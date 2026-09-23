# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Shared relative-humidity and wet-bulb derivations."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest
from src.shared.derived_variables import (
    _saturation_vapor_pressure,
    rh_from_specific_humidity,
    wet_bulb_K,
)

_EPSILON = 0.621981


def _q_from_rh(t_K: float, rh_pct: float, p_Pa: float) -> float:
    """Specific humidity (kg/kg) for a target RH: inverse of the RH derivation."""
    e = (rh_pct / 100.0) * float(_saturation_vapor_pressure(t_K))
    return _EPSILON * e / (p_Pa - e * (1.0 - _EPSILON))


def test_rh_reference_value() -> None:
    rh = float(rh_from_specific_humidity(298.15, 0.010, 96000.0))
    assert rh == pytest.approx(48.5, abs=0.5)


def test_rh_supersaturation_clips_and_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        rh = float(rh_from_specific_humidity(290.0, 0.02, 101325.0))
    assert rh == 100.0  # clipped
    assert any("supersaturation" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "derivation",
    [rh_from_specific_humidity, wet_bulb_K],
    ids=["relative-humidity", "wet-bulb"],
)
def test_derivations_preserve_nonfinite_inputs_as_nan(
    derivation: Callable[..., object],
) -> None:
    T = np.array([300.0, np.nan, 300.0, 300.0])
    q = np.array([0.01, 0.01, np.nan, 0.01])
    p = np.array([101325.0, 101325.0, 101325.0, np.inf])

    out = np.asarray(derivation(T, q, p))

    assert np.isfinite(out[0])
    assert np.all(np.isnan(out[1:]))


@pytest.mark.parametrize(
    ("temperature", "expected", "tolerance"),
    [
        pytest.param(240.0, 27.21, 0.05, id="ice"),
        pytest.param(260.0, 200.4, 0.5, id="mixed-phase"),
        pytest.param(300.0, 3531.56, 0.5, id="water"),
    ],
)
def test_saturation_reference_values(
    temperature: float, expected: float, tolerance: float
) -> None:
    assert float(_saturation_vapor_pressure(temperature)) == pytest.approx(
        expected, abs=tolerance
    )


def test_wet_bulb_K_independent_saturation_anchor() -> None:
    # _q_from_rh uses derived_variables.py's own es, so an es error would cancel between the input
    # and the solve. Here q is built from an INDEPENDENT Magnus-Tetens es, so a wrong
    # saturation curve in the module cannot hide. Chart: 30 degC / 50% RH -> ~22.0 degC.
    def q_tetens(t_K: float, rh_pct: float, p_Pa: float) -> float:
        t_c = t_K - 273.15
        es = 610.94 * np.exp(17.625 * t_c / (243.04 + t_c))
        e = (rh_pct / 100.0) * es
        return _EPSILON * e / (p_Pa - e * (1.0 - _EPSILON))

    t_K, p = 303.15, 101325.0
    got = float(wet_bulb_K(t_K, q_tetens(t_K, 50.0, p), p))
    assert got == pytest.approx(295.15, abs=0.4)


def test_wet_bulb_K_pressure_dependence() -> None:
    # Same T and RH (same vapour pressure), lower pressure -> larger depression -> lower
    # wet-bulb. This is the elevation correction the sea-level Stull fit misses.
    T, rh = 303.15, 40.0
    tw_sea = float(wet_bulb_K(T, _q_from_rh(T, rh, 101325.0), 101325.0))
    tw_alt = float(wet_bulb_K(T, _q_from_rh(T, rh, 80000.0), 80000.0))
    assert tw_alt < tw_sea - 0.3


def test_wet_bulb_K_vectorized() -> None:
    T = np.array([303.15, 298.15, 263.15])
    p = np.array([101325.0, 90000.0, 101325.0])
    q = np.array([_q_from_rh(float(t), 50.0, float(pp)) for t, pp in zip(T, p)])
    out = wet_bulb_K(T, q, p)
    assert out.shape == (3,)
    assert np.all(out < T)  # wet-bulb below dry-bulb at sub-saturation


def test_wet_bulb_K_below_freezing() -> None:
    # Sub-freezing solve stays finite and physical (uses the mixed-phase saturation).
    T, p = 263.15, 101325.0  # -10 degC
    q = _q_from_rh(T, 80.0, p)
    tw = float(wet_bulb_K(T, q, p))
    assert T - 5.0 < tw < T


@pytest.mark.parametrize("rh_pct", [100.0, 120.0], ids=["saturated", "supersaturated"])
def test_wet_bulb_K_saturation_resolves_to_dry_bulb(rh_pct: float) -> None:
    T, p = 298.15, 101325.0
    q = _q_from_rh(T, rh_pct, p)

    assert float(wet_bulb_K(T, q, p)) == pytest.approx(T, abs=0.05)
