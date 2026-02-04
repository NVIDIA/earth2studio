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

"""Unit tests for the Planetary Computer lexicons."""

from __future__ import annotations

import numpy as np
import pytest

from earth2studio.lexicon import (
    ECMWFOpenDataIFSLexicon,
    MODISFireLexicon,
    OISSTLexicon,
    Sentinel3AODLexicon,
)


@pytest.mark.parametrize(
    "variable, expected",
    [
        ("sst", 273.15),
        ("ssta", 0.0),
        ("sstu", 0.0),
        ("sic", 0.01),
    ],
)
def test_planetary_computer_oisst_lexicon(variable: str, expected: float) -> None:
    """Check OISST modifiers and labels."""
    label, modifier = OISSTLexicon[variable]
    assert isinstance(label, str)

    sample = np.array([1.0, 2.0], dtype=np.float32)
    out = modifier(sample.copy())
    assert out.shape == sample.shape

    if variable == "sst":
        assert np.allclose(out, sample + expected)
    elif variable == "sic":
        assert np.allclose(out, sample / 100.0)
    else:
        assert np.allclose(out, sample)


@pytest.mark.parametrize(
    "variable",
    [
        "s3sy01aod",
        "s3sy02ssa",
        "s3sy03sr",
        "s3sysunzen",
        "s3sycloudfrac",
        "s3sy_lat",
        "s3sy_lon",
    ],
)
def test_planetary_computer_sentinel3_aod_lexicon(variable: str) -> None:
    """Ensure Sentinel-3 SYNERGY lexicon entries map cleanly."""
    label, modifier = Sentinel3AODLexicon[variable]
    assert isinstance(label, str)

    sample = np.random.rand(4, 4).astype(np.float32)
    out = modifier(sample.copy())
    assert np.allclose(out, sample)


@pytest.mark.parametrize("variable", ["fmask", "mfrp", "qa"])
def test_planetary_computer_modis_fire_lexicon(variable: str) -> None:
    """Check MODIS Fire lexicon mappings."""
    label, modifier = MODISFireLexicon[variable]
    assert isinstance(label, str)

    sample = np.random.rand(3, 3).astype(np.float32)
    out = modifier(sample.copy())
    assert np.allclose(out, sample)


@pytest.mark.parametrize(
    "variable",
    ["u10m", "z500", "t850", "v100"],
)
def test_planetary_computer_ifs_lexicon(variable: str) -> None:
    """Ensure IFS lexicon entries other than geopotential map cleanly."""
    label, modifier = ECMWFOpenDataIFSLexicon[variable]
    assert isinstance(label, str)

    sample = np.random.rand(4, 4).astype(np.float32)
    out = modifier(sample.copy())
    if variable == "z500":
        assert np.allclose(out, sample * 9.81)
    else:
        assert np.allclose(out, sample)
