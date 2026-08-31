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

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nnja_ai", reason="nnja-ai not installed")

from earth2studio.data import NNJAAIObsConv, NNJAAIObsSat  # noqa: E402
from earth2studio.data.nnja_ai import (  # noqa: E402
    _specific_humidity_from_dewpoint,
)


def test_nnja_ai_obs_sat_validate_satellites():
    with pytest.raises(ValueError):
        NNJAAIObsSat(satellites=["not-a-real-satellite"])
    ds = NNJAAIObsSat(satellites=["metop-b", "n20"])
    assert ds._satellites == ["metop-b", "n20"]


def test_nnja_ai_obs_sat_unsupported_variable():
    ds = NNJAAIObsSat()
    with pytest.raises(ValueError):
        ds(datetime(2024, 1, 1), ["not-a-sensor"])


def test_nnja_ai_obs_sat_validate_time():
    with pytest.raises(ValueError):
        NNJAAIObsSat()(datetime(1990, 1, 1), ["amsua"])


def test_nnja_ai_obs_sat_available():
    assert NNJAAIObsSat.available(datetime(2020, 1, 1))
    assert not NNJAAIObsSat.available(datetime(1990, 1, 1))
    assert NNJAAIObsSat.available(np.datetime64("2020-01-01"))


def test_nnja_ai_obs_conv_unknown_variable_raises():
    ds = NNJAAIObsConv()
    with pytest.raises(ValueError):
        ds(datetime(2024, 1, 1), ["not-a-variable"])


def test_nnja_ai_obs_conv_gpsro_warns_and_skips(monkeypatch):
    ds = NNJAAIObsConv()
    monkeypatch.setattr(ds, "_fetch_adpupa", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ds, "_fetch_adpsfc", lambda *a, **k: pd.DataFrame())
    with pytest.warns(UserWarning, match="GPS-RO"):
        out = ds(datetime(2024, 1, 1), ["gps"])
    assert out.empty
    assert list(out.columns) == ds.SCHEMA_COLUMNS


def test_nnja_ai_obs_conv_validate_time():
    with pytest.raises(ValueError):
        NNJAAIObsConv()(datetime(1970, 1, 1), ["t"])


def test_nnja_ai_obs_conv_available():
    assert NNJAAIObsConv.available(datetime(2020, 1, 1))
    assert not NNJAAIObsConv.available(datetime(1900, 1, 1))


def test_nnja_ai_specific_humidity_from_dewpoint_matches_bolton():
    # At the dewpoint, actual vapor pressure == saturation vapor pressure.
    # Sanity check against a known approximate value: T_d = 273.15 K (0 C)
    # gives e_s ~= 6.112 hPa; at p = 1000 hPa this is q ~= 3.8e-3 kg/kg.
    dewpoint_k = np.array([273.15])
    pressure_pa = np.array([100000.0])
    q = _specific_humidity_from_dewpoint(dewpoint_k, pressure_pa)
    assert q.dtype == np.float32
    assert 3.7e-3 < q[0] < 3.9e-3


def test_nnja_ai_specific_humidity_increases_with_dewpoint():
    pressure_pa = np.full(3, 100000.0)
    dewpoint_k = np.array([260.0, 273.15, 290.0])
    q = _specific_humidity_from_dewpoint(dewpoint_k, pressure_pa)
    assert np.all(np.diff(q) > 0)


def test_nnja_ai_obs_conv_station_id_formatting():
    ds = NNJAAIObsConv()
    wide = pd.DataFrame({"WMOB": [16.0, np.nan, 3.0], "WMOS": [133.0, 500.0, 7.0]})
    station = ds._station_id(wide)
    assert station.tolist() == ["16133", "", "03007"]


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
@pytest.mark.parametrize("time", [datetime(2024, 1, 1, 0)])
@pytest.mark.parametrize("variable", [["amsua"], ["atms", "mhs"]])
def test_nnja_ai_obs_sat_fetch(time, variable):
    ds = NNJAAIObsSat(time_tolerance=timedelta(minutes=30))
    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA_COLUMNS
    assert set(df["variable"].unique()).issubset(set(variable))
    assert not df.empty


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
@pytest.mark.parametrize("time", [datetime(2024, 1, 1, 0)])
@pytest.mark.parametrize("variable", [["pres"], ["t", "u", "v"]])
def test_nnja_ai_obs_conv_fetch(time, variable):
    ds = NNJAAIObsConv(time_tolerance=timedelta(minutes=30))
    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA_COLUMNS
    assert set(df["variable"].unique()).issubset(set(variable))
    assert not df.empty
