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

import asyncio
from unittest.mock import patch

import cftime
import numpy as np
import pytest
import xarray as xr

from earth2studio.data.samudrace import (
    IC_TIMESTAMPS,
    SamudrACEData,
    SamudrACEForcingData,
)

N_LAT = 4
N_LON = 8
# South-to-north latitudes, as in the SamudrACE files on HuggingFace; the
# data sources serve latitude north to south (Earth2Studio convention)
FILE_LAT = np.linspace(-60.0, 60.0, N_LAT)
FILE_LON = np.arange(0.5, 360.5, 360.0 / N_LON)

ATM_VARS = ["surface_temperature", "TMP2m", "PRESsfc"]
OCEAN_VARS = ["sst", "zos", "thetao_0"]


def _spatial_field(seed: int) -> np.ndarray:
    """Build a deterministic (sample, lat, lon) field for a variable."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, N_LAT, N_LON)).astype(np.float32)


@pytest.fixture
def ic_paths(tmp_path):
    """Create minimal atmosphere and ocean IC NetCDF files on disk."""
    coords = {"lat": FILE_LAT, "lon": FILE_LON}

    atm_ds = xr.Dataset(
        {
            v: (["sample", "lat", "lon"], _spatial_field(i))
            for i, v in enumerate(ATM_VARS)
        },
        coords=coords,
    )
    atm_path = tmp_path / "atm_ic.nc"
    atm_ds.to_netcdf(atm_path)

    ocean_ds = xr.Dataset(
        {
            v: (["sample", "lat", "lon"], _spatial_field(100 + i))
            for i, v in enumerate(OCEAN_VARS)
        },
        coords=coords,
    )
    ocean_path = tmp_path / "ocean_ic.nc"
    ocean_ds.to_netcdf(ocean_path)

    return str(atm_path), str(ocean_path)


@pytest.fixture
def ic_source(ic_paths):
    """SamudrACEData with downloads redirected to the local IC files."""
    atm_path, ocean_path = ic_paths

    def fake_fetch(self, filename):
        """Return the local path matching the requested component file."""
        return atm_path if "atmosphere" in filename else ocean_path

    with patch.object(SamudrACEData, "_fetch_file", fake_fetch):
        yield SamudrACEData(verbose=False)


@pytest.fixture
def forcing_path(tmp_path):
    """Create a minimal no-leap forcing NetCDF file on disk."""
    times = [
        cftime.DatetimeNoLeap(5, 1, 1, 0),
        cftime.DatetimeNoLeap(5, 1, 1, 6),
        cftime.DatetimeNoLeap(5, 1, 1, 12),
        cftime.DatetimeNoLeap(5, 1, 1, 18),
        cftime.DatetimeNoLeap(5, 3, 1, 0),
        cftime.DatetimeNoLeap(5, 12, 31, 18),
    ]
    dswrf = np.stack(
        [np.full((N_LAT, N_LON), float(i), dtype=np.float32) for i in range(len(times))]
    )
    ds = xr.Dataset(
        {
            "DSWRFtoa": (["time", "lat", "lon"], dswrf),
            "land_fraction": (["lat", "lon"], _spatial_field(7)[0]),
            "HGTsfc": (["lat", "lon"], _spatial_field(8)[0]),
        },
        coords={"time": times, "lat": FILE_LAT, "lon": FILE_LON},
    )
    path = tmp_path / "forcing.nc"
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def forcing_source(forcing_path):
    """SamudrACEForcingData with downloads redirected to the local file."""

    def fake_fetch(self, filename):
        """Return the local path of the synthetic forcing file."""
        return forcing_path

    with patch.object(SamudrACEForcingData, "_fetch_file", fake_fetch):
        yield SamudrACEForcingData(scenario="0311", verbose=False)


# ---------------------------------------------------------------------------
# SamudrACEData (initial conditions)
# ---------------------------------------------------------------------------
def test_samudrace_data_ic_timestamps():
    """The published IC timestamps are exposed as the valid time domain."""
    assert set(IC_TIMESTAMPS) == {
        "0151-01-06T00:00:00",
        "0311-01-01T00:00:00",
        "0313-01-01T00:00:00",
        "0315-01-01T00:00:00",
        "0317-01-01T00:00:00",
        "0319-01-01T00:00:00",
    }


def test_samudrace_data_call(ic_source):
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    da = ic_source(time, ["skt", "t2m", "sst"])
    assert da.shape == (1, 3, N_LAT, N_LON)
    assert list(da.dims) == ["time", "variable", "lat", "lon"]
    assert list(da.coords["variable"].values) == ["skt", "t2m", "sst"]
    assert (da.coords["time"].values == time).all()
    # Latitude is served north to south
    assert da.coords["lat"].values[0] > da.coords["lat"].values[-1]
    np.testing.assert_allclose(da.coords["lat"].values, FILE_LAT[::-1])
    # Values match the file contents, flipped to north-to-south
    np.testing.assert_array_equal(da.values[0, 0], _spatial_field(0)[0, ::-1])
    np.testing.assert_array_equal(da.values[0, 2], _spatial_field(100)[0, ::-1])


def test_samudrace_data_multiple_times(ic_source):
    time = [
        np.datetime64("0311-01-01T00:00:00"),
        np.datetime64("0313-01-01T00:00:00"),
    ]
    da = ic_source(np.array(time), ["sp", "zos"])
    assert da.shape == (2, 2, N_LAT, N_LON)
    assert not np.isnan(da.values).any()


def test_samudrace_data_invalid_time(ic_source):
    with pytest.raises(ValueError, match="not a published SamudrACE"):
        ic_source(np.array([np.datetime64("2001-01-01T00:00:00")]), ["t2m"])
    # Off by six hours from a published timestamp
    with pytest.raises(ValueError, match="not a published SamudrACE"):
        ic_source(np.array([np.datetime64("0311-01-01T06:00:00")]), ["t2m"])


def test_samudrace_data_unknown_variable(ic_source):
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    with pytest.raises(KeyError, match="not a SamudrACE Earth2Studio variable"):
        ic_source(time, ["fake_variable_xyz"])


def test_samudrace_data_variable_not_in_files(ic_source):
    # In the lexicon, but not present in the (synthetic) IC files
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    with pytest.raises(KeyError, match="not(.|\n)*found"):
        ic_source(time, ["q2m"])


def test_samudrace_data_fetch_async(ic_source):
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    da = asyncio.run(ic_source.fetch(time, ["t2m"]))
    assert da.shape == (1, 1, N_LAT, N_LON)


# ---------------------------------------------------------------------------
# SamudrACEForcingData (exogenous forcing)
# ---------------------------------------------------------------------------
def test_samudrace_forcing_invalid_scenario():
    with pytest.raises(ValueError, match="scenario must be one of"):
        SamudrACEForcingData(scenario="9999")


def test_samudrace_forcing_year_ignoring_match(forcing_source):
    # The same month/day/hour matches regardless of the requested year
    for year in ["0311", "0313", "2001"]:
        time = np.array([np.datetime64(f"{year}-01-01T06:00:00")])
        da = forcing_source(time, ["mtdwswrf"])
        assert da.shape == (1, 1, N_LAT, N_LON)
        np.testing.assert_array_equal(da.values[0, 0], np.full((N_LAT, N_LON), 1.0))


def test_samudrace_forcing_window(forcing_source):
    # A forcing window mixing time-varying and static fields
    time = np.array(
        [
            np.datetime64("0311-01-01T00:00:00"),
            np.datetime64("0311-01-01T06:00:00"),
            np.datetime64("0311-01-01T12:00:00"),
        ]
    )
    da = forcing_source(time, ["mtdwswrf", "land_abs", "z"])
    assert da.shape == (3, 3, N_LAT, N_LON)
    assert list(da.coords["variable"].values) == ["mtdwswrf", "land_abs", "z"]
    assert (da.coords["time"].values == time).all()
    # Time-varying field follows the file's time axis
    for i in range(3):
        np.testing.assert_array_equal(
            da.values[i, 0], np.full((N_LAT, N_LON), float(i))
        )
    # Static fields are broadcast identically to every requested time
    for j in [1, 2]:
        np.testing.assert_array_equal(da.values[0, j], da.values[1, j])
        np.testing.assert_array_equal(da.values[0, j], da.values[2, j])
    # Latitude is served north to south
    assert da.coords["lat"].values[0] > da.coords["lat"].values[-1]
    np.testing.assert_array_equal(da.values[0, 1], _spatial_field(7)[0, ::-1])


def test_samudrace_forcing_year_boundary(forcing_source):
    # A window crossing the year boundary tiles back to January
    time = np.array(
        [
            np.datetime64("0311-12-31T18:00:00"),
            np.datetime64("0312-01-01T00:00:00"),
        ]
    )
    da = forcing_source(time, ["mtdwswrf"])
    np.testing.assert_array_equal(da.values[0, 0], np.full((N_LAT, N_LON), 5.0))
    np.testing.assert_array_equal(da.values[1, 0], np.full((N_LAT, N_LON), 0.0))


def test_samudrace_forcing_out_of_calendar(forcing_source):
    # February 29 has no counterpart on the no-leap forcing calendar
    with pytest.raises(ValueError, match="no counterpart"):
        forcing_source(np.array([np.datetime64("2000-02-29T00:00:00")]), ["mtdwswrf"])
    # Off the 6-hourly grid
    with pytest.raises(ValueError, match="no counterpart"):
        forcing_source(np.array([np.datetime64("0311-01-01T03:00:00")]), ["mtdwswrf"])


def test_samudrace_forcing_unknown_variable(forcing_source):
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    with pytest.raises(KeyError, match="not a SamudrACE Earth2Studio variable"):
        forcing_source(time, ["fake_variable_xyz"])
    # In the lexicon, but not present in the (synthetic) forcing file
    with pytest.raises(KeyError, match="not(.|\n)*found"):
        forcing_source(time, ["ocean_abs"])


def test_samudrace_forcing_fetch_async(forcing_source):
    time = np.array([np.datetime64("0311-01-01T00:00:00")])
    da = asyncio.run(forcing_source.fetch(time, ["land_abs"]))
    assert da.shape == (1, 1, N_LAT, N_LON)
