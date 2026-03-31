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

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.data.samudrace import _IC_TIMESTAMPS, SamudrACEData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
N_LAT = len(ACE_GRID_LAT)
N_LON = len(ACE_GRID_LON)

# Representative atmosphere and ocean variable names (FME-side)
ATM_VARS = ["surface_temperature", "TMP2m", "PRESsfc"]
OCEAN_VARS = ["sst", "zos", "thetao_0"]


@pytest.fixture()
def mock_ic_datasets(tmp_path):
    """Create minimal atmosphere and ocean IC NetCDF files on disk."""
    atm_ds = xr.Dataset()
    for v in ATM_VARS:
        atm_ds[v] = xr.DataArray(
            np.random.randn(1, N_LAT, N_LON).astype(np.float32),
            dims=["sample", "lat", "lon"],
        )
    atm_ds = atm_ds.assign_coords(
        lat=np.linspace(-89, 89, N_LAT).astype(np.float32),
        lon=np.linspace(0.5, 359.5, N_LON).astype(np.float32),
    )
    atm_path = tmp_path / "atm_ic.nc"
    atm_ds.to_netcdf(atm_path)

    ocean_ds = xr.Dataset()
    for v in OCEAN_VARS:
        ocean_ds[v] = xr.DataArray(
            np.random.randn(1, N_LAT, N_LON).astype(np.float32),
            dims=["sample", "lat", "lon"],
        )
    ocean_ds = ocean_ds.assign_coords(
        lat=np.linspace(-89, 89, N_LAT).astype(np.float32),
        lon=np.linspace(0.5, 359.5, N_LON).astype(np.float32),
    )
    ocean_path = tmp_path / "ocean_ic.nc"
    ocean_ds.to_netcdf(ocean_path)

    return atm_path, ocean_path, atm_ds, ocean_ds


def _make_data_source_with_local_ics(mock_ic_datasets):
    """Create a SamudrACEData and inject pre-loaded datasets to skip HF."""
    atm_path, ocean_path, _, _ = mock_ic_datasets
    with patch.object(SamudrACEData, "_download_ic_file"):
        ds = SamudrACEData(ic_timestamp="0311-01-01T00:00:00")
    ds._atm_ds = xr.open_dataset(atm_path, engine="netcdf4")
    ds._ocean_ds = xr.open_dataset(ocean_path, engine="netcdf4")
    return ds


# ---------------------------------------------------------------------------
# Tests: SamudrACEData
# ---------------------------------------------------------------------------
def test_samudrace_data_invalid_timestamp():
    """Invalid ic_timestamp raises ValueError."""
    with pytest.raises(ValueError, match="ic_timestamp must be one of"):
        SamudrACEData(ic_timestamp="9999-01-01T00:00:00")


def test_samudrace_data_call_atm(mock_ic_datasets):
    """Fetch atmosphere variables via __call__."""
    ds = _make_data_source_with_local_ics(mock_ic_datasets)
    time = datetime.datetime(2001, 1, 1, 0)
    da = ds(time, ["skt"])  # skt maps to surface_temperature
    assert da.shape == (1, 1, N_LAT, N_LON)
    np.testing.assert_array_equal(da.coords["lat"].values, ACE_GRID_LAT)
    np.testing.assert_array_equal(da.coords["lon"].values, ACE_GRID_LON)
    assert not np.isnan(da.values).any()


def test_samudrace_data_call_ocean(mock_ic_datasets):
    """Fetch ocean variables via __call__."""
    ds = _make_data_source_with_local_ics(mock_ic_datasets)
    time = [datetime.datetime(2001, 1, 1, 0), datetime.datetime(2001, 1, 2, 0)]
    da = ds(time, ["sst", "zos"])  # identity-mapped ocean var names
    assert da.shape == (2, 2, N_LAT, N_LON)
    # Same spatial fields for both times
    np.testing.assert_array_equal(da.values[0], da.values[1])


def test_samudrace_data_call_mixed(mock_ic_datasets):
    """Fetch a mix of atmosphere + ocean variables."""
    ds = _make_data_source_with_local_ics(mock_ic_datasets)
    time = np.array([np.datetime64("2001-01-01T00:00")])
    da = ds(time, ["skt", "sst"])
    assert da.shape == (1, 2, N_LAT, N_LON)


def test_samudrace_data_unknown_variable(mock_ic_datasets):
    """Unknown variable raises KeyError."""
    ds = _make_data_source_with_local_ics(mock_ic_datasets)
    time = datetime.datetime(2001, 1, 1, 0)
    with pytest.raises(KeyError, match="Unknown SamudrACE variable"):
        ds(time, ["totally_fake_variable_xyz"])


def test_samudrace_data_fetch_async(mock_ic_datasets):
    """Async fetch returns same result as __call__."""
    import asyncio

    ds = _make_data_source_with_local_ics(mock_ic_datasets)
    time = datetime.datetime(2001, 1, 1, 0)
    result = asyncio.get_event_loop().run_until_complete(ds.fetch(time, ["skt"]))
    assert result.shape == (1, 1, N_LAT, N_LON)


def test_samudrace_data_cache_property():
    """Verify cache paths for cached and non-cached modes."""
    with patch.object(SamudrACEData, "__init__", lambda self, **kw: None):
        ds = SamudrACEData.__new__(SamudrACEData)
        ds._cache = True
        ds._ic_timestamp = "0311-01-01T00:00:00"
        ds._verbose = True
        ds._hf_fs = MagicMock()
        ds._atm_ds = None
        ds._ocean_ds = None
        ds.lat = ACE_GRID_LAT
        ds.lon = ACE_GRID_LON
        cached_path = ds.cache
        assert "SamudrACE" in cached_path
        assert "tmp_SamudrACE" not in cached_path

        ds._cache = False
        uncached_path = ds.cache
        assert "tmp_SamudrACE" in uncached_path


def test_samudrace_data_available_timestamps():
    """Verify all documented timestamps are valid."""
    expected = {
        "0151-01-06T00:00:00",
        "0311-01-01T00:00:00",
        "0313-01-01T00:00:00",
        "0315-01-01T00:00:00",
        "0317-01-01T00:00:00",
        "0319-01-01T00:00:00",
    }
    assert set(_IC_TIMESTAMPS.keys()) == expected
