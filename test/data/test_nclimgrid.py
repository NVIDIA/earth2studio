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

import pathlib
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import NClimGrid


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2010, month=7, day=1),
        [
            datetime(year=2010, month=7, day=1),
            datetime(year=2010, month=7, day=2),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m_max", ["t2m_max", "tp"]])
def test_nclimgrid_fetch(time, variable):

    ds = NClimGrid(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] > 100  # CONUS lat grid (~596 points)
    assert shape[3] > 100  # CONUS lon grid (~1385 points)
    assert not np.isnan(data.values).all()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("2010-07-01T00:00")])],
)
@pytest.mark.parametrize("variable", [["t2m_max", "tp"]])
@pytest.mark.parametrize("cache", [True, False])
def test_nclimgrid_cache(time, variable, cache):

    ds = NClimGrid(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] > 100
    assert shape[3] > 100
    assert not np.isnan(data.values).all()

    if cache:
        assert pathlib.Path(ds.cache).is_dir()

    # Reload from cache or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert not np.isnan(data.values).all()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=1940, month=1, day=1),
        datetime(year=1950, month=12, day=31),
    ],
)
@pytest.mark.parametrize("variable", ["t2m_max"])
def test_nclimgrid_valid_time(time, variable):
    with pytest.raises(ValueError):
        ds = NClimGrid()
        ds(time, variable)


@pytest.mark.timeout(15)
def test_nclimgrid_available():
    assert NClimGrid.available(datetime(2010, 7, 1))
    assert NClimGrid.available(datetime(1951, 1, 1))
    assert not NClimGrid.available(datetime(1940, 1, 1))
    assert not NClimGrid.available(datetime(1950, 12, 31))
    assert NClimGrid.available(np.datetime64("2010-07-01"))
    assert not NClimGrid.available(np.datetime64("1940-01-01"))


@pytest.mark.parametrize(
    "time,expected_uri",
    [
        pytest.param(
            datetime(2010, 7, 15),
            "s3://noaa-nclimgrid-daily-pds/access/grids/2010/ncdd-201007-grd-scaled.nc",
            id="2010-07-15",
        ),
        pytest.param(
            datetime(2023, 1, 1),
            "s3://noaa-nclimgrid-daily-pds/access/grids/2023/ncdd-202301-grd-scaled.nc",
            id="2023-01-01",
        ),
        pytest.param(
            datetime(1951, 12, 31),
            "s3://noaa-nclimgrid-daily-pds/access/grids/1951/ncdd-195112-grd-scaled.nc",
            id="1951-12-31",
        ),
    ],
)
def test_nclimgrid_monthly_uri(time, expected_uri):
    ds = NClimGrid()
    assert ds._monthly_nc_uri(time) == expected_uri


def test_nclimgrid_create_tasks():
    ds = NClimGrid()
    times = [datetime(2010, 7, 1), datetime(2010, 7, 2)]
    variables = ["t2m_max", "tp"]

    tasks = ds._create_tasks(times, variables)

    assert len(tasks) == 4  # 2 times x 2 variables
    # Check task metadata
    task = tasks[0]
    assert task.time_index == 0
    assert task.variable_index == 0
    assert task.variable_id == "t2m_max"
    assert task.native_key == "tmax"
    assert "2010" in task.nc_uri
    assert callable(task.modifier)


def test_nclimgrid_create_tasks_invalid_variable():
    ds = NClimGrid()
    tasks = ds._create_tasks([datetime(2010, 7, 1)], ["nonexistent_var"])
    assert len(tasks) == 0


def test_nclimgrid_call_mock(tmp_path: pathlib.Path):
    """Test NClimGrid __call__ with mocked S3 filesystem (no network)."""
    # Create a mock monthly NetCDF with fake CONUS grid
    lat = np.linspace(24.0, 50.0, 10, dtype=np.float32)
    lon = np.linspace(-125.0, -67.0, 15, dtype=np.float32)
    time_coord = [np.datetime64("2010-07-01"), np.datetime64("2010-07-02")]

    mock_ds = xr.Dataset(
        {
            "tmax": (
                ["time", "lat", "lon"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32) * 30,
            ),
            "prcp": (
                ["time", "lat", "lon"],
                np.random.rand(2, len(lat), len(lon)).astype(np.float32) * 10,
            ),
        },
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )
    nc_path = tmp_path / "mock_nclimgrid.nc"
    mock_ds.to_netcdf(nc_path)

    mock_fs = MagicMock()
    mock_fs.open.return_value.__enter__ = lambda s: open(nc_path, "rb")
    mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(NClimGrid, "_async_init", return_value=None):
        ds = NClimGrid(cache=False)
        ds.fs = mock_fs

        data = ds(datetime(2010, 7, 1), ["t2m_max", "tp"])

    assert data.shape == (1, 2, len(lat), len(lon))
    assert list(data.coords["variable"].values) == ["t2m_max", "tp"]
    # Verify unit conversion applied (tmax: C->K, prcp: mm->m)
    assert data.sel(variable="t2m_max").values.min() > 250  # Should be in Kelvin
    assert data.sel(variable="tp").values.max() < 1.0  # Should be in meters
