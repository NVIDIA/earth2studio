# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import os
import pathlib
import shutil

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import (
    DataArrayDirectory,
    DataArrayFile,
    DataArrayPathList,
    DataSetFile,
)


@pytest.fixture
def foo_data_array():

    time = [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
        datetime.datetime(year=2018, month=3, day=1),
    ]
    variable = ["u10m", "v10m", "t2m"]

    da = xr.DataArray(
        data=np.random.randn(len(time), len(variable), 8, 16),
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": time,
            "variable": variable,
        },
    )
    return da


@pytest.fixture
def foo_data_set():

    time = [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
        datetime.datetime(year=2018, month=3, day=1),
    ]
    variable = ["u10m", "v10m", "t2m"]
    ds = xr.Dataset(
        data_vars=dict(
            field1=(
                ["time", "variable", "lat", "lon"],
                np.random.randn(len(time), len(variable), 8, 16),
            ),
            field2=(
                ["time", "variable", "lat", "lon"],
                np.random.randn(len(time), len(variable), 8, 16),
            ),
        ),
        coords={
            "time": time,
            "variable": variable,
        },
    )
    return ds


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["u10m", ["u10m", "v10m"]])
def test_data_array_netcdf(foo_data_array, time, variable):
    foo_data_array.to_netcdf("test.nc")
    # Load data source and request data array
    data_source = DataArrayFile("test.nc")
    data = data_source(time, variable)
    # Delete nc file
    pathlib.Path("test.nc").unlink(missing_ok=True)
    # Check consisten
    assert np.all(
        foo_data_array.sel(time=time, variable=variable).values == data.values
    )


@pytest.mark.parametrize("array", ["field1", "field2"])
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["u10m", ["u10m", "v10m"]])
def test_data_set_netcdf(foo_data_set, array, time, variable):
    foo_data_set.to_netcdf("test.nc")
    # Load data source and request data array
    data_source = DataSetFile("test.nc", array)
    data = data_source(time, variable)
    # Delete nc file
    pathlib.Path("test.nc").unlink(missing_ok=True)
    # Check consisten
    assert np.all(
        foo_data_set[array].sel(time=time, variable=variable).values == data.values
    )


def foo_dat_arr(
    time: list[datetime.datetime] = [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
        datetime.datetime(year=2018, month=3, day=1),
    ]
):
    variable = ["u10m", "v10m", "t2m"]

    da = xr.DataArray(
        data=np.random.randn(len(time), len(variable), 8, 16),
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": time,
            "variable": variable,
        },
    )
    return da


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2019, month=1, day=8),
    ],
)
@pytest.mark.parametrize("variable", ["u10m", ["u10m", "v10m"]])
@pytest.mark.parametrize(
    "data_source_type", ["directory", "path_list_glob", "path_list_explicit"]
)
def test_data_array_sources(time, variable, data_source_type):
    # Create test data structure
    base_dir = "test_data_source"
    os.makedirs(base_dir, exist_ok=True)

    test_files = []

    # Create test data for both years
    test_data = {
        year: foo_dat_arr(
            [
                datetime.datetime(year=year, month=1, day=1),
                datetime.datetime(year=year, month=1, day=8),
            ]
        )
        for year in [2018, 2019]
    }

    # Save data to appropriate locations based on source type
    if data_source_type == "directory":
        # Directory structure with year subdirs
        for year, data in test_data.items():
            year_dir = os.path.join(base_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)
            file_path = os.path.join(year_dir, f"{year}_01.nc")
            data.to_netcdf(file_path)
            test_files.append(file_path)
        data_source = DataArrayDirectory(base_dir)
    elif data_source_type in ["path_list_glob", "path_list_explicit"]:
        # Flat structure for path list
        for year, data in test_data.items():
            file_path = os.path.join(base_dir, f"data_{year}.nc")
            data.to_netcdf(file_path)
            test_files.append(file_path)

        if data_source_type == "path_list_glob":
            data_source = DataArrayPathList(os.path.join(base_dir, "*.nc"))
        elif data_source_type == "path_list_explicit":  # path_list_explicit
            data_source = DataArrayPathList(test_files)

    # Request data
    data_loaded = data_source(time, variable)

    # Cleanup
    shutil.rmtree(base_dir)

    target_data = test_data[time.year]
    assert np.all(
        target_data.sel(time=np.datetime64(time), variable=variable).values
        == data_loaded.values
    )


def test_data_array_path_list_exceptions(tmp_path):
    # Test 1: Missing dimensions
    time = [datetime.datetime(year=2018, month=1, day=1)]
    variable = ["u10m"]
    da = xr.DataArray(
        data=np.random.randn(len(time), len(variable)),
        dims=["time", "variable"],
        coords={
            "time": time,
            "variable": variable,
        },
    )
    missing_dims_file = tmp_path / "missing_dims.nc"
    da.to_netcdf(missing_dims_file)
    with pytest.raises(ValueError):
        DataArrayPathList(missing_dims_file)

    # Test 2: Non-existent file pattern
    with pytest.raises(OSError):
        DataArrayPathList("nonexistent_pattern*.nc")
