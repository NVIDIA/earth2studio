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
    InferenceOuputSource,
    Random,
)
from earth2studio.io import XarrayBackend
from earth2studio.models.px import Persistence
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble


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


@pytest.mark.parametrize(
    "time",
    (
        np.array([np.datetime64("2018-01-01")]),
        np.array([np.datetime64("2018-01-01"), np.datetime64("2018-01-02")]),
    ),
)
@pytest.mark.parametrize(
    "lead_time",
    (
        np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")]),
        np.array(
            [np.timedelta64(0, "h"), np.timedelta64(1, "h"), np.timedelta64(2, "h")]
        ),
    ),
)
@pytest.mark.parametrize(
    "filter_dict",
    (
        {
            "time": np.datetime64("2018-01-01T00:00:00"),
            "ensemble": 0,
        },
        {
            "lead_time": np.timedelta64(1, "h"),
            "ensemble": 0,
        },
    ),
)
def test_inference_output_source(
    time: np.ndarray, lead_time: np.ndarray, filter_dict: dict, tmp_path
):
    variable = ["u10m", "v10m", "t2m"]

    def mock_inference_pipeline(
        run_times: np.ndarray,
        nsteps: int,
        var_names: list[str],
        domain_coords: dict,
        num_members: int = 0,
    ) -> tuple[str, xr.Dataset]:
        domain_coords = {"lat": np.linspace(-10, 10, 4), "lon": np.linspace(0, 30, 8)}
        ds = Random(domain_coords=domain_coords)
        px = Persistence(
            variable=var_names, domain_coords=domain_coords, dt=np.timedelta64(1, "h")
        )

        output_file = f"{tmp_path}/output.nc"
        io = XarrayBackend(coords=domain_coords)
        io = ensemble(
            time=run_times,
            nsteps=nsteps,
            nensemble=num_members,
            prognostic=px,
            data=ds,
            io=io,
            perturbation=Zero(),
        )
        io.root.to_netcdf(output_file)

        return output_file

    output_path = mock_inference_pipeline(
        time,
        lead_time.shape[0],
        variable,
        {"lat": np.linspace(-10, 10, 4), "lon": np.linspace(0, 30, 8)},
        num_members=3,
    )

    ds = InferenceOuputSource(
        output_path,
        filter_dict=filter_dict,
        engine="h5netcdf",
    )
    # Check consistency
    target_da = xr.open_dataset(output_path, engine="h5netcdf")
    target_da = target_da.to_array("variable")
    if "time" in filter_dict:
        for lead_time in target_da.coords["lead_time"].values:
            time_stamp = filter_dict["time"] + lead_time
            da = ds(time=time_stamp, variable=["u10m", "v10m"])
            dat = target_da.sel(
                ensemble=0,
                time=filter_dict["time"],
                lead_time=lead_time,
                variable=["u10m", "v10m"],
            )
            assert np.allclose(da.values, dat.values)
    else:
        for time in target_da.coords["time"].values:
            time_stamp = filter_dict["lead_time"] + time
            da = ds(time=time_stamp, variable=["u10m", "v10m"])
            dat = target_da.sel(
                ensemble=0,
                time=time,
                lead_time=filter_dict["lead_time"],
                variable=["u10m", "v10m"],
            )
            assert np.allclose(da.values, dat.values)
