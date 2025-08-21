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
from zarr.storage import MemoryStore

from earth2studio.data import (
    DataArrayDirectory,
    DataArrayFile,
    DataArrayPathList,
    DataSetFile,
    ModelOutputDatasetSource,
)


def build_model_output_dataset(
    run_times: np.ndarray,
    lead_times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    var_names: list[str],
    *,
    num_members: int = 0,
) -> xr.Dataset:
    dims = (
        ["time", "lead_time"]
        + (["member"] if num_members > 0 else [])
        + [
            "lat",
            "lon",
        ]
    )
    data_vars = {}
    if num_members > 0:
        shape = (run_times.size, lead_times.size, num_members, lat.size, lon.size)
    else:
        shape = (run_times.size, lead_times.size, lat.size, lon.size)

    for name in var_names:
        data = np.random.randn(*shape)
        data_vars[name] = (dims, data)

    coords = {
        "time": run_times,
        "lead_time": lead_times,
        "lat": lat,
        "lon": lon,
    }
    if num_members > 0:
        coords["member"] = np.arange(num_members)

    return xr.Dataset(data_vars=data_vars, coords=coords)


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


@pytest.fixture
def make_model_output_store():
    def _make(
        run_times: np.ndarray,
        lead_times: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        var_names: list[str],
        *,
        num_members: int = 0,
    ) -> tuple[MemoryStore, xr.Dataset]:
        ds = build_model_output_dataset(
            run_times,
            lead_times,
            lat,
            lon,
            var_names,
            num_members=num_members,
        )
        store = MemoryStore()
        ds.to_zarr(store, mode="w")
        return store, ds

    return _make


def test_model_output_dataset_source_basic(make_model_output_store):
    run_times = np.array([np.datetime64("2018-01-01T00:00:00")])
    lead_times = np.array(
        [np.timedelta64(0, "h"), np.timedelta64(6, "h"), np.timedelta64(12, "h")]
    )
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(0, 30, 8)
    store, ds = make_model_output_store(
        run_times, lead_times, lat, lon, ["t2m", "u10m"]
    )
    src = ModelOutputDatasetSource(store, engine="zarr")

    valid_t = (run_times[0] + np.timedelta64(6, "h")).astype("datetime64[ns]")
    out = src(time=np.datetime64(valid_t), variable="t2m")

    expected = ds["t2m"].isel(time=0, lead_time=1).values
    assert out.shape == expected.shape
    assert np.all(out.values.squeeze() == expected)


def test_model_output_dataset_source_filter_and_time_selection(make_model_output_store):
    # Dataset with an extra member dimension and two initialization times
    run_times = np.array(
        [
            np.datetime64("2018-01-01T00:00:00"),
            np.datetime64("2018-01-02T00:00:00"),
        ]
    )
    lead_times = np.array([np.timedelta64(0, "h"), np.timedelta64(12, "h")])
    lat = np.linspace(-5, 5, 2)
    lon = np.linspace(0, 10, 3)

    store, ds = make_model_output_store(
        run_times,
        lead_times,
        lat,
        lon,
        ["t2m"],
        num_members=10,
    )

    # Filter member and time to make it valid
    src = ModelOutputDatasetSource(
        store,
        engine="zarr",
        filter_dict={"member": 0, "time": run_times[0]},
    )

    valid_t = (run_times[0] + np.timedelta64(12, "h")).astype("datetime64[ns]")
    out = src(time=np.datetime64(valid_t), variable="t2m")

    expected = ds["t2m"].sel(member=0).isel(time=0, lead_time=1).values
    assert np.all(out.values.squeeze() == expected)


def test_model_output_dataset_source_requires_filter_for_extra_dims(
    make_model_output_store,
):
    # Dataset with an extra member dimension
    run_times = np.array(
        [
            np.datetime64("2018-01-01T00:00:00"),
        ]
    )
    lead_times = np.array([np.timedelta64(0, "h"), np.timedelta64(12, "h")])
    lat = np.linspace(-5, 5, 2)
    lon = np.linspace(0, 10, 3)

    store, _ = make_model_output_store(
        run_times,
        lead_times,
        lat,
        lon,
        ["t2m"],
        num_members=3,
    )

    with pytest.raises(ValueError):
        ModelOutputDatasetSource(store, engine="zarr")


def test_model_output_dataset_source_too_few_variables_raises(make_model_output_store):
    # Single variable in dataset; requesting an extra variable should fail on selection
    run_times = np.array([np.datetime64("2018-01-01T00:00:00")])
    lead_times = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(6, "h"),
        ]
    )
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(0, 30, 8)

    store, _ = make_model_output_store(
        run_times,
        lead_times,
        lat,
        lon,
        ["t2m"],
    )

    src = ModelOutputDatasetSource(store, engine="zarr")

    valid_t = (run_times[0] + np.timedelta64(6, "h")).astype("datetime64[ns]")
    with pytest.raises(KeyError):
        _ = src(time=np.datetime64(valid_t), variable=["t2m", "u10m"])  # u10m missing


@pytest.mark.parametrize("num_run_times", [1, 2])
@pytest.mark.parametrize("num_lead_times", [1, 2])
@pytest.mark.parametrize("num_var_names", [1, 2])
@pytest.mark.parametrize("num_members", [0, 1, 2])
def test_model_output_dataset_source_parametrized(
    make_model_output_store,
    num_run_times: int,
    num_lead_times: int,
    num_var_names: int,
    num_members: int,
):
    # Build inputs
    base_run = np.datetime64("2018-01-01T00:00:00")
    run_times = np.array(
        [base_run + np.timedelta64(d, "D") for d in range(num_run_times)]
    )
    lead_times = np.array([np.timedelta64(h, "h") for h in (0, 12)[:num_lead_times]])
    lat = np.linspace(-5, 5, 2)
    lon = np.linspace(0, 10, 3)
    all_vars = ["t2m", "u10m"]
    var_names = all_vars[:num_var_names]

    # Create dataset/store
    store, ds = make_model_output_store(
        run_times,
        lead_times,
        lat,
        lon,
        var_names,
        num_members=num_members,
    )

    # Build filter to satisfy ModelOutputDatasetSource requirements
    filter_dict: dict = {}
    if num_run_times > 1:
        filter_dict["time"] = run_times[0]
    if num_members > 0:
        filter_dict["member"] = 0

    # Initialize source
    src = ModelOutputDatasetSource(
        store,
        engine="zarr",
        filter_dict=filter_dict if filter_dict else None,
    )

    # Choose a valid selection to verify data mapping
    lead_idx = 0 if num_lead_times == 1 else 1
    valid_t = (run_times[0] + lead_times[lead_idx]).astype("datetime64[ns]")
    var = var_names[0]

    out = src(time=np.datetime64(valid_t), variable=var)

    # Expected from original ds
    indexer = {"time": 0, "lead_time": lead_idx}
    if num_members > 0:
        expected_vals = ds[var].isel(**indexer, member=0).values
    else:
        expected_vals = ds[var].isel(**indexer).values

    assert np.all(out.values.squeeze() == expected_vals)
