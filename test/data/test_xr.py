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
import tempfile

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
from earth2studio.run import deterministic, ensemble


def build_inference_output_source(
    # Helper for InferenceOuputSource tests
    run_times: np.ndarray,
    nsteps: int,
    var_names: list[str],
    *,
    num_members: int = 0,
    num_run_times: int = 1,
    num_lead_times: int = 1,
) -> tuple[str, xr.Dataset]:
    domain_coords = {"lat": np.linspace(-10, 10, 4), "lon": np.linspace(0, 30, 8)}
    ds = Random(domain_coords=domain_coords)
    px = Persistence(variable=var_names, domain_coords=domain_coords)

    ds_list = []
    for run_time in run_times:
        io = XarrayBackend(coords=domain_coords)
        if num_members and num_members > 0:
            io = ensemble(
                time=[run_time],
                nsteps=nsteps,
                nensemble=num_members,
                prognostic=px,
                data=ds,
                io=io,
                perturbation=Zero(),
            )
        else:
            io = deterministic(
                time=[run_time],
                nsteps=nsteps,
                prognostic=px,
                data=ds,
                io=io,
            )
        ds_list.append(io.root)
    ds = xr.concat(ds_list, dim="time")
    ds = ds.isel(time=slice(0, num_run_times), lead_time=slice(0, num_lead_times))

    # use a tmp file to store the dataset
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    ds.to_netcdf(tmp_file.name)
    store = tmp_file.name

    return store, ds


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
        nsteps: int,
        var_names: list[str],
        *,
        num_members: int = 0,
        num_run_times: int = 1,
        num_lead_times: int = 1,
    ) -> tuple[str, xr.Dataset]:
        store, ds = build_inference_output_source(
            run_times,
            nsteps,
            var_names,
            num_members=num_members,
            num_run_times=num_run_times,
            num_lead_times=num_lead_times,
        )
        return store, ds

    return _make


filter_dict_lt_ensemble = {
    "lead_time": np.timedelta64(60 * 60 * 36, "s"),
    "ensemble": 0,
}
filter_dict_time_ensemble = {
    "time": np.datetime64("2018-01-01T00:00:00"),
    "ensemble": 0,
}
base_run_time = np.datetime64("2018-01-01T00:00:00")
cases = [
    (base_run_time, filter_dict_lt_ensemble, 3, 10, var_names, num_members)
    for num_members in [1, 2, 3]
    for var_names in [["t2m", "d2m"], ["t2m"]]
] + [
    (
        base_run_time,
        filter_dict_time_ensemble,
        num_run_times,
        10,
        var_names,
        num_members,
    )
    for num_members in [1, 2, 3]
    for var_names in [["t2m", "d2m"], ["t2m"]]
    for num_run_times in [1, 2, 3]
]


def _case_id(case: tuple) -> str:
    base_rt, filt, n_runs, n_leads, vars_, n_members = case
    if "lead_time" in filt:
        try:
            hours = int(filt["lead_time"] / np.timedelta64(1, "h"))
            filt_str = f"lt={hours}h"
        except Exception:
            filt_str = f"lt={str(filt['lead_time'])}"
    elif "time" in filt:
        filt_str = f"time={str(filt['time']).replace('T',' ')}"
    else:
        filt_str = "no-filter"
    vars_str = "+".join(vars_)
    return (
        f"members={n_members}|vars={vars_str}|runs={n_runs}|lead={n_leads}|{filt_str}"
    )


_CASE_IDS = [_case_id(c) for c in cases]


@pytest.mark.parametrize(
    "base_run_time,filter_dict,num_run_times,num_lead_times,var_names,num_members",
    cases,
    ids=_CASE_IDS,
)
def test_inference_output_source(
    make_model_output_store,
    base_run_time: np.datetime64,
    filter_dict: dict,
    num_run_times: int,
    num_lead_times: int,
    var_names: list[str],
    num_members: int,
):
    # Build run times
    run_times = np.array(
        [base_run_time + np.timedelta64(d, "D") for d in range(num_run_times)]
    )

    # Create dataset/store via fixture (NetCDF)
    store, ds = make_model_output_store(
        run_times,
        num_lead_times,
        var_names,
        num_members=num_members,
        num_run_times=num_run_times,
        num_lead_times=num_lead_times,
    )

    # Initialize source
    src = InferenceOuputSource(
        store, engine="h5netcdf", filter_dict=filter_dict if filter_dict else None
    )

    # Avoid mutating the shared case filter
    _filt = dict(filter_dict)
    _filt.pop("lead_time", None)
    _filt.pop("time", None)

    # Choose a valid selection to verify data mapping
    run_time = run_times[0]
    lead_time = np.timedelta64(36, "h")
    valid_t = (run_time + lead_time).astype("datetime64[ns]")

    out = src(time=np.datetime64(valid_t), variable=var_names)
    values_from_netcdf = out.values

    # Expected from original ds
    indexer = {"time": run_time, "lead_time": lead_time}
    indexer.update(_filt)
    values_from_xr = ds[var_names].sel(**indexer).squeeze().to_array().values

    assert np.all(values_from_netcdf == values_from_xr)
