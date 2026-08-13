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
import pathlib
import shutil

import numpy as np
import pytest
import xarray as xr

import earth2studio.data.ace2 as ace2_module
from earth2studio.data import ACE2ERA5Data


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2001, month=1, day=1, hour=0),
        [
            datetime.datetime(year=2001, month=1, day=1, hour=0),
            datetime.datetime(year=2001, month=1, day=1, hour=12),
        ],
        np.array([np.datetime64("2001-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["skt", "mtdwswrf"], ["global_mean_co2"]])
def test_ace2era5_forcing_fetch(time, variable):

    ds = ACE2ERA5Data(mode="forcing", cache=False)
    da = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime.datetime):
        time = [time]

    assert da.shape[0] == len(time)
    assert da.shape[1] == len(variable)
    assert da.shape[2] == len(ds.lat)
    assert da.shape[3] == len(ds.lon)
    assert np.array_equal(da.coords["variable"].values, np.array(variable))
    assert not np.isnan(da.values).any()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2001, month=1, day=1, hour=0),
        [
            datetime.datetime(year=2020, month=1, day=1, hour=0),
            datetime.datetime(year=2020, month=2, day=1, hour=0),
        ],
        np.array([np.datetime64("2020-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["skt", "u3k", "v3k", "u10m"]])
def test_ace2era5_ic_fetch(time, variable):
    ds = ACE2ERA5Data(mode="initial_conditions", cache=False)
    da = ds(time, variable)
    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime.datetime):
        time = [time]
    assert da.shape[0] == len(time)
    assert da.shape[1] == len(variable)
    assert da.shape[2] == len(ds.lat)
    assert da.shape[3] == len(ds.lon)
    assert np.array_equal(da.coords["variable"].values, np.array(variable))
    assert not np.isnan(da.values).any()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1979-04-01T00:00")])],
)
@pytest.mark.parametrize("variable", [["t2m"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ace2era5_cache(time, variable, cache):

    ds = ACE2ERA5Data(cache=cache)
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass

    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == time.shape[0]
    assert shape[1] == len(variable)
    assert shape[2] == 180
    assert shape[3] == 360
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == time.shape[0]
    assert shape[1] == len(variable)
    assert shape[2] == 180
    assert shape[3] == 360
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_ace2era5_initial_conditions_year_validation():
    ds = ACE2ERA5Data(mode="initial_conditions")
    # 1981 is not an allowed IC year
    with pytest.raises(ValueError):
        ds(datetime.datetime(year=1981, month=1, day=1, hour=0), ["skt"])


def test_ace2era5_forcing_static_fields_expand_multiple_times(monkeypatch, tmp_path):
    times = np.array(
        ["1950-01-01T00:00:00", "1950-01-01T06:00:00"], dtype="datetime64[ns]"
    )
    lat = np.array([-0.5, 0.5], dtype=np.float32)
    lon = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "surface_temperature": (
                ("time", "lat", "lon"),
                np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            ),
            "HGTsfc": (("lat", "lon"), np.full((2, 3), 10.0, dtype=np.float32)),
            "global_mean_co2": (("time",), np.array([400.0, 401.0], dtype=np.float32)),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )

    cache_file = tmp_path / "ACE2ERA5" / "forcing_data" / "forcing_1950.nc"
    cache_file.parent.mkdir(parents=True)
    cache_file.touch()
    monkeypatch.setattr(ace2_module, "datasource_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(ace2_module.xr, "open_dataset", lambda *_, **__: dataset)

    ds = ACE2ERA5Data(mode="forcing", cache=True, verbose=False)
    ds.lat = lat
    ds.lon = lon

    da = ds(times, ["skt", "z", "global_mean_co2"])

    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.shape == (2, 3, 2, 3)
    assert np.array_equal(da.coords["time"].values, times)
    assert np.array_equal(
        da.coords["variable"].values,
        np.array(["skt", "z", "global_mean_co2"], dtype=object),
    )
    assert np.all(da.sel(variable="z").values == 10.0)
    assert np.all(da.sel(variable="global_mean_co2").values[0] == 400.0)
    assert np.all(da.sel(variable="global_mean_co2").values[1] == 401.0)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ace2era5_co2_fn_override():
    # Define a CO2 function that returns predictable values per-time
    def co2_fn(times):
        # times is a list of datetimes; return array with ascending values
        return np.array([410.0 + i for i, _ in enumerate(times)], dtype=np.float32)

    ds = ACE2ERA5Data(mode="forcing", co2_fn=co2_fn, cache=False)
    times = [
        datetime.datetime(2001, 1, 1, 0),
        datetime.datetime(2001, 1, 1, 12),
    ]
    da = ds(times, ["global_mean_co2"])  # dims: [time, variable, lat, lon]

    # For each time index, the value across all lat/lon should equal co2_fn(times)[i]
    expected = co2_fn(times)
    # Remove variable dimension (size 1)
    vals = da.values[:, 0, :, :]
    # Check each time slice is constant and equals expected
    for i in range(len(times)):
        assert np.allclose(vals[i], expected[i])
