# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import (
    DataArrayFile,
    Random,
    datasource_to_file,
    fetch_data,
    prep_data_array,
)


@pytest.fixture
def foo_data_array():
    time0 = datetime.datetime.now()
    return xr.DataArray(
        data=np.random.rand(8, 16, 32),
        dims=["one", "two", "three"],
        coords={
            "one": [time0 + i * datetime.timedelta(hours=6) for i in range(8)],
            "two": [f"{i}" for i in range(16)],
            "three": np.linspace(0, 1, 32),
        },
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dims", [["one", "two", "three"], ["three", "one", "two"]])
def test_prep_dataarray(foo_data_array, dims, device):

    data_array = foo_data_array.transpose(*dims)
    out, outc = prep_data_array(data_array, device)

    assert str(out.device) == device
    assert list(outc.keys()) == list(data_array.dims)
    for key in outc.keys():
        assert (outc[key] == np.array(data_array.coords[key])).all()
    assert out.shape == data_array.data.shape


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fetch_data(time, lead_time, device):
    variable = np.array(["a", "b", "c"])
    domain = OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1440)})
    r = Random(domain)

    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert not torch.isnan(x).any()


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize(
    "backend",
    ["netcdf", "zarr"],
)
def test_datasource_to_file(time, lead_time, backend, tmp_path):

    variable = np.array(["a", "b", "c"])
    domain = OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1440)})
    ds = Random(domain)

    if backend == "netcdf":
        file_name = str(tmp_path) + "/temp.nc"
    else:
        file_name = str(tmp_path) + "/temp.zarr"
    datasource_to_file(
        file_name,
        ds,
        time=time,
        variable=variable,
        lead_time=lead_time,
        backend=backend,
    )

    # To check attempt to get input data from saved file
    ds = DataArrayFile(file_name)
    x, coords = fetch_data(ds, time, variable, lead_time)

    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert np.all(coords["lat"] == domain["lat"])
    assert np.all(coords["lon"] == domain["lon"])
    assert not torch.isnan(x).any()
