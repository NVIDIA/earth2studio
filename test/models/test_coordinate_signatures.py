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

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import E2S_CRS, E2S_GRID_ID, resolve_grid
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.precipitation_afno import PrecipitationAFNO
from earth2studio.models.px.stormcastconus import StormCastCONUS
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, E2S_STATISTICS


@pytest.fixture(params=[StormScopeGOES, StormScopeMRMS, StormCastCONUS])
def regional_model(request):
    # Coordinate contracts do not require checkpoint loading or inference extras.
    model = request.param.__new__(request.param)
    torch.nn.Module.__init__(model)
    model.variables = np.array(["u10m", "v10m"])
    lat = np.array([[30.0, 30.1, 30.2], [31.0, 31.1, 31.2]])
    lon = np.array([[250.0, 251.0, 252.0], [250.1, 251.1, 252.1]])
    if isinstance(model, StormCastCONUS):
        grid = resolve_grid("hrrr")
        model.hrrr_y = grid.y[10:12]
        model.hrrr_x = grid.x[20:23]
        model.lat, model.lon = lat, lon
    else:
        model.y, model.x = np.arange(2), np.arange(3)
        model.latitudes, model.longitudes = torch.tensor(lat), torch.tensor(lon)
        model._lat_cpu_copy, model._lon_cpu_copy = lat, lon
        model.input_times = np.array([-2, -1, 0], dtype="timedelta64[h]")
        model.output_times = np.array([1, 2], dtype="timedelta64[h]")
    return model


def test_regional_input_signature(regional_model):
    signature = regional_model.input_coords()
    assert isinstance(signature, xr.DataArray)
    assert signature.data.nbytes == 0
    assert signature.attrs[E2S_DYNAMIC_DIMS] == ("batch", "time")
    assert signature.shape[:2] == (0, 0)
    assert signature.shape[-2:] == (2, 3)
    np.testing.assert_array_equal(
        signature.coords["variable"], regional_model.variables
    )
    assert signature.lat.dims == signature.dims[-2:]
    assert signature.lon.dims == signature.dims[-2:]
    if isinstance(regional_model, StormCastCONUS):
        assert signature.dims[-2:] == ("hrrr_y", "hrrr_x")
        assert signature.attrs[E2S_CRS] == resolve_grid("hrrr").crs.to_string()
        assert signature.attrs["dims"] == ["hrrr_y", "hrrr_x"]
        np.testing.assert_array_equal(signature.hrrr_y, regional_model.hrrr_y)
    else:
        assert signature.dims[-2:] == ("y", "x")
        np.testing.assert_array_equal(signature.lat, regional_model._lat_cpu_copy)


def test_regional_output_window(regional_model):
    signature = regional_model.input_coords()
    assert isinstance(signature, xr.DataArray)
    shifted = signature.assign_coords(
        lead_time=signature.lead_time + np.timedelta64(7, "h")
    )
    output = regional_model.output_coords(shifted)
    expected = np.array(
        [8] if isinstance(regional_model, StormCastCONUS) else [8, 9],
        dtype="timedelta64[h]",
    )
    np.testing.assert_array_equal(output.lead_time, expected)
    assert output.data.nbytes == 0
    assert output.attrs == signature.attrs
    xr.testing.assert_identical(output.lat, signature.lat)
    np.testing.assert_array_equal(
        shifted.lead_time, signature.lead_time + np.timedelta64(7, "h")
    )


def test_regional_concrete_batch_dimensions(regional_model):
    signature = regional_model.input_coords()
    assert isinstance(signature, xr.DataArray)
    coords = dict(signature.coords)
    coords.update(
        member=np.arange(2), time=np.array(["2026-09-17"], dtype="datetime64[ns]")
    )
    dims = ("member", "time", *signature.dims[2:])
    array = xr.DataArray(
        np.zeros((2, 1, *signature.shape[2:]), dtype=np.float32),
        dims=dims,
        coords=coords,
        attrs=signature.attrs,
    )
    output = regional_model.output_coords(array)
    assert output.dims == array.dims
    assert output.sizes["member"] == 2
    assert output.attrs[E2S_DYNAMIC_DIMS] == ()
    assert output.data.nbytes == 0
    np.testing.assert_array_equal(output.time, array.time)


def test_regional_signature_validation(regional_model):
    signature = regional_model.input_coords()
    assert isinstance(signature, xr.DataArray)
    with pytest.raises(ValueError, match="variable"):
        regional_model.output_coords(signature.assign_coords(variable=["v10m", "u10m"]))
    with pytest.raises(ValueError, match="size"):
        regional_model.output_coords(signature.isel({signature.dims[-1]: slice(1)}))
    with pytest.raises(ValueError, match="lead_time"):
        regional_model.output_coords(signature.isel(lead_time=slice(0, 0)))
    if not isinstance(regional_model, StormCastCONUS):
        with pytest.raises(ValueError, match="lead_time"):
            regional_model.output_coords(
                signature.assign_coords(
                    lead_time=np.array([-3, -1, 0], dtype="timedelta64[h]")
                )
            )
    else:
        bad_crs = signature.copy()
        bad_crs.attrs[E2S_CRS] = "EPSG:4326"
        with pytest.raises(ValueError, match="CRS"):
            regional_model.output_coords(bad_crs)


def test_precipitation_signature():
    model = PrecipitationAFNO.__new__(PrecipitationAFNO)
    torch.nn.Module.__init__(model)
    signature = model.input_coords()
    assert isinstance(signature, xr.DataArray)
    assert signature.shape == (0, 20, 720, 1440)
    assert signature.attrs[E2S_GRID_ID] == "fcn1"
    assert signature.data.nbytes == 0
    output = model.output_coords(signature)
    assert output.shape == (0, 1, 720, 1440)
    np.testing.assert_array_equal(output.coords["variable"], ["tp"])
    assert output.attrs[E2S_STATISTICS]["tp"]["modifier"] == "sum:6h"
    assert E2S_STATISTICS not in signature.attrs
    assert output.data.nbytes == 0
    with pytest.raises(ValueError, match="variable"):
        model.output_coords(signature.isel(variable=slice(1)))


def test_tensor_batching_with_public_signatures(regional_model):
    class TensorStep:
        input_coords = regional_model.input_coords
        output_coords = regional_model.output_coords
        _input_tensor_coords = regional_model._input_tensor_coords
        _output_tensor_coords = regional_model._output_tensor_coords

        @batch_func()
        def __call__(self, x, coords):
            return x, coords

    coords = regional_model._input_tensor_coords()
    coords["batch"] = np.arange(2)
    coords["time"] = np.array(["2026-09-17"], dtype="datetime64[ns]")
    coords = OrderedDict(
        ("member" if k == "batch" else k, v) for k, v in coords.items()
    )
    x = torch.zeros(tuple(len(v) for v in coords.values()))
    output, output_coords = TensorStep()(x, coords)
    assert output.shape == x.shape
    assert tuple(output_coords) == tuple(coords)


def test_precipitation_tensor_call():
    model = PrecipitationAFNO.__new__(PrecipitationAFNO)
    torch.nn.Module.__init__(model)
    model.core_model = torch.nn.Conv2d(20, 1, 1)
    model.center, model.scale, model.eps = 0.0, 1.0, 1e-5
    signature = model.input_coords()
    assert isinstance(signature, xr.DataArray)
    coords = OrderedDict(
        (str(d), np.asarray(signature.coords[d]))
        for d in signature.dims
        if d != "batch"
    )
    output, output_coords = model(torch.zeros(20, 720, 1440), coords)
    assert output.shape == (1, 720, 1440)
    np.testing.assert_array_equal(output_coords["variable"], ["tp"])
