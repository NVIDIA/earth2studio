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


import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    ProjectedGrid,
    infer_grid,
    resolve_grid,
)
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.precipitation_afno import PrecipitationAFNO
from earth2studio.models.px.stormcastconus import StormCastCONUS
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS
from earth2studio.utils.coords import (
    E2S_DYNAMIC_DIMS,
    E2S_STATISTICS,
    coord_array,
    coord_array_like,
)


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
        model.grid = ProjectedGrid(grid.y[10:12], grid.x[20:23], grid.crs)
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
        assert signature.dims[-2:] == ("y", "x")
        assert signature.attrs[E2S_CRS] == resolve_grid("hrrr").crs.to_string()
        assert signature.attrs["dims"] == ["y", "x"]
        np.testing.assert_array_equal(signature.y, regional_model.grid.y)
        np.testing.assert_array_equal(signature.x, regional_model.grid.x)
        assert infer_grid(signature).fingerprint() == regional_model.grid.fingerprint()
        np.testing.assert_array_equal(regional_model.hrrr_y, signature.y)
        np.testing.assert_array_equal(regional_model.hrrr_x, signature.x)
        assert "hrrr_y" not in regional_model.__dict__
        assert "hrrr_x" not in regional_model.__dict__
        cropped = regional_model.grid
        np.testing.assert_array_equal(signature.lat, cropped.coords()["lat"])
        np.testing.assert_array_equal(signature.lon, cropped.coords()["lon"])
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


@pytest.mark.parametrize("kind", ["datetime", "integer", "nat", "missing"])
def test_regional_lead_time_type(regional_model, kind):
    signature = regional_model.input_coords()
    lead = np.asarray(signature.lead_time)
    if kind == "datetime":
        invalid = signature.assign_coords(lead_time=np.datetime64("2026-09-17") + lead)
    elif kind == "integer":
        invalid = signature.assign_coords(lead_time=lead.astype(np.int64))
    elif kind == "nat":
        lead = lead.copy()
        lead[-1] = np.timedelta64("NaT")
        invalid = signature.assign_coords(lead_time=lead)
    else:
        invalid = signature.drop_vars("lead_time")
    with pytest.raises(ValueError, match="lead_time"):
        regional_model.output_coords(invalid)


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
    np.testing.assert_array_equal(output.coords["variable"], ["tp:sum:6h"])
    assert output.attrs[E2S_STATISTICS]["tp:sum:6h"]["modifier"] == "sum:6h"
    assert E2S_STATISTICS not in signature.attrs
    assert output.data.nbytes == 0
    with pytest.raises(ValueError, match="variable"):
        model.output_coords(signature.isel(variable=slice(1)))


def test_tensor_batching_with_public_signatures(regional_model):
    class TensorStep:
        input_coords = regional_model.input_coords
        output_coords = regional_model.output_coords

        @batch_func()
        def __call__(self, x, coords):
            return x, coords

    coords = coord_array_like(
        regional_model.input_coords(),
        {
            "batch": np.arange(2),
            "time": np.array(["2026-09-17"], dtype="datetime64[ns]"),
        },
    ).rename(batch="member")
    coords = coords.assign_coords(member_label=("member", ["a", "b"]), source="test")
    coords.member_label.attrs["description"] = "member identity"
    x = torch.zeros(coords.shape)
    output, output_coords = TensorStep()(x, coords)
    assert output.shape == x.shape
    assert output_coords.dims == coords.dims
    assert output_coords.attrs == coords.attrs
    xr.testing.assert_identical(
        output_coords.coords.to_dataset(), coords.coords.to_dataset()
    )
    assert output_coords.data.nbytes == 0


@pytest.mark.parametrize("batch_kind", ["none", "batch", "multiple"])
def test_coordinate_pair_batch_layout(regional_model, batch_kind):
    class Step:
        input_coords = regional_model.input_coords

        @batch_func()
        def __call__(self, x, coords):
            return x + 1, coords

    coords = coord_array_like(
        regional_model.input_coords(),
        {"batch": [0, 1], "time": np.array(["2026-09-17"], dtype="datetime64[ns]")},
    )
    if batch_kind == "none":
        coords = coords.isel(batch=0, drop=True).assign_coords(batch="source")
    elif batch_kind == "multiple":
        coords = coords.rename(batch="member")
        coords = coord_array(
            ("ensemble", *coords.dims),
            {**dict(coords.coords), "ensemble": [0, 1, 2]},
            attrs=coords.attrs,
        )
    x = torch.zeros(coords.shape)
    output, actual = Step()(x, coords)
    torch.testing.assert_close(output, x + 1)
    assert actual.dims == coords.dims
    xr.testing.assert_identical(actual.coords.to_dataset(), coords.coords.to_dataset())
    with pytest.raises(ValueError, match="shape"):
        Step()(x.unsqueeze(0), coords)


def test_coordinate_pair_rejects_batch_reordering(regional_model):
    class Step:
        input_coords = regional_model.input_coords

        @batch_func()
        def __call__(self, x, coords):
            return x.flip(0), coords.isel(batch=slice(None, None, -1))

    coords = coord_array_like(
        regional_model.input_coords(),
        {"batch": [0, 1], "time": np.array(["2026-09-17"], dtype="datetime64[ns]")},
    )
    with pytest.raises(ValueError, match="batch"):
        Step()(torch.zeros(coords.shape), coords)


def test_coordinate_pair_preserves_output_metadata(regional_model):
    class Step:
        input_coords = regional_model.input_coords

        @batch_func()
        def __call__(self, x, coords):
            return x, coords.assign_coords(source="forecast", quality=("batch", [3, 4]))

    coords = (
        coord_array_like(
            regional_model.input_coords(),
            {"batch": [0, 1], "time": np.array(["2026-09-17"], dtype="datetime64[ns]")},
        )
        .rename(batch="member")
        .assign_coords(source="input", quality=("member", [1, 2]))
    )
    _, output = Step()(torch.zeros(coords.shape), coords)
    assert output.source.item() == "forecast"
    np.testing.assert_array_equal(output.quality, [3, 4])


def test_precipitation_array_call():
    model = PrecipitationAFNO.__new__(PrecipitationAFNO)
    torch.nn.Module.__init__(model)
    model.core_model = torch.nn.Conv2d(20, 1, 1)
    model.center, model.scale, model.eps = 0.0, 1.0, 1e-5
    signature = model.input_coords()
    assert isinstance(signature, xr.DataArray)
    coords = coord_array_like(signature, {"batch": [0]}).isel(batch=0, drop=True)
    from earth2studio.utils.cupy import from_torch

    output = model(from_torch(torch.zeros(20, 720, 1440), coords))
    assert output.shape == (1, 720, 1440)
    np.testing.assert_array_equal(output["variable"], ["tp:sum:6h"])
    assert output.attrs[E2S_STATISTICS]["tp:sum:6h"]["modifier"] == "sum:6h"


def test_regional_tensor_rollout(regional_model):
    model = regional_model
    coords = coord_array_like(
        model.input_coords(),
        {"batch": [0, 1], "time": np.array(["2026-09-17"], dtype="datetime64[ns]")},
    ).rename(batch="member")
    coords = coords.assign_coords(source="test", member_label=("member", ["a", "b"]))
    x = torch.ones(coords.shape)
    if isinstance(model, StormCastCONUS):
        model.batch_size = 1
        model.conditioning_data_source = object()
        model._get_conditioning = lambda c, b, d: torch.zeros((b, 1, 1, 2, 2, 3))
        model._forward = lambda state, *args, **kwargs: state + 1
    else:
        model.input_interp = None
        model.valid_mask = torch.ones(2, 3, dtype=torch.bool)
        model.conditioning_variables = None
        model.sliding_window = True
        model._inject_auto_observations = lambda state, c: state
        model._forward = lambda state, c, **kwargs: state[:, :, -2:] + 1
    y, output = model(x, coords)
    assert output.dims == coords.dims
    assert tuple(y.shape) == output.shape
    assert output.data.nbytes == 0
    xr.testing.assert_identical(output.lat, coords.lat)
    torch.testing.assert_close(x, torch.ones_like(x))
    iterator = model.create_iterator(x, coords)
    next(iterator)
    first, first_coords = next(iterator)
    second, second_coords = next(iterator)
    assert first_coords.dims == second_coords.dims == coords.dims
    assert second_coords.data.nbytes == 0
    assert (
        np.asarray(second_coords.lead_time)[-1] > np.asarray(first_coords.lead_time)[-1]
    )
    torch.testing.assert_close(second, first + 1)
    iterator.close()
    assert not hasattr(model, "_input_tensor_coords")
    if not isinstance(model, StormCastCONUS):
        model.conditioning_interp = None
        model.conditioning_valid_mask = model.valid_mask
        coupled, coupled_coords = model.call_with_conditioning(x, coords, x, coords)
        torch.testing.assert_close(coupled, y)
        assert coupled_coords.dims == coords.dims
        source = coord_array(
            (*coords.dims[:-2], "lat", "lon"),
            {
                **{str(d): coords[d].variable for d in coords.dims[:-2]},
                "lat": [30.0],
                "lon": [250.0],
            },
        )
        model.input_interp = lambda state: state.expand(*state.shape[:-2], 2, 3)
        regridded, native = model(torch.ones(source.shape), source)
        assert native.dims == coords.dims
        torch.testing.assert_close(regridded, y)


def test_fcn_array_rollout():
    from earth2studio.models.px.fcn import FCN
    from earth2studio.utils.coords import coord_array
    from earth2studio.utils.cupy import from_torch

    model = FCN(torch.nn.Identity(), torch.tensor(0.0), torch.tensor(1.0))
    model.input_coords = lambda: coord_array(
        ("batch", "lead_time", "variable", "lat", "lon"),
        {
            "lead_time": np.array([0], dtype="timedelta64[h]"),
            "variable": ["u10m"],
            "lat": [30.0],
            "lon": [250.0],
        },
        dynamic=("batch",),
    )
    coords = coord_array_like(model.input_coords(), {"batch": [0, 1]}).rename(
        batch="member"
    )
    x = torch.ones(coords.shape)
    iterator = model.create_iterator(from_torch(x, coords))
    for step in range(3):
        output = next(iterator)
        y, _ = output.e2s.to_torch()
        torch.testing.assert_close(y, x)
        assert output.dims == coords.dims
        np.testing.assert_array_equal(
            output.lead_time, np.array([step * 6], dtype="timedelta64[h]")
        )
    iterator.close()
