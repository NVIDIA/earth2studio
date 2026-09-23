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
from collections.abc import Iterable

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.interpmodafno as interp_module
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.dx import DerivedWS
from earth2studio.models.px import DiagnosticWrapper, InterpModAFNO
from earth2studio.models.px.interpmodafno import VARIABLES
from earth2studio.models.px.persistence import Persistence
from earth2studio.utils import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def optional_backend(request, monkeypatch):
    if (
        interp_module.PhysicsNemoModule is None
        and request.node.get_closest_marker("package") is None
    ):
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, interp_module.__file__, raising=False
        )
        monkeypatch.setattr(
            interp_module, "cos_zenith_angle", lambda t, lon, lat: np.ones_like(lon)
        )
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")


def make_input(model, time, device="cpu"):
    signature = model.input_coords()
    signature = coord_array_like(signature, {"batch": [0], "time": time}).isel(
        batch=0, drop=True
    )
    signature = coord_array_like(signature, {"time": time})
    return from_torch(torch.randn(signature.shape, device=device), signature)


class PhooInterpolationModel(torch.nn.Module):
    """Mock interpolation model for testing."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x, t_norm):
        self.batch_sizes.append(x.shape[0])
        return x[:, :73]


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
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_call(time, device, tmp_path):
    """Test basic forward pass of InterpModAFNO model."""
    # Set up base model
    base_model = Persistence(
        variable=[*VARIABLES, "extra"],
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 721),
            "lon": np.arange(1440)[::-1] * 0.25,
        },
        history=2,
    )
    center = torch.ones(1, 73, 1, 1)
    scale = torch.full((1, 73, 1, 1), 2.0)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6 if len(time) == 1 else 1,
    ).to(device)

    # Create domain coordinates
    x = make_input(model, time)
    x.data[...] = x.lon.values
    coords = x

    # Run forward pass
    out = model(x)
    out_coords = out.coords
    assert out.lead_time.values[0] == np.timedelta64(6 // model.num_interp_steps, "h")
    torch.testing.assert_close(
        out.e2s.to_torch()[0].cpu(),
        x.isel(lead_time=slice(-1, None))
        .sel(variable=VARIABLES, lat=out.lat, lon=out.lon)
        .e2s.to_torch()[0],
    )

    if not isinstance(time, Iterable):
        time = [time]

    # Verify output shape and coordinates
    assert out.shape == torch.Size([len(time), 1, 73, 720, 1440])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out.dims == ("time", "lead_time", "variable", "lat", "lon")

    from earth2studio.utils.checkpoint import Checkpoint

    checkpoint = Checkpoint("interp", path=tmp_path, mode="append", level=2)
    with checkpoint as ckpt:
        saved_base = Persistence(
            [*VARIABLES, "extra"], {"lat": x.lat.values, "lon": x.lon.values}, history=2
        )
        iterator = saved_base.create_iterator(x)
        next(iterator)
        saved = next(iterator)
        ckpt.write(lead_time=saved.lead_time.values[-1])
        ckpt.flush()
        iterator.close()
    with checkpoint.select(-1):
        restored = Persistence(
            [*VARIABLES, "extra"], {"lat": x.lat.values, "lon": x.lon.values}, history=2
        )
        model.px_model = restored
        events = []

        def front(state):
            events.append(("front", state.lead_time.values[-1]))
            return state.copy(data=state.data + 10)

        def rear(state):
            events.append(("rear", state.lead_time.values[-1]))
            return state

        model.front_hook, model.rear_hook = front, rear
        resumed = model.create_iterator(x)
        first = next(resumed)
        assert first.sizes["variable"] == 73 and first.sizes["lat"] == 720
        assert first.lead_time.values[0] == np.timedelta64(
            6 + 6 // model.num_interp_steps, "h"
        )
        assert events == [
            ("front", np.timedelta64(6, "h")),
            ("rear", first.lead_time.values[0]),
        ]
        np.testing.assert_allclose(
            first.e2s.to_torch()[0].cpu().numpy()[0, 0, 0, 0], first.lon.values + 10
        )
        resumed.close()
        model.clear_hooks()


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_iter(ensemble, history, device):
    """Test iteration functionality of InterpModAFNO model."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Set up base model
    base_model = Persistence(
        variable=VARIABLES,
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
            "lon": np.arange(1440)[::-1] * 0.25,
        },
        history=history,
    )
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    ).to(device)

    # Create domain coordinates
    x = (
        make_input(model, time)
        .expand_dims(ensemble=np.arange(ensemble))
        .copy(deep=True)
    )
    x = x.assign_coords(lead_time=x.lead_time + np.timedelta64(12, "h"))
    x.name = "forecast"
    x.attrs["nested"] = {"owner": ["caller"]}
    x.encoding = {"nested": {"owner": ["caller"]}}
    x = x.assign_coords(
        aux=("ensemble", np.arange(ensemble)), units=("variable", ["input"] * 73)
    )
    if history == 2:
        x = x.drop_vars("ensemble")
    x.data[...] = x.lon.values
    before = x.copy(deep=True)
    events = []

    def front(field):
        events.append("front")
        assert field.dims == x.dims
        field.attrs["nested"]["owner"].append("front")
        return field.drop_vars("aux", errors="ignore")

    def rear(field):
        events.append("rear")
        return field

    model.front_hook = front
    model.rear_hook = rear
    solar_calls = []
    solar = model._cos_zenith

    def count_solar(times):
        solar_calls.append(times)
        return solar(times)

    model._cos_zenith = count_solar

    # Create iterator
    model_iter = model.create_iterator(x)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    initial = next(model_iter)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    assert not hasattr(model, "sincos_latlon")

    # Test interpolation steps
    frozen = None
    for i, out in enumerate(model_iter):
        np.testing.assert_allclose(
            out.e2s.to_torch()[0].cpu().numpy()[0, 0, 0, 0, 0], out.lon.values
        )
        out_coords = out.coords
        if frozen is None:
            first = out
            frozen = out.copy(deep=True)
        else:
            xr.testing.assert_identical(first, frozen)
        assert out.name == x.name and out.encoding == x.encoding
        assert "aux" not in out.coords and "units" not in out.coords
        np.testing.assert_array_equal(
            out.valid_time, out.time.values[:, None] + out.lead_time.values
        )

        # Check output shape
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 73, 720, 1440])

        # Check coordinates
        assert (
            out_coords["variable"]
            == model.output_coords(model.input_coords())["variable"]
        ).all()
        if "ensemble" in x.coords:
            assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        else:
            assert "ensemble" not in out.coords

        # Check lead time - should be 1 hour increments due to interpolation
        assert out_coords["lead_time"][0] == np.timedelta64(12 + i + 1, "h")

        # Break after testing a few steps
        if i > 10:
            break
    assert events == ["front", *(["rear"] * 6)] * 2
    assert interp_model.batch_sizes == [ensemble] * 10
    assert len(solar_calls) == 10
    xr.testing.assert_identical(x, before)
    model_iter.close()
    model.clear_hooks()


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_exceptions(dc, device):
    """Test exception handling for invalid inputs in InterpModAFNO model."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Set up base model
    base_model = Persistence(
        variable=VARIABLES,
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
    )
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    ).to(device)

    coords = {
        "time": time,
        "lead_time": model.input_coords().lead_time.values,
        "variable": model.input_coords().coords["variable"].values,
        **dc,
    }
    x = xr.DataArray(
        np.zeros(tuple(len(v) for v in coords.values()), dtype=np.float32),
        dims=tuple(coords),
        coords=coords,
    )

    # Expect an exception when running the model with invalid inputs
    with pytest.raises((KeyError, ValueError)):
        model(x)

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        num_interp_steps=6,
    ).to(device)
    with pytest.raises(ValueError):
        model.input_coords()


def test_interpmodafno_conformance():
    base_model = Persistence(
        variable=VARIABLES,
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
    )
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)

    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)
    lsm = torch.zeros(1, 1, 720, 1440)

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    )
    check_prognostic_contract(model)

    # The public initial condition can lack a diagnosed interpolation channel.
    # Use a three-channel composition retaining the 721 -> 720 grid boundary.
    base = Persistence(
        ["u10m", "v10m"],
        {"lat": np.linspace(90, -90, 721), "lon": np.arange(1440) * 0.25},
    )
    from earth2studio.grids import LatLonGrid

    diagnostic = DerivedWS(
        ["10m"],
        grid=LatLonGrid(base.input_coords().lat.values, base.input_coords().lon.values),
    )
    wrapper = DiagnosticWrapper(base, diagnostic)
    model.px_model = wrapper
    model.variables = np.array(["u10m", "v10m", "ws10m"])
    model.center = torch.zeros(1, 3, 1, 1)
    model.scale = torch.ones(1, 3, 1, 1)
    model.prepare_endpoint = wrapper._diagnose

    def interpolate(left, right):
        np.testing.assert_allclose(left.sel(variable="ws10m"), 5)
        assert left.sizes["lat"] == right.sizes["lat"] == 720
        yield left.assign_coords(lead_time=left.lead_time + np.timedelta64(1, "h"))

    model._interpolate = interpolate
    field = make_input(model, np.array([np.datetime64("2024-01-01")]))
    field.loc[{"variable": "u10m"}] = 3
    field.loc[{"variable": "v10m"}] = 4
    iterator = model.create_iterator(field)
    xr.testing.assert_identical(next(iterator), field)
    assert next(iterator).sizes["variable"] == 3
    iterator.close()
    assert model(field).sizes["variable"] == 3


@pytest.fixture(scope="function")
def model() -> InterpModAFNO:
    base_model = Persistence(
        variable=VARIABLES,
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
    )
    # Load the interpolation model
    interp_package = InterpModAFNO.load_default_package()
    model = InterpModAFNO.load_model(interp_package, px_model=base_model)
    return model


@pytest.mark.package
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_package(device, model):
    """Test loading and using the InterpModAFNO model from a package."""
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Test the cached model package
    model = model.to(device)

    # Create domain coordinates
    x = make_input(model, time, device)
    coords = x

    # Run forward pass
    out = model(x)
    out_coords = out.coords

    if not isinstance(time, Iterable):
        time = [time]

    # Verify output shape and coordinates
    assert out.shape == torch.Size([len(time), 1, 73, 720, 1440])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert out.dims == ("time", "lead_time", "variable", "lat", "lon")
