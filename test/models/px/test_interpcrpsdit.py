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

import gc
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.interpcrpsdit as interp_mod
from earth2studio.grids import Region
from earth2studio.models.auto import Package
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import InterpCRPSDiT
from earth2studio.models.px.interpcrpsdit import (
    OPTIONAL_VARIABLES,
    PHYS_EXTRA,
    VARIABLES,
)
from earth2studio.models.px.pangu import VARIABLES as PANGU_VARIABLES
from earth2studio.models.px.persistence import Persistence
from earth2studio.utils import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure

# Small grid keeps the suite fast on CPU without weights or network access
H, W = 16, 32
NVAR = len(VARIABLES)
LAT = np.linspace(90, -90, H, endpoint=False)
LON = np.linspace(0, 360, W, endpoint=False)
Y0 = np.datetime64("2020-01-01T00:00:00")
T0 = Y0 + np.timedelta64(120, "h")
H0 = 120.0  # hours between Y0 and T0

# DiT input layout: [x0 (73), xT (73), cond (87), z (6)]
# with cond = static (12) + cos_zenith + gap + presence mask (73)
GAP_CH = 2 * NVAR + 13
MASK_CH = 2 * NVAR + 14

DEVICES = ["cpu", "cuda:0"]


@pytest.fixture(autouse=True)
def optional_backend(request, monkeypatch):
    if (
        interp_mod.PhysicsNemoModule is None
        and request.node.get_closest_marker("package") is None
    ):
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, interp_mod.__file__, raising=False
        )
        monkeypatch.setattr(
            interp_mod, "cos_zenith_angle", lambda t, lon, lat: np.ones_like(lon)
        )
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")


class PhooDiT(torch.nn.Module):
    # Zero residual (mean latent if noise); records inputs
    patch_size = (2, 2)

    def __init__(self, noise: bool = False, attn_kernel: int | None = None) -> None:
        super().__init__()
        self.noise = noise
        self.detokenizer = types.SimpleNamespace()
        self.inputs: list[torch.Tensor] = []
        self.saw_autocast: bool | None = None
        if attn_kernel is not None:
            self.attn_kernel = attn_kernel

    def forward(self, x, t, condition=None, attn_kwargs=None):
        self.inputs.append(x.detach().clone())
        self.saw_autocast = torch.is_autocast_enabled(x.device.type)
        out = x.new_zeros(x.shape[0], NVAR, x.shape[-2], x.shape[-1])
        if self.noise:
            out = out + x[:, -6:].mean(1, keepdim=True)
        return out


class Ramp(Persistence):
    # Persistence advancing every field by one unit per hour of lead time
    def _ramp(self, y):
        y.data = y.data + np.float32(self._dt / np.timedelta64(1, "h"))
        return y

    def initialize(self, x):
        y, state = super().initialize(x)
        return self._ramp(y), state

    def step(self, y, state):
        y, state = super().step(y, state)
        return self._ramp(y), state


def _model(
    gap_h=6, variables=VARIABLES, px_model=None, center=None, dit=None, **kwargs
):
    lon2d, lat2d = np.meshgrid(LON, LAT)
    kwargs.setdefault("lon_pad", 0)
    if px_model is None:
        px_model = Ramp(
            list(variables), {"lat": LAT, "lon": LON}, dt=np.timedelta64(gap_h, "h")
        )
    return InterpCRPSDiT(
        dit=dit if dit is not None else PhooDiT(),
        center=torch.zeros(1, NVAR, 1, 1) if center is None else center,
        scale=torch.ones(1, NVAR, 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=px_model,
        **kwargs,
    )


def make_input(model, time, device="cpu"):
    signature = model.input_coords()
    signature = coord_array_like(signature, {"batch": [0], "time": time}).isel(
        batch=0, drop=True
    )
    signature = coord_array_like(signature, {"time": time})
    return from_torch(torch.randn(signature.shape, device=device), signature)


def _input(model, times=(T0,), per_var=False, spatial=False):
    # Value = hours since Y0, plus variable index / grid row and column if requested
    times = np.array(times)
    x = make_input(model, times)
    hours = ((times - Y0) / np.timedelta64(1, "h")).astype("float32")
    arr = np.broadcast_to(hours[:, None, None, None, None], x.shape).copy()
    if per_var:
        index = [VARIABLES.index(v) for v in x.coords["variable"].values]
        arr += np.array(index, dtype="float32")[None, None, :, None, None]
    if spatial:
        rows = np.round((90 - x.lat.values) / (180 / H))
        cols = np.round(x.lon.values / (360 / W))
        arr += (rows[:, None] + 0.01 * cols[None, :]).astype("float32")
    x.data[...] = arr
    return x


def _values(out):
    return out.e2s.to_torch()[0].cpu().numpy()


def _close(actual, expected, atol=1e-4):
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected, np.shape(actual)), atol=atol
    )


def _lead_h(out):
    return (_minutes(out) / 60).reshape(-1, 1, 1, 1)


def _minutes(out):
    return (out.lead_time.values / np.timedelta64(1, "m")).astype(int)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [np.datetime64("1999-10-11T12:00"), np.datetime64("2001-06-04T00:00")]
        ),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_interpcrpsdit_call(time, device):
    model = _model(px_model=Persistence(VARIABLES, {"lat": LAT, "lon": LON})).to(device)
    x = make_input(model, time, device)

    out = model(x)

    assert out.shape == (len(time), 6, NVAR, H, W)
    assert out.dims == ("time", "lead_time", "variable", "lat", "lon")
    assert (
        out.coords["variable"].values
        == model.output_coords(x).coords["variable"].values
    ).all()
    assert (out.time.values == time).all()
    np.testing.assert_array_equal(_minutes(out), np.arange(1, 7) * 60)
    assert torch.isfinite(out.e2s.to_torch()[0]).all()

    # Zero residual: persistence base gives back the initial condition
    torch.testing.assert_close(
        out.e2s.to_torch()[0].cpu(),
        x.e2s.to_torch()[0].cpu().expand(out.shape),
    )

    out, state = model.initialize(x)
    resumed, _ = model.step(out, state)
    np.testing.assert_array_equal(_minutes(resumed), np.arange(7, 13) * 60)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_interpcrpsdit_iter(ensemble, device):
    model = _model().to(device)
    x = _input(model).expand_dims(ensemble=np.arange(ensemble)).copy(deep=True)
    x.name = "forecast"
    x.attrs["nested"] = {"owner": ["caller"]}
    before = x.copy(deep=True)

    events = []

    def front(field):
        events.append("front")
        return field

    def rear(field):
        events.append("rear")
        return field

    model.front_hook = front
    model.rear_hook = rear

    iterator = model.create_iterator(x)
    first = next(iterator)
    frozen = first.copy(deep=True)
    for i, out in enumerate([first, next(iterator)]):
        assert out.dims == ("ensemble", "time", "lead_time", "variable", "lat", "lon")
        assert out.shape == (ensemble, 1, 6, NVAR, H, W)
        assert out.name == x.name
        np.testing.assert_array_equal(_minutes(out), (6 * i + np.arange(1, 7)) * 60)
        # Ramp base and zero residual give value = hours since Y0 + lead
        _close(_values(out), H0 + _lead_h(out))
    assert events == ["rear", "front", "rear"]
    xr.testing.assert_identical(first, frozen)
    xr.testing.assert_identical(x, before)

    iterator.close()
    model.clear_hooks()
    if not device.startswith("cpu"):
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(H)}),
        OrderedDict({"lat": np.random.randn(H), "phoo": np.random.randn(W)}),
        OrderedDict({"lat": np.random.randn(H), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_interpcrpsdit_exceptions(dc, device):
    model = _model().to(device)
    signature = model.input_coords()
    coords = {
        "time": np.array([T0]),
        "lead_time": signature.coords["lead_time"].values,
        "variable": signature.coords["variable"].values,
        **dc,
    }
    x = xr.DataArray(
        np.zeros(tuple(len(v) for v in coords.values()), dtype=np.float32),
        dims=tuple(coords),
        coords=coords,
    )
    with pytest.raises((KeyError, ValueError)):
        model(x)


def test_interpcrpsdit_conformance():
    check_prognostic_contract(
        _model(
            px_model=Persistence(VARIABLES, {"lat": LAT, "lon": LON}),
            dit=PhooDiT(noise=True),
        )
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_interp_steps": 0}, "num_interp_steps"),
        ({"amp_dtype": torch.float64}, "amp_dtype must be"),
        ({"drop_variables": ["z500"]}, "trained-optional"),
    ],
)
def test_interpcrpsdit_invalid_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


@pytest.mark.parametrize(
    "gap_h, num_interp_steps",
    [(6, 6), (6, 36), (3, 3), (10, 10)],
)
def test_interpcrpsdit_gap_and_resolution(gap_h, num_interp_steps):
    model = _model(gap_h=gap_h, num_interp_steps=num_interp_steps)
    out = model(_input(model))

    step = gap_h * 60 // num_interp_steps
    np.testing.assert_array_equal(
        _minutes(out), step * np.arange(1, num_interp_steps + 1)
    )
    _close(_values(out), H0 + _lead_h(out), atol=1e-3)


@pytest.mark.parametrize("gap_h, value", [(3, -1.0), (6, 0.0), (9, 1.0)])
def test_interpcrpsdit_gap_channel(gap_h, value):
    dit = PhooDiT()
    model = _model(gap_h=gap_h, dit=dit, num_interp_steps=3)
    model(_input(model))
    assert dit.inputs[0][:, GAP_CH].mean().item() == pytest.approx(value)


@pytest.mark.parametrize(
    "gap_h, num_interp_steps, match",
    [(6, 7, "must divide")],
)
def test_interpcrpsdit_gap_exceptions(gap_h, num_interp_steps, match):
    model = _model(gap_h=gap_h, num_interp_steps=num_interp_steps)
    with pytest.raises(ValueError, match=match):
        model.output_coords(model.input_coords())


def test_interpcrpsdit_fractional_minute_gap():
    px_model = Persistence(
        VARIABLES, {"lat": LAT, "lon": LON}, dt=np.timedelta64(90, "s")
    )
    model = _model(px_model=px_model, num_interp_steps=1)
    with pytest.raises(ValueError, match="whole number of minutes"):
        model.output_coords(model.input_coords())


@pytest.mark.parametrize("gap_h", [12, 2])
def test_interpcrpsdit_out_of_range_gap_warns(gap_h):
    model = _model(gap_h=gap_h, num_interp_steps=2)
    with pytest.warns(UserWarning, match="outside the trained"):
        out = model(_input(model))
    assert torch.isfinite(out.e2s.to_torch()[0]).all()


@pytest.mark.parametrize("amp_dtype, autocast", [(None, False), (torch.bfloat16, True)])
def test_interpcrpsdit_amp_dtype(amp_dtype, autocast):
    dit = PhooDiT()
    model = _model(amp_dtype=amp_dtype, dit=dit)
    out = model(_input(model))
    assert dit.saw_autocast is autocast
    assert torch.isfinite(out.e2s.to_torch()[0]).all()


@pytest.mark.parametrize("lon_pad", [4, 2 * W + 8])
def test_interpcrpsdit_lon_pad(lon_pad):
    model = _model(lon_pad=lon_pad)
    out = model(_input(model))
    assert out.shape[-2:] == (H, W)
    _close(_values(out), H0 + _lead_h(out))


def test_interpcrpsdit_noise():
    times = (T0, T0 + np.timedelta64(12, "h"))

    def run(seed):
        model = _model(dit=PhooDiT(noise=True))
        x = _input(model, times).expand_dims(ensemble=2).copy(deep=True)
        x.data[...] = 0.0
        model.set_rng(seed)
        return _values(model(x))[:, :, 0]

    a, b, c = run(0), run(0), run(1)
    assert np.allclose(a, b)  # same seed reproduces
    assert not np.allclose(a, c)  # different seed differs
    assert not np.allclose(a[0], a[1])  # members differ
    assert not np.allclose(a[:, 0], a[:, 1])  # init times differ


def test_interpcrpsdit_bracket_latent():
    dit = PhooDiT()
    model = _model(num_interp_steps=3, dit=dit)
    model.set_rng(0)
    iterator = model.create_iterator(_input(model))
    next(iterator)
    next(iterator)
    z = [inp[:, -6:] for inp in dit.inputs]
    assert len(z) == 4
    assert z[0].abs().sum() > 0
    assert torch.equal(z[0], z[1])  # reused within a bracket
    assert torch.equal(z[2], z[3])
    assert not torch.equal(z[0], z[2])  # redrawn for the next bracket


@pytest.mark.parametrize(
    "base_variables",
    [
        pytest.param(VARIABLES, id="all-73"),
        pytest.param(PANGU_VARIABLES, id="pangu-69"),
    ],
)
def test_interpcrpsdit_drop_variables(base_variables):
    present = [v for v in VARIABLES if v not in OPTIONAL_VARIABLES]
    drop_idx = [VARIABLES.index(v) for v in OPTIONAL_VARIABLES]
    dit = PhooDiT()
    model = _model(
        variables=base_variables,
        center=torch.full((1, NVAR, 1, 1), 5.0),
        drop_variables=OPTIONAL_VARIABLES,
        dit=dit,
    )
    assert list(model.drop_idx) == drop_idx

    out = model(_input(model, per_var=True))
    assert list(out.coords["variable"].values) == present
    for v in present:
        expected = H0 + VARIABLES.index(v) + _minutes(out) / 60
        np.testing.assert_allclose(
            _values(out.sel(variable=v))[0, :, 0, 0], expected, atol=1e-4
        )

    assert torch.all(dit.inputs[0][:, drop_idx] == 0)
    assert torch.all(dit.inputs[0][:, [MASK_CH + i for i in drop_idx]] == 0)


@pytest.mark.parametrize(
    "halo, exp_halo, base",
    [
        pytest.param(0, (0, 0, 1, 0), "regional", id="halo0-regional"),
        pytest.param(2, (2, 2, 3, 2), "regional", id="halo2-regional"),
        pytest.param(2, (2, 2, 3, 2), "global", id="halo2-global"),
    ],
)
def test_interpcrpsdit_set_domain(halo, exp_halo, base):
    sub = _model().set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=halo, min_cells=4
    )
    assert sub._halo == exp_halo
    if base == "regional":
        sub.px_model = Ramp(
            VARIABLES,
            {"lat": sub.lat2d[:, 0].numpy(), "lon": sub.lon2d[0].numpy()},
        )

    out = sub(_input(sub, spatial=True))

    np.testing.assert_array_equal(out.lat.values, LAT[2:8])
    np.testing.assert_array_equal(out.lon.values, LON[5:18])
    block = (
        np.arange(2, 8, dtype="float32")[:, None]
        + 0.01 * np.arange(5, 18, dtype="float32")[None, :]
    )
    expected = H0 + (_minutes(out) / 60)[:, None, None] + block
    np.testing.assert_allclose(_values(out)[0, :, 0], expected, atol=1e-4)


def test_interpcrpsdit_with_region():
    model = _model()
    model._min_domain_cells = 4
    sub = model.with_region(Region.box(west=50, south=10, east=200, north=70))
    out = sub(_input(sub))
    np.testing.assert_array_equal(out.lat.values, LAT[2:8])
    np.testing.assert_array_equal(out.lon.values, LON[5:18])
    assert model.output_coords(model.input_coords()).coords["lat"].size == H

    with pytest.raises(ValueError, match="only box regions"):
        model.with_region(Region.band(south=10, north=70))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lat_min": -100}, "outside the current grid"),
        ({"min_cells": 1000}, "per-side minimum"),
        ({"halo": -1}, "halo must be non-negative"),
        ({"min_cells": 0}, "min_cells must be >= 1"),
    ],
)
def test_interpcrpsdit_set_domain_exceptions(kwargs, match):
    args = dict(lat_min=10, lat_max=70, lon_min=50, lon_max=200, min_cells=4)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _model().set_domain(**args)


def test_interpcrpsdit_superset_base_grid():
    lat_big = np.concatenate([LAT, [LAT[-1] - (LAT[0] - LAT[1])]])
    model = _model(px_model=Ramp(VARIABLES, {"lat": lat_big, "lon": LON}))
    np.testing.assert_array_equal(model.input_coords().coords["lat"].values, lat_big)

    out = model(_input(model, spatial=True))
    np.testing.assert_array_equal(out.lat.values, LAT)
    block = (
        np.arange(H, dtype="float32")[:, None]
        + 0.01 * np.arange(W, dtype="float32")[None, :]
    )
    expected = H0 + (_minutes(out) / 60)[:, None, None] + block
    np.testing.assert_allclose(_values(out)[0, :, 0], expected, atol=1e-4)


def test_interpcrpsdit_off_grid_base_raises():
    model = _model(px_model=Ramp(VARIABLES, {"lat": LAT + 0.3, "lon": LON}))
    with pytest.raises((KeyError, ValueError)):
        model(_input(model))


def test_interpcrpsdit_front_hook():
    model = _model(num_interp_steps=3)
    model.front_hook = lambda a: a + 5.0
    iterator = model.create_iterator(_input(model))
    first, second = next(iterator), next(iterator)
    _close(_values(first), H0 + _lead_h(first))
    # Edits made before a step feed the base forecast and the next bracket
    _close(_values(second), H0 + 5 + _lead_h(second))


@pytest.mark.parametrize("inplace", [False, True])
def test_interpcrpsdit_rear_hook(inplace):
    def rear(a):
        if inplace:
            a += 100.0
            return a
        return a + 100.0

    model = _model(num_interp_steps=3)
    model.rear_hook = rear
    iterator = model.create_iterator(_input(model))
    for k in range(3):
        out = next(iterator)
        # Hooked output is the next recurrence input, so offsets accumulate
        _close(_values(out), H0 + _lead_h(out) + 100 * (k + 1), atol=1e-3)


def test_interpcrpsdit_to_moves_px_model():
    model = _model()
    calls = []
    model.px_model.to = lambda d: calls.append(d) or model.px_model
    assert model.to("cpu") is model
    assert calls == ["cpu"]


def _write_package(tmp_path, monkeypatch, nlat=4, means=None, stds=None, dit=None):
    class _FakeModule:
        @staticmethod
        def from_checkpoint(path):
            return dit if dit is not None else PhooDiT()

    monkeypatch.setattr(interp_mod, "PhysicsNemoModule", _FakeModule)
    (tmp_path / "CRPSModel.mdlus").write_bytes(b"")
    (tmp_path / "config.json").write_text('{"name": "test"}')
    np.save(
        tmp_path / "global_means.npy",
        np.zeros(NVAR, "float32") if means is None else means,
    )
    np.save(
        tmp_path / "global_stds.npy",
        np.ones(NVAR, "float32") if stds is None else stds,
    )
    nlon = 8
    lat = np.linspace(90, -90, nlat, endpoint=(nlat == 721))
    lon = np.linspace(0, 360, nlon, endpoint=False)
    ramp = np.arange(nlat * nlon, dtype="float32").reshape(nlat, nlon)
    xr.Dataset(
        {
            v: (["latitude", "longitude"], ramp + i)
            for i, v in enumerate(["lsm", "z"] + PHYS_EXTRA)
        },
        coords=dict(latitude=lat, longitude=lon),
    ).to_netcdf(tmp_path / "set_phys.nc")
    return Package(str(tmp_path))


@pytest.mark.parametrize(
    "nlat, attn_kernel, exp_nlat, exp_cells, exp_last_lat",
    [
        pytest.param(4, None, 4, 64, -45.0, id="small"),
        pytest.param(721, 23, 720, 46, -89.75, id="721-crop"),
    ],
)
def test_interpcrpsdit_load_model(
    tmp_path, monkeypatch, nlat, attn_kernel, exp_nlat, exp_cells, exp_last_lat
):
    package = _write_package(
        tmp_path, monkeypatch, nlat=nlat, dit=PhooDiT(attn_kernel=attn_kernel)
    )
    px_model = Persistence(VARIABLES, {"lat": LAT, "lon": LON})
    model = InterpCRPSDiT.load_model(package, px_model=px_model)
    assert model._pole is (nlat == 721)
    assert model.center.shape == (1, NVAR, 1, 1)
    assert model.static_inv.shape == (12, exp_nlat, 8)
    assert torch.isfinite(model.static_inv).all()
    assert model.lat2d.shape == (exp_nlat, 8)
    assert model.lat2d[-1, 0].item() == pytest.approx(exp_last_lat)
    assert model._min_domain_cells == exp_cells


@pytest.mark.parametrize(
    "kind, match",
    [
        ("zero_std", "strictly positive"),
        ("short_means", "must have"),
        ("nonfinite_std", "non-finite"),
    ],
)
def test_interpcrpsdit_load_model_bad_normalization(tmp_path, monkeypatch, kind, match):
    means, stds = np.zeros(NVAR, "float32"), np.ones(NVAR, "float32")
    if kind == "zero_std":
        stds[3] = 0.0
    elif kind == "short_means":
        means = means[:-1]
    else:
        stds[5] = np.nan
    package = _write_package(tmp_path, monkeypatch, means=means, stds=stds)
    px_model = Persistence(VARIABLES, {"lat": LAT, "lon": LON})
    with pytest.raises(ValueError, match=match):
        InterpCRPSDiT.load_model(package, px_model=px_model)


@pytest.fixture(scope="function")
def model() -> InterpCRPSDiT:
    package = InterpCRPSDiT.load_default_package()
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    px_model = Persistence(VARIABLES, {"lat": lat, "lon": lon})
    return InterpCRPSDiT.load_model(package, px_model=px_model)


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NATTEN DiT requires CUDA")
def test_interpcrpsdit_package(model):
    device = "cuda:0"
    torch.cuda.empty_cache()
    model = model.to(device)
    sub = model.with_region(Region.box(west=100, south=10, east=124, north=32))

    time = np.array([np.datetime64("1993-04-05T00:00")])
    x = make_input(sub, time, device)
    center = model.center.expand(1, 1, NVAR, *x.shape[-2:]).clone()
    x = from_torch(center, x)

    out = sub(x)
    expected = sub.output_coords(x)
    assert out.shape == (1, 6, NVAR, *expected.shape[-2:])
    assert out.lead_time.values[-1] == np.timedelta64(6, "h")
    assert torch.isfinite(out.e2s.to_torch()[0]).all()
