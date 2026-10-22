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
import itertools
import types
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.interpcrpsdit as interp_mod
from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import InterpCRPSDiT
from earth2studio.models.px.datareplay import DataReplay
from earth2studio.models.px.interpcrpsdit import (
    OPTIONAL_VARIABLES,
    PHYS_EXTRA,
    VARIABLES,
)
from earth2studio.models.px.pangu import VARIABLES as PANGU_VARIABLES
from earth2studio.models.px.persistence import Persistence
from earth2studio.utils import handshake_dim

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

DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
    ),
]


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


class Source:
    # Value = hours since Y0, plus row/col if spatial, variable index if per_var
    def __init__(
        self, lat=LAT, lon=LON, spatial=False, per_var=False, static=False
    ) -> None:
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.spatial, self.per_var, self.static = spatial, per_var, static

    def __call__(self, time, variable):
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = [variable] if isinstance(variable, str) else list(variable)
        field = np.zeros((len(self.lat), len(self.lon)), dtype="float32")
        if self.spatial:
            rows = np.round((90 - self.lat) / (180 / H))
            cols = np.round(self.lon / (360 / W))
            field = (rows[:, None] + 0.01 * cols[None, :]).astype("float32")
        arr = np.zeros((len(times), len(variable)) + field.shape, dtype="float32")
        for i, t in enumerate(times):
            hours = 0.0 if self.static else float((t - Y0) / np.timedelta64(1, "h"))
            for j, v in enumerate(variable):
                offset = VARIABLES.index(v) if self.per_var else 0
                arr[i, j] = hours + offset + field
        return xr.DataArray(
            arr,
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=times, variable=np.array(variable), lat=self.lat, lon=self.lon
            ),
        )


def _replay(src, variables=VARIABLES, step=np.timedelta64(6, "h")):
    return DataReplay(
        src,
        list(variables),
        OrderedDict(lat=src.lat, lon=src.lon),
        step=step,
    )


def _model(px_model=None, center=None, dit=None, **kwargs):
    lon2d, lat2d = np.meshgrid(LON, LAT)
    kwargs.setdefault("lon_pad", 0)
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


def _inputs(src, variables=VARIABLES, times=(T0,), batch=1):
    times = np.array(times)
    x = torch.from_numpy(src(times, list(variables)).values.astype("float32"))
    x = x[None, :, None].repeat(batch, 1, 1, 1, 1, 1)
    coords = OrderedDict(
        batch=np.arange(batch),
        time=times,
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(variables),
        lat=src.lat,
        lon=src.lon,
    )
    return x, coords


def _build(
    src=None,
    variables=VARIABLES,
    step=np.timedelta64(6, "h"),
    times=(T0,),
    batch=1,
    **kwargs,
):
    src = src if src is not None else Source()
    model = _model(px_model=_replay(src, variables, step), **kwargs)
    x, coords = _inputs(src, variables, times, batch)
    return model, x, coords


def _random_inputs(model, time, device):
    ic = model.input_coords()
    r = Random({k: ic[k] for k in ["lat", "lon"]})
    return fetch_data(r, time, ic["variable"], ic["lead_time"], device=device)


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
    x, coords = _random_inputs(model, time, device)

    out, out_coords = model(x, coords)

    if not isinstance(time, Iterable):
        time = [time]
    assert out.shape == torch.Size([len(time), 1, NVAR, H, W])
    assert torch.allclose(out, x)  # __call__ returns the initial condition
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_interpcrpsdit_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = _model(
        px_model=Persistence(VARIABLES, {"lat": LAT, "lon": LON}), num_interp_steps=6
    ).to(device)
    x, coords = _random_inputs(model, time, device)

    # Add ensemble dim to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    model_iter = model.create_iterator(x, coords)
    assert isinstance(model_iter, Iterable)

    # First step is the initial condition
    out, out_coords = next(model_iter)
    assert torch.allclose(out, x)
    assert out_coords["lead_time"][0] == np.timedelta64(0, "h")

    for i, (out, out_coords) in enumerate(model_iter):
        assert out.shape == torch.Size([ensemble, len(time), 1, NVAR, H, W])
        assert (out_coords["variable"] == np.array(VARIABLES)).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")
        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        if i > 8:
            break

    del model
    gc.collect()
    if device != "cpu":
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
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = _model(px_model=Persistence(VARIABLES, {"lat": LAT, "lon": LON})).to(device)
    ic = model.input_coords()
    x, coords = fetch_data(Random(dc), time, ic["variable"], ic["lead_time"], device)

    # Invalid input domain
    with pytest.raises((KeyError, ValueError)):
        model(x, coords)

    # Missing base model
    model = _model()
    with pytest.raises(ValueError, match="px_model, must be set"):
        model.input_coords()
    x, coords = _inputs(Source())
    with pytest.raises(ValueError, match="px_model, must be set"):
        next(model.create_iterator(x, coords))


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
    model, x, coords = _build(
        step=np.timedelta64(gap_h, "h"), num_interp_steps=num_interp_steps
    )
    step_min = gap_h * 60 // num_interp_steps

    frames = list(
        itertools.islice(model.create_iterator(x, coords), num_interp_steps + 2)
    )

    # Zero residual => linear interpolation
    for k, (xf, cf) in enumerate(frames):
        assert xf.shape == (1, 1, 1, NVAR, H, W)
        assert int(cf["lead_time"][-1] / np.timedelta64(1, "m")) == k * step_min
        assert (xf - (H0 + k * step_min / 60)).abs().max() < 1e-4


@pytest.mark.parametrize("gap_h, expected", [(3, -1.0), (6, 0.0), (9, 1.0)])
def test_interpcrpsdit_gap_channel(gap_h, expected):
    dit = PhooDiT()
    model, x, coords = _build(
        step=np.timedelta64(gap_h, "h"), num_interp_steps=3, dit=dit
    )
    list(itertools.islice(model.create_iterator(x, coords), 2))
    assert float(dit.inputs[0][:, GAP_CH].mean()) == pytest.approx(expected)


@pytest.mark.parametrize(
    "step, num_interp_steps, match",
    [
        (np.timedelta64(6, "h"), 7, "must divide"),
        (np.timedelta64(90, "s"), 6, "whole number of minutes"),
    ],
)
def test_interpcrpsdit_gap_exceptions(step, num_interp_steps, match):
    model, x, coords = _build(step=step, num_interp_steps=num_interp_steps)
    with pytest.raises(ValueError, match=match):
        next(model.create_iterator(x, coords))


@pytest.mark.parametrize("gap_h", [12, 2])
def test_interpcrpsdit_out_of_range_gap_warns(gap_h):
    model, x, coords = _build(step=np.timedelta64(gap_h, "h"))
    with pytest.warns(UserWarning, match="outside the trained"):
        frames = list(itertools.islice(model.create_iterator(x, coords), 2))
    assert all(torch.isfinite(xf).all() for xf, _ in frames)


@pytest.mark.parametrize("amp_dtype, autocast", [(None, False), (torch.bfloat16, True)])
def test_interpcrpsdit_amp_dtype(amp_dtype, autocast):
    dit = PhooDiT()
    model, x, coords = _build(amp_dtype=amp_dtype, dit=dit)
    frames = list(itertools.islice(model.create_iterator(x, coords), 2))
    assert dit.saw_autocast is autocast
    assert torch.isfinite(frames[-1][0]).all()


@pytest.mark.parametrize("lon_pad", [4, 2 * W + 8])
def test_interpcrpsdit_lon_pad(lon_pad):
    model, x, coords = _build(lon_pad=lon_pad)
    frames = list(itertools.islice(model.create_iterator(x, coords), 2))
    assert frames[1][0].shape[-2:] == (H, W)
    assert (frames[1][0] - (H0 + 1)).abs().max() < 1e-4


def test_interpcrpsdit_noise():
    def interior(seed):
        model, x, coords = _build(
            src=Source(static=True),
            times=(T0, T0 + np.timedelta64(12, "h")),
            batch=2,
            seed=seed,
            dit=PhooDiT(noise=True),
        )
        return list(itertools.islice(model.create_iterator(x, coords), 2))[1][0]

    a, b, c = interior(0), interior(0), interior(1)
    assert torch.allclose(a, b)  # same seed reproduces
    assert not torch.allclose(a, c)  # different seed differs
    assert not torch.allclose(a[0], a[1])  # members differ
    assert not torch.allclose(a[:, 0], a[:, 1])  # init times differ


def test_interpcrpsdit_bracket_latent():
    dit = PhooDiT()
    model, x, coords = _build(num_interp_steps=3, seed=0, dit=dit)
    list(itertools.islice(model.create_iterator(x, coords), 7))
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
    model, x, coords = _build(
        src=Source(per_var=True),
        variables=list(base_variables),
        center=torch.full((1, NVAR, 1, 1), 5.0),
        drop_variables=OPTIONAL_VARIABLES,
        dit=dit,
    )
    assert list(model.drop_idx) == drop_idx

    frames = list(itertools.islice(model.create_iterator(x, coords), 2))
    for k, (xf, cf) in enumerate(frames):
        assert list(cf["variable"]) == present
        assert xf.shape[-3] == len(present)
        for j, v in enumerate(present):
            assert (xf[:, :, :, j] - (H0 + k + VARIABLES.index(v))).abs().max() < 1e-4

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
    sub = _model(px_model=None).set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=halo, min_cells=4
    )
    assert sub._halo == exp_halo
    run = sub.run_coords()
    src = (
        Source(run["lat"], run["lon"], spatial=True)
        if base == "regional"
        else Source(spatial=True)
    )
    sub.px_model = _replay(src)
    x, coords = _inputs(src)

    block = (
        np.arange(2, 8, dtype="float32")[:, None]
        + 0.01 * np.arange(5, 18, dtype="float32")[None, :]
    )
    for k, (xf, cf) in enumerate(
        list(itertools.islice(sub.create_iterator(x, coords), 3))
    ):
        np.testing.assert_array_equal(cf["lat"], LAT[2:8])
        np.testing.assert_array_equal(cf["lon"], LON[5:18])
        np.testing.assert_allclose(xf[0, 0, 0, 0].numpy(), H0 + k + block, atol=1e-4)


def test_interpcrpsdit_run_coords():
    model = _model()
    np.testing.assert_array_equal(model.run_coords()["lat"], LAT)
    np.testing.assert_array_equal(model.run_coords()["lon"], LON)

    sub = model.set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=2, min_cells=4
    )
    np.testing.assert_array_equal(sub.run_coords()["lat"], LAT[0:10])
    np.testing.assert_array_equal(sub.run_coords()["lon"], LON[2:20])


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
    src = Source(lat_big, LON, spatial=True)
    model, x, coords = _build(src=src)

    np.testing.assert_array_equal(model.input_coords()["lat"], lat_big)
    np.testing.assert_array_equal(model.run_coords()["lat"], LAT)

    xf, cf = next(model.create_iterator(x, coords))
    np.testing.assert_array_equal(cf["lat"], LAT)
    expected = (
        H0
        + np.arange(H, dtype="float32")[:, None]
        + 0.01 * np.arange(W, dtype="float32")[None, :]
    )
    np.testing.assert_allclose(xf[0, 0, 0, 0].numpy(), expected, atol=1e-4)


def test_interpcrpsdit_regrid_warns_off_grid_base():
    src = Source(LAT + 0.3, LON)
    model, x, coords = _build(src=src)
    with pytest.warns(UserWarning, match="nearest-regridded"):
        next(model.create_iterator(x, coords))


def test_interpcrpsdit_front_hook():
    model, x, coords = _build(num_interp_steps=3)
    model.front_hook = lambda a, c: (a + 5.0, c)
    frames = list(itertools.islice(model.create_iterator(x, coords), 2))
    assert (frames[0][0] - (H0 + 5)).abs().max() < 1e-4
    assert (frames[1][0] - (H0 + 2 + 5)).abs().max() < 1e-4


@pytest.mark.parametrize("inplace", [False, True])
def test_interpcrpsdit_rear_hook(inplace):
    def rear(a, c):
        if inplace:
            a += 100.0
            return a, c
        return a + 100.0, c

    model, x, coords = _build(num_interp_steps=3)
    model.rear_hook = rear
    for k, (xf, _) in enumerate(
        list(itertools.islice(model.create_iterator(x, coords), 7))
    ):
        assert (xf - (H0 + 2 * k + 100)).abs().max() < 1e-4


def test_interpcrpsdit_to_moves_px_model():
    model, _, _ = _build()
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
    model = InterpCRPSDiT.load_model(package)
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
    with pytest.raises(ValueError, match=match):
        InterpCRPSDiT.load_model(package)


@pytest.fixture(scope="function")
def model() -> InterpCRPSDiT:
    package = InterpCRPSDiT.load_default_package()
    return InterpCRPSDiT.load_model(package)


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NATTEN DiT requires CUDA")
def test_interpcrpsdit_package(model):
    device = "cuda:0"
    torch.cuda.empty_cache()
    model = model.to(device)

    sub = model.set_domain(
        lat_min=10.0, lat_max=32.0, lon_min=100.0, lon_max=124.0, halo=0
    )
    run = sub.run_coords()
    sub.px_model = Persistence(VARIABLES, {"lat": run["lat"], "lon": run["lon"]})
    sub.px_model.to(device)

    x = model.center.expand(1, 1, 1, NVAR, len(run["lat"]), len(run["lon"])).clone()
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([np.datetime64("1993-04-05T00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=run["lat"],
        lon=run["lon"],
    )

    (_, _), (out, out_coords) = list(
        itertools.islice(sub.create_iterator(x, coords), 2)
    )
    assert out.shape == (1, 1, 1, NVAR, len(out_coords["lat"]), len(out_coords["lon"]))
    assert out_coords["lead_time"][-1] == np.timedelta64(60, "m")
    assert torch.isfinite(out).all()
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
