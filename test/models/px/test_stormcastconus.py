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
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import earth2studio.models.px.stormcastconus as conus_module
from earth2studio.data import Random, fetch_data
from earth2studio.grids import Region, resolve_grid
from earth2studio.models.base import DomainAdaptable
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.da import StatefulAssimilationModel, StormCastConusSDA
from earth2studio.models.nn.corrdiff import points_in_grid_footprint
from earth2studio.models.px import StormCastCONUS
from earth2studio.models.px.stormcastconus import _cover, _SplitModelWrapper
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


def conditioning_input(model, x):
    signature = coord_array_like(
        x,
        {
            "variable": model.forcing_coords().coords["variable"].values,
            "lead_time": x.lead_time.values + np.timedelta64(1, "h"),
        },
    )
    return from_torch(
        torch.zeros(signature.shape, device=model.means.device), signature
    )


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch, request):
    if conus_module.__file__ not in OptionalDependencyFailure.failures:
        return
    if request.node.get_closest_marker("package"):
        pytest.skip("PhysicsNeMo is unavailable")
    monkeypatch.delitem(OptionalDependencyFailure.failures, conus_module.__file__)

    class Scheduler:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(conus_module, "EDMNoiseScheduler", Scheduler, raising=False)
    monkeypatch.setattr(
        conus_module, "sample", lambda denoiser, latents, **kw: latents, raising=False
    )
    # Exercise public batching/conditioning/ownership when the sampler dependency
    # is absent. Installed backends continue through the actual diffusion path.
    monkeypatch.setattr(
        StormCastCONUS,
        "_forward",
        lambda self, x, conditioning, time, **kw: x + self._sample(x, None),
    )


# Small subdomain aligned to the mock patch size (8, 8) so that crop_model
# validation passes.  Must satisfy:
#   (LAT_START - 17) % 8 == 0  and  (LAT_END - LAT_START) % 8 == 0
#   (LON_START -  3) % 8 == 0  and  (LON_END - LON_START) % 8 == 0
LAT_START, LAT_END = 17, 33  # height = 16
LON_START, LON_END = 3, 19  # width  = 16

NVAR = 4  # must include "refc" – it is always indexed in __init__
NVAR_COND = 5


class _Tokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(
            torch.arange(16, dtype=torch.float32).reshape(16, 1)
        )
        self.input_size = (32, 32)
        self.h_patches = 4
        self.w_patches = 4


class _Detokenizer:
    def __init__(self):
        self.input_size = (32, 32)
        self.h_patches = 4
        self.w_patches = 4


class _PatchConfig:
    def __init__(self):
        self.patch_size = (8, 8)
        self.input_size = (32, 32)
        self.tokenizer = _Tokenizer()
        self.detokenizer = _Detokenizer()


class _InnerModel:
    def __init__(self):
        self.model = _PatchConfig()


class _DiffusionSubmodel:
    def __init__(self):
        self.model = _InnerModel()


class PhooStormCastCONUSDiffusionModel(_SplitModelWrapper):
    """Minimal diffusion model stub for StormCastCONUS unit tests.

    Subclasses :class:`_SplitModelWrapper` so that it passes the ``isinstance``
    check in ``StormCastCONUS.__init__``. Skips the real ``__init__`` but
    provides the model structure used by ``_SplitModelWrapper.crop_model``.
    The forward pass returns the (unchanged) noisy input so the diffusion
    sampler converges trivially.
    """

    def __init__(self, nvar: int):
        # Skip _SplitModelWrapper.__init__; only call torch.nn.Module.__init__
        torch.nn.Module.__init__(self)
        self._nvar = nvar
        self.model_high = _DiffusionSubmodel()
        self.model_pz_low = _DiffusionSubmodel()
        self.model_tq_low = _DiffusionSubmodel()
        self.model_uv_low = _DiffusionSubmodel()
        self.models = {
            "high": self.model_high,
            "pz_low": self.model_pz_low,
            "tq_low": self.model_tq_low,
            "uv_low": self.model_uv_low,
        }
        self.full_grid_shape = (32, 32)
        self.grid_shape = self.full_grid_shape
        self.pos_embed_full = {
            key: model.model.model.tokenizer.pos_embed
            for key, model in self.models.items()
        }
        # Geometry each forward saw on the shared networks (shared with views)
        self.seen = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition=None, **kwargs
    ) -> torch.Tensor:
        with self._bound():
            dit = self.model_high.model.model
            assert dit.detokenizer.h_patches * 8 == x.shape[-2]
            assert dit.detokenizer.w_patches * 8 == x.shape[-1]
            assert dit.tokenizer.pos_embed.shape[0] == x[0, 0].numel() // 64
            self.seen.append(tuple(x.shape[-2:]))
        return x[:, : self._nvar, :, :]


def _build_model(
    device: str = "cpu",
    use_amp: bool = False,
    clamp_values: bool = False,
    hrrr_lat_lim: tuple[int, int] = (LAT_START, LAT_END),
    hrrr_lon_lim: tuple[int, int] = (LON_START, LON_END),
) -> StormCastCONUS:
    """Construct a minimal StormCastCONUS for unit testing."""
    diffusion = PhooStormCastCONUSDiffusionModel(NVAR)

    variables = np.array(["u%02d" % i for i in range(NVAR - 1)] + ["refc"])
    means = torch.zeros(1, NVAR, 1, 1)
    stds = torch.ones(1, NVAR, 1, 1)
    # invariants cover the mock's full 32x32 footprint at HRRR [17:49, 3:35]
    invariants = torch.randn(1, 2, 17 + 32, 3 + 32)
    conditioning_means = torch.zeros(1, NVAR_COND, 1, 1)
    conditioning_stds = torch.ones(1, NVAR_COND, 1, 1)
    conditioning_variables = np.array(["c%02d" % i for i in range(NVAR_COND)])

    return StormCastCONUS(
        diffusion,
        means,
        stds,
        invariants,
        conditioning_means,
        conditioning_stds,
        hrrr_lat_lim=hrrr_lat_lim,
        hrrr_lon_lim=hrrr_lon_lim,
        variables=variables,
        conditioning_variables=conditioning_variables,
        num_diffusion_steps=2,
        use_amp=use_amp,
        clamp_values=clamp_values,
    ).to(device)


def test_stormcastconus_crop_uses_model_region_coordinates():
    model = _build_model()
    diffusion_model = model.diffusion_model

    assert isinstance(diffusion_model, PhooStormCastCONUSDiffusionModel)
    signature = model.input_coords()
    assert signature.data.nbytes == 0
    assert "earth2studio_grid_id" not in signature.attrs
    assert signature.attrs["earth2studio_crs"] == model.grid.crs
    np.testing.assert_array_equal(signature.lat, model.lat)
    np.testing.assert_array_equal(signature.lon, model.lon)
    expected_pos_embed = torch.tensor([[0.0], [1.0], [4.0], [5.0]])
    assert diffusion_model.grid_shape == (16, 16)
    # The shared networks keep their full geometry; forwards bind the crop
    for submodel in diffusion_model.models.values():
        dit = submodel.model.model
        assert dit.input_size == (32, 32)
        assert dit.tokenizer.pos_embed.shape == (16, 1)
    with diffusion_model._bound():
        for submodel in diffusion_model.models.values():
            dit = submodel.model.model
            assert dit.input_size == (16, 16)
            assert dit.tokenizer.input_size == (16, 16)
            assert dit.tokenizer.h_patches == 2
            assert dit.tokenizer.w_patches == 2
            assert torch.equal(dit.tokenizer.pos_embed, expected_pos_embed)
            assert dit.detokenizer.input_size == (16, 16)
            assert dit.detokenizer.h_patches == 2
            assert dit.detokenizer.w_patches == 2
    for key, submodel in diffusion_model.models.items():
        dit = submodel.model.model
        assert dit.input_size == (32, 32)
        assert (dit.tokenizer.h_patches, dit.detokenizer.w_patches) == (4, 4)
        assert dit.tokenizer.pos_embed is diffusion_model.pos_embed_full[key]
    check_prognostic_contract(model)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_sda_iter(device):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if conus_module.TensorDict is None:
        pytest.skip("PhysicsNeMo guidance is unavailable")
    core = _build_model(device)
    core.num_sda_diffusion_steps = 2
    core.time_tolerance = np.timedelta64(10, "m")
    model = StormCastConusSDA(core)
    assert isinstance(model, StatefulAssimilationModel)
    signature = core.input_coords()
    source = Random(OrderedDict((d, signature[d].values) for d in ("y", "x")))
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = fetch_data(
        source,
        time,
        core.variables,
        np.array([np.timedelta64(0, "h")]),
    )
    x = x.assign_coords(core.grid.coords())
    x.attrs = dict(core.grid.attrs, earth2studio_crs=core.grid.crs)
    c = conditioning_input(core, x)
    obs = pd.DataFrame(
        {
            # Observations describe the output valid time, one hour after x.
            "time": [time[0] + np.timedelta64(1, "h")],
            "variable": [core.variables[0]],
            "lat": [core.lat[8, 8]],
            "lon": [core.lon[8, 8]],
            "observation": [1.0],
        }
    )
    original_x, original_c, original_obs = (
        x.copy(deep=True),
        c.copy(deep=True),
        obs.copy(deep=True),
    )
    global_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device != "cpu" else None
    model.set_rng(42)
    free = model(x, c)
    model.set_rng(42)
    y, state = model.initialize(x, c, obs)
    assert np.isfinite(y.e2s.as_numpy()).all()
    assert not np.allclose(y.e2s.as_numpy(), free.e2s.as_numpy())
    assert y.lead_time.values[0] == np.timedelta64(1, "h")
    model.set_rng(42)
    # An observation at the input time lies outside the 10 minute window.
    outside_window = obs.assign(time=time[0])
    xr.testing.assert_allclose(model(x, c, outside_window), free)
    saved = deepcopy(state)
    c1 = conditioning_input(core, y)
    z, _ = model.step(y, c1, None, state)
    model.set_rng(99)
    xr.testing.assert_identical(model.step(y, c1, None, state)[0], z)
    assert saved[0] == state[0]
    for key in saved[1]:
        assert torch.equal(saved[1][key], state[1][key])
    model.set_rng(42)
    iterator = model.create_iterator(x, c, obs)
    xr.testing.assert_identical(next(iterator), y)
    xr.testing.assert_identical(iterator.send((c1, None)), z)
    iterator.close()
    assert torch.equal(global_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    xr.testing.assert_identical(x, original_x)
    xr.testing.assert_identical(c, original_c)
    pd.testing.assert_frame_equal(obs, original_obs)
    with pytest.raises(ValueError, match="conditioning"):
        model.initialize(x, None, obs)
    with pytest.raises(ValueError, match="missing required fields"):
        model.initialize(x, c, obs.drop(columns="observation"))
    with pytest.raises(ValueError):
        model.initialize(x, c.assign_coords(lead_time=x.lead_time), obs)


@pytest.mark.parametrize(
    "hrrr_lat_lim, hrrr_lon_lim, match",
    [
        ((18, 34), (3, 19), r"hrrr_lat_lim\[0\].*must be divisible"),
        ((17, 34), (3, 19), r"hrrr_lat_lim\[1\].*must be divisible"),
        ((17, 33), (4, 20), r"hrrr_lon_lim\[0\].*must be divisible"),
        ((17, 33), (3, 20), r"hrrr_lon_lim\[1\].*must be divisible"),
    ],
)
def test_stormcastconus_crop_requires_patch_aligned_limits(
    hrrr_lat_lim: tuple[int, int],
    hrrr_lon_lim: tuple[int, int],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _build_model(hrrr_lat_lim=hrrr_lat_lim, hrrr_lon_lim=hrrr_lon_lim)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2020-04-05T00:00")]),
        np.array(
            [
                np.datetime64("2020-10-11T12:00"),
                np.datetime64("2020-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("clamp_values", [False, True])
@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_call(time, device, use_amp, clamp_values):
    p = _build_model(device, use_amp=use_amp, clamp_values=clamp_values)

    dc = OrderedDict(
        [
            ("y", p.input_coords()["y"].values),
            ("x", p.input_coords()["x"].values),
        ]
    )
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"].values
    variable = p.input_coords()["variable"].values
    x = fetch_data(r, time, variable, lead_time)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    x.attrs["nested"] = {"items": [1]}
    x.encoding["nested"] = {"items": [2]}
    original = x.copy(deep=True)
    out = p(x, conditioning_input(p, x))
    out.attrs["nested"].clear()
    out.encoding["nested"].clear()
    assert x.attrs["nested"] == {"items": [1]}
    assert x.encoding["nested"] == {"items": [2]}
    out_coords = p.output_coords(x)
    xr.testing.assert_identical(x, original)

    if not isinstance(time, Iterable):
        time = [time]

    ny, nx = LAT_END - LAT_START, LON_END - LON_START
    assert out.shape == torch.Size([len(time), 1, NVAR, ny, nx])
    assert (out_coords["variable"] == p.output_coords(x)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    assert out_coords.dims == ("time", "lead_time", "variable", "y", "x")
    assert out_coords.data.nbytes == 0


@pytest.mark.parametrize(
    "state_dtype, conditioning_dtype, device, use_amp",
    [
        (np.float32, np.float32, "cpu", False),
        (np.float64, np.float32, "cpu", False),
        (np.float32, np.float64, "cuda:0", True),
        (np.float64, np.float64, "cuda:0", True),
    ],
)
def test_stormcastconus_input_dtype(
    state_dtype, conditioning_dtype, device, use_amp, monkeypatch
):
    if conus_module.TensorDict is None:
        pytest.skip("PhysicsNeMo is unavailable")
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    p = _build_model(device, use_amp=use_amp)
    # Exercise a real convolution under the sampler's autocast context. The
    # usual identity denoiser does not detect incompatible input dtypes.
    conv = torch.nn.Conv2d(2 * NVAR + NVAR_COND + 3, NVAR, 1).to(device)

    def denoise(x, t, condition=None, **kwargs):
        return conv(torch.cat((x, condition["cond_concat"]), dim=1)).float()

    monkeypatch.setattr(p.diffusion_model, "forward", denoise)
    time = np.array([np.datetime64("2020-04-05T00:00")])
    signature = p.input_coords()
    r = Random(OrderedDict((d, signature[d].values) for d in ("y", "x")))
    x = fetch_data(
        r,
        time,
        signature["variable"].values,
        signature["lead_time"].values,
        device=device,
    ).astype(state_dtype)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    original = x.copy(deep=True)

    out = p(x, conditioning_input(p, x))
    assert out.dtype == np.float32
    assert out.e2s.to_torch()[0].device == torch.device(device)
    iterator = p.create_iterator(x, conditioning_input(p, x))
    forecast = x
    for hour in (1, 2):
        forecast = (
            next(iterator)
            if hour == 1
            else iterator.send(conditioning_input(p, forecast))
        )
        assert forecast.dtype == np.float32
        np.testing.assert_array_equal(forecast.lead_time, [np.timedelta64(hour, "h")])
    xr.testing.assert_identical(x, original)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("clamp_values", [False, True])
@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_iter(ensemble, device, use_amp, clamp_values):
    time = np.array([np.datetime64("2020-04-05T00:00")])
    p = _build_model(device, use_amp=use_amp, clamp_values=clamp_values)

    dc = OrderedDict(
        [
            ("y", p.input_coords()["y"].values),
            ("x", p.input_coords()["x"].values),
        ]
    )
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"].values
    variable = p.input_coords()["variable"].values
    x = fetch_data(r, time, variable, lead_time)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    x = x.expand_dims(ensemble=np.arange(ensemble)).copy(deep=True)
    x.name = "conus"
    x.attrs["removed"] = True
    x.encoding = {"removed": True}
    x = x.assign_coords(member=("ensemble", np.arange(ensemble)))
    original = x.copy(deep=True)
    events = []

    def front(field):
        assert field.dims == x.dims
        events.append("front")
        return field

    def rear(field):
        events.append("rear")
        field.attrs.pop("removed", None)
        field.encoding.clear()
        return field.drop_vars("member", errors="ignore")

    p.front_hook, p.rear_hook = front, rear
    coords = x
    p_iter = p.create_iterator(x, conditioning_input(p, x))

    ny, nx = LAT_END - LAT_START, LON_END - LON_START

    initial = next(p_iter)
    retained = initial.copy(deep=True)
    assert events == ["rear"]
    initial_coords = coord_array(
        initial.dims, dict(initial.coords), attrs=initial.attrs
    )
    assert initial.shape == x.shape
    assert initial_coords.dims == coords.dims
    assert initial_coords.data.nbytes == 0
    out = initial
    for i in range(4):
        out = p_iter.send(conditioning_input(p, out))
        xr.testing.assert_identical(x, original)
        xr.testing.assert_identical(initial, retained)
        assert out.name == x.name and "removed" not in out.attrs
        assert out.encoding == {} and "member" not in out.coords
        assert events == ["rear"] + ["front", "rear"] * (i + 1)
        out_coords = coord_array(out.dims, dict(out.coords), attrs=out.attrs)
        assert out_coords.dims == coords.dims
        assert out_coords.data.nbytes == 0
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, NVAR, ny, nx])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(i + 2, "h")

        if i > 2:
            break


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_exceptions(device):
    diffusion = PhooStormCastCONUSDiffusionModel(NVAR)

    variables = np.array(["u%02d" % i for i in range(NVAR - 1)] + ["refc"])
    means = torch.zeros(1, NVAR, 1, 1)
    stds = torch.ones(1, NVAR, 1, 1)
    invariants = torch.randn(1, 2, LAT_END, LON_END)
    conditioning_means = torch.zeros(1, NVAR_COND, 1, 1)
    conditioning_stds = torch.ones(1, NVAR_COND, 1, 1)
    conditioning_variables = np.array(["c%02d" % i for i in range(NVAR_COND)])

    p = StormCastCONUS(
        diffusion,
        means,
        stds,
        invariants,
        conditioning_means,
        conditioning_stds,
        hrrr_lat_lim=(LAT_START, LAT_END),
        hrrr_lon_lim=(LON_START, LON_END),
        variables=variables,
        conditioning_variables=conditioning_variables,
        num_diffusion_steps=2,
        use_amp=False,
        clamp_values=False,
    ).to(device)

    dc = OrderedDict(
        [
            ("y", p.input_coords()["y"].values),
            ("x", p.input_coords()["x"].values),
        ]
    )
    r = Random(dc)
    lead_time = p.input_coords()["lead_time"].values
    variable = p.input_coords()["variable"].values
    x = fetch_data(
        r,
        np.array([np.datetime64("2020-04-05T00:00")]),
        variable,
        lead_time,
        device=device,
    )
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)

    with pytest.raises(TypeError):
        p(x)

    with pytest.raises(TypeError):
        p.create_iterator(x)


@pytest.fixture(scope="function")
def model() -> StormCastCONUS:
    package = StormCastCONUS.load_default_package()
    return StormCastCONUS.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize(
    "cond_dims",
    [["time", "variable", "lat", "lon"], ["variable", "time", "lat", "lon"]],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_stormcastconus_package(cond_dims, device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2020-04-05T00:00")])
    p = model.to(device)

    r = Random(
        OrderedDict(
            [
                ("y", p.input_coords()["y"].values),
                ("x", p.input_coords()["x"].values),
            ]
        )
    )

    p.num_diffusion_steps = 2

    lead_time = p.input_coords()["lead_time"].values
    variable = p.input_coords()["variable"].values
    x = fetch_data(r, time, variable, lead_time, device=device)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    out = p(x, conditioning_input(p, x))
    out_coords = p.output_coords(x)

    assert out.shape == torch.Size(
        [len(time), 1, len(p.output_coords(x)["variable"]), 1024, 1792]
    )
    assert (out_coords["variable"] == p.output_coords(x)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    assert out_coords.dims == ("time", "lead_time", "variable", "y", "x")
    assert out_coords.data.nbytes == 0


# Full footprint of the mock networks: HRRR indices [17:49, 3:35]
FOOTPRINT = (slice(17, 49), slice(3, 35))


def _footprint_lat_lon() -> tuple[np.ndarray, np.ndarray]:
    coords = resolve_grid("hrrr").coords()
    return (
        np.asarray(coords["lat"])[FOOTPRINT],
        np.asarray(coords["lon"])[FOOTPRINT],
    )


def _box(rows: slice, cols: slice) -> Region:
    """Box spanning the centers of a block of footprint cells."""
    lat, lon = _footprint_lat_lon()
    lat, lon = lat[rows, cols], lon[rows, cols]
    return Region.box(lon.min(), lat.min(), lon.max(), lat.max())


def _model_input(model: StormCastCONUS) -> xr.DataArray:
    signature = model.input_coords()
    source = Random(OrderedDict((d, signature[d].values) for d in ("y", "x")))
    x = fetch_data(
        source,
        np.array([np.datetime64("2020-04-05T00:00")]),
        signature["variable"].values,
        signature["lead_time"].values,
    )
    x = x.assign_coords(model.grid.coords())
    x.attrs = dict(model.grid.attrs, earth2studio_crs=model.grid.crs)
    return x


@pytest.mark.parametrize(
    "lo, hi, n, patch, minimum, expected",
    [
        (3, 9, 32, 8, 8, (0, 16)),
        (-1, 5, 32, 8, 8, (0, 8)),
        (27, 33, 32, 8, 8, (24, 32)),
        (12, 14, 32, 8, 24, (0, 24)),
        (28, 31, 32, 8, 24, (8, 32)),
        (5, 7, 32, 4, 32, (0, 32)),
    ],
)
def test_stormcastconus_cover(lo, hi, n, patch, minimum, expected):
    assert _cover(lo, hi, n, patch, minimum) == expected


def test_stormcastconus_with_region():
    full = _build_model()
    assert isinstance(full, DomainAdaptable)
    signature = full.input_coords()
    wrapper = full.diffusion_model
    pos_embed = {k: v.clone() for k, v in wrapper.pos_embed_full.items()}

    region = _box(slice(12, 15), slice(20, 23))
    view = full.with_region(region)
    other = full.with_region(_box(slice(2, 6), slice(2, 6)))

    # Views share the networks but own their crop, grid and RNG stream
    assert view.diffusion_model is not wrapper
    for key in wrapper.models:
        assert view.diffusion_model.models[key] is wrapper.models[key]
    assert view.diffusion_model.model_high is wrapper.model_high
    assert view.hrrr_lat_lim != full.hrrr_lat_lim
    assert view.grid.shape == view.lat.shape == view.lat_tensor.shape
    for (lo, hi), origin in zip((view.hrrr_lat_lim, view.hrrr_lon_lim), (17, 3)):
        assert (lo - origin) % 8 == 0 and (hi - lo) % 8 == 0
        assert hi - lo >= 8
    west, south, east, north = region.params
    corners = points_in_grid_footprint(
        np.array([south, south, north, north]),
        np.array([west, east, west, east]) % 360,
        view.lat,
        view.lon,
    )
    assert corners.all()

    # The parent and sibling views are unchanged
    after = full.input_coords()
    assert after.sizes == signature.sizes and after.attrs == signature.attrs
    for name in ("y", "x", "lat", "lon"):
        np.testing.assert_array_equal(after[name], signature[name])
    assert (full.hrrr_lat_lim, full.hrrr_lon_lim) == (
        (LAT_START, LAT_END),
        (LON_START, LON_END),
    )
    assert full.diffusion_model is wrapper and wrapper.grid_shape == (16, 16)
    assert other.hrrr_lat_lim != view.hrrr_lat_lim
    for key, submodel in wrapper.models.items():
        dit = submodel.model.model
        assert dit.input_size == (32, 32)
        assert dit.tokenizer.pos_embed is wrapper.pos_embed_full[key]
        assert torch.equal(dit.tokenizer.pos_embed, pos_embed[key])

    # A view selects from the full footprint, not only from its own grid
    far = _box(slice(26, 29), slice(26, 29))
    reached = other.with_region(far)
    expected = full.with_region(far)
    assert (reached.hrrr_lat_lim, reached.hrrr_lon_lim) == (
        expected.hrrr_lat_lim,
        expected.hrrr_lon_lim,
    )
    large = _box(slice(8, 24), slice(8, 24))
    wide = view.with_region(large)
    assert (wide.hrrr_lat_lim, wide.hrrr_lon_lim) == (
        full.with_region(large).hrrr_lat_lim,
        full.with_region(large).hrrr_lon_lim,
    )
    assert wide.lat.shape[0] > view.lat.shape[0]

    lat, lon = _footprint_lat_lon()
    for bad in (
        Region.band(30, 40),
        Region.rectangle(float(lat[16, 16]), float(lon[16, 16]), 1e4),
        Region.circle(float(lat[16, 16]), float(lon[16, 16]), 1e4),
        Region.box(float(lon.min()) - 5, float(lat.min()), float(lon.max()), 40),
        Region.box(170, float(lat.min()), float(lon.max()), float(lat.max())),
    ):
        with pytest.raises(ValueError):
            full.with_region(bad)


def test_stormcastconus_with_region_forecast():
    if conus_module.TensorDict is None:
        pytest.skip("PhysicsNeMo is unavailable")
    full = _build_model()
    first = full.with_region(_box(slice(12, 15), slice(20, 23)))
    second = full.with_region(_box(slice(8, 24), slice(8, 24)))
    wrapper = full.diffusion_model
    wrapper.seen.clear()

    results = []
    for model in (full, first, second, first, full):
        x = _model_input(model)
        model.set_rng(7)
        out = model(x, conditioning_input(model, x))
        assert out.shape[-2:] == model.lat.shape
        np.testing.assert_array_equal(out.y, model.input_coords().y)
        results.append(out)
        # Shared networks are back on their full geometry after every call
        for key, submodel in wrapper.models.items():
            dit = submodel.model.model
            assert dit.input_size == (32, 32)
            assert dit.tokenizer.pos_embed is wrapper.pos_embed_full[key]
    shapes = list(dict.fromkeys(wrapper.seen))
    assert shapes == [(16, 16), first.lat.shape, second.lat.shape]
    # Running other views in between leaves each view's result unchanged
    xr.testing.assert_identical(results[1], results[3])
    xr.testing.assert_identical(results[0], results[4])


def test_stormcastconus_with_region_rng():
    full = _build_model()
    full.set_rng(3)
    view = full.with_region(_box(slice(12, 15), slice(20, 23)))
    assert view._rng_seed == 3
    view._rng_states["cpu"] = torch.zeros(1)
    assert "cpu" not in full._rng_states


def test_stormcastconus_views_follow_shared_networks():
    full = _build_model()
    # Views copy the wrapper but share its submodules (here a stand-in network)
    shared = full.diffusion_model.shared = torch.nn.Linear(1, 1)
    view = full.with_region(_box(slice(12, 15), slice(20, 23)))
    # Moving a view moves the shared networks but not the parent's buffers
    view.to("meta")
    assert shared.weight.device.type == "meta"
    assert full.means.device.type == "cpu"
    full._follow_network()
    assert full.means.device.type == full.lat_tensor.device.type == "meta"


def test_stormcastconus_cropped_pos_embed_tracks_weight_updates():
    wrapper = _build_model().with_region(_box(slice(12, 15), slice(20, 23)))
    wrapper = wrapper.diffusion_model
    key = next(iter(wrapper.pos_embed_full))
    before = wrapper._cropped_pos_embed(key).clone()
    # load_state_dict copies in place, keeping the parameter's address
    with torch.no_grad():
        wrapper.pos_embed_full[key].add_(1.0)
    torch.testing.assert_close(wrapper._cropped_pos_embed(key), before + 1.0)


def test_stormcastconus_bound_pos_embed_grad():
    wrapper = _build_model().diffusion_model
    with torch.inference_mode():
        wrapper(torch.zeros(1, NVAR, 16, 16), torch.ones(1))
    # The cached crop built under inference mode serves a grad-enabled forward
    x = torch.zeros(1, NVAR, 16, 16, requires_grad=True)
    with wrapper._bound():
        pos_embed = wrapper.model_high.model.model.tokenizer.pos_embed
        assert not pos_embed.is_inference() and not pos_embed.requires_grad
        (x * 2 + pos_embed.sum()).sum().backward()
    assert x.grad is not None
