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

import earth2studio.models.px.stormcastconus as conus_module
from earth2studio.data import Random, Random_FX, fetch_data
from earth2studio.models.conformance import ContractException, check_prognostic_contract
from earth2studio.models.px import StormCastCONUS
from earth2studio.models.px.stormcastconus import _SplitModelWrapper
from earth2studio.utils.coords import coord_array
from earth2studio.utils.imports import OptionalDependencyFailure


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
    # Exercise public batching/conditioning/ownership when the sampler dependency
    # is absent. Installed backends continue through the actual diffusion path.
    monkeypatch.setattr(
        StormCastCONUS,
        "_forward",
        lambda self, x, conditioning, time, **kw: x + torch.randn_like(x),
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

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition=None, **kwargs
    ) -> torch.Tensor:
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
    # invariants must be large enough to be indexed at [LAT_START:LAT_END, LON_START:LON_END]
    invariants = torch.randn(1, 2, LAT_END, LON_END)
    conditioning_means = torch.zeros(1, NVAR_COND, 1, 1)
    conditioning_stds = torch.ones(1, NVAR_COND, 1, 1)
    conditioning_variables = np.array(["c%02d" % i for i in range(NVAR_COND)])

    r_condition = Random(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=181, endpoint=True)),
                ("lon", np.linspace(0, 360, num=360)),
            ]
        )
    )

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
        conditioning_data_source=r_condition,
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
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(model)
    assert exc_info.value.violations == [
        "P13: repeated runs with the same input and seed disagree"
    ]


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
    x = fetch_data(r, time, variable, lead_time, target_grid=p.grid)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    x.attrs["nested"] = {"items": [1]}
    x.encoding["nested"] = {"items": [2]}
    original = x.copy(deep=True)
    out = p(x)
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
    x = fetch_data(r, time, variable, lead_time, target_grid=p.grid)
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
    p_iter = p.create_iterator(x)

    ny, nx = LAT_END - LAT_START, LON_END - LON_START

    initial = next(p_iter)
    retained = initial.copy(deep=True)
    assert events == []
    initial_coords = coord_array(
        initial.dims, dict(initial.coords), attrs=initial.attrs
    )
    assert initial.shape == x.shape
    assert initial_coords.dims == coords.dims
    assert initial_coords.data.nbytes == 0
    for i, out in enumerate(p_iter):
        xr.testing.assert_identical(x, original)
        xr.testing.assert_identical(initial, retained)
        assert out.name == x.name and "removed" not in out.attrs
        assert out.encoding == {} and "member" not in out.coords
        assert events == ["front", "rear"] * (i + 1)
        out_coords = coord_array(out.dims, dict(out.coords), attrs=out.attrs)
        assert out_coords.dims == coords.dims
        assert out_coords.data.nbytes == 0
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, NVAR, ny, nx])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")

        if i > 2:
            break


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_exceptions(device):
    """StormCastCONUS must raise RuntimeError when no conditioning source is set."""
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
        conditioning_data_source=None,
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
        target_grid=p.grid,
    )
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)

    with pytest.raises(RuntimeError):
        p(x)

    iterator = p.create_iterator(x)
    xr.testing.assert_identical(next(iterator), x)
    with pytest.raises(RuntimeError):
        next(iterator)


def test_stormcastconus_conditioning_init_time():
    p = _build_model()

    class _SixHourlyFX(Random_FX):
        def __call__(self, time, lead_time, variable):
            ts = np.asarray(time, dtype="datetime64[h]")
            hours = (ts - ts.astype("datetime64[D]")) / np.timedelta64(1, "h")
            # simulate data source that fails if hour is not one of 00, 06, 12, 18
            if np.any(hours.astype(int) % 6 != 0):
                raise ValueError("forecast only available every 6 hours")
            return super().__call__(time, lead_time, variable)

    p.conditioning_data_source = _SixHourlyFX(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=181, endpoint=True)),
                ("lon", np.linspace(0, 360, num=360)),
            ]
        )
    )
    coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2020-04-05T03:00")])),
            ("lead_time", np.array([np.timedelta64(1, "h")])),
        ]
    )
    device = torch.device("cpu")

    # 03Z is not a 6-hourly cycle — fails without conditioning_init_time
    with pytest.raises(ValueError, match="every 6 hours"):
        p._get_conditioning(coords, batch_size=1, device=device)

    # Pin conditioning to the 00Z cycle so the request is valid
    p.conditioning_init_time = np.array([np.datetime64("2020-04-05T00:00")])
    p._get_conditioning(coords, batch_size=1, device=device)

    # Non-uniform offsets / bad shape
    coords["time"] = np.array(
        [np.datetime64("2020-04-05T03:00"), np.datetime64("2020-04-05T04:00")]
    )
    p.conditioning_init_time = np.array([np.datetime64("2020-04-05T00:00")] * 2)
    with pytest.raises(ValueError, match="uniform lead-time"):
        p._get_conditioning(coords, batch_size=1, device=device)
    p.conditioning_init_time = np.array([np.datetime64("2020-04-05T00:00")] * 3)
    with pytest.raises(ValueError, match="scalar or match"):
        p._get_conditioning(coords, batch_size=1, device=device)


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

    class _RandomWithSpecifiedOrder(Random):
        def __call__(self, time, variable):
            x = super().__call__(time, variable)
            return x.transpose(*cond_dims)

    p.conditioning_data_source = _RandomWithSpecifiedOrder(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=721, endpoint=True)),
                ("lon", np.linspace(0, 360, num=1440)),
            ]
        )
    )
    p.num_diffusion_steps = 2

    lead_time = p.input_coords()["lead_time"].values
    variable = p.input_coords()["variable"].values
    x = fetch_data(r, time, variable, lead_time, device=device, target_grid=p.grid)
    x = x.assign_coords(p.grid.coords())
    x.attrs = dict(p.grid.attrs, earth2studio_crs=p.grid.crs)
    out = p(x)
    out_coords = p.output_coords(x)

    assert out.shape == torch.Size(
        [len(time), 1, len(p.output_coords(x)["variable"]), 1024, 1792]
    )
    assert (out_coords["variable"] == p.output_coords(x)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    assert out_coords.dims == ("time", "lead_time", "variable", "y", "x")
    assert out_coords.data.nbytes == 0
