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

from earth2studio.data import HRRR, Random, fetch_data
from earth2studio.models.px import StormCastCONUS
from earth2studio.models.px.stormcastconus import _SplitModelWrapper
from earth2studio.utils import handshake_dim

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
            ("hrrr_y", HRRR.HRRR_Y[LAT_START:LAT_END]),
            ("hrrr_x", HRRR.HRRR_X[LON_START:LON_END]),
        ]
    )
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    ny, nx = LAT_END - LAT_START, LON_END - LON_START
    assert out.shape == torch.Size([len(time), 1, NVAR, ny, nx])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "hrrr_x", 4)
    handshake_dim(out_coords, "hrrr_y", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("clamp_values", [False, True])
@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcastconus_iter(ensemble, device, use_amp, clamp_values):
    time = np.array([np.datetime64("2020-04-05T00:00")])
    p = _build_model(device, use_amp=use_amp, clamp_values=clamp_values)

    dc = OrderedDict(
        [
            ("hrrr_y", HRRR.HRRR_Y[LAT_START:LAT_END]),
            ("hrrr_x", HRRR.HRRR_X[LON_START:LON_END]),
        ]
    )
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Prepend ensemble dimension
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    ny, nx = LAT_END - LAT_START, LON_END - LON_START

    next(p_iter)  # consume the initial condition
    for i, (out, out_coords) in enumerate(p_iter):
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
            ("hrrr_y", HRRR.HRRR_Y[LAT_START:LAT_END]),
            ("hrrr_x", HRRR.HRRR_X[LON_START:LON_END]),
        ]
    )
    r = Random(dc)
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(
        r,
        np.array([np.datetime64("2020-04-05T00:00")]),
        variable,
        lead_time,
        device=device,
    )

    with pytest.raises(RuntimeError):
        p(x, coords)

    with pytest.raises(RuntimeError):
        next(p.create_iterator(x, coords))


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

    lat_start, lat_end = p.hrrr_lat_lim
    lon_start, lon_end = p.hrrr_lon_lim
    r = Random(
        OrderedDict(
            [
                ("hrrr_y", HRRR.HRRR_Y[lat_start:lat_end]),
                ("hrrr_x", HRRR.HRRR_X[lon_start:lon_end]),
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

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    assert out.shape == torch.Size(
        [len(time), 1, len(p.output_coords(coords)["variable"]), 1024, 1792]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "hrrr_x", 4)
    handshake_dim(out_coords, "hrrr_y", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
