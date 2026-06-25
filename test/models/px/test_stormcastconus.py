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
from earth2studio.utils import handshake_dim

# Small subdomain aligned to the mock patch size (8, 8) so that crop_model
# validation passes.  Must satisfy:
#   (LAT_START - 17) % 8 == 0  and  (LAT_END - LAT_START) % 8 == 0
#   (LON_START -  3) % 8 == 0  and  (LON_END - LON_START) % 8 == 0
LAT_START, LAT_END = 17, 33  # height = 16
LON_START, LON_END = 3, 19  # width  = 16

NVAR = 4  # must include "refc" – it is always indexed in __init__
NVAR_COND = 5


class _PatchConfig:
    patch_size = (8, 8)


class PhooStormCastCONUSDiffusionModel(torch.nn.Module):
    """Minimal diffusion model stub for StormCastCONUS unit tests.

    Exposes ``model_high.model.model.patch_size`` as required by the crop_model
    path in ``StormCastCONUSBase.__init__``.  ``crop_model`` is a no-op.
    The forward pass returns the (unchanged) noisy input so the diffusion
    sampler converges trivially.
    """

    def __init__(self, nvar: int):
        super().__init__()
        self._nvar = nvar
        dit = _PatchConfig()
        inner = type("_Inner", (), {"model": dit})()
        self.model_high = type("_High", (), {"model": inner})()

    def crop_model(self, bbox: tuple) -> None:
        pass

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition=None, **kwargs
    ) -> torch.Tensor:
        return x[:, : self._nvar, :, :]


def _build_model(
    device: str = "cpu",
    use_amp: bool = False,
    clamp_values: bool = False,
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
        hrrr_lat_lim=(LAT_START, LAT_END),
        hrrr_lon_lim=(LON_START, LON_END),
        variables=variables,
        conditioning_variables=conditioning_variables,
        conditioning_data_source=r_condition,
        num_diffusion_steps=2,
        use_amp=use_amp,
        clamp_values=clamp_values,
    ).to(device)


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
