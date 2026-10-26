# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime

import numpy as np
import pytest
import torch

import earth2studio.models.dx.corrdiff_era5_hrrr as corrdiff_module
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import CorrDiffEra5Hrrr
from earth2studio.models.dx.corrdiff_era5_hrrr import ERA5_VARIABLES, OUTPUT_VARIABLES
from earth2studio.utils import handshake_dim
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch


class PhooNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_t = []

    def forward(self, x, t, condition=None):
        self.seen_t.append(t.detach().clone())
        return torch.zeros_like(x)


@pytest.fixture
def rng_model(monkeypatch):
    p = CorrDiffEra5Hrrr.__new__(CorrDiffEra5Hrrr)
    torch.nn.Module.__init__(p)
    p.register_buffer("lat_input_grid", torch.linspace(30, 27, 4))
    p.lat_input_numpy = p.lat_input_grid.numpy()
    p.lon_input_numpy = np.arange(8) + 260.0
    p.register_buffer("lat_output_grid", torch.ones(2, 3) * 28)
    p._lat_out_cpu = p.lat_output_grid.numpy()
    p._lon_out_cpu = np.ones((2, 3)) * 262
    p.hrrr_y, p.hrrr_x = np.arange(2), np.arange(3)
    p.era5_variables, p.output_variables = np.array(["t2m"]), np.array(["t2m"])
    p.number_of_samples = 2
    p.number_of_steps = 2
    p.solver = "euler"
    p.network_kind = "edm"
    p.sigma_min, p.sigma_max, p.rho = 0.01, 1.0, 7.0
    p.t_max, p.shift = 0.99, 1.0
    p.amp = False
    p.register_buffer("invariants", torch.zeros(1, 2, 3))
    p.register_buffer("out_center", torch.zeros(1, 1, 2, 3))
    p.register_buffer("out_scale", torch.ones(1, 1, 2, 3))
    p.preprocess_input = lambda x, time: None

    class Scheduler:
        def __init__(self, **kwargs):
            pass

        def get_denoiser(self, **kwargs):
            return None

        def sigma(self, t):
            return t

    monkeypatch.setattr(corrdiff_module, "EDMNoiseScheduler", Scheduler)
    monkeypatch.setattr(corrdiff_module, "RectifiedFlowNoiseScheduler", Scheduler)
    monkeypatch.setattr(
        corrdiff_module, "sample", lambda denoiser, latents, scheduler, **kw: latents
    )
    p._rf_time_steps = lambda device, scheduler: torch.tensor([1.0, 0.0], device=device)
    return p


def test_corrdiff_era5_hrrr_conformance(rng_model):
    assert check_diagnostic_contract(rng_model) == []


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["edm", "rectified_flow"])
def test_corrdiff_era5_hrrr_rng(rng_model, device, kind):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    p = rng_model.to(device)
    p.network_kind = kind
    x = torch.zeros(1, 4, 8, device=device)
    time = datetime(2025, 7, 1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if x.is_cuda else None
    p.set_rng(42, reset=False)
    first = p._forward(x, time)
    p.set_rng(99, reset=False)
    second = p._forward(x, time)
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    p.set_rng(42)
    assert torch.equal(first, p._forward(x, time))
    assert torch.equal(second, p._forward(x, time))
    p.set_rng(43)
    assert not torch.equal(first, p._forward(x, time))
    assert torch.equal(cpu_state, torch.get_rng_state())
    if x.is_cuda:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))


def test_corrdiff_era5_hrrr_unseeded_rng(rng_model):
    state = torch.get_rng_state()
    rng_model._forward(torch.zeros(1, 4, 8), datetime(2025, 7, 1))
    assert not torch.equal(state, torch.get_rng_state())


@pytest.fixture
def model_args():
    lat, lon = torch.meshgrid(
        torch.linspace(27, 29, 4), torch.linspace(261, 264, 6), indexing="ij"
    )
    return dict(
        network=PhooNet(),
        lat_input_grid=torch.arange(30, 26, -0.5),
        lon_input_grid=torch.arange(260, 265, 0.5),
        lat_output_grid=lat + 0.05 * lon.sin(),
        lon_output_grid=lon,
        hrrr_y=torch.arange(4) * 3000,
        hrrr_x=torch.arange(6) * 3000,
        era5_center=torch.zeros(26),
        era5_scale=torch.ones(26),
        out_center=torch.arange(99).float(),
        out_scale=torch.ones(99) * 2,
        invariants=torch.randn(3, 4, 6),
        presence_flags=["tcwv", "sp"],
        number_of_samples=2,
        number_of_steps=3,
        amp=False,
    )


@pytest.mark.parametrize(
    "kind,prediction,batch_size,device",
    [
        ("rectified_flow", "x0", 1, "cpu"),
        ("rectified_flow", "flow", 2, "cpu"),
        ("edm", "x0", 2, "cpu"),
        pytest.param(
            "rectified_flow",
            "x0",
            2,
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_corrdiff_era5_hrrr(model_args, kind, prediction, batch_size, device):
    dx = CorrDiffEra5Hrrr(
        **model_args, network_kind=kind, prediction_type=prediction
    ).to(device)
    dx.set_rng(0)
    if kind == "edm":
        from physicsnemo.diffusion.preconditioners import EDMPreconditioner

        dx.network = EDMPreconditioner(dx.network, sigma_data=0.5).to(device)
    coords = coord_array_like(
        dx.input_coords(),
        {
            "batch": np.arange(batch_size),
            "time": np.array(["2025-07-01T18", "2025-07-02T00"], dtype="datetime64[h]"),
        },
    )
    x = torch.randn(coords.shape, device=device)
    cond = dx.preprocess_input(x[0, 0], datetime(2025, 7, 1, 18))
    assert cond["cond_concat"].shape == (1, 30, 4, 6)
    assert cond["cond_vec"].shape == (1, 4)
    assert torch.all(cond["cond_vec"][0, 2:] == 1)
    assert cond["cond_concat"][0, 26].abs().max() <= 1
    assert torch.isfinite(cond["cond_concat"]).all()
    field = from_torch(x, coords)
    result = dx(field)
    out, out_coords = result.e2s.to_torch()
    assert out.shape == (batch_size, 2, 2, 99, 4, 6)
    assert out.device == torch.device(device) and torch.isfinite(out).all()
    np.testing.assert_array_equal(coords.coords["variable"], ERA5_VARIABLES)
    np.testing.assert_array_equal(out_coords["variable"], OUTPUT_VARIABLES)
    expected = dx.output_coords(coords)
    for index, dim in enumerate(
        ["batch", "sample", "time", "variable", "hrrr_y", "hrrr_x"]
    ):
        handshake_dim(out_coords, dim, index)
        np.testing.assert_array_equal(out_coords[dim], expected.coords[dim])
    dx.set_rng(0)
    torch.testing.assert_close(out, dx(field).e2s.to_torch()[0])
    assert not torch.allclose(out[:, 0], out[:, 1])
    dx.number_of_samples = 1
    dx.set_rng(0)
    single = dx(field).e2s.to_torch()[0]
    torch.testing.assert_close(single[0, 0, 0], out[0, 0, 0])
    if kind == "rectified_flow":
        from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

        assert 1 < max(float(t.max()) for t in model_args["network"].seen_t) < 999
        scheduler = RectifiedFlowNoiseScheduler(t_max=dx.t_max)
        shifted = dx._rf_time_steps(torch.device(device), scheduler)
        dx.shift = 1
        unshifted = dx._rf_time_steps(torch.device(device), scheduler)
        assert shifted.shape == unshifted.shape and shifted[-1] == 0
        assert torch.all(shifted[1:-1] > unshifted[1:-1])


def test_corrdiff_era5_hrrr_exceptions(model_args):
    dx = CorrDiffEra5Hrrr(**model_args)
    assert dx.network_kind == "rectified_flow"
    coords = coord_array_like(
        dx.input_coords(),
        {"batch": [0], "time": np.array(["2025-07-01"], dtype="datetime64[D]")},
    )
    x = from_torch(torch.zeros(coords.shape), coords)
    for dim in ["time", "variable", "lat", "lon"]:
        wrong = x.copy(deep=True)
        if dim == "time":
            wrong = wrong.rename({dim: "wrong"})
        else:
            values = coords.coords[dim].values
            wrong = wrong.assign_coords(
                {dim: values[::-1] if dim == "variable" else values + 1}
            )
        with pytest.raises((KeyError, ValueError)):
            dx(wrong)
    for key, value in dict(
        network_kind="ddpm",
        prediction_type="epsilon",
        solver="invalid",
        number_of_samples=0,
        number_of_steps=0,
        t_max=1,
        shift=0,
    ).items():
        with pytest.raises(ValueError):
            CorrDiffEra5Hrrr(**(model_args | {key: value}))
    with pytest.raises(ValueError, match="variant"):
        CorrDiffEra5Hrrr.load_model(None, variant="v_pred")
