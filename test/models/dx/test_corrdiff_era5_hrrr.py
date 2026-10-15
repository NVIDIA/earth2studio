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

from earth2studio.models.dx import CorrDiffEra5Hrrr
from earth2studio.models.dx.corrdiff_era5_hrrr import ERA5_VARIABLES, OUTPUT_VARIABLES
from earth2studio.utils import handshake_dim


class PhooNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_t = []

    def forward(self, x, t, condition=None):
        self.seen_t.append(t.detach().clone())
        return torch.zeros_like(x)


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
        seed=0,
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
    if kind == "edm":
        from physicsnemo.diffusion.preconditioners import EDMPreconditioner

        dx.network = EDMPreconditioner(dx.network, sigma_data=0.5).to(device)
    coords = dx.input_coords()
    coords.update(
        batch=np.arange(batch_size),
        time=np.array(["2025-07-01T18", "2025-07-02T00"], dtype="datetime64[h]"),
    )
    x = torch.randn(*(len(c) for c in coords.values()), device=device)
    cond = dx.preprocess_input(x[0, 0], datetime(2025, 7, 1, 18))
    assert cond["cond_concat"].shape == (1, 30, 4, 6)
    assert cond["cond_vec"].shape == (1, 4)
    assert torch.all(cond["cond_vec"][0, 2:] == 1)
    assert cond["cond_concat"][0, 26].abs().max() <= 1
    assert torch.isfinite(cond["cond_concat"]).all()
    out, out_coords = dx(x, coords)
    assert out.shape == (batch_size, 2, 2, 99, 4, 6)
    assert out.device == torch.device(device) and torch.isfinite(out).all()
    np.testing.assert_array_equal(coords["variable"], ERA5_VARIABLES)
    np.testing.assert_array_equal(out_coords["variable"], OUTPUT_VARIABLES)
    expected = dx.output_coords(coords)
    for index, dim in enumerate(
        ["batch", "sample", "time", "variable", "hrrr_y", "hrrr_x"]
    ):
        handshake_dim(out_coords, dim, index)
        np.testing.assert_array_equal(out_coords[dim], expected[dim])
    torch.testing.assert_close(out, dx(x, coords)[0])
    assert not torch.allclose(out[:, 0], out[:, 1])
    dx.number_of_samples = 1
    torch.testing.assert_close(dx(x, coords)[0], out[:, :1])
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
    coords = dx.input_coords()
    coords.update(
        batch=np.array([0]), time=np.array(["2025-07-01"], dtype="datetime64[D]")
    )
    x = torch.zeros(*(len(c) for c in coords.values()))
    for dim in ["time", "variable", "lat", "lon"]:
        wrong = coords.copy()
        if dim == "time":
            wrong["wrong"] = wrong.pop(dim)
        else:
            wrong[dim] = coords[dim][::-1] if dim == "variable" else coords[dim] + 1
        with pytest.raises((KeyError, ValueError)):
            dx(x, wrong)
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


@pytest.fixture
def tiny_distilled():
    """A distilled generator around a toy DiT (timm attention, runs on CPU)."""
    from earth2studio.models.dx.corrdiff_era5_hrrr import CorrDiffEra5HrrrDistilled

    torch.manual_seed(0)
    dit_args = dict(
        input_size=(4, 6),
        in_channels=99 + 30,  # 99 latents + 26 ERA5 + cos zenith + 3 invariants
        out_channels=99,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        condition_dim=4,
        attention_backend="timm",
    )
    net = CorrDiffEra5HrrrDistilled(dit_args, n_grid=8)
    # the DiT initialization zeroes the output projection; give it and the
    # heads some signal so the tests see non-trivial fields
    with torch.no_grad():
        for linear in (net.dit.detokenizer.proj_layer.output_projection, *net.heads):
            torch.nn.init.normal_(linear.weight, std=0.02)
            torch.nn.init.normal_(linear.bias, std=0.02)
    return net.eval().requires_grad_(False)


def _tiny_condition():
    from tensordict import TensorDict

    torch.manual_seed(1)
    return TensorDict(
        {"cond_concat": torch.randn(1, 30, 4, 6), "cond_vec": torch.randn(1, 4)},
        batch_size=[1],
    )


def test_corrdiff_era5_hrrr_distilled_grid_and_heads(tiny_distilled):
    net = tiny_distilled
    grid = net.grid
    assert grid.shape == (net.n_grid + 1,) and len(net.heads) == net.n_grid
    assert float(grid[0]) <= net.t_max and float(grid[-1]) == 0.0
    assert torch.all(grid[1:] < grid[:-1])
    # a head loaded with the backbone's own output projection must reproduce
    # the full DiT forward: features() + head() is an exact split of the DiT
    cond = _tiny_condition()
    x = torch.randn(1, 99, 4, 6)
    t = torch.full((1,), 0.7)
    base = net.dit.detokenizer.proj_layer.output_projection
    net.heads[3].load_state_dict(base.state_dict())
    feats = net.features(x, t, cond)
    assert feats.shape == (1, 2 * 3, 16)
    expected = net.dit(
        torch.cat([x, cond["cond_concat"]], dim=1),
        t * net.time_scale,
        condition=cond["cond_vec"],
    )
    torch.testing.assert_close(net.head(feats, 3), expected)
    assert net.head(feats, 3).shape == (1, 99, 4, 6)


def test_corrdiff_era5_hrrr_distilled_sample(tiny_distilled):
    net = tiny_distilled
    cond = _tiny_condition()
    noise = torch.randn(1, 99, 4, 6)
    out1 = net.sample(noise, cond, nfe=1)
    out2 = net.sample(noise, cond, nfe=2)
    for out in (out1, out2):
        assert out.shape == noise.shape and torch.isfinite(out).all()
    assert not torch.allclose(out1, out2)
    torch.testing.assert_close(net(noise, cond, nfe=2), out2)  # forward aliases sample
    torch.testing.assert_close(net.sample(noise.clone(), cond, nfe=2), out2)
    for nfe in (3, 4, 8):
        with pytest.raises(ValueError, match="1 or 2"):
            net.sample(noise, cond, nfe=nfe)


def test_corrdiff_era5_hrrr_distilled_checkpoint_roundtrip(tiny_distilled, tmp_path):
    from earth2studio.models.dx.corrdiff_era5_hrrr import CorrDiffEra5HrrrDistilled

    net = tiny_distilled
    path = tmp_path / "distilled.mdlus"
    net.save(str(path))
    loaded = CorrDiffEra5HrrrDistilled.from_checkpoint(str(path)).eval()
    assert isinstance(loaded, CorrDiffEra5HrrrDistilled)
    assert loaded.n_grid == net.n_grid and loaded.shift == net.shift
    assert loaded.time_scale == net.time_scale and loaded.x0v_clip == net.x0v_clip
    torch.testing.assert_close(loaded.grid, net.grid)
    cond = _tiny_condition()
    noise = torch.randn(1, 99, 4, 6)
    torch.testing.assert_close(
        loaded.sample(noise, cond, nfe=2), net.sample(noise, cond, nfe=2)
    )


def test_corrdiff_era5_hrrr_distilled_wrapper(model_args, tiny_distilled):
    args = model_args | {"network": tiny_distilled, "number_of_steps": 2}
    dx = CorrDiffEra5Hrrr(**args)
    assert dx.distilled and not CorrDiffEra5Hrrr(**model_args).distilled
    coords = dx.input_coords()
    coords.update(
        batch=np.arange(1),
        time=np.array(["2025-07-01T18", "2025-07-02T00"], dtype="datetime64[h]"),
    )
    x = torch.randn(*(len(c) for c in coords.values()))
    out, out_coords = dx(x, coords)
    assert out.shape == (1, 2, 2, 99, 4, 6) and torch.isfinite(out).all()
    np.testing.assert_array_equal(out_coords["variable"], OUTPUT_VARIABLES)
    torch.testing.assert_close(out, dx(x, coords)[0])  # seeded -> reproducible
    assert not torch.allclose(out[:, 0], out[:, 1])  # members differ
    dx.number_of_steps = 1
    out_1 = dx(x, coords)[0]
    assert out_1.shape == out.shape and not torch.allclose(out_1, out)
    # unsupported layouts are rejected both at construction and at call time
    with pytest.raises(ValueError, match="1 or 2"):
        CorrDiffEra5Hrrr(**(args | {"number_of_steps": 3}))
    dx.number_of_steps = 4
    with pytest.raises(ValueError, match="1 or 2"):
        dx(x, coords)
