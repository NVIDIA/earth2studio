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
