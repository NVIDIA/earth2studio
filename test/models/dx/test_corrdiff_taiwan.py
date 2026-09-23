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

import numpy as np
import pytest
import torch
from test_corrdiff import _input_field
from test_corrdiff import offline_corrdiff as offline_corrdiff

from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import CorrDiffTaiwan


class PhooCorrDiff(torch.nn.Module):
    img_out_channels = 4
    img_resolution = 448
    sigma_min = 0
    sigma_max = float("inf")

    def __init__(self):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    @device.setter
    def device(self, value) -> None:
        # Accept torch.device or string like "cuda:0"/"cpu"
        dev = torch.device(value)
        self.device_buffer = torch.empty(0, device=dev)

    def forward(self, x, img_lr, class_labels=None, force_fp32=False, **model_kwargs):
        return x[:, :4]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 12, 36, 40),
        torch.randn(2, 12, 36, 40),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_corrdiff(x, device):
    # Just test forward pass of CorrDiff with spoof model
    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))

    out_lon, out_lat = torch.meshgrid(lon, lat)
    dx = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
    ).to(device)
    x = x.to(device)

    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out = dx(_input_field(dx, x, coords))
    out_coords = out.coords

    assert out.shape == torch.Size([x.shape[0], 1, 4, 448, 448])
    assert all(
        out_coords["variable"]
        == dx.output_coords(_input_field(dx, x, coords))["variable"]
    )
    assert out.dims == ("batch", "sample", "variable", "y", "x")

    dx.number_of_samples = 2
    out = dx(_input_field(dx, x, coords))
    out_coords = out.coords

    assert out.shape == torch.Size([x.shape[0], 2, 4, 448, 448])
    assert all(
        out_coords["variable"]
        == dx.output_coords(_input_field(dx, x, coords))["variable"]
    )
    assert out.dims == ("batch", "sample", "variable", "y", "x")


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 12, 36, 40),
        torch.randn(2, 12, 36, 40),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_corrdiff_exceptions(x, device):

    # Just test forward pass of CorrDiff with spoof model
    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))

    out_lon, out_lat = torch.meshgrid(lon, lat)
    dx = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
    ).to(device)
    x = x.to(device)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "wrong": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    with pytest.raises((KeyError, ValueError)):
        dx(_input_field(dx, x, wrong_coords))

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )

    with pytest.raises(ValueError):
        dx(
            _input_field(dx, x, wrong_coords).transpose(
                "batch", "variable", "lon", "lat"
            )
        )

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 720),
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(_input_field(dx, x, wrong_coords))


def test_corrdiff_taiwan_conformance():
    """Model contract conformance (dev/spec/MODEL_CONTRACT_SPEC.md).

    Reuses the same mock construction as ``test_corrdiff`` (no ``seed=``
    override, matching the constructor default). CorrDiffTaiwan draws its
    diffusion-sampler latents from a seed that defaults to a fresh
    ``np.random.randint`` draw per call when unset, but the class declares no
    ``stochastic`` attribute and implements no ``set_rng``: it defaults to the
    contract's ``stochastic=False`` reading. Two calls on the same input then
    draw independent sampler noise and disagree -- even with the deterministic
    passthrough ``PhooCorrDiff.forward`` -- which genuinely violates ``D9``.
    This is a wrapper defect (no ``stochastic``/``set_rng`` declaration to make
    diffusion sampling reproducible), not a test issue, and is tracked for a
    follow-up fix rather than papered over here.
    """
    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))
    out_lon, out_lat = torch.meshgrid(lon, lat)
    dx = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
    )
    assert check_diagnostic_contract(dx) == []


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_corrdiff_package(device):
    package = CorrDiffTaiwan.load_default_package()
    dx = CorrDiffTaiwan.load_model(package).to(device)

    x = torch.randn(2, 12, 36, 40).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    field = _input_field(dx, x, coords)
    out = dx(field)
    out_coords = out.coords
    assert out.shape == torch.Size([x.shape[0], 1, 4, 448, 448])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(field)["variable"])
    assert out.dims == ("batch", "sample", "variable", "y", "x")

    dx.number_of_samples = 2
    out = dx(field)
    out_coords = out.coords
    assert out.shape == torch.Size([x.shape[0], 2, 4, 448, 448])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(field)["variable"])
    assert out.dims == ("batch", "sample", "variable", "y", "x")
