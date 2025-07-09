# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.utils import handshake_dim


class PhooCorrDiff(torch.nn.Module):
    img_out_channels = 4
    img_resolution = 448
    sigma_min = 0
    sigma_max = float("inf")

    def forward(
        self, x, img_lr, sigma=None, class_labels=None, force_fp32=False, **model_kwargs
    ):
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

    out, out_coords = dx(x, coords)

    assert out.shape == torch.Size([x.shape[0], 1, 4, 448, 448])
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "sample", 1)
    handshake_dim(out_coords, "batch", 0)

    dx.number_of_samples = 2
    out, out_coords = dx(x, coords)

    assert out.shape == torch.Size([x.shape[0], 2, 4, 448, 448])
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "sample", 1)
    handshake_dim(out_coords, "batch", 0)


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
        dx(x, wrong_coords)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )

    with pytest.raises(ValueError):
        dx(x, wrong_coords)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 720),
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)


@pytest.mark.xfail  # TODO: REMOVE
@pytest.mark.ci_cache
@pytest.mark.timeout(30)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_corrdiff_package(device, model_cache_context):
    # Test the cached model package CorrDiffTaiwan
    # Only cuda supported
    with model_cache_context():
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

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 1, 4, 448, 448])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "sample", 1)
    handshake_dim(out_coords, "batch", 0)

    dx.number_of_samples = 2
    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 2, 4, 448, 448])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "sample", 1)
    handshake_dim(out_coords, "batch", 0)
