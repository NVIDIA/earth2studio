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

from earth2studio.models.dx import ClimateNet
from earth2studio.utils import handshake_dim


class PhooCNet(torch.nn.Module):
    def forward(self, x):
        return x[:, :3, :, :]


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 4, 721, 1440),
        torch.randn(2, 4, 721, 1440),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cnet(x, device):
    # Just test forward pass of AFNO with spoof model
    model = PhooCNet()
    center = torch.zeros(4, 1, 1)
    scale = torch.ones(4, 1, 1)

    dx = ClimateNet(model, center, scale).to(device)
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

    assert out.shape == torch.Size([x.shape[0], 3, 721, 1440])
    assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cnet_exceptions(device):

    model = PhooCNet()
    center = torch.zeros(4)
    scale = torch.ones(4)

    dx = ClimateNet(model, center, scale).to(device)
    x = torch.randn(1).to(device)
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


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cnet_package(device):
    # Only cuda supported
    package = ClimateNet.load_default_package()
    dx = ClimateNet.load_model(package).to(device)
    x = torch.randn(2, 4, 721, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 3, 721, 1440])
    # Check that we get 0-1 masks
    assert torch.all(out <= 1)
    assert torch.all(out >= 0)
    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)
