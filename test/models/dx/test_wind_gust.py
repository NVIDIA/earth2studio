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

from earth2studio.models.dx import WindgustAFNO
from earth2studio.utils import handshake_dim


class PhooAFNOWindgust(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, x):
        return x[:, :1, :, :]


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 1, 1, 17, 720, 1440),  # Single sample
        torch.randn(2, 2, 1, 17, 720, 1440),  # Multiple samples
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_windgust(x, device):
    """Test forward pass of AFNO with mock model."""
    # Create mock model and tensors
    model = PhooAFNOWindgust()
    center = torch.zeros(17, 1, 1)
    scale = torch.ones(17, 1, 1)
    lsm = torch.ones(1, 1, 720, 1440)
    orog = torch.ones(1, 1, 720, 1440)

    # Initialize model
    dx = WindgustAFNO(model, lsm, orog, center, scale).to(device)
    x = x.to(device)

    # Create coordinate system
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    # Run forward pass
    out, out_coords = dx(x, coords)

    # Check output shape and coordinates
    assert out.shape == (x.shape[0], x.shape[1], x.shape[2], 1, 720, 1440)
    assert out_coords["variable"] == ["fg10m"]
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_afno_windgust_package(device):
    package = WindgustAFNO.load_default_package()
    dx = WindgustAFNO.load_model(package).to(device)

    x = torch.randn(2, 1, 1, 17, 720, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)
    assert out.shape == (x.shape[0], x.shape[1], x.shape[2], 1, 720, 1440)
    assert out_coords["variable"] == ["fg10m"]
    assert torch.all(out >= 0)
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_windgust_exceptions(device):
    model = PhooAFNOWindgust()
    center = torch.zeros(17, 1, 1)
    scale = torch.ones(17, 1, 1)
    lsm = torch.ones(1, 720, 1440)
    orog = torch.ones(1, 720, 1440)

    dx = WindgustAFNO(model, lsm, orog, center, scale).to(device)
    x = torch.randn(1, 1, 1, 17, 720, 1440).to(device)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
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
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 721),  # Wrong size
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)
