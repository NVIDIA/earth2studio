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

"""Tests for WindSpeed diagnostic model."""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import WindSpeed
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


def make_input(model, batch: int = 1, device: str = "cpu") -> xr.DataArray:
    signature = coord_array_like(model.input_coords(), {"batch": np.arange(batch)})
    return from_torch(torch.randn(signature.shape, device=device), signature)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_windspeed_call(batch, device):
    """Test forward pass of WindSpeed diagnostic."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    dx = WindSpeed().to(device)
    x = make_input(dx, batch, device)
    before = x.copy(deep=True)

    out = dx(x)

    assert out.shape == (batch, 1, 721, 1440)
    assert list(out.coords["variable"].values) == ["ws10m"]
    assert out.dims == x.dims
    handshake_dataarray(out, dx.output_coords(x))
    xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_windspeed_computation(device):
    """Test wind speed is computed correctly."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    dx = WindSpeed().to(device)
    signature = coord_array_like(dx.input_coords(), {"batch": [0]})
    # Known input: u=3, v=4 -> ws=5
    data = torch.zeros(signature.shape, device=device)
    data[:, 0] = 3.0
    data[:, 1] = 4.0
    x = from_torch(data, signature)

    out = dx(x)

    expected = torch.full((1, 1, 721, 1440), 5.0)
    torch.testing.assert_close(torch.as_tensor(out.e2s.as_numpy().values), expected)


def test_windspeed_exceptions():
    """Test WindSpeed raises on invalid inputs."""
    dx = WindSpeed()
    x = make_input(dx)

    # Wrong variable name
    with pytest.raises((KeyError, ValueError)):
        dx(x.assign_coords(variable=["wrong_u", "wrong_v"]))

    # Wrong dimension order
    with pytest.raises(ValueError):
        dx(x.transpose(*reversed(x.dims)))

    # Wrong grid size (matching grid IDs skip spatial value comparisons)
    with pytest.raises(ValueError):
        dx(x.isel(lat=slice(0, 360)))


def test_windspeed_conformance():
    """Test WindSpeed against the native diagnostic contract."""
    dx = WindSpeed()
    assert isinstance(dx, DiagnosticModel)
    assert check_diagnostic_contract(dx) == [
        "D10: model does not declare itself stochastic"
    ]
