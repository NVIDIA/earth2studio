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

from earth2studio.models.dx import DerivedRH, DerivedVPD, DerivedWS


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        (["10m", "80m"], (1, 2, 16, 32)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_ws(levels, shape, device):
    """Test wind speed derivation"""
    model = DerivedWS(levels).to(device)

    # Test input coordinates
    input_coords = model.input_coords()
    assert "variable" in input_coords
    assert input_coords["variable"].shape[0] == 2 * len(levels)  # u and v components
    assert all(f"u{level}" in input_coords["variable"] for level in levels)
    assert all(f"v{level}" in input_coords["variable"] for level in levels)

    coords = OrderedDict(
        {
            "time": np.arange(1),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, shape[-2]),
            "lon": np.linspace(0, 360, shape[-1]),
        }
    )

    # Check wind speed calculation
    x = torch.randn(shape).to(device)
    u = x[:, ::2]
    v = x[:, 1::2]
    expected_ws = torch.sqrt(u**2 + v**2).to(device)
    out, out_coords = model(x, coords)
    assert torch.allclose(out, expected_ws, rtol=1e-5)
    assert out.device == x.device
    assert out_coords["variable"].shape[0] == len(levels)

    # Test with zero wind components
    x = torch.zeros(shape).to(device)
    out, _ = model(x, coords)
    assert torch.allclose(out, torch.zeros_like(out))

    # Test with known values
    x = torch.ones(shape).to(device)
    out, _ = model(x, coords)
    expected_ws = torch.sqrt(torch.tensor(2.0)).expand_as(out).to(device)
    assert torch.allclose(out, expected_ws)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"time": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"time": np.array([0]), "variable": np.array(["u100"])}
        ),  # Missing v component
    ],
)
def test_derived_ws_invalid_coords(invalid_coords):
    """Test wind speed derivation with invalid coordinates"""
    model = DerivedWS([100])
    x = torch.randn(1, 1, 16, 32)

    with pytest.raises(ValueError):
        model(x, invalid_coords)

    # Wrong number of variables
    x = torch.randn(1, 3, 16, 32)  # 3 variables instead of 2
    with pytest.raises(ValueError):
        model(x, invalid_coords)


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        ([100, 200, 300], (1, 6, 32, 64)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_rh(levels, shape, device):

    model = DerivedRH(levels).to(device)

    batch_size = shape[0]
    n_levels = len(levels)
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )

    # Realistic temperature and specific humidity values
    t = 273.15 + 20
    q = 0.01
    x = torch.ones(shape).to(device)
    x[:, ::2] *= t
    x[:, 1::2] *= q

    out, out_coords = model(x, coords)

    # Check output shape and coordinates
    print(len(levels))
    assert out.shape == (batch_size, n_levels, lat_size, lon_size)
    assert "variable" in out_coords
    assert len(out_coords["variable"]) == len(levels)
    assert all(f"r{level}" in out_coords["variable"] for level in levels)

    # Very cold temperature (210K) should give near 100% RH
    x_cold = torch.ones(shape).to(device)
    x_cold[:, ::2] *= 210.0
    x_cold[:, 1::2] *= 0.001  # Low specific humidity
    out_cold, _ = model(x_cold, coords)
    assert out_cold.device == x_cold.device
    assert torch.all(out_cold <= 101)
    assert torch.all(out_cold >= 0)
    assert torch.all(out_cold >= 98)

    # Very warm temperature (350K) should give low RH
    x_warm = torch.ones(shape).to(device)
    x_warm[:, ::2] *= 350.0
    x_warm[:, 1::2] *= 0.001  # Low specific humidity
    out_warm, _ = model(x_warm, coords)
    assert out_warm.device == x_warm.device
    assert torch.all(out_warm <= 100)
    assert torch.all(out_warm >= 0)
    assert torch.all(out_warm <= 2)
    # Warm temperatures should give lower RH than cold temperatures for same q
    assert torch.all(out_warm < out_cold)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t100"])}
        ),  # Missing q component
    ],
)
def test_derived_rh_invalid_coords(invalid_coords):
    """Test relative humidity derivation with invalid coordinates"""
    model = DerivedRH([100])
    x = torch.randn(1, 1, 16, 32)

    with pytest.raises(ValueError):
        model(x, invalid_coords)


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        ([100, 200, 300], (1, 6, 32, 64)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_vpd(levels, shape, device):
    model = DerivedVPD(levels).to(device)

    batch_size = shape[0]
    n_levels = len(levels)
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )

    # Test with realistic temperature and RH values
    t = 273.15 + 25  # 25°C
    rh = 60  # 60% relative humidity
    x = torch.ones(shape).to(device)
    x[:, ::2] *= t  # Temperature channels
    x[:, 1::2] *= rh  # RH channels

    out, out_coords = model(x, coords)

    # Check output shape and coordinates
    assert out.shape == (batch_size, n_levels, lat_size, lon_size)
    assert "variable" in out_coords
    assert len(out_coords["variable"]) == len(levels)
    assert all(f"vpd{level}" in out_coords["variable"] for level in levels)

    # Test device consistency
    assert out.device == x.device

    # Test physical bounds and behavior
    # At 25°C and 60% RH, VPD should be around 1.2-1.3 kPa or 12-13 mb/hPa
    # https://www.dimluxlighting.com/knowledge/vapor-pressure-deficit-vpd-calculator/
    # https://www.omnicalculator.com/biology/vapor-pressure-deficit#what-is-vapor-pressure-deficit-vpd
    # https://en.wikipedia.org/wiki/Vapour-pressure_deficit
    expected_vpd = 12.5
    assert torch.allclose(
        torch.mean(out), torch.tensor(expected_vpd, device=device), rtol=0.2
    )

    # Test with saturated conditions (RH = 100%)
    x_sat = x.clone()
    x_sat[:, 1::2] = 100
    out_sat, _ = model(x_sat, coords)
    assert torch.allclose(out_sat, torch.zeros_like(out_sat), atol=1e-5)

    # Test with hot and dry conditions (high VPD)
    x_hot_dry = x.clone()
    x_hot_dry[:, ::2] = 273.15 + 35  # 35°C
    x_hot_dry[:, 1::2] = 20  # 20% RH
    out_hot_dry, _ = model(x_hot_dry, coords)
    assert torch.all(out_hot_dry > out)

    # Test with cold and humid conditions (low VPD)
    x_cold_humid = x.clone()
    x_cold_humid[:, ::2] = 273.15 + 10  # 10°C
    x_cold_humid[:, 1::2] = 90  # 90% RH
    out_cold_humid, _ = model(x_cold_humid, coords)
    assert torch.all(out_cold_humid < out)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t100"])}
        ),  # Missing rh component
    ],
)
def test_derived_vpd_invalid_coords(invalid_coords):
    """Test VPD derivation with invalid coordinates"""
    model = DerivedVPD([100])
    x = torch.randn(1, 1, 16, 32)

    with pytest.raises(ValueError):
        model(x, invalid_coords)
