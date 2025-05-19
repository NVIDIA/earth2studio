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

from earth2studio.models.dx import SolarRadiationAFNO
from earth2studio.utils import handshake_dim


class PhooAFNOSolarRadiation(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, x):
        return x[:, :1, :, :]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 16, 32),
        (2, 2, 32, 64),
        (4, 2, 48, 96),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_solarradiation_afno(shape, device):
    """Test basic functionality of SolarRadiationAFNO model."""
    package = SolarRadiationAFNO.load_default_package(cache=True)
    model = SolarRadiationAFNO.load_model(package).to(device)
    
    # Create input tensor and coordinates
    x = torch.randn(shape, device=device)
    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([np.datetime64("2024-01-01")]),
        "lead_time": np.array([np.timedelta64(6, "h")]),
        "variable": np.array(["t2m", "sza"]),
        "lat": np.linspace(90, -90, shape[2], endpoint=False),
        "lon": np.linspace(0, 360, shape[3], endpoint=False),
    })
    
    # Run model
    output, output_coords = model(x, coords)
    
    # Check output shape and coordinates
    assert output.shape == (shape[0], 1, shape[2], shape[3])
    assert "ssrd" in output_coords["variable"]
    assert len(output_coords["variable"]) == 1
    
    # Check physical bounds
    assert torch.all(output >= 0)  # Solar radiation should be non-negative
    assert torch.all(output <= 1e6)  # Reasonable upper bound for 6h accumulated radiation


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t2m"])}
        ),  # Missing sza component
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["sza"])}
        ),  # Missing t2m component
    ],
)
def test_solarradiation_afno_invalid_coords(invalid_coords):
    """Test solar radiation model with invalid coordinates."""
    package = SolarRadiationAFNO.load_default_package(cache=True)
    model = SolarRadiationAFNO.load_model(package)
    
    x = torch.randn((1, len(invalid_coords["variable"]), 16, 32))
    with pytest.raises(ValueError):
        model(x, invalid_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_solarradiation_afno_time_handling(device):
    """Test solar radiation model time handling."""
    package = SolarRadiationAFNO.load_default_package(cache=True)
    model = SolarRadiationAFNO.load_model(package).to(device)
    
    # Test different times of day
    times = [
        np.datetime64("2024-01-01T00:00:00"),  # Midnight
        np.datetime64("2024-01-01T06:00:00"),  # Dawn
        np.datetime64("2024-01-01T12:00:00"),  # Noon
        np.datetime64("2024-01-01T18:00:00"),  # Dusk
    ]
    
    x = torch.randn((1, 2, 16, 32), device=device)
    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([times[0]]),
        "lead_time": np.array([np.timedelta64(6, "h")]),
        "variable": np.array(["t2m", "sza"]),
        "lat": np.linspace(90, -90, 16, endpoint=False),
        "lon": np.linspace(0, 360, 32, endpoint=False),
    })
    
    # Check that radiation values are physically reasonable for different times
    for time in times:
        coords["time"] = np.array([time])
        output, _ = model(x, coords)
        
        # Noon should have higher radiation than midnight
        if time == np.datetime64("2024-01-01T12:00:00"):
            noon_output = output
        elif time == np.datetime64("2024-01-01T00:00:00"):
            midnight_output = output
            assert torch.mean(noon_output) > torch.mean(midnight_output)


@pytest.mark.ci_cache
@pytest.mark.timeout(15)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_solarradiation_afno_package(device, model_cache_context):
    """Test the cached model package SolarRadiationAFNO."""
    # Only cuda supported
    with model_cache_context():
        package = SolarRadiationAFNO.load_default_package()
        model = SolarRadiationAFNO.load_model(package).to(device)

    shape = (2, 2, 720, 1440)
    x = torch.randn(shape).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(shape[0]),
            "variable": model.input_coords()["variable"],
            "lat": model.input_coords()["lat"],
            "lon": model.input_coords()["lon"],
            "time": np.array([np.datetime64("2024-01-01T12:00:00")]),
        }
    )

    out, out_coords = model(x, coords)
    assert out.shape == torch.Size([shape[0], 1, 720, 1440])
    assert out_coords["variable"] == model.output_coords(coords)["variable"]
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_solarradiation_afno_exceptions(device):
    """Test exception handling for invalid inputs."""
    package = SolarRadiationAFNO.load_default_package(cache=True)
    model = SolarRadiationAFNO.load_model(package).to(device)
    
    # Test invalid input shapes
    x = torch.randn((1, 1, 16, 32), device=device)  # Missing sza component
    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([np.datetime64("2024-01-01")]),
        "lead_time": np.array([np.timedelta64(6, "h")]),
        "variable": np.array(["t2m"]),
        "lat": np.linspace(90, -90, 16, endpoint=False),
        "lon": np.linspace(0, 360, 32, endpoint=False),
    })
    
    with pytest.raises(ValueError):
        model(x, coords)
    
    # Test invalid coordinate dimensions
    x = torch.randn((1, 2, 16, 32), device=device)
    coords["lat"] = np.linspace(90, -90, 32, endpoint=False)  # Wrong lat dimension
    with pytest.raises(ValueError):
        model(x, coords) 