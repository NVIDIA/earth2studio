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

from earth2studio.models.dx import SolarRadiationAFNO1H, SolarRadiationAFNO6H
from earth2studio.utils import handshake_dim


class PhooAFNOSolarRadiation(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, x):
        # x: (batch, variables, lat, lon)
        # The model expects input shape (batch, variables, lat, lon)
        # where variables includes the input variables plus sza, sincos_latlon, orography, and landsea_mask
        # We'll return a tensor of the same shape but with only one variable
        return torch.zeros_like(x[:, :1, :, :])


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = PhooAFNOSolarRadiation()
    return model


@pytest.mark.parametrize(
    "x",
    [
        # Single time, single lead time
        torch.randn(1, 1, 1, 24, 721, 1440),
        # Multiple times, single lead time
        torch.randn(1, 2, 1, 24, 721, 1440),
        # Single time, multiple lead times
        torch.randn(1, 1, 2, 24, 721, 1440),
        # Multiple times, multiple lead times
        torch.randn(1, 2, 2, 24, 721, 1440),
        # Multiple batch, multiple times, multiple lead times
        torch.randn(2, 2, 2, 24, 721, 1440),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "model_class,freq",
    [
        (SolarRadiationAFNO1H, "1h"),
        (SolarRadiationAFNO6H, "6h"),
    ],
)
def test_solarradiation_afno(x, device, mock_model, model_class, freq):
    """Test basic functionality of SolarRadiationAFNO models."""
    # Create mock tensors for model initialization
    era5_mean = torch.zeros(24, 1, 1)
    era5_std = torch.ones(24, 1, 1)
    ssrd_mean = torch.zeros(1, 1, 1)
    ssrd_std = torch.ones(1, 1, 1)
    orography = torch.zeros(1, 1, 721, 1440)
    landsea_mask = torch.zeros(1, 1, 721, 1440)
    sincos_latlon = torch.zeros(1, 4, 721, 1440)

    model = model_class(
        core_model=mock_model,
        freq=freq,
        era5_mean=era5_mean,
        era5_std=era5_std,
        ssrd_mean=ssrd_mean,
        ssrd_std=ssrd_std,
        orography=orography,
        landsea_mask=landsea_mask,
        sincos_latlon=sincos_latlon,
    ).to(device)
    x = x.to(device)

    # Create time array based on number of time steps
    times = np.array(
        [
            np.datetime64("2024-01-01") + np.timedelta64(i, "h")
            for i in range(x.shape[1])
        ]
    )
    # Create lead time array based on number of lead time steps
    lead_times = np.array(
        [np.timedelta64(int(freq[:-1]) * i, "h") for i in range(x.shape[2])]
    )

    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "time": times,
            "lead_time": lead_times,
            "variable": model.input_coords()["variable"],
            "lat": model.input_coords()["lat"],
            "lon": model.input_coords()["lon"],
        }
    )

    # Run model
    out, out_coords = model(x, coords)
    assert out.shape == (x.shape[0], x.shape[1], x.shape[2], 1, 721, 1440)
    assert out_coords["variable"] == np.array(["ssrd"])
    assert out_coords["lat"].shape == (721,)
    assert out_coords["lon"].shape == (1440,)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1e6)


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
@pytest.mark.parametrize(
    "model_class,freq",
    [
        (SolarRadiationAFNO1H, "1h"),
        (SolarRadiationAFNO6H, "6h"),
    ],
)
def test_solarradiation_afno_invalid_coords(
    invalid_coords, mock_model, model_class, freq
):
    """Test solar radiation model with invalid coordinates."""
    # Create mock tensors for model initialization
    era5_mean = torch.zeros(24, 1, 1)
    era5_std = torch.ones(24, 1, 1)
    ssrd_mean = torch.zeros(1, 1, 1)
    ssrd_std = torch.ones(1, 1, 1)
    orography = torch.zeros(1, 1, 721, 1440)
    landsea_mask = torch.zeros(1, 1, 721, 1440)
    sincos_latlon = torch.zeros(1, 4, 721, 1440)

    model = model_class(
        core_model=mock_model,
        freq=freq,
        era5_mean=era5_mean,
        era5_std=era5_std,
        ssrd_mean=ssrd_mean,
        ssrd_std=ssrd_std,
        orography=orography,
        landsea_mask=landsea_mask,
        sincos_latlon=sincos_latlon,
    )

    x = torch.randn((1, 1, 1, len(invalid_coords["variable"]), 721, 1440))
    with pytest.raises(ValueError):
        model(x, invalid_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "model_class,freq",
    [
        (SolarRadiationAFNO1H, "1h"),
        (SolarRadiationAFNO6H, "6h"),
    ],
)
def test_solarradiation_afno_exceptions(device, mock_model, model_class, freq):
    """Test exception handling for invalid inputs."""
    # Create mock tensors for model initialization
    era5_mean = torch.zeros(24, 1, 1)
    era5_std = torch.ones(24, 1, 1)
    ssrd_mean = torch.zeros(1, 1, 1)
    ssrd_std = torch.ones(1, 1, 1)
    orography = torch.zeros(1, 1, 721, 1440)
    landsea_mask = torch.zeros(1, 1, 721, 1440)
    sincos_latlon = torch.zeros(1, 4, 721, 1440)

    model = model_class(
        core_model=mock_model,
        freq=freq,
        era5_mean=era5_mean,
        era5_std=era5_std,
        ssrd_mean=ssrd_mean,
        ssrd_std=ssrd_std,
        orography=orography,
        landsea_mask=landsea_mask,
        sincos_latlon=sincos_latlon,
    ).to(device)

    # Test invalid input shapes
    x = torch.randn((1, 1, 1, 24, 721, 1440), device=device)
    coords = OrderedDict(
        {
            "batch": np.ones(1),
            "time": np.array([np.datetime64("2024-01-01")]),
            "lead_time": np.array([np.timedelta64(int(freq[:-1]), "h")]),
            "variable": model.input_coords()["variable"],
            "lat": model.input_coords()["lat"],
            "lon": model.input_coords()["lon"],
        }
    )

    # Test invalid coordinate dimensions
    wrong_coords = coords.copy()
    wrong_coords["lat"] = np.linspace(
        90, -90, 722, endpoint=False
    )  # Wrong lat dimension
    with pytest.raises(ValueError):
        model(x, wrong_coords)

    # Test missing required coordinates
    wrong_coords = coords.copy()
    del wrong_coords["lat"]
    with pytest.raises(ValueError):
        model(x, wrong_coords)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "model_class,freq",
    [
        (SolarRadiationAFNO1H, "1h"),
        (SolarRadiationAFNO6H, "6h"),
    ],
)
def test_solarradiation_afno_package(device, model_class, freq):
    # Only cuda supported
    package = model_class.load_default_package()
    dx = model_class.load_model(package).to(device)
    x = torch.randn(2, 1, 1, 24, 721, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "time": np.array([np.datetime64("2024-01-01T00:00")]),
            "lead_time": np.array([np.timedelta64(int(freq[:-1]), "h")]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 1, 1, 1, 721, 1440])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)
    assert torch.all(out >= 0)
