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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import SFNO, ForecastInterpolation
from earth2studio.utils import handshake_dim


class PhooSFNOModel(torch.nn.Module):
    """Mock SFNO model for testing."""

    def forward(self, x, t):
        return x


class PhooInterpolationModel(torch.nn.Module):
    """Mock interpolation model for testing."""

    def forward(self, x, t_norm):
        # Stack by 6
        return torch.cat([x, x, x, x, x, x], dim=1)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("device", ["cpu"])
def test_forecast_interpolation_call(time, device):
    """Test basic forward pass of ForecastInterpolation model."""
    # Set up base SFNO model
    sfno_model = PhooSFNOModel()
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)
    base_model = SFNO(sfno_model, center, scale).to(device)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = ForecastInterpolation(
        interp_model=interp_model,
        fc_model=base_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        num_interp_steps=6,
    ).to(device)

    # Create domain coordinates
    dc = {k: model.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Run forward pass
    out, out_coords = model(x, coords)

    print(out.shape)
    print(out_coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Verify output shape and coordinates
    assert out.shape == torch.Size([len(time), 1, 73, 720, 1440])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out_coords["lead_time"][0] == np.timedelta64(
        1, "h"
    )  # Should output 1-hour steps

    # Verify coordinate system dimensions
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

    # Verify interpolation steps
    assert len(out_coords["lead_time"]) == 1  # Each call should return one timestep
    assert out_coords["lead_time"][0] == np.timedelta64(
        1, "h"
    )  # Should be hourly output
