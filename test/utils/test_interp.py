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

import numpy as np
import pytest
import torch

from earth2studio.utils.interp import LatLonInterpolation, NearestNeighborInterpolator


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("input_type", ["zeros", "random", "gradient"])
def test_interpolation(device, input_type):
    (lat_in, lon_in) = np.meshgrid(
        np.arange(35.0, 38.0, 0.25), np.arange(5.0, 8.0, 0.25), indexing="ij"
    )
    (lat_out, lon_out) = np.meshgrid(
        np.arange(36.0, 37.0, 0.1), np.arange(6.0, 7.0, 0.1), indexing="ij"
    )

    interp = LatLonInterpolation(lat_in, lon_in, lat_out, lon_out)
    interp.to(device=device)
    if input_type == "zeros":
        x = torch.zeros(lat_in.shape, device=device)
    elif input_type == "random":
        x = torch.rand(*lat_in.shape, device=device)
    elif input_type == "gradient":
        x = (
            torch.linspace(0, 1, lat_in.shape[1], device=device)
            .unsqueeze(0)
            .repeat(lat_in.shape[0], 1)
        )

    y = interp(x)

    if input_type == "zeros":
        assert (y == 0).all()
    elif input_type == "random":
        assert ((y >= 0) & (y <= 1)).all()
    elif input_type == "gradient":
        assert (y[:, 1:] > y[:, :-1]).all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_interpolation_analytical(device):
    lat_in = np.array([[0.0, 0.0], [1.0, 1.0]])
    lon_in = np.array([[0.0, 1.0], [0.0, 1.0]])

    (lat_out, lon_out) = np.mgrid[:1.01:0.25, :1.01:0.25]

    interp = LatLonInterpolation(lat_in, lon_in, lat_out, lon_out)
    interp.to(device=device)

    x = torch.tensor([[0.0, 1.0], [1.0, 2.0]], device=device)
    y = interp(x)

    y_correct = torch.tensor(
        [
            [0.00, 0.25, 0.50, 0.75, 1.00],
            [0.25, 0.50, 0.75, 1.00, 1.25],
            [0.50, 0.75, 1.00, 1.25, 1.50],
            [0.75, 1.00, 1.25, 1.50, 1.75],
            [1.00, 1.25, 1.50, 1.75, 2.00],
        ],
        device=device,
    )

    epsilon = 1e-6  # allow for some FP roundoff
    assert (abs(y - y_correct) < epsilon).all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_nearest_neighbor_interpolator_1d_correctness(device):
    source_lats = torch.tensor([0.0, 45.0])
    source_lons = torch.tensor([0.0, 90.0])
    target_lats = torch.tensor([1.0, 44.0])
    target_lons = torch.tensor([5.0, 85.0])

    interp = NearestNeighborInterpolator(
        source_lats=source_lats,
        source_lons=source_lons,
        target_lats=target_lats,
        target_lons=target_lons,
        max_dist_km=1000.0,
    ).to(device=device)

    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    y = interp(values)

    expected = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    assert torch.equal(y, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_nearest_neighbor_interpolator_2d_correctness(device):
    source_lats = torch.tensor([[0.0, 0.0], [45.0, 45.0]])
    source_lons = torch.tensor([[0.0, 90.0], [0.0, 90.0]])
    target_lats = torch.tensor([[1.0, 1.0], [44.0, 44.0]])
    target_lons = torch.tensor([[5.0, 85.0], [5.0, 85.0]])

    interp = NearestNeighborInterpolator(
        source_lats=source_lats,
        source_lons=source_lons,
        target_lats=target_lats,
        target_lons=target_lons,
        max_dist_km=20000.0,
    ).to(device=device)

    values = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    y = interp(values)

    expected = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    assert torch.equal(y, expected)
