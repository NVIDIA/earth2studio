# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import crps
from earth2studio.statistics.crps import _crps_from_empirical_cdf


@pytest.mark.parametrize(
    "reduction_dimension",
    [
        "ensemble",
        "time",
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_crps(reduction_dimension: str, device: str) -> None:

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop(reduction_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    CRPS = crps(reduction_dimension)

    z, c = CRPS(x, x_coords, y, y_coords)
    assert reduction_dimension not in c
    assert list(z.shape) == [len(val) for val in c.values()]
    assert z.shape == y.shape


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_crps_failures(device: str) -> None:
    reduction_dimension = "ensemble"
    x = torch.randn((10, 1, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    # Raise error for training to pass a list
    with pytest.raises(ValueError):
        crps(["ensemble", "time"])

    CRPS = crps(reduction_dimension)

    # Test reduction_dimension in y error
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = CRPS(x, x_coords, y, y_coords)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            if c != reduction_dimension:
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = CRPS(x, x_coords, y, y_coords)

    # Test reduction_dimension not in x_coords
    with pytest.raises(ValueError):
        y_coords = OrderedDict({})
        for c in x_coords:
            if c != reduction_dimension:
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)

        x_coords.pop("ensemble")
        z, c = CRPS(x, x_coords, y, y_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_crps_accuracy(device: str, rtol: float = 1e-2, atol: float = 1e-2) -> None:
    # Uses eq (5) from Gneiting et al. https://doi.org/10.1175/MWR2904.1
    # crps(N(0, 1), 0.0) = 2 / sqrt(2*pi) - 1/sqrt(pi) ~= 0.23...
    x = 3.0 + torch.randn((10_000, 1), device=device, dtype=torch.float32)
    y = 3.0 + torch.zeros((1,), device=device, dtype=torch.float32)

    # Test pure crps
    c = _crps_from_empirical_cdf(x, y, dim=0)
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    x = 3.0 + torch.randn((1, 10_000), device=device, dtype=torch.float32)

    # Test pure crps
    c = _crps_from_empirical_cdf(x, y, dim=1)
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
