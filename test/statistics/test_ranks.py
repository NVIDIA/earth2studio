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

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import rank_histogram
from earth2studio.utils.coords import handshake_coords, handshake_dim


@pytest.mark.parametrize(
    "ensemble_dimension",
    [
        "ensemble",
        "time",
    ],
)
@pytest.mark.parametrize("number_of_bins", [5, None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rank_histogram(
    ensemble_dimension: str, number_of_bins: int | None, device: str
) -> None:

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    reduction_dimensions = ["lat", "lon"]
    RH = rank_histogram(
        ensemble_dimension, reduction_dimensions, number_of_bins=number_of_bins
    )

    z, c = RH(x, x_coords, y, y_coords)
    for di in [ensemble_dimension] + reduction_dimensions:
        assert di not in c
    assert list(z.shape) == [len(val) for val in c.values()]

    out_test_coords = RH.output_coords(x_coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("number_of_bins", [5, None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rank_histogram_broadcasting(number_of_bins: int | None, device: str) -> None:
    x = torch.randn((1, 2, 10, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "ensemble": np.arange(10),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    RH = rank_histogram("ensemble", ["lat", "lon"], number_of_bins=number_of_bins)
    z, c = RH(x, x_coords, y, y_coords)
    expected_bins = (
        len(x_coords["ensemble"]) + 1 if number_of_bins is None else number_of_bins
    )
    assert len(c["bin"]) == expected_bins
    assert z.shape == (2, expected_bins, 1, 2)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rank_histogram_failures(device: str) -> None:
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
        rank_histogram(["ensemble", "time"], ["lat"])

    RH = rank_histogram(reduction_dimension, ["lat", "lon"])

    # Test reduction_dimension in y error
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = RH(x, x_coords, y, y_coords)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            if c != reduction_dimension:
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = RH(x, x_coords, y, y_coords)

    # Test reduction_dimension not in x_coords
    with pytest.raises(ValueError):
        y_coords = OrderedDict({})
        for c in x_coords:
            if c != reduction_dimension:
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)

        x_coords.pop("ensemble")
        z, c = RH(x, x_coords, y, y_coords)


@pytest.mark.parametrize("reduction_dimension", [20, 50, 80])
@pytest.mark.parametrize("number_of_bins", [3, 5, 8])
@pytest.mark.parametrize("distribution", ["random", "zeros"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rank_histogram_accuracy(
    reduction_dimension: int, number_of_bins: int, distribution: str, device: str
) -> None:
    x_shape = (1000, reduction_dimension, reduction_dimension)
    if distribution == "random":
        x = 1.0 + torch.randn(x_shape, device=device) ** 2
    elif distribution == "zeros":
        x = torch.zeros(*x_shape, device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(1000),
            "lat": np.arange(reduction_dimension),
            "lon": np.arange(reduction_dimension),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y_shape = [len(y_coords[c]) for c in y_coords]
    if distribution == "random":
        y = 1.0 + torch.randn(y_shape, device=device) ** 2
    elif distribution == "zeros":
        y = torch.zeros(*y_shape, device=device)

    reduction_dimensions = ["lat", "lon"]
    RH = rank_histogram("ensemble", reduction_dimensions, number_of_bins=number_of_bins)

    z, _ = RH(x, x_coords, y, y_coords)

    # This should be result in a near uniform distribution but is prone to statistical error
    assert torch.allclose(
        z[1, :],
        reduction_dimension**2 / number_of_bins * torch.ones_like(z[1, :]),
        rtol=2.0 * number_of_bins / reduction_dimension,
    )
