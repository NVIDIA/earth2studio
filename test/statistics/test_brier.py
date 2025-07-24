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

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import brier_score
from earth2studio.utils.coords import handshake_coords, handshake_dim


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs(ensemble_dimension: str, device: str) -> None:

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
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    reduction_dimensions = ["lat", "lon", "time"]

    BS = brier_score(
        reduction_dimensions=reduction_dimensions,
        thresholds=[0.25, 0.75],
        ensemble_dimension=ensemble_dimension,
    )

    z, c = BS(x, x_coords, y, y_coords)
    assert ensemble_dimension not in c
    assert "threshold" in c
    if reduction_dimensions is not None:
        assert all([rd not in c for rd in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]

    out_test_coords = BS.output_coords(x_coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs_failures(device: str) -> None:
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

    BS = brier_score(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        ensemble_dimension="ensemble",
    )

    # Test ensemble_dimension in y error
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = BS(x, x_coords, y, y_coords)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = BS(x, x_coords, y, y_coords)

    # Test rejection of reserved dimension names
    for forbidden_dim in ["threshold", "window_size"]:
        with pytest.raises(ValueError):
            xc = copy.deepcopy(x_coords)
            xc[forbidden_dim] = np.array([1])
            z, c = BS(x, xc, x, xc)

    # Test reduction_dimension not in x_coords
    with pytest.raises(ValueError):
        y_coords = OrderedDict({})
        for c in x_coords:
            y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)

        x_coords.pop("ensemble")
        z, c = BS(x, x_coords, y, y_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs_accuracy(device: str) -> None:
    x = torch.zeros((1, 2, 128, 128), device=device)

    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
        }
    )

    y = torch.zeros_like(x) + 0.75
    y_coords = copy.deepcopy(x_coords)

    BS = brier_score(reduction_dimensions=["lat", "lon", "time"], thresholds=[0.5])

    # test that BS is 0 for comparison to self
    z, c = BS(x, x_coords, x, x_coords)
    assert (z == 0.0).all()

    # test that BS is 1 if x and y are always on different side of threshold
    x = torch.zeros_like(x)
    z, c = BS(x, x_coords, y, y_coords)
    assert (z == 1.0).all()
