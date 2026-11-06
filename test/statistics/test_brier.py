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
import xarray as xr

from earth2studio.statistics import brier_score
from earth2studio.utils.coords import handshake_coords
from earth2studio.utils.cupy import from_torch


def _check_output_coords(stat, x: xr.DataArray, z: xr.DataArray) -> None:
    """Check that the a priori output signature matches the computed output"""
    out_test = stat.output_coords(x)
    assert out_test.dims == z.dims
    assert out_test.shape == z.shape
    assert list(out_test.coords) == list(z.coords)
    for d in z.dims:
        handshake_coords(out_test, z, d)


def _to_np(z: xr.DataArray) -> np.ndarray:
    return z.e2s.to_torch()[0].cpu().numpy()


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", "member", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs(ensemble_dimension: str, device: str) -> None:

    x_coords = OrderedDict(
        {
            ensemble_dimension or "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )
    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), x_coords)

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    reduction_dimensions = ["lat", "lon", "time"]

    BS = brier_score(
        reduction_dimensions=reduction_dimensions,
        thresholds=[0.25, 0.75],
        ensemble_dimension=ensemble_dimension,
    )

    z = BS(x, y)
    assert ensemble_dimension not in z.dims
    assert "threshold" in z.dims
    if reduction_dimensions is not None:
        assert all([rd not in z.dims for rd in reduction_dimensions])
    assert list(z.shape) == [z.sizes[d] for d in z.dims]
    assert z.sizes["threshold"] == 2

    _check_output_coords(BS, x, z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs_failures(device: str) -> None:
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )
    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), x_coords)

    BS = brier_score(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        ensemble_dimension="ensemble",
    )

    # Test ensemble_dimension in y error
    y_coords = copy.deepcopy(x_coords)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)
    with pytest.raises(ValueError):
        BS(x, y)

    # Test x and y don't have broadcastable shapes
    y_coords = OrderedDict({"phony": np.arange(1)})
    for c in x_coords:
        y_coords[c] = x_coords[c]
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)
    with pytest.raises(ValueError):
        BS(x, y)

    # Test rejection of reserved dimension names
    for forbidden_dim in ["threshold", "window_size"]:
        xc = copy.deepcopy(x_coords)
        xc[forbidden_dim] = np.array([1])
        xf = from_torch(x.e2s.to_torch()[0].unsqueeze(-1), xc)
        with pytest.raises(ValueError):
            BS(xf, xf)

    # Test reduction_dimension not in x_coords
    y_coords = copy.deepcopy(x_coords)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    x_coords.pop("ensemble")
    x_noens = from_torch(x.e2s.to_torch()[0][0], x_coords)
    with pytest.raises(ValueError):
        BS(x_noens, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bs_accuracy(device: str) -> None:
    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
        }
    )

    x = from_torch(torch.zeros((1, 2, 128, 128), device=device), x_coords)
    y_coords = copy.deepcopy(x_coords)
    y = from_torch(torch.zeros((1, 2, 128, 128), device=device) + 0.75, y_coords)

    BS = brier_score(reduction_dimensions=["lat", "lon", "time"], thresholds=[0.5])

    # test that BS is 0 for comparison to self
    z = BS(x, x)
    assert (_to_np(z) == 0.0).all()

    # test that BS is 1 if x and y are always on different side of threshold
    z = BS(x, y)
    assert (_to_np(z) == 1.0).all()
