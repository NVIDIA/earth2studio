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

from earth2studio.statistics import fss
from earth2studio.utils.coords import handshake_coords, handshake_dim


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss(ensemble_dimension: str | None, device: str) -> None:

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

    FSS = fss(
        reduction_dimensions=reduction_dimensions,
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
        ensemble_dimension=ensemble_dimension,
    )

    z, c = FSS(x, x_coords, y, y_coords)
    assert ensemble_dimension not in c
    assert "threshold" in c
    assert "window_size" in c
    if reduction_dimensions is not None:
        assert all([rd not in c for rd in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]

    out_test_coords = FSS.output_coords(x_coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_failures(device: str) -> None:
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

    FSS = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
        ensemble_dimension="ensemble",
    )

    # Test ensemble_dimension in y error
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = FSS(x, x_coords, y, y_coords)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        z, c = FSS(x, x_coords, y, y_coords)

    # Test rejection of reserved dimension names
    for forbidden_dim in ["threshold", "window_size"]:
        with pytest.raises(ValueError):
            xc = copy.deepcopy(x_coords)
            xc[forbidden_dim] = np.array([1])
            z, c = FSS(x, xc, x, xc)

    # Test reduction_dimension not in x_coords
    with pytest.raises(ValueError):
        y_coords = OrderedDict({})
        for c in x_coords:
            y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)

        x_coords.pop("ensemble")
        z, c = FSS(x, x_coords, y, y_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_accuracy(device: str) -> None:
    # make a test x with 50% of the data set to 1 and the rest to 0
    x = torch.zeros((1, 2, 128, 128), device=device)
    x[:, :, ::2, ::2] = 1.0
    x[:, :, 1::2, 1::2] = 1.0

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

    FSS = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.5],
        window_sizes=[10],
    )

    z, c = FSS(x, x_coords, y, y_coords)
    # this (==0.8) should be the fss for 50% exceedence
    reference_value = 1 - 0.5**2 / (0.5**2 + 1.0**2)
    # allow some tolerance due to finite-size windows
    print((z - reference_value).cpu().numpy())
    assert (abs(z - reference_value) < 0.01 * reference_value).all()

    # test that FSS is 1 for comparison to self
    z, c = FSS(x, x_coords, x, x_coords)
    assert (z == 1.0).all()

    # test that FSS is 0 if x and y are always on different side of threshold
    x = torch.zeros_like(x)
    z, c = FSS(x, x_coords, y, y_coords)
    assert (z == 0.0).all()


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_spatial_dims(ensemble_dimension: str | None, device: str) -> None:
    # test that we can use spatial dimensions that are not the last 2 dims
    x = torch.zeros((10, 1, 128, 128, 2), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
            "variable": np.array(["t2m", "tcwv"]),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    FSS = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
        spatial_dimensions=["lat", "lon"],
        ensemble_dimension=ensemble_dimension,
    )

    z, c = FSS(x, x_coords, y, y_coords)
    out_test_coords = FSS.output_coords(x_coords)

    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)
