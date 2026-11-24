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

from earth2studio.statistics import fss
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.cupy import from_torch


def _check_output(FSS, x, z) -> None:
    """Check output DataArray against the allocation-free output signature"""
    out = FSS.output_coords(x)
    assert z.dims == out.dims
    assert z.shape == out.shape
    for i, ci in enumerate(z.dims):
        handshake_dim(out, ci, i)
        handshake_coords(out, z, ci)


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss(ensemble_dimension: str | None, device: str) -> None:

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

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    reduction_dimensions = ["lat", "lon", "time"]

    FSS = fss(
        reduction_dimensions=reduction_dimensions,
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
        ensemble_dimension=ensemble_dimension,
    )

    z = FSS(x, y)
    assert ensemble_dimension not in z.dims
    assert "threshold" in z.dims
    assert "window_size" in z.dims
    assert all([rd not in z.dims for rd in reduction_dimensions])
    expected_dims = ("variable", "threshold", "window_size")
    if ensemble_dimension is None:
        expected_dims = ("ensemble",) + expected_dims
    assert z.dims == expected_dims

    _check_output(FSS, x, z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_failures(device: str) -> None:
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
        y = from_torch(torch.randn(y_shape, device=device), y_coords)
        FSS(x, y)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            if c != "ensemble":
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = from_torch(torch.randn(y_shape, device=device), y_coords)
        FSS(x, y)

    # Test rejection of reserved dimension names
    for forbidden_dim in ["threshold", "window_size"]:
        with pytest.raises(ValueError):
            xc = copy.deepcopy(x_coords)
            xc[forbidden_dim] = np.array([1])
            xf = from_torch(torch.randn((10, 1, 2, 361, 720, 1), device=device), xc)
            FSS(xf, xf)

    # Test ensemble dimension not in x (x has no ensemble dim but FSS expects one)
    with pytest.raises(ValueError):
        y_coords = OrderedDict({})
        for c in x_coords:
            y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = from_torch(torch.randn(y_shape, device=device), y_coords)

        xc = copy.deepcopy(x_coords)
        xc.pop("ensemble")
        xs = from_torch(torch.randn((1, 2, 361, 720), device=device), xc)
        FSS(xs, y)

    # Test reduction_dimension not in x and y
    FSS_t = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
    )
    xc = copy.deepcopy(x_coords)
    xc.pop("ensemble")
    xc.pop("time")
    xs = from_torch(torch.randn((2, 361, 720), device=device), xc)
    with pytest.raises((ValueError, KeyError)):
        FSS_t(xs, xs)

    # Test incompatible coordinate values
    yc = copy.deepcopy(x_coords)
    yc.pop("ensemble")
    yc["lat"] = np.linspace(-80.0, 80.0, 361)
    y = from_torch(torch.randn((1, 2, 361, 720), device=device), yc)
    with pytest.raises(ValueError):
        FSS(x, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_accuracy(device: str) -> None:
    # make a test x with 50% of the data set to 1 and the rest to 0
    x_t = torch.zeros((1, 2, 128, 128), device=device)
    x_t[:, :, ::2, ::2] = 1.0
    x_t[:, :, 1::2, 1::2] = 1.0

    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
        }
    )
    x = from_torch(x_t, x_coords)

    y = from_torch(torch.zeros_like(x_t) + 0.75, copy.deepcopy(x_coords))

    FSS = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.5],
        window_sizes=[10],
    )

    z = FSS(x, y).e2s.to_torch()[0]
    # this (==0.8) should be the fss for 50% exceedence
    reference_value = 1 - 0.5**2 / (0.5**2 + 1.0**2)
    # allow some tolerance due to finite-size windows
    assert (abs(z - reference_value) < 0.01 * reference_value).all()

    # test that FSS is 1 for comparison to self
    z = FSS(x, x).e2s.to_torch()[0]
    assert (z == 1.0).all()

    # test that FSS is 0 if x and y are always on different side of threshold
    x = from_torch(torch.zeros_like(x_t), copy.deepcopy(x_coords))
    z = FSS(x, y).e2s.to_torch()[0]
    assert (z == 0.0).all()


@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_spatial_dims(ensemble_dimension: str | None, device: str) -> None:
    # test that we can use spatial dimensions that are not the last 2 dims
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
            "variable": np.array(["t2m", "tcwv"]),
        }
    )
    x = from_torch(torch.zeros((10, 1, 128, 128, 2), device=device), x_coords)

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    FSS = fss(
        reduction_dimensions=["lat", "lon", "time"],
        thresholds=[0.25, 0.75],
        window_sizes=[3, 5],
        spatial_dimensions=("lat", "lon"),
        ensemble_dimension=ensemble_dimension,
    )

    z = FSS(x, y)
    expected_dims = ("variable", "threshold", "window_size")
    if ensemble_dimension is None:
        expected_dims = ("ensemble",) + expected_dims
    assert z.dims == expected_dims
    _check_output(FSS, x, z)
