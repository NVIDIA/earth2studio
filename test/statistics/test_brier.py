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


@pytest.mark.parametrize(
    "ensemble_dimension, ensemble_axis, batch_update",
    [
        ("ensemble", 0, False),
        ("member", 0, False),
        ("realization", 1, False),
        ("member", 2, True),
    ],
)
def test_bs_configured_ensemble_dimension(
    ensemble_dimension: str, ensemble_axis: int, batch_update: bool
) -> None:
    values = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    )
    observations = np.array([[0.0, 2.0], [4.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    thresholds = np.array([1.0, 4.0, 6.0], dtype=np.float32)
    coordinates = {
        ensemble_dimension: np.arange(3),
        "time": np.arange(3),
        "variable": np.array(["t2m", "tcwv"]),
    }
    dimensions = ["time", "variable"]
    dimensions.insert(ensemble_axis, ensemble_dimension)
    x_coords = OrderedDict((dim, coordinates[dim]) for dim in dimensions)
    y_coords = OrderedDict((dim, coordinates[dim]) for dim in ["time", "variable"])
    original_x_coords = copy.deepcopy(x_coords)
    original_y_coords = copy.deepcopy(y_coords)
    x = torch.from_numpy(np.moveaxis(values, 0, ensemble_axis))
    y = torch.from_numpy(observations)
    score = brier_score(["time"], thresholds, ensemble_dimension, batch_update)

    if batch_update:
        time_axis = dimensions.index("time")
        for start, stop in [(0, 1), (1, 3)]:
            xc, yc = copy.deepcopy(x_coords), copy.deepcopy(y_coords)
            xc["time"] = yc["time"] = np.arange(start, stop)
            actual, coords = score(
                x.narrow(time_axis, start, stop - start), xc, y[start:stop], yc
            )
    else:
        actual, coords = score(x, x_coords, y, y_coords)

    probabilities = (values[..., None] >= thresholds).mean(axis=0)
    observed_events = observations[..., None] >= thresholds
    expected = ((probabilities - observed_events) ** 2).mean(axis=0)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)
    assert list(coords) == ["variable", "threshold"]
    for dim, coordinate in score.output_coords(x_coords).items():
        np.testing.assert_array_equal(coords[dim], coordinate)
    for original, current in [
        (original_x_coords, x_coords),
        (original_y_coords, y_coords),
    ]:
        assert list(original) == list(current)
        for dim in original:
            np.testing.assert_array_equal(original[dim], current[dim])
