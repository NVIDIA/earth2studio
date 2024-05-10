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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import lat_weight, moments

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 361)))


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["ensemble"], None),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
        (["lat"], lat_weights),
        (["ensemble", "lat"], lat_weights.repeat(10, 1)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mean(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    mean = moments.mean(reduction_dimensions, weights=weights)

    y, c = mean(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_mean(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(10), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x = torch.randn((10, 361), device=device)

    reduction_dimensions, weights = ["lat"], lat_weights
    mean = moments.mean(reduction_dimensions, weights=weights)

    assert str(mean) == "lat_mean"
    y, c = mean(x, coords)

    # Compute numpy weighted average
    y_np = np.average(x.cpu().numpy(), axis=1, weights=lat_weights.cpu().numpy())

    assert torch.allclose(y, torch.as_tensor(y_np, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_mean(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 1),
            "lon": np.linspace(0.0, 360.0, 1, endpoint=False),
        }
    )

    reduced_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})
    mean = moments.mean(["ensemble"], batch_update=True)
    for inds in range(0, 100, 10):
        x = big_x[inds : inds + 10]
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]

        y, c = mean(x, coords)
        assert torch.allclose(y, torch.mean(big_x[: inds + 10], dim=0), atol=1e-3)
        assert c == reduced_coords


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["ensemble"], None),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
        (["lat"], lat_weights),
        (["ensemble", "lat"], lat_weights.repeat(10, 1)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_var(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    var = moments.variance(reduction_dimensions, weights=weights)

    y, c = var(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]
    assert torch.all(y >= 0.0)


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["ensemble"], None),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
        (["lat"], lat_weights),
        (["ensemble", "lat"], lat_weights.repeat(10, 1)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_std(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    std = moments.std(reduction_dimensions, weights=weights)

    y, c = std(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]
    assert torch.all(y >= 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_var(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(1), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x = torch.randn((1, 361), device=device)

    reduction_dimensions, weights = ["lat"], lat_weights
    var = moments.variance(reduction_dimensions, weights=weights)

    y, c = var(x, coords)

    assert str(var) == "lat_variance"

    # Compute numpy weighted average
    y_np = np.cov(x.cpu().numpy(), aweights=lat_weights.cpu().numpy())

    assert torch.allclose(y, torch.as_tensor(y_np, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_var_std(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    reduced_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})

    std = moments.std(["ensemble"], batch_update=True)
    var = moments.variance(["ensemble"], batch_update=True)

    assert str(std) == "ensemble_std"
    assert str(var) == "ensemble_variance"

    for inds in range(0, 100, 10):
        x = big_x[inds : inds + 10]
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]

        y, c = std(x, coords)
        assert torch.allclose(y, torch.std(big_x[: inds + 10], dim=0))
        assert c == reduced_coords

        y, c = var(x, coords)
        assert torch.allclose(y, torch.var(big_x[: inds + 10], dim=0))
        assert c == reduced_coords


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_moments_failures(device) -> None:
    # Test weights not the same shape as reduction dimensions
    reduction_dimensions = ["lat", "lon"]
    weights = torch.as_tensor([10], device=device)

    with pytest.raises(ValueError):
        moments.mean(reduction_dimensions, weights=weights)

    with pytest.raises(ValueError):
        moments.variance(reduction_dimensions, weights=weights)

    with pytest.raises(ValueError):
        moments.std(reduction_dimensions, weights=weights)

    x = torch.randn((10,), device=device)
    coords = OrderedDict({"lat": np.arange(10)})
    with pytest.raises(ValueError):
        m = moments.mean(reduction_dimensions)

        m(x, coords)

    with pytest.raises(ValueError):
        var = moments.variance(reduction_dimensions)

        var(x, coords)

    with pytest.raises(ValueError):
        std = moments.std(reduction_dimensions)

        std(x, coords)
