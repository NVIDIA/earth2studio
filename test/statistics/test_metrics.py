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

from earth2studio.statistics import lat_weight, rmse, spread_skill_ratio

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
def test_rmse(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

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
    y = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    RMSE = rmse(reduction_dimensions, weights=weights)

    z, c = RMSE(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]

    ## Test broadcasting
    y = torch.randn((1, 1, 2, 361, 720), device=device)
    y_coords["ensemble"] = np.arange(1)

    z, c = RMSE(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_mean(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y = torch.randn((10, 10), device=device)
    y_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})

    RMSE = rmse(["ensemble"], batch_update=True)
    assert str(RMSE) == "ensemble_rmse"
    for inds in range(0, 100, 10):
        x = big_x[inds : inds + 10]
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]

        z, c = RMSE(x, coords, y, y_coords)
        assert torch.allclose(
            z,
            torch.sqrt(torch.mean((big_x[: inds + 10] - y[None]) ** 2, dim=0)),
            atol=1e-3,
        )
        assert c == y_coords


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["time", "lat", "lon"], lat_weights.unsqueeze(1).repeat(2, 1, 720)),
        (["lat"], lat_weights),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_spread_skill(
    reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    x = 3.0 + 2.0 * torch.randn((100, 2, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "time": np.array(
                [np.datetime64("1993-04-05T00:00"), np.datetime64("1993-04-06T00:00")]
            ),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y = 3.0 + 2.0 * torch.randn((2, 2, 361, 720), device=device)

    ensemble_dimension = "ensemble"
    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    SSR = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, reduction_weights=weights
    )

    z, c = SSR(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in ["ensemble"] + reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]
    assert torch.allclose(z, torch.ones_like(z), rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ensemble_batch_spread_skill(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y = torch.randn((10, 10), device=device)
    y_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})

    ensemble_dimension = "ensemble"
    reduction_dimensions = ["lat", "lon"]
    SSR1 = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, ensemble_batch_update=True
    )
    SSR2 = spread_skill_ratio(ensemble_dimension, reduction_dimensions)

    assert str(SSR1) == "ensemble_lat_lon_spread_skill"
    for inds in range(0, 100, 10):
        x1 = big_x[inds : inds + 10]
        coords1 = big_coords.copy()
        coords1["ensemble"] = coords1["ensemble"][inds : inds + 10]

        ss1, c1 = SSR1(x1, coords1, y, y_coords)

        x2 = big_x[: inds + 10]
        coords2 = big_coords.copy()
        coords2["ensemble"] = coords2["ensemble"][: inds + 10]
        ss2, c2 = SSR2(x2, coords2, y, y_coords)

        assert torch.allclose(
            ss1,
            ss2,
            atol=1e-3,
        )

        assert c1 == c2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_time_batch_spread_skill(device) -> None:

    big_x = torch.randn((10, 100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y = torch.randn((100, 10, 10), device=device)
    y_coords = OrderedDict(
        {"time": big_coords["time"], "lat": big_coords["lat"], "lon": big_coords["lon"]}
    )

    ensemble_dimension = "ensemble"
    reduction_dimensions = ["time", "lat", "lon"]
    SSR1 = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, reduction_batch_update=True
    )
    SSR2 = spread_skill_ratio(ensemble_dimension, reduction_dimensions)

    assert str(SSR1) == "ensemble_time_lat_lon_spread_skill"
    for inds in range(0, 100, 10):
        x1 = big_x[:, inds : inds + 10]
        coords1 = big_coords.copy()
        coords1["time"] = coords1["time"][inds : inds + 10]

        y1 = y[inds : inds + 10]
        y_coords1 = y_coords.copy()
        y_coords1["time"] = y_coords1["time"][inds : inds + 10]

        ss1, c1 = SSR1(x1, coords1, y1, y_coords1)

        x2 = big_x[:, : inds + 10]
        coords2 = big_coords.copy()
        coords2["time"] = coords2["time"][: inds + 10]

        y2 = y[: inds + 10]
        y_coords2 = y_coords.copy()
        y_coords2["time"] = y_coords2["time"][: inds + 10]

        ss2, c2 = SSR2(x2, coords2, y2, y_coords2)

        assert torch.allclose(
            ss1,
            ss2,
            atol=1e-3,
        )

        assert c1 == c2
