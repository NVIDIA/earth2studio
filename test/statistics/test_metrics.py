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

from earth2studio.statistics import lat_weight, rmse

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
            "lat": np.linspace(-90.0, 90.0, 1),
            "lon": np.linspace(0.0, 360.0, 1, endpoint=False),
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
