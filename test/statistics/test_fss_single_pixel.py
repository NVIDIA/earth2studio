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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import fss
from earth2studio.utils.type import CoordSystem


def _fields(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, CoordSystem]:
    coords = OrderedDict(time=np.arange(2), lat=np.arange(4), lon=np.arange(5))
    x = (torch.arange(40).reshape(2, 4, 5) % 3).to(dtype)
    y = x.flip(-1)
    return x, y, coords


def _pointwise_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor([0.5, 1.5], dtype=x.dtype)
    forecast = (x[..., None] >= thresholds).to(x.dtype)
    observation = (y[..., None] >= thresholds).to(x.dtype)
    if x.ndim == y.ndim + 1:
        forecast = forecast.mean(dim=0)
    squared_error = ((forecast - observation) ** 2).mean(dim=(-3, -2))
    reference = (forecast**2 + observation**2).mean(dim=(-3, -2))
    return (1 - squared_error / reference).unsqueeze(-1)


@pytest.mark.parametrize(
    "dtype, ensemble, spatial_first",
    [
        (torch.float32, False, False),
        (torch.float64, False, False),
        (torch.float64, True, False),
        (torch.float32, False, True),
    ],
)
def test_fss_single_pixel_matches_pointwise_definition(
    dtype: torch.dtype, ensemble: bool, spatial_first: bool
) -> None:
    x, y, coords = _fields(dtype)
    x_coords = coords.copy()
    if ensemble:
        x = torch.stack([x, x.roll(1, -1), x.roll(1, -2)])
        x_coords = OrderedDict(member=np.arange(3)) | x_coords
    expected = _pointwise_score(x, y)
    if spatial_first:
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        coords = OrderedDict((key, coords[key]) for key in ["lat", "time", "lon"])
        x_coords = coords.copy()
    coords_before = {key: value.copy() for key, value in coords.items()}
    metric = fss(
        reduction_dimensions=["lat", "lon"],
        window_sizes=[1],
        thresholds=[0.5, 1.5],
        ensemble_dimension="member" if ensemble else None,
        spatial_dimensions=("lat", "lon") if spatial_first else None,
    )

    actual, output_coords = metric(x, x_coords, y, coords)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    assert list(output_coords) == ["time", "threshold", "window_size"]
    assert list(actual.shape) == [len(values) for values in output_coords.values()]
    for key in coords:
        np.testing.assert_array_equal(coords[key], coords_before[key])


def test_fss_single_pixel_streaming_matches_concatenated_score() -> None:
    x, y, coords = _fields(torch.float64)
    metric = fss(["time", "lat", "lon"], [1], [0.5, 1.5], batch_update=True)
    for index in range(2):
        batch_coords = coords.copy()
        batch_coords["time"] = coords["time"][index : index + 1]
        actual, output_coords = metric(
            x[index : index + 1], batch_coords, y[index : index + 1], batch_coords
        )
    thresholds = torch.tensor([0.5, 1.5], dtype=x.dtype)
    forecast = (x[..., None] >= thresholds).to(x.dtype)
    observation = (y[..., None] >= thresholds).to(x.dtype)
    numerator = ((forecast - observation) ** 2).mean(dim=(0, 1, 2))
    denominator = (forecast**2 + observation**2).mean(dim=(0, 1, 2))
    expected = (1 - numerator / denominator).unsqueeze(-1)
    torch.testing.assert_close(actual, expected)
    assert list(output_coords) == ["threshold", "window_size"]


@pytest.mark.parametrize("window_size", [1, 2, 3])
def test_fss_neighborhood_coordinates_match_convolution_extent(
    window_size: int,
) -> None:
    _, _, coords = _fields(torch.float32)
    metric = fss(["lat", "lon"], [window_size], [0.5])
    actual = metric._neighborhood_probability_coords(
        coords, metric.output_coords(coords), window_size
    )
    assert len(actual["lat"]) == len(coords["lat"]) - window_size + 1
    assert len(actual["lon"]) == len(coords["lon"]) - window_size + 1
    if window_size == 1:
        np.testing.assert_array_equal(actual["lat"], coords["lat"])
        np.testing.assert_array_equal(actual["lon"], coords["lon"])
    assert not np.shares_memory(actual["lat"], coords["lat"])
    assert not np.shares_memory(actual["lon"], coords["lon"])
