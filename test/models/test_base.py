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

import numpy as np
import pytest

from earth2studio.grids import LatLonGrid, Region
from earth2studio.models.base import DomainAdaptable

PARENT = LatLonGrid(np.arange(60.0, 19.0, -1.0), np.arange(230.0, 301.0, 1.0))


def _covering(centers: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mask of the smallest run of cell centers bracketing ``[low, high]``."""
    if low < centers.min() or high > centers.max():
        raise ValueError("region lies outside the supported footprint")
    return (centers >= centers[centers <= low].max()) & (
        centers <= centers[centers >= high].min()
    )


class CropModel:
    """Minimal crop model covering box regions with cells of a fixed parent grid."""

    def __init__(self, grid: LatLonGrid = PARENT) -> None:
        self.grid = grid

    def with_region(self, region: Region, /) -> "CropModel":
        if region.kind != "box":
            raise ValueError(f"{region.kind} regions are unsupported")
        west, south, east, north = region.params
        lat = _covering(PARENT.latitude, south, north)
        lon = _covering(PARENT.longitude, west % 360, east % 360)
        return CropModel(LatLonGrid(PARENT.latitude[lat], PARENT.longitude[lon]))


class FixedModel:
    grid = PARENT


def test_domain_adaptable_protocol():
    assert isinstance(CropModel(), DomainAdaptable)
    assert not isinstance(FixedModel(), DomainAdaptable)


def test_domain_adaptable_view():
    model = CropModel()
    view = model.with_region(Region.box(-120.5, 30.5, -100.5, 40.5))
    assert model.grid is PARENT
    # The view covers the region with whole cells
    assert view.grid.latitude.min() <= 30.5 and view.grid.latitude.max() >= 40.5
    assert view.grid.longitude.min() <= 239.5 and view.grid.longitude.max() >= 259.5
    assert view.grid.shape == (12, 22)
    # A view reaches the full supported footprint
    assert view.with_region(Region.box(-130, 20, -60, 60)).grid.shape == PARENT.shape


def test_domain_adaptable_rejects():
    model = CropModel()
    # Partial overlap would leave part of the region uncovered
    with pytest.raises(ValueError, match="footprint"):
        model.with_region(Region.box(-125, 10, -115, 40))
    with pytest.raises(ValueError, match="unsupported"):
        model.with_region(Region.rectangle(35, -110, 1e5))
