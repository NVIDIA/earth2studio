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

from earth2studio.grids import GridDefinition, LatLonGrid, Region
from earth2studio.models.domain import DomainAdaptable

PARENT = LatLonGrid(np.arange(60.0, 19.0, -1.0), np.arange(230.0, 301.0, 1.0))


class CropModel:
    """Minimal crop model selecting whole cells from a fixed parent grid."""

    def __init__(self, grid: GridDefinition = PARENT) -> None:
        self.grid = grid

    def domain_for(self, region: Region, /) -> GridDefinition:
        if region.kind != "box":
            raise ValueError(f"{region.kind} regions are unsupported")
        indexers = PARENT.subset_indexers(
            PARENT.coords(only_index=True), bounds=region.params
        )
        return LatLonGrid(
            PARENT.latitude[indexers["lat"]], PARENT.longitude[indexers["lon"]]
        )

    def with_domain(self, grid: GridDefinition, /) -> "CropModel":
        if not isinstance(grid, LatLonGrid) or not (
            np.isin(grid.latitude, PARENT.latitude).all()
            and np.isin(grid.longitude, PARENT.longitude).all()
        ):
            raise ValueError("grid must be a crop of the parent grid")
        return CropModel(grid)


class FixedModel:
    grid = PARENT


def test_domain_adaptable_protocol():
    assert isinstance(CropModel(), DomainAdaptable)
    assert not isinstance(FixedModel(), DomainAdaptable)


def test_domain_adaptable_view():
    model = CropModel()
    grid = model.domain_for(Region.box(-120, 30, -100, 40))
    view = model.with_domain(grid)
    assert view.grid is grid and model.grid is PARENT
    # A view reaches the full supported footprint
    assert view.with_domain(PARENT).grid is PARENT
    with pytest.raises(ValueError, match="unsupported"):
        view.domain_for(Region.around(35, -110, 1e5))
    with pytest.raises(ValueError, match="crop"):
        view.with_domain(LatLonGrid(np.array([35.5]), np.array([250.0])))
