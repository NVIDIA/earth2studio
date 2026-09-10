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
import xarray as xr

from earth2studio.grids import CurvilinearGrid, infer_grid

LAT = np.array([[40.0, 40.1], [41.0, 41.1]])
LON = np.array([[-100.0, -99.0], [-100.1, -99.1]])
GRID = CurvilinearGrid(LAT, LON)


def test_grid_contract(check_grid_contract):
    check_grid_contract(GRID, ("y", "x"), (2, 2), "curvilinear")


def test_grid_coordinates():
    selected = GRID.coords({"y": np.array([1]), "x": np.array([0])})
    np.testing.assert_array_equal(selected["lat"], [[41.0]])
    assert GRID.crs is None
    assert GRID.cell_bounds(GRID.coords(only_index=True)) is None


def test_grid_selection():
    assert GRID.subset_indexers(
        GRID.coords(only_index=True), bounds=(-99.2, 39.5, -98.5, 41.5)
    ) == {"y": slice(0, 2), "x": slice(1, 2)}


def test_grid_validation():
    with pytest.raises(ValueError, match="read-only"):
        GRID.x[0] = 1
    for args, message in (
        (([[0]], [[0, 1]]), "shapes"),
        ((LAT, LON, [0], [0, 1]), "sizes"),
    ):
        with pytest.raises(ValueError, match=message):
            CurvilinearGrid(*args)


def test_grid_inference():
    dataset = xr.Dataset(coords={"lat": (GRID.dims, LAT), "lon": (GRID.dims, LON)})
    assert isinstance(infer_grid(dataset), CurvilinearGrid)
