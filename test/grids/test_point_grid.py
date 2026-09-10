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

from earth2studio.grids import PointGrid, infer_grid

LAT, LON = np.array([35.2, 40.8, 51.0]), np.array([-97.4, -74.0, 0.1])
GRID = PointGrid(LAT, LON)


def test_grid_contract(check_grid_contract):
    check_grid_contract(GRID, ("x",), (3,), "points")


def test_grid_coordinates():
    selected = PointGrid(LAT, LON, [10, 11, 12]).coords({"x": np.array([11])})
    np.testing.assert_array_equal(selected["lat"], [40.8])
    assert GRID.crs is None
    assert GRID.cell_bounds(GRID.coords(only_index=True)) is None


def test_grid_selection():
    assert GRID.subset_indexers(
        GRID.coords(only_index=True), bounds=(-100, 30, -90, 40)
    ) == {"x": slice(0, 1)}


def test_grid_validation():
    with pytest.raises(ValueError, match="read-only"):
        GRID.x[0] = 1
    for args, message in ((([0], [0, 1]), "shapes"), ((LAT, LON, [0]), "size")):
        with pytest.raises(ValueError, match=message):
            PointGrid(*args)


def test_grid_inference():
    points = xr.Dataset(
        coords={"x": np.arange(3), "lat": ("x", LAT), "lon": ("x", LON)}
    )
    assert isinstance(infer_grid(points), PointGrid)
    with pytest.raises(ValueError, match="unsupported layout"):
        infer_grid(points.rename(x="sample"))
