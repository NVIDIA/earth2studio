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

from earth2studio.grids import LatLonGrid, infer_grid

GRID = LatLonGrid([40.0, 39.0], [250.0, 251.0, 252.0])


def test_grid_contract(check_grid_contract):
    check_grid_contract(GRID, ("lat", "lon"), (2, 3), "rectilinear")


def test_grid_coordinates():
    assert GRID.crs.to_epsg() == 4326
    assert GRID.attrs["crs"] == "EPSG:4326"
    assert GRID.cell_bounds(GRID.coords(only_index=True)) is None
    assert (
        GRID.fingerprint()
        != LatLonGrid(GRID.latitude, GRID.longitude, "OGC:CRS84").fingerprint()
    )


def test_grid_selection():
    assert GRID.subset_indexers(
        GRID.coords(only_index=True), bounds=(-110, 38, -90, 41)
    ) == {"lat": slice(0, 2), "lon": slice(0, 3)}
    dateline = LatLonGrid([0], [300, 350, 0, 10, 20])
    assert dateline.subset_indexers(
        dateline.coords(only_index=True), bounds=(350, -1, 10, 1)
    ) == {"lat": slice(0, 1), "lon": slice(1, 4)}


@pytest.mark.parametrize("start", [0.0, -180.0])
@pytest.mark.parametrize(
    "bounds,expected",
    [
        ((0, -1, 180, 1), [*range(0, 181)]),
        ((-190, -1, -170, 1), [*range(170, 191)]),
        ((170, -1, 190, 1), [*range(170, 191)]),
        ((170, -1, -170, 1), [*range(170, 191)]),
        ((-180, -1, 180, 1), [*range(0, 360)]),
        ((10, -1, 10, 1), [10]),
    ],
)
def test_grid_selection_longitude_bounds(start, bounds, expected):
    """Longitude bounds are inclusive and wrap the same on either convention."""
    grid = LatLonGrid([0.0], np.arange(start, start + 360.0, 1.0))
    indexers = grid.subset_indexers(grid.coords(only_index=True), bounds=bounds)
    selected = np.mod(grid.longitude[indexers["lon"]], 360)
    np.testing.assert_array_equal(np.sort(selected), np.sort(np.mod(expected, 360)))


def test_grid_validation():
    with pytest.raises(ValueError, match="read-only"):
        GRID.latitude[0] = 0
    with pytest.raises(ValueError, match="nonempty"):
        LatLonGrid([], [0])


def test_grid_inference():
    array = xr.DataArray(np.ones(GRID.shape), dims=GRID.dims, coords=GRID.coords())
    assert isinstance(infer_grid(array), LatLonGrid)
