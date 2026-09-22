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

from earth2studio.grids import E2S_CRS, ProjectedGrid, infer_grid, resolve_grid

GRID = ProjectedGrid([0.0, 3000.0], [0.0, 3000.0, 6000.0], "EPSG:3857")


def test_grid_contract(check_grid_contract):
    check_grid_contract(GRID, ("y", "x"), (2, 3), "projected")


def test_grid_coordinates():
    coordinates = GRID.coords()
    np.testing.assert_allclose(coordinates["lat"][0, 0], 0.0)
    np.testing.assert_allclose(coordinates["lon"][0, 0], 0.0)
    assert GRID.crs.to_epsg() == 3857
    assert GRID.cell_bounds(GRID.coords(only_index=True)) is None

    hrrr = resolve_grid("hrrr")
    indexes = hrrr.coords(only_index=True)
    subset = {dimension: np.asarray(indexes[dimension][:2]) for dimension in hrrr.dims}
    assert hrrr.coords(subset)["lat"].shape == (2, 2)


def test_grid_selection():
    assert GRID.subset_indexers(
        GRID.coords(), bounds=(-1, -1, 1, 1), bounds_crs="EPSG:3857"
    ) == {"y": slice(0, 1), "x": slice(0, 1)}


def test_grid_validation():
    with pytest.raises(ValueError, match="read-only"):
        GRID.x[0] = 0


def test_grid_inference():
    coordinates = dict(GRID.coords())
    coordinates.update(
        lat=(GRID.dims, np.ones(GRID.shape)),
        lon=(GRID.dims, np.ones(GRID.shape)),
    )
    array = xr.DataArray(
        np.ones(GRID.shape),
        dims=GRID.dims,
        coords=coordinates,
        attrs={E2S_CRS: "EPSG:3857"},
    )
    assert isinstance(infer_grid(array), ProjectedGrid)
