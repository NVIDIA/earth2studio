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
from earth2studio.utils.coords import coord_array

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


def test_coordinate_cache(monkeypatch):
    from earth2studio.grids.projected import Transformer

    grid = ProjectedGrid(GRID.y, GRID.x, GRID.crs)
    original = Transformer.from_crs
    calls = []

    def transform(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(Transformer, "from_crs", transform)
    grid.coords(only_index=True)
    assert not calls
    first = coord_array(("y", "x"), grid=grid)
    second = coord_array(("y", "x"), grid=grid)
    assert len(calls) == 1
    first_grid = grid.coords()
    second_grid = grid.coords()
    for name in ("lat", "lon"):
        assert np.shares_memory(first_grid[name].data, second_grid[name].data)
        with pytest.raises(ValueError, match="read-only"):
            first_grid[name].data[0, 0] = 99
        first_grid[name].attrs["source"] = "changed"
        assert "source" not in second_grid[name].attrs
        first[name].attrs["source"] = "changed"
        assert "source" not in second[name].attrs
        first.coords[name] = (("y", "x"), np.zeros(grid.shape))
        np.testing.assert_array_equal(grid.coords()[name], second[name])
    assert len(calls) == 1


def test_custom_indexes_do_not_populate_full_grid_cache(monkeypatch):
    from earth2studio.grids.projected import Transformer

    grid = ProjectedGrid(GRID.y, GRID.x, GRID.crs)
    original = Transformer.from_crs
    calls = []

    def transform(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(Transformer, "from_crs", transform)
    indexes = {"y": grid.y[:1], "x": grid.x[1:]}
    subset = grid.coords(indexes)
    assert subset["lat"].shape == (1, 2)
    assert len(calls) == 1
    full = grid.coords()
    assert len(calls) == 2
    grid.coords()
    assert len(calls) == 2
    for name in ("lat", "lon"):
        np.testing.assert_allclose(subset[name], full[name][:1, 1:])


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
