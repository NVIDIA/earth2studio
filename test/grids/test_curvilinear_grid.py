# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import CurvilinearGrid, infer_grid

LATITUDE = np.array([[40.0, 40.1], [41.0, 41.1]])
LONGITUDE = np.array([[-100.0, -99.0], [-100.1, -99.1]])


def test_curvilinear_grid_contract(check_grid: Callable[..., None]):
    grid = CurvilinearGrid(LATITUDE, LONGITUDE)
    check_grid(grid, ("y", "x"), (2, 2), "curvilinear")
    assert grid.crs is None


def test_curvilinear_grid_coordinates_and_selection():
    grid = CurvilinearGrid(LATITUDE, LONGITUDE)
    selected = grid.coords({"y": np.array([1]), "x": np.array([0])})
    np.testing.assert_array_equal(selected["lat"], [[41.0]])
    assert grid.subset_indexers(
        grid.coords(only_index=True), bounds=(-99.2, 39.5, -98.5, 41.5)
    ) == {"y": slice(0, 2), "x": slice(1, 2)}
    assert grid.cell_bounds(grid.coords(only_index=True)) is None


def test_curvilinear_grid_validation_and_immutability():
    grid = CurvilinearGrid(LATITUDE, LONGITUDE)
    with pytest.raises(ValueError, match="read-only"):
        grid.x[0] = 1
    with pytest.raises(ValueError, match="shapes must match"):
        CurvilinearGrid([[0]], [[0, 1]])
    with pytest.raises(ValueError, match="sizes must match"):
        CurvilinearGrid(LATITUDE, LONGITUDE, y=[0], x=[0, 1])


def test_infer_curvilinear_grid():
    dataset = xr.Dataset(
        coords={
            "lat": (("y", "x"), LATITUDE),
            "lon": (("y", "x"), LONGITUDE),
        }
    )
    assert isinstance(infer_grid(dataset), CurvilinearGrid)
