# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import PointGrid, infer_grid

LATITUDE = np.array([35.2, 40.8, 51.0])
LONGITUDE = np.array([-97.4, -74.0, 0.1])


def test_point_grid_contract(check_grid: Callable[..., None]):
    check_grid(PointGrid(LATITUDE, LONGITUDE), ("x",), (3,), "points")


def test_point_grid_coordinates_and_selection():
    grid = PointGrid(LATITUDE, LONGITUDE, x=[10, 11, 12])
    selected = grid.coords({"x": np.array([11])})
    np.testing.assert_array_equal(selected["lat"], [40.8])
    assert grid.subset_indexers(
        grid.coords(only_index=True), bounds=(-100, 30, -90, 40)
    ) == {"x": slice(0, 1)}
    assert grid.cell_bounds(grid.coords(only_index=True)) is None


def test_point_grid_validation_and_immutability():
    grid = PointGrid(LATITUDE, LONGITUDE)
    with pytest.raises(ValueError, match="read-only"):
        grid.x[0] = 1
    with pytest.raises(ValueError, match="shapes must match"):
        PointGrid([0], [0, 1])
    with pytest.raises(ValueError, match="size must match"):
        PointGrid(LATITUDE, LONGITUDE, x=[0])


def test_infer_point_grid_and_reject_ambiguous_dimension():
    points = xr.Dataset(
        coords={
            "x": np.arange(2),
            "lat": ("x", [35.2, 40.8]),
            "lon": ("x", [-97.4, -74.0]),
        }
    )
    assert isinstance(infer_grid(points), PointGrid)

    ambiguous = points.rename(x="sample")
    with pytest.raises(ValueError, match="unsupported layout"):
        infer_grid(ambiguous)
