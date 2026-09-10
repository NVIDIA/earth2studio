# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import E2S_CRS, ProjectedGrid, infer_grid, resolve_grid


def test_projected_grid_contract(check_grid: Callable[..., None]):
    grid = ProjectedGrid([0.0, 3000.0], [0.0, 3000.0, 6000.0], "EPSG:3857")
    check_grid(grid, ("y", "x"), (2, 3), "projected")
    assert grid.crs.to_epsg() == 3857


def test_projected_grid_coordinates_and_selection():
    grid = ProjectedGrid([0.0, 3000.0], [0.0, 3000.0], "EPSG:3857")
    coordinates = grid.coords()
    np.testing.assert_allclose(coordinates["lat"][0, 0], 0.0)
    np.testing.assert_allclose(coordinates["lon"][0, 0], 0.0)
    assert grid.subset_indexers(
        coordinates, bounds=(-1, -1, 1, 1), bounds_crs="EPSG:3857"
    ) == {"y": slice(0, 1), "x": slice(0, 1)}
    assert grid.cell_bounds(grid.coords(only_index=True)) is None


def test_projected_grid_builtin_and_immutability():
    hrrr = resolve_grid("hrrr")
    assert hrrr.dims == ("y", "x")
    assert hrrr.shape == (1059, 1799)
    assert hrrr.crs.coordinate_operation is not None
    indexes = hrrr.coords(only_index=True)
    geographic = hrrr.coords(
        {dimension: np.asarray(indexes[dimension][:2]) for dimension in hrrr.dims}
    )
    assert geographic["lat"].shape == (2, 2)
    with pytest.raises(ValueError, match="read-only"):
        hrrr.x[0] = 0


def test_infer_projected_grid_before_auxiliary_coordinates():
    array = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={
            "y": np.arange(2),
            "x": np.arange(3),
            "lat": (("y", "x"), np.ones((2, 3))),
            "lon": (("y", "x"), np.ones((2, 3))),
        },
        attrs={E2S_CRS: "EPSG:3857"},
    )
    assert isinstance(infer_grid(array), ProjectedGrid)
