# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import LatLonGrid, infer_grid


def test_latlon_grid_contract(check_grid: Callable[..., None]):
    grid = LatLonGrid([40.0, 39.0], [250.0, 251.0, 252.0])
    check_grid(grid, ("lat", "lon"), (2, 3), "rectilinear")
    assert grid.crs.to_epsg() == 4326
    assert grid.attrs["crs"] == "EPSG:4326"


def test_latlon_grid_selection_and_dateline():
    grid = LatLonGrid([40.0, 39.0], [250.0, 251.0, 252.0])
    assert grid.subset_indexers(
        grid.coords(only_index=True), bounds=(-110, 38, -90, 41)
    ) == {"lat": slice(0, 2), "lon": slice(0, 3)}

    dateline = LatLonGrid([0], [300, 350, 0, 10, 20])
    assert dateline.subset_indexers(
        dateline.coords(only_index=True), bounds=(350, -1, 10, 1)
    ) == {"lat": slice(0, 1), "lon": slice(1, 4)}


def test_latlon_grid_identity_and_validation():
    grid = LatLonGrid([40.0, 39.0], [250.0, 251.0])
    assert grid.cell_bounds(grid.coords(only_index=True)) is None
    assert (
        grid.fingerprint()
        != LatLonGrid(grid.latitude, grid.longitude, "OGC:CRS84").fingerprint()
    )
    with pytest.raises(ValueError, match="read-only"):
        grid.latitude[0] = 0
    with pytest.raises(ValueError, match="nonempty"):
        LatLonGrid([], [0])


def test_infer_latlon_grid():
    array = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [40.0, 39.0], "lon": [250.0, 251.0, 252.0]},
    )
    assert isinstance(infer_grid(array), LatLonGrid)
