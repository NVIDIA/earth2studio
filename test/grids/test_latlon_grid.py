# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_grid_validation():
    with pytest.raises(ValueError, match="read-only"):
        GRID.latitude[0] = 0
    with pytest.raises(ValueError, match="nonempty"):
        LatLonGrid([], [0])


def test_grid_inference():
    array = xr.DataArray(np.ones(GRID.shape), dims=GRID.dims, coords=GRID.coords())
    assert isinstance(infer_grid(array), LatLonGrid)
