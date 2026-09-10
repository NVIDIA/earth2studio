# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import xarray as xr

from earth2studio.grids import GridDefinition


@pytest.fixture
def check_grid_contract():
    """Return the shared grid contract assertion."""

    def check(grid: GridDefinition, dims, shape, topology):
        assert isinstance(grid, GridDefinition)
        assert (grid.dims, grid.shape, grid.topology) == (dims, shape, topology)

        indexes, coordinates = grid.coords(only_index=True), grid.coords()
        assert tuple(indexes) == dims
        for dimension, size in zip(dims, shape, strict=True):
            assert indexes[dimension].dims == (dimension,)
            assert indexes[dimension].size == size
        latitude, longitude = xr.broadcast(coordinates["lat"], coordinates["lon"])
        assert latitude.dims == longitude.dims == dims
        assert latitude.shape == longitude.shape == shape

        attrs = grid.attrs
        assert (attrs["type"], attrs["dims"], attrs["shape"], attrs["topology"]) == (
            type(grid).__name__,
            list(dims),
            list(shape),
            topology,
        )
        json.dumps(attrs)
        attrs["shape"] = []
        assert grid.attrs["shape"] == list(shape)
        assert grid.fingerprint() == grid.fingerprint()

    return check
