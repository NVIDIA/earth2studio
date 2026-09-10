# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from typing import Any

import pytest
import xarray as xr

from earth2studio.grids import GridDefinition, GridTopology


@pytest.fixture
def check_grid() -> Callable[..., None]:
    """Return a common grid contract checker."""

    def check(
        grid: GridDefinition,
        dims: tuple[str, ...],
        shape: tuple[int, ...],
        topology: GridTopology,
    ) -> None:
        assert isinstance(grid, GridDefinition)
        assert grid.dims == dims
        assert grid.shape == shape
        assert grid.topology == topology

        indexes = grid.coords(only_index=True)
        coordinates = grid.coords()
        assert tuple(indexes) == dims
        for dimension, size in zip(dims, shape, strict=True):
            assert indexes[dimension].dims == (dimension,)
            assert indexes[dimension].size == size
        latitude, longitude = xr.broadcast(coordinates["lat"], coordinates["lon"])
        assert latitude.dims == longitude.dims == dims
        assert latitude.shape == longitude.shape == shape

        attrs: dict[str, Any] = grid.attrs
        assert attrs["type"] == type(grid).__name__
        assert attrs["dims"] == list(dims)
        assert attrs["shape"] == list(shape)
        assert attrs["topology"] == topology
        json.dumps(attrs)
        attrs["shape"] = []
        assert grid.attrs["shape"] == list(shape)
        assert grid.fingerprint()
        assert grid.fingerprint() == grid.fingerprint()

    return check
