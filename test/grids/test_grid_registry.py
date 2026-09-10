# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    GridDefinition,
    ProjectedGrid,
    infer_grid,
    list_grids,
    register_grid,
    resolve_grid,
)


def test_builtin_grid_registry():
    assert {
        "latlon-0.25deg",
        "latlon-0.25deg-south-pole-excluded",
        "hrrr-conus-3km",
        "healpix-l6-nested",
    } <= set(list_grids())
    assert resolve_grid("fcn1").shape == (720, 1440)
    assert isinstance(resolve_grid("hrrr"), GridDefinition)


def test_grid_registration_is_idempotent_and_rejects_conflicts():
    grid = ProjectedGrid([0.0, 3000.0], [0.0, 3000.0, 6000.0], "EPSG:3857")
    register_grid("test-registry-grid", grid, aliases=("test-registry-alias",))
    register_grid("test-registry-grid", grid, aliases=("test-registry-alias",))
    assert resolve_grid("test-registry-alias") is grid

    with pytest.raises(ValueError, match="conflicts"):
        register_grid("test-registry-alias", grid)
    with pytest.raises(ValueError, match="valid CRS"):
        register_grid("EPSG:4326", grid)
    with pytest.raises(TypeError, match="implement GridDefinition"):
        register_grid("invalid-grid", object())  # type: ignore[arg-type]


def test_explicit_grid_id_is_validated_before_use():
    grid = ProjectedGrid([0.0, 3000.0], [0.0, 3000.0, 6000.0], "EPSG:3857")
    register_grid("test-explicit-grid", grid)
    array = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": grid.y, "x": grid.x},
        attrs={E2S_CRS: "EPSG:3857", E2S_GRID_ID: "test-explicit-grid"},
    )
    assert infer_grid(array) is grid
    assert infer_grid(array.expand_dims(time=[0])) is grid
    inferred_subset = infer_grid(array.isel(x=slice(2)))
    assert isinstance(inferred_subset, ProjectedGrid)
    assert inferred_subset.shape == (2, 2)


def test_registry_errors_are_explicit():
    for operation, message in (
        (lambda: resolve_grid("missing"), "Unknown"),
        (lambda: resolve_grid("EPSG:4326"), "does not define"),
        (lambda: infer_grid(xr.DataArray(np.ones(1), dims="x")), "Cannot infer"),
    ):
        with pytest.raises(ValueError, match=message):
            operation()
