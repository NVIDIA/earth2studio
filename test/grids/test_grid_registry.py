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

GRID = ProjectedGrid([0, 3000], [0, 3000, 6000], "EPSG:3857")


def test_grid_registry_builtins():
    assert {"latlon-0.25deg", "hrrr-conus-3km", "healpix-l6-nested"} <= set(
        list_grids()
    )
    assert resolve_grid("fcn1").shape == (720, 1440)
    assert isinstance(resolve_grid("hrrr"), GridDefinition)


def test_grid_registry_registration():
    register_grid("test-registry-grid", GRID, aliases=("test-registry-alias",))
    register_grid("test-registry-grid", GRID, aliases=("test-registry-alias",))
    assert resolve_grid("test-registry-alias") is GRID
    for name, definition, message in (
        ("test-registry-alias", GRID, "conflicts"),
        ("EPSG:4326", GRID, "valid CRS"),
        ("invalid-grid", object(), "implement GridDefinition"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            register_grid(name, definition)  # type: ignore[arg-type]


def test_grid_registry_inference_validation():
    register_grid("test-explicit-grid", GRID)
    array = xr.DataArray(
        np.ones(GRID.shape),
        dims=GRID.dims,
        coords={"y": GRID.y, "x": GRID.x},
        attrs={E2S_CRS: "EPSG:3857", E2S_GRID_ID: "test-explicit-grid"},
    )
    assert infer_grid(array) is GRID
    assert infer_grid(array.expand_dims(time=[0])) is GRID
    assert infer_grid(array.isel(x=slice(2))).shape == (2, 2)


def test_grid_registry_errors():
    for operation, message in (
        (lambda: resolve_grid("missing"), "Unknown"),
        (lambda: resolve_grid("EPSG:4326"), "does not define"),
        (lambda: infer_grid(xr.DataArray(np.ones(1), dims="x")), "Cannot infer"),
    ):
        with pytest.raises(ValueError, match=message):
            operation()
