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

import json
from inspect import Parameter, signature

import pytest
import xarray as xr

from earth2studio.grids import GridDefinition


@pytest.fixture
def check_grid_contract():
    """Return the shared grid contract assertion."""

    def check(grid: GridDefinition, dims, shape, topology):
        assert isinstance(grid, GridDefinition)
        assert (grid.dims, grid.shape, grid.topology) == (dims, shape, topology)
        selection = signature(grid.subset_indexers).parameters
        assert {"coordinates", "bounds", "bounds_crs"} <= set(selection)
        assert all(
            parameter.kind != Parameter.VAR_KEYWORD for parameter in selection.values()
        )

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
