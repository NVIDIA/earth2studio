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

# %%
"""
Earth2Studio Grids
==================

Define, register, infer, and select spatial grid geometry.

A grid describes spatial geometry without containing weather data. In this tutorial
you will learn:

- What the structural grid interface requires
- How to resolve and register grid definitions
- How to infer grids from Xarray coordinates
- How grid definitions support selection and future regridding
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Define a Grid Object
# --------------------
# A grid object structurally implements
# [`GridDefinition`][earth2studio.utils.grid.GridDefinition]. It does not inherit from
# a grid base class. The protocol defines dimension order, shape, topology, coordinate
# generation, metadata, selection, and stable identity.

# %%
# .. literalinclude:: ../../earth2studio/utils/grid.py
#    :language: python
#    :start-after: # sphinx - grid protocol start
#    :end-before: # sphinx - grid protocol end

# %%
import numpy as np
import xarray as xr

import earth2studio.utils.grid as e2s

# %%
# Resolve a Known Grid
# --------------------
# The registry maps stable names and aliases to complete grid definitions.

# %%
hrrr = e2s.resolve_grid("hrrr")
crs = hrrr.crs
grid_summary = {
    "type": type(hrrr).__name__,
    "dims": hrrr.dims,
    "shape": hrrr.shape,
    "topology": hrrr.topology,
    "crs": crs.name if crs is not None else None,
}
print(grid_summary)

# %%
# Register a Projected Grid
# -------------------------
# A projected grid requires ordered ``y`` and ``x`` coordinates plus any CRS accepted
# by PyProj. Registration gives the complete definition a stable identity.

# %%
regional = e2s.ProjectedGrid(
    y=np.arange(3) * 3_000.0,
    x=np.arange(4) * 3_000.0,
    coordinate_reference_system=(
        "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=-97 "
        "+datum=WGS84 +units=m +type=crs"
    ),
)
e2s.register_grid("tutorial-lcc", regional)

indexes = regional.index_coordinates()
geographic = regional.geographic_coordinates(
    {dimension: np.asarray(indexes[dimension]) for dimension in regional.dims}
)
print(regional.shape, geographic["lat"].shape)

# %%
# Populate Xarray Coordinates
# ---------------------------
# Dimension coordinates preserve the grid's spatial order. Latitude and longitude
# are auxiliary coordinates, so projected ``y`` and ``x`` remain the array dimensions.

# %%
geometry = xr.Dataset(coords=indexes).assign_coords(geographic)
print(geometry)

# Field data can later reuse these coordinates with
# ``xr.DataArray(data, dims=regional.dims, coords=geometry.coords)``.

# %%
# Infer Xarray Grids
# ------------------
# Registration is optional. Independent one-dimensional latitude and longitude
# coordinates define a rectilinear grid.

# %%
latlon = xr.DataArray(
    np.zeros((2, 3)),
    dims=("lat", "lon"),
    coords={"lat": [40.0, 39.0], "lon": [250.0, 251.0, 252.0]},
)
print(type(e2s.infer_grid(latlon)).__name__)

# %%
# Arbitrary locations use ``x`` as the ordered index with auxiliary latitude and
# longitude coordinates.

# %%
points = xr.Dataset(
    coords={
        "x": np.arange(3),
        "lat": ("x", [35.2, 40.8, 51.0]),
        "lon": ("x", [-97.4, -74.0, 0.1]),
    }
)
point_grid = e2s.infer_grid(points)
print(point_grid.dims, point_grid.topology, point_grid.index_coordinates())

# %%
# Select a Subdomain
# ------------------
# Geographic bounds are translated into normal Xarray indexers. Field data remains
# outside the grid implementation.

# %%
grid = e2s.infer_grid(latlon)
indexers = grid.subset_indexers(
    latlon.coords,
    bounds=(-110, 38, -90, 41),
)
subset = latlon.isel(indexers)
print(subset)

# %%
# Connect to a Regridder
# ----------------------
# A regridder can accept source and target ``GridDefinition`` objects. It can use
# topology, CRS, geographic centers, and optional cell bounds to select an engine and
# interpolation method. The fingerprint provides a stable key for cached weights.
#
# .. code-block:: python
#
#    source_grid = e2s.infer_grid(source_array)
#    target_grid = e2s.resolve_grid("hrrr")
#    plan = regridder.plan(source_grid, target_grid, method="linear")
#    result = plan(source_array)
#
# The regridder owns capability dispatch, weight construction, and application.
# Unsupported grid pairs or methods should raise before processing field data.
