# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# %%
"""
Earth2Studio Grids
==================

Define, register, infer, and select spatial grid geometry.

A grid describes geometry without containing weather data. This tutorial shows the
structural interface, registry, Xarray coordinates, HEALPix layouts, and selection.
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Define a Grid Object
# --------------------
# A grid object implements [`GridDefinition`][earth2studio.grids.GridDefinition]
# structurally. It does not inherit from a grid base class.

# %%
# .. literalinclude:: ../../earth2studio/grids/base.py
#    :language: python
#    :start-after: # sphinx - grid protocol start
#    :end-before: # sphinx - grid protocol end

# %%
import numpy as np
import xarray as xr

import earth2studio.grids as grids

# %%
# Resolve a Known Grid
# --------------------
# The registry maps stable names and aliases to complete definitions. Metadata is a
# serializable summary suitable for Xarray attributes and external tools.

# %%
hrrr = grids.resolve_grid("hrrr")
print(hrrr.to_metadata())

# %%
# Register a Projected Grid
# -------------------------
# A projected grid requires ordered ``y`` and ``x`` coordinates plus any CRS accepted
# by PyProj.

# %%
regional = grids.ProjectedGrid(
    y=np.arange(3) * 3_000.0,
    x=np.arange(4) * 3_000.0,
    coordinate_reference_system=(
        "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=-97 "
        "+datum=WGS84 +units=m +type=crs"
    ),
)
grids.register_grid("tutorial-lcc", regional)

indexes = regional.coords(only_index=True)
coordinates = regional.coords()
print(regional.shape, coordinates["lat"].shape)

# %%
# Populate Xarray Coordinates
# ---------------------------
# Dimension coordinates preserve spatial order. Latitude and longitude remain
# auxiliary coordinates, so projected ``y`` and ``x`` are the array dimensions.

# %%
geometry = xr.Dataset(coords=coordinates)
print(geometry)

# Field data can reuse these coordinates with
# ``xr.DataArray(data, dims=regional.dims, coords=geometry.coords)``.

# %%
# Represent HEALPix Layouts
# -------------------------
# Ordering and layout are separate. DLESyM uses Earth2Grid's north-origin,
# clockwise XY convention with explicit face, height, and width dimensions.

# %%
dlesym = grids.HEALPixGrid(
    level=2,
    ordering="xy",
    layout="face",
    xy_origin="north",
    xy_clockwise=True,
)
print(dlesym.to_metadata())
print(dlesym.coords(only_index=True))

# %%
# Infer Xarray Grids
# ------------------
# Independent one-dimensional latitude and longitude coordinates define a
# rectilinear grid.

# %%
latlon = xr.DataArray(
    np.zeros((2, 3)),
    dims=("lat", "lon"),
    coords={"lat": [40.0, 39.0], "lon": [250.0, 251.0, 252.0]},
)
print(type(grids.infer_grid(latlon)).__name__)

# %%
# Arbitrary locations use ``x`` as their ordered index.

# %%
points = xr.Dataset(
    coords={
        "x": np.arange(3),
        "lat": ("x", [35.2, 40.8, 51.0]),
        "lon": ("x", [-97.4, -74.0, 0.1]),
    }
)
point_grid = grids.infer_grid(points)
print(point_grid.dims, point_grid.topology, point_grid.coords(only_index=True))

# %%
# Select a Subdomain
# ------------------
# Geographic bounds become normal Xarray indexers. Field data remains outside the
# grid implementation.

# %%
grid = grids.infer_grid(latlon)
indexers = grid.subset_indexers(latlon.coords, bounds=(-110, 38, -90, 41))
print(latlon.isel(indexers))
