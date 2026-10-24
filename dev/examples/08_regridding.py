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
Regridding Fetched Data
=======================

Map source fields onto a model's grid with regridders, regridded sources and
composite sources.

This CPU-only tutorial uses synthetic sources and needs no downloads. See
``dev/spec/REGRID_SPEC.md`` and ``dev/spec/FETCH_DATA_SPEC.md`` for the contracts.
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
from collections.abc import Callable
from datetime import datetime
from typing import cast

import numpy as np
import xarray as xr

from earth2studio.data import CompositeSource, RegriddedSource, fetch_data
from earth2studio.grids import (
    BilinearRegridder,
    GridDefinition,
    LatLonGrid,
    NearestRegridder,
    ProjectedGrid,
    infer_grid,
    list_regridders,
    resolve_grid,
    resolve_regridder,
)
from earth2studio.utils import coord_array
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import TimeArray, VariableArray

# %%
# Synthetic Sources
# -----------------
# Each source returns hourly fields on its native grid. The global source mimics a
# coarse 1 degree analysis; the regional source is a finer lat-lon radar mosaic.

# %%
GLOBAL = resolve_grid("latlon-1deg")
RADAR = LatLonGrid(np.arange(45.0, 30.0, -0.25), np.arange(255.0, 275.0, 0.25))


class Synthetic:
    """Hourly fields ``hours + values(lat, lon)`` on a native grid."""

    time_step = np.timedelta64(1, "h")

    def __init__(
        self,
        grid: GridDefinition,
        values: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.grid, self.values = grid, values

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        time, variable = to_time_array(time), np.atleast_1d(variable)
        coords = self.grid.coords()
        lat, lon = xr.broadcast(coords["lat"], coords["lon"])
        hours = (time - time.min()) / np.timedelta64(1, "h")
        field = self.values(lat.values, lon.values)
        data = hours[:, None, None, None] + field[None, None] * np.ones(
            (1, len(variable), 1, 1)
        )
        return xr.DataArray(
            data,
            dims=("time", "variable", *self.grid.dims),
            coords={"time": time, "variable": variable, **coords},
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return the same fields asynchronously."""
        return self(time, variable)


analysis = Synthetic(GLOBAL, lambda lat, lon: np.cos(np.radians(lat)))
radar = Synthetic(RADAR, lambda lat, lon: (lat > 38).astype(float))

# %%
# A Model Grid
# ------------
# The target is a crop of the HRRR Lambert conformal grid, as a regional model would
# declare it. ``infer_grid`` recovers the grid definition from the signature.

# %%
hrrr = cast(ProjectedGrid, resolve_grid("hrrr"))
crop = ProjectedGrid(hrrr.y[200:900:28], hrrr.x[500:1300:32], hrrr.crs)
signature = coord_array(
    ("batch", "variable", "y", "x"),
    {"variable": ["t2m"]},
    dynamic=("batch",),
    grid=crop,
)
target = infer_grid(signature)
print("Target:", target.dims, target.shape)

# %%
# Regrid to a Target Grid
# -----------------------
# Pass ``target_grid`` to receive fields on it. ``fetch_data`` infers the source
# grid from the fetched field. Fields already on the target pass through, and exact
# crops are selected. Anything else is regridded by nearest neighbor, with a log
# message, and only if the target lies within the source; otherwise it raises.

# %%
time = np.array([np.datetime64("2024-06-01T12")])
field = fetch_data(analysis, time, np.array(["t2m"]), target_grid=target)
print("Field:", dict(field.sizes))
print("Geometry follows the target:", field.attrs["type"], field.attrs["dims"])

try:
    fetch_data(radar, time, np.array(["refc"]), target_grid=target)
except ValueError as error:
    print("Radar does not cover the target:", error)

# %%
# Choose a Method
# ---------------
# ``list_regridders`` reports the methods available for two grids, and
# ``resolve_regridder`` builds one. Passing the regridder to ``fetch_data`` also
# accepts its fill, NaN by default, for points the source does not cover.

# %%
print("Methods:", list_regridders(GLOBAL, target), list_regridders(RADAR, target))
field = fetch_data(
    radar,
    time,
    np.array(["refc"]),
    regridder=resolve_regridder(RADAR, target, "bilinear"),
)
print("Radar points without coverage (NaN):", f"{float(np.isnan(field).mean()):.0%}")

# %%
# Reuse a Regridder
# -----------------
# Mapping onto ``target_grid`` builds a regridder, which computes its indices and
# weights, on every call. Build one once and pass it to reuse them. ``fetch_data`` applies it
# after temporal reductions, so the 6 hour mean regrids one field rather than six.

# %%
bilinear = BilinearRegridder(GLOBAL, target)
field = fetch_data(analysis, time, np.array(["t2m", "t2m:mean:6h"]), regridder=bilinear)
print("Field:", dict(field.sizes))

# %%
# Regridded Sources
# -----------------
# ``RegriddedSource`` pairs a source with its regridder so the pair stands in for a
# source, which is how a model recommends a source on another grid. It is an
# ordinary source: each fetched frame is regridded before any temporal reduction.
# For a mean the order does not matter.

# %%
paired = fetch_data(
    RegriddedSource(analysis, bilinear), time, np.array(["t2m:mean:6h"])
)
direct = fetch_data(analysis, time, np.array(["t2m:mean:6h"]), regridder=bilinear)
print("Mean commutes with bilinear:", np.allclose(paired, direct))

# %%
# Coverage
# --------
# Targets without source support get the fill value, NaN by default. Nearest
# regridders can also limit the search distance. ``coverage`` reports which target
# points are supported, for example to build a model's validity mask.

# %%
nearest = NearestRegridder(RADAR, target, max_dist_km=20.0)
print("Radar coverage:", f"{nearest.coverage.mean():.0%}")

# %%
# Composite Sources
# -----------------
# ``CompositeSource`` routes variables to components on a shared grid. Components on
# other native grids are wrapped in regridded sources with the same target, each
# with the method that suits it.

# %%
inputs = CompositeSource(
    {
        "analysis": RegriddedSource(analysis, bilinear),
        "radar": RegriddedSource(radar, nearest),
    },
    {"analysis": ["t2m"], "radar": ["refc"]},
)
field = fetch_data(inputs, time, np.array(["refc", "t2m"]))
print("Composite:", list(field["variable"].values), dict(field.sizes))
