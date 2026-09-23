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
Allocation-Free Coordinate Signatures
====================================

Describe grids and plan model outputs without allocating forecast fields.

This CPU-only tutorial uses synthetic geometry and the same helpers as
StormScopeGOES, StormScopeMRMS, StormCastCONUS, and PrecipitationAFNO.
Coordinate arrays occupy memory; the field backing array stores only shape/dtype.
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np

from earth2studio.grids import CurvilinearGrid, ProjectedGrid, infer_grid, resolve_grid
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_time,
)

# %%
# Declare Regional Geometry
# -------------------------
# StormScope declares its checkpoint latitude/longitude as a curvilinear grid.
# Dynamic leading dimensions are explicit; all trailing dimensions are fixed.

# %%
lat = np.array([[30.0, 30.1, 30.2], [31.0, 31.1, 31.2]])
lon = np.array([[250.0, 251.0, 252.0], [250.1, 251.1, 252.1]])
regional = coord_array(
    ("batch", "time", "lead_time", "variable", "y", "x"),
    {
        "lead_time": np.array([-2, -1, 0], dtype="timedelta64[h]"),
        "variable": ["u10m", "v10m"],
    },
    dynamic=("batch", "time"),
    grid=CurvilinearGrid(lat, lon),
)
np.testing.assert_equal(regional.data.nbytes, 0)
print("Regional declaration:", regional.dims, regional.shape)

# %%
# Plan Concrete Outputs
# ---------------------
# Leading dimensions can have concrete sizes without allocating fields. Rebase
# lead times for validation, then advance from the final input lead time.
# Converted models accept these coordinates alongside a separate field tensor:
# ``prediction, prediction_coords = model(tensor, concrete)``. The tensor shape
# must equal ``concrete.shape``. Iterators accept and yield the same pair.

# %%
concrete = coord_array_like(
    regional,
    {
        "batch": np.arange(4),
        "time": np.array(["2026-09-17"], dtype="datetime64[ns]"),
        "lead_time": np.array([5, 6, 7], dtype="timedelta64[h]"),
    },
)
handshake_time(concrete)
handshake_time(concrete, "lead_time")
handshake_dataarray(
    concrete.assign_coords(lead_time=concrete.lead_time - concrete.lead_time[-1]),
    regional,
)
forecast = coord_array_like(
    concrete, {"lead_time": np.array([8, 9], dtype="timedelta64[h]")}
)
np.testing.assert_equal(forecast.shape, (4, 1, 2, 2, 2, 3))
np.testing.assert_equal(forecast.data.nbytes, 0)
print("Planned forecast:", forecast.shape, np.asarray(forecast.lead_time))

# %%
# Preserve Native Projected Axes
# -----------------------------
# CONUS crops HRRR's projected axes and reuses its registered CRS. Build a grid
# definition from the selected axes, then pass it to ``coord_array`` just as in
# the regional example above. Cropping preserves the native coordinate values
# in meters; it does not reset them to zero-based pixel indexes.
#
# Use the cropped definition rather than ``grid="hrrr"``: the registry name
# describes the full HRRR domain. Model axis names are mapped at signature
# construction; ``infer_grid`` uses standard names.

# %%
hrrr = resolve_grid("hrrr")
if not isinstance(hrrr, ProjectedGrid):
    raise TypeError("The registered HRRR grid must be a ProjectedGrid")
y_slice, x_slice = slice(10, 12), slice(20, 23)
crop = ProjectedGrid(hrrr.y[y_slice], hrrr.x[x_slice], hrrr.crs)
projected = coord_array(
    ("batch", "time", "lead_time", "variable", "y", "x"),
    {
        "lead_time": np.array([0], dtype="timedelta64[h]"),
        "variable": ["u10m", "v10m"],
    },
    dynamic=("batch", "time"),
    grid=crop,
)

# Geographic auxiliary coordinates are derived by coord_array from this cropped
# ProjectedGrid; callers do not need to provide lat/lon separately.
np.testing.assert_equal(projected.lat.dims, ("y", "x"))
np.testing.assert_equal(projected.lon.dims, ("y", "x"))
np.testing.assert_equal(projected.shape, (0, 0, 1, 2, 2, 3))
np.testing.assert_equal(projected.data.nbytes, 0)
np.testing.assert_array_equal(projected.y, hrrr.y[y_slice])
np.testing.assert_array_equal(projected.x, hrrr.x[x_slice])
inferred = infer_grid(projected)
np.testing.assert_equal(inferred.fingerprint(), crop.fingerprint())
print("Full HRRR grid:", hrrr.shape, "Cropped grid:", crop.shape)
print("Cropped forecast signature:", projected.dims, projected.shape)
print("Native CRS:", projected.attrs["earth2studio_crs"])

# %%
# Declare Accumulated Precipitation
# --------------------------------
# PrecipitationAFNO uses the registered FCN grid. Variable replacement drops
# variable-dependent auxiliaries. Qualified labels declare the output statistic;
# coord_array_like derives matching metadata automatically.

# %%
atmosphere = coord_array(
    ("batch", "variable", "lat", "lon"),
    {"variable": ["u10m", "v10m"]},
    dynamic=("batch",),
    grid="latlon-0.25deg-south-pole-excluded",
)
precipitation = coord_array_like(atmosphere, {"variable": ["tp:sum:6h"]})
np.testing.assert_equal(precipitation.shape, (0, 1, 720, 1440))
np.testing.assert_equal(precipitation.data.nbytes, 0)
print("Precipitation statistics:", precipitation.attrs["earth2studio_statistics"])

# %%
# Validation Includes Geometry
# ----------------------------
# Equal spatial sizes alone do not identify the same curvilinear grid.

# %%
try:
    handshake_dataarray(regional.assign_coords(lat=regional.lat + 1), regional)
except ValueError as error:
    print("Rejected incorrect geometry:", error)
else:
    raise AssertionError("Incorrect latitude must be rejected")
