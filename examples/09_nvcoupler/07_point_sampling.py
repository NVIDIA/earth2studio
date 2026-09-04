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
Grid-to-Point Sampling for Station-Level Applications
======================================================

Delivering a gridded field to a scattered set of locations.

Many applications need a coarse gridded forecast at specific, non-gridded
points rather than on a mesh: verification against station observations,
site-level agriculture fields, or energy-asset locations. A destination
component whose grid is a :class:`~earth2studio.nvcoupler.PointSet` (arbitrary
(lat, lon) locations, not a mesh) tells the Connector to sample instead of
regrid — ``sample="nearest"`` or ``sample="bilinear"``.

In this example you will learn:

- How to declare a point-target component with ``points=PointSet(...)``
- How ``sample="nearest"`` and ``sample="bilinear"`` differ
- What the failure mode looks like when neither ``sample=`` nor a custom
  ``regridder=`` is given for a point target
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Source: a Gridded "Atmosphere"
# -------------------------------
# A 32x64 lat/lon component exporting a field that is exactly linear in
# (lat, lon): ``temperature = lat + 0.1 * lon``. Linearity makes both nearest
# and bilinear sampling hand-checkable — bilinear must reproduce the formula
# exactly anywhere inside the grid, and nearest must reproduce it exactly at
# any point that coincides with a grid cell.

from collections import OrderedDict

import numpy as np
import torch

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import grid_coords

NLAT, NLON = 32, 64


def identity(x, coords):
    return x, coords


atmos = nvc.CallableComponent(
    "atmos", identity, "6h", exports=["air_temperature_2m"]
)
grid = grid_coords(NLAT, NLON)
lat, lon = np.asarray(grid["lat"]), np.asarray(grid["lon"])
temperature = torch.as_tensor(lat).view(-1, 1) + 0.1 * torch.as_tensor(lon).view(1, -1)

clock = nvc.Clock("2024-01-01", "2024-01-02", "6h")
atmos.realize(clock)
atmos.initialize(
    temperature.unsqueeze(0).double(),
    OrderedDict({"variable": np.array(["air_temperature_2m"]), **grid}),
)

# %%
# Destination: Named Stations
# ----------------------------
# Three stations: two sit exactly on grid points (so nearest is exact there
# too), one sits at the midpoint between four cells (nearest and bilinear
# will disagree there).

stations = nvc.PointSet(
    lat=np.array([lat[4], lat[10], (lat[6] + lat[7]) / 2]),
    lon=np.array([lon[3], lon[20], (lon[12] + lon[13]) / 2]),
    names=("boulder", "denver", "midpoint"),
)
site = nvc.CallableComponent(
    "stations", identity, "6h", imports=["air_temperature_2m"], points=stations
)
site.realize(clock)
# initialize() always publishes through State.from_tensor, which requires a
# 'variable' dim even for a component with no exports; a placeholder value
# is skipped (nothing in export_names asks for it).
site.initialize(
    torch.zeros(1, len(stations)),
    OrderedDict({"variable": np.array(["_ic"]), "point": stations.labels()}),
)

# %%
# Sample: Nearest vs. Bilinear
# ------------------------------
nvc.Connector(atmos, site, sample="nearest").execute(clock.start)
nearest = site.import_state["air_temperature_2m"].data.clone()

nvc.Connector(atmos, site, sample="bilinear").execute(clock.start)
bilinear = site.import_state["air_temperature_2m"].data.clone()

expected_exact = lat[[4, 10]] + 0.1 * lon[[3, 20]]  # the two on-grid stations
expected_midpoint = float(
    (lat[6] + lat[7]) / 2 + 0.1 * (lon[12] + lon[13]) / 2
)  # linear field: exact anywhere under bilinear

print(f"stations:          {stations.labels()}")
print(f"nearest sample:     {nearest.numpy()}")
print(f"bilinear sample:     {bilinear.numpy()}")
print(f"expected (on-grid):  {expected_exact}")
print(f"expected (midpoint): {expected_midpoint:.4f}")

if not np.allclose(nearest.numpy()[:2], expected_exact):
    raise ValueError("nearest sampling at on-grid stations did not match")
if not np.allclose(bilinear.numpy(), [*expected_exact, expected_midpoint]):
    raise ValueError("bilinear sampling did not exactly reproduce the linear field")
if np.isclose(nearest.numpy()[2], bilinear.numpy()[2]):
    raise ValueError("nearest and bilinear were expected to disagree at the midpoint")
print("nearest and bilinear both correct; they disagree at the midpoint as expected ✓")

# %%
# The Failure Mode
# -----------------
# A point-target destination with neither ``sample=`` nor a custom
# ``regridder=`` refuses at the exchange, with the fix in the message.

try:
    nvc.Connector(atmos, site).execute(clock.start)
except nvc.CouplingError as e:
    print(f"\nCouplingError: {e}")
