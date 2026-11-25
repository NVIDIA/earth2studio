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
Vertical Coupling for Chemistry Models
======================================

Hybrid sigma-pressure to pressure-level interpolation in a connector.

Chemistry emulators typically live on hybrid model levels
(p_k = a_k + b_k * p_s) while meteorology components export pressure-level
fields — or vice versa. When source and destination declare different
vertical coordinates for a field with a "level" dimension, the connector
interpolates linearly in log-pressure, pulling the surface-pressure field
from the source's exports automatically.

Most earth2studio models encode levels in variable names (z500, t850) and
never touch this machinery.

In this example you will learn:

- How components declare export/import vertical coordinates
- How the connector auto-resolves the surface-pressure dependency
- What the failure mode looks like when ps is missing
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Source on Hybrid Levels
# -----------------------
# A "met" component exports ozone on 3 hybrid levels. With surface pressure
# 1000 hPa the levels realize at 300 / 700 / 1000 hPa. The ozone profile is
# f = log(p), so interpolation results are exact and checkable.

from collections import OrderedDict

import numpy as np
import torch

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.dictionary import DEFAULT_DICTIONARY
from earth2studio.nvcoupler.testing import grid_coords

NLAT, NLON = 8, 16
PS = 100000.0  # Pa

d = nvc.FieldDictionary(DEFAULT_DICTIONARY)
d.register(nvc.FieldEntry("ozone_mixing_ratio", "kg kg-1", aliases=frozenset({"o3"})))

hybrid = nvc.HybridLevels(a=(30000.0, 20000.0, 0.0), b=(0.0, 0.5, 1.0))
target = nvc.PressureLevels((500.0, 850.0))


def identity(x, coords):
    return x, coords


met = nvc.CallableComponent(
    "met",
    identity,
    "6h",
    exports=["ozone_mixing_ratio"],
    export_vertical={"ozone_mixing_ratio": hybrid},
    dictionary=d,
)
chem = nvc.CallableComponent(
    "chem",
    identity,
    "6h",
    imports=["ozone_mixing_ratio"],
    import_vertical={"ozone_mixing_ratio": target},
    dictionary=d,
)

# %%
# Initialize and Exchange
# -----------------------
# The met component publishes ozone = log(p) columns plus the surface
# pressure field the hybrid transform needs.

grid = grid_coords(NLAT, NLON)
p_src = np.array(hybrid.a) + np.array(hybrid.b) * PS  # [30000, 70000, 100000] Pa
o3 = (
    torch.tensor(np.log(p_src), dtype=torch.float64)
    .view(1, 3, 1, 1)
    .expand(1, 3, NLAT, NLON)
    .clone()
)
clock = nvc.Clock("2024-01-01", "2024-01-02", "6h")
met.realize(clock)
chem.realize(clock)
met.initialize(
    o3, OrderedDict({"variable": np.array(["o3"]), "level": np.arange(3.0), **grid})
)
met.export_state.add(
    nvc.Field(
        torch.full((NLAT, NLON), PS, dtype=torch.float64),
        OrderedDict(grid),
        "surface_pressure",
        "Pa",
        valid_time=clock.start,
        source="met",
    )
)
chem.initialize(
    torch.zeros(1, 2, NLAT, NLON, dtype=torch.float64),
    OrderedDict(
        {"variable": np.array(["o3"]), "level": np.array([500.0, 850.0]), **grid}
    ),
)

conn = nvc.Connector(met, chem, fields=["ozone_mixing_ratio"])
conn.execute(clock.start)

# %%
# Verify Exactness
# ----------------
# Linear-in-log-p interpolation of f = log(p) must return log(p_target).

got = chem.import_state["ozone_mixing_ratio"]
expected = np.log(np.array([50000.0, 85000.0]))
print(f"received on levels {list(got.coords['level'])} hPa")
print(f"column values: {got.data[:, 0, 0].numpy()}")
print(f"expected:      {expected}")
if not np.allclose(got.data[:, 0, 0].numpy(), expected):
    raise ValueError("hybrid -> pressure interpolation did not match log(p)")
print("hybrid -> pressure interpolation exact ✓")

# %%
# The Failure Mode
# ----------------
# Remove surface pressure from the source and the connector refuses with a
# fix, at the exchange — not as NaNs three days into a rollout.

del met.export_state["surface_pressure"]
try:
    nvc.Connector(met, chem, fields=["ozone_mixing_ratio"]).execute(clock.start)
except nvc.VerticalMismatchError as e:
    print(f"\nVerticalMismatchError: {e}")
