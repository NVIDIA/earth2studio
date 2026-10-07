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
DataArray IO Backends
=====================

Plan an output store from a model signature and write DataArrays directly.

This CPU-only tutorial uses the Persistence model, a random data source and the
in-memory XarrayBackend. See ``dev/spec/IO_SPEC.md`` for the contract.
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np
import xarray as xr

from earth2studio.grids import LatLonGrid, infer_grid
from earth2studio.io import XarrayBackend
from earth2studio.io.utils import output_schema
from earth2studio.models.px import Persistence

# %%
# Plan the Store
# --------------
# A model's output signature has a dynamic leading ``batch`` dimension and the
# lead times of one step. ``output_schema`` replaces the dynamic prefix with the
# run's leading dimensions and sets the run's lead-time extent. Nothing is
# allocated: the schema holds only coordinates and metadata.

# %%
grid = LatLonGrid(np.linspace(10.0, -10.0, 5), np.linspace(0.0, 20.0, 9))
model = Persistence(["t2m", "tp:sum:6h"], grid, dt=np.timedelta64(6, "h"))
signature = model.output_coords(model.input_coords())
print("Signature:", signature.dims)

nsteps = 3
times = np.array(["2024-01-01T00", "2024-01-01T12"], dtype="datetime64[h]")
leads = np.arange(nsteps + 1) * np.timedelta64(6, "h")
schema = output_schema(signature, {"time": times}, {"lead_time": leads})
print("Schema:", dict(schema.sizes))

# %%
# Create Arrays
# -------------
# ``add_array`` creates one array per variable label. Labels are used verbatim,
# including statistic qualifiers such as ``tp:sum:6h``. Re-adding the same schema
# is a no-op, so resumed runs can call it unconditionally.

# %%
io = XarrayBackend()
io.add_array(schema)
io.add_array(schema)
print("Arrays:", list(io))

# %%
# Write Model Outputs
# -------------------
# Writes locate each field by its coordinate labels, so drivers pass yields
# straight through. Every output here already matches a planned lead time. The
# synthetic initial condition reuses ``output_schema`` on the input signature.

# %%
initial = output_schema(model.input_coords(), {"time": times})
x = xr.DataArray(
    np.random.default_rng(0).standard_normal(initial.shape).astype(np.float32),
    dims=initial.dims,
    coords=initial.coords,
    attrs=initial.attrs,
)
for y in model.create_iterator(x):
    if y.coords["lead_time"].values[-1] > leads[-1]:
        break
    io.write(y)
io.close()

# %%
# Writes may also hold any subset of labels, in any order, including several lead
# times at once, as models computing multiple lead times per step produce. Unknown
# labels raise before anything is written.

# %%
io = XarrayBackend()
io.add_array(schema)
chunk = xr.zeros_like(x.isel(lead_time=0, drop=True)).expand_dims(
    lead_time=leads[[2, 1]], axis=2
)
io.write(chunk)
print("Written lead times:", io["t2m"].notnull().any(["time", "lat", "lon"]).values)
try:
    io.write(chunk.assign_coords(lead_time=leads[[2, 1]] * 10))
except ValueError as error:
    print("Rejected:", error)

# %%
# Read It Back
# ------------
# Stored arrays keep their coordinates, auxiliary coordinates and grid metadata,
# so the grid is recoverable from the output. Signature markers and the
# statistics attribute are not stored; the ``tp:sum:6h`` label carries it.

# %%
stored = io.root["tp:sum:6h"]
print("Stored dims:", stored.dims)
print("Grid recovered:", infer_grid(stored).fingerprint() == grid.fingerprint())
print("Statistics attribute stored:", "earth2studio_statistics" in stored.attrs)
