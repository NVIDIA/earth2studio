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
Coupled Runs under Pipeline
===========================

Run a coupled system over work items with ``Pipeline``.

The :class:`~earth2studio.nvcoupler.Coupler` implements the
:class:`~earth2studio.run.Runner` protocol, so :class:`~earth2studio.run.Pipeline`
supervises coupled runs exactly like single-model ones: it plans one output
store per stream from the coupler's schemas, distributes work items, writes each
step, and can resume. The coupler only decides what each model receives and when.

In this example you will learn:

- planning stores from :meth:`~earth2studio.nvcoupler.Coupler.output_coords`
- predownload planning with :meth:`~earth2studio.nvcoupler.Coupler.data_requests`
- running several initial times through ``Pipeline``
- starting from in-memory initial conditions with ``ics``
"""

# /// script
# dependencies = [
#   "earth2studio[utils] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np

from earth2studio.io import KVBackend
from earth2studio.nvcoupler import Connector, Coupler
from earth2studio.nvcoupler.testing import (
    ATMOS_SHAPE,
    OCEAN_SHAPE,
    ToyAtmosphere,
    ToyOcean,
    toy_grid,
    toy_source,
)
from earth2studio.run import Pipeline, WorkItem, work_items

coupler = Coupler(
    {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={
        "atmos": (toy_source(toy_grid(*ATMOS_SHAPE), 0.0), None),
        "ocean": (toy_source(toy_grid(*OCEAN_SHAPE), 2.0), None),
    },
)

# %%
# Schemas and requests
# --------------------
# ``output_coords`` returns one allocation-free signature per stream covering
# every lead time of an item; ``data_requests`` lists every fetch an item makes.

# %%
final_lead_time = np.timedelta64(96, "h")
for stream, schema in coupler.output_coords(final_lead_time).items():
    print(stream, schema.dims, schema["lead_time"].values.astype("timedelta64[h]"))

item = WorkItem(np.datetime64("2026-01-01T00"), final_lead_time)
for request in coupler.data_requests(item) or ():
    print(request.variable, request.lead_time.astype("timedelta64[h]"))

# %%
# Run through Pipeline
# --------------------
# Streams are keyed by model name, so the output mapping names the models. Two
# initial times become two work items written into the same stores.

# %%
atmos_store, ocean_store = KVBackend(), KVBackend()
items = work_items(
    [np.datetime64("2026-01-01T00"), np.datetime64("2026-01-02T00")], final_lead_time
)
Pipeline(
    coupler, output={"atmos": atmos_store, "ocean": ocean_store}, device="cpu"
).run(items)
print("z1000", atmos_store["z1000"].shape, "sst", ocean_store["sst"].shape)
print("sst per item at 48h:", ocean_store["sst"][:, 1, 1, 1])

# %%
# In-memory initial conditions
# ----------------------------
# Callers already holding arrays pass them as ``ics``; the coupler then skips
# fetching those input slots. Forcing is still served by connectors and sources.

# %%
import xarray as xr

grid = toy_grid(*ATMOS_SHAPE)
ic = xr.DataArray(
    np.full((1, 1, *ATMOS_SHAPE), 5.0, dtype=np.float32),
    dims=("lead_time", "variable", "lat", "lon"),
    coords={
        "lead_time": np.array([0], dtype="timedelta64[h]"),
        "variable": ["z1000"],
        **grid.coords(only_index=True),
    },
)
steps = list(coupler.run_item(item, ics={"atmos": ic}))
print("z1000 at 0h:", float(steps[0]["atmos"].isel(lat=1, lon=1).values.ravel()[0]))
print("z1000 at 6h:", float(steps[1]["atmos"].isel(lat=1, lon=1).values.ravel()[0]))
