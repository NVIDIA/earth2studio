# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
AIDA and WeatherMesh-5c
======================

AIDA is WindBorne's AI data assimilation model: it updates an IFS background
using public observations. WeatherMesh-5c starts from that analysis and returns
global surface and pressure-level forecasts every three hours.

In this example you will learn:

- How to initialize WeatherMesh-5c from an AIDA analysis.
- How to map four forecast members to two IFS backgrounds and fork each pair.
- How to write Zarr output and visualize the difference between two branches.

This is a reduced demonstration: two IFS backgrounds, two branches per
background, and temperature forecasts through +96h. For the operational-style
ensemble and output schedule (24 backgrounds, eight branches each at +72h,
and surface products through +360h), use the standalone
[`wm5c-run`](https://github.com/windborne/weathermesh-5c#end-to-end-forecast).
Both paths use public input sources, not identical production inputs.

Use a CUDA GPU with 40 GB of memory recommended. Model assets total 3.54 GB;
allow several additional GB for the public inputs and about 1 GB for Zarr
output. Budget roughly 10 minutes with warm caches on a datacenter GPU; a
first run also downloads inputs, so time depends on network and cache speed.
"""

# /// script
# dependencies = [
#   "earth2studio[da-aida] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "weathermesh-5c[windborne] @ git+https://github.com/windborne/weathermesh-5c.git@b27c315175982f5bb057c644d4bc0e1a16e0f790",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# - Model: [`AIDAWeatherMesh5C`][earth2studio.models.da.AIDAWeatherMesh5C].
# - Background: [`IFS_ENS`][earth2studio.data.IFS_ENS] and [`OISST`][earth2studio.data.OISST].
# - Observations: [`NNJAObsConv`][earth2studio.data.NNJAObsConv],
#   [`NNJAObsSat`][earth2studio.data.NNJAObsSat], and
#   [`NNJAObsSatwnd`][earth2studio.data.NNJAObsSatwnd].
# - Workflow: [`CoupledRunner`](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/coupling/runner.py),
#   [`Pipeline`](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/run/pipeline.py), and [`ZarrBackend`][earth2studio.io.ZarrBackend].

# %%
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from earth2studio.coupling import CoupledRunner
from earth2studio.io import ZarrBackend
from earth2studio.models.da import AIDAWeatherMesh5C
from earth2studio.run import Pipeline, work_items

load_dotenv()
os.makedirs("outputs", exist_ok=True)

# %%
# Load the Model
# --------------
# At the 09 UTC analysis time, the IFS background is from 00 UTC and the three
# AIDA updates are centered on 03, 06, and 09 UTC. OISST supplies sea surface
# temperatures; only public observations are used. The first forecast is at
# +3h relative to the analysis.
# For two updates from the same 00 UTC background, set `assimilation_steps=2`
# and change the analysis time below to `2026-09-21T06`; updates then occur at
# 03 and 06 UTC. The update count is explicit, not based on the current clock.

# %%
model = AIDAWeatherMesh5C.load_model(
    AIDAWeatherMesh5C.load_default_package(),
    assimilation_steps=3,
    variables=("t2m",),
    fork_hour=72,
)

# %%
# Run with Pipeline
# -----------------
# `source_member` maps members 0 and 1 to IFS ENS member 1 and members 2 and 3
# to IFS ENS member 2. Each pair shares a background until a learned
# perturbation at +72h gives the branches different forecasts.
# `member_batch=2` puts each pair in the same work item so its branches reuse
# observation preparation. Each branch still runs its own AIDA analysis and
# pre-fork forecast.

# %%
runner = CoupledRunner(
    {"forecast": model}, source_member=lambda member: member // 2 + 1, seed=0
)
items = work_items(
    [np.datetime64("2026-09-21T09")],
    final_lead_time=np.timedelta64(96, "h"),
    members=range(4),
    member_batch=2,
)
store = ZarrBackend("outputs/aida-weathermesh.zarr", backend_kwargs={"overwrite": True})
Pipeline(runner, output={"forecast": store}, device="cuda:0").run(items)

# %%
# Plot the Effect of the Fork
# --------------------------
# Compare two branches from the same IFS background at +96h. The difference
# reflects the learned perturbation and its evolution, not different IFS inputs.

# %%
plt.close("all")
temperature = (
    store.to_xarray()["t2m"].sel(lead_time=np.timedelta64(96, "h")).isel(time=0)
)
difference = abs(temperature.sel(ensemble=0) - temperature.sel(ensemble=1))
figure, axis = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(10, 5))
plot = axis.pcolormesh(
    difference.lon,
    difference.lat,
    difference,
    transform=ccrs.PlateCarree(),
    cmap="magma",
)
axis.coastlines()
axis.set_title("WeatherMesh-5c +96h: temperature difference between fork branches")
figure.colorbar(plot, ax=axis, label="Absolute temperature difference (K)", shrink=0.8)
figure.savefig("outputs/05_aida_weathermesh.jpg", bbox_inches="tight")
