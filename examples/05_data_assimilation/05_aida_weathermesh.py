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
=======================

Observation-initialized global ensemble forecasts with WindBorne's AIDA data
assimilation and WeatherMesh-5c forecast models.

AIDA is WindBorne's AI data assimilation model: starting from an IFS ensemble
background, it applies a sequence of three-hourly updates using public satellite
and conventional observations to produce an analysis. WeatherMesh-5c then
forecasts forward from that analysis, returning global surface and
pressure-level fields every three hours.

In this example you will learn:

- How AIDA and WeatherMesh-5c fit together in time.
- How to run the combined model with
  [`CoupledRunner`][earth2studio.coupling.CoupledRunner] and
  [`Pipeline`][earth2studio.run.Pipeline].
- How to map four forecast members to two IFS backgrounds and fork each pair.
- How to write Zarr output and plot a forecast and the difference between two
  branches.

The two models can also be used on their own:
[`AIDA`][earth2studio.models.da.AIDA] returns just the analysis, and
[`WeatherMesh5C`][earth2studio.models.px.WeatherMesh5C] forecasts directly from
IFS fields without assimilation. The combined
[`AIDAWeatherMesh5C`][earth2studio.models.da.AIDAWeatherMesh5C] run through
`CoupledRunner`, as shown here, is the recommended way to produce
observation-initialized forecasts.

This is a reduced demonstration: two IFS backgrounds, two branches per
background, and 2 m temperature through +96h. To scale up toward the
operational setup (24 backgrounds, eight branches each forked at +72h, and
forecasts through +360h), increase `members`, the `source_member` mapping,
`final_lead_time` and the `variables` selection below. WindBorne's standalone
[`wm5c-run`](https://github.com/windborne/weathermesh-5c#end-to-end-forecast)
is a useful reference for that configuration. Both paths use public input
sources, not identical production inputs.

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
# - Workflow: [`CoupledRunner`][earth2studio.coupling.CoupledRunner],
#   [`Pipeline`][earth2studio.run.Pipeline], and [`ZarrBackend`][earth2studio.io.ZarrBackend].

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
# The work item time is the analysis time, here 09 UTC on 21 September 2026.
# The timeline is:
#
# - **00 UTC**: the IFS ensemble background is valid (nine hours before the
#   analysis). OISST supplies sea surface temperatures.
# - **03, 06 and 09 UTC**: AIDA applies three updates, each using public
#   observations within ±1.5 hours of the update time. The 09 UTC update yields
#   the analysis.
# - **12 UTC onward**: WeatherMesh-5c forecasts from the 09 UTC analysis in
#   three-hour steps, so the first output is at +3h (12 UTC) and lead times are
#   relative to the analysis.
#
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
    {"forecast": model},
    source_member=lambda member: member // 2 + 1,
    seed=0,
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
# Plot the Forecast and the Effect of the Fork
# --------------------------------------------
# The left panel shows the +96h 2 m temperature forecast for member 0. The
# right panel compares two branches from the same IFS background: their
# difference reflects the learned perturbation at +72h and its evolution, not
# different IFS inputs.

# %%
plt.close("all")
temperature = (
    store.to_xarray()["t2m"].sel(lead_time=np.timedelta64(96, "h")).isel(time=0)
)
difference = abs(temperature.sel(ensemble=0) - temperature.sel(ensemble=1))
figure, axes = plt.subplots(
    1, 2, subplot_kw={"projection": ccrs.Robinson()}, figsize=(16, 4.5)
)
panels = (
    (temperature.sel(ensemble=0), "Spectral_r", "2 m temperature (K)", "Member 0"),
    (difference, "magma", "Absolute difference (K)", "|Member 0 - Member 1|"),
)
for axis, (field, cmap, label, title) in zip(axes, panels):
    plot = axis.pcolormesh(
        field.lon, field.lat, field, transform=ccrs.PlateCarree(), cmap=cmap
    )
    axis.coastlines()
    axis.set_title(f"WeatherMesh-5c +96h: {title}")
    figure.colorbar(plot, ax=axis, label=label, shrink=0.8)
figure.savefig("outputs/05_aida_weathermesh.png", bbox_inches="tight")
