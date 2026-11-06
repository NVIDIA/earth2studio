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
"""StormScope GOES and MRMS nowcasting with CoupledRunner
=======================================================

StormScope nowcasts storms with two diffusion models. The GOES model forecasts
satellite imagery from a short window of past frames. The MRMS model forecasts
radar reflectivity and lightning from its own window, conditioned on the GOES
window, and then on each new GOES frame as the rollout advances.

``CoupledRunner`` runs the pair from one ``Connector``, and ``Pipeline`` writes
one store per model. This replaces the hand-written loop of the StormScope
GOES/MRMS gallery example (``examples/04_nowcasting/03_stormscope_goes_example.py``)
and reproduces its plot over the southern Great Plains.

Needs the StormScope checkpoints, GOES, MRMS and GLM downloads, and a GPU. The
``stormscope`` and ``data`` extras and cartopy are required.
"""

# %%
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.grids import LatLonGrid, NearestRegridder, Region, infer_grid
from earth2studio.io import XarrayBackend
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)
from earth2studio.run import Pipeline, work_items

os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load regional Flash models
# --------------------------
# The Flash variant runs at 3 km every 10 minutes. ``with_region`` makes a view
# of each model over a box, plus up to 25 gridpoints of context per side. The
# box here covers most of tornado alley, from the Texas Panhandle through
# Oklahoma and Kansas to southern Nebraska: roughly 300 by 400 gridpoints.
# Views made from the same region share a grid, which the connector requires.

region = Region.box(-102.5, 32.0, -95.0, 41.0)
package = StormScopeBase.load_default_package()
goes = StormScopeGOES.load_model(package, model_name="3km_10min_flash", amp=True)
mrms = StormScopeMRMS.load_model(package, model_name="3km_10min_flash", amp=True)
goes = goes.to(device).eval().with_region(region)
mrms = mrms.to(device).eval().with_region(region)
print(goes.input_coords().sizes)

# %%
# Couple the models
# -----------------
# One connector feeds the GOES model's frames into the MRMS model's GOES
# forcing slot. Without ``sources``, every other slot comes from the models'
# ``default_sources``, regridded onto the view's grid: GOES imagery for the
# GOES model, and MRMS radar with GLM lightning for the MRMS model. The MRMS
# model's own GOES source is never built, since the connector feeds that slot.
#
# The schedule follows the model contract. The MRMS model initializes from the
# whole GOES input window, which the GOES model holds as its initial condition.
# Each MRMS step then takes the GOES frame at its own state's lead time,
# published one step earlier. This is the loop the gallery example writes out:
#
# .. code-block:: python
#
#     y_goes, s_goes = goes.initialize(x_goes)
#     y_mrms, s_mrms = mrms.initialize(x_mrms, x_goes)
#     for _ in range(nsteps - 1):
#         y_mrms, s_mrms = mrms.step(y_mrms, y_goes, state=s_mrms)
#         y_goes, s_goes = goes.step(y_goes, state=s_goes)

runner = CoupledRunner({"goes": goes, "mrms": mrms}, [Connector("goes", "mrms")])
print(runner.describe(np.timedelta64(30, "m")))

# %%
# Run with Pipeline
# -----------------
# One work item starts at 21 UTC on 6 May 2024, ahead of that evening's
# Oklahoma tornado outbreak, and runs for an hour. ``Pipeline`` regrids each
# stream onto a regular 0.03 degree grid over the box as it writes, so the
# stores hold only the requested region.

grid = infer_grid(goes.input_coords())
box = LatLonGrid(np.arange(41.0, 31.99, -0.03), np.arange(257.5, 265.01, 0.03))
stores = {"goes": XarrayBackend(), "mrms": XarrayBackend()}
to_box = {name: NearestRegridder(grid, box, max_dist_km=3.0) for name in stores}
start = np.datetime64("2024-05-06T21:00")
items = work_items([start], np.timedelta64(60, "m"))
Pipeline(runner, output=stores, regridder=to_box, device=device).run(items)
print(dict(stores["mrms"].root["refc"].sizes))

# %%
# Plot
# ----
# As in the gallery example: GOES clean infrared (``abi13c``) in grayscale with
# MRMS composite reflectivity above 0 dBZ overlaid, for the observed initial
# condition and two forecast lead times.

ir = stores["goes"].root["abi13c"].isel(time=0)
refc = stores["mrms"].root["refc"].isel(time=0)
leads = np.array([0, 30, 60]) * np.timedelta64(1, "m")
lat, lon = box.latitude, box.longitude

projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)
fig, axes = plt.subplots(
    1,
    len(leads),
    figsize=(3.6 * len(leads) + 2, 5.5),
    subplot_kw={"projection": projection},
    layout="constrained",
)
for ax, lead in zip(axes, leads):
    ax.set_extent([-102.5, -95.0, 32.0, 41.0], crs=ccrs.PlateCarree())
    goes_im = ax.pcolormesh(
        lon,
        lat,
        ir.sel(lead_time=lead).values,
        transform=ccrs.PlateCarree(),
        cmap="gray_r",
        shading="auto",
    )
    field = refc.sel(lead_time=lead).values
    mrms_im = ax.pcolormesh(
        lon,
        lat,
        np.where(field <= 0, np.nan, field),
        transform=ccrs.PlateCarree(),
        cmap="inferno",
        shading="auto",
        vmin=0.0,
        vmax=55.0,
    )
    # Black halo under white lines, visible by day and night
    ax.coastlines(color="black", linewidth=1.2)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=1.0)
    ax.coastlines(color="white", linewidth=0.4)
    ax.add_feature(cfeature.STATES, edgecolor="white", linewidth=0.3)
    label = "observed" if lead == 0 else "forecast"
    ax.set_title(f"+{lead.astype(int)} min ({label})")
fig.colorbar(
    goes_im,
    ax=axes,
    label="GOES Clean IR 10.35um [K]",
    location="right",
    shrink=0.8,
)
fig.colorbar(
    mrms_im,
    ax=axes,
    label="MRMS Reflectivity [dBZ]",
    location="right",
    shrink=0.8,
)
fig.suptitle(
    f"StormScope GOES abi13c with MRMS overlay from "
    f"{np.datetime_as_string(start, unit='m')} UTC initialization"
)
fig.savefig("outputs/11_stormscope_coupling.png", dpi=200, bbox_inches="tight")
