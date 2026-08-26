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
StormScope Meteosat FCI Nowcasting
===================================

StormScope inference workflow with MTG-I1 FCI satellite imagery.

This example will demonstrate how to run inference to generate nowcasts
of Meteosat Third Generation (MTG-I1) FCI satellite imagery using
StormScopeMeteosatEU, a generative diffusion model that predicts the next
10-minute FCI frame from a sliding window of recent observations.

In this example you will learn:

- How to instantiate :py:class:`earth2studio.models.px.StormScopeMeteosatEU`
- Creating a :py:class:`earth2studio.data.MeteosatFCI` data source
- Fetching FCI observations as the model input
- Running iterative nowcasting
- Plotting an RGB composite of FCI channels with cartopy
"""

# /// script
# dependencies = [
#   "earth2studio[stormscope] @ git+https://github.com/NVIDIA/earth2studio.git@0.18.0",
#   "cartopy",
#   "eumdac",
#   "netCDF4",
# ]
# ///

# %%
# Set Up
# ------
# StormScopeMeteosatEU is a pure-observation nowcasting model: it requires no
# external NWP conditioning. The only input is a sliding window of ``L``
# consecutive FCI frames ending at the analysis time.
#
# The model operates on a rectangular sub-region of the full MTG disk defined
# in native FCI pixel coordinates by
# :py:attr:`StormScopeMeteosatEU.Model_FCI_BBox`.  We configure
# :py:class:`earth2studio.data.MeteosatFCI` with the matching ``pixel_bbox``
# so that data fetching returns exactly the pixels the model expects, with
# no interpolation needed.
#
# EUMETSAT Data Store credentials must be available via the environment
# variables ``EUMETSAT_CONSUMER_KEY`` and ``EUMETSAT_CONSUMER_SECRET``, or in
# the ``~/.eumdac/credentials`` file.

# %%
import os
from datetime import datetime, timezone

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from earth2studio.data import fetch_data
from earth2studio.data.meteosat_fci import (
    PERSPECTIVE_POINT_HEIGHT,
    SEMI_MAJOR_AXIS,
    SEMI_MINOR_AXIS,
    MeteosatFCI,
)
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
from earth2studio.utils.coords import map_coords

# %%
# Load Model
# ----------
# Load StormScopeMeteosatEU from a package.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

package = StormScopeMeteosatEU.load_default_package()
model = StormScopeMeteosatEU.load_model(package=package)
model = model.to(device)
model.eval()
model.compile_model()

# %%
# Set Up Data Sources
# -------------------
# ``MeteosatFCI`` is configured with ``pixel_bbox`` matching the model's
# domain (``Model_FCI_BBox``) so that the returned ``y``/``x`` scan-angle
# coordinates align with what the model expects.
#
# Meteosat FCI visible and NIR channels are provided by the EUMETSAT data store
# at 1 km resolution while IR channels are 2 km. The StormScope MTG model uses
# the 2 km grid for all channels, so we need to pull data from both 1 km and 2 km
# sources and do a minimal regrid on the visible and NIR channels.
#
# Here we will set up data sources for both grid resolutions.
# The 1 km grid is exactly 2 times the pixels of the 2 km grid, so we get
# the 1 km bounding box by doubling the 2 km bounding box.

# %%
bbox_2km = StormScopeMeteosatEU.Model_FCI_BBox
FCI_BBox = {"2km": bbox_2km}
FCI_BBox["1km"] = (
    (2 * bbox_2km[0][0], 2 * bbox_2km[0][1]),
    (2 * bbox_2km[1][0], 2 * bbox_2km[1][1]),
)
fci = {
    res: MeteosatFCI(resolution=res, pixel_bbox=bbox)
    for (res, bbox) in FCI_BBox.items()
}

# %%
# Fetch Initial Condition
# -----------------------
# We fetch initial window of 6 consecutive FCI frames ending at
# the chosen analysis time. ``fetch_data`` calls the data source at
# ``start_time + lead_time`` for each lead-time offset defined by the model
# (``[-50 min, -40 min, ..., 0 min]`` for a 6-frame window).
#
# StormScope MTG uses the 2 km grid for all channels, so we use
# ``StormScopeMeteosatEU.combine_1km_2km_inputs`` to resize the channels
# retrieved from the 1 km data source.

# %%
# Start time; replace with any time for which MTG-I1 FCI data is available.
# StormScopeMeteosatEU was trained on November 2024 - May 2026 data; start times
# should generally be outside that period.
start_time = np.datetime64(datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc))
in_coords = model.input_coords()
variables = in_coords["variable"]

with torch.no_grad():
    x_res = {}
    for res in ["2km", "1km"]:
        available_vars = fci[res].available_variables()
        if res == "1km":
            # use 2km version if variable available in both 1km and 2km
            available_vars -= fci["2km"].available_variables()
        variables_res = [var for var in variables if var in available_vars]
        x_res[res] = fetch_data(
            fci[res],
            time=np.array([start_time]),
            variable=variables_res,
            lead_time=in_coords["lead_time"],
            device=device,
        )

    # 2x downsample 1km data to common 2km grid
    x, coords = StormScopeMeteosatEU.combine_1km_2km_inputs(
        *x_res["1km"], *x_res["2km"]
    )
    del x_res
    # ensure data is on model grid (reorders variables if needed)
    x, coords = map_coords(x, coords, in_coords)

# %%
# Add Ensemble Dimension
# ----------------------
# The model expects input of shape ``(ensemble, time, lead_time, variable, y, x)``.
# Up to GPU memory limits, ``ensemble_size`` can be increased to produce multiple
# independent ensemble members in a single forward pass.

# %%
ensemble_size = 1
x = x.expand(ensemble_size, -1, -1, -1, -1, -1)
coords["ensemble"] = np.arange(ensemble_size)
coords.move_to_end("ensemble", last=False)

# %%
# Execute the Nowcast
# -------------------
# ``create_iterator`` handles the autoregressive rollout for us: the first
# yielded value is the (unchanged) initial condition, and each subsequent
# value advances the prediction by one 10-minute interval.

# %%
n_steps = 12  # 2 hours of 10-minute forecast steps

for step, (x_pred, coords_pred) in enumerate(
    tqdm(model.create_iterator(x, coords), total=n_steps + 1)
):
    if step == n_steps:
        break

# %%
# Post Processing
# ---------------
# Plot the final predicted FCI frame as a true-color RGB composite (VIS 0.6 /
# 0.5 / 0.4 µm) in geostationary projection. Off-Earth pixels are already set
# to NaN by ``denormalize``.

# %%
rgb_channels = ["fci06vis", "fci05vis", "fci04vis"]
ch_idx = [list(model.variables).index(ch) for ch in rgb_channels]

# x_pred has shape (batch, time, 1, variable, y, x)
# here we approximately replicate EUMETSAT's recipe for "true-colour" RGB
rgb = (x_pred[0, 0, 0, ch_idx] / 24.0).clamp(min=0, max=1) * 255
rgb = rgb.permute(1, 2, 0).cpu().numpy()
rgb = np.interp(rgb, [0, 30, 60, 120, 190, 255], [0, 110, 160, 210, 240, 255]) / 255.0

proj = ccrs.Geostationary(
    central_longitude=0.0,
    satellite_height=PERSPECTIVE_POINT_HEIGHT,
    globe=ccrs.Globe(semimajor_axis=SEMI_MAJOR_AXIS, semiminor_axis=SEMI_MINOR_AXIS),
)
extent = MeteosatFCI.projection_extent(resolution="2km", pixel_bbox=bbox_2km)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

ax.coastlines(color="white", linewidth=0.8)

im = ax.imshow(rgb, transform=proj, extent=extent, origin="lower")

valid_time = coords_pred["time"][0]
valid_time = datetime.fromisoformat(str(valid_time)).strftime("%Y-%m-%d %H:%M")
lead_min = coords_pred["lead_time"][0].astype("timedelta64[m]").item()
lead_min = int(lead_min / np.timedelta64(1, "m"))
ax.set_title(
    f"StormScopeMeteosatEU — RGB\n" f"Valid {valid_time} UTC  (+{lead_min} min)"
)

fig.tight_layout()
fig.savefig("outputs/04_stormcast_meteosat_example.png", dpi=150)
print("Saved outputs/04_stormcast_meteosat_example.png")
