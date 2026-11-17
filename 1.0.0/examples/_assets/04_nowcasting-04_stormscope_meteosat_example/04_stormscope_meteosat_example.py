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
- Fetching FCI observations on the model grid through the model's default source
- Selecting a regional view for presentation
- Running iterative nowcasting
- Plotting an RGB composite of FCI channels with cartopy
"""

# /// script
# dependencies = [
#   "earth2studio[data,stormscope] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# StormScopeMeteosatEU is a pure-observation nowcasting model: it requires no
# external NWP conditioning. The only input is a sliding window of ``L``
# consecutive FCI frames ending at the analysis time.
#
# The model runs on a crop of the 2 km FCI full-disk grid over Europe. Its
# ``default_sources`` returns a :py:class:`earth2studio.data.MeteosatFCI` source
# that reads exactly that crop: the infrared channels at 2 km, and the visible
# and near-infrared channels, which are only available at 1 km, averaged over
# the 2x2 blocks of 1 km pixels that form each 2 km pixel.
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
)
from earth2studio.grids import Region
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU

# %%
# Load Model
# ----------
# Load StormScopeMeteosatEU from a package. ``region`` optionally selects a
# [`earth2studio.grids.Region`][earth2studio.grids.Region] box for
# presentation. The model always runs on its full European domain, because its
# networks are trained for that input size; a regional view crops the forecast
# with ``crop_output``.

# %% tags=["e2sg-profile:setup"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Region.box(west, south, east, north) in degrees, or None for the full domain.
region: Region | None = None  # Example: Region.box(-5.0, 42.0, 10.0, 52.0)

package = StormScopeMeteosatEU.load_default_package()
model = StormScopeMeteosatEU.load_model(package=package, compile=True)
model = model.to(device)
model.eval()
if region is not None:
    model = model.with_region(region)

# %%
# Fetch Initial Condition
# -----------------------
# We fetch an initial window of 6 consecutive FCI frames ending at the chosen
# analysis time. ``fetch_data`` calls the data source at ``start_time +
# lead_time`` for each lead-time offset defined by the model (``[-50 min, -40
# min, ..., 0 min]`` for a 6-frame window), and the model's default source
# returns every channel on the model grid.

# %% tags=["e2sg-profile:setup"]
# Start time; replace with any time for which MTG-I1 FCI data is available.
# StormScopeMeteosatEU was trained on November 2024 - May 2026 data; start times
# should generally be outside that period.
start_time = np.datetime64(datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc))
in_coords = model.input_coords()

x = fetch_data(
    model.default_sources(),
    time=np.array([start_time]),
    variable=in_coords["variable"].values,
    lead_time=in_coords["lead_time"].values,
    device=device,
)

# %%
# Add Ensemble Dimension
# ----------------------
# The model expects input of shape ``(ensemble, time, lead_time, variable, y, x)``.
# Up to GPU memory limits, ``ensemble_size`` can be increased to produce multiple
# independent ensemble members in a single forward pass.

# %% tags=["e2sg-profile:setup"]
ensemble_size = 1
x = x.expand_dims(ensemble=np.arange(ensemble_size)).copy(deep=True)

# %%
# Execute the Nowcast
# -------------------
# ``create_iterator`` handles the autoregressive rollout for us: each yielded
# value advances the prediction by one 10-minute interval, starting with the
# first forecast.

# %% tags=["e2sg-profile:inference"]
n_steps = 12  # 2 hours of 10-minute forecast steps

for step, x_pred in enumerate(tqdm(model.create_iterator(x), total=n_steps)):
    if step == n_steps - 1:
        break

# %%
# Post Processing
# ---------------
# Plot the final predicted FCI frame as a true-color RGB composite (VIS 0.6 /
# 0.5 / 0.4 µm) in geostationary projection. Off-Earth pixels are already set
# to NaN by ``denormalize``. For a regional view, ``crop_output`` crops the
# forecast to the requested region.

# %% tags=["e2sg-profile:plotting"]
rgb_channels = ["fci06vis", "fci05vis", "fci04vis"]
field = model.crop_output(x_pred).e2s.as_numpy().isel(ensemble=0, time=0, lead_time=0)

# here we approximately replicate EUMETSAT's recipe for "true-colour" RGB
rgb = (field.sel(variable=rgb_channels) / 24.0).clip(min=0, max=1) * 255
rgb = rgb.transpose("y", "x", "variable").values
rgb = np.interp(rgb, [0, 30, 60, 120, 190, 255], [0, 110, 160, 210, 240, 255]) / 255.0

proj = ccrs.Geostationary(
    central_longitude=0.0,
    satellite_height=PERSPECTIVE_POINT_HEIGHT,
    globe=ccrs.Globe(semimajor_axis=SEMI_MAJOR_AXIS, semiminor_axis=SEMI_MINOR_AXIS),
    sweep_axis="y",
)
# Pixel-edge extent in metres from the scan-angle labels; x increases westward.
x_angle, y_angle = field["x"].values, field["y"].values
dx, dy = abs(x_angle[1] - x_angle[0]), abs(y_angle[1] - y_angle[0])
extent = [
    -(x_angle.max() + dx / 2) * PERSPECTIVE_POINT_HEIGHT,
    -(x_angle.min() - dx / 2) * PERSPECTIVE_POINT_HEIGHT,
    (y_angle.min() - dy / 2) * PERSPECTIVE_POINT_HEIGHT,
    (y_angle.max() + dy / 2) * PERSPECTIVE_POINT_HEIGHT,
]

aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
fig_height = 8
fig = plt.figure(figsize=(fig_height * aspect, fig_height))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

ax.coastlines(color="white", linewidth=0.8)

# Rows run south to north and columns west to east, matching the extent.
im = ax.imshow(rgb, transform=proj, extent=extent, origin="lower")

valid_time = field["time"].values + field["lead_time"].values
valid_time = np.datetime_as_string(valid_time, unit="m").replace("T", " ")
lead_min = int(field["lead_time"].values / np.timedelta64(1, "m"))
ax.set_title(
    f"StormScopeMeteosatEU — RGB\n" f"Valid {valid_time} UTC  (+{lead_min} min)"
)

fig.tight_layout()
fig.savefig("outputs/04_stormscope_meteosat_example.png", dpi=150)
