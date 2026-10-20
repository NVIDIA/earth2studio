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
StormScope Satellite and Radar Nowcasting
=========================================

StormScope inference workflow with GOES satellite imagery and MRMS radar data.

This example will demonstrate how to run coupled inference to generate
predictions using StormScope models with both GOES and MRMS data sources.

In this example you will learn:

- How to instantiate StormScope models for GOES and MRMS
- Fetching model inputs on the model grid through the models' default sources
- Selecting a regional domain for Flash inference
- Running iterative prognostic forecasts
- Plotting a single GOES channel with MRMS overlay
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
# This example shows a minimal StormScope workflow with GOES satellite imagery
# and MRMS radar data. We build two models:
#
# - [`earth2studio.models.px.StormScopeGOES`][earth2studio.models.px.StormScopeGOES] to forecast GOES channels.
# - [`earth2studio.models.px.StormScopeMRMS`][earth2studio.models.px.StormScopeMRMS] to forecast radar
#   reflectivity (and a gridded GLM lightning channel).
#
# In the CONUS nowcasting configuration the GOES model is "pure obs" (no external
# conditioning), while the MRMS model is conditioned on GOES — the GOES model
# provides that conditioning during the rollout via ``initialize`` and ``step``.
# The MRMS model additionally has a GLM lightning channel (``glm_density``) as
# part of its state, which evolves autoregressively over the rollout.

# %%
import os
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from earth2studio.data import fetch_data
from earth2studio.grids import Region
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

# %%
# We use the Flash CONUS nowcasting variant (``3km_10min_flash``) for faster
# inference. The GOES model forecasts the eight ABI channels from their recent
# history. The MRMS model forecasts ``[refc, refc_base, glm_density]``
# conditioned on GOES; its Geostationary Lightning Mapper (GLM) channel is both an
# input (observation history) and a predicted output.
#
# Selectable variants (see ``StormScope*.list_available_models``):
#
# - "3km_10min": baseline CONUS nowcasting, 3 km / 10 min
# - "3km_10min_flash": accelerated CONUS nowcasting, 3 km / 10 min (example default)
# - "6km_1hr": legacy 6 km / 60 min nearcasting
#
# StormScope models support regional inference through ``with_region``. Set
# ``region`` to a [`earth2studio.grids.Region`][earth2studio.grids.Region] box
# for local Flash inference within CONUS, or leave it as ``None`` for the full
# domain. A Flash view keeps up to ``region_padding`` (25) gridpoints of real
# context per side and runs on at least 200 x 200 gridpoints; both models share
# the loaded weights with their views. Make the GOES and MRMS views from the same
# region so that their grids match.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "3km_10min_flash"
# Flash only: Region.box(west, south, east, north), in degrees; negative
# longitude means west. Use None for the full CONUS domain.
region: Region | None = None  # Example: Region.box(-97.0, 39.075, -95.0, 41.075)

package = StormScopeBase.load_default_package()

# GOES nowcast model: pure-obs, no external conditioning. We enable automatic
# mixed precision (autocast) and compile the model for faster inference.
model = StormScopeGOES.load_model(
    package=package,
    model_name=model_name,
    amp=True,
    compile=True,
)
model = model.to(device)
model.eval()

# MRMS+GLM nowcast model: conditioned on GOES, which we pass explicitly to
# initialize and step.
model_mrms = StormScopeMRMS.load_model(
    package=package,
    model_name=model_name,
    amp=True,
    compile=True,
)
model_mrms = model_mrms.to(device)
model_mrms.eval()

if region is not None:
    model = model.with_region(region)
    model_mrms = model_mrms.with_region(region)

# %%
# Fetch the Initial Conditions
# ----------------------------
# StormScope runs on the HRRR grid (or a crop of it), while GOES, MRMS and GLM
# each come on their own native grids. Each model's ``default_sources`` returns
# its recommended sources already regridded onto the model grid: GOES and MRMS by
# nearest neighbor, and GLM bilinearly with zero fill. Gridpoints the sources do
# not cover are NaN and masked by the model, so
# [`earth2studio.data.fetch_data`][earth2studio.data.fetch_data] returns inputs
# that match each model's input coordinates.
#
# The MRMS model's input source composes MRMS radar with gridded GLM lightning.
# Its second source provides the GOES conditioning, which is the same GOES
# history that initializes the GOES model, so we reuse that.

# %%
start_date = np.array([np.datetime64(datetime(2024, 1, 8, 18, 0, 0))])

in_coords = model.input_coords()
x = fetch_data(
    model.default_sources(),
    time=start_date,
    variable=in_coords["variable"].values,
    lead_time=in_coords["lead_time"].values,
    device=device,
)

mrms_in_coords = model_mrms.input_coords()
mrms_source, _ = model_mrms.default_sources()
x_mrms = fetch_data(
    mrms_source,
    time=start_date,
    variable=mrms_in_coords["variable"].values,
    lead_time=mrms_in_coords["lead_time"].values,
    device=device,
)

# %%
# Add Batch Dimension
# -------------------
# The models expect a batch dimension: [B, T, L, C, H, W]. Up to GPU memory limits,
# this can be increased to produce multiple ensemble members.

# %%
batch_size = 1
x = x.expand_dims(batch=np.arange(batch_size)).astype(np.float32)
x_mrms = x_mrms.expand_dims(batch=np.arange(batch_size)).astype(np.float32)

# %%
# Execute the Workflow
# --------------------
# Since the StormScope coupled inference is a bit more involved, we will use
# a custom forecast loop rather than a built-in workflow. Here, the GOES model
# predicts future satellite imagery, and the MRMS model predicts radar
# reflectivity (and GLM) conditioned on GOES (initially the observed history, then
# the forecasted GOES imagery) via explicit ``initialize`` and ``step`` calls.

# %%
y_pred, state = model.initialize(x)
y_mrms_pred, state_mrms = model_mrms.initialize(x_mrms, x)

n_steps = 2
for step_idx in trange(n_steps - 1, desc="Forecast steps"):
    y_mrms_pred, state_mrms = model_mrms.step(y_mrms_pred, y_pred, state=state_mrms)
    y_pred, state = model.step(y_pred, state=state)

# %%
# Post Processing
# ---------------
# Let's plot the final forecast step: GOES abi13c (Clean IR 10.35um) in
# grayscale with MRMS reflectivity (refc) overlaid. Forecasts are NaN outside the
# gridpoints the observations cover. For a regional view, ``crop_output`` crops
# the forecast to the requested region; the rollout itself keeps the context.

# %%
goes_channel = "abi13c"
goes_field = model.crop_output(y_pred).e2s.as_numpy().isel(batch=0, time=0, lead_time=0)
mrms_field = (
    model_mrms.crop_output(y_mrms_pred)
    .e2s.as_numpy()
    .isel(batch=0, time=0, lead_time=0)
)
lat_out, lon_out = goes_field["lat"].values, goes_field["lon"].values

# Prepare HRRR Lambert Conformal projection
proj_hrrr = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)
plt.figure(figsize=(9, 6))
ax = plt.axes(projection=proj_hrrr)

# Dual layer coast/state lines for better day/night visibility
# Black halo (thicker)
ax.coastlines(color="black", linewidth=1.2)
ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=1.0)

# White inner line (thinner)
ax.coastlines(color="white", linewidth=0.4)
ax.add_feature(cfeature.STATES, edgecolor="white", linewidth=0.3)

im = ax.pcolormesh(
    lon_out,
    lat_out,
    goes_field.sel(variable=goes_channel).values,
    transform=ccrs.PlateCarree(),
    cmap="gray_r",
    shading="auto",
)

# Overlay MRMS on top of GOES, hiding non-positive reflectivity
field_mrms = mrms_field.sel(variable="refc").values
field_mrms = np.where(field_mrms <= 0, np.nan, field_mrms)
im_mrms = ax.pcolormesh(
    lon_out,
    lat_out,
    field_mrms,
    transform=ccrs.PlateCarree(),
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=55.0,
)
plt.colorbar(
    im,
    label="GOES Clean IR 10.35um [K]",
    orientation="horizontal",
    pad=0.05,
    shrink=0.5,
)
plt.colorbar(
    im_mrms,
    label="MRMS Reflectivity [dBZ]",
    orientation="horizontal",
    pad=0.1,
    shrink=0.5,
)

time = goes_field["time"].values
lead_time = goes_field["lead_time"].values
plt.title(
    f"Predicted GOES {goes_channel} with MRMS overlay from {time} UTC "
    f"initialization (lead {lead_time.astype('timedelta64[m]')})"
)

plt.tight_layout()
plt.savefig("outputs/03_stormscope_goes_example.png", dpi=300)
