# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
- Creating GOES and MRMS data sources
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
# - :py:class:`earth2studio.models.px.StormScopeGOES` to forecast GOES channels.
# - :py:class:`earth2studio.models.px.StormScopeMRMS` to forecast radar reflectivity.
#
# Each model also needs a conditioning data source. For GOES we use
# :py:class:`earth2studio.data.GFS_FX`, so it can be conditioned on synoptic-scale
# z500 data, and for MRMS we condition on GOES. The GOES model will provide the
# conditioning data for the MRMS model in the inference loop as the models are
# rolled out.

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

from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

# %%
# We select the proper GOES platform based on the date and build a single
# initialization timestamp. GOES-19 replaced GOES-16 (both sometimes
# referred to as GOES-East, covering the same CONUS domain) in April 2025.
# Choose pre-trained model names and load them with their conditioning sources.
#
# Model options:
#
# - "6km_60min_natten_cos_zenith_input_eoe_v2" for 1hr timestep GOES model
# - "6km_10min_natten_pure_obs_zenith_6steps" for 10min timestep GOES model
# - "6km_60min_natten_cos_zenith_input_mrms_eoe" for 1hr timestep MRMS model
# - "6km_10min_natten_pure_obs_mrms_obs_6steps" for 10min timestep MRMS model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

goes_model_name = "6km_60min_natten_cos_zenith_input_eoe_v2"
mrms_model_name = "6km_60min_natten_cos_zenith_input_mrms_eoe"

package = StormScopeBase.load_default_package()

# Load GOES model with GFS_FX conditioning (should be set to None for 10min models)
model = StormScopeGOES.load_model(
    package=package,
    conditioning_data_source=GFS_FX(),
    model_name=goes_model_name,
)
model = model.to(device)
model.eval()

# Load MRMS model with GOES conditioning (should be set to None for 10min models)
model_mrms = StormScopeMRMS.load_model(
    package=package,
    conditioning_data_source=GOES(),
    model_name=mrms_model_name,
)
model_mrms = model_mrms.to(device)
model_mrms.eval()

# %%
# Setup GOES Data Source and Interpolators
# ----------------------------------------
# We fetch GOES data for the model inputs and build interpolators that map the
# GOES grid and GFS grid into the StormScope model grid. StormScope operates on
# the HRRR grid, or a downsampled version of it, and for convenience each model
# defines grid coordinates `model.latitudes` and `model.longitudes` to help with
# the regridding functionality.

# %%
start_date = [np.datetime64(datetime(2023, 12, 5, 12, 00, 0))]
goes_satellite = "goes16"
scan_mode = "C"

variables = model.input_coords()["variable"]
lat_out = model.latitudes.detach().cpu().numpy()
lon_out = model.longitudes.detach().cpu().numpy()

goes = GOES(satellite=goes_satellite, scan_mode=scan_mode)
goes_lat, goes_lon = GOES.grid(satellite=goes_satellite, scan_mode=scan_mode)

# Build interpolators for transforming data to model grid
model.build_input_interpolator(goes_lat, goes_lon)
model.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON)

in_coords = model.input_coords()

# Fetch GOES data
x, x_coords = fetch_data(
    goes,
    time=start_date,
    variable=np.array(variables),
    lead_time=in_coords["lead_time"],
    device=device,
)

# %%
# Setup MRMS Data Source and Interpolators
# ----------------------------------------
# MRMS inputs are fetched and interpolated to the model grid. The MRMS model is
# conditioned on GOES, so we also build the GOES conditioning interpolator.

# %%
mrms = MRMS()
mrms_in_coords = model_mrms.input_coords()
x_mrms, x_coords_mrms = fetch_data(
    mrms,
    time=start_date,
    variable=np.array(["refc"]),
    lead_time=mrms_in_coords["lead_time"],
    device=device,
)

model_mrms.build_input_interpolator(x_coords_mrms["lat"], x_coords_mrms["lon"])
model_mrms.build_conditioning_interpolator(goes_lat, goes_lon)

# %%
# Add Batch Dimension
# -------------------
# The models expect a batch dimension: [B, T, L, C, H, W]. Up to GPU memory limits,
# this can be increased to produce multiple ensemble members.

# %%
batch_size = 1
if x.dim() == 5:
    x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    x_coords["batch"] = np.arange(batch_size)
    x_coords.move_to_end("batch", last=False)
if x_mrms.dim() == 5:
    x_mrms = x_mrms.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    x_coords_mrms["batch"] = np.arange(batch_size)
    x_coords_mrms.move_to_end("batch", last=False)

x = x.to(dtype=torch.float32)
x_mrms = x_mrms.to(dtype=torch.float32)

# %%
# Execute the Workflow
# --------------------
# Since the StormScope coupled inference is a bit more involved, we will use
# a custom forecast loop rather than a bilt-in workflow. Here, the GOES model
# predicts future satellite imagery, and the MRMS model predicts radar
# reflectivity conditioned on GOES (initially the raw data, then the forecasted
# GOES imagery) via `call_with_conditioning`.

# %%
y, y_coords = x, x_coords
y_mrms, y_coords_mrms = x_mrms, x_coords_mrms

n_steps = 2
for step_idx in range(n_steps):
    # Run one prognostic step with the GOES model
    y_pred, y_pred_coords = model(y, y_coords)

    # Run one prognostic step with the MRMS model conditioned on GOES
    y_mrms_pred, y_coords_mrms_pred = model_mrms.call_with_conditioning(
        y_mrms, y_coords_mrms, conditioning=y, conditioning_coords=y_coords
    )

    # Update sliding window with new prediction
    y_pred, y_pred_coords = model.next_input(y_pred, y_pred_coords, y, y_coords)
    y_mrms_pred, y_coords_mrms_pred = model_mrms.next_input(
        y_mrms_pred, y_coords_mrms_pred, y_mrms, y_coords_mrms
    )

    # Update the input tensors and coordinate systems for the next step
    y = y_pred
    y_coords = y_pred_coords
    y_mrms = y_mrms_pred
    y_coords_mrms = y_coords_mrms_pred

# %%
# Post Processing
# ---------------
# Let's plot the final forecast step: GOES abi13c (Clean IR 10.35um) in
# grayscale with MRMS reflectivity (refc) overlaid.

# %%
goes_channel = "abi13c"
goes_ch_idx = list(model.variables).index(goes_channel)
mrms_ch_idx = list(model_mrms.variables).index("refc")

# Nan-fill invalid gridpoints
y_pred = torch.where(model.valid_mask, y_pred, torch.nan)
y_mrms_pred = torch.where(model_mrms.valid_mask, y_mrms_pred, torch.nan)

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

field = y_pred[0, 0, 0, goes_ch_idx].detach().cpu().numpy()
im = ax.pcolormesh(
    lon_out,
    lat_out,
    field,
    transform=ccrs.PlateCarree(),
    cmap="gray_r",
    shading="auto",
)

# Overlay MRMS on top of GOES
field_mrms = y_mrms_pred[0, 0, 0, mrms_ch_idx]
field_mrms = (
    torch.where(~model.valid_mask, torch.nan, field_mrms).detach().cpu().numpy()
)
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

time = y_coords["time"][0].item()
lead_time = y_coords["lead_time"][0]
plt.title(
    f"Predicted GOES {goes_channel} with MRMS overlay from {time} UTC "
    f"initialization (lead {lead_time.astype('timedelta64[m]').item()})"
)

plt.tight_layout()
plt.savefig("outputs/20_stormscope_goes_example.png", dpi=300)
