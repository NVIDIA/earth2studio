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
# - [`earth2studio.models.px.StormScopeGOES`][earth2studio.models.px.StormScopeGOES] to forecast GOES channels.
# - [`earth2studio.models.px.StormScopeMRMS`][earth2studio.models.px.StormScopeMRMS] to forecast radar
#   reflectivity.
#
# In the CONUS nowcasting (``3km_10min``) configuration the GOES model is
# "pure obs" (no external conditioning), while the MRMS model is conditioned on
# GOES — the GOES model provides that conditioning during the rollout via
# ``initialize`` and ``step``. The ``3km_10min`` MRMS model
# ([`earth2studio.models.px.StormScopeMRMSGLM`][earth2studio.models.px.StormScopeMRMSGLM])
# additionally consumes gridded GLM lightning observations (``glm_density``) as a
# second forcing input. GLM is not part of the forecast state, it is supplied
# alongside the GOES conditioning at every ``initialize`` and ``step`` call.

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
import xarray as xr
from tqdm import trange

from earth2studio.data import GOES, MRMS, GOESGLMGrid, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMSGLM,
)

# %%
# We use the CONUS nowcasting variant (``3km_10min``), the recommended default.
# The GOES model is "pure obs" (no external conditioning), forecasting the eight
# ABI channels from their recent history. The MRMS model forecasts radar
# reflectivity conditioned on GOES, and additionally consumes a Geostationary
# Lightning Mapper (GLM) forcing input. The GLM field comes from
# [`earth2studio.data.GOESGLMGrid`][earth2studio.data.GOESGLMGrid], a gridded 0.1-degree lightning
# product, which we bilinearly regrid onto the model grid before passing it in.
#
# Other selectable variants (see ``StormScope*.list_available_models``):
#
# - "3km_10min": CONUS nowcasting, 3 km / 10 min (recommended, default)
# - "6km_1hr": legacy 6 km / 60 min nearcasting

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "3km_10min"

package = StormScopeBase.load_default_package()

# GOES nowcast model: pure-obs, no external conditioning source needed.
# We enable automatic mixed precision (autocast) and compile the model for faster inference.
model = StormScopeGOES.load_model(
    package=package,
    model_name=model_name,
    amp=True,
    compile=True,
)
model = model.to(device)
model.eval()

# MRMS+GLM nowcast model: conditioned on GOES and GLM. Here we drive it via
# initialize and step, and fetch the GLM observations ourselves from a gridded
# GLM source. For ``3km_10min`` ``load_model`` returns a ``StormScopeMRMSGLM``.
model_mrms: StormScopeMRMSGLM = StormScopeMRMSGLM.load_model(
    package=package,
    model_name=model_name,
    amp=True,
    compile=True,
)
model_mrms = model_mrms.to(device)
model_mrms.eval()

# %%
# Setup GOES Data Source and Interpolators
# ----------------------------------------
# We fetch GOES data for the model inputs and build interpolators that map the
# GOES grid into the StormScope model grid. StormScope operates on
# the HRRR grid, or a downsampled version of it, and for convenience each model
# defines grid coordinates `model.latitudes` and `model.longitudes` to help with
# the regridding functionality.

# %%
start_date = [np.datetime64(datetime(2024, 1, 8, 18, 0, 0))]
goes_satellite = "goes16"
scan_mode = "C"

variables = model.input_coords().coords["variable"].values
lat_out = model.latitudes.detach().cpu().numpy()
lon_out = model.longitudes.detach().cpu().numpy()

goes = GOES(satellite=goes_satellite, scan_mode=scan_mode)
goes_lat, goes_lon = GOES.grid(satellite=goes_satellite, scan_mode=scan_mode)

# The GOES nowcast model is pure-obs (no external conditioning), so only an input
# interpolator (GOES grid -> model grid) is needed.
model.build_input_interpolator(goes_lat, goes_lon)

in_coords = model.input_coords()

# Fetch GOES data (left on the native GOES grid; the model regrids internally)
x = fetch_data(
    goes,
    time=start_date,
    variable=np.array(variables),
    lead_time=in_coords.coords["lead_time"].values,
    device=device,
)

# %%
# Setup MRMS + GLM Data Sources and Assemble the Inputs
# -----------------------------------------------------
# The MRMS+GLM model forecasts the radar channels in ``model_mrms.variables``.
# These come from [`earth2studio.data.MRMS`][earth2studio.data.MRMS]; the GLM
# forcing comes from [`earth2studio.data.GOESGLMGrid`][earth2studio.data.GOESGLMGrid].
# Because we drive the rollout with ``initialize`` and ``step``, we own the
# inputs: we fetch radar and GLM and regrid each onto the shared model grid (GLM
# uses bilinear regridding, unlike the nearest-neighbor radar/satellite path).
# GLM is a forcing input, so observations are supplied at every step rather than
# predicted by the model. Model execution does not fetch observations
# automatically.

# %%
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch

mrms = MRMS()
mrms_in_coords = model_mrms.input_coords()

x_radar = fetch_data(
    mrms,
    time=start_date,
    variable=np.array(model_mrms.variables),
    lead_time=mrms_in_coords.coords["lead_time"].values,
    device=device,
)

# Interpolators: radar/GOES use nearest-neighbor; GLM is built below (bilinear).
model_mrms.build_input_interpolator(
    x_radar.coords["lat"].values, x_radar.coords["lon"].values
)
model_mrms.build_conditioning_interpolator(goes_lat, goes_lon)

# Regrid the radar channels onto the model grid (nearest-neighbor).
radar_tensor = model_mrms.input_interp(x_radar.e2s.to_torch()[0])
mrms_signature = coord_array_like(
    mrms_in_coords, {"batch": np.array([0]), "time": np.array(start_date)}
)
x_mrms = from_torch(radar_tensor.unsqueeze(0).to(dtype=torch.float32), mrms_signature)

# GLM forcing declaration: the model grid with the GLM variables. The GLM
# interpolator is built once from the gridded GLM source's lat/lon.
glm_source = GOESGLMGrid(satellite="east")
glm_coords_decl = model_mrms.forcing_coords()[1]


def fetch_glm(lead_time: np.ndarray) -> xr.DataArray:
    """Fetch GLM observations and regrid them onto the model grid."""
    glm_field = fetch_data(
        glm_source,
        time=start_date,
        variable=np.asarray(model_mrms.glm_variables),
        lead_time=lead_time,
        device=device,
    )
    glm_tensor, glm_coords = glm_field.e2s.to_torch()
    if model_mrms.glm_interp is None:
        model_mrms.build_glm_interpolator(glm_coords["lat"], glm_coords["lon"])
    glm_tensor = model_mrms.interpolate_glm(glm_tensor).to(dtype=torch.float32)
    signature = coord_array_like(
        glm_coords_decl,
        {
            "batch": np.array([0]),
            "time": np.array(start_date),
            "lead_time": lead_time,
        },
    )
    return from_torch(glm_tensor.unsqueeze(0), signature)


x_glm = fetch_glm(mrms_in_coords.coords["lead_time"].values)

# Regrid the GOES initial history once onto the declared native grid.
goes_tensor = model.input_interp(x.e2s.to_torch()[0])
goes_signature = coord_array_like(
    in_coords, {"batch": np.array([0]), "time": np.array(start_date)}
)
x = from_torch(goes_tensor.unsqueeze(0), goes_signature)

# %%
# Add Batch Dimension
# -------------------
# The models expect a batch dimension: [B, T, L, C, H, W]. Up to GPU memory limits,
# this can be increased to produce multiple ensemble members.

# %%
batch_size = 1
x = (
    x.isel(batch=0, drop=True)
    .expand_dims(batch=np.arange(batch_size))
    .astype(np.float32)
)
x_mrms = (
    x_mrms.isel(batch=0, drop=True)
    .expand_dims(batch=np.arange(batch_size))
    .astype(np.float32)
)
x_glm = (
    x_glm.isel(batch=0, drop=True)
    .expand_dims(batch=np.arange(batch_size))
    .astype(np.float32)
)

# %%
# Execute the Workflow
# --------------------
# Since the StormScope coupled inference is a bit more involved, we will use
# a custom forecast loop rather than a built-in workflow. Here, the GOES model
# predicts future satellite imagery, and the MRMS model predicts radar
# reflectivity conditioned on GOES (initially the raw data, then the forecasted
# GOES imagery) and observed GLM lightning via explicit ``initialize`` and
# ``step`` calls. Each ``step`` receives the GLM observations valid at the lead
# times of the frames just predicted.

# %%
y_pred, state = model.initialize(x)
y_mrms_pred, state_mrms = model_mrms.initialize(x_mrms, x, x_glm)

n_steps = 2
for step_idx in trange(n_steps - 1, desc="Forecast steps"):
    glm_step = fetch_glm(y_pred.coords["lead_time"].values)
    glm_step = glm_step.isel(batch=0, drop=True).expand_dims(
        batch=np.arange(batch_size)
    )
    y_mrms_pred, state_mrms = model_mrms.step(
        y_mrms_pred, y_pred, glm_step, state=state_mrms
    )
    y_pred, state = model.step(y_pred, state=state)
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
y_pred, y_pred_coords = y_pred.e2s.to_torch()
y_mrms_pred, _ = y_mrms_pred.e2s.to_torch()
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

time = y_pred_coords["time"][0].item()
lead_time = y_pred_coords["lead_time"][0]
plt.title(
    f"Predicted GOES {goes_channel} with MRMS overlay from {time} UTC "
    f"initialization (lead {lead_time.astype('timedelta64[m]').item()})"
)

plt.tight_layout()
plt.savefig("outputs/03_stormscope_goes_example.png", dpi=300)
