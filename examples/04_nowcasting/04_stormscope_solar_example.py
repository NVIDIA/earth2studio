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
StormScope Solar Irradiance Nowcasting
======================================

Solar irradiance (GHI) nowcasting using StormScope GOES and NSRDB models.

This example demonstrates how to generate probabilistic GHI (Global Horizontal
Irradiance) forecasts over the Continental United States using geostationary
satellite imagery as input.

In this example you will learn:

- How to instantiate StormScope GOES and NSRDB models
- Running an autoregressive GOES forecast with GHI estimation at each step
- Generating ensemble members for probabilistic forecasts
- Plotting GHI ensemble mean, spread, and individual members
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
# The solar nowcasting pipeline chains two models:
#
# - :py:class:`earth2studio.models.px.StormScopeGOES` (3-km, 10-min) predicts
#   future GOES satellite imagery (8 ABI channels) autoregressively.
# - :py:class:`earth2studio.models.px.StormScopeNSRDB` estimates surface GHI
#   from each predicted GOES frame using regression + stochastic diffusion
#   refinement (SDEdit).
#
# Both models are loaded from the same package. The 3-km GOES model runs
# without GFS conditioning, and the NSRDB model is conditioned on the
# predicted GOES imagery.

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

from earth2studio.data import GOES, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px.stormscope import (
    StormScopeGOES,
    StormScopeNSRDB,
)

# %%
# Load Models
# -----------
# Both the GOES forecast model and the GHI estimation model are loaded from the
# same package. The 3-km GOES model (``3km_goes_nowcast_3_eoe``) runs at 10-minute
# steps with no GFS conditioning (``conditioning_data_source=None``). The NSRDB
# model uses a lightweight U-Net with regression + SDEdit for probabilistic GHI.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

package = Package("/output/stormcast_v2_pkg_clean")

goes_model = StormScopeGOES.load_model(
    package=package,
    model_name="3km_goes_nowcast_3_eoe",
    conditioning_data_source=None,
)
goes_model = goes_model.to(device).eval()

nsrdb_model = StormScopeNSRDB.load_model(
    package=package,
    model_name="stormscope_solar_goes_nsrdb",
    conditioning_data_source=None,
)
nsrdb_model = nsrdb_model.to(device).eval()

nsrdb_model.build_conditioning_interpolator(
    goes_model.latitudes.cpu().numpy(),
    goes_model.longitudes.cpu().numpy(),
)

# %%
# Fetch GOES Initial Conditions
# -----------------------------
# We fetch GOES-16 CONUS data for a clear-sky afternoon case.
# The model needs 6 frames at 10-minute intervals as input context.

# %%
start_date = [np.datetime64(datetime(2024, 7, 15, 18, 0, 0))]
goes_satellite = "goes16"
scan_mode = "C"

goes = GOES(satellite=goes_satellite, scan_mode=scan_mode)
goes_lat, goes_lon = GOES.grid(satellite=goes_satellite, scan_mode=scan_mode)
goes_model.build_input_interpolator(goes_lat, goes_lon)

in_coords = goes_model.input_coords()
x, x_coords = fetch_data(
    goes,
    time=start_date,
    variable=np.array(in_coords["variable"]),
    lead_time=in_coords["lead_time"],
    device=device,
)

# %%
# Create Ensemble
# ---------------
# We replicate the initial condition along the batch dimension to generate
# multiple ensemble members. Each member will evolve differently due to the
# stochastic diffusion sampling in both the GOES and NSRDB models.

# %%
ensemble_size = 4
if x.dim() == 5:
    x = x.unsqueeze(0)
x = x.repeat(ensemble_size, 1, 1, 1, 1, 1)
x_coords["batch"] = np.arange(ensemble_size)
x_coords.move_to_end("batch", last=False)
x = x.to(dtype=torch.float32)

# %%
# Run Forecast Loop
# -----------------
# At each step, we:
#
# 1. Predict future GOES imagery with the GOES model
# 2. Estimate GHI from the predicted GOES using the NSRDB model
# 3. Advance the GOES sliding window for the next step
#
# The NSRDB model uses ``estimate_from_goes()`` which internally:
#
# - Runs a regression U-Net for a deterministic GHI estimate
# - Adds controlled noise (SDEdit) and denoises with a diffusion model
# - This produces calibrated probabilistic GHI with realistic spatial texture

# %%
y, y_coords = x, x_coords
n_steps = 6
ghi_forecasts = []
valid_times = []

for step_idx in range(n_steps):
    torch.manual_seed(step_idx)

    # Forecast GOES
    y_pred, y_pred_coords = goes_model(y, y_coords)

    # Estimate GHI from predicted GOES
    ghi_pred, ghi_coords = nsrdb_model.estimate_from_goes(
        y_pred.clone().float(), y_pred_coords.copy()
    )

    ghi_np = ghi_pred[:, 0, 0, 0].detach().cpu().numpy()
    ghi_forecasts.append(ghi_np)

    lead = y_pred_coords["lead_time"][0]
    valid_times.append(lead.astype("timedelta64[m]").item())
    print(
        f"  Step {step_idx + 1}/{n_steps}: "
        f"lead +{valid_times[-1]}min, "
        f"GHI mean={np.nanmean(ghi_np):.0f} W/m²"
    )

    # Advance GOES sliding window
    y, y_coords = goes_model.next_input(y_pred, y_pred_coords, y, y_coords)

# %%
# Plot Results
# ------------
# We create three figures:
#
# 1. Ensemble mean GHI at selected lead times
# 2. Ensemble spread (standard deviation) showing forecast uncertainty
# 3. Individual ensemble members at the final step

# %%
lat_nsrdb = nsrdb_model.latitudes.detach().cpu().numpy()
lon_nsrdb = nsrdb_model.longitudes.detach().cpu().numpy()

proj_hrrr = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

# Figure 1: Ensemble mean GHI at three lead times
steps_to_show = [0, n_steps // 2, n_steps - 1]
fig, axes = plt.subplots(
    1, len(steps_to_show), figsize=(6 * len(steps_to_show), 5),
    subplot_kw={"projection": proj_hrrr},
)

for i, si in enumerate(steps_to_show):
    ax = axes[i]
    ens_mean = np.nanmean(ghi_forecasts[si], axis=0)
    ax.coastlines(color="black", linewidth=0.8)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.4)
    im = ax.pcolormesh(
        lon_nsrdb, lat_nsrdb, ens_mean,
        transform=ccrs.PlateCarree(), cmap="hot", shading="auto",
        vmin=0, vmax=np.nanpercentile(ens_mean[np.isfinite(ens_mean)], 98),
    )
    ax.set_title(f"+{valid_times[si]} min", fontsize=12)

fig.colorbar(im, ax=axes, shrink=0.6, label="GHI [W/m²]", orientation="horizontal", pad=0.05)
fig.suptitle(
    f"StormScope Solar — Ensemble Mean GHI\n"
    f"Init: {start_date[0]} UTC, {ensemble_size} members",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig("outputs/21_stormscope_solar_ensemble_mean.png", dpi=300)
print("Saved outputs/21_stormscope_solar_ensemble_mean.png")

# Figure 2: Ensemble spread at the same lead times
fig2, axes2 = plt.subplots(
    1, len(steps_to_show), figsize=(6 * len(steps_to_show), 5),
    subplot_kw={"projection": proj_hrrr},
)

for i, si in enumerate(steps_to_show):
    ax = axes2[i]
    ens_std = np.nanstd(ghi_forecasts[si], axis=0)
    ax.coastlines(color="black", linewidth=0.8)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.4)
    im2 = ax.pcolormesh(
        lon_nsrdb, lat_nsrdb, ens_std,
        transform=ccrs.PlateCarree(), cmap="viridis", shading="auto",
        vmin=0, vmax=np.nanpercentile(ens_std[np.isfinite(ens_std)], 95),
    )
    ax.set_title(f"+{valid_times[si]} min", fontsize=12)

fig2.colorbar(im2, ax=axes2, shrink=0.6, label="GHI Spread [W/m²]", orientation="horizontal", pad=0.05)
fig2.suptitle(
    f"StormScope Solar — Ensemble Spread (Std Dev)\n"
    f"Init: {start_date[0]} UTC, {ensemble_size} members",
    fontsize=13,
)
fig2.tight_layout(rect=[0, 0.05, 1, 0.95])
fig2.savefig("outputs/21_stormscope_solar_ensemble_spread.png", dpi=300)
print("Saved outputs/21_stormscope_solar_ensemble_spread.png")

# Figure 3: Individual members at last step
fig3, axes3 = plt.subplots(
    1, ensemble_size, figsize=(5 * ensemble_size, 5),
    subplot_kw={"projection": proj_hrrr},
)
if ensemble_size == 1:
    axes3 = [axes3]

last_ghi = ghi_forecasts[-1]
vmax_mbr = np.nanpercentile(last_ghi[np.isfinite(last_ghi)], 98)

for m in range(ensemble_size):
    ax = axes3[m]
    ax.coastlines(color="black", linewidth=0.8)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.4)
    im3 = ax.pcolormesh(
        lon_nsrdb, lat_nsrdb, last_ghi[m],
        transform=ccrs.PlateCarree(), cmap="hot", shading="auto",
        vmin=0, vmax=vmax_mbr,
    )
    ax.set_title(f"Member {m + 1}", fontsize=11)

fig3.colorbar(im3, ax=axes3, shrink=0.6, label="GHI [W/m²]", orientation="horizontal", pad=0.05)
fig3.suptitle(
    f"StormScope Solar — Individual Members at +{valid_times[-1]} min\n"
    f"Init: {start_date[0]} UTC",
    fontsize=13,
)
fig3.tight_layout(rect=[0, 0.05, 1, 0.95])
fig3.savefig("outputs/21_stormscope_solar_members.png", dpi=300)
print("Saved outputs/21_stormscope_solar_members.png")
