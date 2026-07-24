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
StormCast-CONUS Score-Based Data Assimilation
=============================================

Running StormCast-CONUS with guided diffusion posterior sampling to assimilate observations.

This example demonstrates how to use the StormCast-CONUS generative model with
score-based data assimilation (SDA) over the Continental United States. Sparse
in-situ surface observations from NOAA ISD are assimilated at each forecast step
using diffusion posterior sampling (DPS) guidance.
Two forecasts are run—one without observations and one with ISD surface station
data from the central United States—to illustrate the impact of data assimilation.

In this example you will learn:

- How to load StormCast-CONUS and configure SDA parameters
- Fetching HRRR initial conditions and ISD surface observations
- Running the model iteratively with and without observation assimilation
- Comparing assimilated and non-assimilated forecasts
"""
# /// script
# dependencies = [
#   "earth2studio[data,stormcast-conus] @ git+https://github.com/NVIDIA/earth2studio.git@0.17.0",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Prognostic Model: StormCast-CONUS :py:class:`earth2studio.models.px.StormCastCONUS`
#   configured with SDA parameters.
# - Datasource (state): HRRR analysis :py:class:`earth2studio.data.HRRR`.
# - Datasource (obs): NOAA ISD surface stations :py:class:`earth2studio.data.ISD`.
#
# StormCast-CONUS extends the StormCast generative architecture to the full CONUS
# HRRR domain at 3 km resolution. When an observation DataFrame is passed to each
# generator step, diffusion posterior sampling (DPS) steers the denoising trajectory
# toward the observed values.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import timedelta

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import HRRR, ISD, fetch_data
from earth2studio.models.px import StormCastCONUS
from earth2studio.utils.coords import map_coords

# Load the default model package
package = StormCastCONUS.load_default_package()

# Configure SDA: sda_std_obs is the assumed normalised observation noise std per
# variable (lower = trust observations more).
# sda_gamma is the DPS step-size scaling (lower = stronger guidance from obs).
#
# By default the example runs on the central-US subdomain to reduce GPU memory
# requirements. For the full CONUS domain, comment out hrrr_lat_lim and
# hrrr_lon_lim below (requires ~200 GB VRAM for SDA).
hrrr_lat_lim = (273, 785)
hrrr_lon_lim = (579, 1219)
model = StormCastCONUS.load_model(
    package,
    hrrr_lat_lim=hrrr_lat_lim,  # comment out for full CONUS domain
    hrrr_lon_lim=hrrr_lon_lim,  # comment out for full CONUS domain
    num_diffusion_steps=18,
    num_sda_diffusion_steps=96,
    sda_std_obs=0.15,
    sda_gamma=1e-3,
)
model = model.to("cuda:0")

hrrr = HRRR()

# %%
# Fetch Initial Conditions
# ------------------------
# Pull HRRR analysis data for April 3rd 2025, a date that saw a major tornado
# outbreak across the central United States, and align it to the model's
# coordinate system.

# %%
init_time = np.array([np.datetime64("2025-04-03T18:00")])

x, coords = fetch_data(
    hrrr,
    time=init_time,
    variable=model.variables,
    lead_time=np.array([np.timedelta64(0, "h")]),
    device="cuda:0",
)
x, coords = map_coords(x, coords, model.input_coords())

# %%
# Run Without Observations
# ------------------------
# Step the model forward 6 hours without any observations. This uses the
# standard EDM diffusion sampler, equivalent to running StormCast-CONUS as a
# pure generative forecast model. We store only the 10-m zonal wind (``u10m``)
# used for comparison.

# %%
nsteps = 6
plot_var = "u10m"
plot_vars = ["u10m", "v10m", "t2m"]
var_idx = list(model.variables).index(plot_var)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

no_obs_fields = []
gen = model.create_generator(x.clone(), coords.copy())
x_cur, c_cur = next(gen)  # prime the generator, yields initial state (lead_time = 0 h)

for step in tqdm(range(nsteps), desc="No-obs forecast"):
    logger.info(f"Running no-obs forecast step {step + 1}/{nsteps}")
    x_cur, c_cur = gen.send(None)  # advance one hour without observations
    no_obs_fields.append(x_cur[0, 0, var_idx].cpu().numpy())

gen.close()
no_obs_fields = np.stack(no_obs_fields)  # (nsteps, H, W)

# %%
# Fetch Observations and Plot Station Locations
# ---------------------------------------------
# Fetch NOAA Integrated Surface Database (ISD) surface observations covering
# the model domain.  The bounding box is derived from the model's lat/lon grid
# so it automatically adjusts when running on a subdomain.
# We visualise the station network before running the assimilation forecast.

# %%
# ISD uses [-180, 180] longitude; model.lon is in [0, 360] — convert here.
lat_min = float(model.lat.min())
lat_max = float(model.lat.max())
lon_min = float(model.lon.min()) - 360.0
lon_max = float(model.lon.max()) - 360.0
stations = ISD.get_stations_bbox((lat_min, lon_min, lat_max, lon_max))
isd = ISD(stations=stations, time_tolerance=timedelta(minutes=15), verbose=False)

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Fetch a sample observation at the initial time to retrieve station positions
sample_df = isd(init_time, plot_vars)
station_lats = sample_df["lat"].values
station_lons = sample_df["lon"].values

plt.close("all")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
ax.set_extent(
    [lon_min - 2, lon_max + 2, lat_min - 2, lat_max + 2], crs=ccrs.PlateCarree()
)
ax.add_feature(
    cartopy.feature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="black"
)
ax.add_feature(cartopy.feature.LAND, facecolor="lightyellow")
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
ax.scatter(
    station_lons,
    station_lats,
    s=20,
    marker="x",
    transform=ccrs.PlateCarree(),
    zorder=3,
)
ax.set_title("ISD Station Locations - StormCast-CONUS Domain")
plt.savefig("outputs/03_isd_stations.jpg", dpi=150, bbox_inches="tight")

# %%
# Run Inference With Streaming Observations
# ------------------------------------------
# At each forecast step the next valid time is determined from the current
# generator state, ISD observations are fetched for that time, and the
# observation DataFrame is sent to the generator. Zero-value wind reports
# (anemometer failure) are filtered out before assimilation.

# %%
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

obs_fields = []
gen = model.create_generator(x.clone(), coords.copy())
x_cur, c_cur = next(gen)  # prime the generator, yields initial state

for step in tqdm(range(nsteps), desc="Obs forecast"):
    # Target valid time is one step ahead of the current generator state
    valid_time = np.array(
        [c_cur["time"][0] + c_cur["lead_time"][0] + np.timedelta64(1, "h")]
    )
    obs_df = isd(valid_time, plot_vars)

    # Drop zero-value wind reports (common anemometer failure mode)
    if len(obs_df) > 0:
        is_wind = obs_df["variable"].str.startswith(("u", "v"))
        zero_wind = is_wind & (obs_df["observation"].abs() < 1e-5)
        obs_df = obs_df[~zero_wind]

    obs = obs_df if len(obs_df) > 0 else None
    logger.info(
        f"Step {step + 1}/{nsteps} — {len(obs_df) if obs is not None else 0} obs"
    )
    x_cur, c_cur = gen.send(obs)  # advance one hour with observations
    obs_fields.append(x_cur[0, 0, var_idx].cpu().numpy())

gen.close()
obs_fields = np.stack(obs_fields)  # (nsteps, H, W)

# %%
# Post Processing
# ---------------
# Compare the two forecasts. The top row shows the baseline (no observations),
# the middle row shows the assimilation forecast with station locations overlaid,
# and the bottom row shows the signed difference (assimilated − baseline).

# %%
plt.close("all")

# HRRR Lambert Conformal projection
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

fig, axes = plt.subplots(
    3,
    nsteps,
    subplot_kw={"projection": projection},
    figsize=(4 * nsteps, 8),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.1, right=0.9)

vmin, vmax = -10, 10
for step in range(nsteps):
    lead_hr = step + 1
    no_obs_field = no_obs_fields[step]
    obs_field = obs_fields[step]
    diff_field = obs_field - no_obs_field

    # Row 0: No-obs forecast
    ax = axes[0, step]
    im0 = ax.pcolormesh(
        model.lon,
        model.lat,
        no_obs_field,
        transform=ccrs.PlateCarree(),
        cmap="PRGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )
    ax.set_title(f"+{lead_hr}h")

    # Row 1: With-obs forecast + station locations
    ax = axes[1, step]
    im1 = ax.pcolormesh(
        model.lon,
        model.lat,
        obs_field,
        transform=ccrs.PlateCarree(),
        cmap="PRGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(
        station_lons,
        station_lats,
        s=8,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )

    # Row 2: Difference (assimilated − baseline)
    ax = axes[2, step]
    im2 = ax.pcolormesh(
        model.lon,
        model.lat,
        diff_field,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )

for row_label, ax_row in zip(["No Obs", "Obs", "Difference"], axes[:, 0]):
    ax_row.text(
        -0.07,
        0.5,
        row_label,
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        fontsize=12,
        transform=ax_row.transAxes,
    )

plt.colorbar(im0, ax=axes[0, -1], shrink=0.6, label=f"{plot_var} (m/s)")
plt.colorbar(im1, ax=axes[1, -1], shrink=0.6, label=f"{plot_var} (m/s)")
plt.colorbar(im2, ax=axes[2, -1], shrink=0.6, label=f"{plot_var} (m/s)")

plt.tight_layout()
plt.savefig("outputs/03_stormcast_conus_sda_comparison.jpg", dpi=150)

# %%
# Ground Truth Comparison
# -----------------------
# Fetch HRRR analysis at each valid forecast time and compute the absolute error
# of both the no-obs and assimilation forecasts. This shows whether assimilation
# improves accuracy relative to the actual analysis.

# %%
truth_x, truth_coords = fetch_data(
    hrrr,
    time=init_time,
    variable=np.array([plot_var]),
    lead_time=np.array([np.timedelta64(h + 1, "h") for h in range(nsteps)]),
    device="cpu",
)
truth_x, truth_coords = map_coords(
    truth_x,
    truth_coords,
    {"hrrr_y": model.hrrr_y, "hrrr_x": model.hrrr_x},
)
# truth_x shape: (time=1, lead_time=nsteps, variable=1, H, W)
truth_fields = truth_x[0, :, 0].numpy()  # (nsteps, H, W)

no_obs_err = np.abs(no_obs_fields - truth_fields)
obs_err = np.abs(obs_fields - truth_fields)

# %%
# Plot absolute errors between the StormCast-CONUS predictions and HRRR
# analysis ground truth. In later time-steps the assimilated forecast typically
# shows lower errors near the observation stations.

# %%
plt.close("all")
fig, axes = plt.subplots(
    2,
    nsteps,
    subplot_kw={"projection": projection},
    figsize=(4 * nsteps, 6),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.1, right=0.9)

err_max = 5
for step in range(nsteps):
    lead_hr = step + 1

    # Row 0: No-obs absolute error
    ax = axes[0, step]
    im0 = ax.pcolormesh(
        model.lon,
        model.lat,
        no_obs_err[step],
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=0,
        vmax=err_max,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="grey",
        zorder=2,
    )
    ax.set_title(f"+{lead_hr}h")

    # Row 1: Obs absolute error
    ax = axes[1, step]
    im1 = ax.pcolormesh(
        model.lon,
        model.lat,
        obs_err[step],
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=0,
        vmax=err_max,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="grey",
        zorder=2,
    )

for row_label, ax_row in zip(["|No Obs − Truth|", "|Obs − Truth|"], axes[:, 0]):
    ax_row.text(
        -0.07,
        0.5,
        row_label,
        va="bottom",
        ha="center",
        rotation="vertical",
        rotation_mode="anchor",
        fontsize=12,
        transform=ax_row.transAxes,
    )

plt.colorbar(im0, ax=axes[0, -1], shrink=0.6, label=f"|Δ{plot_var}| (m/s)")
plt.colorbar(im1, ax=axes[1, -1], shrink=0.6, label=f"|Δ{plot_var}| (m/s)")
plt.tight_layout()
plt.savefig("outputs/03_stormcast_conus_sda_gt_comparison.jpg", dpi=150)
