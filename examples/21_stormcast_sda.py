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
StormCast Score-Based Data Assimilation
=======================================

Running StormCast with diffusion posterior sampling to assimilate surface observations.

This example demonstrates how to use the StormCast score-based data assimilation (SDA)
model for convection-allowing regional forecasts that incorporate sparse in-situ
observations using diffusion posterior sampling (DPS).
Two forecasts are run—one without observations and one with ISD surface station data
from Oklahoma, United States region to illustrate the impact of data assimilation.

In this example you will learn:

- How to load and initialise the StormCast score-based data assimilation (SDA) model
- Fetching HRRR initial conditions and ISD surface observations
- Running the model iteratively with and without observation assimilation
- Comparing assimilated and non-assimilated forecasts
"""
# /// script
# dependencies = [
#   "earth2studio[da-stormcast] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: StormCast SDA :py:class:`earth2studio.models.da.StormCastSDA`.
# - Datasource (state): HRRR analysis :py:class:`earth2studio.data.HRRR`.
# - Datasource (obs): NOAA ISD surface stations :py:class:`earth2studio.data.ISD`.
# - Datasource (conditioning): GFS forecasts :py:class:`earth2studio.data.GFS_FX`
#   (loaded automatically by the model).
#
# StormCast score-based data assimilation (SDA) extends StormCast with diffusion
# posterior sampling (DPS) guidance,
# allowing sparse point observations to steer the generative diffusion process.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import HRRR, ISD, fetch_data
from earth2studio.models.da import StormCastSDA
from earth2studio.utils.coords import map_coords_xr

package = StormCastSDA.load_default_package()
# Load the model onto the GPU and configure SDA
# sda_std_obs: assumed observation noise std (lower = trust obs more)
# sda_gamma: DPS guidance scaling factor (higher = stronger assimilation)
model = StormCastSDA.load_model(package, sda_std_obs=0.1, sda_gamma=0.001)
model = model.to("cuda:0")

hrrr = HRRR()

# %%
# Fetch Initial Conditions
# ------------------------
# Pull HRRR analysis data for January 1st 2024 and select the sub-grid that
# StormCast expects. The model's :py:meth:`init_coords` describes the required
# coordinate system.

# %%
time = np.array([np.datetime64("2024-01-01T00:00")])
ic = model.init_coords()[0]

x = fetch_data(
    hrrr,
    time=time,
    variable=ic["variable"],
    lead_time=np.array([np.timedelta64(0, "h")]),
    device="cuda:0",
    legacy=False,
)
x = map_coords_xr(x, ic)

# %%
# Run Without Observations
# ------------------------
# Step the model forward 6 hours without any observations. This is equivalent to using
# the StormCast prognostic model as it will just use EDM diffusion sampling under the
# hood.  Each call to ``model.send(None)`` advances the state by one hour.
# We store only the surface variables used for comparison (u10m, v10m, t2m).

# %%
nsteps = 6
plot_vars = ["u10m", "v10m", "t2m"]

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

no_obs_frames = []
gen = model.create_generator(x.copy())
x_state = next(gen)  # Prime the generator, yields initial state

for step in tqdm(range(nsteps), desc="No-obs forecast"):
    logger.info(f"Running no-obs forecast step {step}")
    x_state = gen.send(None)  # Advance one hour without observations
    no_obs_frames.append(x_state.sel(variable=plot_vars).copy())

gen.close()
no_obs_da = xr.concat(no_obs_frames, dim="lead_time")

# Save to Zarr (convert to numpy for storage)
no_obs_np = no_obs_da.copy(data=no_obs_da.data.get())
no_obs_np.to_dataset(name="prediction").to_zarr("outputs/21_no_obs.zarr", mode="w")

# %%
# Fetch Observations and Run With Assimilation
# ---------------------------------------------
# Fetch NOAA Integrated Surface Database (ISD) surface observations from Oklahoma and
# assimilate them at each forecast step. The observations are fetched for the valid time
# (initialisation time + lead time) so the model assimilates temporally
# relevant data.

# %%
# Get ISD stations in the Oklahoma region and create the data source
stations = ISD.get_stations_bbox((32.0, -105.0, 45.0, -90.0))
isd = ISD(stations=stations, tolerance=timedelta(minutes=15), verbose=False)
init_time = datetime(2024, 1, 1)

# %%
# Plot ISD Station Locations
# --------------------------
# Visualise the ISD stations that will provide observations for assimilation.

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Fetch a sample to get station locations
sample_df = isd(init_time, ["t2m", "u10m", "v10m"])
station_lats = sample_df["lat"].values
station_lons = sample_df["lon"].values

plt.close("all")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6))
ax.set_extent([-120, -80, 30, 45], crs=ccrs.PlateCarree())
ax.add_feature(
    cartopy.feature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="black"
)
ax.add_feature(cartopy.feature.LAND, facecolor="lightyellow")
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

# Color by variable
colors = {"t2m": "red", "u10m": "blue", "v10m": "green"}
for var in sample_df["variable"].unique():
    mask = sample_df["variable"] == var
    ax.scatter(
        station_lons[mask],
        station_lats[mask],
        s=20,
        c=colors.get(var, "black"),
        label=var,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

ax.legend(loc="upper right")
ax.set_title("ISD Station Locations - Oklahoma Region")
plt.savefig("outputs/21_isd_stations.jpg", dpi=150, bbox_inches="tight")

# %%
# Run Inference With Streaming Observations
# ------------------------------------------
# At each forecast step, fetch NOAA ISD observations for the current valid time
# and send them to the model generator. The observations will be used in the SDA
# guidance term when sampling the diffusion model effectively steering the generated
# result to align with the stations data from the ISD data based.

# %%
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

obs_frames = []
gen = model.create_generator(x)
x_state = next(gen)  # Prime the generator, yields initial state

for step in tqdm(range(nsteps), desc="Obs forecast"):
    # Fetch observations for the current forecast step time frame
    valid_time = init_time + timedelta(hours=step + 1)
    obs_df = isd(valid_time, plot_vars)
    logger.info(f"Running obs forecast step {step}, {len(obs_df)} obs")
    x_state = gen.send(obs_df)  # Advance one hour with observations
    obs_frames.append(x_state.sel(variable=plot_vars).copy())

gen.close()
obs_da = xr.concat(obs_frames, dim="lead_time")

# Save to Zarr
obs_np = obs_da.copy(data=obs_da.data.get())
obs_np.to_dataset(name="prediction").to_zarr("outputs/21_with_obs.zarr", mode="w")

# %%
# Post Processing
# ---------------
# Compare the two forecasts.  The top row shows the baseline forecast (no
# observations), the middle row shows the assimilated forecast with observation
# station locations overlaid as unfilled circles, and the bottom row shows the
# difference (assimilated minus baseline).

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
variable = "u10m"

# Load saved forecasts from Zarr stores
no_obs_ds = xr.open_zarr("outputs/21_no_obs.zarr")
obs_ds = xr.open_zarr("outputs/21_with_obs.zarr")

no_obs_vals = no_obs_ds["prediction"].sel(variable=variable).values
obs_vals = obs_ds["prediction"].sel(variable=variable).values

# Lambert Conformal projection matching HRRR
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
    figsize=(5 * nsteps, 8),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.1)

for step in range(nsteps):
    lead_hr = step + 1
    no_obs_field = no_obs_vals[0, step]
    obs_field = obs_vals[0, step]
    diff_field = obs_field - no_obs_field

    vmin = -5
    vmax = 5

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
        label="Stations",
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )

    # Row 2: Difference (assimilated - baseline)
    ax = axes[2, step]
    im2 = ax.pcolormesh(
        model.lon,
        model.lat,
        diff_field,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )
    # No title for difference row

# Set row labels using fig.text (GeoAxes suppresses set_ylabel)
for row, label in enumerate(["No Obs", "Obs", "Difference"]):
    bbox = axes[row, 0].get_position()
    fig.text(
        bbox.x0 - 0.01,
        (bbox.y0 + bbox.y1) / 2,
        label,
        fontsize=12,
        va="center",
        ha="right",
        rotation=90,
    )

# Add colour bars
fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.6, label=f"{variable} (m/s)")
fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.6, label=f"{variable} (m/s)")
fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.6, label=f"{variable} (m/s)")

plt.savefig("outputs/21_stormcast_sda_comparison.jpg", dpi=150, bbox_inches="tight")

# %%
# Ground Truth Comparison
# -----------------------
# Fetch HRRR analysis (ground truth) at each valid forecast time and compute
# the absolute error of both the no-obs and obs forecasts. This shows whether
# assimilation improves accuracy relative to the actual analysis.

# %%
variable = "t2m"

# Fetch HRRR ground truth for each forecast step
truth_times = np.array([np.datetime64(init_time)])
truth = fetch_data(
    hrrr,
    time=truth_times,
    variable=np.array(plot_vars),
    lead_time=np.array([np.timedelta64(h + 1, "h") for h in range(nsteps)]),
    device="cpu",
    legacy=False,
)
ic["variable"] = np.array(plot_vars)

truth = map_coords_xr(truth, {"hrrr_y": ic["hrrr_y"], "hrrr_x": ic["hrrr_x"]})
truth_vals = truth.sel(variable=variable).values  # [nsteps, hrrr_y, hrrr_x]
no_obs_vals = no_obs_ds["prediction"].sel(variable=variable).values
obs_vals = obs_ds["prediction"].sel(variable=variable).values

# Compute absolute errors against ground truth
no_obs_err = np.abs(no_obs_vals[0] - truth_vals[0])
obs_err = np.abs(obs_vals[0] - truth_vals[0])

# %%
# Plot absolute errors between the StormCast predictions and HRRR analysis ground truth.
# In later time-steps it is clear that StormCast with SDA sampline using ISD station
# observations has improved accuracy over the vanilla stormcast prediction.

# %%
plt.close("all")
fig, axes = plt.subplots(
    2,
    nsteps,
    subplot_kw={"projection": projection},
    figsize=(5 * nsteps, 8),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.1)

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
        cmap="magma",
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
        cmap="magma",
        vmin=0,
        vmax=err_max,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="grey",
        zorder=2,
    )

# Set row labels
for row, label in enumerate(["|No Obs - Truth|", "|Obs - Truth|"]):
    bbox = axes[row, 0].get_position()
    fig.text(
        bbox.x0 - 0.01,
        (bbox.y0 + bbox.y1) / 2,
        label,
        fontsize=11,
        va="center",
        ha="right",
        rotation=90,
    )

fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.6, label=f"|Δ{variable}| (m/s)")
fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.6, label=f"|Δ{variable}| (m/s)")

plt.savefig("outputs/21_stormcast_sda_gt_comparison.jpg", dpi=150, bbox_inches="tight")
