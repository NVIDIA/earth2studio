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

This example demonstrates how to use the StormCast SDA model for convection-allowing
regional forecasts that incorporate sparse in-situ observations using diffusion posterior
sampling (DPS). Two forecasts are run—one without observations and one with ISD
surface station data to illustrate the impact of data assimilation.

In this example you will learn:

- How to load and initialise the StormCast SDA model
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
# - Datasource (obs): ISD surface stations :py:class:`earth2studio.data.ISD`.
# - Datasource (conditioning): GFS forecasts :py:class:`earth2studio.data.GFS_FX`
#   (loaded automatically by the model).
#
# StormCast SDA extends StormCast with diffusion posterior sampling (DPS) guidance,
# allowing sparse point observations to steer the generative diffusion process.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from earth2studio.data import HRRR, ISD, fetch_data
from earth2studio.models.da import StormCastSDA

# Load the default model package (downloads checkpoint from HuggingFace)
package = StormCastSDA.load_default_package()
model = StormCastSDA.load_model(package)
model = model.to("cuda:0")

# Data source for initial conditions
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

# Select the StormCast sub-grid from the full HRRR domain
x = x.sel(hrrr_y=ic["hrrr_y"], hrrr_x=ic["hrrr_x"])

# Assign 2-D lat/lon coordinate arrays from the model
x = x.assign_coords(
    lat=(["hrrr_y", "hrrr_x"], model.lat),
    lon=(["hrrr_y", "hrrr_x"], model.lon),
)

# %%
# Run Without Observations
# ------------------------
# Step the model forward 4 hours without any observations.  Each call to
# ``model(x, None)`` advances the state by one hour.  We store only the
# surface variables used for comparison (u10m, v10m, t2m).

# %%
nsteps = 4
plot_vars = ["u10m", "v10m", "t2m"]

no_obs_frames = []
gen = model.create_generator(x)
x_state = next(gen)  # Prime the generator, yields initial state

for step in range(nsteps):
    print(f"Running forecast step {step}")
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
# Fetch ISD surface observations over the CONUS domain.  At each forecast
# step, observations are fetched for the current valid time (initialisation
# time + lead time) so the model assimilates temporally relevant data.

# %%
# Get ISD stations inside the approximate StormCast HRRR bounding box
stations = ISD.get_stations_bbox((25.0, -125.0, 50.0, -65.0))

isd = ISD(stations=stations[:50], tolerance=timedelta(minutes=30), verbose=False)
init_time = datetime(2025, 1, 1)

obs_frames = []
gen = model.create_generator(x)
x_state = next(gen)  # Prime the generator, yields initial state

for step in range(nsteps):
    valid_time = init_time + timedelta(hours=step + 1)
    obs_df = isd(valid_time, plot_vars)
    print(f"Running forecast step {step} - valid {valid_time}, {len(obs_df)} obs")
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

variable = "t2m"

# Load saved forecasts from Zarr stores
no_obs_ds = xr.open_zarr("outputs/21_no_obs.zarr")
obs_ds = xr.open_zarr("outputs/21_with_obs.zarr")

no_obs_vals = (
    no_obs_ds["prediction"].sel(variable=variable).values
)  # [time, lead_time, y, x]
obs_vals = obs_ds["prediction"].sel(variable=variable).values

# Observation locations (convert from 0-360 to -180..180 for plotting)
obs_lons = obs_df["lon"].values.copy()
obs_lons = np.where(obs_lons > 180, obs_lons - 360, obs_lons)
obs_lats = obs_df["lat"].values.copy()

# Plot lon in -180..180 for PlateCarree scatter
plot_model_lon = model.lon.copy()
plot_model_lon = np.where(plot_model_lon > 180, plot_model_lon - 360, plot_model_lon)

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
    figsize=(5 * nsteps, 12),
)

for step in range(nsteps):
    lead_hr = step + 1
    no_obs_field = no_obs_vals[0, step]
    obs_field = obs_vals[0, step]
    diff_field = obs_field - no_obs_field

    vmin = min(no_obs_field.min(), obs_field.min())
    vmax = max(no_obs_field.max(), obs_field.max())

    # Row 0: No-obs forecast
    ax = axes[0, step]
    im0 = ax.pcolormesh(
        plot_model_lon,
        model.lat,
        no_obs_field,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )
    ax.set_title(f"No Obs — +{lead_hr}h")

    # Row 1: With-obs forecast + station locations
    ax = axes[1, step]
    im1 = ax.pcolormesh(
        plot_model_lon,
        model.lat,
        obs_field,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(
        obs_lons,
        obs_lats,
        s=12,
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
    ax.set_title(f"With Obs - +{lead_hr}h")

    # Row 2: Difference (assimilated - baseline)
    ax = axes[2, step]
    abs_max = max(abs(diff_field.min()), abs(diff_field.max()))
    im2 = ax.pcolormesh(
        plot_model_lon,
        model.lat,
        diff_field,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
    )
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )
    ax.set_title(f"Difference — +{lead_hr}h")

# Add colour bars
fig.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.6, label="t2m (K)")
fig.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.6, label="t2m (K)")
fig.colorbar(im2, ax=axes[2, :].tolist(), shrink=0.6, label="Δt2m (K)")

fig.suptitle("StormCast SDA — 2025-01-01 Forecast Comparison", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("outputs/21_stormcast_sda_comparison.jpg", dpi=150, bbox_inches="tight")
