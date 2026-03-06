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
sampling (DPS). Two forecasts are run—one without observations and one with a 5x5 grid
of synthetic surface observations to illustrate the impact of data assimilation.

In this example you will learn:

- How to load and initialise the StormCast SDA model
- Fetching HRRR initial conditions and creating synthetic observations
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
# - Observations: Synthetic surface observations (5x5 grid centered on Oklahoma).
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
import pandas as pd
import torch
import xarray as xr

from earth2studio.data import HRRR, fetch_data
from earth2studio.models.da import StormCastSDA
from earth2studio.utils.coords import map_coords_xr

# Load the default model package (downloads checkpoint from HuggingFace)
package = StormCastSDA.load_default_package()
model = StormCastSDA.load_model(package, sda_std_y=0.5, sda_gamma=0.05)
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
x = map_coords_xr(x, ic)

# %%
# Run Without Observations
# ------------------------
# Step the model forward 4 hours without any observations.  Each call to
# ``model.send(None)`` advances the state by one hour.  We store only the
# surface variables used for comparison (u10m, v10m, t2m).

# %%
nsteps = 4
plot_vars = ["u10m", "v10m", "t2m"]

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

no_obs_frames = []
gen = model.create_generator(x.copy())
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
# Create a 5x5 grid of synthetic observations centered on Oklahoma (35N, 98W)
# with wind speed that increases each time step. At each forecast step,
# observations are provided for the current valid time (initialisation
# time + lead time) so the model assimilates temporally relevant data.

# %%
# Create a 5x5 grid of observation stations centered on Oklahoma (35N, 98W)
center_lat = 40.0
center_lon = -98.0
grid_spacing = 1.0  # degrees

# Create 5x5 grid of stations
grid_size = 5
lats = np.linspace(
    center_lat - (grid_size - 1) * grid_spacing / 2,
    center_lat + (grid_size - 1) * grid_spacing / 2,
    grid_size,
)
lons = np.linspace(
    center_lon - (grid_size - 1) * grid_spacing / 2,
    center_lon + (grid_size - 1) * grid_spacing / 2,
    grid_size,
)

# Create all combinations of lat/lon for the grid
obs_lats, obs_lons = np.meshgrid(lats, lons, indexing="ij")
obs_lats = obs_lats.flatten()
obs_lons = obs_lons.flatten()

init_time = datetime(2024, 1, 1)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %%
# Run inference loop now with streaming observations every forecast step

# %%

obs_frames = []
gen = model.create_generator(x)
x_state = next(gen)  # Prime the generator, yields initial state

for step in range(nsteps):
    valid_time = init_time + timedelta(hours=step + 1)
    # Wind speed increases by 1 m/s each time step, starting at 5 m/s
    ws10m_value = -5.0

    # Create synthetic observation DataFrame for all 25 stations
    obs_df = pd.DataFrame(
        {
            "lat": obs_lats.tolist(),
            "lon": obs_lons.tolist(),
            "variable": ["u10m"] * len(obs_lats),
            "observation": [ws10m_value] * len(obs_lats),
            "time": [valid_time] * len(obs_lats),
        }
    )

    print(
        f"Running forecast step {step} - valid {valid_time}, {len(obs_df)} obs, u10m={ws10m_value:.1f} m/s"
    )
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
        obs_lons,
        obs_lats,
        s=30,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=3,
        label="Observations",
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

fig.suptitle("StormCast SDA 2024-01-01 Forecast Comparison", fontsize=16, y=1.0)
plt.savefig("outputs/21_stormcast_sda_comparison.jpg", dpi=150, bbox_inches="tight")
