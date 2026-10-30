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
============================================

Running StormCast-CONUS with guided diffusion posterior sampling to assimilate observations.

This example demonstrates how to use the StormCastConusSDA generative model with
score-based data assimilation (SDA) over a CONUS subdomain. Sparse
in-situ surface observations from NOAA Global Historical Climatology Network Hourly
(GHCNh) are assimilated at each forecast step using diffusion posterior sampling
(DPS) guidance. Two forecasts from the central United States are run to illustrate the impact of
data assimilation: one without observations and one with GHCNh surface station data.

In this example you will learn:

- How to load StormCastConusSDA and configure SDA parameters
- Fetching HRRR initial conditions, GFS conditioning, and GHCNh surface observations
- Running the model iteratively with and without observation assimilation
- Comparing assimilated and non-assimilated forecasts
"""

# /// script
# dependencies = [
#   "earth2studio[data,stormcast-conus] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
#   "scipy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: :py:class:`earth2studio.models.da.StormCastConusSDA`
#   configured with SDA parameters.
# - Datasource (state): HRRR analysis [`earth2studio.data.HRRR`][earth2studio.data.HRRR].
# - Datasource (conditioning): :py:class:`earth2studio.data.GFS_FX`.
# - Datasource (obs): NOAA GHCNh surface stations
#   [`earth2studio.data.GHCNHourly`][earth2studio.data.GHCNHourly].
#
# StormCast-CONUS uses the HRRR domain at 3 km resolution. When an observation
# DataFrame is passed to each step, diffusion posterior sampling (DPS) steers the
# denoising trajectory toward the observed values.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import timedelta

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import GFS_FX, HRRR, GHCNHourly, fetch_data
from earth2studio.models.da import StormCastConusSDA
from earth2studio.run import _map_field

# Load the default model package
package = StormCastConusSDA.load_default_package()

# Configure SDA: sda_std_obs is the assumed normalised observation noise std per
# variable (lower = trust observations more).
# sda_gamma is the DPS step-size scaling (lower = stronger guidance from obs).
#
# Crop to the central-US subdomain to reduce memory. Omit the limits to use CONUS.
model = StormCastConusSDA.load_model(
    package,
    hrrr_lat_lim=(273, 785),
    hrrr_lon_lim=(579, 1219),
    num_diffusion_steps=18,
    num_sda_diffusion_steps=96,
    time_tolerance=np.timedelta64(10, "m"),
    sda_std_obs=0.15,
    sda_gamma=1e-3,
)
model = model.to("cuda:0")

hrrr = HRRR()

# %%
# Fetch Initial Conditions
# ------------------------
# Pull HRRR analysis data for 17 April 2026, a date that saw a significant tornado
# outbreak across the central United States, and align it to the model's
# coordinate system.

# %%
init_time = np.array([np.datetime64("2026-04-17T18:00")])

x = fetch_data(
    hrrr,
    time=init_time,
    variable=model.input_coords()["variable"].values,
    lead_time=np.array([np.timedelta64(0, "h")]),
    device="cuda:0",
)
# Fetch preserves the source grid. Select native labels explicitly.
input_signature = model.input_coords()
x = x.rename(hrrr_y="y", hrrr_x="x")
x = x.sel(y=input_signature.y, x=input_signature.x)
# Subsetting a projected grid changes inverse-projection roundoff (~1e-13 degrees).
for coordinate in ("lat", "lon"):
    np.testing.assert_allclose(
        x[coordinate], input_signature[coordinate], atol=1e-10, rtol=0
    )
x = x.assign_coords(lat=input_signature.lat, lon=input_signature.lon)
x = _map_field(
    x,
    input_signature,
)

# %%
# Caller-Supplied Conditioning
# ---------------------------
# StormCast-CONUS requires conditioning at the output valid time, including
# when observations are omitted. Fetch GFS from the same forecast initialization
# and interpolate its regular lat/lon fields onto the declared HRRR grid.
# Observations are fetched at that same output valid time.

# %%
conditioning_source = GFS_FX(verbose=False)
conditioning_signature, _ = model.forcing_coords()


def fetch_conditioning(lead_hour: int) -> xr.DataArray:
    """Fetch and align conditioning for one StormCast-CONUS output valid time."""
    lead_time = np.array([np.timedelta64(lead_hour, "h")])
    coarse = fetch_data(
        conditioning_source,
        time=init_time,
        variable=conditioning_signature["variable"].values,
        lead_time=lead_time,
        device="cpu",
    )
    regional = coarse.interp(
        lat=conditioning_signature.lat, lon=conditioning_signature.lon
    )
    dims = tuple(dim for dim in conditioning_signature.dims if dim != "batch")
    regional = regional.transpose(*dims)
    coords = dict(conditioning_signature.coords)
    coords.pop("batch", None)
    coords.update(time=init_time, lead_time=lead_time)
    return xr.DataArray(
        regional.values, dims=dims, coords=coords, attrs=conditioning_signature.attrs
    )


# %%
# Run Without Observations
# ------------------------
# Step the model forward 6 hours without any observations. This uses the
# diffusion sampler without observation guidance, running StormCast as a
# pure generative forecast model. We store only the 10-m zonal wind (``u10m``)
# used for comparison.

# %%
nsteps = 6
plot_var = "u10m"
plot_vars = ["u10m", "v10m", "t2m"]
var_idx = list(input_signature["variable"].values).index(plot_var)

model.set_rng(42)

no_obs_fields = []
iterator = model.create_iterator(x.copy(deep=True), fetch_conditioning(1), None)

for step in tqdm(range(nsteps), desc="No-obs forecast"):
    logger.info(f"Running no-obs forecast step {step + 1}/{nsteps}")
    if step == 0:
        x_cur = next(iterator)  # First computed output at +1 h
    else:
        x_cur = iterator.send((fetch_conditioning(step + 1), None))
    no_obs_fields.append(x_cur[0, 0, var_idx].e2s.as_numpy().values)

iterator.close()
no_obs_fields = np.stack(no_obs_fields)  # (nsteps, H, W)

# %%
# Fetch Observations and Plot Station Locations
# ---------------------------------------------
# Fetch GHCNh surface observations covering the model domain. The bounding box
# is derived from the model's lat/lon grid so it automatically adjusts when
# running on a subdomain. We visualise the station network before running the
# assimilation forecast.

# %%
# Model lon is in [0, 360); convert to [-180, 180) for the western-hemisphere
# model domain. GHCNHourly.get_stations_bbox accepts either convention.
lat_min = float(input_signature.lat.min())
lat_max = float(input_signature.lat.max())
lon_min = float(input_signature.lon.min()) - 360.0
lon_max = float(input_signature.lon.max()) - 360.0
stations = GHCNHourly.get_stations_bbox((lat_min, lon_min, lat_max, lon_max))
ghcn = GHCNHourly(
    stations=stations, time_tolerance=timedelta(minutes=15), verbose=False
)

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Fetch a sample observation at the initial time to retrieve station positions.
# GHCNh returns lon in [0, 360); shift to [-180, 180) for PlateCarree plots.
sample_df = ghcn(init_time, plot_vars)
station_lats = sample_df["lat"].values
station_lons = sample_df["lon"].values - 360.0

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
ax.set_title("GHCNh Station Locations - StormCast-CONUS Domain")
plt.savefig("outputs/01_ghcn_stations.jpg", dpi=150, bbox_inches="tight")

# %%
# Run Inference With Streaming Observations
# -----------------------------------------
# At each forecast step GHCNh observations are fetched for the output valid time.
# Initialize with the first conditioning/observation pair; send later pairs to
# continue the trajectory. The first yield is an analysis, not the input state.
# Zero-value wind reports (likely anemometer failure) are filtered out before
# assimilation.

# %%
model.set_rng(42)

obs_fields = []

for step in tqdm(range(nsteps), desc="Obs forecast"):
    # Observations and conditioning share the output valid time.
    valid_time = init_time + np.timedelta64(step + 1, "h")
    obs_df = ghcn(valid_time, plot_vars)

    # Drop zero-value wind reports (common anemometer failure mode)
    if len(obs_df) > 0:
        is_wind = obs_df["variable"].str.startswith(("u", "v"))
        zero_wind = is_wind & (obs_df["observation"].abs() < 1e-5)
        obs_df = obs_df[~zero_wind]

    obs = obs_df if len(obs_df) > 0 else None
    logger.info(
        f"Step {step + 1}/{nsteps} — {len(obs_df) if obs is not None else 0} obs"
    )
    conditioning = fetch_conditioning(step + 1)
    if step == 0:
        iterator = model.create_iterator(x.copy(deep=True), conditioning, obs)
        x_cur = next(iterator)  # Computes and yields the +1 h analysis
    else:
        x_cur = iterator.send((conditioning, obs))
    obs_fields.append(x_cur[0, 0, var_idx].e2s.as_numpy().values)

iterator.close()
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
        input_signature.lon,
        input_signature.lat,
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
        input_signature.lon,
        input_signature.lat,
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
        input_signature.lon,
        input_signature.lat,
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
plt.savefig("outputs/01_stormcast_conus_sda_comparison.jpg", dpi=150)

# %%
# Ground Truth Comparison
# -----------------------
# Fetch HRRR analysis at each valid forecast time and compute the absolute error
# of both the no-obs and assimilation forecasts. This shows whether assimilation
# improves accuracy relative to the actual analysis.

# %%
truth_x = fetch_data(
    hrrr,
    time=init_time,
    variable=np.array([plot_var]),
    lead_time=np.array([np.timedelta64(h + 1, "h") for h in range(nsteps)]),
    device="cpu",
)
truth_signature = model.input_coords().sel(variable=[plot_var])
truth_x = truth_x.rename(hrrr_y="y", hrrr_x="x")
truth_x = truth_x.sel(y=input_signature.y, x=input_signature.x)
for coordinate in ("lat", "lon"):
    np.testing.assert_allclose(
        truth_x[coordinate], input_signature[coordinate], atol=1e-10, rtol=0
    )
truth_x = truth_x.assign_coords(lat=input_signature.lat, lon=input_signature.lon)
truth_x = _map_field(
    truth_x,
    truth_signature,
)
# truth_x shape: (time=1, lead_time=nsteps, variable=1, H, W)
truth_fields = truth_x[0, :, 0].values  # (nsteps, H, W)

no_obs_err = np.abs(no_obs_fields - truth_fields)
obs_err = np.abs(obs_fields - truth_fields)

# %%
# Plot absolute errors between the StormCast predictions and HRRR
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
        input_signature.lon,
        input_signature.lat,
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
        input_signature.lon,
        input_signature.lat,
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
plt.savefig("outputs/01_stormcast_conus_sda_gt_comparison.jpg", dpi=150)
