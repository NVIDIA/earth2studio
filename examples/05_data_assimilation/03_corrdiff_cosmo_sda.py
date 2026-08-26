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
CorrDiff COSMO-REA2 Score-Based Data Assimilation
=================================================

Assimilate weather-station observations into COSMO-REA2 with CorrDiff.

This example covers a dense-station region spanning the Netherlands, NW Germany,
and the adjacent North Sea. Given an ERA5 driving state for a historical time and
sparse GHCNHourly 10 m wind reports, diffusion posterior sampling (DPS) steers the
diffusion downscaler's denoising trajectory toward the observations, producing a
high-resolution COSMO-REA analysis guided toward what the stations actually
measured.

CorrDiff-COSMO is a single-shot *diagnostic* downscaler. It does not propagate
state between times; each call produces an independent analysis conditioned on
the ERA5 input for one time.

A random subset of stations is assimilated and the analysis is compared to the
*held-out* stations. This single case illustrates observation behavior at the
held-out sites -- it is not a statistical skill assessment, and one time / split /
ensemble does not guarantee improvement.

In this example you will learn:

- Load ``CorrDiffCosmoEra5SDA`` on a COSMO-REA2 sub-domain
- Fetch an ERA5 driving state (ARCO) and regrid it onto the downscaler's input grid
- Fetch GHCNHourly 10 m wind observations over the model domain
- Produce a prior (free, no-obs) downscaling and an observation-guided analysis
- Compare both fields to held-out stations (prior vs analysis RMSE)

!!! note
    The default ~206 x 206-cell sub-domain completed in a few minutes and used about
    5 GB of GPU memory in one test run (bf16). Set ``DOMAIN = None`` for the full domain;
    its time and memory requirements were not validated.

"""

# /// script
# dependencies = [
#   "earth2studio[data,da-cosmo] @ git+https://github.com/NVIDIA/earth2studio.git@0.18.0",
#   # PhysicsNeMo's RoPE/NATTEN backend is not on PyPI yet; pin the Git source
#   # explicitly -- the repo's [tool.uv.sources] pin does not reach a PEP 723 script.
#   "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@bf0ad4f43275b84a7beab35127d57a99cd359260",
#   "cartopy",
#   "matplotlib",
#   "scipy",
# ]
# ///

# %%
# Set Up
# ------

# %% tags=["e2sg-profile:setup"]
import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

INIT_TIME = datetime(2024, 1, 1, 0)
ASSIMILATE = ("u10m", "v10m")
# Set to None for the full COSMO-REA2 domain.
DOMAIN = dict(lat_min=50.2, lat_max=53.8, lon_min=4.6, lon_max=10.4)
ENSEMBLE_SIZE = 1
SAMPLER_STEPS = 12  # Reduced from 18 for this example.
SDA_STD_OBS = 0.5
SDA_GAMMA = 5e-5
VAL_FRAC = 0.3
OBS_TIME_TOLERANCE = timedelta(minutes=30)
DEVICE = "cuda:0"
DATA = ccrs.PlateCarree()
PROJ = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=40.0)

os.makedirs("outputs", exist_ok=True)


# %%
# Load Assimilation Model
# ------------------------
# ``CorrDiffCosmoEra5SDA`` wraps a diffusion-mode ``CorrDiffCosmoEra5`` downscaler.
# ``assimilate_variables`` is required and must be identity-transform, unit-scale
# output channels -- here the 10 m wind, which matches the station reports' height.

# %% tags=["e2sg-profile:setup"]
from earth2studio.data import ARCO, GHCNHourly, fetch_data
from earth2studio.models.da import CorrDiffCosmoEra5SDA

package = CorrDiffCosmoEra5SDA.load_default_package()

sda = CorrDiffCosmoEra5SDA.load_model(
    package,
    assimilate_variables=ASSIMILATE,
    resolution="rea2",
    domain=DOMAIN,
    time_tolerance=OBS_TIME_TOLERANCE,
    number_of_samples=ENSEMBLE_SIZE,
    sampler_steps=SAMPLER_STEPS,
    sda_std_obs=SDA_STD_OBS,
    sda_gamma=SDA_GAMMA,
    amp=True,
).to(DEVICE)
sda.seed = 0

# %%
# Fetch Coarse-Resolution State
# ------------------------------
# Fetch an ERA5 analysis (ARCO) for the historical time and regrid it onto the
# downscaler's regional input grid. The result is an ``xr.DataArray`` with dims
# ``(time, variable, lat, lon)`` -- the same driving state the downscaler
# conditions on; the ``time`` coord also drives its day/night (solar) input.


# %% tags=["e2sg-profile:setup"]
def regrid_to_input(x_src, src_coords, dvars, dlat, dlon):
    """Subset to the downscaler's ERA5 variables and bilinearly regrid a global
    regular lat/lon field onto its regional grid. Returns [n_var, n_lat, n_lon]."""
    svars = list(src_coords["variable"])
    slat = np.asarray(src_coords["lat"]).astype(float)
    slon = np.asarray(src_coords["lon"]).astype(float)
    field = x_src.reshape(-1, len(svars), len(slat), len(slon))[0].float().cpu().numpy()
    field = field[[svars.index(v) for v in dvars]]  # select the ERA5 channels
    if slat[0] > slat[-1]:  # ensure ascending latitude
        slat, field = slat[::-1], field[:, ::-1, :]
    field_w = np.concatenate([field, field[:, :, 0:1]], axis=-1)  # lon wrap column
    slon_w = np.concatenate([slon, [slon[0] + 360.0]])
    lon2d, lat2d = np.meshgrid(dlon % 360.0, dlat)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    out = np.empty((len(dvars), len(dlat), len(dlon)), np.float32)
    for c in range(len(dvars)):
        out[c] = RegularGridInterpolator(
            (slat, slon_w), field_w[c], bounds_error=False, fill_value=None
        )(pts).reshape(len(dlat), len(dlon))
    return out


ic = sda.init_coords()[0]
dvars = list(ic["variable"])
dlat, dlon = np.asarray(ic["lat"]), np.asarray(ic["lon"])

t = np.array([np.datetime64(INIT_TIME)])
x_src, c_src = fetch_data(
    ARCO(),
    time=t,
    variable=np.array(dvars),
    lead_time=np.array([np.timedelta64(0, "h")]),
    device=DEVICE,
)
era5 = regrid_to_input(x_src, c_src, dvars, dlat, dlon)
x_da = xr.DataArray(
    data=era5[None],
    dims=["time", "variable", "lat", "lon"],
    coords={"time": t, "variable": np.array(dvars), "lat": dlat, "lon": dlon},
)

# %%
# Fetch Observations
# ------------------
# Fetch paired wind reports and split stations into assimilated and held-out sets.

# %% tags=["e2sg-profile:setup"]
glat = sda.model.lat_output_numpy
glon = sda.model.lon_output_numpy
bbox = (float(glat.min()), float(glon.min()), float(glat.max()), float(glon.max()))

stations = GHCNHourly.get_stations_bbox(bbox)
ghcn = GHCNHourly(stations=stations, time_tolerance=OBS_TIME_TOLERANCE, verbose=False)
raw = ghcn(INIT_TIME, list(ASSIMILATE))
raw = raw[raw["variable"].isin(ASSIMILATE)].dropna(subset=["observation"]).copy()
raw["dt"] = (pd.to_datetime(raw["time"]) - INIT_TIME).abs()
raw = raw.sort_values("dt").drop_duplicates(["station", "variable"], keep="first")
complete_stations = raw.groupby("station")["variable"].nunique().eq(len(ASSIMILATE))
raw = raw[raw["station"].isin(complete_stations[complete_stations].index)]
station_ids = sorted(raw["station"].unique())
print(f"Stations with {'+'.join(ASSIMILATE)} at {INIT_TIME}: {len(station_ids)}")
if len(station_ids) < 2:
    raise RuntimeError(
        f"Need at least 2 usable stations (got {len(station_ids)}); widen DOMAIN, "
        "loosen OBS_TIME_TOLERANCE, or choose another time."
    )

rng = np.random.default_rng(0)
n_val = min(len(station_ids) - 1, max(1, int(round(len(station_ids) * VAL_FRAC))))
val_ids = set(rng.choice(station_ids, size=n_val, replace=False))
obs_cols = ["time", "lat", "lon", "variable", "observation"]
is_held_out = raw["station"].isin(val_ids)
assimilated_reports = raw[~is_held_out]
held_out_reports = raw[is_held_out]
assimilation_obs = assimilated_reports[obs_cols]
n_assim = len(station_ids) - len(val_ids)
print(f"Assimilated: {n_assim} stations | Held-out: {len(val_ids)} stations")

# %%
# Prior vs. Observation-Guided Analysis
# -------------------------------------
# Run the downscaler twice on the same ERA5 state: once with no observations (the
# prior, ``obs=None``) and once with the observations selected for assimilation.
# Both return an ensemble with dims ``(time, sample, variable, y, x)``.


# %% tags=["e2sg-profile:inference"]
def to_numpy(da):
    """Return the DataArray data as a NumPy array."""
    arr = da.data
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


post = sda(x_da, assimilation_obs)  # analysis (observation-guided)
free = sda(x_da)  # prior (free, no-obs downscaling)

# Locate the wind-component channels and output grid used for evaluation and plotting.
output_variables = list(post["variable"].values)
u_index = output_variables.index("u10m")
v_index = output_variables.index("v10m")
output_lat = np.asarray(post["lat"])
output_lon = np.asarray(post["lon"])

post_np, free_np = to_numpy(post), to_numpy(free)  # [time, sample, variable, y, x]

# %%
# Single Posterior Sample
# -----------------------
# Compare the observation-guided analysis with the prior using 10 m wind speed and
# vector-wind RMSE.

# %% tags=["e2sg-profile:plotting"]
ws_post = np.hypot(post_np[0, :, u_index], post_np[0, :, v_index]).mean(0)
ws_free = np.hypot(free_np[0, :, u_index], free_np[0, :, v_index]).mean(0)
analysis_u = post_np[0, :, u_index].mean(0)
analysis_v = post_np[0, :, v_index].mean(0)
prior_u = free_np[0, :, u_index].mean(0)
prior_v = free_np[0, :, v_index].mean(0)
print(
    f"Finite values: Posterior={np.isfinite(post_np).all()}, "
    f"Prior={np.isfinite(free_np).all()}"
)

# %%
# Compare to Held-Out Stations
# ----------------------------
# Compute the vector-wind RMSE of each analysis against the observations, at the
# stations' nearest output cells. The *held-out* stations were never assimilated,
# so their prior-vs-posterior RMSE illustrates how the analysis behaves at
# unassimilated sites for this single case -- it is a diagnostic, not a statistical
# skill claim, and a single time/split/ensemble does not guarantee improvement.


# %% tags=["e2sg-profile:plotting"]
def vector_rmse_at(df, u_field, v_field):
    """Compute vector wind RMSE at station locations."""
    u_reports = df[df["variable"] == "u10m"][["station", "lat", "lon", "observation"]]
    v_reports = df[df["variable"] == "v10m"][["station", "observation"]]
    merged = u_reports.merge(v_reports, on="station", suffixes=("_u", "_v"))
    if not len(merged):
        return float("nan")
    tree = cKDTree(np.column_stack([output_lat.ravel(), (output_lon % 360).ravel()]))
    _, flat = tree.query(np.column_stack([merged["lat"], merged["lon"] % 360]))
    bi, bj = np.unravel_index(flat, output_lat.shape)
    u_error = u_field[bi, bj] - merged["observation_u"].values
    v_error = v_field[bi, bj] - merged["observation_v"].values
    return float(np.sqrt(np.mean(u_error**2 + v_error**2)))


held_out_prior = vector_rmse_at(held_out_reports, prior_u, prior_v)
held_out_analysis = vector_rmse_at(held_out_reports, analysis_u, analysis_v)
assimilated_prior = vector_rmse_at(assimilated_reports, prior_u, prior_v)
assimilated_analysis = vector_rmse_at(assimilated_reports, analysis_u, analysis_v)
print("Vector 10 m wind RMSE vs. GHCNHourly (m/s)")
print(f"Held-out: Prior={held_out_prior:.3f}, Analysis={held_out_analysis:.3f}")
print(
    f"Assimilated: Prior={assimilated_prior:.3f}, Analysis={assimilated_analysis:.3f}"
)

# %%
# Plot the Analyses and Their Difference
# --------------------------------------
# Finally, we can plot the results:
#
# - Left: prior 10 m wind speed.
# - Center: observation-guided analysis with assimilated and held-out stations.
# - Right: signed analysis increment.


# %% tags=["e2sg-profile:plotting"]
title_box = {"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2}


def format_map(ax, title, left_labels=False):
    """Style a map axis."""
    ax.set_title(title, fontsize=9, y=0.95, color="black", bbox=title_box, zorder=5)
    ax.set_extent(extent, crs=PROJ)
    ax.coastlines(resolution="50m", linewidth=0.6, color="0.3")
    gridlines = ax.gridlines(
        crs=DATA, draw_labels=True, linewidth=0.3, color="0.5", alpha=0.5
    )
    gridlines.x_inline = gridlines.y_inline = False
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.left_labels = left_labels
    gridlines.rotate_labels = False


assimilated_stations = assimilated_reports.drop_duplicates("station")
held_out_stations = held_out_reports.drop_duplicates("station")
rp = PROJ.transform_points(DATA, output_lon, output_lat)
x_rotated, y_rotated = rp[..., 0], rp[..., 1]
extent = [x_rotated.min(), x_rotated.max(), y_rotated.min(), y_rotated.max()]
speed_max = float(max(np.nanpercentile(ws_free, 99), np.nanpercentile(ws_post, 99)))
speed_style = {
    "transform": DATA,
    "shading": "nearest",
    "cmap": "viridis",
    "vmin": 0,
    "vmax": speed_max,
}
increment = ws_post - ws_free
increment_limit = float(np.nanpercentile(np.abs(increment), 99))

plt.close("all")
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.8), subplot_kw={"projection": PROJ}, layout="constrained"
)
prior_mesh = axes[0].pcolormesh(output_lon, output_lat, ws_free, **speed_style)
axes[1].pcolormesh(output_lon, output_lat, ws_post, **speed_style)
increment_mesh = axes[2].pcolormesh(
    output_lon,
    output_lat,
    increment,
    transform=DATA,
    shading="nearest",
    cmap="RdBu_r",
    vmin=-increment_limit,
    vmax=increment_limit,
)

format_map(axes[0], f"Prior ({INIT_TIME:%Y-%m-%d %HZ})", left_labels=True)
format_map(axes[1], "Observation-guided analysis")
format_map(axes[2], "Analysis - prior")
station_styles = [
    (assimilated_stations, {"c": "#D55E00", "linewidths": 0.4, "label": "Assimilated"}),
    (held_out_stations, {"facecolors": "none", "linewidths": 0.8, "label": "Held-out"}),
]
for stations, style in station_styles:
    axes[1].scatter(
        stations["lon"],
        stations["lat"],
        s=24,
        edgecolors="k",
        transform=DATA,
        zorder=3,
        **style,
    )
axes[1].legend(loc="lower right", fontsize=7)
colorbar_style = {"shrink": 0.82, "pad": 0.02}
fig.colorbar(prior_mesh, ax=axes[:2], label="10 m wind speed (m/s)", **colorbar_style)
fig.colorbar(
    increment_mesh, ax=axes[2], label="10 m wind increment (m/s)", **colorbar_style
)
plt.savefig("outputs/03_corrdiff_cosmo_sda.jpg", dpi=120)
