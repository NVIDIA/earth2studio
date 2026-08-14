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

Assimilate real in-situ station observations into the COSMO-REA2 (2.2 km)
downscaler with ``CorrDiffCosmoEra5SDA``, over a dense-station region spanning the
Netherlands, NW Germany and the adjacent North Sea. Given an ERA5 driving state for a
historical time and sparse GHCNHourly 10 m wind reports, diffusion posterior sampling (DPS) steers the
diffusion downscaler's denoising trajectory toward the observations, producing a
high-resolution COSMO-REA analysis that is guided toward what the stations
actually measured.

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

.. note::
   The model package is hosted on Hugging Face
   (``hf://nvidia/corrdiff-cosmo-era5``) and fetched by ``load_default_package()``.

.. note::
   The default ~206 x 206-cell sub-domain completed in a few minutes and used about
   5 GB of GPU memory in one test run (bf16). Set ``DOMAIN = None`` for the full domain;
   its time and memory requirements were not validated.

.. note::
   The assimilated in-situ observations are NOAA NCEI GHCN-Hourly station data.
"""

# /// script
# dependencies = [
#   "earth2studio[data,da-cosmo] @ git+https://github.com/NVIDIA/earth2studio.git",
#   # PhysicsNeMo's RoPE/NATTEN backend is not on PyPI yet; pin the Git source
#   # explicitly -- the repo's [tool.uv.sources] pin does not reach a PEP 723 script.
#   "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@ced75d93d014f70bb691372788eee2d201171c12",
#   "cartopy",
#   "matplotlib",
#   "scipy",
# ]
# ///

# %%
# Configuration
# -------------
import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

INIT_TIME = datetime(2024, 1, 26, 0)  # historical time (ARCO + GHCNHourly cover it)
ASSIMILATE = ("u10m", "v10m")  # 10 m wind, an identity/unit-scale REA2 channel
# Square sub-domain (the Netherlands + NW Germany, ~206 x 206 native cells, many hourly
# stations); shifted N/NW so the North Sea occupies the northwestern edge of the frame.
# Set DOMAIN = None for the full COSMO-REA2 domain (untested, high VRAM).
DOMAIN = dict(lat_min=50.2, lat_max=53.8, lon_min=4.6, lon_max=10.4)
ENSEMBLE_SIZE = 1  # single posterior analysis (one DPS draw; sufficient here)
SAMPLER_STEPS = 12  # 12 steps used to reduce example runtime; model default is 18
SDA_STD_OBS = 0.5  # assumed obs-noise std (physical units; lower trusts obs more)
SDA_GAMMA = 5e-5  # DPS guidance scaling (lower = stronger assimilation)
VAL_FRAC = 0.3  # fraction of stations held out for comparison
# One window for both the obs fetch and the model's assimilation filter, so the
# stations shown/counted are exactly the set handed to the assimilation.
OBS_TIME_TOLERANCE = timedelta(minutes=30)
DEVICE = "cuda:0"
# COSMO-REA is on a rotated-pole grid; plot in that native projection so the crop is
# axis-aligned (square panels) rather than a slanted PlateCarree parallelogram.
DATA = ccrs.PlateCarree()  # station lat/lon and the model's geographic coordinates
PROJ = ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=40.0)  # from rea2 metadata

os.makedirs("outputs", exist_ok=True)


def geo_axes(ax, labels=True, left=True):
    """Add coastlines + (optionally labeled) gridlines to a cartopy GeoAxes.

    ``left=False`` drops the latitude (left) labels -- used on the inner panels of a
    shared-latitude row so only the leftmost panel is labelled.
    """
    ax.coastlines(resolution="50m", linewidth=0.6, color="0.3")
    # x_inline/y_inline=False keeps lon/lat labels on the axis edges (below / left)
    # rather than inline inside the rotated-pole map; don't rotate them to the grid.
    gl = ax.gridlines(
        crs=DATA,
        draw_labels=labels,
        x_inline=False,
        y_inline=False,
        linewidth=0.3,
        color="0.5",
        alpha=0.5,
    )
    if labels:
        gl.top_labels = gl.right_labels = False
        gl.left_labels = left
        gl.rotate_labels = False


def to_numpy(da):
    """Analysis data as NumPy. On CUDA the analysis is CuPy-backed (same-device
    contract), so use ``.data`` + ``.get()`` rather than ``.values`` (which would
    force an implicit -- disallowed -- CuPy->NumPy conversion)."""
    arr = da.data
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


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


# %%
# Load the assimilation model
# ---------------------------
# ``CorrDiffCosmoEra5SDA`` wraps a diffusion-mode ``CorrDiffCosmoEra5`` downscaler.
# ``assimilate_variables`` is required and must be identity-transform, unit-scale
# output channels -- here the 10 m wind, which matches the station reports' height.
from earth2studio.data import ARCO, GHCNHourly, fetch_data
from earth2studio.models.da import CorrDiffCosmoEra5SDA

package = CorrDiffCosmoEra5SDA.load_default_package()

sda = CorrDiffCosmoEra5SDA.load_model(
    package,
    assimilate_variables=ASSIMILATE,
    resolution="rea2",
    domain=DOMAIN,  # sub-domain by default; DOMAIN = None runs the full REA2 domain
    time_tolerance=OBS_TIME_TOLERANCE,  # match the obs fetch window below
    number_of_samples=ENSEMBLE_SIZE,
    sampler_steps=SAMPLER_STEPS,
    sda_std_obs=SDA_STD_OBS,
    sda_gamma=SDA_GAMMA,
    amp=True,  # bf16 autocast on the guided sampler (CUDA); set False for fp32
).to(DEVICE)
sda.seed = 0  # reproducible ensemble

# %%
# ERA5 driving state
# ------------------
# Fetch an ERA5 analysis (ARCO) for the historical time and regrid it onto the
# downscaler's regional input grid. The result is an ``xr.DataArray`` with dims
# ``(time, variable, lat, lon)`` -- the same driving state the downscaler
# conditions on; the ``time`` coord also drives its day/night (solar) input.
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
# GHCNHourly 10 m wind observations
# ---------------------------------
# Fetch GHCNHourly reports over the model's output domain, keep one report per
# (station, variable) closest to the analysis time, and keep only stations that
# report both wind components. Then split the stations into an assimilated set and
# a held-out set so the observation-guided analysis can be compared at sites that
# were never assimilated.
glat = sda.model.lat_output_numpy
glon = sda.model.lon_output_numpy
bbox = (float(glat.min()), float(glon.min()), float(glat.max()), float(glon.max()))

stations = GHCNHourly.get_stations_bbox(bbox)
ghcn = GHCNHourly(stations=stations, time_tolerance=OBS_TIME_TOLERANCE, verbose=False)
raw = ghcn(INIT_TIME, list(ASSIMILATE))
raw = raw[raw["variable"].isin(ASSIMILATE)].dropna(subset=["observation"]).copy()
# one report per (station, variable): the one closest to INIT_TIME
raw["dt"] = (pd.to_datetime(raw["time"]) - INIT_TIME).abs()
raw = raw.sort_values("dt").drop_duplicates(["station", "variable"], keep="first")
# keep only stations reporting BOTH u10m and v10m
both = raw.groupby("station")["variable"].nunique().eq(len(ASSIMILATE))
raw = raw[raw["station"].isin(both[both].index)]
station_ids = sorted(raw["station"].unique())
print(
    f"stations with {'+'.join(ASSIMILATE)} in domain @ {INIT_TIME}: {len(station_ids)}"
)
if len(station_ids) < 2:
    raise RuntimeError(
        f"need >= 2 usable stations to split (got {len(station_ids)}); widen DOMAIN, "
        "loosen time_tolerance, or pick a time/region with more reports."
    )

rng = np.random.default_rng(0)
n_val = min(len(station_ids) - 1, max(1, int(round(len(station_ids) * VAL_FRAC))))
val_ids = set(rng.choice(station_ids, size=n_val, replace=False))
obs_cols = ["time", "lat", "lon", "variable", "observation"]
is_held_out = raw["station"].isin(val_ids)
assimilated_reports = raw[~is_held_out]
held_out_reports = raw[is_held_out]
# The model does not need the station ID, so pass only its observation columns.
assimilation_obs = assimilated_reports[obs_cols]
n_assim = len(station_ids) - len(val_ids)
print(
    f"provided for assimilation: {n_assim} stations "
    f"| held-out: {len(val_ids)} stations"
)

# %%
# Prior vs observation-guided analysis
# -------------------------------------
# Run the downscaler twice on the same ERA5 state: once with no observations (the
# prior, ``obs=None``) and once with the observations selected for assimilation.
# Both return an ensemble with dims ``(time, sample, variable, y, x)``.
post = sda(x_da, assimilation_obs)  # analysis (observation-guided)
free = sda(x_da)  # prior (free, no-obs downscaling)

# Locate the wind-component channels and output grid used for evaluation and plotting.
output_variables = list(post["variable"].values)
u_index = output_variables.index("u10m")
v_index = output_variables.index("v10m")
output_lat = np.asarray(post["lat"])
output_lon = np.asarray(post["lon"])

post_np, free_np = to_numpy(post), to_numpy(free)  # [time, sample, variable, y, x]
# One posterior draw (ENSEMBLE_SIZE = 1): the map shows that analysis' wind speed and
# the RMSE below uses its u/v components. The ``.mean(0)`` over ``sample`` is a no-op
# here; with ENSEMBLE_SIZE > 1 it yields the mean per-member speed (map) and the mean
# components (RMSE) -- averaging speeds, not components, avoids direction cancellation.
ws_post = np.hypot(post_np[0, :, u_index], post_np[0, :, v_index]).mean(0)
ws_free = np.hypot(free_np[0, :, u_index], free_np[0, :, v_index]).mean(0)
analysis_u = post_np[0, :, u_index].mean(0)
analysis_v = post_np[0, :, v_index].mean(0)
prior_u = free_np[0, :, u_index].mean(0)
prior_v = free_np[0, :, v_index].mean(0)
print(
    f"finite: posterior {np.isfinite(post_np).all()} free {np.isfinite(free_np).all()}"
)

# %%
# Compare to held-out stations
# ----------------------------
# Compute the vector-wind RMSE of each analysis against the observations, at the
# stations' nearest output cells. The *held-out* stations were never assimilated,
# so their prior-vs-posterior RMSE illustrates how the analysis behaves at
# unassimilated sites for this single case -- it is a diagnostic, not a statistical
# skill claim, and a single time/split/ensemble does not guarantee improvement.


def vector_rmse_at(df, u_field, v_field):
    """Vector 10 m wind RMSE (m/s) vs obs at each station's nearest output cell.
    Components are paired by ``station`` (not lat/lon) to avoid cross-pairing."""
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


print("=== vector 10 m wind RMSE vs GHCNHourly (m/s) ===")
print(
    f"  held-out:    prior {vector_rmse_at(held_out_reports, prior_u, prior_v):.3f}"
    f"  ->  analysis {vector_rmse_at(held_out_reports, analysis_u, analysis_v):.3f}"
)
print(
    f"  assimilated: prior "
    f"{vector_rmse_at(assimilated_reports, prior_u, prior_v):.3f}"
    f"  ->  analysis "
    f"{vector_rmse_at(assimilated_reports, analysis_u, analysis_v):.3f}"
)

# %%
# Plot the analyses and their difference
# --------------------------------------
# Left: prior (free, no-obs) 10 m wind speed. Middle: observation-guided analysis,
# with assimilated (filled) and held-out (open) stations overlaid. Right: the signed
# increment the observations add (analysis - prior), with per-held-out-station error-
# change markers (up = error reduced, down = error increased, marker area proportional
# to |error change|).


def station_locations(df):
    """Unique (lat, lon) of the stations in ``df``."""
    loc = df.drop_duplicates("station")
    return loc["lat"].values, loc["lon"].values


def held_out_error_change(df):
    """Per-station change in vector-wind error at each station's nearest output cell:
    ``delta = prior_error - posterior_error`` (positive = the analysis is closer to
    the station than the prior). One split is a single case, not a skill assessment.
    Returns (lat, lon, delta) paired by ``station``."""
    # Pair the observed u/v components by station.
    u_reports = df[df["variable"] == "u10m"][["station", "lat", "lon", "observation"]]
    v_reports = df[df["variable"] == "v10m"][["station", "observation"]]
    merged = u_reports.merge(v_reports, on="station", suffixes=("_u", "_v"))

    # Sample each model field at the station's nearest output-grid cell.
    tree = cKDTree(np.column_stack([output_lat.ravel(), (output_lon % 360).ravel()]))
    _, flat_indices = tree.query(np.column_stack([merged["lat"], merged["lon"] % 360]))
    row_indices, col_indices = np.unravel_index(flat_indices, output_lat.shape)
    observed_u = merged["observation_u"].values
    observed_v = merged["observation_v"].values
    prior_error = np.hypot(
        prior_u[row_indices, col_indices] - observed_u,
        prior_v[row_indices, col_indices] - observed_v,
    )
    analysis_error = np.hypot(
        analysis_u[row_indices, col_indices] - observed_u,
        analysis_v[row_indices, col_indices] - observed_v,
    )
    return (
        merged["lat"].values,
        merged["lon"].values,
        prior_error - analysis_error,
    )


tr_lat, tr_lon = station_locations(assimilated_reports)
va_lat, va_lon = station_locations(held_out_reports)

plt.close("all")
fig, axs = plt.subplots(1, 3, figsize=(21, 6.2), subplot_kw={"projection": PROJ})


def add_colorbar(mesh, ax, label):
    """Colorbar the exact height of its (square) map, via an axes divider."""
    cax = make_axes_locatable(ax).append_axes(
        "right", size="4%", pad=0.08, axes_class=plt.Axes
    )
    return fig.colorbar(mesh, cax=cax, label=label)


# frame each map to the data footprint in the rotated-pole frame, where the crop is an
# axis-aligned (near-square) rectangle; colorbars are matched to the map height below.
rp = PROJ.transform_points(
    DATA, output_lon, output_lat
)  # true lat/lon -> rotated-pole coords
extent = [
    float(rp[..., 0].min()),
    float(rp[..., 0].max()),
    float(rp[..., 1].min()),
    float(rp[..., 1].max()),
]
# shared color limit across prior and analysis so neither panel saturates
vmax = float(max(np.nanpercentile(ws_free, 99), np.nanpercentile(ws_post, 99)))
speed_kw = dict(transform=DATA, shading="nearest", cmap="viridis", vmin=0, vmax=vmax)

m0 = axs[0].pcolormesh(output_lon, output_lat, ws_free, **speed_kw)
axs[0].set_title(f"Prior 10 m wind speed, no obs  ({INIT_TIME:%Y-%m-%d %HZ})")
geo_axes(axs[0])
axs[0].set_extent(extent, crs=PROJ)
add_colorbar(m0, axs[0], "10 m wind speed (m/s)")

m1 = axs[1].pcolormesh(output_lon, output_lat, ws_post, **speed_kw)
axs[1].scatter(
    tr_lon,
    tr_lat,
    s=30,
    c="red",
    edgecolors="k",
    linewidths=0.5,
    transform=DATA,
    zorder=3,
    label="assimilated",
)
axs[1].scatter(
    va_lon,
    va_lat,
    s=30,
    facecolors="none",
    edgecolors="k",
    linewidths=1.0,
    transform=DATA,
    zorder=3,
    label="held-out",
)
axs[1].legend(loc="lower right", fontsize=8)
axs[1].set_title("Observation-guided analysis 10 m wind speed")
geo_axes(axs[1], left=False)
axs[1].set_extent(extent, crs=PROJ)
add_colorbar(m1, axs[1], "10 m wind speed (m/s)")

diff = ws_post - ws_free
dlim = float(np.nanpercentile(np.abs(diff), 99))
m2 = axs[2].pcolormesh(
    output_lon,
    output_lat,
    diff,
    transform=DATA,
    shading="nearest",
    cmap="RdBu_r",
    vmin=-dlim,
    vmax=dlim,
)
axs[2].set_title("Increment (analysis - prior)")
geo_axes(axs[2], left=False)
axs[2].set_extent(extent, crs=PROJ)
# Overlay the per-held-out-station error change (delta = prior_error - posterior_error):
# up-triangle where the analysis sits closer to the withheld obs than the prior,
# down-triangle where it is worse. Marker area is on an absolute scale (points^2 per
# m/s) so sizes are comparable across figures/dates. Held-out stations only --
# assimilated sites are fit by construction, so their change is not a fair comparison.
hv_lat, hv_lon, hv_d = held_out_error_change(held_out_reports)
AREA_PER_MS = 70.0  # marker area (points^2) per m/s of |delta error|; |delta|=1 -> 70
sizes = AREA_PER_MS * np.clip(np.abs(hv_d), 0.2, 3.0)
better = hv_d >= 0
# colorblind-safe (Okabe-Ito) fills with a white edge to read on the red/blue field
skill_kw = dict(transform=DATA, edgecolors="w", linewidths=1.1, zorder=4)
axs[2].scatter(
    hv_lon[better], hv_lat[better], marker="^", c="#009E73", s=sizes[better], **skill_kw
)
axs[2].scatter(
    hv_lon[~better],
    hv_lat[~better],
    marker="v",
    c="#D55E00",
    s=sizes[~better],
    **skill_kw,
)
# equal-size, colorblind-safe legend proxies drawn at the |delta error| = 1 m/s size,
# so the two entries differ only in shape/color (sign), and the size sets the scale
ref_ms = float(np.sqrt(AREA_PER_MS))
skill_legend = [
    Line2D(
        [0],
        [0],
        marker="^",
        linestyle="none",
        markerfacecolor="#009E73",
        markeredgecolor="0.3",
        markersize=ref_ms,
        label="held-out: error reduced",
    ),
    Line2D(
        [0],
        [0],
        marker="v",
        linestyle="none",
        markerfacecolor="#D55E00",
        markeredgecolor="0.3",
        markersize=ref_ms,
        label="held-out: error increased",
    ),
]
axs[2].legend(
    handles=skill_legend,
    loc="lower right",
    fontsize=7,
    framealpha=0.9,
    title="marker area ∝ |Δ error|",
    title_fontsize=6,
)
add_colorbar(m2, axs[2], "10 m wind increment (m/s)")

fig.subplots_adjust(wspace=0.18)
plt.savefig("outputs/03_corrdiff_cosmo_sda.jpg", dpi=150, bbox_inches="tight")
