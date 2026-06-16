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

This minimal example chains two models to nowcast Global Horizontal Irradiance
(GHI) over the Continental US from geostationary satellite imagery:

- :py:class:`earth2studio.models.px.StormScopeGOES` (3-km, 10-min) autoregressively
  predicts future GOES imagery (open-source ``nvidia/stormscope-goes-mrms``).
- :py:class:`earth2studio.models.px.StormScopeNSRDB` estimates surface GHI from each
  predicted GOES frame (regression + SDEdit diffusion refinement).

In this example you will learn how to load both models, run a short
autoregressive forecast estimating GHI at each step (including t=0), and plot the
result. GHI is produced on the 3-km HRRR grid.
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
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeNSRDB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load Models
# -----------
# The 3-km GOES model is loaded from the open-source Hugging Face package; the
# NSRDB solar model is loaded from its (private) repo -- authenticate first with
# ``export HF_TOKEN=...`` or ``huggingface-cli login``.

# %%
goes_model = StormScopeGOES.load_model(
    package=StormScopeGOES.load_default_package(),
    model_name="3km_10min_natten_pure_obs_cos_zenith_input_eoe",
    conditioning_data_source=None,
).to(device).eval()

solar_package = Package(
    "hf://acarpentieri/Stormscope-nsrdb",
    cache_options={
        "cache_storage": Package.default_cache("stormscope_nsrdb"),
        "same_names": True,
    },
)
nsrdb_model = StormScopeNSRDB.load_model(
    package=solar_package,
    model_name="stormscope_solar_goes_nsrdb",
    conditioning_data_source=None,
).to(device).eval()

# %%
# Fetch GOES Initial Conditions
# -----------------------------
# We fetch GOES-16 CONUS data for a clear-sky afternoon case and build the grid
# interpolators for both models.

# %%
start_date = [np.datetime64(datetime(2024, 7, 15, 18, 0, 0))]
goes = GOES(satellite="goes16", scan_mode="C")
goes_lat, goes_lon = GOES.grid(satellite="goes16", scan_mode="C")
goes_model.build_input_interpolator(goes_lat, goes_lon)

# Build the NSRDB conditioning interpolator from the RAW GOES satellite grid (not
# the GOES model grid). This makes ``conditioning_valid_mask`` correctly flag model
# pixels with no GOES coverage as invalid, so ``_forward`` zeroes them after
# normalization. Building it from the model grid instead leaves off-disk 0.0 fills
# that normalize to out-of-distribution values, corrupting the denoiser's GroupNorm
# statistics and biasing GHI high. ``use_ckdtree=True`` matches NSRDB training.
nsrdb_model.build_conditioning_interpolator(goes_lat, goes_lon, use_ckdtree=True)

in_coords = goes_model.input_coords()
x, x_coords = fetch_data(
    goes,
    time=start_date,
    variable=np.array(in_coords["variable"]),
    lead_time=in_coords["lead_time"],
    device=device,
)
if x.dim() == 5:
    x = x.unsqueeze(0)
x = x.to(dtype=torch.float32)
x_coords["batch"] = np.arange(1)
x_coords.move_to_end("batch", last=False)

# %%
# Run Forecast Loop (+ GHI for t=0)
# ---------------------------------
# At each step we (1) predict future GOES imagery, (2) estimate GHI from it, and
# (3) advance the GOES sliding window. ``num_steps`` controls the SDEdit
# refinement steps (fewer = faster/smoother, more = sharper/stochastic).

# %%
nsrdb_model.sampler_args = {"num_steps": 12}
goes_model.sampler_args = {"num_steps": 48, "S_churn": 0.0}

n_steps = 2
ghi_forecasts, valid_times = [], []
y, y_coords = x, x_coords
for step_idx in range(n_steps):
    torch.manual_seed(step_idx)
    y_pred, y_pred_coords = goes_model(y, y_coords)
    ghi_pred, _ = nsrdb_model.estimate_from_goes(
        y_pred.clone().float(), y_pred_coords.copy()
    )
    ghi_forecasts.append(ghi_pred[0, 0, 0, 0].detach().cpu().numpy())
    lead = int(y_pred_coords["lead_time"][0] / np.timedelta64(1, "m"))
    valid_times.append(lead)
    print(f"  Step {step_idx + 1}/{n_steps}: lead +{lead}min, "
          f"GHI mean={np.nanmean(ghi_forecasts[-1]):.0f} W/m²")
    y, y_coords = goes_model.next_input(y_pred, y_pred_coords, y, y_coords)

# Estimate GHI for the initial t=0 window (regrid the raw IC onto the GOES grid).
x_goes_grid, x_goes_coords = goes_model.prep_input(x.clone().float(), x_coords.copy())
ghi_init, _ = nsrdb_model.estimate_from_goes(x_goes_grid, x_goes_coords)
ghi_forecasts = [ghi_init[0, 0, 0, 0].detach().cpu().numpy()] + ghi_forecasts
valid_times = [0] + valid_times

# %%
# Plot Results
# ------------
# GHI at t=0 and each forecast lead, on the 3-km NSRDB grid.

# %%
lat_nsrdb = nsrdb_model.latitudes.detach().cpu().numpy()
lon_nsrdb = nsrdb_model.longitudes.detach().cpu().numpy()
proj = ccrs.LambertConformal(
    central_longitude=262.5, central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

fig, axes = plt.subplots(
    1, len(ghi_forecasts), figsize=(6 * len(ghi_forecasts), 5),
    subplot_kw={"projection": proj},
)
axes = np.atleast_1d(axes)
for ax, ghi, lead in zip(axes, ghi_forecasts, valid_times):
    ax.coastlines(color="black", linewidth=0.8)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.4)
    im = ax.pcolormesh(
        lon_nsrdb, lat_nsrdb, ghi, transform=ccrs.PlateCarree(), cmap="hot",
        shading="auto", vmin=0, vmax=np.nanpercentile(ghi[np.isfinite(ghi)], 98),
    )
    ax.set_title(f"+{lead} min", fontsize=12)

fig.colorbar(im, ax=axes, shrink=0.6, label="GHI [W/m²]",
             orientation="horizontal", pad=0.05)
fig.suptitle(
    f"StormScope Solar — GHI Nowcast\n"
    f"Init: {start_date[0]} UTC | {lat_nsrdb.shape[0]}x{lon_nsrdb.shape[0]} (NSRDB 3-km)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig("outputs/04_stormscope_solar_nowcast.png", dpi=300)
print("Saved outputs/04_stormscope_solar_nowcast.png")
