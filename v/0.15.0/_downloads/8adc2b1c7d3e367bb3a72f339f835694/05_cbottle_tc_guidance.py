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
CBottle Tropical Cyclone Guidance
==================================

Guided tropical cyclone sampling with cBottle and odds-ratio diagnostics.

This example demonstrates the cBottle TC guidance model for generating synthetic
tropical cyclone samples at user-specified locations and computing the log-odds ratio
between the guided and unguided distributions.  The odds ratio quantifies how much more
likely a particular sample is under guidance compared to the base model.

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

For more information on the odds ratio see:

- https://arxiv.org/abs/2605.03802

In this example you will learn:

- Running guided TC sampling with :py:class:`earth2studio.models.dx.CBottleTCGuidance`
- Visualizing a guided sample over a regional domain
- Reloading the model with second-order derivative support for odds-ratio computation
- Computing and interpreting the log-odds ratio of a guided sample
"""
# /// script
# dependencies = [
#   "earth2studio[cbottle] @ git+https://github.com/NVIDIA/earth2studio.git@0.15.0",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# For this example we need the cBottle TC guidance diagnostic model. We load it twice:
#
# 1. Default (fast) path for standard guided sampling
# 2. Second-order-derivative path for odds-ratio computation
#
# Thus, we need the following:
#
# - Diagnostic Model: Use the built in CBottle TC Guidance Model :py:class:`earth2studio.models.dx.CBottleTCGuidance`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime

import numpy as np
import torch

from earth2studio.models.dx import CBottleTCGuidance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the default model package which downloads the checkpoint from NGC
package = CBottleTCGuidance.load_default_package()

# %%
# Guided TC Sampling (Fast Path)
# ------------------------------
# The guidance tensor marks the location where we want TC activity.  Here we place a
# single guidance point near Florida and request one timestamp during hurricane season.
# The fast model path (``allow_second_order_derivatives=False``) is optimized for
# standard guided inference.

# %%
lat = torch.tensor([27.0], device=device)  # Near Florida
lon = torch.tensor([-82.0], device=device)  # Converted internally to [0, 360)
times = [datetime(2005, 10, 11, 12)]

model = CBottleTCGuidance.load_model(package, seed=0).to(device)
# Create guidance tensor
guidance, coords = model.create_guidance_tensor(lat, lon, times)
guidance = guidance.to(device)
# Run guided sampling
guided_sample, guided_coords = model(guidance, coords)

# %%
# Post Processing Guided Sample
# -----------------------------
# Plot the 10-metre zonal wind (u10m) over a Caribbean domain to visualize the
# generated tropical cyclone structure.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")

variables = guided_coords["variable"]
u_var = "u10m"
u_idx = int(np.where(variables == u_var)[0][0])

# guided_sample dims: [time, lead_time, variable, lat, lon]
u = guided_sample[0, 0, u_idx].detach().cpu().numpy()
lat_coords = guided_coords["lat"]
lon_coords = guided_coords["lon"]

# Caribbean box in 0-360 longitude convention
lon_min, lon_max = 260.0, 300.0  # 100W to 60W
lat_min, lat_max = 15.0, 40.0  # 15N to 40N
lon_mask = (lon_coords >= lon_min) & (lon_coords <= lon_max)
lat_mask = (lat_coords >= lat_min) & (lat_coords <= lat_max)

u_carib = u[np.ix_(lat_mask, lon_mask)]
lat_carib = lat_coords[lat_mask]
lon_carib = lon_coords[lon_mask]
# Convert to -180..180 for plotting
lon_carib_deg = ((lon_carib + 180.0) % 360.0) - 180.0

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 4.5))
pcm = ax.pcolormesh(
    lon_carib_deg,
    lat_carib,
    u_carib,
    shading="auto",
    cmap="RdBu_r",
    vmin=-30,
    vmax=30,
    transform=ccrs.PlateCarree(),
)
ax.set_extent([-100.0, -60.0, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines(resolution="110m", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
plt.colorbar(pcm, ax=ax, label=f"{u_var} (m/s)", pad=0.08, shrink=0.92)
ax.set_title("Guided TC Sample: 10m Zonal Wind")
plt.tight_layout()
plt.savefig("outputs/05_cbottle_tc_guided_sample.jpg")

# %%
# Computing the Odds Ratio
# -------------------------
# The odds ratio requires computing Hutchinson divergence terms that need second-order
# derivatives through the model.  We reload with ``allow_second_order_derivatives=True``
# for odds ratio calculations.
#
# Note: ``sampler_steps`` is also reduced in this section to speed up runtime. Use
# the default sampler settings for improved quality and more stable odds-ratio values.

# %%
model = CBottleTCGuidance.load_model(
    package,
    seed=0,
    sampler_steps=2,
    allow_second_order_derivatives=True,
).to(device)

log_odds_ratio, forward_latents, latent_coords = model.calculate_odds_ratio(
    guidance,
    coords,
)

print(f"Log odds ratio: {log_odds_ratio:.4f}")
print(f"Forward latents shape: {tuple(forward_latents.shape)}")

# %%
# Post Processing Forward Latents
# --------------------------------
# The ``forward_latents`` tensor is returned on the same grid as the model output
# (lat-lon when ``lat_lon=True``).  We visualize a single channel (u10m) over the same
# Caribbean domain.  This shows the latent-space representation that the odds-ratio
# computation operates on.

# %%
plt.close("all")

# Identify the u10m channel in output variable ordering
latent_variables = latent_coords["variable"]
u_latent_idx = int(np.where(latent_variables == u_var)[0][0])
latent_u = forward_latents[0, u_latent_idx].detach().cpu().numpy()

latent_u_carib = latent_u[np.ix_(lat_mask, lon_mask)]

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 4.5))
pcm = ax.pcolormesh(
    lon_carib_deg,
    lat_carib,
    latent_u_carib,
    shading="auto",
    cmap="viridis",
    transform=ccrs.PlateCarree(),
)
ax.set_extent([-100.0, -60.0, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines(resolution="110m", linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
plt.colorbar(pcm, ax=ax, label=f"Forward Latent ({u_var})", pad=0.08, shrink=0.92)
ax.set_title(f"Forward Latents: {u_var} Channel")
plt.tight_layout()
plt.savefig("outputs/05_cbottle_tc_forward_latents.jpg")
