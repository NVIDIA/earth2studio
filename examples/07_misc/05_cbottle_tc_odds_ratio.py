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
CBottle TC Odds Ratio
===============================

Minimal example showing how to use guided sampling from the cBottle TC model 
and how to calculate the ratio between the probability of a guided sample under the 
guided model and the probability of the sample under the unguided model (odds ratio). 

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

For more information on the odds ratio see:

- https://arxiv.org/abs/2605.03802

In this example you will:
- Run one guided sample with the default model, producing a sample with a TC in a user-specified location
- Reload the model with ``allow_second_order_derivatives=True`` and produce a new guided sample 
- Compute the new TC sample's odds ratio
"""
# /// script
# dependencies = [
#   "earth2studio[cbottle] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# We load cBottle twice in this example:
# 1) fast/default path for guided generation
# 2) second-order-derivative path for odds-ratio diagnostics
#
# We also route caches to Lustre to avoid filling local disk during model downloads.
from datetime import datetime
import os

import torch

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from earth2studio.models.dx import CBottleTCGuidance
from earth2studio.lexicon import CBottleLexicon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
package = CBottleTCGuidance.load_default_package()

# %%
# Build a Single Guidance Sample
# ------------------------------
# The guidance tensor marks where we want TC activity. Here we provide one point near
# the Caribbean and one timestamp.
lat = torch.tensor([27.0], device=device)   # Near Florida
lon = torch.tensor([-82.0], device=device)  # Converted internally to [0, 360)
times = [datetime(2005, 10, 11, 12)]

# Use the default fast model path for standard guided sampling.
model_fast = CBottleTCGuidance.load_model(
    package,
    seed=0,
    sampler_steps=18,
    allow_second_order_derivatives=False,
).to(device)
guidance, coords = model_fast.create_guidance_tensor(lat, lon, times)
guidance = guidance.to(device)

# %%
# One Guided Sample (Fast Path)
# -----------------------------
# Run a normal guided sample first (no odds-ratio machinery yet).
guided_sample, guided_coords = model_fast(guidance, coords)
print(f"guided_sample shape: {tuple(guided_sample.shape)}")
print(f"n_output_variables: {guided_coords['variable'].shape[0]}")

# %%
# Plot one near-surface zonal wind field over a Caribbean window.

variables = guided_coords["variable"]
if "u10m" in variables:
    u_var = "u10m"
elif "u1000" in variables:
    u_var = "u1000"
else:
    u_var = next((v for v in variables if str(v).startswith("u")), None)
    if u_var is None:
        raise RuntimeError("Could not find a U-wind variable in output coords.")

u_idx = int(np.where(variables == u_var)[0][0])

# guided_sample dims: [time, lead_time, variable, lat, lon]
u = guided_sample[0, 0, u_idx].detach().cpu().numpy()
lat = guided_coords["lat"]
lon = guided_coords["lon"]

# Caribbean box in 0-360 longitude convention.
lon_min, lon_max = 260.0, 300.0  # 100W to 60W
lat_min, lat_max = 15.0, 40.0  # 15N to 40N
lon_mask = (lon >= lon_min) & (lon <= lon_max)
lat_mask = (lat >= lat_min) & (lat <= lat_max)

u_carib = u[np.ix_(lat_mask, lon_mask)]
lat_carib = lat[lat_mask]
lon_carib = lon[lon_mask]
lon_carib_deg = ((lon_carib + 180.0) % 360.0) - 180.0

fig = plt.figure(figsize=(8, 4.5))
ax = plt.axes(projection=ccrs.PlateCarree())
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
gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
gl.right_labels = False
gl.top_labels = False
plt.colorbar(pcm, ax=ax, label=f"{u_var} (m/s)", pad=0.08, shrink=0.92)
ax.set_title("Guided sample: Caribbean zonal wind")
plt.tight_layout()
plt.show()
# %%
# Reload and Calculate Odds Ratio
# -------------------------------
# Odds-ratio uses divergence terms that require second-order derivatives.
# We therefore reload with allow_second_order_derivatives=True.
model_odds = CBottleTCGuidance.load_model(
    package,
    seed=0,
    sampler_steps=2,
    allow_second_order_derivatives=True,
).to(device)
if not hasattr(model_odds.core_model, "calculate_odds_ratio"):
    raise RuntimeError(
        "The installed cBottle version does not expose calculate_odds_ratio. "
        "Please update cBottle before running this example."
    )
if model_odds.device_buffer.device.type != "cuda":
    raise RuntimeError(
        "Odds-ratio is running on CPU, which is usually very slow. "
        "Run on a CUDA device to use default odds-ratio settings."
    )
# This will create a new guided sample and calculate the odds ratio 
log_odds_ratio, forward_latents = model_odds.calculate_odds_ratio(
    guidance,
    coords,
)


print(f"log_odds_ratio: {log_odds_ratio:.4f}")
print(f"forward_latents shape: {tuple(forward_latents.shape)}")

# %%
# Plot forward_latents 
# ------------------------------
# forward_latents are in the model's native HEALPix space [batch, channel, time, hpx].
# Regrid one latent slice (uas channel, time=0) to lat/lon for a simple map view.
# In Earth2Studio naming, cBottle "uas" corresponds to "u10m".
uas_var = next(
    var for var, (native_name, level) in CBottleLexicon.VOCAB.items()
    if native_name == "uas" and level == -1
)
uas_idx = int(np.where(guided_coords["variable"] == uas_var)[0][0])
latent_native = forward_latents[:, uas_idx : uas_idx + 1, :1, :]
# forward_latents are raw (pre-postprocess) tensors in model domain pixel ordering
# (typically HEALPIX_PAD_XY). Use the domain grid explicitly for regridding.
domain_grid = getattr(model_odds.core_model.net.domain, "_grid", model_odds.core_model.net.domain)
latent_ll = (
    model_odds.regrid_hpx_to_latlon(latent_native, grid=domain_grid)
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)

latent_carib = latent_ll[np.ix_(lat_mask, lon_mask)]

fig = plt.figure(figsize=(8, 4.5))
ax = plt.axes(projection=ccrs.PlateCarree())
pcm = ax.pcolormesh(
    lon_carib_deg,
    lat_carib,
    latent_carib,
    shading="auto",
    cmap="viridis",
    transform=ccrs.PlateCarree(),
)
ax.set_extent([-100.0, -60.0, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.coastlines(resolution="110m", linewidth=0.8)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
gl.right_labels = False
gl.top_labels = False
plt.colorbar(
    pcm,
    ax=ax,
    label=f"forward_latents[{uas_var} (= uas)]",
    pad=0.08,
    shrink=0.92,
)
ax.set_title(f"Forward Latents ({uas_var} channel)")
plt.tight_layout()
plt.show()

