# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
CBottle Video Inference
=======================

Climate in a Bottle (cBottle) video model inference workflows.

This example will demonstrate the cBottle video diffusion model for generating
temporal sequences of global weather states. The CBottleVideo model predicts 12
frames at a time (in 6-hour increments) and can operate in both unconditional and
conditional modes.

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

In this example you will learn:

- Running unconditional video generation with CBottleVideo
- Running conditional video generation using ERA5 data with CBottleInfill
- Post-processing and visualizing the temporal forecasts
- Creating animated videos of the forecasts
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
# For this example we will use the CBottleVideo prognostic model. Unlike typical
# prognostic models, CBottleVideo is a diffusion-based video model that generates
# 12 frames (0-66 hours in 6-hour steps) at once.

# %%
# We need the following components:
#
# - Prognostic Model: Use the CBottleVideo model :py:class:`earth2studio.models.px.CBottleVideo`.
# - Datasource: Pull data from the WeatherBench2 data api :py:class:`earth2studio.data.WB2ERA5`.
# - Diagnostic Model: Use the CBottle Infill Model :py:class:`earth2studio.models.dx.CBottleInfill` for conditional generation.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from datetime import datetime

from earth2studio.models.px import CBottleVideo
from earth2studio.data import WB2ERA5
from earth2studio.models.dx import CBottleInfill
from earth2studio.data.utils import fetch_data

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CBottleVideo model
package = CBottleVideo.load_default_package()
cbottle_video = CBottleVideo.load_model(package, seed=42)
cbottle_video = cbottle_video.to(device)

# %%
# Unconditional Video Generation
# -------------------------------
# First, let's generate a video sequence unconditionally. This means the model will
# generate plausible weather states based only on the timestamp and SST data, without
# conditioning on any specific initial atmospheric state. We do this by providing a
# tensor of NaN values as input.

# %%

# Prepare input coordinates
time = np.array([datetime(2022, 6, 1)], dtype="datetime64[ns]")
coords = cbottle_video.input_coords()
coords["time"] = time
coords["batch"] = np.array([0])

# Create NaN tensor for unconditional sampling
x_uncond = torch.full(
    (1, 1, 1, len(cbottle_video.VARIABLES), 721, 1440),
    float("nan"),
    dtype=torch.float32,
    device=device,
)

# Run inference - the model generates 12 frames at once
print("Running unconditional generation...")
iterator = cbottle_video.create_iterator(x_uncond, coords)
uncond_outputs = []
uncond_coords_list = []
for step, (output, output_coords) in enumerate(iterator):
    lead_time = output_coords["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    print(f"Step {step}: lead_time = +{hours}h")
    uncond_outputs.append(output.cpu())
    uncond_coords_list.append(output_coords)
    if step >= 11:  # Get first 12 frames (0-66 hours)
        break

# %%
# Conditional Video Generation with ERA5
# ---------------------------------------
# Next, let's demonstrate conditional generation using real ERA5 data. Since ERA5
# doesn't contain all 45 variables required by CBottleVideo, we first use the
# CBottleInfill model to generate the missing variables, then use that complete
# state to condition the video model.

# %%

# Load ERA5 data source
era5_ds = WB2ERA5()

# Load CBottleInfill model to generate all required variables
input_variables = [
    "u10m",
    "v10m",
    "t2m",
    "msl",
    "z50",
    "u50",
    "v50",
    "z500",
    "u500",
    "v500",
    "z1000",
    "u1000",
    "v1000",
]

package_infill = CBottleInfill.load_default_package()
cbottle_infill = CBottleInfill.load_model(
    package_infill, input_variables=input_variables, sampler_steps=18
)
cbottle_infill = cbottle_infill.to(device)
cbottle_infill.set_seed(42)

# Fetch ERA5 data
print("Fetching ERA5 data and running infill...")
times = np.array([datetime(2022, 6, 1)], dtype="datetime64[ns]")
era5_x, era5_coords = fetch_data(era5_ds, times, input_variables, device=device)

# Infill to get all 45 CBottleVideo variables
infilled_x, infilled_coords = cbottle_infill(era5_x, era5_coords)

# Reshape for CBottleVideo input: [batch, time, lead_time, variable, lat, lon]
x_cond = infilled_x.unsqueeze(2)  # Add lead_time dimension
print(f"Conditioned input shape: {x_cond.shape}")

# Update coordinates for CBottleVideo
coords_cond = cbottle_video.input_coords()
coords_cond["time"] = times
coords_cond["batch"] = np.array([0])
coords_cond["variable"] = infilled_coords["variable"]

# Run conditional inference
print("Running conditional generation...")
cbottle_video.set_seed(42)  # Set seed for reproducibility
iterator_cond = cbottle_video.create_iterator(x_cond, coords_cond)
cond_outputs = []
cond_coords_list = []
for step, (output, output_coords) in enumerate(iterator_cond):
    lead_time = output_coords["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    print(f"Step {step}: lead_time = +{hours}h")
    cond_outputs.append(output.cpu())
    cond_coords_list.append(output_coords)
    if step >= 11:  # Get first 12 frames (0-66 hours)
        break

# %%
# Post Processing and Visualization - Static Plots
# -------------------------------------------------
# Let's visualize the results by comparing unconditional and conditional generation.
# We'll plot the mean sea level pressure (msl) at different time steps to see how
# the weather patterns evolve.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

variable = "msl"
var_idx = np.where(uncond_coords_list[0]["variable"] == variable)[0][0]

# Select time steps to visualize (0, 24h, 48h, 66h)
time_steps = [0, 4, 8, 11]

plt.close("all")
projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=45.0)

# Create a figure with subplots
fig, axes = plt.subplots(
    2, 4, subplot_kw={"projection": projection}, figsize=(16, 8)
)


def plot_field(ax, data, coords, title):
    """Helper function to plot a field"""
    im = ax.pcolormesh(
        coords["lon"],
        coords["lat"],
        data,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=95000,
        vmax=105000,
    )
    ax.coastlines()
    ax.gridlines()
    ax.set_title(title)
    return im


# Plot unconditional generation
for i, step in enumerate(time_steps):
    lead_time = uncond_coords_list[step]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    plot_field(
        axes[0, i],
        uncond_outputs[step][0, 0, 0, var_idx].numpy(),
        uncond_coords_list[step],
        f"Unconditional: +{hours}h",
    )

# Plot conditional generation
for i, step in enumerate(time_steps):
    lead_time = cond_coords_list[step]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    im = plot_field(
        axes[1, i],
        cond_outputs[step][0, 0, 0, var_idx].numpy(),
        cond_coords_list[step],
        f"Conditional (ERA5): +{hours}h",
    )

# Add colorbar
fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.05, label="MSL (Pa)")

plt.tight_layout()
plt.savefig("outputs/19_cbottle_video_static.jpg", dpi=150, bbox_inches="tight")
print("Saved static visualization to outputs/19_cbottle_video_static.jpg")

# %%
# Creating Animated Videos
# ------------------------
# Now let's create animated videos from our forecasts. We'll create separate videos
# for unconditional and conditional generation to better visualize the temporal
# evolution of the weather patterns.

# %%

import matplotlib.animation as animation
import matplotlib.colors
import pandas as pd

# Set the variable to visualize
video_variable = "msl"
video_var_idx = np.where(uncond_coords_list[0]["variable"] == video_variable)[0][0]

# %%
# Unconditional Generation Video
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, create a video from the unconditional generation results.

# %%

plt.style.use("dark_background")
fig_uncond = plt.figure(figsize=(12, 8))
ax_uncond = fig_uncond.add_subplot(111, projection=projection)

# Set up the first frame
data_uncond = uncond_outputs[0][0, 0, 0, video_var_idx].numpy()
norm = matplotlib.colors.Normalize(vmin=95000, vmax=105000)

img_uncond = ax_uncond.pcolormesh(
    uncond_coords_list[0]["lon"],
    uncond_coords_list[0]["lat"],
    data_uncond,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    norm=norm,
)
ax_uncond.coastlines()
ax_uncond.gridlines()
plt.colorbar(
    img_uncond,
    ax=ax_uncond,
    orientation="horizontal",
    shrink=0.5,
    pad=0.05,
    label="MSL (Pa)",
)

lead_time = uncond_coords_list[0]["lead_time"][0]
hours = int(lead_time / np.timedelta64(1, "h"))
time_str = pd.Timestamp(
    uncond_coords_list[0]["time"][0] + uncond_coords_list[0]["lead_time"][0]
).strftime("%Y-%m-%d %H:%M")
title_uncond = ax_uncond.set_title(
    f"Unconditional Generation: {video_variable} +{hours:03d}h ({time_str})"
)

fig_uncond.tight_layout()


def update_uncond(frame):
    """Update unconditional animation frame"""
    data = uncond_outputs[frame][0, 0, 0, video_var_idx].numpy()
    img_uncond.set_array(data.ravel())

    lead_time = uncond_coords_list[frame]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    time_str = pd.Timestamp(
        uncond_coords_list[frame]["time"][0] + uncond_coords_list[frame]["lead_time"][0]
    ).strftime("%Y-%m-%d %H:%M")
    title_uncond.set_text(
        f"Unconditional Generation: {video_variable} +{hours:03d}h ({time_str})"
    )
    return [img_uncond, title_uncond]


# Create animation
print("Creating unconditional video...")
anim_uncond = animation.FuncAnimation(
    fig_uncond, update_uncond, frames=len(uncond_outputs), interval=500, blit=True
)

# Save video
writer = animation.FFMpegWriter(fps=2)
anim_uncond.save("outputs/19_cbottle_video_unconditional.mp4", writer=writer, dpi=100)
plt.close()
print("Unconditional video saved to outputs/19_cbottle_video_unconditional.mp4")

# %%
# Conditional Generation Video
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now create a video from the conditional (ERA5-based) generation results.

# %%

fig_cond = plt.figure(figsize=(12, 8))
ax_cond = fig_cond.add_subplot(111, projection=projection)

# Set up the first frame
data_cond = cond_outputs[0][0, 0, 0, video_var_idx].numpy()

img_cond = ax_cond.pcolormesh(
    cond_coords_list[0]["lon"],
    cond_coords_list[0]["lat"],
    data_cond,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    norm=norm,
)
ax_cond.coastlines()
ax_cond.gridlines()
plt.colorbar(
    img_cond, ax=ax_cond, orientation="horizontal", shrink=0.5, pad=0.05, label="MSL (Pa)"
)

lead_time = cond_coords_list[0]["lead_time"][0]
hours = int(lead_time / np.timedelta64(1, "h"))
time_str = pd.Timestamp(
    cond_coords_list[0]["time"][0] + cond_coords_list[0]["lead_time"][0]
).strftime("%Y-%m-%d %H:%M")
title_cond = ax_cond.set_title(
    f"Conditional Generation (ERA5): {video_variable} +{hours:03d}h ({time_str})"
)

fig_cond.tight_layout()


def update_cond(frame):
    """Update conditional animation frame"""
    data = cond_outputs[frame][0, 0, 0, video_var_idx].numpy()
    img_cond.set_array(data.ravel())

    lead_time = cond_coords_list[frame]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    time_str = pd.Timestamp(
        cond_coords_list[frame]["time"][0] + cond_coords_list[frame]["lead_time"][0]
    ).strftime("%Y-%m-%d %H:%M")
    title_cond.set_text(
        f"Conditional Generation (ERA5): {video_variable} +{hours:03d}h ({time_str})"
    )
    return [img_cond, title_cond]


# Create animation
print("Creating conditional video...")
anim_cond = animation.FuncAnimation(
    fig_cond, update_cond, frames=len(cond_outputs), interval=500, blit=True
)

# Save video
anim_cond.save("outputs/19_cbottle_video_conditional.mp4", writer=writer, dpi=100)
plt.close()
print("Conditional video saved to outputs/19_cbottle_video_conditional.mp4")

# %%
# Visualizing Different Variables
# --------------------------------
# The CBottleVideo model outputs 45 different weather variables. You can easily
# change the variable being visualized by modifying the `video_variable` parameter
# above. Here are some interesting variables to try:
#
# - "msl": Mean sea level pressure
# - "t2m": 2-meter temperature
# - "tcwv": Total column water vapor
# - "u10m", "v10m": 10-meter wind components
# - "tpf": Total precipitation flux
# - "sst": Sea surface temperature
#
# Simply change `video_variable = "msl"` to any of these and re-run the video
# creation cells to generate animations for different weather fields!
