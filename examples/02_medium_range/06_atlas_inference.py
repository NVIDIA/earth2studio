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
Running Atlas Inference
=======================

Deterministic inference using the Atlas prognostic model.

This example demonstrates how to run a single-member forecast using the Atlas model,
a generative AI weather model that uses stochastic interpolants and autoencoders to
produce 6-hour forecasts on a 0.25 degree global grid. Atlas requires two input lead
times (t-6h and t) and maintains an internal latent state for autoregressive rollouts.

In this example you will learn:

- How to instantiate the Atlas prognostic model
- How to run a single-member forecast with Atlas
- How to visualize predicted 10m u-wind and total column water vapour
"""
# /// script
# dependencies = [
#   "torch==2.9.1", # Match lock file to avoid torch-harmonics issue
#   "earth2studio[atlas] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, we will use the deterministic workflow
# :py:meth:`earth2studio.run.deterministic`. Note the model itself is not
# deterministic, but we use this workflow for simplicity since we're just
# generating a single ensemble member.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :start-after: # sphinx - deterministic start
#    :end-before: # sphinx - deterministic end

# %%
# We need the following:
#
# - Prognostic Model: Use the Atlas model :py:class:`earth2studio.models.px.Atlas`.
# - Datasource: Pull data from the ARCO data api :py:class:`earth2studio.data.ARCO`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# .. note::
#   Atlas requires two input lead times (t-6h and t) to produce a forecast.
#   The deterministic workflow handles this automatically via the model's
#   ``input_coords`` definition.
#
# .. note::
#   Atlas was trained on ERA5 data and in-filled NaNs in SST over landmasses with a value
#   of 0 K using the ERA5 land-sea mask. If you are using a different SST dataset, you will
#   need to in-fill the NaNs using the appropriate land-sea mask.
#   

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import torch
import numpy as np

from earth2studio.data.utils import fetch_data
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Atlas

# Load the default model package which downloads the checkpoint from HuggingFace
package = Atlas.load_default_package()
model = Atlas.load_model(package)

# Create the data source
data = ARCO()

# Create the IO handler
io = ZarrBackend()

# %%
# Manual Forward Pass
# -------------------
# For more control over the inference loop, Atlas can be called directly using
# :py:meth:`earth2studio.data.utils.fetch_data` for initial conditions and
# :py:meth:`earth2studio.models.px.Atlas.prep_next_input` to advance the sliding
# window between steps. This is useful when you need access to intermediate tensors
# or want to customize the rollout logic.
#
# .. note::
#   For autoregressive rollouts longer than one step, prefer
#   ``create_iterator`` which correctly manages the internal latent state.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get model input coordinate requirements
input_coords = model.input_coords()

# Fetch initial conditions matching the model's expected variables and lead times
time = np.array([np.datetime64("2024-01-01T00:00")])
x, coords = fetch_data(
    source=data,
    time=time,
    variable=input_coords["variable"],
    lead_time=input_coords["lead_time"],
    device=device,
)

# Add a batch dimension
x = x.unsqueeze(0)
coords["batch"] = np.arange(1)
coords.move_to_end("batch", last=False)

# Run two forward steps manually
n_manual_steps = 2
for step in range(n_manual_steps):
    y, y_coords = model(x, coords)
    lead_hrs = y_coords["lead_time"][0] / np.timedelta64(1, "h")
    print(f"Step {step + 1}: lead_time = {lead_hrs:.0f}h, shape = {y.shape}")
    x, coords = model.prep_next_input(y, y_coords, x, coords)


# %%
# Using the model iterator interface
# --------------------
# For a less verbose approach, we can use the model iterator interface directly,
# which automatically handles the internal latent state during autoregressive rollout.
# Simply pass the model and other instantiated components to the workflow, which will
# automatically use the iterator interface internally and return outputs to the IO handler.
# %%
import earth2studio.run as run

nsteps = 20
io = run.deterministic(
    ["2024-01-01"],
    nsteps,
    model,
    data,
    io,
    output_coords={"variable": np.array(["u10m", "tcwv"])},
)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# The last step is to post process our results. We will plot the predicted 10m u-wind
# component (u10m) and total column water vapour (tcwv) at day 3 of the forecast.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = "2024-01-01"
step = 12  # lead time = 72 hrs (day 3)

plt.close("all")
projection = ccrs.Robinson()

fig, ax = plt.subplots(1, 2, subplot_kw={"projection": projection}, figsize=(16, 5))

# Plot u10m
im0 = ax[0].pcolormesh(
    io["lon"][:],
    io["lat"][:],
    io["u10m"][0, step],
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-20,
    vmax=20,
)
ax[0].set_title(f"{forecast} - u10m - Lead time: {6*step}hrs")
ax[0].coastlines()
ax[0].gridlines()
plt.colorbar(im0, ax=ax[0], shrink=0.6, pad=0.04, label="m/s")

# Plot tcwv
im1 = ax[1].pcolormesh(
    io["lon"][:],
    io["lat"][:],
    io["tcwv"][0, step],
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    vmin=0,
    vmax=70,
)
ax[1].set_title(f"{forecast} - tcwv - Lead time: {6*step}hrs")
ax[1].coastlines()
ax[1].gridlines()
plt.colorbar(im1, ax=ax[1], shrink=0.6, pad=0.04, label="kg/m²")

plt.tight_layout()
plt.savefig("outputs/06_atlas_u10m_tcwv.jpg", dpi=300)
