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
Temporal Interpolation
======================

Temporal Interpolation inference using InterpModAFNO and InterpCRPSDiT models.

This example demonstrates how to use the InterpModAFNO and InterpCRPSDiT models to
interpolate forecasts from a base model to a finer time resolution.
Many of the existing prognostic models have a step size of 6 hours which may prove
insufficient for some applications.
InterpModAFNO provides a AI driven method for getting hourly resolution given 6 hour
predictions.
InterpCRPSDiT is a probabilistic alternative: it pins the coarse endpoints exactly and
each ensemble member draws its own random latent for the frames in between, so the
interpolation uncertainty is sampled with an ensemble.

In this example you will learn:

- How to load a base prognostic model
- How to load the InterpModAFNO model
- How to run the interpolation model
- How to visualize the results
- How to load the InterpCRPSDiT model and produce an interpolation ensemble
"""

# /// script
# dependencies = [
#   "torch==2.13.0", # Match torch-harmonics examples
#   "earth2studio[sfno,interp-modafno,interp-crps-dit] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# First, import the necessary modules and set up our environment and load the models.
# We will use SFNO as the base prognostic model and the InterpModAFNO model to
# interpolate its output to a finer time resolution.
# The prognostic model must predict the needed variables in the interpolation model.
#
# This example needs the following:
#
# - Interpolation Model: [`earth2studio.models.px.InterpModAFNO`][earth2studio.models.px.InterpModAFNO].
# - Probabilistic Interpolation Model: [`earth2studio.models.px.InterpCRPSDiT`][earth2studio.models.px.InterpCRPSDiT].
# - Prognostic Base Model: Use SFNO model [`earth2studio.models.px.SFNO`][earth2studio.models.px.SFNO].
# - Datasource: Pull data from the GFS data api [`earth2studio.data.GFS`][earth2studio.data.GFS].

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO, InterpCRPSDiT, InterpModAFNO
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble

# Create output directory
os.makedirs("outputs", exist_ok=True)

sfno_package = SFNO.load_default_package()
px_model = SFNO.load_model(sfno_package)

# Load the interpolation model
interp_package = InterpModAFNO.load_default_package()
interp_model = InterpModAFNO.load_model(interp_package, px_model=px_model)

# Create the data source
data = GFS()

# Create the IO handler
io = ZarrBackend()

# %%
# Run the Interpolation Model
# ---------------------------
# Now run the interpolation model to get forecasts at a finer time resolution.
# The base model (SFNO) produces forecasts at 6-hour intervals, and the
# interpolation model will interpolate to 1-hour intervals.

# %%

# Define forecast parameters
forecast_date = "2024-01-01"
nsteps = 5  # Number of interpolated forecast steps

# Run the model
from earth2studio.run import deterministic

io = deterministic([forecast_date], nsteps, interp_model, data, io)

print(io.root.tree())

# %%
# Visualize Results
# -----------------
# Let's visualize the total column water vapour (tcwv) at each time step
# and save them as separate files.

# %%

# Get the number of time steps
n_steps = io["tcwv"].shape[1]

# Create a single figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
axs = axs.ravel()

# Create plots for each time step
for step in range(min([n_steps, 6])):
    im = axs[step].imshow(
        io["tcwv"][0, step], cmap="twilight_shifted", aspect="auto", vmin=0, vmax=85
    )
    axs[step].set_title(f"Water Vapour - Step: {step} hrs")
    fig.colorbar(im, ax=axs[step], label="kg/m^2")

plt.tight_layout()
# Save the figure
plt.savefig("outputs/12_tcwv_steps.jpg")

# %%
# Probabilistic Interpolation with InterpCRPSDiT
# ----------------------------------------------
# [`earth2studio.models.px.InterpCRPSDiT`][earth2studio.models.px.InterpCRPSDiT] is a
# one-shot, endpoint-pinned interpolator trained with a CRPS objective. Between two
# coarse SFNO frames it reproduces both endpoints exactly and fills in the interior
# frames with a learned correction driven by a per-member random latent.
# Because the spread is produced by the model itself, we use
# [`earth2studio.run.ensemble`][earth2studio.run.ensemble] with a
# [`Zero`][earth2studio.perturbation.Zero] (no-op) perturbation: every member shares the
# same initial condition and the diversity comes entirely from the interpolator.
# `batch_size` caps GPU memory by running members in chunks.
#
# .. note::
#
#    The default package for InterpCRPSDiT may not be hosted yet. In that case load a local
#    bundle instead with `Package("/path/to/bundle")`
#    (`from earth2studio.models.auto import Package`).

# %%
crps_package = InterpCRPSDiT.load_default_package()
crps_model = InterpCRPSDiT.load_model(crps_package, px_model=px_model)

nmembers = 4
io_crps = ensemble(
    [forecast_date],
    nsteps,
    nmembers,
    crps_model,
    data,
    ZarrBackend(),
    Zero(),
    batch_size=1,
    output_coords={"variable": np.array(["tcwv"])},
)

print(io_crps.root.tree())

# %%
# Visualize Ensemble Results
# --------------------------
# Plot the total column water vapour at the 3 hour midpoint of the first coarse
# interval for the ensemble mean and each member.

# %%
# io_crps dims: (ensemble, time, lead_time, lat, lon); hourly leads so index 3 is 3 hrs
members_mid = np.asarray(io_crps["tcwv"])[:, 0, 3]

fig, axs = plt.subplots(2, 3, figsize=(15, 6))
panels = [("Ensemble mean", members_mid.mean(0))]
panels += [(f"Member {m}", members_mid[m]) for m in range(nmembers)]
for ax, (title, fld) in zip(axs.ravel(), panels):
    im = ax.imshow(fld, cmap="twilight_shifted", aspect="auto", vmin=0, vmax=85)
    ax.set_title(f"{title} - 3 hrs")
    fig.colorbar(im, ax=ax, label="kg/m^2")
for ax in axs.ravel()[len(panels) :]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/04_tcwv_crps_ensemble.jpg")
