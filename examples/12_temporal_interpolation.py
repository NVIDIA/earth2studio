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
Temporal Interpolation
======================

Temporal Interpolation inference using InterpModAFNO model.

This example demonstrates how to use the InterpModAFNO model to interpolate
forecasts from a base model to a finer time resolution.
Many of the existing prognostic models have a step size of 6 hours which may prove
insufficient for some applications.
InterpModAFNO provides a AI driven method for getting hourly resolution given 6 hour
predictions.

In this example you will learn:

- How to load a base prognostic model
- How to load the InterpModAFNO model
- How to run the interpolation model
- How to visualize the results
"""

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
# - Interpolation Model: :py:class:`earth2studio.models.px.InterpModAFNO`.
# - Prognostic Base Model: Use SFNO model :py:class:`earth2studio.models.px.SFNO`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.

# %%
import os

import matplotlib.pyplot as plt

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO, InterpModAFNO

# Create output directory
os.makedirs("outputs", exist_ok=True)

sfno_package = SFNO.load_default_package()
base_model = SFNO.load_model(sfno_package)

# Load the interpolation model
interp_package = InterpModAFNO.load_default_package()
interp_model = InterpModAFNO.load_model(interp_package)
interp_model.px_model = base_model  # Set the base model

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
