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
Temporal Interpolation of Forecasts
==================================

This example demonstrates how to use the ForecastInterpolation model to interpolate
forecasts from a base model to a finer time resolution.

In this example you will learn:

- How to load a base prognostic model (e.g., SFNO)
- How to load the ForecastInterpolation model
- How to run the interpolation model to get forecasts at a finer time resolution
- How to visualize the results
"""

# %%
# Set Up
# ------
# First, let's import the necessary modules and set up our environment.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO, ForecastInterpolation

# Create output directory
os.makedirs("outputs", exist_ok=True)

# %%
# Load Models
# -----------
# We'll use SFNO as our base model and the ForecastInterpolation model to
# interpolate its output to a finer time resolution.

# %%
# Load the base model (SFNO)
sfno_package = SFNO.load_default_package()
base_model = SFNO.load_model(sfno_package)

# Load the interpolation model
interp_package = ForecastInterpolation.load_default_package()
interp_model = ForecastInterpolation.load_model(interp_package, fc_model=base_model)

# Create the data source
data = GFS()

# Create the IO handler
io = ZarrBackend()

# %%
# Run the Interpolation Model
# ---------------------------
# Now we'll run the interpolation model to get forecasts at a finer time resolution.
# The base model (SFNO) produces forecasts at 6-hour intervals, and the
# interpolation model will interpolate to 1-hour intervals.

# %%
# Define forecast parameters
forecast_date = "2024-01-01"
nsteps = 1  # Number of forecast steps from the base model
# The interpolation model will automatically interpolate between these steps

# Run the model
from earth2studio.run import deterministic

io = deterministic([forecast_date], nsteps, interp_model, data, io)

# %%
# Visualize Results
# ----------------
# Let's visualize the temperature at 2 meters (t2m) at each time step
# and save them as separate files.

# %%
# Get the number of time steps
n_steps = io["t2m"].shape[1]

# Create a separate plot for each time step
for step in range(n_steps):
    # Create a new figure for each time step
    plt.figure(figsize=(10, 6))

    # Create the plot - flip the data vertically and adjust extent to rotate 180 degrees
    im = plt.imshow(
        np.flipud(io["t2m"][0, step]),  # Flip the data vertically
        cmap="Spectral_r",
        origin="lower",
        extent=[0, 360, -90, 90],  # Keep the same extent
        aspect="auto",
    )

    # Set title
    plt.title(f"Temperature at 2m - Step: {step}hrs")

    # Add colorbar
    plt.colorbar(im, label="Temperature (K)")

    # Save the figure
    plt.savefig(f"outputs/12_t2m_step_{step}.jpg")
    plt.close()  # Close the figure to free memory
