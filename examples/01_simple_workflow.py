# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
Running Simple Inference
========================

Simple deterministic inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
simple determinstic forecast using one of the built in models of Earth-2 Inference
Studio.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Running a simple built in workflow
- Post-processing results
"""

# %%
# Set Up
# ------
# All workflows inside Earth-2 Inference Studio require constructed components to be
# handed to them. In this example, lets take a look at the most basic:
# :py:meth:`earth2studio.run.deterministic`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :lines: 35-42

# %%
# Thus we need the following:
#
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Lets save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
import numpy as np
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.models.px import FCN
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
model = FCN.load_model(package)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
io = ZarrBackend()

# %%
# Execute the Workflow
# --------------------
# With all componments intialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for two days (these will get executed as a batch) for
# 20 forecast steps which is 5 days.

# %%
import earth2studio.run as run

nsteps = 20
io = run.deterministic(["2024-01-01"], nsteps, model, data, io)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a greate library for plotting
# fields on projects of a sphere. Here we will just plot the temperature at 2 meters
# (t2m) 1 day into the forecast.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import os

os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

forecast = "2024-01-01"
variable = "t2m"
step = 4  # lead time = 24 hrs

plt.close("all")
# Create a Robinson projection
projection = ccrs.Robinson()

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 6))

# Plot the field using pcolormesh
im = ax.pcolormesh(
    io["lon"][:],
    io["lat"][:],
    io[variable][0, step],
    transform=ccrs.PlateCarree(),
    cmap="coolwarm",
)

# Set title
ax.set_title(f"{forecast} - Lead time: {6*step}hrs")

# Add coastlines and gridlines
ax.coastlines()
ax.gridlines()
plt.savefig("outputs/01_t2m_prediction.jpg")
