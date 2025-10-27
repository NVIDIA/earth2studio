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
Running StormCast Inference
===========================

Basic StormCast inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
basic determinstic forecast using StormCast. For details about the stormcast model,
see

 - https://arxiv.org/abs/2408.10958

"""
# /// script
# dependencies = [
#   "earth2studio[data,stormcast] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, let's take a look at the most basic:
# :py:meth:`earth2studio.run.deterministic`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :start-after: # sphinx - deterministic start
#    :end-before: # sphinx - deterministic end

# %%
# Thus, we need the following:
#
# - Prognostic Model: Use the built in StormCast Model :py:class:`earth2studio.models.px.StormCast`.
# - Datasource: Pull data from the HRRR data api :py:class:`earth2studio.data.HRRR`.
# - IO Backend: Let's save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# StormCast also requires a conditioning data source. We use a forecast data source here,
# GFS_FX :py:class:`earth2studio.data.GFS_FX` which is the default, but a non-forecast
# data source such as ARCO could also be used with appropriate time stamps.

# %%
from datetime import datetime, timedelta

from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast

# Load the default model package which downloads the check point from NGC
# Use the default conditioning data source GFS_FX
package = StormCast.load_default_package()
model = StormCast.load_model(package)

# Create the data source
data = HRRR()

# Create the IO handler, store in memory
io = ZarrBackend()

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 4 hours

# %%
import earth2studio.run as run

nsteps = 4
today = datetime.today() - timedelta(days=1)
date = today.isoformat().split("T")[0]
io = run.deterministic([date], nsteps, model, data, io)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a great library for plotting
# fields on projections of a sphere. Here we will just plot the temperature at 2 meters
# (t2m) 4 hours into the forecast.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = f"{date}"
variable = "t2m"
step = 4  # lead time = 1 hr

plt.close("all")

# Create a correct Lambert Conformal projection
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 6))

# Plot the field using pcolormesh
im = ax.pcolormesh(
    model.lon,
    model.lat,
    io[variable][0, step],
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
)

# Set state lines
ax.add_feature(
    cartopy.feature.STATES.with_scale("50m"), linewidth=0.5, edgecolor="black", zorder=2
)

# Set title
ax.set_title(f"{forecast} - Lead time: {step}hrs")

# Add coastlines and gridlines
ax.coastlines()
ax.gridlines()
plt.savefig(f"outputs/09_{date}_t2m_prediction.jpg")
