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
Running StormCast Ensemble Inference
===============================

Ensemble StormCast inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
ensemble forecast using StormCast. For details about the stormcast model,
see

 - https://arxiv.org/abs/2408.10958

"""
# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, let's take a look at the most basic ensemble workflow:
# :py:meth:`earth2studio.run.ensemble`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :start-after: # sphinx - ensemble start
#    :end-before: # sphinx - ensemble end

# %%
# Thus, we need the following:
#
# - Prognostic Model: Use the built in StormCast Model :py:class:`earth2studio.models.px.StormCast`.
# - perturbation_method: Use the Zero Method :py:class:`earth2studio.perturbation.Zero`. We will not
#    perturb the initial data because StormCast has stochastic generation of  ensemble members.
# - Datasource: Pull data from the HRRR data api :py:class:`earth2studio.data.HRRR`.
# - IO Backend: Let's save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# StormCast also requires a conditioning data source. We use a forecast data source here,
# ARCO :py:class:`earth2studio.data.ARCO`, but a forecast data source such as GFS_FX
# could also be used with appropriate time stamps.

# %%
import numpy as np
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import ARCO, HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero

# Load the default model package which downloads the check point from NGC
package = StormCast.load_default_package()
model = StormCast.load_model(package)

# Instantiate the (Zero) perturbation method
z = Zero()

# Create the data source
data = HRRR()

# Create and set the conditioning data source
conditioning_data_source = ARCO()
model.conditioning_data_source = conditioning_data_source

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
nensemble = 4
batch_size = 2

date = "2022-11-04T21:00:00"
io = run.ensemble(
    [date],
    nsteps,
    nensemble,
    model,
    data,
    io,
    z,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "refc"])},
)

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
step = nsteps  # 4 hours, since lead_time = 1 hr


def plot_(axi, data, title, cmap, vmin=None, vmax=None):
    """Convenience function for plotting pcolormesh."""
    # Plot the field using pcolormesh
    im = axi.pcolormesh(
        io["lon"][:],
        io["lat"][:],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=axi, shrink=0.6, pad=0.04)
    # Set title
    axi.set_title(title)

    # Add coastlines and gridlines
    axi.coastlines()
    axi.gridlines()

    # Set state lines
    axi.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )


# Create a correct Lambert Conformal projection
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

# Plot 2-meter temperature
variable = "t2m"
cmap = "Spectral_r"
x = io[variable]

plt.close("all")
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(20, 6)
)
plot_(
    ax1,
    x[0, 0, step],
    f"{forecast} - Lead time: {step}hrs - Member: {0}",
    cmap,
)
plot_(
    ax2,
    io[variable][1, 0, step],
    f"{forecast} - Lead time: {step}hrs - Member: {1}",
    cmap,
)
plot_(
    ax3,
    x[:, 0, step].std(axis=0),
    f"{forecast} - Lead time: {step}hrs - Std",
    cmap,
)
plt.savefig(f"outputs/10_{date}_{variable}_{step}_ensemble.jpg")

# Plot refc
variable = "refc"
cmap = "gist_ncar"
x = io[variable]

plt.close("all")
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(20, 6)
)
plot_(
    ax1,
    np.where(x[0, 0, step] > 0, x[0, 0, step], np.nan),
    f"{forecast} - Lead time: {step}hrs - Member: {0}",
    cmap,
    vmin=0,
    vmax=60,
)
plot_(
    ax2,
    np.where(x[1, 0, step] > 0, x[1, 0, step], np.nan),
    f"{forecast} - Lead time: {step}hrs - Member: {1}",
    cmap,
    vmin=0,
    vmax=60,
)
plot_(
    ax3,
    np.where(x[:, 0, step].mean(axis=0) > 0, x[:, 0, step].std(axis=0), np.nan),
    f"{forecast} - Lead time: {step}hrs - Std",
    cmap,
    vmin=0,
    vmax=60,
)
plt.savefig(f"outputs/10_{date}_{variable}_{step}_ensemble.jpg")
