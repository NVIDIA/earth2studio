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
Running Ensemble Inference
==========================

Simple ensemble inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
ensemble forecast using one of the built in models of Earth-2 Inference
Studio.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Select a perturbation method
- Running a simple built in workflow
- Post-processing results
"""

# %%
# Creating a Simple Ensemble Workflow
# -----------------------------------
#
# To start lets begin with creating a simple ensemble workflow to use. We encourage
# users to explore and experiment with their own custom workflows that borrow ideas from
# built in workflows inside :py:obj:`earth2studio.run` or the examples.
#
# Creating our own generalizable ensemble workflow is easy when we rely on the component
# interfaces defined in Earth2Studio (use dependency injection). Here we create a run
# method that accepts the following:
#
# - time: Input list of datetimes / strings to run inference for
# - nsteps: Number of forecast steps to predict
# - nensemble: Number of ensembles to run for
# - prognostic: Our initialized prognostic model
# - perturbation_method: Our initialized pertubation method
# - data: Initialized data source to fetch initial conditions from
# - io: IOBackend

# %%
# Set Up
# ------
#
# We need the following:
#
# - Prognostic Model: Use the built in FourCastNet model :py:class:`earth2studio.models.px.FCN`.
# - perturbation_method: Use the Spherical Gaussian Method :py:class:`earth2studio.perturbation.SphericalGaussian`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Lets save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
model = FCN.load_model(package)

# Instantiate the pertubation method
sg = SphericalGaussian(noise_amplitude=0.15)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1}
io = ZarrBackend(file_name="outputs/02_ensemble_sg.zarr", chunks=chunks)

# %%
# Execute the Workflow
# --------------------
# With all componments intialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 steps (for FCN, this is 60 hours) with 8 ensemble
# members which will be ran in 2 batches with batch size 4.
# %%

nsteps = 10
nensemble = 8
batch_size = 4
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    batch_size=batch_size,
    perturbation_method=sg,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
    device="cuda:0",
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a greate library for plotting
# fields on projects of a sphere.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = "2024-01-01"


def plot_(axi, data, title, cmap):
    """Convenience function for plotting pcolormesh."""
    # Plot the field using pcolormesh
    im = axi.pcolormesh(
        io["lon"][:],
        io["lat"][:],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    plt.colorbar(im, ax=axi, shrink=0.6, pad=0.04)
    # Set title
    axi.set_title(title)
    # Add coastlines and gridlines
    axi.coastlines()
    axi.gridlines()


for variable, cmap in zip(["t2m", "tcwv"], ["coolwarm", "Blues"]):
    step = 0  # lead time = 24 hrs

    plt.close("all")
    # Create a Robinson projection
    projection = ccrs.Robinson()

    # Create a figure and axes with the specified projection
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(16, 3)
    )

    plot_(
        ax1,
        io[variable][0, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {0}",
        cmap,
    )
    plot_(
        ax2,
        io[variable][1, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {1}",
        cmap,
    )
    plot_(
        ax3,
        np.std(io[variable][:, 0, step], axis=0),
        f"{forecast} - Lead time: {6*step}hrs - Std",
        cmap,
    )

    plt.savefig(f"outputs/02_{forecast}_{variable}_{step}_ensemble.jpg")
