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
- Running a simple built in workflow for ensembling
- Post-processing results
"""

# /// script
# dependencies = [
#   "torch==2.13.0", # Match torch-harmonics examples
#   "earth2studio[fcn,perturbation] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "scipy>=1.15.2",
#   "cartopy",
# ]
#
# [tool.uv]
# no-build-isolation-package = ["torch-harmonics"]
# ///

# %%
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be handed to
# them. In this example, let's take a look at the ensemble workflow:
# [`earth2studio.run.ensemble`](https://nvidia.github.io/earth2studio/main/modules/generated/workflows/ensemble/#earth2studio.run.ensemble).
#
# Thus, we need the following:
#
# - Prognostic Model: Use the built in FourCastNet model [`earth2studio.models.px.FCN`][earth2studio.models.px.FCN].
# - Perturbation Method: Use the Spherical Gaussian Method [`earth2studio.perturbation.SphericalGaussian`][earth2studio.perturbation.SphericalGaussian].
# - Datasource: Pull data from the GFS data api [`earth2studio.data.GFS`][earth2studio.data.GFS].
# - IO Backend: Save the outputs into a Zarr store [`earth2studio.io.ZarrBackend`][earth2studio.io.ZarrBackend].

# %% tags=["e2sg-profile:setup"]
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble
from earth2studio.utils.time import to_time_array

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
model = FCN.load_model(package)

# Instantiate the pertubation method
sg = SphericalGaussian(noise_amplitude=0.15)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1, "lead_time": 1}
io = ZarrBackend(
    file_name="outputs/03_ensemble_sg.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)

# %%
# Fetch Data
# ----------
# You can easily fetch raw Xarray data from an initial condition data source with
# a simple call. By default, this caches the data locally on your machine, so you
# won't have to re-download it if you access it again or use it in an inference
# pipeline.

# %% tags=["e2sg-profile:setup"]
sample = data(
    to_time_array(["2024-01-01"]),
    model.input_coords()["variable"],
)
print(f"Cached GFS input shape: {sample.shape}")

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 steps (for FCN, this is 60 hours) with 8 ensemble
# members which will be ran in 2 batches with batch size 4.

# %% tags=["e2sg-profile:inference"]

nsteps = 10
nensemble = 8
batch_size = 2
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    sg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a great library for plotting
# fields on projections of a sphere.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %% tags=["e2sg-profile:plotting"]
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


for variable, cmap in zip(["tcwv"], ["Blues"]):
    step = 4  # lead time = 24 hrs

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

    plt.savefig(f"outputs/03_{forecast}_{variable}_{step}_ensemble.jpg")
