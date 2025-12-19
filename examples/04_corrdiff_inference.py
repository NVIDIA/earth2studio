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
Generative Downscaling
========================

Generative downscaling over Taiwan using CorrDiff diffusion model.

This example will demonstrate how to user Nvidia's CorrDiff model, trained for
predicting weather over Taiwan, to perform generative downscaling from quarter degree
global forecast data to ~3km.

This checkpoint was trained on ERA5 data and WRF data that spans 2018-2021 at one hour
time resolution. In this example, we demonstrate an application to GFS data for a typhoon
super-resolution from 2023. The model's performance on GFS data and on data from this year
has not been evaluated.

In this example you will learn:

- Creating a custom workflow for running CorrDiff inference
- Creating a data-source for CorrDiff's input
- Initializing and running CorrDiff diagnostic model
- Post-processing results.
"""
# /// script
# dependencies = [
#   "earth2studio[corrdiff] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Creating a Simple CorrDiff Workflow
# -----------------------------------
#
# As usual, we start with creating a simple workflow to run CorrDiff in. To maximize the
# generalization of this workflow, we use dependency injection following the pattern
# provided inside :py:obj:`earth2studio.run`. Since CorrDiff is a diagnostic model, this
# workflow won't predict a time-series, rather just an instantaneous prediction.
#
#
# For this workflow, we specify
#
# - time: Input list of datetimes / strings to run inference for
# - corrdiff: The initialized CorrDiffTaiwan model
# - data: Initialized data source to fetch initial conditions from
# - io: IOBackend
# - number_of_samples: Number of samples to generate from the model

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from loguru import logger

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array


def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    corrdiff: CorrDiffTaiwan,
    data: DataSource,
    io: IOBackend,
    number_of_samples: int = 1,
) -> IOBackend:
    """CorrDiff infernce workflow

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    corrdiff : CorrDiffTaiwan
        CorrDiff mode
    data : DataSource
        Data source
    io : IOBackend
        IO object
    number_of_samples : int, optional
        Number of samples to generate, by default 1

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running corrdiff inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    corrdiff = corrdiff.to(device)
    # Update the number of samples for corrdiff to generate
    corrdiff.number_of_samples = number_of_samples

    # Fetch data from data source and load onto device
    time = to_time_array(time)
    x, coords = prep_data_array(
        data(time, corrdiff.input_coords()["variable"]), device=device
    )
    x, coords = map_coords(x, coords, corrdiff.input_coords())

    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    output_coords = corrdiff.output_coords(corrdiff.input_coords())
    total_coords = OrderedDict(
        {
            "time": coords["time"],
            "sample": output_coords["sample"],
            "lat": output_coords["lat"],
            "lon": output_coords["lon"],
        }
    )
    io.add_array(total_coords, output_coords["variable"])

    logger.info("Inference starting!")
    x, coords = corrdiff(x, coords)
    io.write(*split_coords(x, coords))

    logger.success("Inference complete")
    return io


# %%
# Set Up
# ------
# With the workflow defined, the next step is initializing the needed components from
# Earth-2 studio
#
# It's clear we need the following:
#
# - Diagnostic Model: CorrDiff model for Taiwan :py:class:`earth2studio.models.dx.CorrDiffTaiwan`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#

# %%
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend

# Create CorrDiff model
package = CorrDiffTaiwan.load_default_package()
corrdiff = CorrDiffTaiwan.load_model(package)

# Create the data source
data = GFS()

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
# For the inference we will predict 1 sample for a particular timestamp representing
# Typhoon Koinu.

# %%
io = run(["2023-10-04T18:00:00"], corrdiff, data, io, number_of_samples=1)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a great library for plotting
# fields on projections of a sphere.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

projection = ccrs.LambertConformal(
    central_longitude=io["lon"][:].mean(),
)

fig = plt.figure(figsize=(4 * 8, 8))

ax0 = fig.add_subplot(1, 3, 1, projection=projection)
c = ax0.pcolormesh(
    io["lon"],
    io["lat"],
    io["mrr"][0, 0],
    transform=ccrs.PlateCarree(),
    cmap="inferno",
)
plt.colorbar(c, ax=ax0, shrink=0.6, label="mrr dBz")
ax0.coastlines()
ax0.gridlines()
ax0.set_title("Radar Reflectivity")

ax1 = fig.add_subplot(1, 3, 2, projection=projection)
c = ax1.pcolormesh(
    io["lon"],
    io["lat"],
    io["t2m"][0, 0],
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
)
plt.colorbar(c, ax=ax1, shrink=0.6, label="K")
ax1.coastlines()
ax1.gridlines()
ax1.set_title("2-meter Temperature")

ax2 = fig.add_subplot(1, 3, 3, projection=projection)
c = ax2.pcolormesh(
    io["lon"],
    io["lat"],
    np.sqrt(io["u10m"][0, 0] ** 2 + io["v10m"][0, 0] ** 2),
    transform=ccrs.PlateCarree(),
    cmap="Greens",
)
plt.colorbar(c, ax=ax2, shrink=0.6, label="w10m m s^-1")
ax2.coastlines()
ax2.gridlines()
ax2.set_title("10-meter Wind Speed")

plt.savefig("outputs/04_corr_diff_prediction.jpg")
