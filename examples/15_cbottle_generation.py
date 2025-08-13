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
CBottle Data Generation and Infilling
=====================================

Climate in a Bottle (cBottle) inference workflows for global weather data synthesis.

This example will demonstrate the cBottle diffusion model data source and infilling
diagnostic model for generating global climate and weather data. Both the cBottle data
source and infilling diagnostic use the same diffusion model but the sampling procedure
is different enabling two unique modes of interaction.

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

In this example you will learn:

- Generating synthetic climate data with cBottle data source
- Instantiating cBottle infill diagnostic model
- Creating a simple infilling inference workflow
"""
# /// script
# dependencies = [
#   "earth2studio[cbottle] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# For this example we will use the cBottle data source and infill diagnostic. Unlike
# other data sources the cBottle3D data source needs to be loaded similar to a
# prognostic or diagnostic model.

# %%
# Thus, we need the following:
#
# - Datasource: Generate data from the CBottle3D data api :py:class:`earth2studio.data.CBottle3D`.
# - Datasource: Pull data from the WeatherBench2 data api :py:class:`earth2studio.data.WB2ERA5`.
# - Diagnostic Model: Use the built in CBottle Infill Model :py:class:`earth2studio.models.dx.CBottleInfill`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import torch

from earth2studio.data import WB2ERA5, CBottle3D
from earth2studio.models.dx import CBottleInfill

# Load the default model package which downloads the check point from NGC
package = CBottle3D.load_default_package()
cbottle_ds = CBottle3D.load_model(package)
# This is an AI data source, so also move it to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cbottle_ds = cbottle_ds.to(device)

# Create the ground truth data source
era5_ds = WB2ERA5()

# %%
# Generating Synthetic Weather Data
# ---------------------------------
# Once loaded, generating data from cBottle is as easy as any other data source.
# Under the hood the model is conditioned on the timestamp requested as well as a
# mid-month SST field which is internally handle for users but limits the range of the
# data source to years between 1970 and 2022.
#
# Note that this diffusion model is stochastic, so querying the same timestamp will
# generate different fields that are reflective of the requested time and SST state.
# One can use `set_seed` for reproducibility.

# %%

from datetime import datetime

n_samples = 5
timestamp = datetime(2022, 9, 5)

# Fetch the ground truth
era5_da = era5_ds([timestamp], ["msl", "tcwv"])
# Generate some samples from cBottle
cbottle_da = cbottle_ds([timestamp for i in range(n_samples)], ["msl", "tcwv"])

print(era5_da)
print(cbottle_da)

# %%
# Post Processing CBottle Data
# ----------------------------
# Let's visualize this data to better understand what the cBottle data source is able to
# provide.
# It is clear that each sample is indeed unique, yet remains physically realizable.
# In other words the cBottle data source can be used to create climates that
# do not exist but could based on the conditional distribution learned from the training
# data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

variable = "tcwv"

plt.close("all")
projection = ccrs.Orthographic(central_longitude=300.0)

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(2, 3, subplot_kw={"projection": projection}, figsize=(11, 6))
ax = ax.flatten()

ax[0].pcolormesh(
    era5_da.coords["lon"],
    era5_da.coords["lat"],
    era5_da.sel(variable=variable).isel(time=0),
    transform=ccrs.PlateCarree(),
    cmap="cubehelix",
)
ax[0].set_title("ERA5")

for i in range(n_samples):
    ax[i + 1].pcolormesh(
        cbottle_da.coords["lon"],
        cbottle_da.coords["lat"],
        cbottle_da.sel(variable=variable).isel(time=i),
        transform=ccrs.PlateCarree(),
        cmap="cubehelix",
        vmin=0,
        vmax=90,
    )
    ax[i + 1].set_title(f"CBottle Sample {i}")

for ax0 in ax:
    ax0.coastlines()
    ax0.gridlines()

plt.tight_layout()
plt.savefig("outputs/15_tcwv_cbottle_datasource.jpg")

# %%
# Variable Infilling with CBottleInfill Diagnostic
# ------------------------------------------------
# Next lets look at using the same model but for variable infilling.
# CBottleInfill allows users to generate global weather fields like the data source but
# condition it on a set of input fields that can be configured.
# This means that this diagnostic is extremely flexible and can be used with all types
# of data sources and models.
#
# To demonstrate this lets consider two instances of the infilling diagnostic with a
# different set of inputs and then compare the resulting infilled variables.
# Note that the outputs of both configurations are the same size with the same
# variables.

# %%
import numpy as np

from earth2studio.data.utils import fetch_data

# Input variables
input_variables = ["u10m", "v10m"]

# Load the default model package which downloads the check point from NGC
package = CBottleInfill.load_default_package()
model = CBottleInfill.load_model(package, input_variables=input_variables)
model = model.to(device)

model.set_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

times = np.array([timestamp] * n_samples, dtype="datetime64[ns]")
x, coords = fetch_data(era5_ds, times, input_variables, device=device)
output_0, output_coords = model(x, coords)
print(output_0.shape)

# %%
# Now repeat the process above but with an expanded set of variables.
# In this instance we provide a lot more data to the model to condition it with more
# information.

# %%
input_variables = [
    "u10m",
    "v10m",
    "t2m",
    "msl",
    "z50",
    "u50",
    "v50",
    "z500",
    "u500",
    "v500",
    "z1000",
    "u1000",
    "v1000",
]

# Load the default model package which downloads the check point from NGC
package = CBottleInfill.load_default_package()
model = CBottleInfill.load_model(package, input_variables=input_variables)
model = model.to(device)

model.set_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

x, coords = fetch_data(era5_ds, times, input_variables, device=device)
output_1, output_coords = model(x, coords)
print(output_1.shape)

# %%
# Post Processing CBottleInfill
# -----------------------------
# To post process the results, we take a look at a infilled variable, total column water
# vapour.
# Compared to the samples from the CBottle3D, the results are much more aligned with
# the ground truth since the infill model is sampling a conditional distribution.
# Additionally, the model provided more variables is better aligned with the ground
# truth due the additional information provided.

# %%

variable = "tcwv"
var_idx = np.where(output_coords["variable"] == "tcwv")[0][0]
era5_data, _ = fetch_data(era5_ds, times[:1], [variable], device=device)

plt.close("all")
projection = ccrs.Mollweide(central_longitude=0)

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(2, 3, subplot_kw={"projection": projection}, figsize=(10, 6))


def plot_contour(
    ax0: plt.axes,
    data: torch.Tensor,
    cmap: str = "jet",
    vrange: tuple[int, int] = (0, 90),
) -> None:
    """Contour helper"""
    ax0.contourf(
        output_coords["lon"],
        output_coords["lat"],
        data.cpu(),
        vmin=vrange[0],
        vmax=vrange[1],
        transform=ccrs.PlateCarree(),
        levels=20,
        cmap=cmap,
    )
    ax0.coastlines()
    ax0.gridlines()


plot_contour(ax[0, 0], era5_data[0, 0, 0])
plot_contour(ax[0, 1], torch.mean(output_0[:, 0, var_idx], axis=0))
plot_contour(ax[0, 2], torch.mean(output_1[:, 0, var_idx], axis=0))
plot_contour(
    ax[1, 1], torch.std(output_0[:, 0, var_idx], axis=0), cmap="inferno", vrange=(0, 10)
)
plot_contour(
    ax[1, 2], torch.std(output_1[:, 0, var_idx], axis=0), cmap="inferno", vrange=(0, 10)
)

ax[0, 0].set_title("ERA5")
ax[0, 1].set_title("3 Input Variables Mean")
ax[0, 2].set_title("13 Input Variables Mean")
ax[1, 1].set_title("3 Input Variables Std")
ax[1, 2].set_title("13 Input Variables Std")

plt.tight_layout()
plt.savefig("outputs/15_tcwv_cbottle_infill.jpg")
