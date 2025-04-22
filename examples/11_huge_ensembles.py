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
Huge Ensembles (HENS) Checkpoints
=================================

Basic multi-checkpoint Huge Ensembles (HENS) inference workflow.

This example provides a basic example to load the Huge Ensemble checkpoints to perform
ensemble inference.
This notebook aims to demonstrate the foundations of running a multi-checkpoint workflow
from Earth2Studio components.
For more details about HENS, see:

- https://arxiv.org/abs/2408.03100
- https://github.com/ankurmahesh/earth2mip-fork


.. warning::

    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

For the complete HENS workflow, we encourage users to have a look at the HENS recipe
which provides a end-to-end solution to leverage HENS for downstream analysis such as
tropical cyclone tracking:

- coming soon

In this example you will learn:

- How to load the HENS checkpoints with a custom model package
- How to load the HENS perturbation method
- How to create a simple ensemble inference loop
- How to visualize results

"""

# %%
# Set Up
# ------
# First, import the necessary modules and set up our environment and load the required
# modules.
# HENS has checkpoints conveniently stored on `HuggingFace <https://huggingface.co/datasets/maheshankur10/hens/tree/main/earth2mip_prod_registry>`_
# that we will use.
# Rather than loading the default checkpoint from the original SFNO paper, create a
# model package that points to the specific HENS checkpoint we want to use instead.
#
# This example also needs the following:
#
# - Prognostic Base Model: Use SFNO model architecture :py:class:`earth2studio.models.px.SFNO`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - Perturbation Method: HENS uses a novel perturbation method :py:class:`earth2studio.perturbation.HemisphericCentredBredVector`.
# - Seeding Perturbation Method: Perturbation method to seed the Bred Vector :py:class:`earth2studio.perturbation.CorrelatedSphericalGaussian`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import SFNO
from earth2studio.perturbation import (
    CorrelatedSphericalGaussian,
    HemisphericCentredBredVector,
)
from earth2studio.run import ensemble

# Set up two model packages for each checkpoint
# Note the modification of the cache location to avoid overwriting
model_package_1 = Package(
    "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed102",
    cache_options={
        "cache_storage": Package.default_cache("hens_1"),
        "same_names": True,
    },
)

model_package_2 = Package(
    "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed103",
    cache_options={
        "cache_storage": Package.default_cache("hens_2"),
        "same_names": True,
    },
)

# Create the data source
data = GFS()

# %%
# Execute the Workflow
# --------------------
# Next we execute the ensemble workflow for each model but loop through each checkpoint.
# Note that the models themselves have not been loaded into memory yet, this will be
# done one at a time to minimize the memory footprint of inference on a GPU.
# Before the ensemble workflow can get executed the following set up is needed:
#
# - Initialize the SFNO model from checkpoint
# - Initialize the perturbation method with the prognostic model
# - Initialize the IO zarr store for this model
#
# If multiple GPUs are being used, one could parallelize inference using different
# checkpoints on each card.

# %%

import gc
from datetime import datetime, timedelta

import numpy as np
import torch

start_date = datetime(2024, 1, 1)
nsteps = 4
nensemble = 2

for i, package in enumerate([model_package_1, model_package_2]):
    # Load SFNO model from package
    # HENS checkpoints use different inputs than default SFNO (inclusion of d2m)
    # Can find this in the config.json, the load_model function in SFNO handles this
    model = SFNO.load_model(package)

    # Perturbation method
    # Here we will simplify the process that's in the original paper for conciseness
    noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0])
    noise_amplification[40] = 1.0  # z500
    noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)

    seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
    perturbation = HemisphericCentredBredVector(
        model, data, seed_perturbation, noise_amplitude=noise_amplification
    )

    # IO object
    io = ZarrBackend(
        file_name=f"outputs/11_hens_{i}.zarr",
        chunks={"ensemble": 1, "time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True},
    )

    io = ensemble(
        ["2024-01-01"],
        nsteps,
        nensemble,
        model,
        data,
        io,
        perturbation,
        batch_size=1,
        output_coords={"variable": np.array(["u10m", "v10m"])},
    )

    print(io.root.tree())
    # Do some manual clean up to free up VRAM
    del model
    del perturbation
    gc.collect()
    torch.cuda.empty_cache()

# %%
# Post Processing
# ---------------
# The result of the workflow is two zarr stores with the ensemble data for the
# respective checkpoints used.
# The rest of the example is focused on some basic post processing to visualize the
# results.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

lead_time = 4
plot_date = start_date + timedelta(hours=int(6 * lead_time))

# Load data from both zarr stores
ds0 = xr.open_zarr("outputs/11_hens_0.zarr")
ds1 = xr.open_zarr("outputs/11_hens_1.zarr")

# Combine the datasets
ds = xr.concat([ds0, ds1], dim="ensemble")

# Calculate wind speed magnitude
wind_speed = np.sqrt(ds.u10m**2 + ds.v10m**2)

# Get mean and std of 4th timestep across ensemble
mean_wind = wind_speed.isel(time=0, lead_time=lead_time).mean(dim="ensemble")
std_wind = wind_speed.isel(time=0, lead_time=lead_time).std(dim="ensemble")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(15, 4), subplot_kw={"projection": ccrs.PlateCarree()}
)

# Plot mean
p1 = ax1.contourf(
    mean_wind.coords["lon"],
    mean_wind.coords["lat"],
    mean_wind,
    levels=15,
    transform=ccrs.PlateCarree(),
    cmap="nipy_spectral",
)
ax1.coastlines()
ax1.set_title(f'Mean Wind Speed\n{plot_date.strftime("%Y-%m-%d %H:%M UTC")}')
fig.colorbar(p1, ax=ax1, label="m/s")

# Plot standard deviation
p2 = ax2.contourf(
    std_wind.coords["lon"],
    std_wind.coords["lat"],
    std_wind,
    levels=15,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
)
ax2.coastlines()
ax2.set_title(
    f'Wind Speed Standard Deviation\n{plot_date.strftime("%Y-%m-%d %H:%M UTC")}'
)
fig.colorbar(p2, ax=ax2, label="m/s")

plt.tight_layout()
# Save the figure
plt.savefig(f"outputs/11_hens_step_{plot_date.strftime('%Y_%m_%d')}.jpg")
