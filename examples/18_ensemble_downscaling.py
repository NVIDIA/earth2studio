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
Ensemble Forecasting with Downscaling
=====================================

Custom ensembling workflow with generative downscaling using CorrDiff.


This example demonstrates an ensemble forecasting pipeline that runs a
prognostic model ensemble and applies CorrDiff generative downscaling to each
step and member. While this example uses SFNO for the prognostic model, the
pipeline is model-agnostic and can be reused with other prognostic models that
follow the Earth2Studio interfaces.

In this example you will learn:

- How to create an ensemble forecast pipeline with CorrDiff downscaling
- Saving output ensemble data to a Zarr store
- Post-process results
"""
# /// script
# dependencies = [
#     "earth2studio[sfno] @ git+https://github.com/NVIDIA/earth2studio.git",
#     "earth2studio[corrdiff] @ git+https://github.com/NVIDIA/earth2studio.git",
#     "cartopy",
#     "matplotlib",
# ]
# ///

# %%
# Creating an Ensemble Downscaling Workflow
# -----------------------------------------
#
# To create our own ensemble forecasting with downscaling workflow, we will use the
# built-in ensemble workflow :py:meth:`earth2studio.run.ensemble` as the reference to
# start with. For this to work we need to update how the output coordinates are
# calculated for the IO object, as well as add the CorrDiff model's forward call
# into the forecast loop.
#
# As in previous examples, we use dependency injection to define the signature of the
# pipeline method.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime
from math import ceil

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import CorrDiff
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array


def corrdiff_on_hens_ensemble(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    nsamples: int,
    prognostic: PrognosticModel,
    corrdiff: CorrDiff,
    data: DataSource,
    io: IOBackend,
    perturbation: Perturbation,
    batch_size: int | None = None,
) -> IOBackend:
    """Ensemble CorrDiff pipeline

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        Forecast start times
    nsteps : int
        Number of forecast steps for prognostic model to take
    nensemble : int
        Number of forecast ensemble members
    nsamples : int
        Number of samples from CorrDiff model to generate
    prognostic : PrognosticModel
        Prognostic model
    corrdiff : CorrDiff
        CorrDiff model
    data : DataSource
        Data source
    io : IOBackend
        IO Backend
    perturbation : Perturbation
        Perturbation method
    batch_size : int | None, optional
        Ensemble batch size during forecasting. If None, uses nensemble; default None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prognostic = prognostic.to(device)
    corrdiff = corrdiff.to(device)
    corrdiff.number_of_samples = nsamples

    # Fetch initial data for the ensemble
    prognostic_ic = prognostic.input_coords()
    time = to_time_array(time)
    x0, coords0 = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
        interp_to=prognostic_ic if hasattr(prognostic, "interp_method") else None,
        interp_method=getattr(prognostic, "interp_method", "nearest"),
    )

    # Prepare CorrDiff output coordinates for IO backend
    total_coords = corrdiff.output_coords(corrdiff.input_coords())
    if "batch" in total_coords:
        del total_coords["batch"]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords["ensemble"] = np.arange(nensemble)
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    total_coords.move_to_end("ensemble", last=False)
    variables_to_save = total_coords.pop("variable")
    io.add_array(total_coords, variables_to_save)

    # Determine batch size and number of batches
    batch_size = min(nensemble, batch_size or nensemble)
    number_of_batches = ceil(nensemble / batch_size)
    logger.info(
        f"Starting {nensemble} member ensemble inference with {number_of_batches} batches."
    )

    # Main ensemble loop
    for batch_id in tqdm(
        range(0, nensemble, batch_size),
        total=number_of_batches,
        desc="Ensemble Batches",
    ):
        mini_batch_size = min(batch_size, nensemble - batch_id)
        x = x0.to(device)
        # Set up coordinates for this batch
        coords = {
            "ensemble": np.arange(batch_id, batch_id + mini_batch_size),
            **coords0.copy(),
        }
        # Repeat initial condition for each ensemble member in the batch
        x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * x.ndim))
        x, coords = map_coords(x, coords, prognostic_ic)
        x, coords = perturbation(x, coords)
        model = prognostic.create_iterator(x, coords)

        with tqdm(
            total=nsteps + 1,
            desc=f"Batch {batch_id} inference",
            position=1,
            leave=False,
        ) as pbar:
            for step, (x, coords) in enumerate(model):
                # Map prognostic outputs to CorrDiff inputs if needed
                x, coords = map_coords(x, coords, corrdiff.input_coords())
                # CorrDiff workflow: generate and write CorrDiff outputs
                x, coords = corrdiff(x, coords)
                io.write(*split_coords(x, coords))
                pbar.update(1)
                if step == nsteps:
                    break

    logger.success("Inference complete")
    return io


# %%
# Set Up
# ------
# With the inference pipeline function defined, next let's create the required
# components as usual. We need the following:
#
# - Prognostic Model: Use the built in SFNO model :py:class:`earth2studio.models.px.SFNO`.
# - CorrDiff Model: Use the built in CorrDiff Taiwan model :py:class:`earth2studio.models.dx.CorrDiffTaiwan`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Let's save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# For the prognostic checkpoint, we will use a HENS checkpoint conveniently stored
# on `HuggingFace <https://huggingface.co/datasets/maheshankur10/hens/tree/main/earth2mip_prod_registry>`_.
# Refer to the previous examples for more information about loading these models.

# %%

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.models.px import SFNO
from earth2studio.perturbation import (
    CorrelatedSphericalGaussian,
    HemisphericCentredBredVector,
)

# Create data source
data = GFS()
# Load prognostic model
hens_package = Package(
    "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed103",
    cache_options={
        "cache_storage": Package.default_cache("hens_1"),
        "same_names": True,
    },
)
model = SFNO.load_model(hens_package)
# Set up perturbation method
noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0])
index_z500 = list(model.input_coords()["variable"]).index("z500")
noise_amplification[index_z500] = 39.27
noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)
seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
perturbation = HemisphericCentredBredVector(
    model, data, seed_perturbation, noise_amplitude=noise_amplification
)
# Load the CorrDiffTaiwan model
corrdiff = CorrDiffTaiwan.load_model(CorrDiffTaiwan.load_default_package())
# Set up IO backend
io = ZarrBackend(
    file_name="outputs/18_ensemble_corrdiff.zarr",
    chunks={"ensemble": 1, "sample": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

# %%
# Run
# ---
# Execute the pipeline. For this example, we use the period of Typhoon Doksuri which
# has a track over the Taiwan region.
# https://en.wikipedia.org/wiki/Typhoon_Doksuri

# %%
start_date = datetime(2023, 7, 26, 12)
corrdiff_on_hens_ensemble(
    time=[start_date],
    nsteps=4,
    nensemble=2,
    nsamples=3,
    prognostic=model,
    corrdiff=corrdiff,
    data=data,
    io=io,
    perturbation=perturbation,
    batch_size=1,
)

# %%
# Post-processing
# ---------------
# Plot the mean and standard deviation of 10m wind speed magnitude for a sequence of
# lead times.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_zarr("outputs/18_ensemble_corrdiff.zarr")
lead_time = 4
arr = np.sqrt(ds["u10m"] ** 2 + ds["v10m"] ** 2)
mean_field = arr.mean(dim=["ensemble", "sample"])
std_field = arr.std(dim=["ensemble", "sample"])
fig, ax = plt.subplots(
    2, lead_time, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)

for i in range(lead_time):
    p1 = ax[0, i].contourf(
        ds["lon"],
        ds["lat"],
        mean_field.isel(time=0, lead_time=i),
        levels=20,
        vmin=0,
        vmax=40,
        transform=ccrs.PlateCarree(),
        cmap="nipy_spectral",
    )
    ax[0, i].coastlines()
    ax[0, i].set_title(f"lead_time={6*i}hr")

    p2 = ax[1, i].contourf(
        ds["lon"],
        ds["lat"],
        std_field.isel(time=0, lead_time=i),
        levels=20,
        vmin=0,
        vmax=4,
        transform=ccrs.PlateCarree(),
        cmap="magma",
    )
    ax[1, i].coastlines()

fig.colorbar(p1, ax=ax[0, -1], label="wind speed mean")
fig.colorbar(p2, ax=ax[1, -1], label="wind speed std")
fig.suptitle(
    f"Start date: {np.datetime_as_string(ds['time'].values[0], unit='h')}", fontsize=12
)

# Leave room for suptitle
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("outputs/18_ensemble_corrdiff_w10m.jpg")
plt.show()
