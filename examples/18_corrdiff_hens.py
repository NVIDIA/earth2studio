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
CorrDiff on HENS Ensemble
=========================

This example shows how to run CorrDiff on a HENS ensemble.

In this example you will learn:

- How to run CorrDiff on a HENS ensemble
- How to save the output to a Zarr backend
- How to post process the output

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
# Imports and utility functions
import os
from collections import OrderedDict
from datetime import datetime
from math import ceil

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import GFS, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiff
from earth2studio.models.px import SFNO, PrognosticModel
from earth2studio.perturbation import (
    CorrelatedSphericalGaussian,
    HemisphericCentredBredVector,
    Perturbation,
)
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array


# %%
# Helper function to write data to the Zarr backend with the correct coordinates.
# Handles both sampled (CorrDiff) and non-sampled (HENS) variables.
def write_to_zarr(
    io,
    coords_for_io,
    corrdiff_coords,
    batch_id,
    ens_idx,
    step,
    var_name,
    var_data,
    is_sampled,
):
    """
    Helper function to write data to the Zarr backend with the correct coordinates.
    Handles both sampled (CorrDiff) and non-sampled (HENS) variables.
    """
    coords_to_write = {
        "ensemble": np.array([coords_for_io["ensemble"][batch_id + ens_idx]]),
        "time": np.array([coords_for_io["time"][0]]),
        "lead_time": np.array([coords_for_io["lead_time"][step]]),
    }
    if is_sampled:
        coords_to_write["sample"] = coords_for_io["sample"]
    for k in coords_for_io:
        if k not in ("ensemble", "time", "lead_time", "sample"):
            v = corrdiff_coords.get(k)
            if v is not None:
                coords_to_write[k] = np.array(v) if not isinstance(v, np.ndarray) else v
    io.write(var_data, coords_to_write, array_name=var_name)


# %%
# Main function to run CorrDiff on a HENS ensemble
def corrdiff_on_hens_ensemble(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    prognostic: PrognosticModel,
    corrdiff: CorrDiff,
    nsamples: int,
    data: GFS,
    io: ZarrBackend,
    perturbation: Perturbation,
    batch_size: int | None = None,
    device: torch.device | None = None,
    save_hens_vars: list[str] = None,
) -> ZarrBackend:
    """
    Run CorrDiff generative downscaling on top of a HENS/SFNO ensemble.
    The output Zarr will have both 'ensemble' and 'sample' dimensions.
    Optionally, also save selected HENS variables interpolated to the CorrDiff output grid.
    """
    logger.info("Running HENS + CorrDiff ensemble inference!")
    # Set device (GPU if available, else CPU)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Prepare CorrDiff output coordinates for IO backend
    corrdiff_ic = corrdiff.input_coords()
    corrdiff_oc = corrdiff.output_coords(corrdiff_ic)
    coords_for_io = OrderedDict()
    coords_for_io["ensemble"] = np.arange(nensemble)
    coords_for_io["time"] = time
    coords_for_io["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    coords_for_io["sample"] = np.arange(nsamples)
    for k, v in corrdiff_oc.items():
        if k != "batch":
            coords_for_io[k] = v
    variables_to_save = coords_for_io.pop("variable")
    io.add_array(coords_for_io, variables_to_save)

    # Optionally, add HENS variables to Zarr backend (without sample axis)
    if save_hens_vars:
        for hens_var in save_hens_vars:
            if hens_var not in io.root:
                hens_coords = coords_for_io.copy()
                hens_coords.pop("sample", None)
                io.add_array(hens_coords, [hens_var])

    # Determine batch size and number of batches
    batch_size = min(nensemble, batch_size or nensemble)
    number_of_batches = ceil(nensemble / batch_size)
    logger.info(
        f"Starting {nensemble} Member Ensemble Inference with {number_of_batches} batches."
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
                # For each ensemble member in the batch
                for ens_idx in range(mini_batch_size):
                    ens_x = x[ens_idx : ens_idx + 1]
                    ens_coords = {
                        k: (
                            v[ens_idx : ens_idx + 1]
                            if isinstance(v, np.ndarray)
                            and v.shape[0] == mini_batch_size
                            else v
                        )
                        for k, v in coords.items()
                    }
                    ens_x, ens_coords = map_coords(
                        ens_x, ens_coords, corrdiff.input_coords()
                    )
                    # Optionally, collect and write HENS outputs for selected variables
                    if save_hens_vars:
                        for hens_var in save_hens_vars:
                            var_idx = list(ens_coords["variable"]).index(hens_var)
                            hens_interp = corrdiff._interpolate(ens_x)
                            hens_interp = hens_interp[..., var_idx, :, :]
                            corrdiff_coords = corrdiff.output_coords(
                                corrdiff.input_coords()
                            )

                            write_to_zarr(
                                io,
                                coords_for_io,
                                corrdiff_coords,
                                batch_id,
                                ens_idx,
                                step,
                                hens_var,
                                hens_interp,
                                is_sampled=False,
                            )
                    # CorrDiff workflow: generate and write CorrDiff outputs
                    corrdiff_out, corrdiff_coords = corrdiff(ens_x, ens_coords)
                    for var_idx, var_name in enumerate(variables_to_save):
                        var_data = corrdiff_out[..., :, var_idx, :, :]
                        write_to_zarr(
                            io,
                            coords_for_io,
                            corrdiff_coords,
                            batch_id,
                            ens_idx,
                            step,
                            var_name,
                            var_data,
                            is_sampled=True,
                        )
                pbar.update(1)
                if step == nsteps:
                    break

    logger.success("Inference complete")
    return io


# %%
# Set up environment and data source
os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

data = GFS()

# %%
# Load HENS/SFNO model
hens_package = Package(
    "hf://datasets/maheshankur10/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed103",
    cache_options={
        "cache_storage": Package.default_cache("hens_1"),
        "same_names": True,
    },
)
model = SFNO.load_model(hens_package)

# %%
# Set up perturbation for ensemble generation
noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0])
index_z500 = list(model.input_coords()["variable"]).index("z500")
noise_amplification[index_z500] = 39.27
noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)
seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
perturbation = HemisphericCentredBredVector(
    model, data, seed_perturbation, noise_amplitude=noise_amplification
)

# %%
# Load CorrDiff model
corrdiff_package = Package(
    "mini_package",
    cache_options={
        "cache_storage": Package.default_cache("corrdiff"),
        "same_names": True,
    },
)
corrdiff = CorrDiff.load_model(corrdiff_package)

# %%
# Set up IO backend for output
io = ZarrBackend(
    file_name="outputs/hens_corrdiff.zarr",
    chunks={"ensemble": 1, "sample": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

# %%
# Run the workflow, saving selected HENS variables interpolated to CorrDiff grid
corrdiff_on_hens_ensemble(
    time=[datetime(2022, 9, 1, 12)],
    nsteps=4,
    nensemble=2,
    prognostic=model,
    corrdiff=corrdiff,
    nsamples=3,
    data=data,
    io=io,
    perturbation=perturbation,
    batch_size=1,
    save_hens_vars=["u10m", "v10m", "t2m", "z500"],
)

# %%
# Post Processing: Plot mean and std of a variable at a given lead_time
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_zarr("outputs/hens_corrdiff.zarr")

lead_time = 3
var = "u10m"  # or any variable present in ds
arr = ds[var]
n_ens = arr.sizes["ensemble"]
n_samples = arr.sizes["sample"] if "sample" in arr.dims else 1
arr2d = arr.isel(time=0, lead_time=lead_time)
mean_field = arr2d.mean(
    dim=["ensemble", "sample"] if "sample" in arr.dims else ["ensemble"]
)
std_field = arr2d.std(
    dim=["ensemble", "sample"] if "sample" in arr.dims else ["ensemble"]
)

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)
p1 = ax1.contourf(
    ds["lon"],
    ds["lat"],
    mean_field,
    levels=15,
    transform=ccrs.PlateCarree(),
    cmap="nipy_spectral",
)
ax1.coastlines()
ax1.set_title(f"Mean {var} (lead_time={lead_time})")
fig.colorbar(p1, ax=ax1, label=var)

p2 = ax2.contourf(
    ds["lon"],
    ds["lat"],
    std_field,
    levels=15,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
)
ax2.coastlines()
ax2.set_title(f"Std {var} (lead_time={lead_time})")
fig.colorbar(p2, ax=ax2, label=var)

plt.tight_layout()
plt.savefig(f"outputs/hens_corrdiff_{var}_lead{lead_time}.jpg")
plt.show()
