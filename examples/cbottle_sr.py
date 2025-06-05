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

# %%

# %%
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("cache", exist_ok=True)  # Create cache directory
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from collections import OrderedDict
from datetime import datetime
import hashlib

import numpy as np
import torch
from loguru import logger

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import CBottleSR
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array


def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    cbottle_sr: CBottleSR,
    data: DataSource,
    io: IOBackend,
    number_of_samples: int = 1,
) -> IOBackend:
    """CorrDiff infernce workflow

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    cbottle_sr : CBottleSR
        CBottleSR model
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

    cbottle_sr = cbottle_sr.to(device)

    # Fetch data from data source and load onto device
    time = to_time_array(time)

    # Implement caching for data
    cache_key = hashlib.md5(str(time).encode()).hexdigest()
    cache_file = f"cache/data_{cache_key}.npz"
    
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        cached_data = np.load(cache_file, allow_pickle=True)
        x = torch.from_numpy(cached_data['x']).to(device)
        # Convert numpy arrays back to torch tensors for each coordinate
        coords = {}
        for key in cached_data.files:
            if key.startswith('coords_'):
                coord_key = key[7:]  # Remove 'coords_' prefix
                coords[coord_key] = np.array(cached_data[key])
    else:
        logger.info("Generating new data")
        x, coords = prep_data_array(
            data(time, cbottle_sr.input_coords()["variable"]), device=device
        )
        x, coords = map_coords(x, coords, cbottle_sr.input_coords())
        
        # Cache the data - save each coordinate tensor separately
        save_dict = {'x': x.cpu().numpy()}
        for key, tensor in coords.items():
            save_dict[f'coords_{key}'] = tensor
        
        np.savez(cache_file, **save_dict)
        logger.info(f"Saved data to cache: {cache_file}")

    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    output_coords =   cbottle_sr.output_coords(cbottle_sr.input_coords())
    total_coords = OrderedDict(
        {
            "time": coords["time"],
            "lat": output_coords["lat"],
            "lon": output_coords["lon"],
        }
    )
    io.add_array(total_coords, output_coords["variable"])

    logger.info("Inference starting!")
    x, coords = cbottle_sr(x, coords)
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
from earth2studio.data import CBottle3D
from earth2studio.io import ZarrBackend

# Create CBottleSR model
package = CBottleSR.load_default_package()
cbottle_sr = CBottleSR.load_model(package, num_steps=18)

# Create the data source
package = CBottle3D.load_default_package()
data = CBottle3D.load_model(package)

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
io = run(["2020-10-04T18:00:00"], cbottle_sr, data, io, number_of_samples=1)

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

# Create a Robinson projection
projection = ccrs.Robinson()

# Create a figure and axes with the specified projection
fig = plt.figure(figsize=(15, 5))

ax0 = fig.add_subplot(1, 3, 1, projection=projection)
c = ax0.pcolormesh(
    io["lon"][:],
    io["lat"][:],
    io["tpf"][0],
    transform=ccrs.PlateCarree(),
    cmap="viridis",
)
plt.colorbar(c, ax=ax0, shrink=0.6, label="mm/h")
ax0.coastlines()
ax0.gridlines()
ax0.set_title("Precipitation")

ax1 = fig.add_subplot(1, 3, 2, projection=projection)
c = ax1.pcolormesh(
    io["lon"][:],
    io["lat"][:],
    io["t2m"][0],
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
)
plt.colorbar(c, ax=ax1, shrink=0.6, label="K")
ax1.coastlines()
ax1.gridlines()
ax1.set_title("2-meter Temperature")

ax2 = fig.add_subplot(1, 3, 3, projection=projection)
c = ax2.pcolormesh(
    io["lon"][:],
    io["lat"][:],
    np.sqrt(io["u10m"][0] ** 2 + io["v10m"][0] ** 2),
    transform=ccrs.PlateCarree(),
    cmap="Greens",
)
plt.colorbar(c, ax=ax2, shrink=0.6, label="m s^-1")
ax2.coastlines()
ax2.gridlines()
ax2.set_title("10-meter Wind Speed")

plt.tight_layout()
plt.savefig("outputs/04_cbottle_sr_prediction.jpg", dpi=300, bbox_inches='tight')
