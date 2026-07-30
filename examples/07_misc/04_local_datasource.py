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
Creating a Local Data Source
============================

Create and save an offline dataset to use in an inference pipeline.

This example demonstrates how to:

- Build a small offline dataset by fetching data and writing to a Zarr store
- Load the local store as a data source for an inference pipeline with the Microsoft Aurora model
- Run the deterministic workflow and plot results
"""
# /// script
# dependencies = [
#   "earth2studio[aurora] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# For this example, the following are needed:
#
# - Prognostic Model: Use the built-in Aurora 6-hour model :py:class:`earth2studio.models.px.Aurora`.
# - Data source: Pull data from the WeatherBench2 data API :py:class:`earth2studio.data.WB2ERA5`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

from earth2studio.data import WB2ERA5, fetch_data
from earth2studio.models.px import Aurora

# Load the default model package which downloads the checkpoint from GCP
package = Aurora.load_default_package()
model = Aurora.load_model(package)

# Create the data source, cache is false
wb2 = WB2ERA5(cache=False, verbose=False)

# %%
# Creating a Local Zarr Store from a Datasource
# ---------------------------------------------
# Start with creating a local dataset from the WeatherBench2 data store. Since data
# sources return in-memory data arrays, there are a variety of ways this could be done.
# The following is a simple method using Earth2Studio IO objects to pack the requested
# data into a single Zarr store.
#
# For this example, let's download some data for a Microsoft aurora forecast.

# %%
from collections import OrderedDict

import numpy as np

from earth2studio.io import ZarrBackend
from earth2studio.utils.coords import split_coords

times = np.array(
    [np.datetime64("2022-01-01T00:00:00"), np.datetime64("2022-01-01T06:00:00")]
)
variables = model.input_coords()["variable"]
zarr_path = "./outputs/19_wb2_dataset.zarr"
# Create Zarr store to pack data into
zb = ZarrBackend(file_name=zarr_path, backend_kwargs={"overwrite": True})
full_coords = OrderedDict(
    [
        ("time", np.atleast_1d(times)),
        ("lead_time", np.array([np.timedelta64(0, "h")])),
        ("lat", np.linspace(90, -90, 721)),
        ("lon", np.linspace(0, 359.75, 1440)),
    ]
)
zb.add_array(full_coords, array_name=list(variables))

# Loop over timestamps, fetch data and write slices into the pre-created arrays
for t in np.atleast_1d(times):
    x, coords = fetch_data(
        wb2,
        time=np.array([t]),
        variable=variables,
        lead_time=np.array([np.timedelta64(0, "h")]),
        device="cpu",
    )
    xs, reduced_coords, var_names = split_coords(x, coords, dim="variable")
    zb.write(xs, reduced_coords, array_name=list(var_names))

# %%
# Note that the Zarr store we just created can be used for more than just Earth2Studio
# inference pipelines. Open it with zarr or xarray to explore/process what
# you just downloaded.

# %%
import zarr

zg = zarr.group(store=zarr.storage.LocalStore(zarr_path))
print(zg.tree())

# %%
# Execute the Workflow
# --------------------
# To use the saved dataset as a data source, we could create our own class that implements
# the interface required by :py:class:`earth2studio.data.base.DataSource`, which needs
# just a ``__call__(time, variable)`` method.
#
# However, since we used an IO backend from Earth2Studio we can use the
# :py:class:`earth2studio.data.xr.InferenceOutputSource` which is a convenience class
# that supports the output of inference pipelines.

# %%
import earth2studio.run as run
from earth2studio.data import InferenceOutputSource

offline_source = InferenceOutputSource(zarr_path)
out_zarr_path = "./outputs/19_pangu_output.zarr"
io = ZarrBackend(file_name=out_zarr_path, backend_kwargs={"overwrite": True})
io = run.deterministic(
    times[-1:],
    4,
    model,
    offline_source,
    io,
    output_coords=OrderedDict({"variable": np.array(["msl"])}),
)

# %%
# Post Processing
# ---------------
# The last step is to post-process our results.

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
projection = ccrs.Robinson()
fig, axes = plt.subplots(
    2,
    2,
    subplot_kw={"projection": projection},
    figsize=(12, 7),
    constrained_layout=True,
)
axes = axes.ravel()

lon = io["lon"][:]
lat = io["lat"][:]
lead_steps = [1, 2, 3, 4]  # 6h, 12h, 18h, 24h
for ax, step in zip(axes, lead_steps):
    im = ax.pcolormesh(
        lon,
        lat,
        io["msl"][0, step],
        transform=ccrs.PlateCarree(),
        cmap="PiYG",
    )
    ax.set_title(f"msl - Lead time: {6*step}h")
    ax.coastlines()
    ax.gridlines(draw_labels=False)

fig.colorbar(
    im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.07, label="msl"
)
plt.savefig("outputs/19_msl_1day.png", dpi=150)
