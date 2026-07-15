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
Running StormCast-CONUS Inference
==================================

Basic StormCast-CONUS inference workflow for the full Continental United States.

This example demonstrates how to run a single-member forecast using
StormCast-CONUS, a generative convection-allowing model at 3 km resolution over
the full CONUS HRRR domain. For details about the model see

 - https://arxiv.org/abs/2408.10958

In this example you will learn:

- How to instantiate StormCast-CONUS from a model package
- Creating an HRRR initial-condition data source and a GFS forecast conditioning source
- Running a deterministic forecast with :py:meth:`earth2studio.run.deterministic`
- Post-processing and plotting composite reflectivity and 2-m temperature
"""
# /// script
# dependencies = [
#   "earth2studio[data,stormcast] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
#   "numpy",
# ]
# ///

# %%
# Set Up
# ------
# StormCast-CONUS requires a low-resolution global conditioning source. We use
# :py:class:`earth2studio.data.GFS_FX` (the default), which provides GFS forecast
# fields interpolated to the HRRR grid. An analysis source such as
# :py:class:`earth2studio.data.ARCO` can also be used.
#
# .. note::
#    StormCast-CONUS does not yet have a publicly released default package.
#    Set the ``STORMCAST_CONUS_MODEL_PATH`` environment variable to the local
#    directory or remote URI of the model package before running this example.

# %%
import os
from datetime import datetime, timedelta

import numpy as np
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import StormCastCONUS

# Load the model package from a local path or remote URI.
# Set STORMCAST_CONUS_MODEL_PATH to override the default location.
model_path = os.environ.get("STORMCAST_CONUS_MODEL_PATH")
if model_path is None:
    package = StormCastCONUS.load_default_package()
else:
    package = Package(
        model_path,
        cache_options={
            "cache_storage": Package.default_cache("stormcast-conus"),
            "same_names": True,
        },
    )
# Uses GFS_FX as the conditioning data source by default
model = StormCastCONUS.load_model(package)

# Create the HRRR initial-condition data source
data = HRRR()

# Create the IO handler, store in memory
io = ZarrBackend()

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# We save only ``t2m`` and ``refc`` to keep memory usage manageable; the full model
# state has 99 channels on the 1024 × 1792 CONUS grid.
#
# For the forecast we will predict 4 hours ahead.

# %%
import earth2studio.run as run

nsteps = 4
today = datetime.today() - timedelta(days=1)
date = today.isoformat().split("T")[0]

io = run.deterministic(
    [date],
    nsteps,
    model,
    data,
    io,
    output_coords={"variable": np.array(["t2m", "refc"])},
)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# The last step is to post-process and visualise the results. We use Cartopy with the
# HRRR Lambert Conformal projection to plot the CONUS domain. Both composite
# reflectivity and 2-m temperature are shown at the final lead time.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = f"{date}"
step = nsteps  # lead time in hours (1 h time step)

# HRRR Lambert Conformal projection
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


def plot_field(ax, data, title, cmap, vmin=None, vmax=None):
    """Plot a 2-D field on the HRRR Lambert Conformal grid."""
    im = ax.pcolormesh(
        model.lon,
        model.lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    ax.set_title(title)
    ax.coastlines()
    ax.gridlines()
    ax.add_feature(
        cartopy.feature.STATES.with_scale("50m"),
        linewidth=0.5,
        edgecolor="black",
        zorder=2,
    )


plt.close("all")
fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, subplot_kw={"projection": projection}, figsize=(20, 6)
)

# Composite reflectivity — mask sub-zero values which are not physically meaningful
refc = io["refc"][0, step]
plot_field(
    ax1,
    np.where(refc > 0, refc, np.nan),
    f"{forecast} - Reflectivity (dBZ) - Lead time: {step}h",
    cmap="gist_ncar",
    vmin=0,
    vmax=60,
)

# 2-m temperature
plot_field(
    ax2,
    io["t2m"][0, step],
    f"{forecast} - 2m Temperature (K) - Lead time: {step}h",
    cmap="Spectral_r",
)

plt.tight_layout()
plt.savefig(f"outputs/04_{date}_refc_t2m.jpg", dpi=150, bbox_inches="tight")
