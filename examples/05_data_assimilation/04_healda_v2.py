# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
HealDA v2 Global Data Assimilation
==================================

Producing a 0.25 degree global analysis from GPS-RO and satellite-wind observations.

This example runs the HealDA v2 data assimilation model on NNJA observations. The
model is observation-only: it reads an eight-frame, six-hourly window ending at the
analysis time and every observation within three hours of each frame, and returns
the analysis of the last frame on the 721 x 1440 equiangular grid.

In this example you will learn:

- How to load the HealDA v2 model with the `healda` package as its backend
- Fetching NNJA GPS-RO and satellite-wind DataFrames over the model's 48 hour window
- Running the model and comparing the analysis against ERA5
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda-v2] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA v2 [`earth2studio.models.da.HealDAv2`][earth2studio.models.da.HealDAv2].
# - Datasource (GPS-RO): NNJA conventional observations
#   [`earth2studio.data.NNJAObsConv`][earth2studio.data.NNJAObsConv].
# - Datasource (winds): NNJA satellite winds
#   [`earth2studio.data.NNJAObsSatwnd`][earth2studio.data.NNJAObsSatwnd].
#
# The checkpoint was trained on the full NNJA observing system (radiances, PrepBUFR
# conventional reports, GPS-RO and satellite winds). GPS-RO and satellite winds are the
# two streams with Earth2Studio adapters today, so the analysis here is a degraded
# subset of what the recipe can produce.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

from datetime import timedelta

import numpy as np
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import NCAR_ERA5, NNJAObsConv, NNJAObsSatwnd, fetch_dataframe
from earth2studio.models.da import HealDAv2

# The network runs on a CUDA device only; building the 0.25 degree recipe fetches the
# ERA5 static fields into the healda cache the first time.
package = HealDAv2.load_default_package()
model = HealDAv2.load_model(package, device="cuda:0")

# %%
# Fetch Observations
# ------------------
# One analysis reads observations spanning ``[t - 45h, t + 3h)``. The NNJA sources
# return DataFrames matching `HealDAv2.input_coords`; `fetch_dataframe` attaches the
# ``request_time`` metadata the model requires.

# %%
analysis_time = np.array([np.datetime64("2024-01-05T00:00")])
tolerance = (timedelta(hours=-45), timedelta(hours=3) - timedelta(seconds=1))

gpsro_schema, satwnd_schema = model.input_coords()
gpsro_df = fetch_dataframe(
    NNJAObsConv(time_tolerance=tolerance),
    time=analysis_time,
    variable=np.array(gpsro_schema["variable"]),
    fields=np.array(list(gpsro_schema.keys())),
)
logger.info(f"Fetched {len(gpsro_df)} GPS-RO rows")

satwnd_df = fetch_dataframe(
    NNJAObsSatwnd(time_tolerance=tolerance),
    time=analysis_time,
    variable=np.array(satwnd_schema["variable"]),
    fields=np.array(list(satwnd_schema.keys())),
)
logger.info(f"Fetched {len(satwnd_df)} satellite-wind rows")

# %%
# Run the Model
# -------------
# The direct call returns the analysis as an `xr.DataArray` on the model's device.

# %%
analysis = model(gpsro_obs=gpsro_df, satwnd_obs=satwnd_df)
logger.info(f"Analysis shape: {analysis.shape}")

# %%
# HealDA v2 vs ERA5
# -----------------
# The output already uses Earth2Studio variable names on the 0.25 degree grid, so ERA5
# from the NCAR archive can be queried and compared without regridding.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plot_vars = ["t2m", "z500"]
era5 = NCAR_ERA5()(analysis_time, plot_vars)


def to_numpy(arr):
    """CuPy / Numpy helper function"""
    return arr.get() if hasattr(arr, "get") else arr


lat = analysis.coords["lat"].values
lon = analysis.coords["lon"].values
fig, axes = plt.subplots(
    2, len(plot_vars), subplot_kw={"projection": ccrs.Robinson()}, figsize=(14, 6)
)
for col, var in enumerate(plot_vars):
    field = to_numpy(analysis.sel(variable=var).data[0])
    truth = era5.sel(variable=var).values[0]
    logger.info(f"{var} MAE vs ERA5: {float(np.abs(field - truth).mean()):.4f}")
    for row, (title, values) in enumerate((("HealDA v2", field), ("ERA5", truth))):
        ax = axes[row, col]
        im = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{title} {var}")
        fig.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle(f"HealDA v2 analysis {str(analysis_time[0])[:16]} UTC", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/04_healda_v2_analysis.jpg", dpi=150)
