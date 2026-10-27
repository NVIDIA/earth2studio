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

Producing a 0.25 degree global analysis from the NNJA observing system.

This example runs the HealDA v2 data assimilation model on NNJA observations. The
model is observation-only: it reads an eight-frame, six-hourly window ending at the
analysis time and every observation within three hours of each frame, and returns
the analysis of the last frame on the 721 x 1440 equiangular grid.

In this example you will learn:

- How to load the HealDA v2 model with the `healda` package as its backend
- Fetching NNJA conventional, satellite-wind and radiance DataFrames over the model's
  48 hour window
- Running the model and comparing the analysis against ERA5
"""

# /// script
# dependencies = [
#   "earth2studio[da-healda-v2] @ git+https://github.com/NVIDIA/earth2studio.git@0.20.0",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA v2 [`earth2studio.models.da.HealDAv2`][earth2studio.models.da.HealDAv2].
# - Datasource (PrepBUFR and GPS-RO): NNJA conventional observations
#   [`earth2studio.data.NNJAObsConv`][earth2studio.data.NNJAObsConv].
# - Datasource (winds): NNJA satellite winds
#   [`earth2studio.data.NNJAObsSatwnd`][earth2studio.data.NNJAObsSatwnd].
# - Datasource (radiances): NNJA satellite radiances
#   [`earth2studio.data.NNJAObsSat`][earth2studio.data.NNJAObsSat].
#
# Decoding 48 hours of NNJA BUFR takes a long time on first use; the raw files are
# cached, so later runs over the same window skip the download.

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

from earth2studio.data import (
    NCAR_ERA5,
    NNJAObsConv,
    NNJAObsSat,
    NNJAObsSatwnd,
    fetch_dataframes,
)
from earth2studio.models.da import HealDAv2

# The network runs on a CUDA device only; building the 0.25 degree recipe fetches the
# ERA5 static fields into the healda cache the first time.
package = HealDAv2.load_default_package()
model = HealDAv2.load_model(package).to("cuda:0")

# %%
# Fetch Observations
# ------------------
# One analysis reads observations spanning ``[t - 45h, t + 3h]``. Any source returning
# DataFrames that match `HealDAv2.input_coords` works; here the NNJA sources are
# configured as in training, in the wide layout the model reads (one row per report
# level, wind or footprint): each PrepBUFR observation's original event, satellite
# winds from `NNJAObsSatwnd` only, and only the IR sounder channels the model reads.
# A wide request returns one stream (PrepBUFR, GPS-RO, satellite winds or one satellite
# sensor), and `input_coords` gives one schema per model argument, in the order
# ``(prepbufr, gpsro, satwnd, *sensors)``. `fetch_dataframes` fetches every stream
# concurrently, in that order, and attaches the ``request_time`` metadata the model
# requires. A sensor outside its archive period comes back empty with a warning.

# %%
analysis_time = np.array([np.datetime64("2026-01-15T00:00")])
tolerance = (timedelta(hours=-45), timedelta(hours=3))

conv_source = NNJAObsConv(
    time_tolerance=tolerance,
    original_event=True,
    exclude_message_types=("SATWND",),
    layout="wide",
)
satwnd_source = NNJAObsSatwnd(time_tolerance=tolerance, layout="wide")
sat_source = NNJAObsSat(
    time_tolerance=tolerance, sensor_indices=model.sensor_indices, layout="wide"
)

schemas = model.input_coords()  # (prepbufr, gpsro, satwnd, *sensors)
sources = [conv_source, conv_source, satwnd_source] + [sat_source] * (len(schemas) - 3)
obs = fetch_dataframes(list(zip(sources, schemas)), analysis_time)
logger.info(f"Fetched rows per stream: {[len(frame) for frame in obs]}")

# %%
# Run the Model
# -------------
# The direct call returns the analysis as an `xr.DataArray` on the model's device.

# %%
analysis = model(*obs)
logger.info(f"Analysis shape: {analysis.shape}")

# %%
# HealDA v2 vs ERA5
# -----------------
# The output already uses Earth2Studio variable names on the 0.25 degree grid, so ERA5
# from the NCAR archive can be queried and compared without regridding.

# %%
from collections import OrderedDict

import cartopy.crs as ccrs
import cupy
import matplotlib.pyplot as plt
import torch

from earth2studio.statistics import lat_weight, rmse

plot_vars = ["t2m", "z500"]
era5 = NCAR_ERA5()(analysis_time, plot_vars)

lat = analysis.coords["lat"].values
lon = analysis.coords["lon"].values
grid = OrderedDict(lat=lat, lon=lon)
weights = lat_weight(torch.as_tensor(lat))[:, None].repeat(1, lon.size)
score = rmse(["lat", "lon"], weights=weights)
fig, axes = plt.subplots(
    3, len(plot_vars), subplot_kw={"projection": ccrs.Robinson()}, figsize=(14, 9)
)
for col, var in enumerate(plot_vars):
    field = cupy.asnumpy(analysis.sel(variable=var).data[0])
    truth = era5.sel(variable=var).values[0]
    diff = field - truth
    # Symmetric color limit covering the 0.5-99.5 percentile range of the difference
    limit = float(np.abs(np.percentile(diff, [0.5, 99.5])).max())
    error = float(score(torch.as_tensor(field), grid, torch.as_tensor(truth), grid)[0])
    logger.info(f"{var} RMSE vs ERA5: {error:.4f}")
    scale = {"vmin": truth.min(), "vmax": truth.max()}
    panels = (
        (f"HealDA v2 {var}", field, scale),
        (f"ERA5 {var}", truth, scale),
        (
            f"HealDA v2 - ERA5 {var} (RMSE {error:.3g})",
            diff,
            {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit},
        ),
    )
    for row, (title, values, style) in enumerate(panels):
        ax = axes[row, col]
        im = ax.imshow(
            values, extent=(0, 360, -90, 90), transform=ccrs.PlateCarree(), **style
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle(f"HealDA v2 analysis {str(analysis_time[0])[:16]} UTC", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/04_healda_v2_analysis.jpg", dpi=150)
