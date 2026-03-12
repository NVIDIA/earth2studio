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
HealDA Global Data Assimilation
================================

Producing a global weather analysis from satellite and conventional observations.

This example demonstrates how to use the HealDA data assimilation model to produce
a global weather analysis on a HEALPix grid from sparse in-situ (conventional) and
satellite radiance observations sourced from the NOAA UFS replay archive.
Three runs are compared: conventional observations only, satellite observations only,
and both combined to illustrate the impact of each observation type.

In this example you will learn:

- How to load and initialise the HealDA data assimilation model
- Using the built-in ``lat_lon`` option for lat-lon output
- Fetching UFS conventional and satellite observation DataFrames
- Running the model with different observation combinations
"""
# /// script
# dependencies = [
#   "earth2studio[da-healda] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA :py:class:`earth2studio.models.da.HealDA`.
# - Datasource (conv): UFS conventional observations
#   :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (sat): UFS satellite observations
#   :py:class:`earth2studio.data.UFSObsSat`.
#
# HealDA is a stateless neural-network-based data assimilation model that ingests
# conventional (radiosonde, surface station, GPS-RO, etc.) and satellite radiance
# observations and produces a single global weather analysis on a HEALPix level-6
# grid.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import timedelta

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.models.da import HealDA

# Load the default model package (downloads checkpoint from HuggingFace)
# Setting lat_lon=True regrids the native HEALPix output to a regular lat-lon grid.
# The default output_resolution is (721, 1440) for ~0.25° resolution.
package = HealDA.load_default_package()
model = HealDA.load_model(package, lat_lon=True, output_resolution=(181, 360))
model = model.to("cuda:0")

# %%
# Fetch Observations
# ------------------
# Pull conventional and satellite observation DataFrames for the analysis time.
# The UFS data sources return pandas DataFrames that match the schemas expected by
# :py:meth:`HealDA.input_coords`.  We use
# :py:func:`earth2studio.data.fetch_dataframe` which attaches ``request_time``
# metadata required by the model.

# %%
analysis_time = np.array([np.datetime64("2024-01-01T00:00")])

# Conventional observations (radiosonde, surface, GPS-RO)
conv_source = UFSObsConv(tolerance=timedelta(hours=3))
conv_schema, sat_schema = model.input_coords()
conv_df = fetch_dataframe(
    conv_source,
    time=analysis_time,
    variable=np.array(conv_schema["variable"]),
)
logger.info(f"Fetched {len(conv_df)} conventional observations")

# Satellite observations (ATMS, MHS, AMSU-A, AMSU-B)
sat_source = UFSObsSat(tolerance=timedelta(hours=3))
sat_df = fetch_dataframe(
    sat_source,
    time=analysis_time,
    variable=np.array(sat_schema["variable"]),
)
logger.info(f"Fetched {len(sat_df)} satellite observations")

# %%
# Run With Conventional Observations Only
# ----------------------------------------
# Call the model with only conventional observations to see its standalone impact.

# %%
torch.manual_seed(42)
result_conv = model(conv_obs=conv_df)
logger.info(f"Conv-only analysis shape: {result_conv.shape}")

# %%
# Run With Satellite Observations Only
# -------------------------------------
# Call the model with only satellite observations.

# %%
torch.manual_seed(42)
result_sat = model(sat_obs=sat_df)
logger.info(f"Sat-only analysis shape: {result_sat.shape}")

# %%
# Run With Both Observation Types
# --------------------------------
# Combine conventional and satellite observations for the fullest analysis.

# %%
torch.manual_seed(42)
result_both = model(conv_obs=conv_df, sat_obs=sat_df)
logger.info(f"Combined analysis shape: {result_both.shape}")

# %%
# Post Processing
# ---------------
# Because we loaded the model with ``lat_lon=True`` the output is already on a
# regular equiangular lat-lon grid, so no manual regridding is needed.
# Compare the three runs for surface temperature (``tas``) and 500 hPa geopotential
# (``Z500``).  Each row shows a different observation configuration.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
plot_vars = ["tas", "Z500"]
titles = ["Conv only", "Sat only", "Conv + Sat"]
results = [result_conv, result_sat, result_both]
projection = ccrs.Robinson()

fig, axes = plt.subplots(
    len(results),
    len(plot_vars),
    subplot_kw={"projection": projection},
    figsize=(7 * len(plot_vars), 4 * len(results)),
)

lat = results[0].coords["lat"].values
lon = results[0].coords["lon"].values

for row, (title, da) in enumerate(zip(titles, results)):
    for col, var in enumerate(plot_vars):
        ax = axes[row, col]
        field = da.sel(variable=var).values[0]  # [nlat, nlon]
        if hasattr(field, "get"):
            field = field.get()  # cupy -> numpy
        im = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=ccrs.PlateCarree(),
            cmap="Spectral_r",
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        if row == 0:
            ax.set_title(var, fontsize=14)
        if col == 0:
            bbox = ax.get_position()
            fig.text(
                bbox.x0 - 0.01,
                (bbox.y0 + bbox.y1) / 2,
                title,
                fontsize=12,
                va="center",
                ha="right",
                rotation=90,
            )

fig.suptitle(
    f"HealDA Analysis — {str(analysis_time[0])[:16]} UTC",
    fontsize=16,
    y=1.01,
)
plt.savefig("outputs/22_healda_analysis.jpg", dpi=150, bbox_inches="tight")

# %%
# Difference Plot
# ---------------
# Show the difference between the combined analysis and each single-source run.
# This highlights where adding the other observation type changes the analysis
# most.

# %%
plt.close("all")

fig, axes = plt.subplots(
    2,
    len(plot_vars),
    subplot_kw={"projection": projection},
    figsize=(7 * len(plot_vars), 8),
)

diff_titles = ["(Conv+Sat) − Conv", "(Conv+Sat) − Sat"]
diff_pairs = [(result_both, result_conv), (result_both, result_sat)]

for row, (diff_title, (da_a, da_b)) in enumerate(zip(diff_titles, diff_pairs)):
    for col, var in enumerate(plot_vars):
        ax = axes[row, col]
        field_a = da_a.sel(variable=var).values[0]
        field_b = da_b.sel(variable=var).values[0]
        if hasattr(field_a, "get"):
            field_a = field_a.get()
        if hasattr(field_b, "get"):
            field_b = field_b.get()
        diff = field_a - field_b
        vmax = np.nanpercentile(np.abs(diff), 98)
        im = ax.pcolormesh(
            lon,
            lat,
            diff,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        if row == 0:
            ax.set_title(var, fontsize=14)
        if col == 0:
            bbox = ax.get_position()
            fig.text(
                bbox.x0 - 0.01,
                (bbox.y0 + bbox.y1) / 2,
                diff_title,
                fontsize=12,
                va="center",
                ha="right",
                rotation=90,
            )

fig.suptitle(
    f"HealDA Analysis Differences — {str(analysis_time[0])[:16]} UTC",
    fontsize=16,
    y=1.01,
)
plt.savefig("outputs/22_healda_differences.jpg", dpi=150, bbox_inches="tight")
