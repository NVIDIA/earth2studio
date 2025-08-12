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
# - Datasource: Pull data from a CMIP6 model via :py:class:`earth2studio.data.CMIP6`.
# - Diagnostic Model: Use the built in CBottle Infill Model :py:class:`earth2studio.models.dx.CBottleInfill`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import ARCO, CMIP6
from earth2studio.data.utils import fetch_data
from earth2studio.models.dx import CBottleInfill

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ground-truth data source (CMIP6 CanESM5, SSP585, daily table)
cmip6_ds_atmos = CMIP6(
    experiment_id="ssp585",
    source_id="CanESM5",
    table_id="day",
    variant_label="r1i1p2f1",
)
cmip6_ds_ocean = CMIP6(
    experiment_id="ssp585",
    source_id="CanESM5",
    table_id="Omon",
    variant_label="r1i1p2f1",
)

# ERA5 ARCO datasource for reference plots
arco_ds = ARCO(cache=False, verbose=False)

# Input variables
input_variables = ["u10m", "v10m"]
timestamp = datetime(2015, 1, 15, 12)
n_samples = 1

# Load the default model package which downloads the check point from NGC
package = CBottleInfill.load_default_package()
model = CBottleInfill.load_model(package, input_variables=input_variables + ["sst"])
# model = CBottleInfill.load_model(package, input_variables=input_variables)
model = model.to(device)

# Build target interpolation grid based on the model coordinates
lat_1d = model.input_coords()["lat"]
lon_1d = model.input_coords()["lon"]
_lat, _lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
interp_grid = {"_lat": _lat, "_lon": _lon}

model.set_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Prepare conditioning data
times = np.array([timestamp] * n_samples, dtype="datetime64[ns]")
x_atmos, coords_atmos = fetch_data(
    cmip6_ds_atmos,
    times,
    input_variables,
    device=device,
    interp_to=interp_grid,
    interp_method="linear",
)
x_ocean, coords_ocean = fetch_data(
    cmip6_ds_ocean,
    times,
    ["sst"],
    device=device,
    interp_to=interp_grid,
    interp_method="linear",
)

# ------------------------------------------------------------------
# Clean NaNs in SST: nearest-neighbour fill along longitude (dim = -1)
# ------------------------------------------------------------------


def _fill_nan_nearest(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fill NaNs by nearest neighbour along a single dimension.

    1. Forward-fill to get value/distance to previous valid sample.
    2. Back-fill (using reversed tensor) to get next valid sample.
    3. Pick whichever of the two is closer.  If one side is missing
       (row begins/ends with NaNs) fall back to the other side.
    """

    if not torch.isnan(x).any():
        return x

    # Helper: forward fill (propagate last valid value) *and* return index of that value
    def _forward_fill(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nan = torch.isnan(t)
        idx = torch.arange(t.shape[dim], device=t.device)
        view = [1] * t.ndim
        view[dim] = -1
        idx = idx.view(view).expand_as(t)
        idx = idx.masked_fill(nan, 0)
        last_idx = torch.cummax(idx, dim=dim).values
        filled = torch.take_along_dim(t, last_idx, dim=dim)
        return filled, last_idx

    # forward & backward (reverse) fills
    fwd_val, fwd_idx = _forward_fill(x)
    rev_val, rev_idx = _forward_fill(torch.flip(x, [dim]))
    bwd_val = torch.flip(rev_val, [dim])
    bwd_idx = x.shape[dim] - 1 - torch.flip(rev_idx, [dim])

    # Current index tensor for distance calculation
    idx = torch.arange(x.shape[dim], device=x.device)
    view = [1] * x.ndim
    view[dim] = -1
    idx = idx.view(view).expand_as(x)

    dist_prev = idx - fwd_idx
    dist_next = bwd_idx - idx

    # Conditions: choose nearer valid; if one side NaN use the other.
    use_prev = (torch.isnan(bwd_val)) | (
        ~torch.isnan(fwd_val) & (dist_prev <= dist_next)
    )

    filled = torch.where(use_prev, fwd_val, bwd_val)

    return torch.where(torch.isnan(x), filled, x)


# Apply filling
x_ocean = _fill_nan_nearest(x_ocean, dim=-1)

# Debug plot: visualise filled SST field
plt.figure(figsize=(8, 4))
plt.contourf(
    lon_1d,
    lat_1d,
    x_ocean[0, 0, 0].cpu(),
    levels=100,
    cmap="turbo",
)
plt.colorbar(label="SST (K)")
plt.title("SST after NaN filling")
plt.savefig("outputs/sst_filled_debug.jpg", dpi=150, bbox_inches="tight")
plt.close()

# Combine atmos and ocean data
x = torch.cat([x_atmos, x_ocean], dim=2)
coords = coords_atmos
coords["variable"] = np.concatenate([coords["variable"], coords_ocean["variable"]])
coords.pop("_lat")
coords.pop("_lon")
coords["lat"] = lat_1d
coords["lon"] = lon_1d

# Run the model
output, output_coords = model(x, coords)

# Post Processing
variable = "tcwv"
var_idx = np.where(output_coords["variable"] == "tcwv")[0][0]
era5_data, _ = fetch_data(
    arco_ds,
    times[:1],
    [variable],
    device=device,
    interp_to=interp_grid,
    interp_method="linear",
)

plt.close("all")
projection = ccrs.Mollweide(central_longitude=0)

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(1, 2, subplot_kw={"projection": projection}, figsize=(10, 6))


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


plot_contour(ax[0], era5_data[0, 0, 0])
plot_contour(ax[1], torch.mean(output[:, 0, var_idx], axis=0))

ax[0].set_title("ERA5 (ARCO)")
ax[1].set_title(f"Input Variables Mean: {input_variables}")

plt.tight_layout()
plt.savefig("outputs/15_infill_cmip6_cbottle.jpg")
