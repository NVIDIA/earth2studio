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
Running DLESyM Inference
===============================

Basic deterministic inference workflow for the DLESyM model.

This example will demonstrate how to run a simple inference workflow with the
DLESyM model, which differs from other prognostic models in earth2studio.
DLESyM performs global earth system modeling, including atmosphere and ocean
components which evolve on different timescales (different temporal resolution).
Internally, the model uses a HEALPix nside=64 (approximately 1 degree) resolution
grid for the physical variables of interest. The model also uses some derived input
variables which are not provided by standard data sources but can be computed
from the standard variables.

In this example you will learn:

- How to instantiate the DLESyM model
- How to prepare input data from an ERA5 data source
- How to use the model API to generate a forecast
- How to use the output selection and regridding methods to select appropriate data
"""

# %%
# Set Up
# ------
# The first step is fetching appropriate input data for the model. The ERA5 data sources
# in earth2studio provide data on the lat/lon grid, so we need to either:
#  -  use the :py:class:`earth2studio.models.DLESyMLatLon` model. This version of DLESyM
#     accepts inputs on the lat/lon grid and regrids them to the HEALPix grid internally,
#     before returning the output regirdded back to the lat/lon grid.
#  -  use the :py:class:`earth2studio.models.DLESyM` model, and handle the regridding of
#     input lat/lon data ourselves.
# Let's load both of these models and inspect the input coordinates.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function
import numpy as np
import torch

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px.dlesym import DLESyM, DLESyMLatLon

device = "cpu" if not torch.cuda.is_available() else "cuda"

# Create the data source
data = ARCO()

# Load the default model package which downloads the check point from NGC
# We will instantiate both versions of the model to demonstrate usage of each.
package = DLESyMLatLon.load_default_package()
model_ll = DLESyMLatLon.load_model(package).to(device)
package = DLESyM.load_default_package()
model_hpx = DLESyM.load_model(package).to(device)

in_coords_ll = model_ll.input_coords()
in_coords_hpx = model_hpx.input_coords()
print(
    "DLESyM LatLon input coord shapes: ",
    [(k, v.shape) for k, v in in_coords_ll.items()],
)
print(
    "DLESyM HPX input coord shapes: ", [(k, v.shape) for k, v in in_coords_hpx.items()]
)
print("Variable names: ", in_coords_ll["variable"])

# %%
# Preparing data
# --------------------
# Note the spatial dimensions are different between the lat/lon and HEALPix
# versions of the model. Also note the variable names: we need to construct
# the following from the ERA5 data source:
# - `tau300-700` (geopotential thickness) is defined as the difference between
#  z300 and z700 geopotential levels
# - `ws10m` (wind speed at 10m above surface) is defined as the square root of
#  the sum of the squared zonal and meridional wind components, i.e.
#  `sqrt(u10m **2 + v10m **2)`
# - `sst` (sea surface temperature) is supported by the data source but filled
#  with nans over landmasses. We will deal with these using a custom interpolation
#  scheme
# Let's pull the data from the ERA5 data source and construct the above variables.
# %%
import xarray as xr

ic_date = np.datetime64("2021-06-15")

# Replace the derived variables with those in the data source so we can compute the derived variables
variables_to_fetch = [
    v for v in list(in_coords_ll["variable"]) if v not in ["tau300-700", "ws10m"]
]
variables_to_fetch.extend(["u10m", "v10m", "z300", "z700"])
x, coords = fetch_data(
    source=data,
    time=np.array([ic_date]),
    variable=np.array(variables_to_fetch),
    lead_time=in_coords_ll["lead_time"],
    device=device,
)


# Use helper functions to prepare the derived variables
def nan_interpolate_sst(sst, coords):
    """
    Interpolate the SST data to fill nans over landmasses.

    Args:
        sst (torch.Tensor): The SST data.
        coords (dict): The coordinates of the input data.

    Returns:
        torch.Tensor: The interpolated SST data.
    """

    # First pass: interpolate along longitude
    da_sst = xr.DataArray(sst.cpu().numpy(), dims=coords.keys())
    da_interp = da_sst.interpolate_na(dim="lon", method="linear", use_coordinate=False)

    # Second pass: roll, interpolate along longitude, and unroll
    roll_amount_lon = int(len(da_interp.lon) / 2)
    da_double_interp = (
        da_interp.roll(lon=roll_amount_lon, roll_coords=False)
        .interpolate_na(dim="lon", method="linear", use_coordinate=False)
        .roll(lon=len(da_interp.lon) - roll_amount_lon, roll_coords=False)
    )

    # Third pass do a similar roll along latitude
    roll_amount_lat = int(len(da_double_interp.lat) / 2)
    da_triple_interp = (
        da_double_interp.roll(lat=roll_amount_lat, roll_coords=False)
        .interpolate_na(dim="lat", method="linear", use_coordinate=False)
        .roll(lat=len(da_double_interp.lat) - roll_amount_lat, roll_coords=False)
    )

    return torch.from_numpy(da_triple_interp.values).to(sst.device)


def prepare_derived_variables(x, coords, target_vars):
    """
    Prepare the derived variables for the DLESyM model.

    Parameters
    ----------
    x : torch.Tensor
        The input data tensor from an earth2studio data source.
    coords : dict
        The coordinates of the input data.
    target_vars : list
        The target variables to prepare.

    Returns
    -------
    x : torch.Tensor
        The input data tensor with the derived variables.
    coords : dict
        The coordinates of the input data with the derived variables.
    """
    # Fetch the base variables
    base_vars = list(coords["variable"])
    src_vars = {
        v: x[..., base_vars.index(v) : base_vars.index(v) + 1, :, :] for v in base_vars
    }

    # Compute the derived variables
    out_vars = {
        "ws10m": torch.sqrt(src_vars["u10m"] ** 2 + src_vars["v10m"] ** 2),
        "tau300-700": src_vars["z300"] - src_vars["z700"],
    }
    out_vars.update(src_vars)

    # Fill SST nans with a constant value
    out_vars["sst"] = nan_interpolate_sst(out_vars["sst"], coords)

    # Update the tensor with the derived variables and return
    coords["variable"] = target_vars
    x_out = torch.empty(*[v.shape[0] for v in coords.values()], device=x.device)
    for i, v in enumerate(coords["variable"]):
        x_out[..., i : i + 1, :, :] = out_vars[v]

    # Add a batch dim
    if x.ndim < 7:
        x_out = x_out.unsqueeze(0)
        coords["batch"] = np.array([0])
        coords.move_to_end("batch", last=False)

    return x_out, coords


x, coords = prepare_derived_variables(x, coords, target_vars=in_coords_ll["variable"])

print("Fetched and pre-processed variables: ", coords["variable"])


# %%
# Making predictions, regridding, and selecting outputs
# ---------------
# Since we now have the appropriate input physical variables prepared, we can
# make predictions using the DLESyM model. As the data source provides lat/lon
# data, we can use the :py:class:`earth2studio.models.DLESyMLatLon` model.

# In addition, we demonstrate how to use the regridding utilities provided by
# `DLESyMLatLon` to regrid onto the HEALPix grid. The :py:class:`earth2studio.models.DLESyM`
# model can then be used directly with the HEALPix data.

# Finally, a key aspect of the DLESyM model is that it makes predictions for the atmosphere
# and ocean components at different timesteps, becasue the atmosphere is faster-evolving
# than the ocean. The atmosphere is predicted every 6 hours, while the ocean is only
# predicted every 48 hours. Thus, not all output lead times are valid for the ocean
# component. For convenience, we can use a method that selects only the valid outputs
# for each of the atmosphere and ocean components.

# %%
# Can call the `DLESyMLatLon` model directly with the input lat/lon data
y, y_coords = model_ll(x, coords)

# Or, we can use the regridding utilities ourself to regrid the data onto the HEALPix grid
# Then run directly with `DLESyM`, which expects HEALPix data
x_hpx, coords_hpx = model_ll.to_hpx(x), model_ll.coords_to_hpx(coords)
y_hpx, y_coords_hpx = model_hpx(x_hpx, coords_hpx)

# Retrieve the valid outputs for atmos/ocean components from the predictions
y_atmos, y_atmos_coords = model_ll.retrieve_valid_atmos_outputs(y, y_coords)
y_ocean, y_ocean_coords = model_ll.retrieve_valid_ocean_outputs(y, y_coords)

print(
    "Atmosphere outputs (variables, lead_time [hrs]):",
    y_atmos_coords["variable"],
    y_atmos_coords["lead_time"].astype("timedelta64[h]"),
)
print(
    "Ocean outputs (variables, lead_time [hrs]):",
    y_ocean_coords["variable"],
    y_ocean_coords["lead_time"].astype("timedelta64[h]"),
)

# %%
# Model iteration for longer forecasts
# ------------------------------
# Similar to other models in earth2studio, we can use the model iterator to loop
# over forecasted outputs. A single step of the DLESyM model produces predictions
# over 4 days (96 hours), so to make a sub-seasonal forecast we can take 15 steps for
# a total of 60 days.

# %%
n_steps = 16
model_iter_ll = model_ll.create_iterator(x, coords)

for i in range(n_steps):
    x, coords = next(model_iter_ll)
    print(i)
    if i > 0:
        # Don't retrieve the first step as it is the initial condition
        x_atmos, x_atmos_coords = model_ll.retrieve_valid_atmos_outputs(x, coords)
        print(x_atmos.min(), x_atmos.mean(), x_atmos.max())
        x_ocean, x_ocean_coords = model_ll.retrieve_valid_ocean_outputs(x, coords)

print(f"Completed forecast with {n_steps} steps")
# %%
# Plotting the outputs
# --------------------
# Let's plot some of the forecasted outputs for the atmosphere and ocean components.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

lat = x_atmos_coords["lat"]
lon = x_atmos_coords["lon"]
atmos_var, atmos_units = "ws10m", "m/s"
ocean_var, ocean_units = "sst", "K"
atmos_var_idx = list(x_atmos_coords["variable"]).index(atmos_var)
ocean_var_idx = list(x_ocean_coords["variable"]).index(ocean_var)
lead_time_idx = -1

plt.close("all")
# Create a Robinson projection
projection = ccrs.Robinson()

# Create a figure and axes with the specified projection
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": projection}, figsize=(15, 6))

# Plot the field using pcolormesh
im = axs[0].pcolormesh(
    lon,
    lat,
    x_atmos[0, 0, lead_time_idx, atmos_var_idx, :, :].cpu().numpy(),
    transform=ccrs.PlateCarree(),
    cmap="cividis",
)

# Set title
axs[0].set_title(
    f"{ic_date} - Lead time: {x_atmos_coords['lead_time'][lead_time_idx].astype('timedelta64[h]')}"
)

# Add coastlines and gridlines
axs[0].coastlines()
axs[0].gridlines()

cbar = fig.colorbar(im, ax=axs[0], orientation="horizontal", pad=0.05)
cbar.set_label(f"{atmos_var} [{atmos_units}]")

# Plot the ocean component
im = axs[1].pcolormesh(
    lon,
    lat,
    x_ocean[0, 0, lead_time_idx, ocean_var_idx, :, :].cpu().numpy(),
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
)

axs[1].set_title(
    f"{ic_date} - Lead time: {x_ocean_coords['lead_time'][lead_time_idx].astype('timedelta64[h]')}"
)

# Add coastlines and gridlines
axs[1].add_feature(cfeature.LAND, color="grey", zorder=100)
axs[1].coastlines()
axs[1].gridlines()

cbar = fig.colorbar(im, ax=axs[1], orientation="horizontal", pad=0.05)
cbar.set_label(f"{ocean_var} [{ocean_units}]")

plt.tight_layout()
plt.savefig("outputs/14_ws10m_sst_prediction.png")
