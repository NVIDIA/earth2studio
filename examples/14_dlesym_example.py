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
========================

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
- How to use the model API to generate a forecast
- How to use the output selection and regridding methods to select appropriate data
- How to use the DLESyMLatLon model with earth2studio workflows
"""
# /// script
# dependencies = [
#   "earth2studio[dlesym] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# The first step is fetching appropriate input data for the model. The ERA5 data sources
# in earth2studio provide data on the lat/lon grid, so have two options:
#
# - Use the :py:class:`earth2studio.models.px.DLESyMLatLon` model. This version of DLESyM
#   accepts inputs on the lat/lon grid and regrids them to the HEALPix grid internally,
#   before returning the output regridded back to the lat/lon grid. This is the
#   recommended approach for most users as it can be used directly with earth2studio
#   data sources and workflows, since it performs regridding and pre-processing
#   internally.
# - Use the :py:class:`earth2studio.models.px.DLESyM` model, and handle the regridding of
#   input lat/lon data ourselves. Since the model uses some derived variables which
#   are not provided by the data source, we would also need to prepare these derived
#   variables ourselves.
#
# Let's load both of these models and inspect the expected input coordinates for each.
# Also note the input and output variable set for each model.

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

device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("GPU/CUDA required for DLESyM")

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
print("Lat-lon input variable names: ", in_coords_ll["variable"])
print(
    "Lat-lon output variable names: ", model_ll.output_coords(in_coords_ll)["variable"]
)
print("HEALPix input variable names: ", in_coords_hpx["variable"])
print(
    "HEALPix output variable names: ",
    model_hpx.output_coords(in_coords_hpx)["variable"],
)
# %%
# Making Predictions, Regridding, and Selecting Outputs
# -----------------------------------------------------
# Let's now pull some example data and make predictions with the model. As the
# data source provides lat/lon data, we can use the :py:class:`earth2studio.models.px.DLESyMLatLon` model.
#
# In addition, we demonstrate how to use the regridding utilities provided by
# `DLESyMLatLon` to regrid onto the HEALPix grid. The :py:class:`earth2studio.models.px.DLESyM`
# model can then be used directly with the HEALPix data.
#
# Finally, a key aspect of the DLESyM model is that it makes predictions for the atmosphere
# and ocean components at different timesteps, because the atmosphere is faster-evolving
# than the ocean. The atmosphere is predicted every 6 hours, while the ocean is only
# predicted every 48 hours. Thus, not all output lead times are valid for the ocean
# component. For convenience, we can use a method that selects only the valid outputs
# for each of the atmosphere and ocean components.

# %%
ic_date = np.datetime64("2021-06-15")

# Fetch some example data
x, coords = fetch_data(
    source=data,
    time=np.array([ic_date]),
    variable=np.array(in_coords_ll["variable"]),
    lead_time=in_coords_ll["lead_time"],
    device=device,
)

# Can call the `DLESyMLatLon` model directly with the input lat/lon data
y, y_coords = model_ll(x, coords)

# Or, we can use the pre-processing and regridding utilities to regrid the data onto
# the HEALPix grid, and then run directly with `DLESyM`, which expects HEALPix data
x_prep, coords_prep = model_ll._prepare_derived_variables(x, coords)
x_hpx, coords_hpx = model_ll.to_hpx(x_prep), model_ll.coords_to_hpx(coords_prep)
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
# Model Iteration for Longer Forecasts
# ------------------------------------
# Similar to other models in earth2studio, we can use the model iterator to loop
# over forecasted outputs. A single step of the DLESyM model produces predictions
# over 4 days (96 hours), so to make a sub-seasonal forecast we can take 15 steps for
# a total of 60 days.

# %%
n_steps = 16
model_iter_ll = model_ll.create_iterator(x, coords)

for i in range(n_steps):
    x, x_coords = next(model_iter_ll)
    if i > 0:  # Don't retrieve the first step as it is the initial condition
        x_atmos, x_atmos_coords = model_ll.retrieve_valid_atmos_outputs(x, x_coords)
        x_ocean, x_ocean_coords = model_ll.retrieve_valid_ocean_outputs(x, x_coords)

print(f"Completed forecast with {n_steps} steps")


# %%
# Using Built-in Deterministic Workflow
# -------------------------------------
# Because the `DLESyMLatLon` model permits usage of data coming directly from an
# earth2studio data source, we can use the built-in deterministic workflow to generate
# a forecast as well. The only caveat is we need to explitcitly specify the output
# lead time coordinates that will be generated by the model, since it has different
# input and output lead time dimensions.

# %%
import earth2studio.run as run
from earth2studio.io import KVBackend

io = KVBackend()

output_coords = model_ll.output_coords(coords)
inp_lead_time = model_ll.input_coords()["lead_time"]
out_lead_times = [
    output_coords["lead_time"] + output_coords["lead_time"][-1] * i
    for i in range(n_steps)
]
output_coords["lead_time"] = np.concatenate([inp_lead_time, *out_lead_times])
io = run.deterministic(
    [ic_date], n_steps, model_ll, data, io, output_coords=output_coords
)

ds = io.to_xarray()
print(ds)

# %%
# Plotting the Outputs
# --------------------
# Let's plot some of the forecasted outputs for the atmosphere and ocean components.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# lat = x_atmos_coords["lat"]
# lon = x_atmos_coords["lon"]
atmos_var, atmos_units = "ws10m", "m/s"
ocean_var, ocean_units = "sst", "K"
# atmos_var_idx = list(x_atmos_coords["variable"]).index(atmos_var)
# ocean_var_idx = list(x_ocean_coords["variable"]).index(ocean_var)
lead_time = ds.lead_time.values[-1]

plt.close("all")
# Create a Robinson projection
projection = ccrs.Robinson()

# Create a figure and axes with the specified projection
fig, axs = plt.subplots(1, 2, subplot_kw={"projection": projection}, figsize=(15, 6))

# Plot the field using pcolormesh
im = axs[0].pcolormesh(
    ds.lon.values,
    ds.lat.values,
    ds[atmos_var].sel(time=ic_date, lead_time=lead_time).values,
    transform=ccrs.PlateCarree(),
    cmap="cividis",
)

# Set title
axs[0].set_title(
    f"Initialization: {ic_date} - Lead time: {lead_time.astype('timedelta64[h]')}"
)

# Add coastlines and gridlines
axs[0].coastlines()
axs[0].gridlines()

cbar = fig.colorbar(im, ax=axs[0], orientation="horizontal", pad=0.05)
cbar.set_label(f"{atmos_var} [{atmos_units}]")

# Plot the ocean component
im = axs[1].pcolormesh(
    ds.lon.values,
    ds.lat.values,
    ds[ocean_var].sel(time=ic_date, lead_time=lead_time).values,
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
)

axs[1].set_title(
    f"Initialization: {ic_date} - Lead time: {lead_time.astype('timedelta64[h]')}"
)

# Add coastlines and gridlines
axs[1].add_feature(cfeature.LAND, color="grey", zorder=100)
axs[1].coastlines()
axs[1].gridlines()

cbar = fig.colorbar(im, ax=axs[1], orientation="horizontal", pad=0.05)
cbar.set_label(f"{ocean_var} [{ocean_units}]")

plt.tight_layout()
plt.savefig("outputs/14_ws10m_sst_prediction.png")
