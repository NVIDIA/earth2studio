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
Running the DLESyM ISCCP-ERA5 Model
===================================

Coupled inference with the upstream DLESyM checkpoints, including precipitation.

This example demonstrates the upstream DLESyM model distributed by the
[AtmosSci-DLESM/DLESyM](https://github.com/AtmosSci-DLESM/DLESyM) group
(the University of Washington team behind Cresswell-Clay et al. 2024). It shares
the coupled atmosphere/ocean HEALPix architecture of the base
[`earth2studio.models.px.DLESyM`][earth2studio.models.px.DLESyM] model, but differs in a few ways that
are worth highlighting:

- The atmosphere carries an outgoing longwave radiation (OLR) channel. The model
  was trained on ISCCP-distributed OLR, so the wrapper accepts ERA5 ``ttr`` and
  applies a per-day-of-year moment-matching transform to convert it to OLR
  internally (controlled by the ``use_ttr`` flag).
- A separate [`earth2studio.models.dx.DLESyMv0_ISCCP_ERA5Precip`][earth2studio.models.dx.DLESyMv0_ISCCP_ERA5Precip]
  diagnostic predicts 6-hourly accumulated precipitation (``tp06``) from the
  full coupled state.

In this example you will learn:

- How to instantiate the upstream prognostic and precipitation models
- How the ``ttr`` -> OLR transform shows up in the input/output variable sets
- How to run a coupled forecast with the lat/lon convenience wrapper
- How to chain the precipitation diagnostic off the prognostic output
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
# As with the base DLESyM model, the upstream checkpoints run on a HEALPix
# ``nside=64`` grid, with derived input variables that no data source provides.
# [`earth2studio.models.px.DLESyMv0_ISCCP_ERA5`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5]
# recommends a source that prepares them: `DLESyMInputSource` fetches the base
# variables from a lat/lon source (ARCO ERA5 by default), computes the derived
# variables, fills the sea surface temperature over land and regrids onto HEALPix.
# The model, the precipitation diagnostic and their outputs all stay on the native
# grid; we regrid onto lat/lon only for plotting.
#

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function
import numpy as np
import torch

from earth2studio.data.utils import fetch_data
from earth2studio.models.dx import DLESyMv0_ISCCP_ERA5Precip
from earth2studio.models.px import DLESyMv0_ISCCP_ERA5

device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("GPU/CUDA required for DLESyM")

# Load the coupled prognostic and the precip diagnostic.
package = DLESyMv0_ISCCP_ERA5.load_default_package()
model = DLESyMv0_ISCCP_ERA5.load_model(package).to(device)

# The model's recommended source: ARCO ERA5 prepared as native HEALPix inputs
data = model.default_sources()

# The prognostic output is already in OLR space, so we load the precip
# diagnostic with ``use_ttr=False`` -- no further TTR -> OLR transform is needed
# when chaining off the model's own output. (Pass ``use_ttr=True`` to run the
# diagnostic standalone from an ERA5 initial condition instead.)
precip = DLESyMv0_ISCCP_ERA5Precip.load_model(package, use_ttr=False).to(device)

# %%
# Inspecting the Variable Sets
# ----------------------------
# Note that ``ttr`` appears in the prognostic input variables (the wrapper
# converts it to OLR internally), while the output variables report ``rlut``
# (OLR) -- the model variable space. The precip diagnostic consumes the full
# 10-variable coupled state and emits a single ``tp06`` field.

# %%
in_coords = model.input_coords()
out_coords_vars = model.output_coords(in_coords)["variable"]
print("Prognostic input variables: ", in_coords["variable"])
print("Prognostic output variables:", out_coords_vars)
print("Precip input variables:     ", precip.input_coords()["variable"])

# %%
# Making a Coupled Prediction
# ---------------------------
# We fetch an ERA5 initial condition, prepared on the HEALPix grid by the
# source, and run the model directly. As with the base DLESyM model, the
# atmosphere is predicted every 6 hours while the ocean only advances every
# 48 hours, so we use the ``retrieve_valid_*`` helpers to select the valid lead
# times for each component.

# %%
ic_date = np.datetime64("2021-06-15")

x = fetch_data(
    source=data,
    time=np.array([ic_date]),
    variable=np.array(in_coords["variable"]),
    lead_time=in_coords.coords["lead_time"].values,
    device=device,
)

# Run a single coupled step on the native grid
from earth2studio.run import _dimension_coords, map_field

x = map_field(x, in_coords)
y = model(x)

y_atmos = model.retrieve_valid_atmos_outputs(y)
y_ocean = model.retrieve_valid_ocean_outputs(y)
y_atmos_coords = _dimension_coords(y_atmos)
y_ocean_coords = _dimension_coords(y_ocean)

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
# Rolling Out to a 16-Day Forecast
# --------------------------------
# A single coupled step advances 96 hours (4 days), so to reach a 16-day lead
# time we roll the iterator forward four steps. The iterator yields the initial
# condition first, then one coupled step per iteration; we keep the final step
# and re-select the valid atmosphere/ocean outputs from it.

# %%
forecast_days = 16
hours_per_step = int(model.atmos_output_times[-1] / np.timedelta64(1, "h"))  # 96h
n_steps = int(np.ceil(forecast_days * 24 / hours_per_step))

model_iter = model.create_iterator(x)
# The iterator yields forecasts only (no initial condition)
for _ in range(n_steps):
    y = next(model_iter)

y_atmos = model.retrieve_valid_atmos_outputs(y)
y_ocean = model.retrieve_valid_ocean_outputs(y)
y_coords = _dimension_coords(y)
y_atmos_coords = _dimension_coords(y_atmos)
y_ocean_coords = _dimension_coords(y_ocean)
print(
    "Final forecast lead time:",
    y_coords["lead_time"][-1].astype("timedelta64[h]"),
)

# %%
# Diagnosing Precipitation
# ------------------------
# The precip diagnostic predicts 6-hourly accumulated precipitation from two
# consecutive 6-hourly atmosphere timesteps of the full coupled state. The
# 16-day prognostic output ``y`` already contains exactly the 10 coupled
# variables the diagnostic expects (in OLR / ``rlut`` space), so we take its
# last two atmosphere lead times as the ``[-6, 0]`` history window. Both models
# are on the same HEALPix grid, so no regridding is needed.

# %%
# The prognostic output carries the coupled variables in the prognostic's
# channel order, which differs from the order the precip diagnostic expects
# (e.g. ``ws10m`` and ``rlut`` are swapped). Reorder the variable axis to match
# the diagnostic's input variables before feeding it.
precip_vars = precip.input_coords().coords["variable"].values

# Last two atmosphere lead times form the [-6h, 0h] window relative to the
# diagnosed valid time; relative spacing is what the diagnostic validates.
precip_in = y.isel(lead_time=slice(-2, None)).sel(variable=precip_vars)

tp = precip(precip_in)
tp_coords = _dimension_coords(tp)
print(
    "Precip output (variable, valid lead_time [hrs]):",
    tp_coords["variable"],
    tp_coords["lead_time"].astype("timedelta64[h]"),
)

# %%
# Plotting the Outputs
# --------------------
# We plot a forecasted atmosphere field, the ocean SST, and the diagnosed
# precipitation, regridded from HEALPix onto a 1 degree lat/lon grid with a
# standalone [`earth2studio.grids.NearestRegridder`][earth2studio.grids.NearestRegridder].

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from earth2studio.grids import LatLonGrid, NearestRegridder

lat = np.linspace(90, -90, 181)
lon = np.linspace(0, 360, 360, endpoint=False)
to_latlon = NearestRegridder(
    in_coords.attrs["earth2studio_grid_id"], LatLonGrid(lat, lon)
)
atmos_var, atmos_units = "ws10m", "m/s"
ocean_var, ocean_units = "sst", "K"
atmos_idx = list(y_atmos_coords["variable"]).index(atmos_var)
ocean_idx = list(y_ocean_coords["variable"]).index(ocean_var)

plt.close("all")
projection = ccrs.Robinson()
fig, axs = plt.subplots(1, 3, subplot_kw={"projection": projection}, figsize=(20, 6))

# Atmosphere: 10m wind speed at the final atmos lead time
atmos_lead = y_atmos_coords["lead_time"][-1]
im = axs[0].pcolormesh(
    lon,
    lat,
    to_latlon(y_atmos[0, -1, atmos_idx].e2s.as_numpy()).values,
    transform=ccrs.PlateCarree(),
    cmap="cividis",
)
axs[0].set_title(f"{atmos_var} @ {atmos_lead.astype('timedelta64[h]')}")
axs[0].coastlines()
axs[0].gridlines()
cbar = fig.colorbar(im, ax=axs[0], orientation="horizontal", pad=0.05)
cbar.set_label(f"{atmos_var} [{atmos_units}]")

# Ocean: SST at the final ocean lead time
ocean_lead = y_ocean_coords["lead_time"][-1]
im = axs[1].pcolormesh(
    lon,
    lat,
    to_latlon(y_ocean[0, -1, ocean_idx].e2s.as_numpy()).values,
    transform=ccrs.PlateCarree(),
    cmap="Spectral_r",
)
axs[1].set_title(f"{ocean_var} @ {ocean_lead.astype('timedelta64[h]')}")
axs[1].add_feature(cfeature.LAND, color="grey", zorder=100)
axs[1].coastlines()
axs[1].gridlines()
cbar = fig.colorbar(im, ax=axs[1], orientation="horizontal", pad=0.05)
cbar.set_label(f"{ocean_var} [{ocean_units}]")

# Precip: 6-hourly accumulated precipitation on a log color scale. We pin the
# color range to a fixed physical window (0.1 mm .. 50 mm per 6 h) rather than
# auto-scaling: the log-precip inverse transform can produce a handful of
# grid-scale outliers, and an auto-scaled norm would chase those and wash out
# the real field. Values are clipped into the window for display only.
precip_lead = tp_coords["lead_time"][-1]
precip_field = np.clip(to_latlon(tp[0, 0, 0].e2s.as_numpy()).values, 0.0, None)
vmin, vmax = 1e-4, 5e-2  # metres of accumulated precip over 6 h
im = axs[2].pcolormesh(
    lon,
    lat,
    np.clip(precip_field, vmin, vmax),
    transform=ccrs.PlateCarree(),
    cmap="GnBu",
    norm=LogNorm(vmin=vmin, vmax=vmax),
)
axs[2].set_title(f"tp06 @ {precip_lead.astype('timedelta64[h]')}")
axs[2].coastlines()
axs[2].gridlines()
cbar = fig.colorbar(im, ax=axs[2], orientation="horizontal", pad=0.05, extend="both")
cbar.set_label("tp06 [m] (log scale)")

plt.suptitle(f"Upstream DLESyM forecast - Initialization: {ic_date}")
plt.tight_layout()
plt.savefig("outputs/03_dlesym_climate_prediction.png")
