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
Running DLESyM Inference
========================

DLESyM under ``Pipeline``, fused and as coupled atmosphere and ocean components.

DLESyM is a global earth system model with an atmosphere that advances every
6 hours and an ocean that advances every 48 hours, both on a HEALPix nside=64
(approximately 1 degree) grid. The fused
[`earth2studio.models.px.DLESyM`][earth2studio.models.px.DLESyM] model steps both
inside one 96-hour cycle and returns every atmosphere lead time, with the ocean
valid only at its own. This example runs the model in two ways:

- fused, as any other prognostic model under
  [`earth2studio.run.Pipeline`][earth2studio.run.Pipeline], which rolls it out with a
  `PrognosticRunner` and writes one store;
- split into [`earth2studio.models.px.DLESyMAtmos`][earth2studio.models.px.DLESyMAtmos]
  and [`earth2studio.models.px.DLESyMOcean`][earth2studio.models.px.DLESyMOcean]
  components under [`earth2studio.coupling.CoupledRunner`][earth2studio.coupling.CoupledRunner],
  where the sea surface temperature and the 48-hour atmospheric means the two halves
  exchange pass through explicit
  [`earth2studio.coupling.Connector`][earth2studio.coupling.Connector] edges, and each
  component writes its own stream at its own cadence.

In this example you will learn:

- How to instantiate DLESyM and run it from a lat/lon data source with `Pipeline`
- How to split DLESyM into atmosphere and ocean components that share its weights
- How to serve the components' native HEALPix inputs from ERA5 with a data source adapter
- How to declare the coupling with connectors, read the derived schedule and run it
  under `Pipeline` with one store per component
- How to regrid the native outputs for plotting
"""

# /// script
# dependencies = [
#   "earth2studio[dlesym] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# The ERA5 data sources in Earth2Studio provide data on the lat/lon grid, so we load
# the [`earth2studio.models.px.DLESyMLatLon`][earth2studio.models.px.DLESyMLatLon]
# wrapper. It accepts lat/lon inputs, computes the derived variables the model needs
# (``ws10m`` from the wind components and ``tau300-700`` from geopotential), fills the
# sea surface temperature over land, regrids onto HEALPix internally and returns
# lat/lon outputs. It also carries everything the split components need, so one
# package load serves both runs.
#
# - Prognostic Model: [`earth2studio.models.px.DLESyMLatLon`][earth2studio.models.px.DLESyMLatLon]
# - Datasource: [`earth2studio.data.ARCO_ERA5`][earth2studio.data.ARCO_ERA5]
# - IO Backend: [`earth2studio.io.KVBackend`][earth2studio.io.KVBackend], in memory

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function
import numpy as np
import torch

from earth2studio.data import ARCO_ERA5
from earth2studio.io import KVBackend
from earth2studio.models.px import DLESyMLatLon
from earth2studio.run import Pipeline, map_field, work_items

device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("GPU/CUDA required for DLESyM")

data = ARCO_ERA5()

package = DLESyMLatLon.load_default_package()
model = DLESyMLatLon.load_model(package).to(device)

in_coords = model.input_coords()
print("Lat/lon input variables:", list(in_coords["variable"].values))
print("Input lead times [h]:", in_coords["lead_time"].values.astype("timedelta64[h]"))
out_coords = model.output_coords(in_coords)
print("Output variables:", list(out_coords["variable"].values))
print("Output lead times [h]:", out_coords["lead_time"].values.astype("timedelta64[h]"))

# %%
# The Fused Model Under Pipeline
# ------------------------------
# A `Pipeline` drives a runner over work items. `Pipeline.from_model` wraps the
# prognostic model in a `PrognosticRunner` that fetches the initial history from the
# data source, rolls the model forward and writes every step into the store. One
# DLESyM step is a 96-hour cycle, so the final lead time is a multiple of 96 hours.
#
# Only the fields named in ``output_variables`` are written. The ocean's ``sst`` is
# valid at 48-hour lead times only: the fused model fills the other lead times with
# NaNs, because its one output slot carries the atmosphere's lead times.

# %%
ic_date = np.datetime64("2021-06-15")
n_cycles = 8  # 8 cycles of 96 hours: 32 days
items = work_items([ic_date], final_lead_time=np.timedelta64(96 * n_cycles, "h"))

fused = KVBackend()
Pipeline.from_model(
    model,
    data,
    output={"forecast": fused},
    output_variables={"forecast": ["ws10m", "sst"]},
    device=device,
).run(items)

fused_ds = fused.to_xarray()
print(fused_ds)
valid_sst = fused_ds["sst"].notnull().any(["lat", "lon"]).squeeze("time")
print(
    "Lead times with valid sst [h]:",
    fused_ds["lead_time"].values[valid_sst.values].astype("timedelta64[h]"),
)

# %%
# Splitting DLESyM Into Components
# --------------------------------
# A fused model whose outputs advance at different cadences cannot be coupled as it
# is. `DLESyM.split` returns the atmosphere and ocean as two ordinary prognostic
# models on the native HEALPix grid that share the fused model's weights, so nothing
# is copied. Each declares its own slots:
#
# - The atmosphere takes its 6-hourly history and, as forcing, the sea surface
#   temperature at the history's final lead time. In the fused model this is the
#   SST at the start of each cycle, the ocean's previous output.
# - The ocean takes its history at -48 h and 0 h and, as forcing, the means of
#   ``z1000`` and ``ws10m`` over the atmosphere frames leading up to each of its
#   output times, declared as qualified labels such as ``z1000:mean:-42h:+6h``.
#   In the fused model these are averages of the atmosphere's frames from the
#   same cycle.
#
# The coupler reads these declarations; nothing in it knows about DLESyM.

# %%
from earth2studio.coupling import Connector, CoupledRunner

atmos, ocean = model.split()


def hours(coords):
    """Lead times of a signature in whole hours."""
    return coords["lead_time"].values.astype("timedelta64[h]")


def print_slots(name, component):
    """Print a component's slots, as declared."""
    inputs, forcing = component.input_coords(), component.forcing_coords()
    outputs = component.output_coords(inputs)
    print(f"{name} input:   {list(inputs['variable'].values)} at {hours(inputs)}")
    print(f"{name} forcing: {list(forcing['variable'].values)} at {hours(forcing)}")
    print(f"{name} output:  {list(outputs['variable'].values)} at {hours(outputs)}")


print_slots("atmos", atmos)
print_slots("ocean", ocean)

# %%
# Native Inputs From a Lat/Lon Source
# -----------------------------------
# The components' input slots are on HEALPix with the model's derived variables, which
# no data source provides. Models never fetch data, so the preparation belongs in a
# source: a small `DataSource` adapter fetches the lat/lon base variables from ERA5
# and applies `DLESyMLatLon.to_native`, the same preparation the fused wrapper gives
# its own inputs. The runner then fetches each component's history through it, one
# call per lead time.


class NativeERA5:
    """Serve DLESyM's native HEALPix input variables from a lat/lon source."""

    def __init__(self, source, model):
        self.source = source
        self.model = model
        self.signature = model.input_coords()

    def __call__(self, time, variable):
        """Fetch the base variables, then return the native ones requested."""
        x = self.source(time, self.signature["variable"].values)
        # Lay the fields out on the model's declared lat/lon grid before preparing
        # them; map_field also checks the grid metadata matches.
        x = map_field(x, self.signature)
        return self.model.to_native(x).sel(variable=np.atleast_1d(variable))


native = NativeERA5(data, model)

# %%
# Declaring the Coupling
# ----------------------
# Components are named models and connectors are edges between them. Both components
# are on the same HEALPix grid, so no regridder is needed.
#
# - The ocean feeds the atmosphere's SST slot with ``time="hold"``: the atmosphere
#   takes the latest SST the ocean published before the cycle, which is what the fused
#   model does. With ``exact`` the two would wait for each other, and the runner
#   would report the cycle when built.
# - The atmosphere feeds the ocean's window means with the default ``exact`` timing:
#   the coupler reduces the atmosphere's 6-hourly frames over each 48-hour window.
#
# Every slot not fed by a connector, here the two histories, is fetched from its
# source. Construction validates the declarations and dry-runs the schedule, and
# `describe` prints it: each row is a call, when it is due, what it publishes and
# where each connected slot reads from.

# %%
runner = CoupledRunner(
    {"atmos": atmos, "ocean": ocean},
    [
        Connector("ocean", "atmos", time="hold"),
        Connector("atmos", "ocean"),
    ],
    sources={"atmos": native, "ocean": native},
)
print(runner.describe(np.timedelta64(192, "h")))

# %%
# Running the Coupled Model Under Pipeline
# ----------------------------------------
# A coupled runner is a `Runner`, so the same `Pipeline` drives it over the same
# work items. It plans one store per component stream, each at that component's own
# lead times: the atmosphere every 6 hours and the ocean every 48 hours, with no
# NaN padding. Both start with the initial condition at lead time zero.
#
# !!! note
#     The default checkpoints sample conditional layer norm noise. Each component
#     draws its own stream, seeded from the component's name, so the coupled run is
#     another realization of the same model rather than a bitwise copy of the fused
#     run. With deterministic cores the two agree exactly.

# %%
atmos_io, ocean_io = KVBackend(), KVBackend()
Pipeline(runner, output={"atmos": atmos_io, "ocean": ocean_io}, device=device).run(
    items
)

atmos_ds, ocean_ds = atmos_io.to_xarray(), ocean_io.to_xarray()
print(atmos_ds)
print(ocean_ds)
print(
    "Atmosphere lead times [h]:", atmos_ds["lead_time"].values.astype("timedelta64[h]")
)
print("Ocean lead times [h]:", ocean_ds["lead_time"].values.astype("timedelta64[h]"))

# %%
# Post Processing
# ---------------
# The coupled streams are on the native HEALPix grid. For plotting, `DLESyMLatLon`
# exposes its regridder as `to_ll`, which maps the trailing ``(face, height, width)``
# dimensions onto the 0.25 degree lat/lon grid. We plot the 10 m wind speed from the
# atmosphere stream and the sea surface temperature from the ocean stream at the
# final lead time, next to the fused model's wind speed at the same lead time.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def to_latlon(field):
    """Regrid a HEALPix field with trailing (face, height, width) onto lat/lon."""
    tensor = torch.as_tensor(np.asarray(field), device=device)
    return model.to_ll(tensor).cpu().numpy()


lead_time = atmos_ds["lead_time"].values[-1]
ws10m = to_latlon(atmos_ds["ws10m"].sel(time=ic_date, lead_time=lead_time).values)
sst = to_latlon(ocean_ds["sst"].sel(time=ic_date, lead_time=lead_time).values)
ws10m_fused = fused_ds["ws10m"].sel(time=ic_date, lead_time=lead_time).values
label = f"Initialization: {ic_date} - Lead time: {lead_time.astype('timedelta64[h]')}"

plt.close("all")
projection = ccrs.Robinson()
fig, axs = plt.subplots(1, 3, subplot_kw={"projection": projection}, figsize=(21, 6))

for ax, field, title, cmap, units in (
    (axs[0], ws10m_fused, "Fused ws10m", "cividis", "m/s"),
    (axs[1], ws10m, "Coupled ws10m", "cividis", "m/s"),
    (axs[2], sst, "Coupled sst", "Spectral_r", "K"),
):
    im = ax.pcolormesh(
        model.lon, model.lat, field, transform=ccrs.PlateCarree(), cmap=cmap
    )
    ax.set_title(f"{title}\n{label}")
    if title.endswith("sst"):
        ax.add_feature(cfeature.LAND, color="grey", zorder=100)
    ax.coastlines()
    ax.gridlines()
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label(f"{title.split()[-1]} [{units}]")

plt.tight_layout()
plt.savefig("outputs/02_dlesym_coupled_ws10m_sst.jpg")
