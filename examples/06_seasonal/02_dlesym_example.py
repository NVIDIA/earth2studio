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
- as the standalone [`earth2studio.models.px.DLESyMAtmos`][earth2studio.models.px.DLESyMAtmos]
  and [`earth2studio.models.px.DLESyMOcean`][earth2studio.models.px.DLESyMOcean]
  components under [`earth2studio.coupling.CoupledRunner`][earth2studio.coupling.CoupledRunner],
  where the sea surface temperature and the 48-hour atmospheric means the two halves
  exchange pass through explicit
  [`earth2studio.coupling.Connector`][earth2studio.coupling.Connector] edges, and each
  component writes its own stream at its own cadence.

In this example you will learn:

- How to instantiate DLESyM and run it from a lat/lon data source with `Pipeline`
- How the model's recommended source serves its native HEALPix inputs from ERA5
- How to load the atmosphere and ocean as standalone components
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
# DLESyM runs on the HEALPix grid and takes derived variables (``ws10m`` from the
# wind components and ``tau300-700`` from geopotential) that no data source provides.
# Models never fetch data, so the preparation belongs in a source: the model
# recommends `DLESyMInputSource`, which fetches the base variables from a lat/lon
# source, ARCO ERA5 by default, computes the derived variables, fills the sea surface
# temperature over land and regrids onto HEALPix. To initialize from another lat/lon
# source, wrap it: ``DLESyMInputSource(WB2ERA5(), nside=model.nside)``.
#
# - Prognostic Model: [`earth2studio.models.px.DLESyM`][earth2studio.models.px.DLESyM]
# - Datasource: [`earth2studio.data.ARCO_ERA5`][earth2studio.data.ARCO_ERA5], through
#   the model's recommended `DLESyMInputSource`
# - IO Backend: [`earth2studio.io.KVBackend`][earth2studio.io.KVBackend], in memory

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function
import numpy as np
import torch

from earth2studio.io import KVBackend
from earth2studio.models.px import DLESyM, DLESyMAtmos, DLESyMOcean
from earth2studio.run import Pipeline, work_items

device = "cuda"
if not torch.cuda.is_available():
    raise RuntimeError("GPU/CUDA required for DLESyM")

package = DLESyM.load_default_package()
model = DLESyM.load_model(package).to(device)

in_coords = model.input_coords()
print("Input variables:", list(in_coords["variable"].values))
print("Input lead times [h]:", in_coords["lead_time"].values.astype("timedelta64[h]"))
out_coords = model.output_coords(in_coords)
print("Output variables:", list(out_coords["variable"].values))
print("Output lead times [h]:", out_coords["lead_time"].values.astype("timedelta64[h]"))
print("Recommended source:", type(model.default_sources()).__name__)

# %%
# The Fused Model Under Pipeline
# ------------------------------
# A `Pipeline` drives a runner over work items. `Pipeline.from_model` wraps the
# prognostic model in a `PrognosticRunner` that fetches the initial history from the
# model's recommended source, rolls the model forward and writes every step into the
# store. One DLESyM step is a 96-hour cycle, so the final lead time is a multiple of
# 96 hours.
#
# Only the fields named in ``output_variables`` are written, on the native HEALPix
# grid. The ocean's ``sst`` is valid at 48-hour lead times only: the fused model fills
# the other lead times with NaNs, because its one output slot carries the atmosphere's
# lead times.

# %%
ic_date = np.datetime64("2021-06-15")
n_cycles = 8  # 8 cycles of 96 hours: 32 days
items = work_items([ic_date], final_lead_time=np.timedelta64(96 * n_cycles, "h"))

fused = KVBackend()
Pipeline.from_model(
    model,
    output={"forecast": fused},
    output_variables={"forecast": ["ws10m", "sst"]},
    device=device,
).run(items)

fused_ds = fused.to_xarray()
print(fused_ds)
valid_sst = fused_ds["sst"].notnull().any(["face", "height", "width"]).squeeze("time")
print(
    "Lead times with valid sst [h]:",
    fused_ds["lead_time"].values[valid_sst.values].astype("timedelta64[h]"),
)

# %%
# The Atmosphere and Ocean as Components
# --------------------------------------
# A fused model whose outputs advance at different cadences cannot be coupled as it
# is. `DLESyMAtmos` and `DLESyMOcean` are the atmosphere and ocean as two ordinary
# prognostic models on the native HEALPix grid. Each loads only its own network from
# the same package, like any other model, and declares its own slots:
#
# - The atmosphere takes its 6-hourly history and, as forcing, the sea surface
#   temperature at the history's final lead time. In the fused model this is the
#   SST at the start of each cycle, the ocean's previous output.
# - The ocean takes its history at -48 h and 0 h and, as forcing, the means of
#   ``z1000`` and ``ws10m`` over the 48 hours of atmosphere frames leading up to each
#   of its output times, declared as qualified labels such as ``z1000:mean:48h``.
#   In the fused model these are averages of the atmosphere's frames from the
#   same cycle.
#
# The coupler reads these declarations; nothing in it knows about DLESyM.

# %%
from earth2studio.coupling import Connector, CoupledRunner

# The fused model is no longer needed: free its copy of the two networks.
del model
torch.cuda.empty_cache()

atmos = DLESyMAtmos.load_model(package).to(device)
ocean = DLESyMOcean.load_model(package).to(device)


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
# source. The components recommend the same `DLESyMInputSource` over ARCO ERA5 as the
# fused model, so no ``sources=`` is needed; pass
# ``sources={"atmos": source, "ocean": source}`` to initialize from another one.
# Construction validates the declarations and dry-runs the schedule, and `describe`
# prints it: each row is a call, when it is due, what it publishes and where each
# connected slot reads from.

# %%
runner = CoupledRunner(
    {"atmos": atmos, "ocean": ocean},
    [
        Connector("ocean", "atmos", time="hold"),
        Connector("atmos", "ocean"),
    ],
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
# Every stream is on the native HEALPix grid. For plotting we regrid onto a 1 degree
# lat/lon grid with a standalone
# [`earth2studio.grids.NearestRegridder`][earth2studio.grids.NearestRegridder], built
# once from the model's grid and reused for every field. We plot the 10 m wind speed
# from the atmosphere stream and the sea surface temperature from the ocean stream at
# the final lead time, next to the fused model's wind speed at the same lead time.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from earth2studio.grids import LatLonGrid, NearestRegridder

lat = np.linspace(90, -90, 181)
lon = np.linspace(0, 360, 360, endpoint=False)
to_latlon = NearestRegridder(
    atmos.input_coords().attrs["earth2studio_grid_id"], LatLonGrid(lat, lon)
)

lead_time = atmos_ds["lead_time"].values[-1]
ws10m = to_latlon(atmos_ds["ws10m"].sel(time=ic_date, lead_time=lead_time)).values
sst = to_latlon(ocean_ds["sst"].sel(time=ic_date, lead_time=lead_time)).values
ws10m_fused = to_latlon(fused_ds["ws10m"].sel(time=ic_date, lead_time=lead_time)).values
label = f"Initialization: {ic_date} - Lead time: {lead_time.astype('timedelta64[h]')}"

plt.close("all")
projection = ccrs.Robinson()
fig, axs = plt.subplots(1, 3, subplot_kw={"projection": projection}, figsize=(21, 6))

for ax, field, title, cmap, units in (
    (axs[0], ws10m_fused, "Fused ws10m", "cividis", "m/s"),
    (axs[1], ws10m, "Coupled ws10m", "cividis", "m/s"),
    (axs[2], sst, "Coupled sst", "Spectral_r", "K"),
):
    im = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(), cmap=cmap)
    ax.set_title(f"{title}\n{label}")
    if title.endswith("sst"):
        ax.add_feature(cfeature.LAND, color="grey", zorder=100)
    ax.coastlines()
    ax.gridlines()
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label(f"{title.split()[-1]} [{units}]")

plt.tight_layout()
plt.savefig("outputs/02_dlesym_coupled_ws10m_sst.jpg")
