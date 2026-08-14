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
Running a Coupled Atmosphere-Ocean System
=========================================

The core nvcoupler workflow on synthetic components.

This example builds the smallest complete coupled system: a fast "atmosphere"
(6 h step, 32x64 grid) and a slow "ocean" (48 h step, 16x32 grid) exchanging
fields through a trailing-average mediator, exactly the cadence structure of
DLESyM. Every number below is hand-computable.

In this example you will learn:

- How to declare components with imports/exports by standard name
- How the run-sequence DSL schedules runs and exchanges
- How the Driver validates and executes the coupled loop
- How to inspect exchanges (probe) and collect results as xarray
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# The toy components live in ``earth2studio.nvcoupler.testing``. The
# atmosphere steps ``z1000 += 1 + 0.1 * sst`` and imports SST; the ocean
# steps ``sst += 0.01 * z48m`` and imports the trailing 48 h mean of z1000.
# Grids differ, so the connector regrids automatically.

import os

os.makedirs("outputs", exist_ok=True)

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

atmos = fake_atmos()  # 6h step,  exports geopotential_at_1000hpa, imports SST
ocean = fake_ocean()  # 48h step, exports sea_surface_temperature
med = nvc.TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])

print(atmos)
print(ocean)
print(med)

# %%
# The Run Sequence
# ----------------
# NUOPC-style: one slot per cadence, actions in order. Placing
# ``ocean -> atmos`` *before* ``atmos`` makes the coupling lagged — the
# atmosphere always sees the ocean's most recent completed state.

SEQUENCE = """
@6h
  atmos -> med          # accumulate z1000 into the mediator
  ocean -> atmos        # lagged SST forcing
  atmos
@48h
  med.compute           # reduce the 48h window
  med -> ocean          # hand the mean to the ocean
  ocean
@
"""

# %%
# Execute the Coupled Loop
# ------------------------
# The Driver validates everything at initialize (names, cadences, field
# matching, units) and then runs 96 hours: 16 atmosphere steps, 2 ocean
# steps, 2 mediator reductions.

driver = nvc.Driver(
    {"atmos": atmos, "ocean": ocean, "med": med},
    sequence=SEQUENCE,
    clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
)
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
datasets = driver.run()  # dict[str, xr.Dataset] (in-memory collection)

print(f"atmos ran {atmos.run_count}x, ocean {ocean.run_count}x, med {med.run_count}x")

# %%
# Inspect the Results
# -------------------
# Expected values: z grows 1.2/step under SST=2, so z(48h)=9.6; the first
# 48h mean is 4.2, giving sst=2.042; z then grows 1.2042/step to
# z(96h)=19.2336, and sst(96h)=2.180147.

z = datasets["atmos"]["geopotential_at_1000hpa"]
sst = datasets["ocean"]["sea_surface_temperature"]
zmean = datasets["med"]["geopotential_at_1000hpa_48h_mean"]

print("\ntime series (area means):")
print(f"{'time':>20} {'z1000':>10} {'sst':>10}")
for t in z.time.values:
    z_t = float(z.sel(time=t).mean())
    row = f"{str(t)[:16]:>20} {z_t:>10.4f}"
    if t in sst.time.values:
        row += f" {float(sst.sel(time=t).mean()):>10.6f}"
    print(row)
print(f"\n48h means from the mediator: {zmean.mean(('lat', 'lon')).values}")

# %%
# Plot the Coupled Time Series
# ----------------------------
# The same area means, visualized: z1000 grows a little faster after each
# 48 h coupling event because the ocean warms in response to the mean z1000
# it received — the coupling feedback is visible as the kinks at 48 h/96 h.

import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax0.plot(z.time, z.mean(("lat", "lon")), "o-", color="tab:blue")
ax0.set_ylabel("z1000 area mean")
ax0.set_title("Coupled toy system: 6 h atmosphere, 48 h ocean")

ax1.plot(sst.time, sst.mean(("lat", "lon")), "s-", color="tab:red")
ax1.set_ylabel("sst area mean")
ax1.set_xlabel("time")

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("outputs/01_coupled_toy_timeseries.jpg")

# %%
# Probe an Exchange
# -----------------
# Every connector remembers the last fields it moved — useful when a coupled
# run misbehaves and you need to see what actually crossed the interface.

transfer = driver.probe("ocean->atmos")
f = transfer["sea_surface_temperature"]
print(f"last ocean->atmos transfer: {f}")
print(f"regridded to the atmos grid: {tuple(f.data.shape)}")
