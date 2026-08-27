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
fields through two connectors — one of them a windowed (trailing 48 h mean)
reduction — exactly the cadence structure of DLESyM. The system is declared
as components plus connections; the run sequence is derived from the coupling
graph. Every number below is hand-computable.

In this example you will learn:

- How to declare components with imports/exports by standard name
- How to declare the coupling graph and let the Driver derive the schedule
- How a windowed connector (window=, reduce=) bridges a cadence gap
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
# Grids differ, so the connectors regrid automatically.

import os

os.makedirs("outputs", exist_ok=True)

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

atmos = fake_atmos()  # 6h step,  exports geopotential_at_1000hpa, imports SST
ocean = fake_ocean()  # 48h step, exports sea_surface_temperature

print(atmos)
print(ocean)

# %%
# Declare the Coupling Graph
# --------------------------
# Two edges. The SST hand-off is a plain connector — a bare ``(src, dst)``
# tuple builds the default. The z1000 hand-off is a *windowed* connector:
# ``window="48h", reduce="mean"`` folds the atmosphere's export into a
# running mean every step and delivers it as the derived field
# ``geopotential_at_1000hpa_48h_mean`` (declared by a CellMethod entry in the
# ocean's dictionary) on each 48 h boundary. No mediator needed for a
# single-source reduction.

connectors = [
    ("ocean", "atmos"),  # lagged SST forcing
    nvc.Connector(atmos, ocean, window="48h", reduce="mean"),
]

# %%
# Build the Driver — No Run Sequence Required
# -------------------------------------------
# With no ``sequence=`` the Driver derives the canonical (lagged) schedule
# from the coupling graph: one slot per cadence, connects before runs.
# ``describe()`` shows the whole plan — components, connectors, and the
# derived sequence — before anything runs. (An explicit run-sequence DSL
# remains the escape hatch when the *ordering* is the experiment; see
# example 02.)

driver = nvc.Driver(
    {"atmos": atmos, "ocean": ocean},
    clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=connectors,
)
print(driver.describe())

# %%
# Execute the Coupled Loop
# ------------------------
# The Driver validates everything at initialize (names, cadences, field
# matching, units) and then runs 96 hours: 16 atmosphere steps, 2 ocean
# steps, 2 window deliveries. We iterate with ``steps()`` to also capture
# each 48 h mean as the windowed connector delivers it.

driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})

z48_means, seen = [], set()
for time, states in driver.steps():
    f = driver.probe("atmos->ocean").get("geopotential_at_1000hpa_48h_mean")
    if f is not None and f.valid_time not in seen:
        seen.add(f.valid_time)
        z48_means.append(float(f.data.mean()))
datasets = driver.to_xarray()  # dict[str, xr.Dataset] (in-memory collection)

print(f"atmos ran {atmos.run_count}x, ocean {ocean.run_count}x")

# %%
# Inspect the Results
# -------------------
# Expected values: z grows 1.2/step under SST=2, so z(48h)=9.6; the first
# 48h mean is 4.2, giving sst=2.042; z then grows 1.2042/step to
# z(96h)=19.2336, and sst(96h)=2.180147.

import numpy as np

z = datasets["atmos"]["geopotential_at_1000hpa"]
sst = datasets["ocean"]["sea_surface_temperature"]

print("\ntime series (area means):")
print(f"{'time':>20} {'z1000':>10} {'sst':>10}")
for t in z.time.values:
    z_t = float(z.sel(time=t).mean())
    row = f"{str(t)[:16]:>20} {z_t:>10.4f}"
    if t in sst.time.values:
        row += f" {float(sst.sel(time=t).mean()):>10.6f}"
    print(row)
print(f"\n48h means delivered by the windowed connector: {z48_means}")

if not np.isclose(float(z.values[-1].mean()), 19.2336, atol=1e-4):
    raise ValueError("z1000(96h) does not match the hand-computed value")
if not np.isclose(float(sst.values[-1].mean()), 2.180147, atol=1e-6):
    raise ValueError("sst(96h) does not match the hand-computed value")
if not np.allclose(z48_means, [4.2, 13.8147], atol=1e-4):
    raise ValueError("48h means do not match the hand-computed values")

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
# The windowed connector's probe carries the *derived* standard name.

f = driver.probe("ocean->atmos")["sea_surface_temperature"]
print(f"last ocean->atmos transfer: {f}")
print(f"regridded to the atmos grid: {tuple(f.data.shape)}")
z48 = driver.probe("atmos->ocean")["geopotential_at_1000hpa_48h_mean"]
print(f"last atmos->ocean transfer: {z48}")

# %%
# One-Call Auto-Wiring
# --------------------
# ``couple()`` goes one step further: it discovers both edges (including the
# windowed one, from the ocean's derived import) by matching standard names,
# so the whole system above is:

driver2 = nvc.couple(fake_atmos(), fake_ocean(), start="2024-01-01", stop="2024-01-05")
driver2.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
z96 = float(driver2.run()["atmos"]["geopotential_at_1000hpa"].values[-1].mean())
print(f"couple() reproduces z1000(96h) = {z96:.4f}")
