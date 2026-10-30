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
Weather-to-Impact Chains with Windowed Reductions
=================================================

Feeding an impact model with accumulated weather quantities.

Impact models (flood, crop, energy, fire) rarely want instantaneous fields —
they want precipitation *sums*, temperature *maxima*, degree days. Derived
fields are declared through CF-style cell-method entries in the field
dictionary, and nvcoupler offers two mechanisms to produce them: a **windowed
connector** (``window=``, ``reduce=``) when one source feeds one destination,
and an **AccumulationMediator** for terminal or multi-source reductions. Any
Python function can be the impact model via CallableComponent (it does not
need to be ML).

In this example you will learn:

- How to register derived fields with a CellMethod (no suffix parsing)
- How a windowed connector reduces a fast field for a slow consumer
- When an AccumulationMediator is still the right tool
- How the run sequence is derived from the coupling graph
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# A Toy Weather Component
# -----------------------
# Exports 6 h precipitation (1.0 kg m-2 every step) and 2 m temperature
# (steps upward 1 K per step from 280 K) on a 6 h cadence.

from collections import OrderedDict

import numpy as np
import torch

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import grid_coords

GRID = (16, 32)


def weather_step(x, coords):
    tp06, t2m = x[0], x[1]
    return torch.stack([tp06, t2m + 1.0]), coords


weather = nvc.CallableComponent(
    "weather",
    weather_step,
    timestep="6h",
    exports=["total_precipitation_6h", "air_temperature_2m"],
)

# %%
# Derived Fields via CellMethod
# -----------------------------
# ``total_precipitation_48h_sum`` and ``air_temperature_2m_24h_max`` ship in
# the default dictionary. Windowed connectors and mediators alike read the
# base field, reduction, and window off the entry — nothing is inferred from
# name strings.

from earth2studio.nvcoupler.dictionary import DEFAULT_DICTIONARY

entry = DEFAULT_DICTIONARY.resolve("total_precipitation_48h_sum")
print(f"{entry.standard_name}: {entry.cell_method}")

# %%
# A (Non-ML) Impact Model
# -----------------------
# A trivial flood index: 0.1 x the 48 h precipitation sum. The index field is
# registered in a per-component dictionary copy; the imported sum arrives as
# a state variable (the default VariableOverwriteAdapter pattern).

d = nvc.FieldDictionary(DEFAULT_DICTIONARY)
d.register(nvc.FieldEntry("flood_risk_index", "", "toy flood index"))


def flood_step(x, coords):
    _index, p48 = x[0], x[1]
    return torch.stack([0.1 * p48, p48]), coords


flood = nvc.CallableComponent(
    "flood",
    flood_step,
    timestep="48h",
    imports=["total_precipitation_48h_sum"],
    exports=["flood_risk_index"],
    variable_aliases={
        "findex": "flood_risk_index",
        "p48": "total_precipitation_48h_sum",
    },
    dictionary=d,
)

# %%
# Wire the Chain
# --------------
# The precip sum has one source and one destination, so it is a **windowed
# connector** — no mediator, no extra component. The 24 h t2m max is a
# *terminal* product (nothing imports it), so it needs an
# **AccumulationMediator**: mediators are components with export states that
# land in the collected output. The run sequence is derived from the graph —
# weather feeds both reductions every 6 h, the max reduces every 24 h, the
# flood model runs every 48 h on the freshly delivered sum.

t2m_max = nvc.AccumulationMediator("tmax", ["air_temperature_2m_24h_max"])

driver = nvc.Driver(
    {"weather": weather, "tmax": t2m_max, "flood": flood},
    clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=[
        nvc.Connector(weather, flood, window="48h", reduce="sum"),
        ("weather", "tmax"),
    ],
)
print(driver.describe())

# %%
# Run It
# ------

ic_weather = (
    torch.stack([torch.full(GRID, 1.0), torch.full(GRID, 280.0)]),
    OrderedDict({"variable": np.array(["tp06", "t2m"]), **grid_coords(*GRID)}),
)
ic_flood = (
    torch.zeros(2, *GRID),
    OrderedDict({"variable": np.array(["findex", "p48"]), **grid_coords(*GRID)}),
)

# Note: initialize logs warnings that tmax's and flood's exports have no
# consumer — correct here, they are the chain's terminal outputs.
driver.initialize({"weather": ic_weather, "flood": ic_flood})
ds = driver.run()

# %%
# Check the Numbers
# -----------------
# 8 samples of 1.0 kg m-2 per 48 h window -> sum 8.0 -> flood index 0.8.
# t2m rises 1 K/step; the max over each 24 h window is the last sample.

p48 = driver.probe("weather->flood")["total_precipitation_48h_sum"]
tmax_series = ds["tmax"]["air_temperature_2m_24h_max"].mean(("lat", "lon")).values
flood_series = ds["flood"]["flood_risk_index"].mean(("lat", "lon")).values

print(f"last 48h precip sum:  {float(p48.data.mean())}")
print(f"24h t2m maxima:       {tmax_series}")
print(f"flood risk index:     {flood_series}")
if not np.isclose(float(p48.data.mean()), 8.0) or not np.allclose(
    flood_series[1:], 0.8
):
    raise ValueError("impact chain did not reproduce the analytic values")
print("\nimpact chain produced the analytic values ✓")
