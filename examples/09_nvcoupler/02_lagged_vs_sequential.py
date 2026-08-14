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
Coupling Order as a One-Line Experiment
=======================================

Lagged vs sequential coupling by reordering run-sequence actions.

In physical coupled modeling, whether the atmosphere sees the ocean state
from *this* coupling step (sequential/implicit-ish) or the *previous* one
(lagged/explicit) is an architectural decision baked deep into the coupler.
In nvcoupler it is one line of the run sequence: a ConnectAction placed
before the destination's RunAction in a slot is lagged; after, sequential.

In this example you will learn:

- How action order inside a slot defines coupling semantics
- How to run the same system twice under both orderings
- Why the two differ by exactly one coupling window (hand-computable)
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Two Sequences, One Line Apart
# -----------------------------
# The SST hand-off to the atmosphere is confined to the 48 h slot so the
# orderings are cleanly comparable. Only the position of ``ocean -> atmos``
# changes.

import numpy as np

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

LAGGED = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  ocean -> atmos        # BEFORE the ocean runs: atmos gets the OLD sst
  med -> ocean
  ocean
@
"""

SEQUENTIAL = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  med -> ocean
  ocean
  ocean -> atmos        # AFTER the ocean runs: atmos gets the FRESH sst
@
"""


def run(sequence: str) -> float:
    driver = nvc.Driver(
        {
            "atmos": fake_atmos(),
            "ocean": fake_ocean(),
            "med": nvc.TrailingAverageMediator(
                "med", ["geopotential_at_1000hpa_48h_mean"]
            ),
        },
        sequence=sequence,
        clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
    )
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    ds = driver.run()
    return float(ds["atmos"]["geopotential_at_1000hpa"].values[-1].mean())


# %%
# Compare
# -------
# Hand computation: at 48 h the ocean updates SST from 2.0 to 2.042. Under
# the lagged ordering the atmosphere is forced by SST=2.0 for the next 8
# steps; under the sequential ordering by 2.042. The difference in final
# z1000 is exactly 8 steps x 0.1 x 0.042 = 0.0336.

z_lagged = run(LAGGED)
z_sequential = run(SEQUENTIAL)

print(f"z1000(96h) lagged:     {z_lagged:.4f}")
print(f"z1000(96h) sequential: {z_sequential:.4f}")
print(f"difference:            {z_sequential - z_lagged:.4f}")
print(f"expected 8*0.1*0.042 = {8 * 0.1 * 0.042:.4f}")
if not np.isclose(z_sequential - z_lagged, 8 * 0.1 * 0.042, atol=1e-5):
    raise ValueError("coupling-order difference does not match the analytic value")
print("\ncoupling-order experiment reproduced the analytic difference ✓")
