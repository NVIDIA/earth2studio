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
HiRO-ACEv2
==========

Couple ACE2S-X-SHiELD to the HiROv2 downscaler.

ACE2S-X-SHiELD steps a global 1-degree atmosphere every 6 hours. HiROv2 reads
the coarse precipitation rate, 10 m winds and sea-level pressure and draws a
3 km ensemble over one region. ``CoupledRunner`` keeps the two models separate:
a connector passes the coarse fields, and each model writes its own store.

The checkpoints, initial condition and forcing all come from
``allenai/HiRO-ACEv2`` (several GB on first use). Install the ``ace2`` extra.
The first ACE2S state used here is 1 January 2014, one of the published
initial-condition years.
"""

# /// script
# dependencies = [
#   "earth2studio[ace2] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import os

os.makedirs("outputs", exist_ok=True)

import numpy as np
import torch

from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.grids import NearestRegridder, infer_grid
from earth2studio.io import XarrayBackend
from earth2studio.models.dx import HiROv2
from earth2studio.models.px import ACE2SXSHiELD
from earth2studio.run import Pipeline, work_items

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load both checkpoints from the same package
# -------------------------------------------
# The region is a slice of the ACE Gaussian grid over the Pacific Northwest.
# It has to cover at least one HiROv2 training patch. Overlap of one coarse
# cell matches the published HiRO-ACEv2 config.

package = ACE2SXSHiELD.load_default_package()
ace = ACE2SXSHiELD.load_model(package).to(device)
hiro = HiROv2.load_model(
    package,
    lat_extent=(34.0, 50.0),
    lon_extent=(233.0, 250.0),
    number_of_samples=1,
    overlap=1,
).to(device)
ace.set_rng(0)
hiro.set_rng(0)
print(list(hiro.input_coords().coords["variable"].values))
print(hiro.input_coords().sizes)

# %%
# Feed ACE2S into HiROv2
# ----------------------
# The connector names whatever coarse variables the loaded HiROv2 checkpoint
# declares, so a later checkpoint with a different variable list still lines
# up. ACE2S keeps its own forcing source. HiROv2 has no source of its own.

variables = [str(name) for name in hiro.input_coords().coords["variable"].values]
regridder = NearestRegridder("gaussian-f90", infer_grid(hiro.input_coords()))
runner = CoupledRunner(
    {"ace2s": ace, "hiro": hiro},
    [
        Connector(
            "ace2s",
            "hiro",
            variables=variables,
            regridder=regridder,
        )
    ],
)
print(runner.describe(np.timedelta64(6, "h")))

# %%
# One 6-hour step
# ---------------

stores = {"ace2s": XarrayBackend(), "hiro": XarrayBackend()}
items = work_items([np.datetime64("2014-01-01T00:00")], np.timedelta64(6, "h"))
Pipeline(runner, output=stores, device=device).run(items)
print(dict(stores["hiro"].root["tp"].sizes))
