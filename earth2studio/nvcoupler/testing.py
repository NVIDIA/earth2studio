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

"""Synthetic toy components for tests and demos.

A deterministic two-component system with the DLESyM cadence structure:

- fake atmosphere, 6 h step, 32x64 lat/lon grid, state [z1000, sst]:
      z1000 <- z1000 + 1.0 + gain * 0.1 * sst        (sst = imported SST)
- fake ocean, 48 h step, 16x32 lat/lon grid, state [sst, z48m]:
      sst   <- sst + gain * 0.01 * z48m              (z48m = imported 48 h
                                                       mean of atmos z1000)

With spatially-constant initial conditions every intermediate value is
hand-computable, which the end-to-end driver tests rely on. Pass
``gain=torch.tensor(1.0, requires_grad=True)`` to check that gradients flow
across the exchange.
"""

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem

from .component import CallableComponent

ATMOS_GRID = (32, 64)
OCEAN_GRID = (16, 32)


def grid_coords(nlat: int, nlon: int) -> CoordSystem:
    return OrderedDict(
        {
            "lat": np.linspace(90.0, -90.0, nlat),
            "lon": np.linspace(0.0, 360.0, nlon, endpoint=False),
        }
    )


def fake_atmos(
    gain: torch.Tensor | float = 1.0, timestep: str = "6h"
) -> CallableComponent:
    """Fast toy component: imports SST, exports z1000."""

    def step(x: torch.Tensor, coords: CoordSystem):
        z1000, sst = x[0], x[1]
        z_next = z1000 + 1.0 + gain * 0.1 * sst
        return torch.stack([z_next, sst]), coords

    return CallableComponent(
        "atmos",
        step,
        timestep=timestep,
        imports=["sea_surface_temperature"],
        exports=["geopotential_at_1000hpa"],
    )


def fake_ocean(
    gain: torch.Tensor | float = 1.0,
    timestep: str = "48h",
    with_mask: bool = False,
) -> CallableComponent:
    """Slow toy component: imports the 48 h mean of z1000, exports SST.

    With ``with_mask=True`` the exported SST carries a land mask covering the
    northern half of the grid (True = valid ocean point).
    """

    def step(x: torch.Tensor, coords: CoordSystem):
        sst, z48m = x[0], x[1]
        sst_next = sst + gain * 0.01 * z48m
        return torch.stack([sst_next, z48m]), coords

    export_masks = None
    if with_mask:
        mask = torch.ones(*OCEAN_GRID, dtype=torch.bool)
        mask[: OCEAN_GRID[0] // 2, :] = False  # northern half is land
        export_masks = {"sea_surface_temperature": mask}

    return CallableComponent(
        "ocean",
        step,
        timestep=timestep,
        imports=["geopotential_at_1000hpa_48h_mean"],
        exports=["sea_surface_temperature"],
        variable_aliases={"z48m": "geopotential_at_1000hpa_48h_mean"},
        export_masks=export_masks,
    )


def atmos_ic(z0: float = 0.0, sst0: float = 2.0) -> tuple[torch.Tensor, CoordSystem]:
    coords = OrderedDict(
        {"variable": np.array(["z1000", "sst"]), **grid_coords(*ATMOS_GRID)}
    )
    x = torch.stack([torch.full(ATMOS_GRID, z0), torch.full(ATMOS_GRID, sst0)])
    return x, coords


def ocean_ic(sst0: float = 2.0, z48m0: float = 0.0) -> tuple[torch.Tensor, CoordSystem]:
    coords = OrderedDict(
        {"variable": np.array(["sst", "z48m"]), **grid_coords(*OCEAN_GRID)}
    )
    x = torch.stack([torch.full(OCEAN_GRID, sst0), torch.full(OCEAN_GRID, z48m0)])
    return x, coords
