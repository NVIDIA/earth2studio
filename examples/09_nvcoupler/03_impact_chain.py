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
Impact Chains with Diagnostics
==============================

Feed diagnostics from a coupled model, including over time windows.

Diagnostic models join a coupled system as they are: their ``input_coords()``
are imports, their ``output_coords()`` exports, and they run whenever a step
delivers their inputs. A diagnostic that declares a qualified label such as
``z1000:max:24h`` receives the trailing maximum of its source's frames.

In this example you will learn:

- coupling a :class:`~earth2studio.models.dx.DiagnosticModel` to a prognostic
- windowed inputs declared by the diagnostic itself
- unit-like conversions with a connector ``transform``
- why a windowed diagnostic skips the first steps
"""

# /// script
# dependencies = [
#   "earth2studio[utils] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np
import xarray as xr

from earth2studio.nvcoupler import Connector, Coupler
from earth2studio.nvcoupler.testing import (
    ToyAtmosphere,
    ToyImpact,
    toy_grid,
    toy_source,
)
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array, coord_array_like

GRID = toy_grid(32, 64)


class DailyMax:
    """Diagnostic over a time window: the trailing 24 h maximum of ``z1000``."""

    stochastic = False

    def input_coords(self) -> xr.DataArray:
        """Declare the windowed input through its qualified label."""
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": np.array(["z1000:max:24h"])},
            dynamic=("batch",),
            grid=GRID,
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Rename the single variable."""
        return coord_array_like(input_coords, {"variable": np.array(["z1000_dmax"])})

    def default_sources(self) -> None:
        """Recommend no source; the input is coupled."""
        return None

    def to(self, device: object) -> "DailyMax":
        """Return the model; it holds no device state."""
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Pass the delivered maximum through under the output label."""
        return x.assign_coords(variable=np.array(["z1000_dmax"]))


# %%
# Declare the chain
# -----------------
# The atmosphere's SST forcing comes from a data source here (a constant 2 K),
# so only one prognostic is involved. Two diagnostics hang off it: ``ToyImpact``
# turns each ``z1000`` frame into ``ws10m``, and ``DailyMax`` takes the trailing
# 24 h maximum. A ``transform`` on the first connector converts geopotential to
# geopotential height before delivery.

# %%
coupler = Coupler(
    {"atmos": ToyAtmosphere(), "wind": ToyImpact(), "daily": DailyMax()},
    connectors=[
        Connector("atmos", "wind", transform=lambda x: x / 9.81),
        Connector("atmos", "daily"),
    ],
    sources={"atmos": (toy_source(GRID, 0.0), toy_source(GRID, 2.0))},
)
print(coupler.describe())

# %%
# Both diagnostics inherit the atmosphere's 6 h cadence and run after it in the
# same step. Run two days:

# %%
item = WorkItem(np.datetime64("2026-01-01T00"), np.timedelta64(48, "h"))
for step in coupler.run_item(item):
    lead = int(
        step["atmos"]["lead_time"].values[0].astype("timedelta64[h]").astype(int)
    )
    values = {
        name: float(out.isel(lat=1, lon=1).values.ravel()[0])
        for name, out in step.items()
    }
    print(f"{lead:3d}h", {k: round(v, 4) for k, v in values.items()})

# %%
# ``wind`` is ``0.01 * z1000 / 9.81`` at every step, initial condition included.
# ``daily`` first appears at 24 h: its window ``[T - 24h, T)`` needs four frames,
# so earlier steps are skipped rather than computed from a partial window. From
# then on it is the maximum of the previous four 6-hourly frames, i.e. the frame
# at ``T - 6h`` for this monotonically increasing toy.
