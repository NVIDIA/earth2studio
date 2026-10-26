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
Vertical Coupling for Chemistry Models
======================================

Hybrid sigma-pressure to pressure-level interpolation in a connector.

Chemistry emulators typically live on hybrid model levels
(``p_k = a_k + b_k * p_s``) while meteorology models export pressure-level
fields, or the other way round. A connector ``transform`` built with
:func:`~earth2studio.nvcoupler.vertical.to_pressure_levels` interpolates a
field with a ``level`` dimension linearly in log pressure before delivery.
Most Earth2Studio models encode levels in variable names (``z500``, ``t850``)
and never touch this.

In this example you will learn:

- declaring a ``level`` dimension in a model signature
- interpolating between vertical coordinates with a connector ``transform``
- the failure mode when a hybrid source lacks surface pressure
"""

# /// script
# dependencies = [
#   "earth2studio[utils] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np
import xarray as xr

from earth2studio.nvcoupler import Connector, Coupler, CouplingError
from earth2studio.nvcoupler.testing import ToyPrognostic, toy_grid
from earth2studio.nvcoupler.vertical import HybridLevels, to_pressure_levels
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array, coord_array_like

GRID = toy_grid(8, 16)
PS = 100000.0  # Pa, taken constant here


class Layered(ToyPrognostic):
    """Toy prognostic carrying ozone on a ``level`` dimension."""

    def __init__(self, levels, **kwargs):
        super().__init__(["o3"], **kwargs)
        self.levels = np.asarray(levels, dtype=np.float64)

    def input_coords(self):
        """Declare ``o3`` on this model's levels."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "level", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": np.array(["o3"]),
                "level": self.levels,
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def forcing_coords(self):
        """Declare the forcing slot on the same levels."""
        if not self.forcing_variables:
            return None
        return coord_array_like(
            self.input_coords(),
            {
                "lead_time": np.array([self.forcing_lead]),
                "variable": np.array(self.forcing_variables),
            },
        )


# %%
# Source on hybrid levels
# -----------------------
# The meteorology model holds ozone on three hybrid levels that realise at
# 300 / 700 / 1000 hPa for a 1000 hPa surface. Its profile is ``log(p)``, so
# interpolation results are exact and checkable. The chemistry model wants
# the same field as forcing on 500 and 850 hPa.

# %%
hybrid = HybridLevels(a=(30000.0, 20000.0, 0.0), b=(0.0, 0.5, 1.0))
received = []


def chem_rule(x, forcing):
    """Record the delivered column and persist the state."""
    received.append(
        forcing.isel(time=0, lead_time=0, lat=0, lon=0).sel(variable="o3").values
    )
    return x


met = Layered(np.arange(3.0), grid=GRID, advance=lambda x, f: x)
chem = Layered([500.0, 850.0], grid=GRID, forcing=["o3"], advance=chem_rule)

pressure = np.array(hybrid.a) + np.array(hybrid.b) * PS
ozone = xr.DataArray(
    np.broadcast_to(
        np.log(pressure)[None, None, :, None, None], (1, 1, 3, 8, 16)
    ).astype(np.float32),
    dims=("lead_time", "variable", "level", "lat", "lon"),
    coords={
        "lead_time": np.array([0], dtype="timedelta64[h]"),
        "variable": ["o3"],
        "level": np.arange(3.0),
        **GRID.coords(only_index=True),
    },
)
# The chemistry model starts from zeros on its own two levels.
chem_ic = xr.zeros_like(ozone.isel(level=[0, 1])).assign_coords(level=[500.0, 850.0])
ics = {"met": ozone, "chem": chem_ic}

# %%
# Interpolate in the connector
# ----------------------------
# ``to_pressure_levels`` returns the ``transform``; the connector applies it to
# each delivered field after regridding, and the coupler checks the result
# against the chemistry model's declared ``level`` coordinate. Both models
# start from in-memory initial conditions here, so no sources are bound.

# %%
coupler = Coupler(
    {"met": met, "chem": chem},
    connectors=[
        Connector("met", "chem", transform=to_pressure_levels(hybrid, [500, 850], PS))
    ],
)
item = WorkItem(np.datetime64("2026-01-01"), np.timedelta64(6, "h"))
list(coupler.run_item(item, ics=ics))
print("received:", received[0])
print("expected:", np.log([50000.0, 85000.0]))
np.testing.assert_allclose(received[0], np.log([50000.0, 85000.0]), rtol=1e-6)

# %%
# The failure mode
# ----------------
# A hybrid source without surface pressure is refused at the exchange, not as
# NaNs three days into a run.

# %%
broken = Coupler(
    {"met": met, "chem": chem},
    connectors=[
        Connector("met", "chem", transform=to_pressure_levels(hybrid, [500, 850]))
    ],
)
try:
    list(broken.run_item(item, ics=ics))
except CouplingError as error:
    print(error)
