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

"""Coupled runs under Pipeline
===============================

The ``Coupler`` is a ``Runner``, so ``Pipeline`` supervises coupled runs exactly
like single-model ones: it plans one store per stream from the coupler's
schemas, distributes work items, writes each step and can resume. The coupler
only decides what each model receives and when. Reuses the toy atmosphere and
ocean of ``10_coupling.py``, defined inline.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator

import numpy as np
import xarray as xr

from earth2studio.coupling import Connector, Coupler
from earth2studio.data import Constant
from earth2studio.grids import LatLonGrid
from earth2studio.io import KVBackend
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import Pipeline, WorkItem, work_items
from earth2studio.utils import coord_array, coord_array_like

T0 = np.datetime64("2026-01-01T00")
ATMOS = LatLonGrid(np.linspace(90, -90, 32), np.linspace(0, 360, 64, endpoint=False))
OCEAN = LatLonGrid(np.linspace(90, -90, 16), np.linspace(0, 360, 32, endpoint=False))
Rule = Callable[[xr.DataArray, xr.DataArray], xr.DataArray]


def constant(grid: LatLonGrid, value: float) -> Constant:
    """Constant source on ``grid``."""
    coords = grid.coords(only_index=True)
    domain = OrderedDict(lat=np.asarray(coords["lat"]), lon=np.asarray(coords["lon"]))
    return Constant(domain, value)  # type: ignore[arg-type]


class Toy(PrognosticMixin):
    """One state variable, one forcing variable, ``rule(x, newest forcing)``."""

    def __init__(
        self,
        variable: str,
        forcing: str,
        forcing_lead: int,
        step: int,
        grid: LatLonGrid,
        rule: Rule,
    ) -> None:
        self.variable, self.forcing, self.grid, self.rule = (
            variable,
            forcing,
            grid,
            rule,
        )
        self.forcing_lead = np.timedelta64(forcing_lead, "h")
        self.step_size = np.timedelta64(step, "h")

    def input_coords(self) -> xr.DataArray:
        """Declare the input slot."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": np.array([self.variable]),
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def forcing_coords(self) -> xr.DataArray:
        """Declare the forcing slot."""
        return coord_array_like(
            self.input_coords(),
            {
                "lead_time": np.array([self.forcing_lead]),
                "variable": np.array([self.forcing]),
            },
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Advance the input signature by one step."""
        lead = input_coords["lead_time"].values + self.step_size
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        f = forcing.isel(lead_time=-1, drop=True).sel(variable=self.forcing, drop=True)
        y = self.rule(x, f).assign_coords(lead_time=x["lead_time"] + self.step_size)
        return y.transpose(*x.dims).assign_attrs(x.attrs)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, int]:
        """Return the first forecast and a step counter."""
        return self._advance(x, forcing), 0

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray, state: int = 0
    ) -> tuple[xr.DataArray, int]:
        """Advance one step from the previous forecast."""
        return self._advance(y, forcing), state + 1

    def __call__(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> xr.DataArray:
        """Apply the model."""
        return self._default_call(x, forcing)  # type: ignore[return-value]

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        """Iterate forecasts, taking forcing through ``send``."""
        return self._default_create_iterator(x, forcing)  # type: ignore[return-value]

    def to(self, device: object) -> "Toy":
        """Return the model; it holds no device state."""
        return self


coupler = Coupler(
    {
        "atmos": Toy("z1000", "sst", 0, 6, ATMOS, lambda x, sst: x + 1 + 0.1 * sst),
        "ocean": Toy("sst", "z1000:mean:48h", 48, 48, OCEAN, lambda x, z: x + 0.01 * z),
    },
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={
        "atmos": (constant(ATMOS, 0.0), None),
        "ocean": (constant(OCEAN, 2.0), None),
    },
)

# %%
# Schemas and requests are known before running: one allocation-free signature
# per stream covering every lead time of an item, and every fetch an item makes.

final_lead_time = np.timedelta64(96, "h")
for stream, schema in coupler.output_coords(final_lead_time).items():
    print(stream, schema.dims, schema["lead_time"].values.astype("timedelta64[h]"))
for request in coupler.data_requests(WorkItem(T0, final_lead_time)) or ():
    print(request.variable, request.lead_time.astype("timedelta64[h]"))

# %%
# Streams are keyed by model name, so the output mapping names the models. Two
# initial times become two work items written into the same stores.

atmos_store, ocean_store = KVBackend(), KVBackend()
items = work_items([T0, T0 + np.timedelta64(1, "D")], final_lead_time)
Pipeline(
    coupler, output={"atmos": atmos_store, "ocean": ocean_store}, device="cpu"
).run(items)
print(
    "z1000", tuple(atmos_store["z1000"].shape), "sst", tuple(ocean_store["sst"].shape)
)
np.testing.assert_allclose(ocean_store["sst"][:, 1, 1, 1], [2.042, 2.042])

# %%
# Callers already holding arrays pass them as ``ics``; the coupler then skips
# fetching those input slots. Forcing is still served by connectors and sources.

ic = xr.DataArray(
    np.full((1, 1, 32, 64), 5.0, dtype=np.float32),
    dims=("lead_time", "variable", "lat", "lon"),
    coords={
        "lead_time": np.array([0], dtype="timedelta64[h]"),
        "variable": ["z1000"],
        **ATMOS.coords(only_index=True),
    },
)
steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(6, "h")), ics={"atmos": ic}))
np.testing.assert_allclose(
    float(steps[1]["atmos"].isel(lat=1, lon=1).values.ravel()[0]), 6.2
)
print("z1000 at 6h from an in-memory IC of 5:", 6.2)
