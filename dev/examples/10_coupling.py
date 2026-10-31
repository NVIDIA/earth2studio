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

"""Coupling: models on one clock
=================================

A ``Coupler`` advances ordinary prognostic and diagnostic models together and
serves each model's declared forcing from the others' outputs. Nothing about the
models is coupler-specific: imports come from ``forcing_coords()`` or a
diagnostic's ``input_coords()``, exports from ``output_coords()``, cadence from
the lead-time advance, grids from the signatures. See
``dev/spec/COUPLING_SPEC.md``.

Toy models with hand-computable arithmetic stand in for networks. Needs no
weights, downloads or GPU (SciPy for the regridders).
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import xarray as xr

from earth2studio.coupling import (
    Connector,
    Coupler,
    CouplingError,
    HybridLevels,
    RunSequence,
    Slot,
    to_pressure_levels,
)
from earth2studio.data import Constant
from earth2studio.grids import LatLonGrid
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array, coord_array_like

T0 = np.datetime64("2026-01-01T00")
Rule = Callable[[xr.DataArray, xr.DataArray | None], xr.DataArray]


def grid(nlat: int, nlon: int) -> LatLonGrid:
    """Regular global grid, poles included."""
    return LatLonGrid(
        np.linspace(90, -90, nlat), np.linspace(0, 360, nlon, endpoint=False)
    )


def constant(grid: LatLonGrid, value: float) -> Constant:
    """Constant source on ``grid`` for initial conditions."""
    coords = grid.coords(only_index=True)
    domain = OrderedDict(lat=np.asarray(coords["lat"]), lon=np.asarray(coords["lon"]))
    return Constant(domain, value)  # type: ignore[arg-type]


def newest(forcing: xr.DataArray | None, label: str) -> xr.DataArray:
    """The newest frame of one forcing variable, without lead_time and variable."""
    assert forcing is not None  # noqa: S101
    return forcing.isel(lead_time=-1, drop=True).sel(variable=label, drop=True)


def point(step: dict, name: str) -> float:
    """One grid point of a published field."""
    return float(step[name].isel(lat=1, lon=1).values.ravel()[0])


class Toy(PrognosticMixin):
    """One-slot prognostic with an optional forcing slot and a rule on the values."""

    def __init__(
        self,
        variables: Sequence[str],
        forcing: Sequence[str] = (),
        forcing_lead: str = "0h",
        step: str = "6h",
        grid: LatLonGrid = grid(32, 64),
        rule: Rule | None = None,
        levels: Sequence[float] | None = None,
    ) -> None:
        self.variables, self.forcing_variables = list(variables), list(forcing)
        self.forcing_lead = np.timedelta64(int(forcing_lead[:-1]), "h")
        self.step_size = np.timedelta64(int(step[:-1]), "h")
        self.grid, self.levels = grid, levels
        self.rule: Rule = rule or (lambda x, f: x + 1)
        self.calls: list[list[int]] = []

    def input_coords(self) -> xr.DataArray:
        """Declare the input slot, with a level dimension when levels are given."""
        level = ("level",) if self.levels is not None else ()
        dims = ("batch", "time", "lead_time", "variable", *level, "lat", "lon")
        coords: dict[str, np.ndarray] = {
            "lead_time": np.array([0], dtype="timedelta64[h]"),
            "variable": np.array(self.variables),
        }
        if self.levels is not None:
            coords["level"] = np.asarray(self.levels, dtype=np.float64)
        return coord_array(dims, coords, dynamic=("batch", "time"), grid=self.grid)

    def forcing_coords(self) -> xr.DataArray | None:
        """Declare the forcing slot, or none."""
        if not self.forcing_variables:
            return None
        return coord_array_like(
            self.input_coords(),
            {
                "lead_time": np.array([self.forcing_lead]),
                "variable": np.array(self.forcing_variables),
            },
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Advance the input signature by one step."""
        lead = input_coords["lead_time"].values + self.step_size
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
        if forcing is not None:
            leads = forcing["lead_time"].values.astype("timedelta64[h]").astype(int)
            self.calls.append([int(v) for v in leads])
        y = self.rule(x, forcing).assign_coords(
            lead_time=x["lead_time"] + self.step_size
        )
        return y.transpose(*x.dims).assign_attrs(x.attrs)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> tuple[xr.DataArray, int]:
        """Return the first forecast and a step counter."""
        return self._advance(x, forcing), 0

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray | None = None, state: int = 0
    ) -> tuple[xr.DataArray, int]:
        """Advance one step from the previous forecast."""
        return self._advance(y, forcing), state + 1

    def __call__(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> xr.DataArray:
        """Apply the model."""
        args = (x,) if forcing is None else (x, forcing)
        return self._default_call(*args)  # type: ignore[return-value]

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray | None = None
    ) -> Iterator[xr.DataArray]:
        """Iterate forecasts, taking forcing through ``send``."""
        args = (x,) if forcing is None else (x, forcing)
        return self._default_create_iterator(*args)  # type: ignore[return-value]

    def to(self, device: object) -> "Toy":
        """Return the model; it holds no device state."""
        return self


# %%
# Atmosphere and ocean
# --------------------
# A 6-hourly atmosphere forced by SST at its input time, and a 48-hourly ocean
# forced by the 48 h mean of z1000 over the step it is taking. The qualified
# label ``z1000:mean:48h`` is the whole declaration of the windowed exchange.

atmos = Toy(
    ["z1000"], forcing=["sst"], rule=lambda x, f: x + 1 + 0.1 * newest(f, "sst")
)
ocean = Toy(
    ["sst"],
    forcing=["z1000:mean:48h"],
    forcing_lead="48h",
    step="48h",
    grid=grid(16, 32),
    rule=lambda x, f: x + 0.01 * newest(f, "z1000:mean:48h"),
)
coupler = Coupler(
    {"atmos": atmos, "ocean": ocean},
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={
        "atmos": (constant(grid(32, 64), 0.0), None),
        "ocean": (constant(grid(16, 32), 2.0), None),
    },
)
print(coupler.describe())

# %%
# The plan shows what was derived: a 6 h clock, bilinear regridders between the
# grids, and the atmosphere ahead of the ocean because the ocean needs the
# atmosphere's frames from the same 48 h step while the atmosphere's SST is
# lagged. Run four days and check the numbers by hand.

steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(96, "h"))))
np.testing.assert_allclose(point(steps[1], "atmos"), 1.2)  # 0 + 1 + 0.1 * 2
np.testing.assert_allclose(point(steps[8], "ocean"), 2.042)  # 2 + 0.01 * mean(0..8.4)
np.testing.assert_allclose(point(steps[9], "atmos"), 10.8042)  # 9.6 + 1 + 0.2042
print([sorted(step) for step in steps[7:10]])

# %%
# Lagged versus same-step coupling
# --------------------------------
# A destination declaring its forcing at lead ``0h`` sees its partner's previous
# frame; at ``+6h`` it needs the frame produced this step, so ``derive_sequence``
# runs the source first. Here ``a`` counts 0, 1, 2 and ``b`` adds it.

for lead in ("0h", "6h"):
    b = Toy(
        ["b"],
        forcing=["a"],
        forcing_lead=lead,
        rule=lambda x, f: x + newest(f, "a").values,
    )
    system = Coupler(
        {"b": b, "a": Toy(["a"])},
        [Connector("a", "b")],
        {"a": constant(grid(32, 64), 0.0), "b": constant(grid(32, 64), 0.0)},
    )
    values = [
        point(s, "b") for s in system.run_item(WorkItem(T0, np.timedelta64(18, "h")))
    ]
    print(lead, system.sequence.slots[0].models, values, b.calls)

try:
    Coupler(
        {"b": Toy(["b"], forcing=["a"], forcing_lead="6h"), "a": Toy(["a"])},
        [Connector("a", "b")],
        {"a": constant(grid(32, 64), 0.0), "b": constant(grid(32, 64), 0.0)},
        sequence=RunSequence((Slot("6h", ("b", "a")),)),
    )
except CouplingError as error:
    print(error)

# %%
# Diagnostics and time windows
# ----------------------------
# A diagnostic joins as it is and runs after its sources each step. One that
# declares ``z1000:max:24h`` receives the trailing maximum and skips the first
# steps, until the window exists.


class DailyMax:
    """Diagnostic over a time window."""

    stochastic = False

    def input_coords(self) -> xr.DataArray:
        """Declare the input slot."""
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": np.array(["z1000:max:24h"])},
            dynamic=("batch",),
            grid=grid(32, 64),
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Advance the input signature by one step."""
        return coord_array_like(input_coords, {"variable": np.array(["z1000_dmax"])})

    def default_sources(self) -> None:
        """Recommend no source."""
        return None

    def to(self, device: object) -> "DailyMax":
        """Return the model; it holds no device state."""
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Apply the model."""
        return x.assign_coords(variable=np.array(["z1000_dmax"]))


chain = Coupler(
    {
        "atmos": Toy(
            ["z1000"], forcing=["sst"], rule=lambda x, f: x + 1 + 0.1 * newest(f, "sst")
        ),
        "daily": DailyMax(),
    },
    [Connector("atmos", "daily")],
    {"atmos": (constant(grid(32, 64), 0.0), constant(grid(32, 64), 2.0))},
)
for step in chain.run_item(WorkItem(T0, np.timedelta64(36, "h"))):
    hours = int(
        step["atmos"]["lead_time"].values[0].astype("timedelta64[h]").astype(int)
    )
    print(f"{hours:3d}h", "daily" in step)

# %%
# Vertical interpolation
# ----------------------
# A field with a ``level`` dimension is interpolated onto the destination's
# pressure levels by a connector ``transform`` (linear in log pressure). Ozone
# is ``log(p)`` here, so the result is exact.

hybrid = HybridLevels(a=(30000.0, 20000.0, 0.0), b=(0.0, 0.5, 1.0))  # 300/700/1000 hPa
received: list[np.ndarray] = []


def record(x: xr.DataArray, f: xr.DataArray | None) -> xr.DataArray:
    """Record the delivered column and persist the state."""
    received.append(newest(f, "o3").isel(lat=0, lon=0).values.ravel())
    return x


met = Toy(["o3"], grid=grid(8, 16), levels=np.arange(3.0), rule=lambda x, f: x)
chem = Toy(["o3"], forcing=["o3"], grid=grid(8, 16), levels=[500.0, 850.0], rule=record)
pressure = np.array(hybrid.a) + np.array(hybrid.b) * 1e5
ozone = xr.DataArray(
    np.broadcast_to(
        np.log(pressure)[None, None, :, None, None], (1, 1, 3, 8, 16)
    ).astype(np.float32),
    dims=("lead_time", "variable", "level", "lat", "lon"),
    coords={
        "lead_time": np.array([0], dtype="timedelta64[h]"),
        "variable": ["o3"],
        "level": np.arange(3.0),
        **grid(8, 16).coords(only_index=True),
    },
)
chem_ic = xr.zeros_like(ozone.isel(level=[0, 1])).assign_coords(level=[500.0, 850.0])
vertical = Coupler(
    {"met": met, "chem": chem},
    [Connector("met", "chem", transform=to_pressure_levels(hybrid, [500, 850], 1e5))],
)
list(
    vertical.run_item(
        WorkItem(T0, np.timedelta64(6, "h")), ics={"met": ozone, "chem": chem_ic}
    )
)
np.testing.assert_allclose(received[0], np.log([50000.0, 85000.0]), rtol=1e-6)
print("hybrid -> pressure exact:", received[0])
