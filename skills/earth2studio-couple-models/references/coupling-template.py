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

"""Template: couple two models with CoupledRunner and run them under Pipeline.

Runs as is on CPU, with no weights or downloads:

    uv run python skills/earth2studio-couple-models/references/coupling-template.py

A 6-hourly atmosphere feeds a daily land model the previous day's rain total
and mean temperature. The land model feeds soil moisture back, lagged with
``time="hold"``. Sections marked ADAPT are where real models, sources and
connectors go; the toy models only stand in for them.
"""

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.coupling import Connector, CoupledRunner, CouplingError
from earth2studio.grids import BilinearRegridder, LatLonGrid, infer_grid
from earth2studio.io import XarrayBackend
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import Pipeline, WorkItem, work_items
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem, TimeArray, VariableArray

HOUR = np.timedelta64(1, "h")
ATMOS_GRID = LatLonGrid(np.array([20.0, 10.0, 0.0]), np.array([0.0, 10.0, 20.0, 30.0]))
LAND_GRID = LatLonGrid(np.array([15.0, 5.0]), np.array([5.0, 25.0]))


# %%
# Stand-in models (ADAPT: load real models instead)
# -------------------------------------------------
# Components are ordinary models; the coupler never wraps them. Anything that
# satisfies PrognosticModel, StatefulAssimilationModel or DiagnosticModel works,
# e.g. ``SomeModel.load_model(SomeModel.load_default_package())``.


def signature(variables: list[str], grid: LatLonGrid, lead: int) -> CoordinateSystem:
    """Slot signature: dynamic ``time``, one lead time in hours, and a grid."""
    return coord_array(
        ("time", "lead_time", "variable", "lat", "lon"),
        {
            "lead_time": np.array([lead], dtype="timedelta64[h]"),
            "variable": variables,
            "lat": grid.latitude,
            "lon": grid.longitude,
        },
        dynamic=("time",),
    )


class Atmosphere(PrognosticMixin):
    """6 h model: ``t2m`` warms 0.5 K a step, cooled by wet soil; ``tp`` is 6 h rain."""

    def input_coords(self) -> CoordinateSystem:
        """State at the initial time."""
        return signature(["t2m", "tp"], ATMOS_GRID, 0)

    def forcing_coords(self) -> CoordinateSystem:
        """Soil moisture at the state's valid time."""
        return signature(["swvl1"], ATMOS_GRID, 0)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Advance by 6 hours."""
        lead = input_coords["lead_time"].values[-1:] + 6 * HOUR
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, soil: xr.DataArray) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        t2m = last.sel(variable=["t2m"]) + 0.5 - 2.0 * float(soil.mean())
        tp = 0.004 - last.sel(variable=["tp"])
        y = xr.concat([t2m, tp], dim="variable")
        return y.assign_coords(lead_time=last["lead_time"] + 6 * HOUR)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, soil: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        """First forecast and an empty state."""
        return self._advance(x, soil), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, soil: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        """Advance one step."""
        return self._advance(y, soil), None

    def to(self, device: torch.device) -> "Atmosphere":
        """Stay on the CPU."""
        return self


class Land(PrognosticMixin):
    """24 h bucket: rain fills it; warmth, in degrees Celsius, dries it."""

    def input_coords(self) -> CoordinateSystem:
        """Soil moisture at the initial time, on the coarse land grid."""
        return signature(["swvl1"], LAND_GRID, 0)

    def forcing_coords(self) -> CoordinateSystem:
        """The previous day's rain total and mean temperature.

        Qualified labels (``base:statistic:window``) ask the coupler to reduce
        the producer's frames over ``[T - 24h, T)``.
        """
        return signature(["tp:sum:24h", "t2m:mean:24h"], LAND_GRID, 24)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Advance by one day."""
        lead = input_coords["lead_time"].values[-1:] + 24 * HOUR
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, weather: xr.DataArray) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        rain = weather.sel(variable="tp:sum:24h").values
        celsius = weather.sel(variable="t2m:mean:24h").values
        y = (last + 2.0 * rain - 0.0005 * celsius).clip(0.0, 0.5)
        return y.assign_coords(lead_time=last["lead_time"] + 24 * HOUR)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, weather: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        """First daily analysis and an empty state."""
        return self._advance(x, weather), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, weather: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        """Advance one day."""
        return self._advance(y, weather), None

    def to(self, device: torch.device) -> "Land":
        """Stay on the CPU."""
        return self


# %%
# Stand-in sources (ADAPT: real data sources, or omit to use recommendations)
# ---------------------------------------------------------------------------


class Constant:
    """Data source returning one constant per variable on a grid."""

    def __init__(self, grid: LatLonGrid, values: dict[str, float]) -> None:
        self.grid, self.values = grid, values

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Constant fields for each time and variable."""
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        shape = (len(time), len(self.grid.latitude), len(self.grid.longitude))
        data = np.stack([np.full(shape, self.values[str(v)]) for v in variable], 1)
        return xr.DataArray(
            data.astype(np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords=OrderedDict(
                time=time,
                variable=variable,
                lat=self.grid.latitude,
                lon=self.grid.longitude,
            ),
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async variant of ``__call__``."""
        return self(time, variable)


# %%
# 1. Inspect what each model declares
# -----------------------------------
# Connectors are derived from these declarations: slot variables (with any
# ``base:statistic:window`` labels), lead times, and grids. Read them first.


def inspect(name: str, model: object) -> None:
    """Print a model's slots: variables, lead times and grid size."""

    def slots(signature: object) -> tuple:
        if signature is None:
            return ()
        return signature if isinstance(signature, tuple) else (signature,)

    inputs = model.input_coords()  # type: ignore[attr-defined]
    outputs = model.output_coords(inputs)  # type: ignore[attr-defined]
    forcing = getattr(model, "forcing_coords", lambda: None)()
    print(f"{name}:")
    for kind, group in (("input", inputs), ("forcing", forcing), ("output", outputs)):
        for i, x in enumerate(slots(group)):
            if not isinstance(x, xr.DataArray):
                print(f"  {kind}[{i}] DataFrame {list(x)}")
                continue
            leads = x.coords["lead_time"].values if "lead_time" in x.dims else []
            print(
                f"  {kind}[{i}] {list(x.coords['variable'].values)} "
                f"leads={[str(v.astype('timedelta64[h]')) for v in leads]} "
                f"grid={infer_grid(x).shape}"
            )


models = {"atmos": Atmosphere(), "land": Land()}  # Declaration order breaks ties.
for name, model in models.items():
    inspect(name, model)

# %%
# 2. Declare connectors (ADAPT)
# -----------------------------
# One connector per consumer slot. ``exact`` (default) waits for the producer's
# frame at each required time, or reduces frames for qualified labels. ``hold``
# takes the latest frame published earlier, which breaks feedback cycles.
# A regridder is required whenever grids differ; ``transform`` converts units
# after regridding and must keep ``variable`` and ``lead_time``.


def kelvin_to_celsius(x: xr.DataArray) -> xr.DataArray:
    """Convert the mean temperature label to degrees Celsius."""
    return x.where(x["variable"] != "t2m:mean:24h", x - 273.15)


connectors = [
    Connector(
        "atmos",
        "land",
        regridder=BilinearRegridder(ATMOS_GRID, LAND_GRID),
        transform=kelvin_to_celsius,
    ),
    Connector(
        "land",
        "atmos",
        regridder=BilinearRegridder(LAND_GRID, ATMOS_GRID),
        time="hold",
    ),
]

# Sources per component, in input-then-forcing slot order. ``None`` keeps the
# model's recommendation; connected slots use the source only for frames at or
# before the item time that the producer does not hold.
sources = {
    "atmos": (
        Constant(ATMOS_GRID, {"t2m": 288.15, "tp": 0.0}),
        Constant(ATMOS_GRID, {"swvl1": 0.2}),
    ),
    "land": Constant(LAND_GRID, {"swvl1": 0.2}),
}

# %%
# 3. Build: validation runs here and names the broken rule (C1-C12)
# -----------------------------------------------------------------

try:
    runner = CoupledRunner(models, connectors, sources=sources, seed=0)
except CouplingError as error:
    print(f"CouplingError: {error}")
    raise

# %%
# 4. Check the schedule before running anything
# ---------------------------------------------
# Each row is a call: when it is due, what it produces, and where every
# connected slot's data comes from. Confirm the lags and reduction windows are
# the ones intended.

print(runner.describe(np.timedelta64(48, "h")))

# %%
# 5. Run one item
# ---------------
# ``run_item`` yields each component's output under its stream name as it is
# published; a component with several output slots yields ``name.0``, ...

start = np.datetime64("2026-06-01T00")
land = {}
for published in runner.run_item(WorkItem(start, np.timedelta64(72, "h"))):
    if "land" in published:
        x = published["land"]
        land[int(x["lead_time"].values[0] // HOUR)] = round(float(x.mean()), 4)
print(land)
assert sorted(land) == [0, 24, 48, 72]

# %%
# 6. Run many items with Pipeline
# -------------------------------
# One store per stream, each at its component's own lead times and grid.
# ADAPT: Pipeline(runner, output=..., device="cuda") moves components that are
# not pinned with ``devices=`` and stages fields onto each component's device.

stores = {"atmos": XarrayBackend(), "land": XarrayBackend()}
items = work_items([start, start + 24 * HOUR], np.timedelta64(72, "h"))
Pipeline(runner, output=stores, device="cpu").run(items)
print(dict(stores["atmos"].root["t2m"].sizes))
print(dict(stores["land"].root["swvl1"].sizes))
