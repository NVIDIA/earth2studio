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
"""Coupling models at their own cadences
=====================================

``CoupledRunner`` runs several models together, each at its own cadence and on
its own grid, and passes fields between them through ``Connector`` edges. The
models are ordinary Earth2Studio models: nothing wraps or subclasses them. A
coupled run is a ``Runner``, so ``Pipeline`` writes one store per component.

This example couples a synthetic 6-hourly weather model to a daily soil
moisture bucket that assimilates satellite retrievals arriving every three
days, then closes the loop by feeding soil moisture back to the weather model.
Everything is CPU-only and synthetic: no weights, downloads, or GPU.
"""

from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import xarray as xr

from earth2studio.coupling import Connector, CoupledRunner, CouplingError
from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import XarrayBackend
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import Pipeline, WorkItem, work_items
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import (
    CoordinateSystem,
    FrameSchema,
    TimeArray,
    VariableArray,
)

HOUR = np.timedelta64(1, "h")
WEATHER_GRID = LatLonGrid(
    np.array([20.0, 10.0, 0.0]), np.array([0.0, 10.0, 20.0, 30.0])
)
SOIL_GRID = LatLonGrid(np.array([15.0, 5.0]), np.array([5.0, 25.0]))


def signature(variables: list[str], grid: LatLonGrid, lead: int) -> CoordinateSystem:
    """Signature with a dynamic time prefix, one lead time in hours, and a grid."""
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


# %%
# A 6-hourly weather model
# ------------------------
# The weather model warms ``t2m`` by half a kelvin every step and alternates
# 6-hour precipitation accumulations between 4 mm and nothing. With
# ``feedback=True`` it also takes soil moisture as forcing at its state's valid
# time and cools where the soil is wet; we use that at the end.


class Weather(PrognosticMixin):
    """Synthetic 6 h weather model with ``t2m`` and 6 h ``tp`` accumulations."""

    def __init__(self, feedback: bool = False) -> None:
        self.feedback = feedback

    def input_coords(self) -> CoordinateSystem:
        """Declare the state at the initial time."""
        return signature(["t2m", "tp"], WEATHER_GRID, 0)

    def forcing_coords(self) -> CoordinateSystem | None:
        """Declare soil moisture forcing at the state's valid time, if coupled back."""
        return signature(["swvl1"], WEATHER_GRID, 0) if self.feedback else None

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Advance the input signature by 6 hours."""
        lead = input_coords["lead_time"].values[-1:] + 6 * HOUR
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, soil: xr.DataArray | None) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        t2m = last.sel(variable=["t2m"]) + 0.5
        if soil is not None:
            t2m = t2m - 2.0 * float(soil.mean())
        tp = 0.004 - last.sel(variable=["tp"])
        y = xr.concat([t2m, tp], dim="variable")
        return y.assign_coords(lead_time=last["lead_time"] + 6 * HOUR)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, soil: xr.DataArray | None = None
    ) -> tuple[xr.DataArray, None]:
        """Return the first 6-hour forecast and an empty state."""
        return self._advance(x, soil), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, soil: xr.DataArray | None = None, state: None = None
    ) -> tuple[xr.DataArray, None]:
        """Advance one 6-hour step."""
        return self._advance(y, soil), None

    def to(self, device: torch.device) -> "Weather":
        """Stay on the CPU."""
        return self


# %%
# A daily soil bucket that assimilates retrievals
# -----------------------------------------------
# The impact model is a stateful assimilation model: explicit state, a 24-hour
# step, and two forcing slots. The gridded slot asks for the previous day's
# precipitation sum and mean temperature with qualified labels; the coupler
# reduces the weather model's 6-hour frames to them. The second slot takes
# satellite retrievals as a DataFrame and is ``None`` on days without one, so
# the bucket nudges toward observations only when they exist.


class SoilBucket(PrognosticMixin):
    """Daily bucket: rain fills it, warmth dries it, retrievals nudge it."""

    def input_coords(self) -> CoordinateSystem:
        """Declare soil moisture at the initial time on the coarse soil grid."""
        return signature(["swvl1"], SOIL_GRID, 0)

    def forcing_coords(self) -> tuple[CoordinateSystem, FrameSchema]:
        """Declare the previous day's weather, then optional retrievals."""
        return (
            signature(["tp:sum:24h", "t2m:mean:24h"], SOIL_GRID, 24),
            FrameSchema(
                {
                    "time": np.empty(0, dtype="datetime64[ns]"),
                    "lat": np.empty(0, dtype=np.float32),
                    "lon": np.empty(0, dtype=np.float32),
                    "observation": np.empty(0, dtype=np.float32),
                    "variable": np.array(["swvl1"]),
                }
            ),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Advance the input signature by one day."""
        lead = input_coords["lead_time"].values[-1:] + 24 * HOUR
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(
        self, x: xr.DataArray, weather: xr.DataArray, obs: pd.DataFrame | None
    ) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        rain = weather.sel(variable="tp:sum:24h").values
        warmth = weather.sel(variable="t2m:mean:24h").values - 273.15
        y = last + 2.0 * rain - 0.0005 * warmth
        if obs is not None:
            y = y + 0.3 * (float(obs["observation"].mean()) - y)
        return y.clip(0.0, 0.5).assign_coords(lead_time=last["lead_time"] + 24 * HOUR)

    def initialize(
        self, x: xr.DataArray, weather: xr.DataArray, obs: pd.DataFrame | None = None
    ) -> tuple[xr.DataArray, None]:
        """Return the first daily analysis and an empty state."""
        return self._advance(x, weather, obs), None

    def step(
        self,
        y: xr.DataArray,
        weather: xr.DataArray,
        obs: pd.DataFrame | None,
        state: None = None,
    ) -> tuple[xr.DataArray, None]:
        """Advance one day, assimilating ``obs`` when present."""
        return self._advance(y, weather, obs), None

    def to(self, device: torch.device) -> "SoilBucket":
        """Stay on the CPU."""
        return self


# %%
# Sources
# -------
# Initial conditions come from data sources, as for any runner. Retrievals come
# from a DataFrame source that only has rows every 72 hours.


class Climatology:
    """Data source returning one constant per variable on a grid."""

    def __init__(self, grid: LatLonGrid, values: dict[str, float]) -> None:
        self.grid, self.values = grid, values

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return constant fields for each time and variable."""
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        data = np.stack(
            [
                np.full(
                    (len(self.grid.latitude), len(self.grid.longitude)), self.values[v]
                )
                for v in variable
            ]
        )
        return xr.DataArray(
            np.broadcast_to(data, (len(time), *data.shape)).astype(np.float32),
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


class Retrievals:
    """DataFrame source with a soil moisture retrieval of 0.30 every 72 hours."""

    SCHEMA = pa.schema(
        [
            ("time", pa.timestamp("ns")),
            ("lat", pa.float32()),
            ("lon", pa.float32()),
            ("observation", pa.float32()),
            ("variable", pa.string()),
        ]
    )

    def __init__(self, start: np.datetime64, every: int = 72) -> None:
        self.start, self.every = start, every

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Return the retrievals valid at ``time``; empty when there are none."""
        rows = [
            {
                "time": t,
                "lat": 10.0,
                "lon": 15.0,
                "observation": 0.30,
                "variable": "swvl1",
            }
            for t in np.atleast_1d(time)
            if int((t - self.start) // HOUR) % self.every == 0
        ]
        return pd.DataFrame(
            rows, columns=["time", "lat", "lon", "observation", "variable"]
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async variant of ``__call__``."""
        return self(time, variable, fields)


start = np.datetime64("2026-06-01T00")
weather_ic = Climatology(WEATHER_GRID, {"t2m": 288.15, "tp": 0.0})
soil_ic = Climatology(SOIL_GRID, {"swvl1": 0.20})

# %%
# Declare the coupling
# --------------------
# Components are named models. One connector feeds the weather model's fields
# into the bucket's first forcing slot, regridded onto the soil grid. The
# retrieval slot is not connected, so it is fetched from its source; ``None``
# in the bucket's source tuple keeps the recommendation (none here) for the
# gridded slot, which the connector feeds instead.

to_soil = BilinearRegridder(WEATHER_GRID, SOIL_GRID)
runner = CoupledRunner(
    {"weather": Weather(), "soil": SoilBucket()},
    [Connector("weather", "soil", regridder=to_soil)],
    sources={"weather": weather_ic, "soil": (soil_ic, None, Retrievals(start))},
)

# %%
# The schedule is derived from what each model declares. ``describe`` dry-runs
# it: the bucket initializes at +24h, once the weather model has produced the
# frames from +0h through +18h its 24-hour window reduces over.

print(runner.describe(np.timedelta64(72, "h")))

# %%
# Run one item
# ------------
# ``run_item`` yields each component's output as it is published, under the
# component's name. Day three's analysis is pulled toward the retrieval.

soil = {}
for step in runner.run_item(WorkItem(start, np.timedelta64(96, "h"))):
    if "soil" in step:
        x = step["soil"]
        soil[int(x["lead_time"].values[0] // HOUR)] = float(x.mean())
print({lead: round(value, 4) for lead, value in soil.items()})
# The jump into day three is more than twice a normal day's: pulled toward 0.30.
np.testing.assert_array_less(2 * (soil[48] - soil[24]), soil[72] - soil[48])

# %%
# Run many items with Pipeline
# ----------------------------
# A coupled runner is a ``Runner``: ``Pipeline`` plans one store per component
# stream, each at that component's own lead times and grid.

stores = {"weather": XarrayBackend(), "soil": XarrayBackend()}
items = work_items([start, start + 24 * HOUR], np.timedelta64(96, "h"))
Pipeline(runner, output=stores, device="cpu").run(items)
print(dict(stores["weather"].root["t2m"].sizes))
print(dict(stores["soil"].root["swvl1"].sizes))

# %%
# Feedback
# --------
# Now the weather model also takes soil moisture at each state's valid time.
# With ``exact`` edges both ways, the weather model waits for soil moisture the
# bucket has not produced, while the bucket waits for a day of weather. The
# runner finds the cycle when it is built, before running anything.

to_weather = BilinearRegridder(SOIL_GRID, WEATHER_GRID)
components = {"weather": Weather(feedback=True), "soil": SoilBucket()}
sources: dict[str, Climatology | tuple[Climatology, None, Retrievals]] = {
    "weather": weather_ic,
    "soil": (soil_ic, None, Retrievals(start)),
}
try:
    CoupledRunner(
        components,
        [
            Connector("weather", "soil", regridder=to_soil),
            Connector("soil", "weather", regridder=to_weather),
        ],
        sources=sources,
    )
except CouplingError as error:
    print(f"CouplingError: {error}")

# %%
# ``time="hold"`` makes the edge lag: the weather model takes the bucket's latest
# analysis from an earlier step, and the loop runs. No run sequence is
# declared; the order follows from the edges.

coupled = CoupledRunner(
    components,
    [
        Connector("weather", "soil", regridder=to_soil),
        Connector("soil", "weather", regridder=to_weather, time="hold"),
    ],
    sources=sources,
)
print(coupled.describe(np.timedelta64(30, "h")))
t2m = [
    float(step["weather"].sel(variable="t2m").mean())
    for step in coupled.run_item(WorkItem(start, np.timedelta64(48, "h")))
    if "weather" in step
]
print([round(value, 3) for value in t2m])
