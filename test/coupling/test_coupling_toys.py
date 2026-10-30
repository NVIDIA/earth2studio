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

"""Tiny CPU components for coupling tests.

Shared by the coupling tests. A uniquely named module, not ``conftest``, so
tests import it unambiguously when collected alongside other directories.

The atmosphere/ocean pair reproduces the hand-computable toy of PR #1229:
spatially constant fields, so every value can be checked by hand.

- atmosphere, 6 h: ``z1000 <- z1000 + 1 + 0.1 * sst`` with ``sst`` forcing at
  the state's valid time;
- ocean, 48 h: ``sst <- sst + 0.01 * z1000:mean:48h`` with the 48 h mean of
  the atmosphere's ``z1000`` over the window ending at the output valid time.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.grids import LatLonGrid, NearestRegridder
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.coords import handshake_dataarray
from earth2studio.utils.type import CoordinateSystem, FrameSchema

T0 = np.datetime64("2024-01-01T00")
H = np.timedelta64(1, "h")
ATMOS_GRID = LatLonGrid(np.array([45.0, -45.0]), np.array([0.0, 90.0, 180.0, 270.0]))
OCEAN_GRID = LatLonGrid(np.array([0.0]), np.array([0.0, 180.0]))


def _check(x: CoordinateSystem, expected: CoordinateSystem) -> None:
    """Validate ``x`` against ``expected`` at whatever lead times ``x`` carries."""
    if "lead_time" in expected.dims:
        if "lead_time" not in x.dims:
            raise ValueError("Input has no lead_time dimension")
        expected = coord_array_like(expected, {"lead_time": x["lead_time"].values})
    handshake_dataarray(x, expected)


def signature(
    variables: list[str], grid: LatLonGrid, leads: list[int] | None = (0,)  # type: ignore[assignment]
) -> CoordinateSystem:
    """Signature with a dynamic ``time`` prefix, lead times in hours, and a grid."""
    dims: tuple[str, ...] = ("time", "variable", "lat", "lon")
    coords: dict = {"variable": variables, "lat": grid.latitude, "lon": grid.longitude}
    if leads is not None:
        dims = ("time", "lead_time", "variable", "lat", "lon")
        coords["lead_time"] = np.array(leads, dtype="timedelta64[h]")
    return coord_array(dims, coords, dynamic=("time",))


class Fill:
    """Data source returning one constant per variable on a grid."""

    def __init__(self, grid: LatLonGrid, values: dict[str, float]) -> None:
        self.grid = grid
        self.values = values
        self.calls: list[tuple[list, list]] = []

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.calls.append((list(time), list(variable)))
        shape = (
            len(time),
            len(variable),
            len(self.grid.latitude),
            len(self.grid.longitude),
        )
        data = np.empty(shape, dtype=np.float32)
        for i, name in enumerate(variable):
            data[:, i] = self.values[str(name)]
        return xr.DataArray(
            data,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": self.grid.latitude,
                "lon": self.grid.longitude,
            },
        )


class _Toy(PrognosticMixin):
    """Single-state model forced by one slot: ``x <- x + a + b * mean(forcing)``."""

    def __init__(
        self,
        state: str,
        forcing: str,
        grid: LatLonGrid,
        step: int,
        offset: float,
        gain: float,
        forcing_lead: int = 0,
    ) -> None:
        self.state, self.forcing = state, forcing
        self.grid, self.hours = grid, step
        self.offset, self.gain = offset, gain
        self.forcing_lead = forcing_lead
        self.seen: list[list[np.timedelta64]] = []

    def input_coords(self) -> CoordinateSystem:
        return signature([self.state], self.grid)

    def forcing_coords(self) -> CoordinateSystem:
        return signature([self.forcing], self.grid, [self.forcing_lead])

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        _check(input_coords, self.input_coords())
        leads = input_coords["lead_time"].values[-1:] + self.hours * H
        return coord_array_like(input_coords, {"lead_time": leads})

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        _check(forcing, self.forcing_coords())
        self.seen.append(list(forcing["lead_time"].values))
        y = x.isel(lead_time=[-1]) + self.offset + self.gain * float(forcing.mean())
        return y.assign_coords(lead_time=y["lead_time"] + self.hours * H)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        return self._advance(x, forcing), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, forcing), None

    def __call__(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        return self._default_call(x, forcing)  # type: ignore[return-value]

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, forcing)  # type: ignore[return-value]

    def to(self, device: torch.device) -> _Toy:
        return self


def atmosphere() -> _Toy:
    """6 h atmosphere forced by SST at the state's valid time."""
    return _Toy("z1000", "sst", ATMOS_GRID, 6, 1.0, 0.1)


def ocean() -> _Toy:
    """48 h ocean forced by the atmosphere's trailing 48 h mean ``z1000``."""
    return _Toy("sst", "z1000:mean:48h", OCEAN_GRID, 48, 0.0, 0.01, forcing_lead=48)


class Double:
    """Diagnostic publishing ``z2 = 2 * z1000``."""

    stochastic = False

    def __init__(self, grid: LatLonGrid = ATMOS_GRID) -> None:
        self.grid = grid
        self.calls = 0

    def input_coords(self) -> CoordinateSystem:
        return signature(["z1000"], self.grid)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        _check(input_coords, self.input_coords())
        return coord_array_like(input_coords, {"variable": ["z2"]})

    def default_sources(self) -> None:
        return None

    def to(self, device: torch.device) -> Double:
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        self.calls += 1
        return (2 * x).assign_coords(variable=["z2"])


class Bucket(PrognosticMixin):
    """Daily soil bucket nudged toward sparse observations (stateful assimilation).

    ``swvl1 <- swvl1 + 0.001 * tp:sum:24h``, then, when an observation exists,
    ``swvl1 <- swvl1 + 0.5 * (obs - swvl1)``.
    """

    def __init__(self, grid: LatLonGrid = ATMOS_GRID) -> None:
        self.grid = grid
        self.observed: list[bool] = []

    def input_coords(self) -> CoordinateSystem:
        return signature(["swvl1"], self.grid)

    def forcing_coords(self) -> tuple[CoordinateSystem, FrameSchema]:
        return (
            signature(["tp:sum:24h"], self.grid, [24]),
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
        leads = input_coords["lead_time"].values[-1:] + 24 * H
        return coord_array_like(input_coords, {"lead_time": leads})

    def _advance(
        self, x: xr.DataArray, rain: xr.DataArray, obs: pd.DataFrame | None
    ) -> xr.DataArray:
        y = x.isel(lead_time=[-1]) + 0.001 * float(rain.mean())
        self.observed.append(obs is not None)
        if obs is not None:
            y = y + 0.5 * (float(obs["observation"].mean()) - y)
        return y.assign_coords(lead_time=y["lead_time"] + 24 * H)

    def initialize(
        self, x: xr.DataArray, rain: xr.DataArray, obs: pd.DataFrame | None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(x, rain, obs), None

    def step(
        self,
        y: xr.DataArray,
        rain: xr.DataArray,
        obs: pd.DataFrame | None,
        state: None = None,
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, rain, obs), None

    def to(self, device: torch.device) -> Bucket:
        return self


class Rain(PrognosticMixin):
    """6 h model whose ``tp`` is a constant 6 h accumulation of 2.0."""

    def __init__(self, grid: LatLonGrid = ATMOS_GRID) -> None:
        self.grid = grid

    def input_coords(self) -> CoordinateSystem:
        return signature(["tp"], self.grid)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        _check(input_coords, self.input_coords())
        leads = input_coords["lead_time"].values[-1:] + 6 * H
        return coord_array_like(input_coords, {"lead_time": leads})

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return self._default_call(x)  # type: ignore[return-value]

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x)  # type: ignore[return-value]

    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        y = x.isel(lead_time=[-1]) * 0 + 2.0
        return y.assign_coords(lead_time=y["lead_time"] + 6 * H)

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:  # type: ignore[override]
        return self._advance(x), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y), None

    def to(self, device: torch.device) -> Rain:
        return self


class Observations:
    """Data frame source with soil moisture retrievals of 0.4 every 72 hours."""

    def __init__(self, every: int = 72) -> None:
        self.every = every
        self.requests: list[np.datetime64] = []

    def __call__(
        self, time: np.ndarray, variable: np.ndarray, fields: object = None
    ) -> pd.DataFrame:
        rows = []
        for t in np.atleast_1d(time):
            self.requests.append(t)
            hours = int((t - T0) // H)
            if hours % self.every == 0:
                rows.append(
                    {
                        "time": t,
                        "lat": 0.0,
                        "lon": 0.0,
                        "observation": 0.4,
                        "variable": "swvl1",
                    }
                )
        return pd.DataFrame(
            rows, columns=["time", "lat", "lon", "observation", "variable"]
        )


class Mixed(PrognosticMixin):
    """6 h atmosphere carrying SST in its state: ``z1000 <- z1000 + 1 + 0.1 * sst``."""

    def input_coords(self) -> CoordinateSystem:
        return signature(["z1000", "sst"], ATMOS_GRID)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        _check(input_coords, self.input_coords())
        leads = input_coords["lead_time"].values[-1:] + 6 * H
        return coord_array_like(input_coords, {"lead_time": leads})

    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        y = x.isel(lead_time=[-1]).copy()
        sst = float(y.sel(variable="sst").mean())
        y.loc[{"variable": "z1000"}] = y.sel(variable="z1000") + 1 + 0.1 * sst
        return y.assign_coords(lead_time=y["lead_time"] + 6 * H)

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:  # type: ignore[override]
        return self._advance(x), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y), None

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return self._default_call(x)  # type: ignore[return-value]

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x)  # type: ignore[return-value]

    def to(self, device: torch.device) -> Mixed:
        return self


TO_OCEAN = NearestRegridder(ATMOS_GRID, OCEAN_GRID)
TO_ATMOS = NearestRegridder(OCEAN_GRID, ATMOS_GRID)


def toy_runner(ic_sources: Mapping[str, Fill]) -> CoupledRunner:
    """The atmosphere/ocean toy: the ocean held into the atmosphere, means back."""
    return CoupledRunner(
        {"atmos": atmosphere(), "ocean": ocean()},
        [
            Connector(
                "ocean",
                "atmos",
                time="hold",
                regridder=TO_ATMOS,
            ),
            Connector("atmos", "ocean", regridder=TO_OCEAN),
        ],
        sources=dict(ic_sources),
    )


def run(runner: CoupledRunner, hours: int) -> dict[str, list[xr.DataArray]]:
    """Run one item to ``hours``, collecting each stream's frames."""
    streams: dict[str, list[xr.DataArray]] = {}
    for step in runner.run_item(WorkItem(T0, np.timedelta64(hours, "h"))):
        for name, x in step.items():
            streams.setdefault(name, []).append(x)
    return streams


def series(frames: list[xr.DataArray]) -> dict[int, float]:
    """Spatial mean of each frame by lead time in hours."""
    return {int(x["lead_time"].values[0] // H): float(x.mean()) for x in frames}
