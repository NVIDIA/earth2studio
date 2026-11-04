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

"""Testing template: check a coupling with hand-computable toy models.

Run as is:

    uv run pytest skills/earth2studio-couple-models/references/testing-template.py

Toys hold spatially constant fields and add a known increment each step, so
every value a consumer receives can be computed by hand. Mirror the real
models' cadences, slot labels and grids in the toys, then assert:

- the exact values each component publishes,
- the lead times each component was forced at (``seen``),
- the schedule ``describe`` prints for the lags you intend,
- that a wrong wiring fails at construction with ``CouplingError``.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.coupling import Connector, CoupledRunner, CouplingError
from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import XarrayBackend
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import Pipeline, WorkItem, work_items
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")
H = np.timedelta64(1, "h")
GRID = LatLonGrid(np.array([45.0, -45.0]), np.array([0.0, 90.0, 180.0, 270.0]))
COARSE = LatLonGrid(np.array([0.0]), np.array([0.0, 180.0]))


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


class Toy(PrognosticMixin):
    """``state <- state + offset + gain * mean(forcing)`` every ``hours``."""

    def __init__(
        self,
        state: str,
        forcing: str,
        hours: int,
        offset: float,
        gain: float,
        forcing_lead: int = 0,
        grid: LatLonGrid = GRID,
    ) -> None:
        self.state, self.forcing, self.hours = state, forcing, hours
        self.offset, self.gain = offset, gain
        self.forcing_lead, self.grid = forcing_lead, grid
        self.seen: list[list[np.timedelta64]] = []

    def input_coords(self) -> CoordinateSystem:
        return signature([self.state], self.grid, 0)

    def forcing_coords(self) -> CoordinateSystem:
        return signature([self.forcing], self.grid, self.forcing_lead)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = input_coords["lead_time"].values[-1:] + self.hours * H
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
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

    def to(self, device: torch.device) -> "Toy":
        return self


class Constant:
    """Data source returning one constant per variable on a grid."""

    def __init__(self, grid: LatLonGrid, values: dict[str, float]) -> None:
        self.grid, self.values = grid, values

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
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


def components(feedback: str = "c") -> dict[str, Toy]:
    """``fast`` adds 1 every 6 h; ``slow`` adds the mean of ``u`` over each day."""
    return {
        "fast": Toy("u", feedback, 6, 1.0, 0.0),
        "slow": Toy("v", "u:mean:24h", 24, 0.0, 1.0, forcing_lead=24),
    }


SOURCES = {
    "fast": (Constant(GRID, {"u": 0.0}), Constant(GRID, {"c": 0.0, "v": 0.0})),
    "slow": Constant(GRID, {"v": 0.0}),
}


def test_values_follow_by_hand() -> None:
    models = components()
    runner = CoupledRunner(models, [Connector("fast", "slow")], sources=SOURCES)
    slow = {}
    for published in runner.run_item(WorkItem(T0, np.timedelta64(48, "h"))):
        if "slow" in published:
            x = published["slow"]
            slow[int(x["lead_time"].values[0] // H)] = float(x.mean())
    # Day 1 averages u at +0..+18h = 0, 1, 2, 3; day 2 averages 4, 5, 6, 7.
    assert slow == pytest.approx({0: 0.0, 24: 1.5, 48: 1.5 + 5.5})
    assert models["slow"].seen == [[24 * H], [48 * H]]


def test_schedule_shows_the_intended_window() -> None:
    runner = CoupledRunner(components(), [Connector("fast", "slow")], sources=SOURCES)
    plan = runner.describe(np.timedelta64(48, "h"))
    rows = [row.split() for row in plan.splitlines()]
    # The day-1 analysis is due at +24h and reduces fast's frames from +0h..+18h.
    first = next(row for row in rows if row[1:3] == ["slow", "initialize"])
    assert first[0] == "+24h"
    assert " ".join(first[-5:]) == "fast at +24h from +0h..+18h]"


def test_exact_feedback_is_a_cycle_and_hold_breaks_it() -> None:
    models = components(feedback="v")
    forward = Connector("fast", "slow")
    with pytest.raises(CouplingError, match="cycle"):
        CoupledRunner(models, [forward, Connector("slow", "fast")], sources=SOURCES)
    runner = CoupledRunner(
        models, [forward, Connector("slow", "fast", time="hold")], sources=SOURCES
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(24, "h"))))


def test_grids_that_differ_need_a_regridder() -> None:
    models = components()
    models["slow"] = Toy("v", "u:mean:24h", 24, 0.0, 1.0, forcing_lead=24, grid=COARSE)
    sources = {**SOURCES, "slow": Constant(COARSE, {"v": 0.0})}
    with pytest.raises(CouplingError, match="regridder"):
        CoupledRunner(models, [Connector("fast", "slow")], sources=sources)
    regridder = BilinearRegridder(GRID, COARSE)
    CoupledRunner(
        models, [Connector("fast", "slow", regridder=regridder)], sources=sources
    )


def test_pipeline_writes_one_store_per_component() -> None:
    runner = CoupledRunner(components(), [Connector("fast", "slow")], sources=SOURCES)
    stores = {"fast": XarrayBackend(), "slow": XarrayBackend()}
    items = work_items([T0], np.timedelta64(48, "h"))
    Pipeline(runner, output=stores, device="cpu").run(items)
    assert stores["fast"].root["u"].sizes["lead_time"] == 9
    assert stores["slow"].root["v"].sizes["lead_time"] == 3
