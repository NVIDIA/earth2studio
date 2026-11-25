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

"""Cross-component seam tests: exports must be exchange-shaped.

PrognosticComponent wraps models whose state carries singleton batch /
time / lead_time dims. Its published Fields must nevertheless be plain
spatial (lat, lon) tensors, or they break every consumer downstream:
VariableOverwriteAdapter's slice broadcast, State.as_tensor stacking with
fields from other components, and DiagnosticComponent's input conformance.
These tests exercise those three seams end to end.
"""

from collections import OrderedDict

import numpy as np
import torch
import xarray as xr

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import (
    CallableComponent,
    DataComponent,
    DiagnosticComponent,
    PrognosticComponent,
)
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")
NLAT, NLON = 8, 16
GRID = (NLAT, NLON)


class BatchTimePrognostic:
    """MockPrognostic with explicit singleton batch and time dims.

    Mirrors real earth2studio prognostics whose state tensors are
    (batch, time, lead_time, variable, lat, lon); +1.0 per 6 h step.
    """

    def __init__(self):
        self._in = OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([T0]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["z1000", "sst"]),
                **grid_coords(NLAT, NLON),
            }
        )

    def input_coords(self):
        return OrderedDict({k: v.copy() for k, v in self._in.items()})

    def output_coords(self, input_coords):
        out = OrderedDict({k: v.copy() for k, v in input_coords.items()})
        out["lead_time"] = input_coords["lead_time"] + np.timedelta64(6, "h")
        return out

    def __call__(self, x, coords):
        return x + 1.0, self.output_coords(coords)

    def to(self, device):
        return self


def _prognostic_ic(z0=0.0, sst0=3.0):
    """IC tensor matching BatchTimePrognostic.input_coords()."""
    coords = BatchTimePrognostic().input_coords()
    x = torch.empty(1, 1, 1, 2, NLAT, NLON)
    x[..., 0, :, :] = z0
    x[..., 1, :, :] = sst0
    return x, coords


def _sink_atmos():
    """CallableComponent on the same grid: z += 1 + 0.1 * sst."""

    def step(x, coords):
        z1000, sst = x[0], x[1]
        return torch.stack([z1000 + 1.0 + 0.1 * sst, sst]), coords

    return CallableComponent(
        "sink",
        step,
        timestep="6h",
        imports=["sea_surface_temperature"],
        exports=["geopotential_at_1000hpa"],
    )


def _sink_ic(z0=0.0, sst0=0.0):
    coords = OrderedDict(
        {"variable": np.array(["z1000", "sst"]), **grid_coords(NLAT, NLON)}
    )
    x = torch.stack([torch.full(GRID, z0), torch.full(GRID, sst0)])
    return x, coords


class HalfDiagnostic:
    """z500 = 0.5 * z1000 on the seam grid (models/dx/base.py interface)."""

    def __init__(self):
        self.grid = grid_coords(NLAT, NLON)

    def input_coords(self):
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["z1000"]),
                "lat": self.grid["lat"],
                "lon": self.grid["lon"],
            }
        )

    def output_coords(self, input_coords):
        out = OrderedDict(input_coords)
        out["variable"] = np.array(["z500"])
        return out

    def __call__(self, x, coords):
        return 0.5 * x, self.output_coords(coords)

    def to(self, device):
        return self


class ConstantDataSource:
    """In-memory DataSource: constant value per variable on the seam grid."""

    def __init__(self, values):
        self.values = values
        self.grid = grid_coords(NLAT, NLON)

    def __call__(self, time, variable) -> xr.DataArray:
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = np.atleast_1d(np.asarray(variable))
        lat, lon = self.grid["lat"], self.grid["lon"]
        data = np.empty((len(time), len(variable), len(lat), len(lon)))
        for j, v in enumerate(variable):
            data[:, j] = self.values[str(v)]
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": variable, "lat": lat, "lon": lon},
        )


def test_prognostic_export_feeds_callable_component():
    """(a) PrognosticComponent -> Connector -> CallableComponent import."""
    clock = Clock(T0, "2024-01-02", "6h")
    prog = PrognosticComponent("prog", BatchTimePrognostic())
    sink = _sink_atmos()
    prog.realize(clock)
    sink.realize(clock)
    prog.initialize(*_prognostic_ic(z0=0.0, sst0=3.0))
    sink.initialize(*_sink_ic())

    # exports must be exchange-shaped: plain (lat, lon), no batch/time/lead
    sst0 = prog.export_state["sea_surface_temperature"]
    assert list(sst0.coords) == ["lat", "lon"]
    assert sst0.data.shape == GRID

    conn = Connector(prog, sink)
    t1 = clock.advance()
    prog.run(t1)  # sst: 3 -> 4
    conn.execute(t1)
    sink.run(t1)  # z = 0 + 1 + 0.1 * 4 = 1.4

    z = sink.export_state["geopotential_at_1000hpa"]
    assert torch.allclose(z.data, torch.full(GRID, 1.4))


def test_data_and_prognostic_exports_stack_in_one_import_state():
    """(b) DataComponent + PrognosticComponent exports into one component's
    imports, stacked via State.as_tensor without cat_coords errors."""
    clock = Clock(T0, "2024-01-02", "6h")
    prog = PrognosticComponent("prog", BatchTimePrognostic())
    data = DataComponent(
        "data",
        source=ConstantDataSource({"t2m": 280.0}),
        exports=["air_temperature_2m"],
        timestep="6h",
    )

    def step(x, coords):
        return x, coords

    dst = CallableComponent(
        "dst",
        step,
        timestep="6h",
        imports=["geopotential_at_1000hpa", "air_temperature_2m"],
        exports=[],
    )
    for comp in (prog, data, dst):
        comp.realize(clock)
    prog.initialize(*_prognostic_ic(z0=0.0, sst0=3.0))
    data.initialize()
    dst.initialize(
        torch.zeros(1, NLAT, NLON),
        OrderedDict({"variable": np.array(["z1000"]), **grid_coords(NLAT, NLON)}),
    )

    t1 = clock.advance()
    prog.run(t1)  # z1000: 0 -> 1
    data.run(t1)
    Connector(prog, dst, fields=["geopotential_at_1000hpa"]).execute(t1)
    Connector(data, dst, fields=["air_temperature_2m"]).execute(t1)

    names = ["geopotential_at_1000hpa", "air_temperature_2m"]
    stacked, coords = dst.import_state.as_tensor(names)
    assert list(coords["variable"]) == names
    assert stacked.shape == (2, NLAT, NLON)
    assert torch.allclose(stacked[0], torch.full(GRID, 1.0))
    assert torch.allclose(stacked[1], torch.full(GRID, 280.0))


def test_diagnostic_consumes_prognostic_export_in_driver():
    """(c) DiagnosticComponent consuming a PrognosticComponent export."""
    components = {
        "prog": PrognosticComponent("prog", BatchTimePrognostic()),
        "diag": DiagnosticComponent("diag", HalfDiagnostic(), timestep="6h"),
    }
    dsl = """
@6h
  prog
  prog -> diag
  diag
@
"""
    driver = Driver(components, dsl, Clock(T0, "2024-01-02", "6h"))
    driver.initialize({"prog": _prognostic_ic(z0=0.0, sst0=0.0), "diag": (None, None)})
    ds = driver.run()

    assert driver.components["diag"].run_count == 4
    z500 = ds["diag"]["geopotential_at_500hpa"]
    assert z500.dims == ("time", "lat", "lon")
    # z1000_n = n (starts 0, +1 each step); diagnostic halves it
    assert np.allclose(z500.values, 0.5 * np.arange(1, 5)[:, None, None], atol=1e-6)
