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

"""DataComponent (prescribed forcing) and DiagnosticComponent tests.

Everything is mocked — no network, no model registries:

- MockDataSource returns the exact xr.DataArray shape fetch_data expects
  from a non-forecast DataSource: dims (time, variable, lat, lon).
- MockDiagnostic implements the models/dx/base.py interface and computes
  z500 = 0.5 * z1000, a hand-checkable single-step transform.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import DataComponent, DiagnosticComponent
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field
from earth2studio.nvcoupler.testing import (
    ATMOS_GRID,
    atmos_ic,
    fake_atmos,
    grid_coords,
)

T0 = "2024-01-01"
T24 = "2024-01-02"
T48 = "2024-01-03"


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class MockDataSource:
    """Deterministic in-memory DataSource: constant value per variable on a
    small lat/lon grid, dims (time, variable, lat, lon)."""

    def __init__(self, values: dict[str, float], nlat: int = 16, nlon: int = 32):
        self.values = values
        self.grid = grid_coords(nlat, nlon)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, time, variable) -> xr.DataArray:
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = np.atleast_1d(np.asarray(variable))
        self.calls.append((time.copy(), variable.copy()))
        lat, lon = self.grid["lat"], self.grid["lon"]
        data = np.empty((len(time), len(variable), len(lat), len(lon)))
        for j, v in enumerate(variable):
            data[:, j] = self.values[str(v)]
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": variable, "lat": lat, "lon": lon},
        )


class MockDiagnostic:
    """z500 = 0.5 * z1000 on the atmos grid (models/dx/base.py interface)."""

    def __init__(self, nlat: int = 32, nlon: int = 64):
        self.grid = grid_coords(nlat, nlon)
        self.call_count = 0

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
        # the component must present variables in the model's raw vocabulary
        assert list(coords["variable"]) == ["z1000"]
        assert x.ndim == len(coords)
        self.call_count += 1
        return 0.5 * x, self.output_coords(coords)

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# DataComponent
# ---------------------------------------------------------------------------
def make_data_ocean(values=None, **kwargs):
    source = MockDataSource(values or {"sst": 3.0})
    comp = DataComponent(
        "ocean",
        source=source,
        exports=["sea_surface_temperature"],
        timestep="24h",
        **kwargs,
    )
    return comp, source


def test_data_component_fetch_and_publish():
    comp, source = make_data_ocean()
    comp.realize(Clock(T0, T48, "6h"))
    comp.initialize()  # no IC: fetches at clock.start
    sst = comp.export_state["sea_surface_temperature"]
    assert list(sst.coords) == ["lat", "lon"]  # time/lead squeezed away
    assert sst.data.shape == (16, 32)
    assert torch.allclose(sst.data, torch.full((16, 32), 3.0))
    assert sst.valid_time == np.datetime64(T0)
    assert sst.units == "K"
    # raw name resolved through dictionary aliases, fetched at clock.start
    assert list(source.calls[0][1]) == ["sst"]
    assert source.calls[0][0][0] == np.datetime64(T0)
    # grid is known after the first fetch
    grid = comp.grid_coords()
    assert grid is not None and len(grid["lat"]) == 16 and len(grid["lon"]) == 32

    t1 = np.datetime64(T24)
    comp.run(t1)
    assert comp.run_count == 1
    assert comp.export_state["sea_surface_temperature"].valid_time == t1
    assert source.calls[-1][0][0] == t1


def test_data_component_variable_map():
    comp, source = make_data_ocean(
        values={"analysed_sst": 5.0},
        variable_map={"sea_surface_temperature": "analysed_sst"},
    )
    comp.realize(Clock(T0, T24, "6h"))
    comp.initialize()
    assert list(source.calls[0][1]) == ["analysed_sst"]
    sst = comp.export_state["sea_surface_temperature"]
    assert torch.allclose(sst.data, torch.full((16, 32), 5.0))


def test_data_component_initialize_before_realize_raises():
    comp, _ = make_data_ocean()
    with pytest.raises(CouplingError, match="realize"):
        comp.initialize()


def test_data_component_replaces_modeled_ocean_in_driver():
    """Prescribed forcing: swap fake_ocean for a DataComponent; the atmos,
    connector, and sequence stay as-is. z += 1 + 0.1 * sst each step."""
    ocean, source = make_data_ocean(values={"sst": 3.0})
    components = {"atmos": fake_atmos(), "ocean": ocean}
    dsl = """
@6h
  ocean -> atmos
  atmos
@24h
  ocean
@
"""
    driver = Driver(components, dsl, Clock(T0, T48, "6h"))
    # DataComponent needs no IC tensor; (None, None) means "fetch at t0"
    driver.initialize({"atmos": atmos_ic(), "ocean": (None, None)})
    ds = driver.run()

    atmos = driver.components["atmos"]
    assert atmos.run_count == 8  # 48h at 6h
    assert ocean.run_count == 2  # 24h and 48h
    # constant sst=3 from t0 on: z_n = n * (1 + 0.1 * 3) = 1.3 n
    z = ds["atmos"]["geopotential_at_1000hpa"]
    assert np.allclose(z.values, 1.3 * np.arange(9)[:, None, None], atol=1e-5)
    # the connector regridded the 16x32 source field onto the atmos grid
    sst_in = driver.probe("ocean->atmos")["sea_surface_temperature"]
    assert sst_in.data.shape == ATMOS_GRID
    assert torch.allclose(sst_in.data, torch.full(ATMOS_GRID, 3.0), atol=1e-6)


def test_data_component_no_arg_initialize_standalone():
    """DataComponent needs no IC: requires_ic is False and initialize()
    with no arguments fetches at clock.start."""
    comp, _ = make_data_ocean()
    assert comp.requires_ic is False
    comp.realize(Clock(T0, T24, "6h"))
    comp.initialize()  # no arguments
    assert "sea_surface_temperature" in comp.export_state


# ---------------------------------------------------------------------------
# DiagnosticComponent
# ---------------------------------------------------------------------------
def test_diagnostic_defaults_from_model_coords():
    comp = DiagnosticComponent("diag", MockDiagnostic(), timestep="6h")
    assert comp.import_names == ["geopotential_at_1000hpa"]
    assert comp.export_names == ["geopotential_at_500hpa"]
    # grid_coords available straight from the model's input_coords
    grid = comp.grid_coords()
    assert grid is not None and len(grid["lat"]) == 32 and len(grid["lon"]) == 64


def test_diagnostic_run_standalone():
    model = MockDiagnostic()
    comp = DiagnosticComponent("diag", model, timestep="6h")
    comp.realize(Clock(T0, T24, "6h"))
    comp.initialize()  # tolerant of no state tensor
    assert len(comp.export_state) == 0

    t1 = np.datetime64("2024-01-01T06")
    comp.import_state.add(
        Field(
            data=torch.full(ATMOS_GRID, 4.0),
            coords=grid_coords(*ATMOS_GRID),
            standard_name="geopotential_at_1000hpa",
            units="m2 s-2",
            valid_time=t1,
        )
    )
    comp.run(t1)
    out = comp.export_state["geopotential_at_500hpa"]
    assert list(out.coords) == ["lat", "lon"]  # batch dim squeezed back off
    assert torch.allclose(out.data, torch.full(ATMOS_GRID, 2.0))
    assert out.valid_time == t1
    assert model.call_count == 1


def test_diagnostic_no_arg_initialize_standalone():
    """DiagnosticComponent needs no IC: requires_ic is False and a no-arg
    initialize() derives its grid from the model's input_coords()."""
    comp = DiagnosticComponent("diag", MockDiagnostic(), timestep="6h")
    assert comp.requires_ic is False
    comp.initialize()  # no arguments, not even realized
    grid = comp.grid_coords()
    assert grid is not None and len(grid["lat"]) == 32 and len(grid["lon"]) == 64


def test_stateful_components_require_ic():
    assert fake_atmos().requires_ic is True


def test_diagnostic_missing_import_raises():
    comp = DiagnosticComponent("diag", MockDiagnostic(), timestep="6h")
    comp.realize(Clock(T0, T24, "6h"))
    comp.initialize()
    with pytest.raises(CouplingError, match="missing imports"):
        comp.run(np.datetime64("2024-01-01T06"))


def test_diagnostic_chain_in_driver():
    """fake_atmos exports z1000 -> diagnostic derives z500 = 0.5 * z1000."""
    model = MockDiagnostic()
    components = {
        "atmos": fake_atmos(),
        "diag": DiagnosticComponent("diag", model, timestep="6h"),
    }
    dsl = """
@6h
  atmos
  atmos -> diag
  diag
@
"""
    # atmos's sst import is intentionally unfed (held at its IC value)
    driver = Driver(components, dsl, Clock(T0, T24, "6h"), allow_unfed_imports=True)
    driver.initialize({"atmos": atmos_ic(), "diag": (None, None)})
    ds = driver.run()

    diag = driver.components["diag"]
    assert diag.run_count == 4  # every 6h over 24h
    assert model.call_count == 4
    # atmos: sst held at IC (2.0), so z_n = 1.2 n; diagnostic halves it
    z500 = ds["diag"]["geopotential_at_500hpa"]
    assert z500.dims == ("time", "lat", "lon")
    assert np.allclose(
        z500.values, 0.5 * 1.2 * np.arange(1, 5)[:, None, None], atol=1e-5
    )
    # exports valid at the driver time, same cadence as the source
    out = diag.export_state["geopotential_at_500hpa"]
    assert out.valid_time == np.datetime64(T24)


def test_diagnostic_gradient_flows_through_chain():
    gain = torch.tensor(1.0, requires_grad=True)
    components = {
        "atmos": fake_atmos(gain=gain),
        "diag": DiagnosticComponent("diag", MockDiagnostic(), timestep="6h"),
    }
    dsl = "@6h\n  atmos\n  atmos -> diag\n  diag\n@"
    # atmos's sst import is intentionally unfed (held at its IC value)
    driver = Driver(components, dsl, Clock(T0, T24, "6h"), allow_unfed_imports=True)
    driver.initialize({"atmos": atmos_ic(), "diag": (None, None)})
    with torch.enable_grad():
        states = driver.rollout(4)
    loss = states["diag"]["geopotential_at_500hpa"].data.sum()
    loss.backward()
    assert gain.grad is not None and gain.grad != 0
