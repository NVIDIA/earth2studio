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

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler import Connector, Coupler, CouplingError
from earth2studio.nvcoupler.testing import ToyPrognostic, toy_grid
from earth2studio.nvcoupler.vertical import (
    HybridLevels,
    PressureLevels,
    interp_to_pressure,
    to_pressure_levels,
)
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array, coord_array_like

GRID = toy_grid(8, 16)
PS = 100000.0


def _pressure_array() -> xr.DataArray:
    levels = np.array([100.0, 500.0, 1000.0])
    return xr.DataArray(
        np.log(levels * 100.0)[:, None, None],
        dims=("level", "lat", "lon"),
        coords={"level": levels, "lat": [0.0], "lon": [0.0]},
    )


def test_pressure_interpolation_is_exact_for_log_pressure():
    output = interp_to_pressure(
        _pressure_array(), PressureLevels((100, 500, 1000)), PressureLevels((200, 750))
    )
    assert output.dims == ("level", "lat", "lon")
    np.testing.assert_allclose(output[:, 0, 0], np.log([20000.0, 75000.0]))
    np.testing.assert_array_equal(output["level"], [200.0, 750.0])
    same = interp_to_pressure(
        _pressure_array(),
        PressureLevels((100, 500, 1000)),
        PressureLevels((100, 500, 1000)),
    )
    assert same is _pressure_array() or same.identical(_pressure_array())


def test_pressure_interpolation_clamps_endpoints():
    output = interp_to_pressure(
        _pressure_array(), PressureLevels((100, 500, 1000)), PressureLevels((50, 1100))
    )
    np.testing.assert_allclose(output[:, 0, 0], _pressure_array()[[0, -1], 0, 0])


@pytest.mark.parametrize("surface_pressure", [PS, "field"])
def test_hybrid_interpolation(surface_pressure):
    source = HybridLevels((1000.0, 5000.0, 10000.0), (0.0, 0.4, 0.9))
    pressure = np.array(source.a) + np.array(source.b) * PS
    array = xr.DataArray(
        np.log(pressure)[:, None, None],
        dims=("level", "lat", "lon"),
        coords={"level": np.arange(3), "lat": [0.0], "lon": [0.0]},
    )
    if surface_pressure == "field":
        surface_pressure = xr.DataArray(
            [[PS]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]}
        )
    output = interp_to_pressure(
        array, source, PressureLevels((200, 500)), surface_pressure
    )
    np.testing.assert_allclose(output[:, 0, 0], np.log([20000.0, 50000.0]))


@pytest.mark.parametrize(
    "levels,match",
    [
        ((1000.0, 500.0), "strictly increasing"),
        ((100.0, 100.0), "strictly increasing"),
        ((), "positive"),
        ((-1.0,), "positive"),
    ],
)
def test_pressure_levels_validation(levels, match):
    with pytest.raises(ValueError, match=match):
        PressureLevels(levels)


@pytest.mark.parametrize(
    "a,b,match",
    [
        ((1.0,), (0.0, 1.0), "equally long"),
        ((1000.0, 1000.0), (0.5, 0.5), "strictly increasing"),
    ],
)
def test_hybrid_levels_validation(a, b, match):
    with pytest.raises(ValueError, match=match):
        HybridLevels(a, b)


def test_interpolation_errors():
    hybrid = HybridLevels((1000.0, 5000.0), (0.0, 0.9))
    two = _pressure_array().isel(level=[0, 1])
    with pytest.raises(CouplingError, match="needs surface_pressure"):
        interp_to_pressure(two, hybrid, PressureLevels((500,)))
    with pytest.raises(
        CouplingError, match="2 levels but the source coordinate declares 3"
    ):
        interp_to_pressure(
            two, PressureLevels((100, 500, 1000)), PressureLevels((500,))
        )
    with pytest.raises(CouplingError, match="does not match the declared"):
        interp_to_pressure(
            _pressure_array(), PressureLevels((100, 400, 1000)), PressureLevels((500,))
        )
    with pytest.raises(CouplingError, match="no 'level'"):
        interp_to_pressure(
            _pressure_array().isel(level=0, drop=True),
            PressureLevels((100,)),
            PressureLevels((100,)),
        )
    bad_ps = xr.DataArray([1.0, 2.0], dims=("x",))
    with pytest.raises(CouplingError, match="not broadcastable"):
        interp_to_pressure(two, hybrid, PressureLevels((500,)), bad_ps)


class _Layered(ToyPrognostic):
    """Toy prognostic carrying one variable on a level dimension."""

    def __init__(self, levels, **kwargs):
        super().__init__(["o3"], **kwargs)
        self.levels = np.asarray(levels, dtype=np.float64)

    def input_coords(self):
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
        if not self.forcing_variables:
            return None
        return coord_array_like(
            self.input_coords(),
            {
                "lead_time": np.array([self.forcing_lead]),
                "variable": np.array(self.forcing_variables),
            },
        )


def test_connector_transform_interpolates_hybrid_levels_onto_pressure_levels():
    hybrid = HybridLevels((30000.0, 20000.0, 0.0), (0.0, 0.5, 1.0))  # 300/700/1000 hPa
    seen = []

    def chem_rule(x, forcing):
        seen.append(
            forcing.isel(time=0, lead_time=0, lat=0, lon=0).sel(variable="o3").values
        )
        return x

    met = _Layered(np.arange(3.0), grid=GRID, advance=lambda x, f: x)
    chem = _Layered(
        [500.0, 850.0], grid=GRID, forcing=["o3"], forcing_lead="0h", advance=chem_rule
    )
    pressure = np.array(hybrid.a) + np.array(hybrid.b) * PS
    ic = xr.DataArray(
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
    chem_ic = xr.zeros_like(ic.isel(level=[0, 1])).assign_coords(level=[500.0, 850.0])
    coupler = Coupler(
        {"met": met, "chem": chem},
        [
            Connector(
                "met", "chem", transform=to_pressure_levels(hybrid, [500, 850], PS)
            )
        ],
    )
    item = WorkItem(np.datetime64("2026-01-01"), np.timedelta64(6, "h"))
    # Neither model has a source: initial conditions must come from ics.
    with pytest.raises(CouplingError, match="no bound source"):
        list(coupler.run_item(item, ics={"met": ic}))
    list(coupler.run_item(item, ics={"met": ic, "chem": chem_ic}))
    np.testing.assert_allclose(seen[0], np.log([50000.0, 85000.0]), rtol=1e-6)
    # Without surface pressure a hybrid source is refused at the exchange.
    broken = Coupler(
        {"met": met, "chem": chem},
        [Connector("met", "chem", transform=to_pressure_levels(hybrid, [500, 850]))],
    )
    with pytest.raises(CouplingError, match="needs surface_pressure"):
        list(broken.run_item(item, ics={"met": ic, "chem": chem_ic}))
