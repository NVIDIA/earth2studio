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

"""Pull-pattern coupling tests (PullAdapter / StateDataSource).

MockPullModel is protocol-faithful to StormCast's coupling surface: it holds
a settable ``conditioning_data_source`` and, inside its own ``__call__``,
fetches its conditioning variables through the REAL
``earth2studio.data.utils.fetch_data`` — the same code path the production
model runs. If the shim satisfies fetch_data, it satisfies StormCast's
mechanics (the model-weights physics is exactly what these tests cannot
cover, and says so).
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.data.utils import fetch_data
from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import CallableComponent, PrognosticComponent
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field, State
from earth2studio.nvcoupler.pull import PullAdapter, StateDataSource
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")
GRID = (8, 16)


class MockPullModel:
    """StormCast-shaped mock: pulls u10m/t2m via fetch_data inside __call__,
    then steps its single state variable by the conditioning mean:
        refc <- refc + 1 + mean(u10m) + 0.1 * mean(t2m)
    Deterministic and hand-computable."""

    conditioning_variables = np.array(["u10m", "t2m"])

    def __init__(self):
        self.conditioning_data_source = None
        self.pull_log: list[np.ndarray] = []

    def input_coords(self):
        return OrderedDict(
            {
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["refc"]),
                **grid_coords(*GRID),
            }
        )

    def output_coords(self, input_coords):
        out = OrderedDict({k: v.copy() for k, v in input_coords.items()})
        out["lead_time"] = input_coords["lead_time"] + np.timedelta64(1, "h")
        return out

    def __call__(self, x, coords):
        if self.conditioning_data_source is None:
            raise RuntimeError("conditioning_data_source not set")
        # the REAL fetch path StormCast uses
        cond, _ = fetch_data(
            self.conditioning_data_source,
            time=np.atleast_1d(coords["time"]),
            variable=self.conditioning_variables,
        )
        self.pull_log.append(cond.numpy().copy())
        u_mean = cond[0, 0, 0].mean()
        t_mean = cond[0, 0, 1].mean()
        return x + 1.0 + u_mean + 0.1 * t_mean, self.output_coords(coords)

    def to(self, device):
        return self


def _field(name, value, valid=T0):
    return Field(
        torch.full(GRID, float(value)),
        grid_coords(*GRID),
        name,
        "m s-1" if "wind" in name else "K",
        valid_time=valid,
    )


def _imports(u=2.0, t=280.0):
    return State(
        "imports",
        [_field("eastward_wind_10m", u), _field("air_temperature_2m", t)],
    )


class TestStateDataSource:
    def test_serves_import_state_via_real_fetch_data(self):
        src = StateDataSource(
            _imports(u=3.0, t=290.0),
            raw_to_std={"u10m": "eastward_wind_10m", "t2m": "air_temperature_2m"},
        )
        x, coords = fetch_data(
            src, time=np.array([T0]), variable=np.array(["u10m", "t2m"])
        )
        assert x.shape == (1, 1, 2, *GRID)
        assert torch.all(x[0, 0, 0] == 3.0) and torch.all(x[0, 0, 1] == 290.0)
        assert list(coords["variable"]) == ["u10m", "t2m"]

    def test_unknown_variable_actionable_error(self):
        src = StateDataSource(_imports(), raw_to_std={})
        with pytest.raises(CouplingError, match="wire a connector"):
            src(np.array([T0]), np.array(["msl"]))

    def test_strict_time_mismatch_raises(self):
        src = StateDataSource(
            _imports(),
            raw_to_std={"u10m": "eastward_wind_10m", "t2m": "air_temperature_2m"},
            strict_time=True,
        )
        with pytest.raises(CouplingError, match="run-sequence ordering"):
            src(np.array([T0 + np.timedelta64(6, "h")]), np.array(["u10m"]))


class TestPullAdapter:
    def test_redirects_model_pull_to_imports(self):
        model = MockPullModel()
        comp = PrognosticComponent(
            "stormcast",
            model,
            imports=["eastward_wind_10m", "air_temperature_2m"],
            exports=["radar_reflectivity"],
            import_adapter=PullAdapter(),
            variable_aliases={"refc": "radar_reflectivity"},
            dictionary=_dict_with_refc(),
        )
        clock = Clock(T0, "2024-01-01T04:00", "1h")
        comp.realize(clock)
        ic = model.input_coords()
        ic["time"] = np.array([T0])
        comp.initialize(torch.zeros(1, 1, 1, *GRID), ic)
        comp.import_state.add(_field("eastward_wind_10m", 2.0))
        comp.import_state.add(_field("air_temperature_2m", 280.0))
        comp.run(clock.advance())
        # refc = 0 + 1 + 2 + 28 = 31, pulled through the model's own fetch
        got = comp.export_state["radar_reflectivity"]
        assert torch.allclose(got.data, torch.full(GRID, 31.0))
        assert len(model.pull_log) == 1

    def test_missing_attribute_actionable_error(self):
        class NoPull:
            def input_coords(self):
                return MockPullModel().input_coords()

            def output_coords(self, c):
                return MockPullModel().output_coords(c)

        comp = PrognosticComponent(
            "x",
            NoPull(),
            imports=["eastward_wind_10m"],
            exports=["radar_reflectivity"],
            import_adapter=PullAdapter(),
            variable_aliases={"refc": "radar_reflectivity"},
            dictionary=_dict_with_refc(),
        )
        comp.realize(Clock(T0, "2024-01-01T02:00", "1h"))
        ic = comp.model.input_coords()
        ic["time"] = np.array([T0])
        comp.initialize(torch.zeros(1, 1, 1, *GRID), ic)
        comp.import_state.add(_field("eastward_wind_10m", 1.0))
        with pytest.raises(CouplingError, match="ConditioningKwargAdapter"):
            comp.run(np.datetime64("2024-01-01T01:00"))


def _dict_with_refc():
    from earth2studio.nvcoupler.dictionary import (
        DEFAULT_DICTIONARY,
        FieldDictionary,
        FieldEntry,
    )

    d = FieldDictionary(DEFAULT_DICTIONARY)
    d.register(FieldEntry("radar_reflectivity", "dBZ", aliases=frozenset({"refc"})))
    return d


class TestCoupledPullWorkflow:
    """The Jussi-workflow shape end-to-end: a 'global' component's exports
    flow per-step into a pull-pattern regional model — no staging, no
    InferenceOutputSource, sequential exchange in one slot."""

    def test_per_step_conditioning_updates(self):
        def global_step(x, coords):
            # u10m grows 1 m/s per step; t2m constant
            return torch.stack([x[0] + 1.0, x[1]]), coords

        glob = CallableComponent(
            "global",
            global_step,
            timestep="1h",
            exports=["eastward_wind_10m", "air_temperature_2m"],
        )
        model = MockPullModel()
        stormcast = PrognosticComponent(
            "stormcast",
            model,
            imports=["eastward_wind_10m", "air_temperature_2m"],
            exports=["radar_reflectivity"],
            import_adapter=PullAdapter(),
            variable_aliases={"refc": "radar_reflectivity"},
            dictionary=_dict_with_refc(),
        )
        driver = Driver(
            {"global": glob, "stormcast": stormcast},
            sequence="""
            @1h
              global
              global -> stormcast
              stormcast
            @
            """,
            clock=Clock(T0, "2024-01-01T03:00", "1h"),
            connectors=[Connector(glob, stormcast)],
        )
        ic_glob = (
            torch.stack([torch.full(GRID, 2.0), torch.full(GRID, 280.0)]),
            OrderedDict(
                {"variable": np.array(["u10m", "t2m"]), **grid_coords(*GRID)}
            ),
        )
        ic_sc = MockPullModel().input_coords()
        ic_sc["time"] = np.array([T0])
        driver.initialize(
            {"global": ic_glob, "stormcast": (torch.zeros(1, 1, 1, *GRID), ic_sc)}
        )
        driver.run()
        # sequential coupling: stormcast at hour k pulls u = 2 + k (fresh)
        # refc increments: k=1: 1+3+28=32; k=2: 1+4+28=33; k=3: 1+5+28=34
        got = stormcast.export_state["radar_reflectivity"]
        assert torch.allclose(got.data, torch.full(GRID, 32.0 + 33.0 + 34.0))
        # three pulls, each seeing the CURRENT global state (3, 4, 5)
        pulled_u = [p[0, 0, 0].mean() for p in model.pull_log]
        assert pulled_u == [3.0, 4.0, 5.0]
