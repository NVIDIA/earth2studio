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

"""Smoke tests: PrognosticComponent against a real earth2studio prognostic.

Uses the Persistence model (models/px/persistence.py) — a real
PrognosticModel implementing the full protocol (input_coords /
output_coords / batch_func-decorated __call__) that returns its input
unchanged and needs no weights. This exercises the seam between
nvcoupler's component phases and the earth2studio model conventions:
'batch' dims of np.empty(0) in input_coords, lead_time windows, and
batch_func's compress/decompress behavior.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.px import Persistence
from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import PrognosticComponent
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")
VARIABLES = ["t2m", "z1000"]
NLAT, NLON = 8, 16


def _make_component() -> PrognosticComponent:
    model = Persistence(VARIABLES, grid_coords(NLAT, NLON))
    return PrognosticComponent("persist", model)


def _make_ic(model: Persistence) -> tuple[torch.Tensor, OrderedDict]:
    """Build an IC tensor matching model.input_coords() with the empty
    'batch' dim scrubbed (batch_func re-inserts it on call, the pattern
    run.py uses when preparing model inputs)."""
    coords = model.input_coords()
    del coords["batch"]  # np.empty(0) placeholder, not a real axis
    shape = tuple(len(v) for v in coords.values())
    torch.manual_seed(0)
    x = torch.randn(shape)
    return x, coords


def test_timestep_inferred_from_output_coords():
    comp = _make_component()
    assert comp.timestep == np.timedelta64(6, "h")
    # exports resolved from raw variable names to dictionary standard names
    assert comp.export_names == [
        "air_temperature_2m",
        "geopotential_at_1000hpa",
    ]


def test_realize_initialize_run_loop_persists_values():
    comp = _make_component()
    clock = Clock(T0, "2024-01-02", "6h")
    comp.realize(clock)

    x0, coords0 = _make_ic(comp.model)
    comp.initialize(x0, coords0)

    # initialize seeds the exports at t0 for lagged coupling
    for std in comp.export_names:
        assert std in comp.export_state
        assert comp.export_state[std].valid_time == clock.start
        assert list(comp.export_state[std].coords) == ["lat", "lon"]

    var_axis = list(coords0).index("variable")
    # exchange-shaped IC slices: the singleton lead_time dim dropped, since
    # published Fields must carry plain (lat, lon) coords
    ic_slices = {
        "air_temperature_2m": x0.select(var_axis, 0)[0],
        "geopotential_at_1000hpa": x0.select(var_axis, 1)[0],
    }

    for i in range(1, 4):  # three 6 h steps
        time = T0 + i * np.timedelta64(6, "h")
        assert comp.should_run(time)
        comp.run(time)
        for std, ic in ic_slices.items():
            field = comp.export_state[std]
            assert field.standard_name == std
            assert field.valid_time == time
            assert field.source == "persist"
            # exports are exchange-shaped: no batch/time/lead_time singletons
            assert list(field.coords) == ["lat", "lon"]
            # persistence: values are the IC, unchanged, on the same grid
            assert torch.equal(field.data, ic)
            assert np.array_equal(field.coords["lat"], coords0["lat"])
            assert np.array_equal(field.coords["lon"], coords0["lon"])
    assert comp.run_count == 3


def test_internal_state_window_stays_model_shaped():
    comp = _make_component()
    clock = Clock(T0, "2024-01-02", "6h")
    comp.realize(clock)
    x0, coords0 = _make_ic(comp.model)
    comp.initialize(x0, coords0)
    comp.run(T0 + np.timedelta64(6, "h"))
    x, coords = comp.state
    # next_input rewound lead_time to the model's input window
    assert np.array_equal(coords["lead_time"], comp.model.input_coords()["lead_time"])
    assert list(coords) == ["lead_time", "variable", "lat", "lon"]
    assert x.shape == (1, len(VARIABLES), NLAT, NLON)


def test_multi_history_needs_next_input_hook():
    """A history-2 Persistence outputs 1 lead time but takes 2 as input:
    the default next_input cannot manage the window and must say so."""
    from earth2studio.nvcoupler.errors import CouplingError

    model = Persistence(VARIABLES, grid_coords(NLAT, NLON), history=2)
    comp = PrognosticComponent("persist2", model)
    clock = Clock(T0, "2024-01-02", "6h")
    comp.realize(clock)
    coords = model.input_coords()
    del coords["batch"]
    x = torch.randn(tuple(len(v) for v in coords.values()))
    comp.initialize(x, coords)
    with pytest.raises(CouplingError, match="next_input"):
        comp.run(T0 + np.timedelta64(6, "h"))
