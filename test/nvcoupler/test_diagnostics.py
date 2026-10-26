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

from earth2studio.nvcoupler import (
    Connector,
    Coupler,
    CouplingError,
    RunSequence,
    Slot,
)
from earth2studio.nvcoupler.testing import (
    ATMOS_SHAPE,
    ToyAtmosphere,
    ToyImpact,
    toy_grid,
    toy_source,
)
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array_like

T0 = np.datetime64("2026-01-01T00")
GRID = toy_grid(*ATMOS_SHAPE)


class _DailyMax(ToyImpact):
    def input_coords(self):
        return coord_array_like(
            super().input_coords(), {"variable": np.array(["z1000:max:24h"])}
        )

    def output_coords(self, input_coords):
        return coord_array_like(input_coords, {"variable": np.array(["zmax"])})


def _value(step: dict, stream: str) -> float:
    return float(step[stream].isel(lat=1, lon=1).values.ravel()[0])


def test_windowed_diagnostic_skips_steps_until_its_window_is_complete():
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "daily": _DailyMax()},
        [Connector("atmos", "daily")],
        {"atmos": (toy_source(GRID, 0.0), toy_source(GRID, 2.0))},
    )
    steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(36, "h"))))
    present = [lead for lead, step in zip(range(0, 37, 6), steps) if "daily" in step]
    assert present == [24, 30, 36]
    # Maximum over [T - 24h, T) of a growing field is the frame at T - 6h.
    np.testing.assert_allclose(_value(steps[4], "daily"), 0.01 * 1.2 * 3)
    np.testing.assert_allclose(_value(steps[6], "daily"), 0.01 * 1.2 * 5)


def test_windowed_diagnostic_needs_no_same_step_frame():
    coupler = Coupler(
        {"daily": _DailyMax(), "atmos": ToyAtmosphere()},
        [Connector("atmos", "daily")],
        {"atmos": (toy_source(GRID, 0.0), toy_source(GRID, 2.0))},
    )
    # The window ends at the current lead, so declaration order is kept.
    assert coupler.sequence.slots[0].models == ("daily", "atmos")
    steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(30, "h"))))
    assert [lead for lead, step in zip(range(0, 31, 6), steps) if "daily" in step] == [
        24,
        30,
    ]


def test_diagnostic_gap_after_warmup_is_an_error():
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "wind": ToyImpact()},
        [Connector("atmos", "wind")],
        {"atmos": (toy_source(GRID, 0.0), toy_source(GRID, 2.0))},
    )
    # Bypass validation: the diagnostic now runs before its source each step.
    coupler.sequence = RunSequence((Slot("6h", ("wind", "atmos")),))
    with pytest.raises(CouplingError, match="advance 'atmos' first"):
        list(coupler.run_item(WorkItem(T0, np.timedelta64(12, "h"))))
