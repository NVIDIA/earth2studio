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

"""Connector declarations and the valid-time frame buffer."""

import numpy as np
import pytest
import xarray as xr

from earth2studio.coupling import Connector, CouplingError
from earth2studio.coupling._buffer import _Buffer
from earth2studio.coupling.connector import _split_label

HOUR = 3_600_000_000_000


def test_connector_is_a_frozen_declaration() -> None:
    names = {"tp": "tp06"}
    connector = Connector("atmos", "soil", 1, variables=names, time="hold")
    names["t2m"] = "t2m"
    assert connector.variables == {"tp": "tp06"}
    assert connector.source_variable("tp") == "tp06"
    assert connector.source_variable("t2m") == "t2m"
    assert str(connector) == "atmos -> soil[1] (hold)"
    with pytest.raises(AttributeError):
        connector.slot = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"src": "a", "dst": "a"}, "itself"),
        ({"src": "", "dst": "a"}, "component names"),
        ({"src": "a", "dst": "b", "slot": -1}, "non-negative"),
        ({"src": "a", "dst": "b", "time": "interpolate"}, "'exact' or 'hold'"),
        ({"src": "a", "dst": "b", "regridder": object()}, "Regridder"),
    ],
)
def test_connector_rejects_invalid_declarations(kwargs: dict, match: str) -> None:
    with pytest.raises(CouplingError, match=match):
        Connector(**kwargs)


def test_coupling_error_is_a_value_error() -> None:
    assert issubclass(CouplingError, ValueError)


def test_split_label() -> None:
    assert _split_label("tp:sum:24h") == ("tp", "sum:24h")
    assert _split_label("tp:mean:-6h:0h") == ("tp", "mean:-6h:0h")
    assert _split_label("t2m") == ("t2m", None)


def _frame(lead: int, value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((1, 1, 2), value),
        dims=("lead_time", "variable", "lat"),
        coords={
            "lead_time": [np.timedelta64(lead, "ns")],
            "variable": ["z"],
            "lat": [0, 1],
        },
    )


def test_buffer_selects_frames_by_valid_time() -> None:
    buffer = _Buffer()
    for k in range(4):
        buffer.add(k * 6 * HOUR, frozenset({"z"}), k, _frame(k * 6 * HOUR, k))
    assert buffer.has("z", 12 * HOUR)
    assert not buffer.has("z", 3 * HOUR)
    assert not buffer.has("q", 12 * HOUR)
    np.testing.assert_array_equal(buffer.get("z", 12 * HOUR).values, 2)
    with pytest.raises(KeyError):
        buffer.get("z", 3 * HOUR)


def test_buffer_hold_uses_frames_published_on_earlier_steps() -> None:
    buffer = _Buffer()
    buffer.add(0, frozenset({"z"}), 0)
    buffer.add(6 * HOUR, frozenset({"z"}), 1)
    assert buffer.latest("z", 9 * HOUR, before=2) == 6 * HOUR
    assert buffer.latest("z", 9 * HOUR, before=1) == 0
    assert buffer.latest("z", -HOUR, before=2) is None
    assert buffer.latest("q", 9 * HOUR, before=2) is None


def test_buffer_prune_keeps_each_variables_latest_frame() -> None:
    buffer = _Buffer()
    buffer.add(0, frozenset({"z", "sst"}), 0)
    for k in range(1, 4):
        buffer.add(k * 6 * HOUR, frozenset({"z"}), k)
    buffer.prune(12 * HOUR)
    assert not buffer.has("z", 6 * HOUR)
    assert buffer.has("z", 12 * HOUR) and buffer.has("z", 18 * HOUR)
    assert buffer.latest("sst", 18 * HOUR, before=9) == 0
