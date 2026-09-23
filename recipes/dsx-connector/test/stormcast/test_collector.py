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

"""Direct tests for the model-specific StormCastCollector (src/stormcast/collector.py).

Exercises the derivation (regroup source vars -> ForecastSeries) at the collector level,
asserting on ForecastSeries rather than wire payloads. No GPU, no model, no bus.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.stormcast.collector import StormCastCollector


class _FakeExtractor:
    def __init__(self, site_lon: dict[str, float]) -> None:
        self.site_lon = site_lon

    def extract(self, field: np.ndarray) -> dict[str, float]:
        v = float(np.asarray(field).ravel()[0])
        return {sid: v for sid in self.site_lon}


def _collector(site_lon_deg: float = 262.5) -> StormCastCollector:
    return StormCastCollector(
        _FakeExtractor({"s": site_lon_deg}), "test-model", init_ms=1000
    )


def test_collect_produces_forecast_series() -> None:
    c = _collector()
    c._buffer[("s", 0)] = {
        "t2m": 300.0,
        "u10m": 3.0,
        "v10m": 4.0,
        "q1hl": 0.01,
        "p1hl": 96000.0,
    }
    c._buffer[("s", 1)] = {"t2m": 295.0}  # temperature only at lead 1
    series = {fs.variable: fs for fs in c.collect()}

    assert set(c._buffer) == {("s", 0), ("s", 1)}
    assert set(series) == {
        "Temperature",
        "WindU",
        "WindV",
        "RelativeHumidity",
        "WetBulb",
    }
    temp = series["Temperature"]
    assert temp.site_id == "s"
    assert temp.model == "test-model"
    assert temp.init_ms == 1000
    assert temp.lead_seconds == [0, 1]
    assert temp.member_values == [[300.0, 295.0]]
    assert series["WindU"].member_values == [[3.0]]
    assert series["WindV"].member_values == [[4.0]]
    wetbulb = series["WetBulb"]
    assert wetbulb.lead_seconds == [0]
    assert 273.15 < wetbulb.member_values[0][0] < 300.0  # Kelvin, <= dry-bulb


def test_write_rejects_field_name_count_mismatch() -> None:
    c = _collector()
    coords = {"lead_time": np.array([np.timedelta64(0, "h")])}

    with pytest.raises(ValueError, match=r"2 field\(s\).*1 variable name"):
        c.write(
            [np.ones((1, 1)), np.ones((1, 1))],
            coords,
            np.array(["t2m"]),
        )


def test_write_rejects_duplicate_variable() -> None:
    c = _collector()
    coords = {"lead_time": np.array([np.timedelta64(0, "h")])}
    c.write(np.ones((1, 1)), coords, "t2m")

    with pytest.raises(ValueError, match="duplicate write"):
        c.write(np.ones((1, 1)), coords, "t2m")


def test_write_filters_non_source_variables() -> None:
    c = _collector()
    coords = {"lead_time": np.timedelta64(0, "h")}

    c.write(
        [np.array([[300.0]]), np.array([[1.0]])],
        coords,
        np.array(["t2m", "garbage"]),
    )

    assert c._buffer[("s", 0)] == {"t2m": 300.0}


def test_to_yx_squeezes_singleton_leading_dimensions() -> None:
    field = np.arange(12.0).reshape(1, 1, 3, 4)

    result = StormCastCollector._to_yx(field)

    assert result.shape == (3, 4)
    assert result[0, 0] == 0.0


def test_to_yx_rejects_nonsingleton_leading_dimensions() -> None:
    with pytest.raises(ValueError, match="singleton"):
        StormCastCollector._to_yx(np.zeros((2, 3, 4)))


def test_to_yx_rejects_field_with_fewer_than_two_dimensions() -> None:
    with pytest.raises(ValueError, match="at least two dimensions"):
        StormCastCollector._to_yx(np.ones((2,)))


def test_begin_cycle_resets_buffer_and_init() -> None:
    c = _collector()
    c._buffer[("s", 1)] = {"t2m": 290.0}

    c.begin_cycle(2000)

    assert c.init_ms == 2000
    assert not c._buffer
