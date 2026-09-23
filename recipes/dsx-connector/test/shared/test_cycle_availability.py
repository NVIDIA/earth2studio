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


"""Shared forecast-cycle scheduling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from src.shared.cycle_availability import floor_to_cycle, resolve_latest


def test_floor_to_cycle() -> None:
    t = datetime(2026, 7, 7, 14, 37, 12, tzinfo=timezone.utc)
    assert floor_to_cycle(t, 6) == datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
    assert floor_to_cycle(t, 1) == datetime(2026, 7, 7, 14, 0, tzinfo=timezone.utc)
    # Naive datetimes are treated as UTC.
    naive = datetime(2026, 7, 7, 3, 5)
    assert floor_to_cycle(naive, 6) == datetime(2026, 7, 7, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("step_hours", [0, 5, 1.0, True])
def test_floor_to_cycle_rejects_invalid_step_hours(step_hours: object) -> None:
    with pytest.raises(
        ValueError,
        match="step_hours must be a positive integer that divides evenly into 24",
    ):
        floor_to_cycle(datetime(2026, 7, 7, tzinfo=timezone.utc), step_hours)  # type: ignore[arg-type]


def test_resolve_latest_walks_back_to_available_cycle() -> None:
    # GFS 18Z not out yet at 20:45 -> resolver returns the latest available cycle (12Z).
    now = datetime(2026, 7, 7, 20, 45, tzinfo=timezone.utc)
    available = {datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)}
    got = resolve_latest(lambda t: t in available, now, 6, 12)
    assert got == datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def test_resolve_latest_hourly_grid() -> None:
    now = datetime(2026, 7, 7, 14, 20, tzinfo=timezone.utc)
    seen: list[datetime] = []

    def available(t: datetime) -> bool:
        seen.append(t)
        return t.hour == 13  # 14Z HRRR not ready, 13Z is

    got = resolve_latest(available, now, 1, 6)
    assert got == datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
    assert seen[0] == datetime(
        2026, 7, 7, 14, 0, tzinfo=timezone.utc
    )  # tried newest first


def test_resolve_latest_returns_none_on_outage() -> None:
    now = datetime(2026, 7, 7, 20, 45, tzinfo=timezone.utc)
    assert resolve_latest(lambda t: False, now, 6, 12) is None


def test_resolve_latest_zero_lookback_checks_only_current_cycle() -> None:
    now = datetime(2026, 7, 7, 14, 20, tzinfo=timezone.utc)
    seen: list[datetime] = []

    def available(candidate: datetime) -> bool:
        seen.append(candidate)
        return True

    assert resolve_latest(available, now, 1, 0) == datetime(
        2026, 7, 7, 14, 0, tzinfo=timezone.utc
    )
    assert seen == [datetime(2026, 7, 7, 14, 0, tzinfo=timezone.utc)]


@pytest.mark.parametrize("max_lookback_hours", [-1, 1.5, True])
def test_resolve_latest_rejects_invalid_lookback(max_lookback_hours: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_lookback_hours must be a non-negative integer",
    ):
        resolve_latest(
            lambda candidate: True,
            datetime(2026, 7, 7, tzinfo=timezone.utc),
            6,
            max_lookback_hours,  # type: ignore[arg-type]
        )
