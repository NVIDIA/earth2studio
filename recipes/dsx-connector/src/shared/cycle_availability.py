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

"""Find the newest available input-data cycle.

Weather data sources publish updates at regular cycle times. These helpers round the current time
down to a cycle boundary, then search backward until the supplied availability check succeeds.
They support whole-hour UTC cycles that repeat daily from 00:00 UTC.

The caller provides the availability check, such as ``HRRR.available`` or ``GFS.available``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone


def floor_to_cycle(time: datetime, step_hours: int) -> datetime:
    """Round a time down to the most recent cycle boundary.

    Parameters
    ----------
    time : datetime
        Time to round down. A datetime without a timezone is treated as UTC.
    step_hours : int
        Number of hours between cycles, such as 6 for GFS or 1 for HRRR. It must be a
        positive integer that divides evenly into 24.

    Returns
    -------
    datetime
        Most recent cycle time in UTC.

    Raises
    ------
    ValueError
        If ``step_hours`` is not a positive integer that divides evenly into 24.
    """
    if (
        isinstance(step_hours, bool)
        or not isinstance(step_hours, int)
        or step_hours <= 0
        or 24 % step_hours != 0
    ):
        raise ValueError(
            "step_hours must be a positive integer that divides evenly into 24"
        )

    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    time = time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return time - timedelta(hours=time.hour % step_hours)


def resolve_latest(
    available: Callable[[datetime], bool],
    now: datetime,
    step_hours: int,
    max_lookback_hours: int,
) -> datetime | None:
    """Find the newest available cycle within a lookback window.

    The search starts at the most recent cycle boundary and moves backward one cycle at a time
    until ``available`` returns ``True`` or the lookback window is exhausted.

    Parameters
    ----------
    available : Callable[[datetime], bool]
        Function that returns whether data exists for a cycle time, such as ``HRRR.available``.
    now : datetime
        Current time. A datetime without a timezone is treated as UTC.
    step_hours : int
        Number of hours between cycles. It must be a positive integer that divides evenly into 24.
    max_lookback_hours : int
        Maximum number of hours to search backward from the most recent cycle boundary. Zero checks
        only that cycle.

    Returns
    -------
    datetime | None
        Newest available cycle, or ``None`` if no cycle is available within the window.

    Raises
    ------
    ValueError
        If ``step_hours`` is invalid or ``max_lookback_hours`` is not a non-negative integer.
    """
    if (
        isinstance(max_lookback_hours, bool)
        or not isinstance(max_lookback_hours, int)
        or max_lookback_hours < 0
    ):
        raise ValueError("max_lookback_hours must be a non-negative integer")

    candidate = floor_to_cycle(now, step_hours)
    earliest_allowed = candidate - timedelta(hours=max_lookback_hours)
    while candidate >= earliest_allowed:
        if available(candidate):
            return candidate
        candidate -= timedelta(hours=step_hours)
    return None
