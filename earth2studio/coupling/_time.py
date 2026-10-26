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

"""Timedelta helpers shared by the coupler modules."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Literal, TypeAlias

import numpy as np

DeltaLike: TypeAlias = str | np.timedelta64

_DURATION = re.compile(r"^(-?\d+)\s*([a-zA-Z]+)$")
_UNITS: dict[str, Literal["D", "h", "m", "s"]] = {
    "d": "D",
    "day": "D",
    "days": "D",
    "h": "h",
    "H": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    "m": "m",
    "min": "m",
    "minute": "m",
    "minutes": "m",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
}

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_MINUTE = 60_000_000_000


def as_timedelta(value: DeltaLike) -> np.timedelta64:
    """Coerce a duration string such as ``"6h"`` or a timedelta to nanoseconds."""
    if isinstance(value, np.timedelta64):
        if np.datetime_data(value.dtype)[0] in {"M", "Y"}:
            raise ValueError("Calendar-relative durations are not supported")
        return value.astype("timedelta64[ns]")
    if isinstance(value, str):
        match = _DURATION.match(value.strip())
        # Units are case-sensitive so 'M' (month) is never read as minutes.
        if match is None or match.group(2) not in _UNITS:
            raise ValueError(
                f"Cannot parse duration {value!r}; expected e.g. '6h', '30min', '2d'"
            )
        magnitude, unit = match.groups()
        return np.timedelta64(int(magnitude), _UNITS[unit]).astype("timedelta64[ns]")
    raise TypeError(
        f"Expected a duration string or numpy.timedelta64, got {type(value).__name__}"
    )


def nanoseconds(value: np.timedelta64) -> int:
    """Return a timedelta as integer nanoseconds."""
    return int(np.timedelta64(value).astype("timedelta64[ns]").astype(np.int64))


def fmt_timedelta(value: np.timedelta64) -> str:
    """Format a timedelta as whole days, hours or minutes where possible."""
    ns = nanoseconds(value)
    if ns and ns % (24 * _NS_PER_HOUR) == 0:
        return f"{ns // (24 * _NS_PER_HOUR)}d"
    if ns % _NS_PER_HOUR == 0:
        return f"{ns // _NS_PER_HOUR}h"
    if ns % _NS_PER_MINUTE == 0:
        return f"{ns // _NS_PER_MINUTE}min"
    return str(value)


def is_multiple(interval: np.timedelta64, step: np.timedelta64) -> bool:
    """Whether ``interval`` is a positive whole multiple of ``step``."""
    interval_ns, step_ns = nanoseconds(interval), nanoseconds(step)
    return step_ns > 0 and interval_ns > 0 and interval_ns % step_ns == 0


def gcd_timedelta(values: Iterable[np.timedelta64]) -> np.timedelta64:
    """Greatest common divisor of positive timedeltas."""
    ns = [nanoseconds(value) for value in values]
    if not ns or any(value <= 0 for value in ns):
        raise ValueError("Cadences must be positive timedeltas")
    return np.timedelta64(math.gcd(*ns), "ns")
