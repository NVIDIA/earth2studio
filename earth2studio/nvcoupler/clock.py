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

"""Clock and Alarm: coupled-system time management (ESMF analogs).

The Driver owns one Clock stepping at the coupling interval dt; every
Component owns an Alarm at its own cadence. A component runs only on
driver steps where its alarm rings, which is how a 6 h atmosphere and a
48 h ocean coexist in one loop.

All times are np.datetime64 / np.timedelta64 to match earth2studio's
TimeArray / LeadTimeArray conventions.
"""

from collections.abc import Iterator

import numpy as np

from .errors import CadenceError

TimeLike = str | np.datetime64
DeltaLike = str | np.timedelta64


def as_datetime(t: TimeLike) -> np.datetime64:
    """Coerce to nanosecond datetime64 (accepts ISO strings)."""
    return np.datetime64(t).astype("datetime64[ns]")


def as_timedelta(d: DeltaLike) -> np.timedelta64:
    """Coerce to nanosecond timedelta64. Strings use '<n><unit>' (e.g. '6h')."""
    if isinstance(d, np.timedelta64):
        return d.astype("timedelta64[ns]")
    if isinstance(d, (bool, int, np.integer)):
        raise ValueError(
            f"Bare number {d!r} is ambiguous as a timedelta (hours? steps?) — "
            "pass a string like '6h' or '2D', or a np.timedelta64"
        )
    if isinstance(d, str):
        s = d.strip()
        i = 0
        while i < len(s) and (s[i].isdigit() or s[i] == "-"):
            i += 1
        if i == 0 or i == len(s):
            raise ValueError(f"Cannot parse timedelta {d!r}; expected e.g. '6h', '2D'")
        value, unit = int(s[:i]), s[i:]
        unit = {"d": "D", "H": "h", "min": "m", "S": "s"}.get(unit, unit)
        return np.timedelta64(value, unit).astype("timedelta64[ns]")
    raise TypeError(f"Cannot interpret {d!r} as a timedelta")


def fmt_timedelta(d: np.timedelta64) -> str:
    """Human-readable timedelta: whole hours/days where possible."""
    ns = d.astype("timedelta64[ns]").astype(np.int64)
    hour = 3_600_000_000_000
    if ns % (24 * hour) == 0:
        return f"{ns // (24 * hour)}D"
    if ns % hour == 0:
        return f"{ns // hour}h"
    if ns % 60_000_000_000 == 0:
        return f"{ns // 60_000_000_000}m"
    return str(d)


def is_multiple(interval: np.timedelta64, dt: np.timedelta64) -> bool:
    interval_ns = interval.astype("timedelta64[ns]").astype(np.int64)
    dt_ns = dt.astype("timedelta64[ns]").astype(np.int64)
    return dt_ns > 0 and interval_ns > 0 and interval_ns % dt_ns == 0


class Alarm:
    """Rings when (time - start - offset) is a whole multiple of interval.

    NUOPC-alarm analog used for per-component cadences: an interval of 48 h
    on a 6 h driver clock rings every 8th step.
    """

    def __init__(self, interval: DeltaLike, offset: DeltaLike | None = None):
        self.interval = as_timedelta(interval)
        self.offset = (
            as_timedelta(offset) if offset is not None else np.timedelta64(0, "ns")
        )
        if self.interval <= np.timedelta64(0, "ns"):
            raise ValueError(f"Alarm interval must be positive, got {interval!r}")

    def is_ringing(self, time: np.datetime64, start: np.datetime64) -> bool:
        elapsed = (as_datetime(time) - as_datetime(start) - self.offset).astype(
            "timedelta64[ns]"
        )
        elapsed_ns = elapsed.astype(np.int64)
        interval_ns = self.interval.astype(np.int64)
        return elapsed_ns >= 0 and elapsed_ns % interval_ns == 0

    def __repr__(self) -> str:
        return f"Alarm(interval={self.interval}, offset={self.offset})"


class Clock:
    """Driver clock stepping from start to stop (inclusive) at dt.

    Iterating yields each time after the start: the initial condition is at
    ``start`` and is the caller's step 0; the first yielded time is
    ``start + dt``. This mirrors run.py where the iterator's 0th output is
    the IC and each subsequent step advances one dt.
    """

    def __init__(self, start: TimeLike, stop: TimeLike, dt: DeltaLike):
        self.start = as_datetime(start)
        self.stop = as_datetime(stop)
        self.dt = as_timedelta(dt)
        if self.dt <= np.timedelta64(0, "ns"):
            raise ValueError(f"Clock dt must be positive, got {dt!r}")
        if self.stop <= self.start:
            raise ValueError(f"Clock stop {stop!r} must be after start {start!r}")
        span = (self.stop - self.start).astype("timedelta64[ns]")
        if not is_multiple(span, self.dt):
            raise CadenceError("Clock span (stop - start)", str(span), str(self.dt))
        self._step = 0

    @property
    def current(self) -> np.datetime64:
        return self.start + self._step * self.dt

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def n_steps(self) -> int:
        span_ns = (self.stop - self.start).astype("timedelta64[ns]").astype(np.int64)
        return int(span_ns // self.dt.astype(np.int64))

    def elapsed(self) -> np.timedelta64:
        return self.current - self.start

    def done(self) -> bool:
        return self.current >= self.stop

    def advance(self) -> np.datetime64:
        if self.done():
            raise StopIteration(f"Clock already at stop time {self.stop}")
        self._step += 1
        return self.current

    def reset(self) -> None:
        self._step = 0

    def times(self) -> np.ndarray:
        """All times from start to stop inclusive (length n_steps + 1)."""
        return self.start + np.arange(self.n_steps + 1) * self.dt

    def __iter__(self) -> Iterator[np.datetime64]:
        while not self.done():
            yield self.advance()

    def __repr__(self) -> str:
        return (
            f"Clock({np.datetime_as_string(self.start, unit='m')} -> "
            f"{np.datetime_as_string(self.stop, unit='m')}, "
            f"dt={fmt_timedelta(self.dt)}, step={self._step})"
        )
