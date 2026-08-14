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

from earth2studio.nvcoupler.clock import Alarm, Clock, as_timedelta, is_multiple
from earth2studio.nvcoupler.errors import CadenceError


def test_as_timedelta_parsing():
    assert as_timedelta("6h") == np.timedelta64(6, "h")
    assert as_timedelta("2D") == np.timedelta64(2, "D")
    assert as_timedelta("2d") == np.timedelta64(2, "D")
    assert as_timedelta(np.timedelta64(30, "m")) == np.timedelta64(30, "m")
    with pytest.raises(ValueError):
        as_timedelta("h6")


def test_bare_int_timedelta_rejected():
    # finding 2e: a unit-less number is ambiguous (hours? steps?) — reject it
    # with an actionable message instead of a TypeError deep in numpy
    with pytest.raises(ValueError, match="'6h'"):
        as_timedelta(6)
    with pytest.raises(ValueError, match="np.timedelta64"):
        Clock("2024-01-01", "2024-01-02", 6)
    with pytest.raises(ValueError, match="'6h'"):
        Alarm(24)


def test_clock_iteration():
    clock = Clock("2024-01-01", "2024-01-05", "6h")
    assert clock.n_steps == 16
    times = list(clock)
    assert len(times) == 16
    assert times[0] == np.datetime64("2024-01-01T06:00")
    assert times[-1] == np.datetime64("2024-01-05T00:00")
    assert clock.done()
    with pytest.raises(StopIteration):
        clock.advance()
    clock.reset()
    assert clock.current == np.datetime64("2024-01-01")
    assert len(clock.times()) == 17  # includes start


def test_clock_validation():
    with pytest.raises(CadenceError):
        Clock("2024-01-01", "2024-01-02T01:00", "6h")  # span not multiple of dt
    with pytest.raises(ValueError):
        Clock("2024-01-02", "2024-01-01", "6h")  # stop before start


def test_alarm_cadences():
    start = np.datetime64("2024-01-01")
    fast = Alarm("6h")
    slow = Alarm("48h")
    clock = Clock(start, "2024-01-05", "6h")
    fast_rings = sum(fast.is_ringing(t, start) for t in clock)
    clock.reset()
    slow_rings = sum(slow.is_ringing(t, start) for t in clock)
    assert fast_rings == 16  # every step over 96h
    assert slow_rings == 2  # at 48h and 96h
    # alarms also ring at t=start (step 0 / initialization)
    assert fast.is_ringing(start, start) and slow.is_ringing(start, start)


def test_alarm_offset():
    start = np.datetime64("2024-01-01")
    offset = Alarm("24h", offset="6h")
    assert not offset.is_ringing(start, start)  # before offset
    assert offset.is_ringing(np.datetime64("2024-01-01T06:00"), start)
    assert offset.is_ringing(np.datetime64("2024-01-02T06:00"), start)
    assert not offset.is_ringing(np.datetime64("2024-01-02T00:00"), start)


def test_is_multiple():
    assert is_multiple(np.timedelta64(48, "h"), np.timedelta64(6, "h"))
    assert not is_multiple(np.timedelta64(7, "h"), np.timedelta64(6, "h"))
