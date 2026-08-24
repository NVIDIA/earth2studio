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

from datetime import datetime, timedelta

import numpy as np
import pytest

from earth2studio.utils.time import (
    leadtimearray_to_timedelta,
    normalize_time_precision,
    timearray_to_datetime,
    to_time_array,
)


@pytest.mark.parametrize(
    "time",
    [
        [datetime.fromisoformat("1993-04-05T00:00")],
        [
            datetime.fromisoformat("1999-10-11T12:00"),
            datetime.fromisoformat("2001-06-04T00:00"),
        ],
    ],
)
def test_to_datetime(time):
    array = np.array(time, dtype=np.datetime64)
    time_np = timearray_to_datetime(array)
    assert all(t1 == t2 for t1, t2 in zip(time, time_np))


@pytest.mark.parametrize(
    "timedelta",
    [
        [timedelta(hours=1, minutes=2, seconds=3)],
        [
            timedelta(weeks=1, hours=2, microseconds=3),
            timedelta(days=1, seconds=2, milliseconds=3),
        ],
    ],
)
def test_to_timedelta(timedelta):
    array = np.array(timedelta, dtype="timedelta64[us]")
    timedelta_np = leadtimearray_to_timedelta(array)
    assert all(t1 == t2 for t1, t2 in zip(timedelta, timedelta_np))


@pytest.mark.parametrize(
    "time",
    [
        [
            np.datetime64("1999-10-11"),
            "2001-06-04T00:00",
            datetime.fromisoformat("2001-02-27"),
        ],
        ["1999-10-11", datetime.fromisoformat("2001-06-04T00:00"), "2001-02-27"],
    ],
)
def test_to_timearray(time):
    target = np.array(["1999-10-11", "2001-06-04", "2001-02-27"], dtype=np.datetime64)
    out = to_time_array(time)
    assert all(t1 == t2 for t1, t2 in zip(target, out))


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("2024-01-01T00:00:00", "s")],
        # Sub-second resolution must survive the cast
        [np.datetime64("2024-01-01T00:00:00.123456", "us")],
        # Just inside the datetime64[ns] range
        [np.datetime64("1700-01-01T00:00:00", "s")],
        [np.datetime64("2200-01-01T00:00:00", "s")],
    ],
)
def test_normalize_time_precision_representable(time):
    array = np.array(time)
    out = normalize_time_precision(array)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert all(t1 == t2 for t1, t2 in zip(array, out))


@pytest.mark.parametrize(
    "time",
    [
        # CM4 model years used by SamudrACE initial conditions
        [np.datetime64("0151-01-06T00:00:00", "s")],
        [np.datetime64("0311-01-01T00:00:00", "s")],
        # Beyond the upper datetime64[ns] bound
        [np.datetime64("2300-01-01T00:00:00", "s")],
        # A mixed array falls back for every entry, not just the offender
        [
            np.datetime64("2024-01-01T00:00:00", "s"),
            np.datetime64("0311-01-01T00:00:00", "s"),
        ],
    ],
)
def test_normalize_time_precision_out_of_range(time):
    array = np.array(time)
    out = normalize_time_precision(array)
    # The array keeps its own precision rather than silently wrapping
    assert out.dtype == array.dtype
    assert all(t1 == t2 for t1, t2 in zip(array, out))


def test_normalize_time_precision_empty():
    out = normalize_time_precision(np.array([], dtype="datetime64[s]"))
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out.size == 0


def test_to_timearray_out_of_ns_range():
    """CM4 model-year timestamps must not wrap to unrelated dates."""
    time = [np.datetime64("0311-01-01T00:00:00", "s")]
    out = to_time_array(time)
    assert out[0] == time[0]
    assert out[0].astype("datetime64[s]").item().year == 311
