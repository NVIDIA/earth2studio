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
