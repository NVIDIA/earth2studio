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

from earth2studio.nvcoupler._time import (
    as_timedelta,
    fmt_timedelta,
    gcd_timedelta,
    is_multiple,
)


@pytest.mark.parametrize(
    "value,hours",
    [
        ("6h", 6),
        ("2d", 48),
        ("90min", 1.5),
        (np.timedelta64(12, "h"), 12),
        ("-6H", -6),
    ],
)
def test_as_timedelta(value, hours):
    assert as_timedelta(value) == np.timedelta64(int(hours * 60), "m")


@pytest.mark.parametrize("value", ["6", "h", "1M", 6, np.timedelta64(1, "M")])
def test_as_timedelta_rejects(value):
    with pytest.raises((ValueError, TypeError)):
        as_timedelta(value)


@pytest.mark.parametrize(
    "value,text",
    [("48h", "2d"), ("6h", "6h"), ("30min", "30min"), ("0h", "0h")],
)
def test_fmt_timedelta(value, text):
    assert fmt_timedelta(as_timedelta(value)) == text


def test_is_multiple_and_gcd():
    six, forty_eight = as_timedelta("6h"), as_timedelta("48h")
    assert is_multiple(forty_eight, six)
    assert not is_multiple(six, forty_eight)
    assert not is_multiple(six, np.timedelta64(0, "h"))
    assert gcd_timedelta([six, forty_eight, as_timedelta("9h")]) == as_timedelta("3h")
    with pytest.raises(ValueError):
        gcd_timedelta([])
