# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import datetime
import pathlib
import shutil

import numpy as np
import pytest

from earth2studio.data import ARCO


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1959, month=1, day=31),
        [
            datetime.datetime(year=1971, month=6, day=1, hour=6),
            datetime.datetime(year=2021, month=11, day=23, hour=12),
        ],
        np.array([np.datetime64("1993-04-05T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["u500", "u200"]])
def test_arco_fetch(time, variable):

    ds = ARCO(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()
    # assert ARCO.available(time[0])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1993-04-05T00:00")])],
)
@pytest.mark.parametrize("variable", [["z500", "q200"]])
@pytest.mark.parametrize("cache", [True, False])
def test_arco_cache(time, variable, cache):

    ds = ARCO(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime(year=2024, month=1, day=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_arco_available(time, variable):
    assert not ARCO.available(time)
    with pytest.raises(ValueError):
        ds = ARCO()
        ds(time, variable)
