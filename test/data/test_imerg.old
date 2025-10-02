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

from earth2studio.data import IMERG


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2002, month=10, day=1),
        [
            datetime.datetime(year=2006, month=8, day=8, hour=4, minute=30),
            datetime.datetime(year=2016, month=4, day=20, hour=20),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tp", ["tpi"]])
def test_imerg_fetch(time, variable):

    ds = IMERG(cache=False)
    data = ds(time, variable)

    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 1800
    assert shape[3] == 3600
    assert not np.isnan(data.values).any()
    assert IMERG.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T16:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["tp", "tpp"]])
@pytest.mark.parametrize("cache", [True, False])
def test_imerge_cache(time, variable, cache):

    ds = IMERG(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 1800
    assert shape[3] == 3600
    assert not np.isnan(data.values).any()
    assert IMERG.available(time[0])
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1800
    assert shape[3] == 3600
    assert not np.isnan(data.values).any()
    assert IMERG.available(time[0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2020, month=2, day=1, hour=12, minute=2),
        datetime.datetime(year=2000, month=6, day=1),
        datetime.datetime(year=2050, month=1, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["tp"])
def test_imerg_available(time, variable):
    assert not IMERG.available(time)
    with pytest.raises((ValueError, FileNotFoundError)):
        ds = IMERG()
        ds(time, variable)
