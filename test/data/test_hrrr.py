# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.data import HRRR


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2022, month=12, day=25),
        [
            datetime.datetime(year=2022, month=1, day=1, hour=6),
            datetime.datetime(year=2023, month=2, day=3, hour=18),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["u10m", "u100"]])
def test_hrrr_fetch(time, variable):

    ds = HRRR(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    if isinstance(variable, str):
        variable = [variable]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "sp"]])
@pytest.mark.parametrize("cache", [True, False])
def test_hrrr_cache(time, variable, cache):

    ds = HRRR(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
    assert (data.coords["variable"] == np.array(variable)).all()
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == 1059
    assert shape[3] == 1799
    assert not np.isnan(data.values).any()
    assert HRRR.available(time[0])
    assert (data.coords["variable"] == np.array(variable[0])).all()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2014, month=8, day=4, hour=0),
    ],
)
@pytest.mark.parametrize("variable", ["u100"])
def test_hrrr_available(time, variable):
    assert not HRRR.available(time)
    with pytest.raises(ValueError):
        ds = HRRR()
        ds(time, variable)
