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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest

from earth2studio.data import AIFS, AIFS_ENS, IFS, IFS_ENS


def now6h():
    """Get closest 6 hour timestamp"""
    nt = datetime.now()
    delta_hr = nt.hour % 6
    return datetime(nt.year, nt.month, nt.day, nt.hour - delta_hr)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=0),
        [
            timedelta(hours=3),
            timedelta(hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_fetch(time, lead_time, variable):
    ds = IFS(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=0),
        [
            timedelta(hours=3),
            timedelta(hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_ens_fetch(time, lead_time, variable):
    members = [1, 2, 3]
    ds = IFS_ENS(cache=False, members=members)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == len(members)
    assert shape[4] == 721
    assert shape[5] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([now6h() - timedelta(days=2)], dtype="datetime64"),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", [["u10m", "tcwv"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ifs_cache(time, lead_time, variable, cache):
    ds = IFS(cache=cache)
    data = ds(time, lead_time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    data = ds(time, lead_time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [IFS, IFS_ENS])
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_time_available(data_source, time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = data_source(source="ecmwf")
        ds(time, lead_time, variable)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [IFS, IFS_ENS])
@pytest.mark.parametrize("time", [now6h() - timedelta(days=2)])
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=1),
        timedelta(hours=147),
        timedelta(hours=366),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_leadtime_available(data_source, time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = data_source()
        ds(time, lead_time, variable)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [AIFS, AIFS_ENS])
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", ["msl"])
def test_aifs_time_available(data_source, time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = data_source(source="ecmwf")
        ds(time, lead_time, variable)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [AIFS, AIFS_ENS])
@pytest.mark.parametrize("time", [now6h() - timedelta(days=2)])
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=3),
        timedelta(hours=147),
        timedelta(hours=366),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_aifs_leadtime_available(data_source, time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = data_source()
        ds(time, lead_time, variable)
