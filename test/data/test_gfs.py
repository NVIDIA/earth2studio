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

from earth2studio.data import GFS, GFS_FX


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
@pytest.mark.parametrize("variable", ["t2m", ["msl"]])
def test_gfs_fetch(time, variable):

    ds = GFS(cache=False)
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
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "lead_time",
    [
        datetime.timedelta(hours=1),
        [datetime.timedelta(hours=2), datetime.timedelta(hours=3)],
        np.array([np.timedelta64(0, "h")]),
    ],
)
def test_gfs_fx_fetch(lead_time):
    time = datetime.datetime(year=2022, month=12, day=25)
    variable = "t2m"
    ds = GFS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, datetime.timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime.datetime):
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
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "msl"]])
@pytest.mark.parametrize("cache", [True, False])
def test_gfs_cache(time, variable, cache):

    ds = GFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert GFS.available(time[0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2021, month=2, day=16),
        datetime.datetime(year=2023, month=1, day=1, hour=13),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_gfs_available(time, variable):
    assert not GFS.available(time)
    with pytest.raises(ValueError):
        ds = GFS()
        ds(time, variable)


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "lead_time",
    [
        datetime.timedelta(hours=-1),
        [datetime.timedelta(hours=2), datetime.timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(385, "h")]),
    ],
)
def test_gfs_fx_available(lead_time):
    time = datetime.datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GFS_FX()
        ds(time, lead_time, variable)
