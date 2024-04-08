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

from earth2studio.data import IFS


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2023, month=1, day=31),
        [
            datetime.datetime(year=2023, month=10, day=15, hour=18),
            datetime.datetime(year=2023, month=12, day=1, hour=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_fetch(time, variable):

    ds = IFS(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 451
    assert shape[3] == 900
    assert not np.isnan(data.values).any()
    assert IFS.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["u10m", "tcwv"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ifs_cache(time, variable, cache):

    ds = IFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert shape[2] == 451
    assert shape[3] == 900
    assert not np.isnan(data.values).any()
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 451
    assert shape[3] == 900
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2023, month=1, day=17),
        datetime.datetime(year=2023, month=2, day=2, hour=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_ifs_available(time, variable):
    assert not IFS.available(time)
    with pytest.raises(ValueError):
        ds = IFS()
        ds(time, variable)
