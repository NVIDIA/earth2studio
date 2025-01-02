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

import pathlib
import shutil
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import NCAR_ERA5


@pytest.mark.slow
@pytest.mark.xfail
# @pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2022, month=12, day=25),
        [
            datetime(year=2022, month=1, day=1, hour=6),
            datetime(year=2023, month=2, day=3, hour=18),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["msl"]])
def test_ncar_fetch(time, variable):

    ds = NCAR_ERA5(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
# @pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "msl"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ncar_cache(time, variable, cache):

    ds = NCAR_ERA5(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    if cache:
        assert pathlib.Path(ds.cache).is_dir()
    else:
        assert not pathlib.Path(ds.cache).is_dir()

    # Load from cache or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    if cache:
        shutil.rmtree(ds.cache)


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2021, month=2, day=16, hour=0, minute=3),
        datetime(year=2023, month=1, day=1, hour=13, second=45),
    ],
)
@pytest.mark.parametrize("variable", ["not_available"])
def test_ncar_valid_time(time, variable):
    with pytest.raises(ValueError):
        ds = NCAR_ERA5()
        ds(time, variable)
