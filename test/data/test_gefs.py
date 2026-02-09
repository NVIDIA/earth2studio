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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest

from earth2studio.data import GEFS_FX, GEFS_FX_721x1440


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=3), timedelta(hours=384)],
            "t2m",
        ),
        (
            [
                datetime(year=2021, month=8, day=8, hour=6),
                datetime(year=2022, month=4, day=20, hour=12),
            ],
            timedelta(hours=0),
            ["msl"],
        ),
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=240)],
            np.array(["tcwv", "u10m"]),
        ),
    ],
)
def test_gefs_0p50_fetch(time, lead_time, variable):

    ds = GEFS_FX(cache=False)
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
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (
            datetime(year=2020, month=11, day=1),
            [timedelta(hours=0), timedelta(hours=240)],
            "t2m",
        ),
    ],
)
def test_gefs_0p25_fetch(time, lead_time, variable):

    ds = GEFS_FX_721x1440(cache=False)
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
    assert GEFS_FX_721x1440.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize("product", ["gec00", "gep01", "gep30"])
def test_gefs_products(product):
    time = datetime(year=2022, month=12, day=25)
    lead_time = timedelta(hours=3)
    variable = "u100m"

    ds = GEFS_FX(product, cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array([variable]))


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
def test_gefs_cache(time, variable, cache):

    lead_time = np.array([np.timedelta64(3, "h")])

    ds = GEFS_FX(cache=cache)
    data = ds(time, lead_time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 2
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, lead_time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1
    assert shape[3] == 361
    assert shape[4] == 720
    assert not np.isnan(data.values).any()
    assert GEFS_FX.available(time[0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2020, month=9, day=22),
        datetime.now(),
    ],
)
def test_gefs_available(time):
    variable = ["mpl"]
    lead_time = timedelta(hours=0)
    assert not GEFS_FX.available(time)
    with pytest.raises(ValueError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(243, "h")]),
        np.array([np.timedelta64(390, "h")]),
    ],
)
def test_gefs_invalid_lead(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(241, "h")]),
    ],
)
def test_gefs_0p25_invalid_lead(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = GEFS_FX_721x1440(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "variable",
    ["aaa", "t1m"],
)
def test_gefs_invalid_variable(variable):
    time = datetime(year=2022, month=12, day=25)
    lead_time = timedelta(hours=0)
    with pytest.raises(KeyError):
        ds = GEFS_FX(cache=False)
        ds(time, lead_time, variable)


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "product",
    ["gec0", "gep31", "gep00"],
)
def test_gefs_invalid_product(product):
    with pytest.raises(ValueError):
        GEFS_FX(product, cache=False)
