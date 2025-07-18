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

import asyncio
import pathlib
import shutil
from datetime import datetime, timedelta

import gcsfs
import numpy as np
import pytest
import s3fs
from fsspec.implementations.http import HTTPFileSystem

from earth2studio.data import HRRR, HRRR_FX


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
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
@pytest.mark.parametrize("variable", ["t2m", ["u10m", "u100"], ["u1hl"]])
def test_hrrr_fetch(time, variable):

    ds = HRRR(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
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
    "time,lead_time",
    [
        (datetime(year=2022, month=12, day=25), timedelta(hours=1)),
        (
            datetime(year=2022, month=12, day=25),
            [timedelta(hours=0), timedelta(hours=3)],
        ),
        (
            np.array(
                [np.datetime64("2024-01-01T00:00"), np.datetime64("2024-02-01T00:00")]
            ),
            np.array([np.timedelta64(1, "h")]),
        ),
    ],
)
def test_hrrr_fx_fetch(time, lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "tp"
    ds = HRRR_FX(cache=False)
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
    assert shape[3] == 1059
    assert shape[4] == 1799
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.timeout(15)
def test_hrrr_init():
    """Test HRRR initialization with different sources and parameters"""
    # Test AWS source
    ds = HRRR(source="aws", cache=True, verbose=True, async_timeout=300)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(ds._async_init())

    assert ds.uri_prefix == "noaa-hrrr-bdp-pds"
    assert isinstance(ds.fs, s3fs.S3FileSystem)
    assert ds.async_timeout == 300

    # Test Google source
    ds = HRRR(source="google", cache=False, verbose=False)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(ds._async_init())
    assert ds.uri_prefix == "high-resolution-rapid-refresh"
    assert isinstance(ds.fs, gcsfs.GCSFileSystem)

    # Test Nomads source
    ds = HRRR(source="nomads")
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(ds._async_init())
    assert ds.uri_prefix == "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/"
    assert isinstance(ds.fs, HTTPFileSystem)

    # Test invalid source
    with pytest.raises(ValueError):
        HRRR(source="invalid_source")

    # Test Azure source (not implemented)
    with pytest.raises(NotImplementedError):
        HRRR(source="azure")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "sp", "t10hl"]])
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


def test_hrrr_validate_inputs():
    ds = HRRR(cache=False)

    # Test valid time
    valid_time = datetime(year=2022, month=12, day=25, hour=12)
    ds._validate_time([valid_time])  # Should not raise

    # Test invalid hour interval
    invalid_time = datetime(year=2022, month=12, day=25, hour=12, minute=30)
    with pytest.raises(ValueError):
        ds._validate_time([invalid_time])

    # Test time before 2018-07-12 13:00
    old_time = datetime(year=2018, month=7, day=12, hour=12)
    with pytest.raises(ValueError):
        ds._validate_time([old_time])

    # Test invalid variable
    with pytest.raises(KeyError):
        ds(datetime(year=2024, month=12, day=25), "invalid variable")


@pytest.mark.timeout(15)
def test_hrrr_fx_validate_leadtime():
    ds = HRRR_FX(cache=False)
    # Test valid lead times
    times = [datetime(2024, 1, 1)]
    valid_lead_times = [timedelta(hours=1), timedelta(hours=12), timedelta(hours=48)]
    ds._validate_leadtime(times, valid_lead_times)

    # Test invalid lead times
    invalid_lead_times = [
        timedelta(hours=49),  # > 48 hours
        timedelta(hours=-1),  # < 0 hours
        timedelta(hours=1, minutes=30),  # Not hourly
    ]
    for lt in invalid_lead_times:
        with pytest.raises(ValueError):
            ds._validate_leadtime(times, [lt])

    times = [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)]
    for time0 in times:
        with pytest.raises(ValueError):
            ds._validate_leadtime([time0], [timedelta(hours=19)])


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-1),
        [timedelta(hours=2), timedelta(hours=2, minutes=1)],
        np.array([np.timedelta64(49, "h")]),
    ],
)
def test_hrrr_fx_available(lead_time):
    time = datetime(year=2022, month=12, day=25)
    variable = "t2m"
    with pytest.raises(ValueError):
        ds = HRRR_FX()
        ds(time, lead_time, variable)
