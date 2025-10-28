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

from earth2studio.data import CosineSolarZenith, LandSeaMask, SurfaceGeoPotential


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
def test_lsm_fetch(time):

    ds = LandSeaMask(cache=False)
    data = ds(time, "lsm")
    shape = data.shape

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(["lsm"]))
    assert not np.isnan(data.values).any()
    assert (np.logical_and(data.values <= 1, data.values >= 0)).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1993-04-05T00:00")])],
)
@pytest.mark.parametrize("cache", [True, False])
def test_lsm_cache(time, cache):

    ds = LandSeaMask(cache=cache)
    data = ds(time, "lsm")
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, "lsm")
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
def test__fetch(time):

    ds = SurfaceGeoPotential(cache=False)
    data = ds(time, "z")
    shape = data.shape

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(["z"]))
    assert not np.isnan(data.values).any()


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
@pytest.mark.parametrize(
    "domain_coords",
    [
        {
            "lat": np.linspace(-90, 90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
        {
            "phoo": np.arange(10),
            "lat": np.linspace(-90, 90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
    ],
)
def test_uvcossza_fetch(time, domain_coords):
    ds = CosineSolarZenith(domain_coords)
    data = ds(time, "")

    if isinstance(time, datetime.datetime):
        time = [time]

    assert data.shape[0] == len(time)
    assert data.shape[1] == 1
    for i, (key, value) in enumerate(domain_coords.items()):
        assert data.shape[i + 2] == value.shape[0]
        assert np.all(data.coords[key] == value)
    assert np.array_equal(data.coords["variable"].values, np.array(["uvcossza"]))
    assert not np.isnan(data.values).any()
