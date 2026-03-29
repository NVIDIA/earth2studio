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

import datetime
import pathlib
import shutil

import numpy as np
import pytest

from earth2studio.data import CAMS, CAMS_FX

YESTERDAY = datetime.datetime.now(datetime.UTC).replace(
    hour=0, minute=0, second=0, microsecond=0
) - datetime.timedelta(days=1)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        [YESTERDAY],
        np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))]),
    ],
)
@pytest.mark.parametrize("variable", ["dust", ["dust", "pm2p5"]])
def test_cams_fetch(time, variable):
    ds = CAMS(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert len(data.coords["lat"]) > 0
    assert len(data.coords["lon"]) > 0
    assert not np.isnan(data.values).all()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("variable", [["dust", "so2sfc"]])
@pytest.mark.parametrize("cache", [True, False])
def test_cams_cache(variable, cache):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    ds = CAMS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert not np.isnan(data.values).all()
    assert pathlib.Path(ds.cache).is_dir() == cache

    data = ds(time, variable[0])
    assert data.shape[1] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
def test_cams_invalid():
    with pytest.raises((ValueError, KeyError)):
        ds = CAMS()
        ds(YESTERDAY, "nonexistent_var")


def test_cams_time_validation():
    with pytest.raises(ValueError):
        ds = CAMS()
        ds(datetime.datetime(2018, 1, 1), "dust")


def test_cams_available():
    assert CAMS.available(datetime.datetime(2024, 1, 1))
    assert not CAMS.available(datetime.datetime(2015, 1, 1))


# ---- CAMS_FX tests ----


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("variable", ["dust", ["dust", "pm2p5"]])
@pytest.mark.parametrize(
    "lead_time",
    [
        datetime.timedelta(hours=0),
        [datetime.timedelta(hours=0), datetime.timedelta(hours=24)],
    ],
)
def test_cams_fx_fetch(variable, lead_time):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    ds = CAMS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, datetime.timedelta):
        lead_time = [lead_time]

    assert shape[0] == 1  # time
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert len(data.coords["lat"]) > 0
    assert len(data.coords["lon"]) > 0
    assert not np.isnan(data.values).all()


def test_cams_fx_available():
    assert CAMS_FX.available(datetime.datetime(2024, 1, 1))
    assert not CAMS_FX.available(datetime.datetime(2010, 1, 1))
