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

import asyncio
import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest

from earth2studio.data import AIFS_ENS_FX, AIFS_FX, IFS, IFS_ENS, IFS_ENS_FX, IFS_FX


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
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_fetch(time, variable):
    ds = IFS(cache=False)
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

    ds = IFS_FX(cache=False)
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert np.allclose(data.values, data_fx.values[:, 0])


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
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
@pytest.mark.parametrize("member", [0, 1])
def test_ifs_ens_fetch(time, variable, member):
    ds = IFS_ENS(cache=False, member=member)
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

    ds = IFS_ENS_FX(cache=False, member=member)
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.allclose(data.values, data_fx.values[:, 0])


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
def test_ifs_ens_fx_fetch(time, lead_time, variable):
    ds = IFS_ENS_FX(cache=False, member=0)
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
    "lead_time",
    [
        timedelta(hours=0),
        [
            timedelta(hours=6),
            timedelta(hours=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["tcc", "tp"]])
@pytest.mark.parametrize("member", [0, 1])
def test_aifs_ens_fx_fetch(lead_time, variable, member):
    time = now6h() - timedelta(hours=12)
    ds = AIFS_ENS_FX(cache=False, member=member)
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
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([now6h() - timedelta(days=2)], dtype="datetime64"),
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
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    data = ds(time, variable[0])
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


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
async def test_ifs_async_fetch():
    t = now6h() - timedelta(hours=12)
    lt = timedelta(hours=0)
    variable = "msl"

    ds_ifs = IFS(cache=False)
    ds_ifs_fx = IFS_FX(cache=False)
    ds_ifs_ens = IFS_ENS(cache=False, member=1)
    ds_ifs_ens_fx = IFS_ENS_FX(cache=False, member=1)

    da_ifs, da_fx, da_ens, da_ens_fx = await asyncio.gather(
        ds_ifs.fetch(t, variable),
        ds_ifs_fx.fetch(t, lt, variable),
        ds_ifs_ens.fetch(t, variable),
        ds_ifs_ens_fx.fetch(t, lt, variable),
    )

    # IFS (analysis): [time, variable, lat, lon]
    assert da_ifs.shape[0] == 1
    assert da_ifs.shape[1] == 1
    assert da_ifs.shape[2] == 721
    assert da_ifs.shape[3] == 1440
    assert not np.isnan(da_ifs.values).any()

    # IFS_FX (forecast): [time, lead_time, variable, lat, lon]
    assert da_fx.shape[0] == 1
    assert da_fx.shape[1] == 1
    assert da_fx.shape[2] == 1
    assert da_fx.shape[3] == 721
    assert da_fx.shape[4] == 1440
    assert not np.isnan(da_fx.values).any()

    # IFS_ENS (analysis): [time, variable, lat, lon]
    assert da_ens.shape[0] == 1
    assert da_ens.shape[1] == 1
    assert da_ens.shape[2] == 721
    assert da_ens.shape[3] == 1440
    assert not np.isnan(da_ens.values).any()

    # IFS_ENS_FX (forecast): [time, lead_time, variable, lat, lon]
    assert da_ens_fx.shape[0] == 1
    assert da_ens_fx.shape[1] == 1
    assert da_ens_fx.shape[2] == 1
    assert da_ens_fx.shape[3] == 721
    assert da_ens_fx.shape[4] == 1440
    assert not np.isnan(da_ens_fx.values).any()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [IFS, IFS_ENS])
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_time_available(data_source, time, variable):
    with pytest.raises(ValueError):
        ds = IFS(source="ecmwf")
        ds(time, ["msl"])

        ds = IFS_FX(source="ecmwf")
        ds(time, [timedelta(hours=0)], ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=1),
        timedelta(hours=147),
        timedelta(hours=366),
    ],
)
def test_ifs_leadtime_available(lead_time):
    with pytest.raises(ValueError):
        time = now6h() - timedelta(days=2)
        ds = IFS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])

        ds = IFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", ["msl"])
def test_aifs_time_available(time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = AIFS_FX(source="ecmwf")
        ds(time, lead_time, variable)

    with pytest.raises(ValueError):
        ds = AIFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, variable)
