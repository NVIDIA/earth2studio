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
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import NCAR_ERA5


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
@pytest.mark.parametrize("variable", ["sd", ["smlt"]])
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
    assert np.array_equal(data.coords["lat"].values, NCAR_ERA5.NCAR_EAR5_LAT)
    assert np.array_equal(data.coords["lon"].values, NCAR_ERA5.NCAR_EAR5_LON)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["sd", "smlt"]])
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
        assert not any(pathlib.Path(ds.cache).iterdir())

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


@pytest.mark.parametrize(
    "time,expected_uri,expected_initial_time,expected_fc_hour",
    [
        # Jan 1 2025 00:00 -> initial_time = Dec 31 2024 18:00, fc_hour = 6
        pytest.param(
            datetime(2025, 1, 1, 0),
            "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu/202412/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.2024121606_2025010106.nc",
            datetime(2024, 12, 31, 18),
            6,
            id="2025-01-01T00",
        ),
        # Dec 31 2024 19:00 -> initial_time = Dec 31 2024 18:00, fc_hour = 1
        pytest.param(
            datetime(2024, 12, 31, 19),
            "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu/202412/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.2024121606_2025010106.nc",
            datetime(2024, 12, 31, 18),
            1,
            id="2024-12-31T19",
        ),
        # Jan 16 2025 06:00 -> initial_time = Jan 15 2025 18:00, fc_hour = 12
        pytest.param(
            datetime(2025, 1, 16, 6),
            "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu/202501/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.2025010106_2025011606.nc",
            datetime(2025, 1, 15, 18),
            12,
            id="2025-01-16T06",
        ),
        # Jan 16 2025 07:00 -> initial_time = Jan 16 2025 06:00, fc_hour = 1
        pytest.param(
            datetime(2025, 1, 16, 7),
            "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu/202501/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.2025011606_2025020106.nc",
            datetime(2025, 1, 16, 6),
            1,
            id="2025-01-16T07",
        ),
        # Feb 16 2025 07:00 -> initial_time = Feb 16 2025 06:00, fc_hour = 1
        pytest.param(
            datetime(2025, 2, 16, 7),
            "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu/202502/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.2025021606_2025030106.nc",
            datetime(2025, 2, 16, 6),
            1,
            id="2025-02-16T07",
        ),
    ],
)
def test_ncar_create_tasks_accumulated(
    time, expected_uri, expected_initial_time, expected_fc_hour
):
    ds = NCAR_ERA5()
    tasks = ds._create_tasks([time], ["lsp"])

    assert len(tasks) == 1
    task = next(iter(tasks.values()))

    assert task.ncar_file_uri == expected_uri
    assert "lsp" in task.ncar_level_indices.values()

    # Check metadata
    time_idx = next(iter(task.ncar_meta.keys()))
    meta = task.ncar_meta[time_idx]
    assert meta["forecast_initial_time"] == expected_initial_time
    assert meta["forecast_hour"] == expected_fc_hour
    assert meta["time"] == np.datetime64(time)


def test_ncar_create_tasks_accumulated_mult():
    # Test with multiple accumulated variables
    times = [
        datetime(2025, 1, 1, 0),
        datetime(2025, 1, 1, 6),
        datetime(2025, 1, 1, 12),
        datetime(2025, 1, 1, 19),
    ]
    variables = ["lsp", "cp"]

    ds = NCAR_ERA5()
    tasks = ds._create_tasks(times, variables)
    # 4 files: 2 for lsp (times 0,1 share file; times 2,3 in separate files)
    #          2 for cp  (same pattern)
    assert len(tasks) == 4

    for task in tasks.values():
        assert len(task.ncar_meta) == 2

    # Verify all times are accounted for
    all_times = set()
    for task in tasks.values():
        for meta in task.ncar_meta.values():
            all_times.add(meta["time"].astype("datetime64[us]").astype(datetime))
    assert all_times == set(times)

    # Verify both variables are in the tasks
    all_vars = set()
    for task in tasks.values():
        all_vars.update(task.ncar_level_indices.values())
    assert all_vars == set(variables)
