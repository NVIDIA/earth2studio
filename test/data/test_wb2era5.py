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
from importlib.metadata import version

import numpy as np
import pytest

from earth2studio.data import WB2ERA5, WB2ERA5_32x64, WB2ERA5_121x240


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.skipif(
    int(version("zarr").split(".")[0]) < 3, reason="Test requires zarr version > 3.0"
)
@pytest.mark.timeout(15)
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
@pytest.mark.parametrize("variable", ["tcwv", ["u500", "u200"]])
@pytest.mark.parametrize("Datasource", [WB2ERA5, WB2ERA5_121x240, WB2ERA5_32x64])
def test_wb2era5_fetch(time, variable, Datasource):

    ds = Datasource(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == Datasource.WB2_ERA5_LAT.shape[0]
    assert shape[3] == Datasource.WB2_ERA5_LON.shape[0]
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.skipif(
    int(version("zarr").split(".")[0]) < 3, reason="Test requires zarr version > 3.0"
)
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1993-04-05T00:00")])],
)
@pytest.mark.parametrize("variable", [["z500", "q200"]])
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("Datasource", [WB2ERA5, WB2ERA5_121x240, WB2ERA5_32x64])
def test_wb2era5_cache(time, variable, cache, Datasource):

    ds = Datasource(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == Datasource.WB2_ERA5_LAT.shape[0]
    assert shape[3] == Datasource.WB2_ERA5_LON.shape[0]
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == Datasource.WB2_ERA5_LAT.shape[0]
    assert shape[3] == Datasource.WB2_ERA5_LON.shape[0]
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.skipif(
    int(version("zarr").split(".")[0]) < 3, reason="Test requires zarr version > 3.0"
)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime(year=2024, month=1, day=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
@pytest.mark.parametrize("Datasource", [WB2ERA5, WB2ERA5_121x240, WB2ERA5_32x64])
def test_wb2era5_available(time, variable, Datasource):
    with pytest.raises(ValueError):
        ds = Datasource(cache=False)
        ds(time, variable)


@pytest.mark.timeout(30)
def test_wb2era5_time_statistics_mock(mock_statistic_zarr, monkeypatch):
    from earth2studio.data import WB2Climatology
    from earth2studio.data.utils import fetch_data

    ds = WB2ERA5(cache=True, verbose=True)
    reads = mock_statistic_zarr(ds)
    time = datetime.datetime(2020, 1, 2, 12)
    index = ds._get_time_index(time)
    variables = ["t2m", "t2m:mean:24h", "t2m:sum:24h", "t2m:max:12h", "t2m:min:12h"]
    data = ds(time, variables)

    assert data.shape == (1, 5, 2, 2)
    values = [float(data.sel(variable=v).values[0, 0, 0]) for v in variables]
    # Window [-24h, 0h) is the four 6-hourly fields index-4 .. index-1
    assert values == [index, index - 2.5, 4 * index - 10, index - 1, index - 2]
    assert set(data.attrs["earth2studio_statistics"]) == set(variables[1:])
    # Each distinct 6-hourly field is read once, including the instantaneous one
    assert sorted(reads) == list(range(index - 4, index + 1))
    # A window that is not a multiple of the 6 hour cadence is rejected
    with pytest.raises(ValueError, match="divisible"):
        ds(time, ["t2m:mean:9h"])

    reads.clear()
    data = ds(time, ["t2m", "t2m:mean:+0h:+12h"])
    np.testing.assert_allclose(data.values[0, :, 0, 0], [index, index + 0.5])
    assert sorted(reads) == [index, index + 1]

    reads.clear()
    data = fetch_data(
        ds,
        np.array([time], dtype="datetime64[ns]"),
        np.array(["t2m:mean:24h"]),
        delta_t=np.timedelta64(12, "h"),
    )
    assert data.values[0, 0, 0, 0, 0] == index - 3
    assert sorted(reads) == [index - 4, index - 2]

    # Both native entry points reject equivalent requests before fetching samples.
    reads.clear()
    for duplicate in (
        ["t2m", "t2m"],
        ["t2m:mean:24h", "t2m:mean:1d"],
        ["t2m:mean:24h", "t2m:mean:-24h:0h"],
    ):
        with pytest.raises(ValueError, match="duplicate quantities"):
            ds(time, duplicate)
        for cadence in (None, np.timedelta64(6, "h")):
            with pytest.raises(ValueError, match="duplicate quantities"):
                fetch_data(
                    ds,
                    np.array([time], dtype="datetime64[ns]"),
                    np.array(duplicate),
                    delta_t=cadence,
                )
    assert reads == []

    climate = WB2Climatology()
    assert not hasattr(climate, "time_statistics")
    assert not hasattr(climate.lexicon, "TIME_STATISTICS")

    async def fetch(time, variable):
        assert variable == ["t2m"]
        return await ds.fetch(time, variable)

    monkeypatch.setattr(climate, "fetch", fetch)
    data = fetch_data(
        climate,
        np.array([time], dtype="datetime64[ns]"),
        np.array(["t2m:mean:24h"]),
        delta_t=np.timedelta64(6, "h"),
    )
    assert data.values[0, 0, 0, 0, 0] == index - 2.5
