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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from earth2studio.data import ISD


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2020, month=1, day=1),
        [
            datetime(year=2022, month=1, day=1, hour=6),
            datetime(year=2023, month=2, day=3, hour=18),
        ],
    ],
)
@pytest.mark.parametrize(
    "stations, variable, tol",
    [
        (
            ["72788324220", "72063800224"],
            ["station", "time", "lat", "lon", "t2m"],
            timedelta(hours=4),
        ),
        (["72781024243"], ["station", "time", "u10m", "v10m"], timedelta(hours=12)),
    ],
)
def test_isd_fetch(stations, time, variable, tol):
    ds = ISD(stations=stations, tolerance=tol, cache=False)
    df = ds(time, variable)

    assert list(df.columns) == variable
    assert set(df["station"].unique()).issubset(set(stations))

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    # Check all rows are within requested times / tolerances
    time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
    for t in time:
        df_times = pd.to_datetime(df["time"])
        min_time = t - tol
        max_time = t + tol
        # Get bool df of rows in this time range
        time_union["time"] = time_union["time"] | (
            df_times.ge(min_time) & df_times.le(max_time)
        )

    assert time_union["time"].all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "tcc"]])
@pytest.mark.parametrize("cache", [True, False])
def test_isd_cache(time, variable, cache):

    ds = ISD(
        stations=["72781024243"],
        tolerance=timedelta(hours=12),
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert df.shape[1] == len(variable)
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    df = ds(time, variable)

    assert df.shape[1] == len(variable)

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_isd_exceptions():
    # Throw key error for invalid variable
    ds = ISD(
        stations=["72781024243"],
        tolerance=timedelta(hours=12),
        cache=False,
        verbose=False,
    )
    with pytest.raises(KeyError):
        df = ds(np.datetime64("2025-01-01T12:00:00"), ["invalid"])

    # For a invalidation station / one that it cannot find data for should return empty
    ds = ISD(stations=["invalid"], cache=False, verbose=False)
    df = ds(
        np.array(["2025-01-01T12:00:00"], dtype=np.datetime64), ["lat", "lon", "u10m"]
    )
    assert df.empty

    # For a for time that there is no data for, should return empty
    ds = ISD(stations=["72781024243"], cache=False, verbose=True)
    df = ds(np.array(["2050-01-01T12:00:00"], dtype=np.datetime64), ["t2m"])
    assert df.empty
    assert list(df.columns) == ["t2m"]


@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "bbox",
    [
        # [-180, 180)
        (35.0, -123.0, 36.0, -121.0),
        # [0, 360)
        (35.0, 237.0, 36.0, 239.0),
    ],
)
def test_isd_station_bbox(bbox):
    lat_min, lon_min, lat_max, lon_max = bbox

    stations = ISD.get_stations_bbox((lat_min, lon_min, lat_max, lon_max))
    # Load station history to verify coordinates
    df = ISD.get_station_history()
    # Normalize columns/types
    df["USAF"] = df["USAF"].astype(str).str.zfill(6)
    df["WBAN"] = (
        pd.to_numeric(df["WBAN"], errors="coerce")
        .fillna(0)
        .astype(int)
        .map(lambda x: f"{x:05d}")
    )
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])

    lon_180 = df["LON"]
    lon_360 = (df["LON"] + 360.0) % 360.0

    # Verify each returned station falls inside bbox
    for sid in stations:
        usaf, wban = sid[:6], sid[6:]
        rows = df[(df["USAF"] == usaf) & (df["WBAN"] == wban)]
        if rows.empty:
            continue
        lat_ok = (rows["LAT"] >= lat_min) & (rows["LAT"] <= lat_max)
        if lon_min >= 0 or lon_max > 180:
            lon_ok = (lon_360.loc[rows.index] >= lon_min) & (
                lon_360.loc[rows.index] <= lon_max
            )
        else:
            lon_ok = (lon_180.loc[rows.index] >= lon_min) & (
                lon_180.loc[rows.index] <= lon_max
            )
        assert bool((lat_ok & lon_ok).any())
