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
import pandas as pd
import pyarrow as pa
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
            ["t2m"],
            timedelta(hours=1),
        ),
        (
            ["72781024243"],
            ["u10m", "v10m", "t2m", "d2m", "fg10m"],
            timedelta(hours=4),
        ),
    ],
)
def test_isd_fetch(stations, time, variable, tol):
    ds = ISD(stations=stations, tolerance=tol, cache=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["station"].unique()).issubset(set(stations))
    assert set(df["variable"].unique()).issubset(set(variable))

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    # Check all rows are within requested times / tolerances
    time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
    for t in time:
        df_times = df["time"]
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

    # Check columns match schema
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_isd_schema_fields():
    station = "72033063853"
    time = np.array(["2025-01-01T12:00:00"], dtype=np.datetime64)
    tol = timedelta(minutes=10)

    ds = ISD(stations=[station], tolerance=tol)

    # Test with default schema (all fields)
    df_full = ds(time, ["t2m"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    # Test with subset of fields (must include required fields)
    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t2m"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_isd_exceptions():
    # Throw key error for invalid variable
    ds = ISD(
        stations=["72781024243"],
        tolerance=timedelta(hours=12),
        cache=False,
        verbose=False,
    )
    with pytest.raises(KeyError):
        df = ds(np.array([np.datetime64("2025-01-01T12:00:00")]), ["invalid"])

    # For a invalid station / one that it cannot find data for should return empty
    ds = ISD(stations=["invalid"], cache=False, verbose=False)
    df = ds(np.array(["2025-01-01T12:00:00"], dtype=np.datetime64), ["u10m"])
    assert df.empty

    # Time that there is no data for, should return empty
    ds = ISD(stations=["72781024243"], cache=False, verbose=True)
    df = ds(np.array(["2050-01-01T12:00:00"], dtype=np.datetime64), ["t2m"])
    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert (df["variable"] == "t2m").all()

    with pytest.raises(KeyError):
        ds(
            np.datetime64("2025-01-01T12:00:00"),
            ["t2m"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(np.datetime64("2025-01-01T12:00:00"), ["t2m"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),  # Should be timestamp, not string
        ]
    )
    with pytest.raises(TypeError):
        ds(np.datetime64("2025-01-01T12:00:00"), ["t2m"], fields=wrong_type_schema)


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


@pytest.fixture(scope="class")
def _shared_isd(request):
    # shared datasource (its internal methods operate on the passed DataFrame)
    request.cls.isd = ISD(
        stations=[], tolerance=timedelta(0), cache=False, verbose=False
    )


@pytest.mark.usefixtures("_shared_isd")
class TestISDExtractors:
    @pytest.mark.parametrize(
        "wnd, expected",
        [
            (None, [np.nan]),  # missing column -> NaN
            (["100,0,X,9999,0"], [np.nan]),  # speed missing code
            (["090,0,X,0500,0"], [50.0]),  # 500 -> 50.0 m/s
        ],
    )
    def test_extract_ws10m(self, wnd, expected):
        if wnd is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"WND": wnd})
        out = self.isd._extract_ws10m(df.copy())
        np.testing.assert_allclose(
            out["ws10m"].to_numpy(), np.array(expected), equal_nan=True
        )

    @pytest.mark.parametrize(
        "wnd, expected_u, expected_v",
        [
            (None, [np.nan], [np.nan]),  # missing column
            (["100"], [np.nan], [np.nan]),  # insufficient parts
            (["999,0,C,0000,0"], [0.0], [0.0]),  # calm -> 0
            (["090,0,X,0100,0"], [-10.0], [0.0]),  # 10 m/s from 90 deg
        ],
    )
    def test_extract_uv(self, wnd, expected_u, expected_v):
        if wnd is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"WND": wnd})
        out = self.isd._extract_uv(df.copy())
        u = out["u10m"].to_numpy()
        v = out["v10m"].to_numpy()
        np.testing.assert_allclose(u, np.array(expected_u), atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(v, np.array(expected_v), atol=1e-6, equal_nan=True)

    @pytest.mark.parametrize(
        "aa1, expected",
        [
            (None, [np.nan]),  # missing column
            (["01"], [np.nan]),  # insufficient parts
            (["01,0010,9,5"], [0.001]),  # 10 / 10000 m
        ],
    )
    def test_extract_tp(self, aa1, expected):
        if aa1 is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"AA1": aa1})
        out = self.isd._extract_tp(df.copy())
        np.testing.assert_allclose(
            out["tp"].to_numpy(), np.array(expected), equal_nan=True
        )

    @pytest.mark.parametrize(
        "tmp, expected",
        [
            (None, [np.nan]),  # missing column
            ([""], [np.nan]),  # unparsable -> NaN
            (["+0273,5"], [300.45]),  # 27.3C -> K
        ],
    )
    def test_extract_t2m(self, tmp, expected):
        if tmp is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"TMP": tmp})
        out = self.isd._extract_t2m(df.copy())
        np.testing.assert_allclose(
            out["t2m"].to_numpy(), np.array(expected), atol=1e-6, equal_nan=True
        )

    @pytest.mark.parametrize(
        "oc1, expected",
        [
            (None, [np.nan]),  # missing
            ([""], [np.nan]),  # insufficient/unparsable
            (["0050,0"], [5.0]),  # 50 -> 5.0 m/s
        ],
    )
    def test_extract_fg10m(self, oc1, expected):
        if oc1 is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"OC1": oc1})
        out = self.isd._extract_fg10m(df.copy())
        np.testing.assert_allclose(
            out["fg10m"].to_numpy(), np.array(expected), equal_nan=True
        )

    @pytest.mark.parametrize(
        "dew, expected",
        [
            (None, [np.nan]),  # missing
            ([""], [np.nan]),  # insufficient/unparsable
            (["+0100,5"], [283.15]),  # 10.0C -> K
        ],
    )
    def test_extract_d2m(self, dew, expected):
        if dew is None:
            df = pd.DataFrame({"OTHER": ["x"]})
        else:
            df = pd.DataFrame({"DEW": dew})
        out = self.isd._extract_d2m(df.copy())
        np.testing.assert_allclose(
            out["d2m"].to_numpy(), np.array(expected), atol=1e-6, equal_nan=True
        )

    @pytest.mark.parametrize(
        "col, value, expected",
        [
            ("GA1", "04,xx", 0.5),  # GA mapping -> 4 => 0.5
            ("GD1", "3,xx", 0.75),  # GD mapping -> 3 => 0.75
            ("GF1", "08,xx", 1.0),  # GF mapping -> 8 => 1.0
        ],
    )
    def test_extract_tcc(self, col, value, expected):
        df = pd.DataFrame({col: [value]})
        out = self.isd._extract_tcc(df.copy())
        np.testing.assert_allclose(
            out["tcc"].to_numpy(), np.array([expected]), equal_nan=True
        )
