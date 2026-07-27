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

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import DWDSynopReports
from earth2studio.data import dwd_synop_reports as mod
from earth2studio.lexicon import DWDSynopReportsLexicon


def _wide_frame() -> pd.DataFrame:
    # Two stations: a WMO-id station and a named (automatic) station.
    return pd.DataFrame(
        {
            "station": ["10015", "Berlin"],
            "time": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:00"]),
            "lat": [54.0, 52.5],
            "lon": [8.0, 13.4],
            "elev": [10.0, 50.0],
            "airTemperature": [288.0, 290.0],
            "dewpointTemperature": [283.0, 284.0],
            "windSpeed": [5.0, 10.0],
            "windDirection": [0.0, 90.0],
            "pressureReducedToMeanSeaLevel": [101300.0, 101000.0],
        }
    )


# ======================================================================
# 1. Network fetch test (slow, xfail)
# ======================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_dwd_synop_reports_fetch():
    ds = DWDSynopReports(
        time_tolerance=timedelta(minutes=45), cache=False, verbose=False
    )
    # Query a recent synoptic hour within the retention window.
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    df = ds(t, ["t2m", "u10m", "v10m"])
    assert list(df.columns) == ds.SCHEMA.names
    # A real synoptic query must return data for many stations (also proves
    # non-empty).
    assert df["station"].nunique() > 10
    # Each requested variable must be present, not only temperature.
    assert {"t2m", "u10m", "v10m"} == set(df["variable"].unique())
    # Physical sanity: 2 m temperature in Kelvin and wind speed magnitude.
    t2m = df[df["variable"] == "t2m"]["observation"]
    assert t2m.between(220.0, 330.0).all()
    # Pivot so u10m/v10m are aligned per (station, time) before combining.
    wind = df[df["variable"].isin(["u10m", "v10m"])].pivot_table(
        index=["station", "time"], columns="variable", values="observation"
    )
    speed = np.hypot(wind["u10m"].to_numpy(), wind["v10m"].to_numpy())
    assert (speed < 120.0).all()


# ======================================================================
# 2. Cache test (slow, xfail)
# ======================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_dwd_synop_reports_cache(cache, tmp_path, monkeypatch):

    # Isolate to a temp cache root so the test never removes a developer's cache.
    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    ds = DWDSynopReports(cache=cache, verbose=False)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    df = ds(t, ["t2m"])
    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache
    shutil.rmtree(ds.cache, ignore_errors=True)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_cache_hit_skips_download(tmp_path, monkeypatch):
    # A second fetch of the same URL must reuse the cached file rather than
    # download it again: _cat_file is called only once.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    calls = {"n": 0}

    class _FS:
        async def _cat_file(self, url):
            calls["n"] += 1
            return b"BUFR"

    ds = DWDSynopReports(verbose=False)
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    ds.fs = _FS()

    url = "https://x/f.bin"
    p1 = asyncio.run(ds._fetch_remote_file(url))
    p2 = asyncio.run(ds._fetch_remote_file(url))
    assert p1 == p2
    assert calls["n"] == 1


@pytest.mark.timeout(15)
def test_dwd_synop_reports_prune_cache(tmp_path, monkeypatch):
    # _prune_cache removes cache files older than the retention window (their
    # bulletin URLs are permanently gone) while keeping fresh ones.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    ds = DWDSynopReports(verbose=False)
    cache = pathlib.Path(ds.cache)
    cache.mkdir(parents=True, exist_ok=True)
    old = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()

    # Stale/fresh cache files, an orphaned atomic-write .tmp file, and per-instance
    # tmp_* dirs left by crashed cache=False runs: stale artifacts must be swept,
    # fresh ones kept.
    stale = cache / "dwd_synop_reports_stale.bin"
    fresh = cache / "dwd_synop_reports_fresh.bin"
    stale_tmp = cache / "dwd_synop_reports_x.bin.abcd1234.tmp"
    stale_dir = cache / "tmp_deadbeef"
    fresh_dir = cache / "tmp_cafebabe"
    for f in (stale, fresh, stale_tmp):
        f.write_bytes(b"BUFR")
    for d in (stale_dir, fresh_dir):
        d.mkdir()
        (d / "dwd_synop_reports_inner.bin").write_bytes(b"BUFR")
    for p in (stale, stale_tmp, stale_dir):
        os.utime(p, (old, old))

    ds._prune_cache()
    assert not stale.exists()
    assert not stale_tmp.exists()
    assert not stale_dir.exists()
    assert fresh.exists()
    assert fresh_dir.exists()


# ======================================================================
# 3. Mock test — no network
# ======================================================================


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    df = ds._compile_dataframe(
        _wide_frame(), ["t2m", "u10m", "v10m", "msl"], windows, None
    )

    assert list(df.columns) == ds.SCHEMA.names
    # Direct variable values.
    sub = df[df["station"] == "10015"].set_index("variable")["observation"]
    assert np.isclose(sub["t2m"], 288.0)
    assert np.isclose(sub["msl"], 101300.0)
    # Wind components (meteorological, blows-from): dir=0,spd=5 -> u=0,v=-5.
    assert np.isclose(sub["u10m"], 0.0, atol=1e-4)
    assert np.isclose(sub["v10m"], -5.0, atol=1e-4)
    # dir=90,spd=10 -> u=-10,v=0.
    sub2 = df[df["station"] == "Berlin"].set_index("variable")["observation"]
    assert np.isclose(sub2["u10m"], -10.0, atol=1e-4)
    assert np.isclose(sub2["v10m"], 0.0, atol=1e-4)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_bbox_and_window():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # bbox excludes Berlin (lat 52.5), keeps station 10015 (lat 54.0).
    df = ds._compile_dataframe(_wide_frame(), ["t2m"], windows, (53.0, 5.0, 56.0, 10.0))
    assert set(df["station"].unique()) == {"10015"}
    # A window with no overlap returns an empty frame with the schema columns.
    empty = ds._compile_dataframe(
        _wide_frame(),
        ["t2m"],
        [(datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1))],
        None,
    )
    assert empty.empty
    assert list(empty.columns) == ds.SCHEMA.names


def _wide_pair(stations, at_values, msl_values):
    # Two rows at the same coordinates and time with the given ids/fields.
    return pd.DataFrame(
        {
            "station": stations,
            "time": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:00"]),
            "lat": [54.0, 54.0],
            "lon": [8.0, 8.0],
            "elev": [10.0, 10.0],
            "airTemperature": at_values,
            "dewpointTemperature": [np.nan, np.nan],
            "windSpeed": [np.nan, np.nan],
            "windDirection": [np.nan, np.nan],
            "pressureReducedToMeanSeaLevel": msl_values,
        }
    )


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_coalesces_retransmission():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Same station and time twice with complementary fields (t2m vs msl).
    wide = _wide_pair(["10015", "10015"], [288.0, np.nan], [np.nan, 101300.0])
    df = ds._compile_dataframe(wide, ["t2m", "msl"], windows, None)
    assert set(df["station"]) == {"10015"}
    got = df.set_index("variable")["observation"]
    assert np.isclose(got["t2m"], 288.0)
    assert np.isclose(got["msl"], 101300.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_keeps_colocated_stations():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Two different station ids at identical coordinates/time stay separate.
    wide = _wide_pair(["A", "B"], [288.0, 289.0], [np.nan, np.nan])
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    assert set(df["station"]) == {"A", "B"}
    assert len(df) == 2


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_newest_bulletin_wins():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Same station and time, conflicting t2m; the later bulletin holds the
    # corrected value and must win.
    wide = _wide_pair(["10015", "10015"], [288.0, 290.0], [np.nan, np.nan])
    wide["_bulletin"] = pd.to_datetime(["2024-06-01 12:01", "2024-06-01 12:06"])
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    got = df.set_index("variable")["observation"]
    assert np.isclose(got["t2m"], 290.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_calm_wind_zero_components():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Calm wind: zero speed with a missing direction (SYNOP omits direction when
    # calm) must still yield u10m = v10m = 0, not be dropped as undefined.
    wide = _wide_frame().iloc[[0]].copy()
    wide["windSpeed"] = 0.0
    wide["windDirection"] = np.nan
    df = ds._compile_dataframe(wide, ["u10m", "v10m"], windows, None)
    got = df.set_index("variable")["observation"]
    assert np.isclose(got["u10m"], 0.0)
    assert np.isclose(got["v10m"], 0.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_coalesces_null_id_stations():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Null-id (unnamed) stations coalesce by (lat, lon, time): two rows at the
    # same coordinates with complementary fields merge to one; a third null-id
    # row at different coordinates stays separate.
    wide = pd.DataFrame(
        {
            "station": [None, None, None],
            "time": pd.to_datetime(["2024-06-01 12:00"] * 3),
            "lat": [54.0, 54.0, 48.0],
            "lon": [8.0, 8.0, 11.0],
            "elev": [10.0, 10.0, 500.0],
            "airTemperature": [288.0, np.nan, 293.0],
            "dewpointTemperature": [np.nan] * 3,
            "windSpeed": [np.nan] * 3,
            "windDirection": [np.nan] * 3,
            "pressureReducedToMeanSeaLevel": [np.nan, 101300.0, np.nan],
        }
    )
    df = ds._compile_dataframe(wide, ["t2m", "msl"], windows, None)
    # The colocated pair merges (t2m + msl at 54,8); the 48,11 station is separate.
    at_54 = df[(df["lat"] == 54.0) & (df["variable"] == "t2m")]["observation"]
    msl_54 = df[(df["lat"] == 54.0) & (df["variable"] == "msl")]["observation"]
    assert np.isclose(at_54.iloc[0], 288.0)
    assert np.isclose(msl_54.iloc[0], 101300.0)
    assert set(zip(df["lat"], df["lon"])) == {(54.0, 8.0), (48.0, 11.0)}


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_name_fallback_needs_coordinates():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Two distinct stations share a generic fallback name (non-WMO id) but sit
    # at different coordinates; they must not be merged into one.
    wide = pd.DataFrame(
        {
            "station": ["Auto Station", "Auto Station"],
            "time": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:00"]),
            "lat": [54.0, 48.0],
            "lon": [8.0, 11.0],
            "elev": [10.0, 500.0],
            "airTemperature": [288.0, 293.0],
            "dewpointTemperature": [np.nan, np.nan],
            "windSpeed": [np.nan, np.nan],
            "windDirection": [np.nan, np.nan],
            "pressureReducedToMeanSeaLevel": [np.nan, np.nan],
        }
    )
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    assert set(zip(df["lat"], df["lon"])) == {(54.0, 8.0), (48.0, 11.0)}
    assert len(df) == 2


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_numeric_fallback_name_not_merged():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Two distinct non-WMO stations share a 5-digit fallback name. The decode-time
    # _is_wmo=False flag must keep them keyed on coordinates so they stay separate;
    # a bare \\d{5} check would wrongly treat the name as a WMO id and merge them.
    wide = pd.DataFrame(
        {
            "station": ["12345", "12345"],
            "_is_wmo": [False, False],
            "time": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:00"]),
            "lat": [54.0, 48.0],
            "lon": [8.0, 11.0],
            "elev": [10.0, 500.0],
            "airTemperature": [288.0, 293.0],
            "dewpointTemperature": [np.nan, np.nan],
            "windSpeed": [np.nan, np.nan],
            "windDirection": [np.nan, np.nan],
            "pressureReducedToMeanSeaLevel": [np.nan, np.nan],
        }
    )
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    assert set(zip(df["lat"], df["lon"])) == {(54.0, 8.0), (48.0, 11.0)}
    assert len(df) == 2


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_normalizes_negative_longitude():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # A western-hemisphere raw BUFR longitude (-100) must normalize to [0, 360).
    wide = _wide_frame().iloc[[0]].copy()
    wide["lon"] = -100.0
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    assert np.isclose(df["lon"].iloc[0], 260.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_bbox_uses_corrected_coordinates():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Same WMO station, two bulletins: the older places it inside the bbox, the
    # newer corrects it outside. Because bbox filtering runs after coalescing
    # (newest wins), the corrected position governs and the station is excluded.
    wide = pd.DataFrame(
        {
            "station": ["10015", "10015"],
            "_is_wmo": [True, True],
            "_bulletin": pd.to_datetime(["2024-06-01 12:01", "2024-06-01 12:06"]),
            "time": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:00"]),
            "lat": [54.0, 40.0],  # older inside, newer outside the bbox below
            "lon": [8.0, 8.0],
            "elev": [10.0, 10.0],
            "airTemperature": [288.0, 288.0],
            "dewpointTemperature": [np.nan, np.nan],
            "windSpeed": [np.nan, np.nan],
            "windDirection": [np.nan, np.nan],
            "pressureReducedToMeanSeaLevel": [np.nan, np.nan],
        }
    )
    df = ds._compile_dataframe(wide, ["t2m"], windows, (53.0, 5.0, 56.0, 10.0))
    assert df.empty


@pytest.mark.timeout(15)
def test_dwd_synop_reports_compile_rejects_invalid_coordinates():
    ds = DWDSynopReports(cache=False, verbose=False)
    windows = [(datetime(2024, 6, 1, 11, 30), datetime(2024, 6, 1, 12, 30))]
    # Out-of-range and non-finite coordinates must be dropped before longitude
    # normalization so a malformed BUFR value cannot become a plausible location.
    # Covers out-of-range (999), non-finite (inf, reaches np.isfinite because
    # dropna does not remove it), and missing (nan, removed by dropna) values.
    wide = pd.DataFrame(
        {
            "station": ["10015", "bad_lon", "bad_lat", "inf_lon", "nan_lat"],
            "time": pd.to_datetime(["2024-06-01 12:00"] * 5),
            "lat": [54.0, 48.0, 999.0, 48.0, np.nan],
            "lon": [8.0, 999.0, 11.0, np.inf, 11.0],
            "elev": [10.0, 10.0, 10.0, 10.0, 10.0],
            "airTemperature": [288.0, 290.0, 291.0, 292.0, 293.0],
            "dewpointTemperature": [np.nan] * 5,
            "windSpeed": [np.nan] * 5,
            "windDirection": [np.nan] * 5,
            "pressureReducedToMeanSeaLevel": [np.nan] * 5,
        }
    )
    df = ds._compile_dataframe(wide, ["t2m"], windows, None)
    assert set(df["station"]) == {"10015"}


@pytest.mark.timeout(15)
def test_dwd_synop_reports_call_mock(tmp_path, monkeypatch):
    # No-network mock test of the full public call path. Only the filesystem
    # (_ls/_cat_file) and pdbufr are stubbed; every DWD transform runs for real,
    # exercising the wiring seam: fetch -> _create_tasks -> _fetch_wrapper
    # -> _fetch_remote_file -> _decode_paths -> _decode_synop_bufr ->
    # _compile_dataframe.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    # A recent synoptic hour within the retention window; bulletin published a
    # few minutes earlier so it falls in the padded discovery window.
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    bulletin = t - timedelta(minutes=10)
    fname = f"{mod.GERMANY_URL}Z__C_EDZW_{bulletin:%Y%m%d%H%M%S}_synop.bin"

    class _FS:
        async def _ls(self, path, detail=False):
            return [fname]

        async def _cat_file(self, url):
            return b"BUFR"

    raw = pd.DataFrame(
        {
            "latitude": [54.0],
            "longitude": [8.0],
            "heightOfStationGroundAboveMeanSeaLevel": [10.0],
            "blockNumber": [10.0],
            "stationNumber": [1.0],
            "stationOrSiteName": ["Foo"],
            "year": [t.year],
            "month": [t.month],
            "day": [t.day],
            "hour": [t.hour],
            "minute": [0],
            "airTemperature": [288.0],
            "dewpointTemperature": [283.0],
            "windSpeed": [5.0],
            "windDirection": [0.0],
            "pressureReducedToMeanSeaLevel": [101300.0],
        }
    )
    monkeypatch.setattr(mod.pdbufr, "read_bufr", lambda *a, **k: raw)

    ds = DWDSynopReports(time_tolerance=timedelta(minutes=30), verbose=False)
    ds.fs = _FS()  # set before fetch so _async_init is skipped

    df = ds(t, ["t2m", "d2m", "ws10m", "u10m", "v10m", "msl"])

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()) == {
        "t2m",
        "d2m",
        "ws10m",
        "u10m",
        "v10m",
        "msl",
    }
    assert set(df["station"].unique()) == {"10001"}
    assert df.attrs["source"] == ds.SOURCE_ID
    got = df.set_index("variable")["observation"]
    assert np.isclose(got["t2m"], 288.0)
    assert np.isclose(got["d2m"], 283.0)
    assert np.isclose(got["ws10m"], 5.0)
    assert np.isclose(got["msl"], 101300.0)
    # dir=0, spd=5 (wind from north) -> u=0, v=-5.
    assert np.isclose(got["u10m"], 0.0, atol=1e-4)
    assert np.isclose(got["v10m"], -5.0, atol=1e-4)

    # Requesting a subset of fields narrows the returned columns.
    sub = ds(t, ["t2m"], fields=["time", "observation", "variable"])
    assert list(sub.columns) == ["time", "observation", "variable"]


@pytest.mark.timeout(15)
def test_dwd_synop_reports_decode_bufr_transform(tmp_path, monkeypatch):

    # pdbufr.read_bufr returns one row per station report; stub it to exercise
    # the station-id / time / missing-value transform in _decode.
    raw = pd.DataFrame(
        {
            "latitude": [54.0, 48.1, 50.0],
            "longitude": [8.0, 11.5, 7.0],
            "heightOfStationGroundAboveMeanSeaLevel": [10.0, 50.0, 200.0],
            # 3rd station uses the raw CODES_MISSING_LONG sentinel (not NaN)
            # for its id -> must be treated as missing (falls back to name -> None).
            "blockNumber": [10.0, np.nan, 2147483647],  # WMO id / name / missing
            "stationNumber": [1.0, np.nan, 2147483647],
            "stationOrSiteName": ["Foo", "Berlin", ""],
            "year": [2024, 2024, 2024],
            "month": [6, 6, 6],
            "day": [1, 1, 1],
            "hour": [12, 12, 12],
            "minute": [0, 30, np.nan],
            # 2nd missing (NaN), 3rd is the CODES_MISSING_DOUBLE sentinel.
            "airTemperature": [288.0, np.nan, -1e100],
            "dewpointTemperature": [283.0, 284.0, 285.0],
            "windSpeed": [5.0, 10.0, 3.0],
            "windDirection": [0.0, 90.0, 180.0],
            "pressureReducedToMeanSeaLevel": [101300.0, 101000.0, 100500.0],
        }
    )
    monkeypatch.setattr(mod.pdbufr, "read_bufr", lambda *a, **k: raw)
    path = tmp_path / "synop.bin"
    path.write_bytes(b"BUFR")

    df = mod._decode_synop_bufr(str(path))

    assert list(df.columns) == list(mod._META_COLUMNS) + list(mod._BUFR_ELEMENTS) + [
        "_is_wmo"
    ]
    assert len(df) == 3
    # WMO block+station id, station-name fallback, empty name -> None.
    assert list(df["station"]) == ["10001", "Berlin", None]
    # Only the block+station id is flagged as a true WMO id.
    assert list(df["_is_wmo"]) == [True, False, False]
    # NaN and sentinel-missing airTemperature both become NaN; others decode.
    assert np.isclose(df.iloc[0]["airTemperature"], 288.0)
    assert pd.isna(df.iloc[1]["airTemperature"])
    assert pd.isna(df.iloc[2]["airTemperature"])
    # Observation time from the per-report date/time group (missing minute -> 0).
    assert df.iloc[0]["time"] == pd.Timestamp("2024-06-01 12:00")
    assert df.iloc[1]["time"] == pd.Timestamp("2024-06-01 12:30")
    assert df.iloc[2]["time"] == pd.Timestamp("2024-06-01 12:00")
    assert np.isclose(df.iloc[0]["windSpeed"], 5.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_decode_bufr_empty(tmp_path, monkeypatch):

    monkeypatch.setattr(mod.pdbufr, "read_bufr", lambda *a, **k: pd.DataFrame())
    path = tmp_path / "synop.bin"
    path.write_bytes(b"BUFR")
    df = mod._decode_synop_bufr(str(path))
    assert df.empty
    assert list(df.columns) == list(mod._META_COLUMNS) + list(mod._BUFR_ELEMENTS) + [
        "_is_wmo"
    ]


@pytest.mark.timeout(15)
def test_dwd_synop_reports_decode_bufr_raises_returns_none(tmp_path, monkeypatch):

    # A pdbufr decode error (unsupported edition, corrupt content) must be
    # swallowed as None so the caller can classify it as a failed file.
    def _boom(*a, **k):
        raise ValueError("unsupported BUFR edition")

    monkeypatch.setattr(mod.pdbufr, "read_bufr", _boom)
    path = tmp_path / "synop.bin"
    path.write_bytes(b"BUFR")
    assert mod._decode_synop_bufr(str(path)) is None


@pytest.mark.timeout(15)
def test_dwd_synop_reports_decode_transform_tolerates_absent_column(
    tmp_path, monkeypatch
):

    # Scope: this checks that the DWD transform in _decode_synop_bufr tolerates a
    # column missing from pdbufr's output (here windSpeed) -- it fills the absent
    # variable with NaN rather than crashing. Whether real pdbufr.read_bufr
    # accepts a requested BUFR key absent from a bulletin is outside this mocked
    # test's scope.
    raw = pd.DataFrame(
        {
            "latitude": [54.0],
            "longitude": [8.0],
            "heightOfStationGroundAboveMeanSeaLevel": [10.0],
            "blockNumber": [10.0],
            "stationNumber": [1.0],
            "stationOrSiteName": ["Foo"],
            "year": [2024],
            "month": [6],
            "day": [1],
            "hour": [12],
            "minute": [0],
            "airTemperature": [288.0],
            "dewpointTemperature": [283.0],
            # windSpeed intentionally absent
            "windDirection": [0.0],
            "pressureReducedToMeanSeaLevel": [101300.0],
        }
    )
    captured = {}

    def _read_bufr(*a, **k):
        captured.update(k)
        return raw

    monkeypatch.setattr(mod.pdbufr, "read_bufr", _read_bufr)
    path = tmp_path / "synop.bin"
    path.write_bytes(b"BUFR")
    df = mod._decode_synop_bufr(str(path))
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["windSpeed"])
    assert np.isclose(df.iloc[0]["airTemperature"], 288.0)
    # Verify the sparse-report policy is forwarded to pdbufr (only position and
    # date/hour required). Exercising it against a genuinely sparse BUFR file is
    # outside this unit test's scope.
    assert captured["required_columns"] == mod._REQUIRED_BUFR_COLUMNS


@pytest.mark.timeout(10)
def test_dwd_synop_reports_create_tasks_selection():

    base = "https://opendata.dwd.de/weather/weather_reports/synoptic/germany/"

    def fn(ts):
        return f"{base}Z__C_EDZW_{ts}_bda01,synop_bufr_GER_999999_999999__MW_1.bin"

    names = [
        fn("20240601120000"),  # in window
        fn("20240601124500"),  # +45 min: delayed but within the 90-min lag pad
        fn("20240601140000"),  # +120 min: beyond lag pad -> excluded
        fn("20240601114000"),  # -20 min: before the 15-min lead pad -> excluded
        fn("20240601180000"),  # in the second (disjoint) window
        fn("20240601150000"),  # BETWEEN the two windows -> excluded
        base + "Z__C_EDZW_latest_bda01,synop_bufr_GER__MW_1.bin",  # 'latest' skip
        base + "readme.txt",  # non-.bin skip
    ]
    ds = DWDSynopReports(cache=False, verbose=False)
    ds.fs = MagicMock()
    ds.fs._ls = AsyncMock(return_value=names)

    # Two disjoint windows (12:00 and 18:00): files between them are NOT pulled.
    windows = [
        (datetime(2024, 6, 1, 12, 0), datetime(2024, 6, 1, 12, 0)),
        (datetime(2024, 6, 1, 18, 0), datetime(2024, 6, 1, 18, 0)),
    ]
    tasks = asyncio.run(ds._create_tasks(windows))
    assert sorted(t.file_time for t in tasks) == [
        datetime(2024, 6, 1, 12, 0),
        datetime(2024, 6, 1, 12, 45),
        datetime(2024, 6, 1, 18, 0),
    ]


@pytest.mark.timeout(15)
def test_dwd_synop_reports_feed_listing_failure_raises():
    # A listing error must surface as RuntimeError (an outage), not be swallowed
    # into an empty task list that would mask the outage as "no observations".
    ds = DWDSynopReports(cache=False, verbose=False)
    ds.fs = MagicMock()
    ds.fs._ls = AsyncMock(side_effect=OSError("boom"))
    windows = [(datetime(2024, 6, 1, 12, 0), datetime(2024, 6, 1, 12, 0))]
    with pytest.raises(RuntimeError, match="Failed to list"):
        asyncio.run(ds._create_tasks(windows))


@pytest.mark.timeout(15)
def test_dwd_synop_reports_empty_when_no_files(monkeypatch):
    ds = DWDSynopReports(cache=False, verbose=False)

    async def _no_tasks(*args, **kwargs):
        return []

    # No files in window -> a schema-typed empty frame (not an error).
    monkeypatch.setattr(ds, "_create_tasks", _no_tasks)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=1)
    df = ds(t, ["t2m"])
    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert df.attrs["source"] == ds.SOURCE_ID


# ======================================================================
# 4. Exception / error handling tests
# ======================================================================


@pytest.mark.timeout(10)
def test_dwd_synop_reports_exceptions():
    # Invalid variable in lexicon.
    with pytest.raises(KeyError):
        DWDSynopReportsLexicon["nonexistent_variable"]

    # Unsupported feed and out-of-range execution parameters are rejected.
    with pytest.raises(ValueError):
        DWDSynopReports(feed="mars")
    for bad in [{"async_workers": 0}, {"retries": -1}, {"async_timeout": 0}]:
        with pytest.raises(ValueError):
            DWDSynopReports(**bad)

    # Time older than the retention window.
    ds = DWDSynopReports(cache=False, verbose=False)
    with pytest.raises(ValueError):
        ds._validate_time([datetime(2000, 1, 1, 0)])
    # Time in the future.
    with pytest.raises(ValueError):
        ds._validate_time([datetime(2999, 1, 1, 0)])


# ======================================================================
# 5. Available classmethod test
# ======================================================================


@pytest.mark.timeout(5)
def test_dwd_synop_reports_available():
    now = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )
    assert DWDSynopReports.available(now - timedelta(hours=1)) is True
    assert DWDSynopReports.available(datetime(2000, 1, 1, 0)) is False
    assert DWDSynopReports.available(np.datetime64("2000-01-01T00:00")) is False
    # A timezone-aware datetime must not raise and is normalized to UTC.
    assert DWDSynopReports.available(datetime.now(timezone.utc)) is True
    # A missing time (NaT) must return False, not raise.
    assert DWDSynopReports.available(np.datetime64("NaT")) is False
    # A far-future date outside the datetime64[ns] range must not wrap into the
    # window (it would with a datetime64[ns] intermediate) and report available.
    assert DWDSynopReports.available(np.datetime64("2611-02-14")) is False
    # Beyond Python's datetime range (year > 9999) .item() yields an int, not a
    # datetime; the guard must reject it rather than raise on comparison.
    assert DWDSynopReports.available(np.datetime64("12000-01-01")) is False


# ======================================================================
# 6. resolve_fields and service-failure handling
# ======================================================================


@pytest.mark.timeout(5)
def test_dwd_synop_reports_resolve_fields():

    schema = DWDSynopReports.SCHEMA
    assert DWDSynopReports.resolve_fields(None) == schema
    assert DWDSynopReports.resolve_fields("time").names == ["time"]
    assert DWDSynopReports.resolve_fields(["time", "lat", "observation"]).names == [
        "time",
        "lat",
        "observation",
    ]
    sub = pa.schema([schema.field("observation"), schema.field("variable")])
    assert DWDSynopReports.resolve_fields(sub) == sub
    with pytest.raises(KeyError):
        DWDSynopReports.resolve_fields(["nonexistent_field"])
    with pytest.raises(TypeError):
        DWDSynopReports.resolve_fields(pa.schema([pa.field("time", pa.string())]))


@pytest.mark.timeout(15)
def test_dwd_synop_reports_all_downloads_failed(monkeypatch):

    ds = DWDSynopReports(cache=False, verbose=False)

    async def _one_task(*args, **kwargs):
        return [
            mod._DWDAsyncTask(url="https://x/f.bin", file_time=datetime(2024, 6, 1))
        ]

    async def _fail(task):
        return task.file_time, None, True

    monkeypatch.setattr(ds, "_create_tasks", _one_task)
    monkeypatch.setattr(ds, "_fetch_wrapper", _fail)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=1)
    with pytest.raises(RuntimeError):
        ds(t, ["t2m"])


@pytest.mark.timeout(15)
def test_dwd_synop_reports_mixed_prune_and_failure_raises(monkeypatch):
    # Nothing fetched because one file failed and one was pruned. The raise must
    # report both counts, not just the failure count.

    ds = DWDSynopReports(cache=False, verbose=False)

    async def _two_tasks(*args, **kwargs):
        return [
            mod._DWDAsyncTask(url="https://x/fail.bin", file_time=datetime(2024, 6, 1)),
            mod._DWDAsyncTask(
                url="https://x/pruned.bin", file_time=datetime(2024, 6, 1)
            ),
        ]

    async def _fetch(task):
        # fail.bin is a genuine failure; pruned.bin is an expected live-feed miss.
        failed = task.url.endswith("fail.bin")
        return task.file_time, None, failed

    monkeypatch.setattr(ds, "_create_tasks", _two_tasks)
    monkeypatch.setattr(ds, "_fetch_wrapper", _fetch)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=1)
    with pytest.raises(RuntimeError, match="pruned"):
        ds(t, ["t2m"])


@pytest.mark.timeout(15)
def test_dwd_synop_reports_all_pruned_returns_empty(tmp_path, monkeypatch):
    # A file present in the listing is pruned from the live feed before download
    # (its _cat_file raises FileNotFoundError). This must be classified as an
    # expected miss -> empty frame, not a "feed unavailable" RuntimeError. Drives
    # the real _create_tasks -> _fetch_wrapper -> _fetch_remote_file path.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    bulletin = t - timedelta(minutes=10)
    fname = f"{mod.GERMANY_URL}Z__C_EDZW_{bulletin:%Y%m%d%H%M%S}_synop.bin"

    class _FS:
        async def _ls(self, path, detail=False):
            return [fname]

        async def _cat_file(self, url):
            raise FileNotFoundError(url)

    ds = DWDSynopReports(time_tolerance=timedelta(minutes=30), verbose=False)
    ds.fs = _FS()

    df = ds(t, ["t2m"])
    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "bbox",
    [
        (47.0, 5.8, 55.1, 15.1),  # [-180, 180] box
        (30.0, 270.0, 40.0, 285.0),  # [0, 360] box
    ],
)
def test_dwd_synop_reports_valid_bbox(bbox):
    DWDSynopReports(lat_lon_bbox=bbox)  # does not raise


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "bbox",
    [
        (55.1, 5.8, 47.0, 15.1),  # inverted latitude
        (47.0, 170.0, 55.1, -170.0),  # antimeridian-crossing ([-180, 180] seam)
        (47.0, 350.0, 55.1, 10.0),  # seam-crossing in [0, 360] (lon_min > lon_max)
        (-100.0, 5.0, 100.0, 15.0),  # latitude out of [-90, 90]
        (47.0, 300.0, 55.1, 400.0),  # longitude out of [0, 360]
        (47.0, -10.0, 55.1, 200.0),  # mixed longitude conventions
        (47.0, 5.8, 55.1),  # wrong length
    ],
)
def test_dwd_synop_reports_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        DWDSynopReports(lat_lon_bbox=bbox)


@pytest.mark.timeout(5)
def test_dwd_synop_reports_filter_bbox_seam_endpoints():
    # Stations across the [0, 360) seam and both antimeridian representations.
    # lon is stored in [0, 360); a station at 0/360 is canonicalized to 0.0.
    df = pd.DataFrame(
        {
            "lat": [50.0, 50.0, 50.0, 50.0],
            "lon": [0.0, 5.0, 180.0, 355.0],
        }
    )

    def lons(bbox):
        return sorted(DWDSynopReports._filter_bbox(df, bbox)["lon"].tolist())

    # Box ending on the seam (lon_max == 360) includes the canonical 0.0 but not
    # a nearby eastern longitude (5.0).
    assert lons((45.0, 350.0, 55.0, 360.0)) == [0.0, 355.0]
    # Full [0, 360] box covers every longitude.
    assert lons((45.0, 0.0, 55.0, 360.0)) == [0.0, 5.0, 180.0, 355.0]
    # Antimeridian endpoints in either convention include lon 180 (stored 180.0).
    assert lons((45.0, 170.0, 55.0, 180.0)) == [180.0]
    assert lons((45.0, -180.0, 55.0, -170.0)) == [180.0]
    # [-180, 180] box ending on the antimeridian (lon_max == 180) includes the
    # station stored at 180.0 (which canonicalizes to -180.0 in that convention).
    # Box (-3, 180): includes 0.0/5.0, includes 180.0 via the seam guard, and
    # excludes 355.0 (== -5.0, below lon_min).
    assert lons((45.0, -3.0, 55.0, 180.0)) == [0.0, 5.0, 180.0]


@pytest.mark.timeout(15)
def test_dwd_synop_reports_all_decodes_failed(monkeypatch):

    ds = DWDSynopReports(cache=False, verbose=False)

    async def _one_task(*args, **kwargs):
        return [
            mod._DWDAsyncTask(url="https://x/f.bin", file_time=datetime(2024, 6, 1))
        ]

    async def _ok_fetch(task):
        return task.file_time, "unused_path.bin", False

    # Downloads succeed but every BUFR file fails to decode -> raise (a decoder
    # or feed problem must not be mistaken for an empty feed).
    monkeypatch.setattr(ds, "_create_tasks", _one_task)
    monkeypatch.setattr(ds, "_fetch_wrapper", _ok_fetch)
    monkeypatch.setattr(mod, "_decode_synop_bufr", lambda path: None)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=1)
    with pytest.raises(RuntimeError):
        ds(t, ["t2m"])


@pytest.mark.timeout(15)
def test_dwd_synop_reports_partial_decode(monkeypatch):

    ds = DWDSynopReports(cache=False, verbose=False)
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=1)

    async def _two_tasks(*args, **kwargs):
        return [
            mod._DWDAsyncTask(url="https://x/a.bin", file_time=t),
            mod._DWDAsyncTask(url="https://x/b.bin", file_time=t),
        ]

    async def _ok_fetch(task):
        return task.file_time, task.url, False

    good = pd.DataFrame(
        {c: [] for c in list(mod._META_COLUMNS) + list(mod._BUFR_ELEMENTS)}
    )
    good.loc[0] = ["10015", pd.Timestamp(t), 54.0, 8.0, 10.0, 288.0, *[np.nan] * 4]

    # One file decodes, the other fails -> partial result returned (no raise).
    def _decode(path):
        return good.copy() if path.endswith("a.bin") else None

    monkeypatch.setattr(ds, "_create_tasks", _two_tasks)
    monkeypatch.setattr(ds, "_fetch_wrapper", _ok_fetch)
    monkeypatch.setattr(mod, "_decode_synop_bufr", _decode)
    df = ds(t, ["t2m"])
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["station"]) == {"10015"}
    t2m = df[df["variable"] == "t2m"]["observation"]
    assert np.isclose(t2m.iloc[0], 288.0)


@pytest.mark.timeout(10)
def test_dwd_synop_reports_atomic_write_cleanup(tmp_path, monkeypatch):

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    ds = DWDSynopReports(verbose=False)
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)

    class _FS:
        async def _cat_file(self, url):
            return b"payload"

    ds.fs = _FS()

    def _boom(src, dst):
        raise OSError("replace failed")

    # A failed atomic rename must not leave a stray .tmp file behind.
    monkeypatch.setattr(mod.os, "replace", _boom)
    with pytest.raises(OSError):
        asyncio.run(ds._fetch_remote_file("https://x/f.bin"))
    assert not list(pathlib.Path(ds.cache).rglob("*.tmp"))


# ======================================================================
# 8. International feed (BUFR-only; non-BUFR/TAC bulletins are skipped)
# ======================================================================


@pytest.mark.timeout(15)
def test_dwd_synop_reports_international_bufr_call(tmp_path, monkeypatch):
    # End-to-end international fetch: a BUFR file is decoded like the germany feed.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    bulletin = t - timedelta(minutes=10)
    fname = f"{mod.INTERNATIONAL_URL}Z__C_EDZW_{bulletin:%Y%m%d%H%M%S}_synop.bin"

    class _FS:
        async def _ls(self, path, detail=False):
            return [fname]

        async def _cat_file(self, url):
            return b"BUFR\x00\x00payload"  # sniffed as BUFR

    raw = pd.DataFrame(
        {
            "latitude": [48.1],
            "longitude": [11.5],
            "heightOfStationGroundAboveMeanSeaLevel": [500.0],
            "blockNumber": [10.0],
            "stationNumber": [870.0],
            "stationOrSiteName": ["MUNICH"],
            "year": [t.year],
            "month": [t.month],
            "day": [t.day],
            "hour": [t.hour],
            "minute": [0],
            "airTemperature": [290.0],
            "dewpointTemperature": [285.0],
            "windSpeed": [3.0],
            "windDirection": [180.0],
            "pressureReducedToMeanSeaLevel": [101500.0],
        }
    )
    monkeypatch.setattr(mod.pdbufr, "read_bufr", lambda *a, **k: raw)

    ds = DWDSynopReports(
        feed="international", time_tolerance=timedelta(minutes=30), verbose=False
    )
    ds.fs = _FS()
    df = ds(t, ["t2m", "msl"])
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["station"].unique()) == {"10870"}
    got = df.set_index("variable")["observation"]
    assert np.isclose(got["t2m"], 290.0)
    assert np.isclose(got["msl"], 101500.0)


@pytest.mark.timeout(15)
def test_dwd_synop_reports_decode_paths_skips_tac(tmp_path, monkeypatch):
    # On the international feed: BUFR is decoded, a recognised TAC (AAXX/BBXX)
    # file is skipped+counted, and an unknown non-BUFR payload is a decode
    # failure -- none are silently treated as empty.

    t = datetime(2026, 7, 25, 0)
    bufr_path = tmp_path / "a.bin"
    bufr_path.write_bytes(b"BUFR\x00\x00stuff")
    tac_path = tmp_path / "b.bin"
    tac_path.write_bytes(b"\r\r\nSIXX01 EDZW 250000\r\r\nAAXX 25001\r\r\n72219 ...=")
    junk_path = tmp_path / "c.bin"
    junk_path.write_bytes(b"<html><body>503 Service Unavailable</body></html>")

    good = pd.DataFrame(
        [["10870", pd.Timestamp(t), 48.1, 11.5, 500.0, 290.0, *[np.nan] * 4]],
        columns=list(mod._META_COLUMNS) + list(mod._BUFR_ELEMENTS),
    )
    monkeypatch.setattr(mod, "_decode_synop_bufr", lambda p: good.copy())

    ds = DWDSynopReports(feed="international", cache=False, verbose=False)
    wide, n_failed, n_tac_skipped = ds._decode_paths(
        [(t, str(bufr_path)), (t, str(tac_path)), (t, str(junk_path))]
    )
    assert n_tac_skipped == 1  # the AAXX/TAC file was skipped, not decoded
    assert n_failed == 1  # the junk/HTML payload is a decode failure, not TAC
    assert set(wide["station"]) == {"10870"}


@pytest.mark.timeout(15)
def test_dwd_synop_reports_international_all_tac_raises(tmp_path, monkeypatch):
    # A window whose only file is TAC (no BUFR) must raise, not return empty.

    monkeypatch.setattr(
        "earth2studio.data.dwd_synop_reports.datasource_cache_root",
        lambda: str(tmp_path),
    )
    t = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=2)
    bulletin = t - timedelta(minutes=10)
    fname = f"{mod.INTERNATIONAL_URL}Z__C_EDZW_{bulletin:%Y%m%d%H%M%S}_synop.bin"

    class _FS:
        async def _ls(self, path, detail=False):
            return [fname]

        async def _cat_file(self, url):
            return b"\r\r\nSIRA20 RUHB 000000\r\r\nAAXX 25001\r\r\n72219 ...="  # TAC

    ds = DWDSynopReports(
        feed="international", time_tolerance=timedelta(minutes=30), verbose=False
    )
    ds.fs = _FS()
    with pytest.raises(RuntimeError, match="traditional alphanumeric"):
        ds(t, ["t2m"])
