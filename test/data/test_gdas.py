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
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import earth2studio.data.gdas as gdas_data
from earth2studio.data import NomadsGDASObsConv, utils_ncep
from earth2studio.lexicon import GDASObsConvLexicon


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        datetime(2026, 4, 4, 0),
        [datetime(2026, 4, 4, 0), datetime(2026, 4, 4, 6)],
    ],
)
@pytest.mark.parametrize("variable", ["t", ["t", "pres"]])
def test_nomads_gdas_fetch(time, variable):
    ds = NomadsGDASObsConv(
        time_tolerance=timedelta(minutes=180),
        cache=False,
    )
    df = ds(time, variable)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ds.SCHEMA.names
    assert not df.empty

    if isinstance(variable, str):
        variable = [variable]
    assert set(df["variable"].unique()).issubset(set(variable))

    # Validate coordinate ranges
    assert (df["lat"] >= -90).all() and (df["lat"] <= 90).all()
    assert (df["lon"] >= 0).all() and (df["lon"] < 360).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("variable", [["u", "v"]])
def test_nomads_gdas_wind(variable):
    ds = NomadsGDASObsConv(
        time_tolerance=timedelta(minutes=180),
        cache=False,
    )
    time = datetime(2026, 4, 4, 0)
    df = ds(time, variable)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df["variable"].unique()).issubset({"u", "v"})

    # Wind components should have reasonable values
    obs = df["observation"].dropna()
    assert (obs.abs() < 200).all()  # m/s, no wind > 200 m/s


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("2026-04-04T00:00")])],
)
@pytest.mark.parametrize("variable", [["t", "pres"]])
@pytest.mark.parametrize("cache", [True, False])
def test_nomads_gdas_cache(time, variable, cache):
    ds = NomadsGDASObsConv(
        time_tolerance=timedelta(minutes=180),
        cache=cache,
    )
    df = ds(time, variable)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Reload from cache if caching
    df2 = ds(time, variable[0])
    assert isinstance(df2, pd.DataFrame)

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_nomads_gdas_schema_fields():
    ds = NomadsGDASObsConv()

    schema_full = ds.resolve_fields(None)
    assert schema_full.equals(ds.SCHEMA, check_metadata=True)

    subset = ["time", "lat", "lon", "observation", "variable"]
    schema_subset = ds.resolve_fields(subset)
    assert schema_subset.names == subset

    schema_str = ds.resolve_fields("observation")
    assert schema_str.names == ["observation"]

    with pytest.raises(KeyError):
        ds.resolve_fields("nonexistent_field")

    schema = ds.resolve_fields(ds.SCHEMA)
    assert schema.equals(ds.SCHEMA, check_metadata=True)

    wrong_type = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        ds.resolve_fields(wrong_type)


@pytest.mark.timeout(15)
def test_nomads_gdas_exceptions():
    ds = NomadsGDASObsConv()

    # Invalid variable – use a recent time so validation passes before KeyError
    recent = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)
    with pytest.raises(KeyError):
        ds(recent, "nonexistent_var")

    with pytest.raises(KeyError):
        ds(recent, "t", fields="nonexistent_field")

    # Future time
    with pytest.raises(ValueError):
        ds(datetime(2030, 1, 1, 0), "t")

    # Too old
    with pytest.raises(ValueError):
        ds(datetime(2020, 1, 1, 0), "t")


@pytest.mark.asyncio
async def test_nomads_gdas_invalid_requests_do_not_initialize_store(monkeypatch):
    ds = NomadsGDASObsConv(cache=True, verbose=False)
    initialized = False

    async def initialize():
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(ds._store, "_async_init", initialize)
    recent = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)

    with pytest.raises(ValueError):
        await ds.fetch(datetime(2020, 1, 1, 0), "t")
    with pytest.raises(KeyError):
        await ds.fetch(recent, "t", fields="nonexistent_field")
    with pytest.raises(KeyError):
        await ds.fetch(recent, "nonexistent_var")

    assert not initialized


@pytest.mark.timeout(15)
def test_nomads_gdas_available():
    # Future time should not be available
    assert not NomadsGDASObsConv.available(datetime(2030, 1, 1))

    # Very old time should not be available
    assert not NomadsGDASObsConv.available(datetime(2020, 1, 1))

    # numpy datetime64
    assert not NomadsGDASObsConv.available(np.datetime64("2030-01-01"))

    # Recent time should be available (use relative time within 2-day window)
    recent = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)
    assert NomadsGDASObsConv.available(recent)


@pytest.mark.timeout(15)
def test_nomads_gdas_url_builder():
    url = NomadsGDASObsConv._build_prepbufr_uri(datetime(2026, 4, 5, 12))
    expected = (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/"
        "gdas.20260405/gdas.t12z.prepbufr.nr"
    )
    assert url == expected

    gpsro_url = NomadsGDASObsConv._build_gpsro_uri(datetime(2026, 4, 5, 12))
    assert gpsro_url == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/"
        "gdas.20260405/gdas.t12z.gpsro.tm00.bufr_d.nr"
    )


@pytest.mark.timeout(15)
def test_nomads_gdas_lexicon():
    # All expected variables should be in the lexicon
    expected_vars = ["u", "v", "t", "q", "pres", "gps"]
    for var in expected_vars:
        assert var in GDASObsConvLexicon.VOCAB

    # get_item should return (key, modifier)
    key, mod = GDASObsConvLexicon.get_item("t")
    assert key == "TOB"
    assert callable(mod)

    # Wind vars should have decomposition modifiers
    key_u, mod_u = GDASObsConvLexicon.get_item("u")
    assert key_u == "wind::u"

    key_v, mod_v = GDASObsConvLexicon.get_item("v")
    assert key_v == "wind::v"

    key_gps, mod_gps = GDASObsConvLexicon.get_item("gps")
    assert key_gps == "gpsro::15037"
    assert callable(mod_gps)
    assert not {"gps_l1", "gps_l2", "gps_t", "gps_q"}.intersection(
        GDASObsConvLexicon.VOCAB
    )


@pytest.mark.timeout(15)
def test_nomads_gdas_cache_path():
    ds = NomadsGDASObsConv(cache=True)
    url1 = "https://example.com/gdas.20260405/gdas.t00z.prepbufr.nr"
    url2 = "https://example.com/gdas.20260405/gdas.t06z.prepbufr.nr"

    path1 = ds._cache_path(url1)
    path2 = ds._cache_path(url2)

    # Different URLs should produce different cache paths
    assert path1 != path2

    # Same URL should produce same cache path
    assert ds._cache_path(url1) == path1

    # Should be in the cache directory
    assert path1.startswith(ds.cache)
    assert path1.endswith(".bin")


@pytest.mark.timeout(15)
def test_nomads_gdas_tolerance_conversion():
    # Symmetric tolerance: lower is negative, upper is positive
    ds1 = NomadsGDASObsConv(time_tolerance=timedelta(hours=3))
    assert ds1._tolerance_lower == timedelta(hours=-3)
    assert ds1._tolerance_upper == timedelta(hours=3)

    # Asymmetric tolerance: (lower, upper) passed through as-is
    ds2 = NomadsGDASObsConv(time_tolerance=(timedelta(hours=-1), timedelta(hours=6)))
    assert ds2._tolerance_lower == timedelta(hours=-1)
    assert ds2._tolerance_upper == timedelta(hours=6)


@pytest.mark.timeout(30)
def test_nomads_gdas_call_mock(tmp_path):
    # Use a recent time that passes validation
    base_time = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)
    obs_time1 = base_time + timedelta(minutes=30)
    obs_time2 = base_time + timedelta(hours=1)

    # Create a mock DataFrame that would come from _decode_prepbufr
    # Observation values are in raw PrepBUFR units (t in DEG C)
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime([obs_time1, obs_time2]),
            "pres": np.array([50000.0, 85000.0], dtype=np.float32),
            "elev": np.array([5000.0, 1500.0], dtype=np.float32),
            "type": np.array([101, 106], dtype=np.uint16),
            "level_cat": pd.array([1, 1], dtype="uint16[pyarrow]"),
            "class": ["1", "1"],
            "lat": np.array([40.0, 35.0], dtype=np.float32),
            "lon": np.array([250.0, 280.0], dtype=np.float32),
            "station": ["72451", "72520"],
            "station_elev": np.array([300.0, 200.0], dtype=np.float32),
            "quality": np.array([0, 0], dtype=np.uint16),
            "pressure_quality": pd.array([1, 1], dtype="uint16[pyarrow]"),
            "observation": np.array([250.0, 288.0], dtype=np.float32),
            "variable": ["t", "t"],
        }
    )

    dummy_file = tmp_path / "mock.bin"
    dummy_file.write_bytes(b"dummy")
    ds = NomadsGDASObsConv(
        time_tolerance=timedelta(hours=3),
        cache=False,
    )
    fetched: list[str] = []

    async def mock_fetch_files(urls):
        fetched.extend(urls)

    ds._store.fetch_files = mock_fetch_files
    ds._store.local_path = lambda _url: str(dummy_file)
    with patch.object(ds, "_decode_file", return_value=mock_df) as decode:
        df = ds(base_time, ["t"])

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()) == {"t"}
    assert fetched
    decode.assert_called()


def test_nomads_gdas_fetch_uses_store(tmp_path, monkeypatch):
    cached_file = tmp_path / "cached.bufr"
    cached_file.write_bytes(b"fixture")

    class FakeStore:
        def __init__(self):
            self.fetched: list[str] = []
            self.cleanup_calls = 0

        async def fetch_files(self, uris):
            self.fetched = list(uris)

        def local_path(self, uri):
            return str(cached_file)

        def cleanup(self):
            self.cleanup_calls += 1

    cycle = datetime.now(timezone.utc).replace(
        tzinfo=None, minute=0, second=0, microsecond=0
    )
    cycle = cycle.replace(hour=(cycle.hour // 6) * 6)
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime([cycle]),
            "observation": [273.15],
            "variable": ["t"],
        }
    )
    source = NomadsGDASObsConv(
        time_tolerance=timedelta(0),
        cache=True,
        verbose=False,
    )
    store = FakeStore()
    source._store = store
    monkeypatch.setattr(source, "_decode_file", lambda path, task: frame)

    result = source(
        cycle,
        ["t"],
        fields=["time", "observation", "variable"],
    )

    assert store.fetched == [source._build_prepbufr_uri(cycle)]
    assert result.equals(frame)
    assert result.attrs == {"source": source.SOURCE_ID}
    assert store.cleanup_calls == 1

    async def fail_fetch(uris):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(store, "fetch_files", fail_fetch)
    with pytest.raises(RuntimeError):
        source(
            cycle,
            ["t"],
            fields=["time", "observation", "variable"],
        )
    assert store.cleanup_calls == 2


@pytest.mark.timeout(15)
def test_nomads_gdas_create_tasks():
    ds = NomadsGDASObsConv(
        time_tolerance=timedelta(hours=3),
    )

    # Single time, single variable
    tasks = ds._create_tasks(
        [datetime(2026, 4, 4, 0)],
        ["t"],
    )
    assert len(tasks) >= 1
    assert all(t.uri.endswith(".prepbufr.nr") for t in tasks)
    assert all("t" in t.var_plan for t in tasks)

    # Task URLs should be for cycle times around midnight
    for task in tasks:
        assert "gdas." in task.uri

    mixed_ds = NomadsGDASObsConv(time_tolerance=timedelta(0))
    mixed_tasks = mixed_ds._create_tasks([datetime(2026, 4, 4, 0)], ["t", "gps"])
    assert len(mixed_tasks) == 2
    prepbufr_task = next(task for task in mixed_tasks if task.route == "prepbufr")
    gpsro_task = next(task for task in mixed_tasks if task.route == "gpsro")
    assert prepbufr_task.uri.endswith(".prepbufr.nr")
    assert gpsro_task.uri.endswith(".gpsro.tm00.bufr_d.nr")
    assert set(prepbufr_task.var_plan) == {"t"}
    assert set(gpsro_task.var_plan) == {"gps"}
    prepbufr_key, prepbufr_modifier = prepbufr_task.var_plan["t"]
    gpsro_descriptor, gpsro_modifier = gpsro_task.var_plan["gps"]
    assert isinstance(prepbufr_key, str)
    assert gpsro_descriptor == utils_ncep.GPSRO_BNDA
    assert callable(prepbufr_modifier)
    assert callable(gpsro_modifier)

    windowed_ds = NomadsGDASObsConv(time_tolerance=timedelta(minutes=20))
    windowed_tasks = windowed_ds._create_tasks(
        [datetime(2026, 4, 4, 1), datetime(2026, 4, 4, 2)], ["t", "gps"]
    )
    assert len(windowed_tasks) == 2
    assert {task.route for task in windowed_tasks} == {"prepbufr", "gpsro"}
    assert all(
        task.datetime_min == datetime(2026, 4, 4, 0, 40)
        and task.datetime_max == datetime(2026, 4, 4, 2, 20)
        for task in windowed_tasks
    )


@pytest.mark.timeout(15)
def test_nomads_gdas_empty_result():
    ds = NomadsGDASObsConv()
    empty_df = ds._compile_dataframe([], ds.SCHEMA)

    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty
    assert list(empty_df.columns) == ds.SCHEMA.names


@pytest.mark.asyncio
async def test_nomads_store_fetches_and_reuses_cached_files(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    store = gdas_data._NomadsObsStore(
        cache=True,
        verbose=False,
        max_workers=2,
        retries=4,
    )
    requests: list[str] = []
    initialized = False

    class FakeHTTPFileSystem:
        async def _cat_file(self, url):
            requests.append(url)
            return url.encode()

    async def initialize():
        nonlocal initialized
        initialized = True
        store.fs = FakeHTTPFileSystem()

    monkeypatch.setattr(store, "_async_init", initialize)
    urls = [
        "https://example.com/gdas.t00z.prepbufr.nr",
        "https://example.com/gdas.t06z.prepbufr.nr",
    ]

    await store.fetch_files(urls)

    assert initialized
    assert sorted(requests) == sorted(urls)
    for url in urls:
        assert pathlib.Path(store.local_path(url)).read_bytes() == url.encode()

    await store.fetch_files(urls)
    assert sorted(requests) == sorted(urls)


@pytest.mark.asyncio
async def test_nomads_store_preserves_retry_and_concurrency_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    store = gdas_data._NomadsObsStore(True, False, 3, 5)
    store.fs = object()
    retries: list[tuple[str, dict]] = []
    concurrency: dict = {}

    async def fake_retry(_function, uri, **kwargs):
        retries.append((uri, kwargs))

    async def fake_gather(coros, **kwargs):
        concurrency.update(kwargs)
        await asyncio.gather(*coros)

    monkeypatch.setattr(gdas_data, "async_retry", fake_retry)
    monkeypatch.setattr(gdas_data, "gather_with_concurrency", fake_gather)

    await store.fetch_files(["first", "second"])

    assert [uri for uri, _kwargs in retries] == ["first", "second"]
    assert all(
        kwargs
        == {
            "retries": 5,
            "backoff": 1.0,
            "task_timeout": 120.0,
            "exceptions": (OSError, IOError, TimeoutError, ConnectionError),
        }
        for _uri, kwargs in retries
    )
    assert concurrency == {
        "max_workers": 3,
        "desc": "Fetching GDAS conventional observations",
        "verbose": True,
    }


def test_nomads_store_temporary_cache_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    first = gdas_data._NomadsObsStore(False, False, 1, 0)
    second = gdas_data._NomadsObsStore(False, False, 1, 0)
    first_cache = pathlib.Path(first.cache)

    assert first.cache == str(first_cache)
    assert first.cache != second.cache
    first_cache.mkdir(parents=True)
    pathlib.Path(first.local_path("https://example.com/file")).write_bytes(b"raw")

    first.cleanup()

    assert not first_cache.exists()


def test_nomads_gdas_cache_false_cleans_up_after_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    source = NomadsGDASObsConv(cache=False, verbose=False)
    cache = pathlib.Path(source.cache)
    cache.mkdir(parents=True)
    (cache / "raw").write_bytes(b"raw")

    with pytest.raises(ValueError):
        source(datetime(2020, 1, 1), "t")

    assert not cache.exists()


@pytest.mark.timeout(15)
def test_nomads_gdas_lexicon_modifiers():
    """Verify lexicon modifiers convert raw PrepBUFR units to E2S standard."""
    # Temperature: DEG C -> K (+273.15)
    _, mod_t = GDASObsConvLexicon.get_item("t")
    df_t = pd.DataFrame({"observation": np.array([0.0, 20.0, -40.0], dtype=np.float32)})
    df_t = mod_t(df_t)
    np.testing.assert_allclose(df_t["observation"], [273.15, 293.15, 233.15], rtol=1e-5)

    # Specific humidity: mg/kg -> kg/kg (÷1e6)
    _, mod_q = GDASObsConvLexicon.get_item("q")
    df_q = pd.DataFrame({"observation": np.array([10000.0, 5000.0], dtype=np.float32)})
    df_q = mod_q(df_q)
    np.testing.assert_allclose(df_q["observation"], [0.01, 0.005], rtol=1e-5)

    # Pressure: hPa (MB) -> Pa (×100)
    _, mod_p = GDASObsConvLexicon.get_item("pres")
    df_p = pd.DataFrame({"observation": np.array([1013.25, 500.0], dtype=np.float32)})
    df_p = mod_p(df_p)
    np.testing.assert_allclose(df_p["observation"], [101325.0, 50000.0], rtol=1e-5)

    # Wind u/v: already m/s, no change
    _, mod_u = GDASObsConvLexicon.get_item("u")
    df_u = pd.DataFrame({"observation": np.array([5.0, -3.0], dtype=np.float32)})
    df_u = mod_u(df_u)
    np.testing.assert_allclose(df_u["observation"], [5.0, -3.0], rtol=1e-5)

    _, mod_v = GDASObsConvLexicon.get_item("v")
    df_v = pd.DataFrame({"observation": np.array([2.0, -7.0], dtype=np.float32)})
    df_v = mod_v(df_v)
    np.testing.assert_allclose(df_v["observation"], [2.0, -7.0], rtol=1e-5)
