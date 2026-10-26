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
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from loguru import logger

import earth2studio.data.nnja as nnja
from earth2studio.data import (
    NNJAObsConv,
    NNJAObsSat,
    NNJAObsSatwnd,
    fetch_dataframe,
    utils_bufr,
    utils_ncep,
)
from earth2studio.data.utils import radiance_to_bt
from earth2studio.data.utils_bufr import datetime_ns
from earth2studio.data.utils_ir import wavenumber_cm_inverse
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    NCEP_MICROWAVE_OUTPUT_SCHEMA,
    NCEP_SATWND_PUBLIC_SCHEMA,
    _to_long,
    channel_axis,
    compile_tables,
    cycle_windows,
    decode_gpsro,
    decode_prepbufr,
    decode_satellite,
    decode_satwnd,
    observation_cycle_times,
)

pytest.importorskip("earth2bufr", reason="earth2bufr not installed")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [datetime(year=2024, month=1, day=1, hour=0)],
)
@pytest.mark.parametrize(
    "variable, tol",
    [
        (["t"], timedelta(0)),
        (["u", "v"], timedelta(0)),
    ],
)
def test_nnja_obs_conv_fetch(time, variable, tol):
    ds = NNJAObsConv(time_tolerance=tol, cache=False, verbose=False, decode_workers=16)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert not df.empty


@pytest.mark.parametrize("cache", [True, False])
def test_nnja_obs_conv_cache_mock(cache, tmp_path):
    """Test NNJAObsConv cache behavior with mocked S3 fetch."""
    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "report_time": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:00:00"]
            ),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "level_cat": [0, 0],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "report_lat": [40.0, 41.0],
            "report_lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "pressure_quality": [1, 1],
            "radius_curvature": [np.nan, np.nan],
            "geoid_undulation": [np.nan, np.nan],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
            "cycle_time": pd.to_datetime(["2024-01-01 00:00:00"] * 2),
        }
    )

    with patch("earth2studio.data.nnja._sync_async") as mock_sync:
        mock_sync.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=cache, verbose=False)

        # First fetch
        df = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df.columns) == ds.SCHEMA.names

        # Second fetch (should use cache if enabled)
        df2 = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df2.columns) == ds.SCHEMA.names

    # Clean up
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_nnja_obs_conv_concurrent_calls_share_temp_cache(monkeypatch):
    ds = NNJAObsConv(cache=False, verbose=False)
    cache_dir = ds.cache
    slow_started, fast_done = threading.Event(), threading.Event()

    async def fetch(time, variable, fields):
        path = os.path.join(ds.cache, f"{variable}.bin")
        with open(path, "wb") as f:
            f.write(b"bufr")
        if variable == "slow":
            slow_started.set()
            await asyncio.to_thread(fast_done.wait, 30)
        with open(path, "rb") as f:
            assert f.read() == b"bufr"
        return pd.DataFrame()

    monkeypatch.setattr(ds, "fetch", fetch)
    with ThreadPoolExecutor(1) as pool:
        slow = pool.submit(ds, datetime(2024, 1, 1), "slow")
        assert slow_started.wait(30)
        ds(datetime(2024, 1, 1), "fast")
        assert os.path.isdir(cache_dir)
        fast_done.set()
        slow.result(timeout=30)
    assert not os.path.exists(cache_dir)


def test_nnja_obs_conv_exceptions():

    # Invalid source
    with pytest.raises(ValueError):
        NNJAObsConv(source="not_a_source", cache=False, verbose=False)

    with pytest.raises(NotImplementedError, match="raw dump streams"):
        NNJAObsConv(source="convbufr", cache=False, verbose=False)

    # Invalid variable - test via lexicon lookup directly (avoids network)
    with pytest.raises(KeyError):
        from earth2studio.lexicon import NNJAObsConvLexicon

        NNJAObsConvLexicon["invalid_variable"]

    # Invalid fields - test via resolve_fields directly (avoids network)
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["observation", "variable", "invalid_field"])

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type_schema)


def test_nnja_obs_conv_original_event():
    latest = NNJAObsConv(cache=False, verbose=False)
    original = NNJAObsConv(original_event=True, cache=False, verbose=False)

    assert not latest._original_event
    assert original._original_event


@pytest.mark.parametrize("year, expected", [(99, 1999), (100, 2000), (6, 2006)])
def test_edition3_year_of_century(year, expected):
    message = SimpleNamespace(
        source_message_index=0,
        edition=3,
        year=year,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
    )
    info = SimpleNamespace(data_messages=[message], messages=[message])
    base_us = utils_ncep._message_base_times(info)
    assert base_us[0] == datetime_ns(datetime(expected, 1, 1)) // 1000


def test_nnja_obs_conv_validate_time():
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 0)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 6)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 12)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 18)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 1)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 0, 30)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(1970, 1, 1, 0)])


def test_nnja_obs_conv_tolerance_conversion():

    ds = NNJAObsConv(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
    assert ds._tolerance_lower == timedelta(hours=-1)
    assert ds._tolerance_upper == timedelta(hours=1)

    ds_np = NNJAObsConv(
        time_tolerance=np.timedelta64(2, "h"), cache=False, verbose=False
    )
    assert ds_np._tolerance_lower == timedelta(hours=-2)
    assert ds_np._tolerance_upper == timedelta(hours=2)

    ds_asym = NNJAObsConv(
        time_tolerance=(np.timedelta64(-3, "h"), np.timedelta64(1, "h")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(hours=-3)
    assert ds_asym._tolerance_upper == timedelta(hours=1)


def test_nnja_obs_conv_resolve_fields():
    schema_full = NNJAObsConv.resolve_fields(None)
    assert schema_full.names == NNJAObsConv.SCHEMA.names

    schema_subset = NNJAObsConv.resolve_fields(
        ["time", "lat", "lon", "observation", "variable"]
    )
    assert schema_subset.names == ["time", "lat", "lon", "observation", "variable"]

    schema_str = NNJAObsConv.resolve_fields("time")
    assert schema_str.names == ["time"]

    sub = pa.schema(
        [
            NNJAObsConv.SCHEMA.field("time"),
            NNJAObsConv.SCHEMA.field("observation"),
        ]
    )
    out = NNJAObsConv.resolve_fields(sub)
    assert out.names == ["time", "observation"]

    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["nonexistent"])

    bad_schema = pa.schema([pa.field("nonexistent", pa.float32())])
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(bad_schema)

    wrong_type = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type)


def test_nnja_obs_conv_mock_fetch():
    """Test NNJAObsConv data processing with mocked S3 fetch."""

    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "report_time": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:00:00"]
            ),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "level_cat": [0, 0],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "report_lat": [40.0, 41.0],
            "report_lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "pressure_quality": [1, 1],
            "radius_curvature": [np.nan, np.nan],
            "geoid_undulation": [np.nan, np.nan],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
            "cycle_time": pd.to_datetime(["2024-01-01 00:00:00"] * 2),
        }
    )

    with patch.object(NNJAObsConv, "fetch") as mock_fetch:
        mock_fetch.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

        # Patch _sync_async to call the mock directly
        with patch("earth2studio.data.nnja._sync_async") as mock_sync:
            mock_sync.return_value = mock_df
            df = ds(datetime(2024, 1, 1, 0), ["t"])

    assert list(df.columns) == ds.SCHEMA.names
    assert len(df) == 2
    assert set(df["variable"].unique()) == {"t"}
    assert df["observation"].iloc[0] == pytest.approx(273.15)


def test_nnja_obs_conv_fetch_uses_store(tmp_path, monkeypatch):
    """Exercise the composed request/transport seam through __call__.

    Verifies the request path drives the transport ``fetch_files`` and
    ``local_path`` and delegates teardown to ``cleanup`` in the ``__call__``
    finally, including when the request fails mid-flight.
    """

    cached_file = tmp_path / "cached.bufr"
    cached_file.write_bytes(b"fixture")
    frame = pa.table(
        {
            "time": pa.array([np.datetime64("2024-01-01", "ns")]),
            "observation_t": pa.array([273.15], type=pa.float32()),
        }
    )
    source = NNJAObsConv(cache=True, verbose=False)

    fetched: list[str] = []
    cleanup_calls = {"n": 0}

    async def fake_fetch_files(uris):
        fetched.extend(uris)

    def fake_cleanup():
        cleanup_calls["n"] += 1

    monkeypatch.setattr(source, "fetch_files", fake_fetch_files)
    monkeypatch.setattr(source, "local_path", lambda uri: str(cached_file))
    monkeypatch.setattr(source, "cleanup", fake_cleanup)
    monkeypatch.setattr(source, "_decode_file", lambda path, task: frame)

    result = source(
        datetime(2024, 1, 1),
        ["t"],
        fields=["time", "observation", "variable"],
    )

    assert fetched == [source._build_uri("prepbufr", datetime(2024, 1, 1))]
    assert result["observation"].tolist() == [pytest.approx(273.15)]
    assert result["variable"].tolist() == ["t"]
    assert result.attrs == {"source": source.SOURCE_ID}
    assert cleanup_calls["n"] == 1

    # cleanup must still run if the request fails mid-flight
    async def boom_fetch(uris):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(source, "fetch_files", boom_fetch)
    with pytest.raises(RuntimeError):
        source(
            datetime(2024, 1, 1),
            ["t"],
            fields=["time", "observation", "variable"],
        )
    assert cleanup_calls["n"] == 2


@pytest.mark.parametrize("failing", ["decode", "download"])
def test_nnja_cleanup_waits_for_download_and_decode(tmp_path, monkeypatch, failing):
    """When one side fails, cleanup runs only after the other has stopped."""
    cached_file = tmp_path / "cached.bufr"
    cached_file.write_bytes(b"fixture")
    source = NNJAObsConv(cache=True, verbose=False)
    running = {"download": False, "decode": False}
    running_at_cleanup = []

    async def fetch_files(uris):
        running["download"] = True
        try:
            if failing == "download":
                raise RuntimeError("download failed")
            await asyncio.sleep(5)
        finally:
            running["download"] = False

    def decode(path, task):
        running["decode"] = True
        try:
            if failing == "decode":
                raise RuntimeError("decode failed")
            time.sleep(0.5)
        finally:
            running["decode"] = False
        return pa.table({"time": pa.array([np.datetime64("2024-01-01", "ns")])})

    monkeypatch.setattr(source, "fetch_files", fetch_files)
    monkeypatch.setattr(source, "local_path", lambda uri: str(cached_file))
    monkeypatch.setattr(source, "_decode_file", decode)
    monkeypatch.setattr(
        source, "cleanup", lambda: running_at_cleanup.append(dict(running))
    )
    with pytest.raises(Exception, match="failed"):
        source(datetime(2024, 1, 1), ["t"])
    assert running_at_cleanup == [{"download": False, "decode": False}]


@pytest.mark.parametrize(
    "cls, variable", [(NNJAObsConv, "t"), (nnja.NNJAObsSatwnd, "u")]
)
def test_nnja_obs_conv_decode_failure_raises(tmp_path, monkeypatch, cls, variable):
    """A file that fails to decode fails the request instead of returning no rows."""
    cached_file = tmp_path / "cached.bufr"
    cached_file.write_bytes(b"fixture")
    source = cls(cache=True, verbose=False)

    async def fake_fetch_files(uris):
        pass

    def broken_decode(path, task):
        raise ModuleNotFoundError("No module named 'bitarray'")

    monkeypatch.setattr(source, "fetch_files", fake_fetch_files)
    monkeypatch.setattr(source, "local_path", lambda uri: str(cached_file))
    monkeypatch.setattr(source, "_decode_file", broken_decode)
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as err:
        source(datetime(2024, 1, 1), [variable])
    assert err.value.context["reason"] == "task_failure"
    assert isinstance(err.value.__cause__, ModuleNotFoundError)


def test_nnja_obs_conv_available():
    """Test NNJAObsConv.available() classmethod with both datetime types."""
    assert NNJAObsConv.available(datetime(2024, 1, 1, 0)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 6)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 12)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 18)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 1)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 7, 30)) is True

    # Before MIN_DATE
    assert NNJAObsConv.available(datetime(1970, 1, 1, 0)) is False

    assert NNJAObsConv.available(np.datetime64("2024-01-01T00:00:00")) is True
    assert NNJAObsConv.available(np.datetime64("2024-01-01T06:00:00")) is True
    assert NNJAObsConv.available(np.datetime64("2024-01-01T01:00:00")) is True

    assert NNJAObsConv.available(np.datetime64("1970-01-01T00:00:00")) is False


def test_nnja_obs_conv_build_uris():
    """Test URI building methods for prepbufr and gpsro."""
    ds = NNJAObsConv(cache=False, verbose=False)

    # Test prepbufr URI
    cycle = datetime(2024, 1, 15, 6)
    prepbufr_uri = ds._build_uri("prepbufr", cycle)
    assert "2024" in prepbufr_uri
    assert "01" in prepbufr_uri
    assert "20240115" in prepbufr_uri
    assert "t06z" in prepbufr_uri
    assert "prepbufr" in prepbufr_uri

    # Test gpsro URI
    gpsro_uri = ds._build_uri("gpsro", cycle)
    assert "2024" in gpsro_uri
    assert "01" in gpsro_uri
    assert "20240115" in gpsro_uri
    assert "t06z" in gpsro_uri
    assert "gpsro" in gpsro_uri


def test_nnja_obs_conv_create_tasks():
    """Test _create_tasks method for prepbufr variables."""
    # Use zero tolerance to get exactly one task per cycle
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    # Test prepbufr-only variable
    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert len(tasks) == 1
    assert tasks[0].route == "prepbufr"
    assert "prepbufr" in tasks[0].uri
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)

    # Test multiple variables (same route)
    tasks_multi = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t", "u", "v"])
    assert len(tasks_multi) == 1  # Same file, combined extraction keys
    assert "t" in tasks_multi[0].var_plan
    assert "u" in tasks_multi[0].var_plan
    assert "v" in tasks_multi[0].var_plan

    # Test multiple times
    tasks_times = ds._create_tasks(
        [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)], ["t"]
    )
    assert len(tasks_times) == 2  # Two different cycles

    # A file holds reports within 3h of its cycle, so 01Z is in the 00Z file
    tasks_offset = ds._create_tasks([datetime(2024, 1, 1, 1)], ["t"])
    assert len(tasks_offset) == 1
    assert tasks_offset[0].datetime_file == datetime(2024, 1, 1, 0)

    ds_window = NNJAObsConv(
        time_tolerance=(timedelta(0), timedelta(hours=4)),
        cache=False,
        verbose=False,
    )
    tasks_window = ds_window._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert [task.datetime_file for task in tasks_window] == [
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 6),
    ]


def test_nnja_obs_conv_create_tasks_gpsro_route():
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["gps"])

    assert len(tasks) == 1
    assert tasks[0].route == "gpsro"
    assert "gpsro" in tasks[0].uri
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)
    assert tasks[0].var_plan["gps"][0] == "15037"


def test_nnja_obs_conv_create_tasks_mixed_prepbufr_and_gpsro():
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["gps", "t"])

    assert len(tasks) == 2
    conv_task = next(t for t in tasks if t.route == "prepbufr")
    gpsro_task = next(t for t in tasks if t.route == "gpsro")
    assert set(conv_task.var_plan) == {"t"}
    assert set(gpsro_task.var_plan) == {"gps"}
    assert gpsro_task.var_plan["gps"][0] == "15037"


@pytest.mark.asyncio
async def test_nnja_obs_sat_fetch_and_task_failures_are_structured(
    tmp_path, monkeypatch
):
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=True,
        verbose=False,
        decode_workers=1,
    )
    requested_uri = source._build_satellite_uri(datetime(2024, 1, 1), "atms")

    async def failed_fetch(_uris):
        raise OSError("fetch failed")

    monkeypatch.setattr(source, "fetch_files", failed_fetch)
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as fetch_error:
        await source.fetch(datetime(2024, 1, 1), "atms")
    assert fetch_error.value.context == {
        "reason": "fetch_failure",
        "requested_uri_count": 1,
        "requested_uris": (requested_uri,),
        "cause_type": "OSError",
        "cause_message": "fetch failed",
    }
    # A missing cycle file is an archive gap: warned and skipped, not fatal.
    source._handle_missing_file(requested_uri)

    local_path = tmp_path / "atms.bufr"
    local_path.write_bytes(b"fixture")

    async def successful_fetch(_uris):
        return None

    monkeypatch.setattr(source, "fetch_files", successful_fetch)
    monkeypatch.setattr(source, "local_path", lambda _uri: str(local_path))

    def _failing_decode(_path, _task):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(source, "_decode_file", _failing_decode)
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as task_error:
        await source.fetch(datetime(2024, 1, 1), "atms")
    assert task_error.value.context == {
        "reason": "task_failure",
        "uri": requested_uri,
        "task_index": 1,
        "task_count": 1,
        "cause_type": "RuntimeError",
        "cause_message": "decode failed",
    }


def test_nnja_obs_sat_tasks_group_fields_and_use_verified_archive_routes():
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    # AMSU-B ends in 2014
    cycle = datetime(2013, 1, 1)
    tasks = source._create_tasks(
        [cycle], ["atms", "atms_antenna_temperature", "mhs", "amsua", "amsub"]
    )

    assert len(tasks) == 4
    atms = next(task for task in tasks if task.sensor == "atms")
    assert atms.var_plan == {
        "atms": "TMBR",
        "atms_antenna_temperature": "TMANT",
    }
    assert atms.datetime_min == cycle
    assert atms.datetime_max == cycle
    assert atms.uri.endswith("gdas.20130101.t00z.atms.tm00.bufr_d")
    amsub = next(task for task in tasks if task.sensor == "amsub")
    assert amsub.var_plan == {"amsub": "TMBR"}
    assert amsub.uri.endswith("gdas.20130101.t00z.1bamub.tm00.bufr_d")


def test_nnja_obs_sat_cycle_windows_follow_nnja_cycle_selection():
    source = NNJAObsSat(cache=False, verbose=False, decode_workers=1)
    requested = datetime(2024, 1, 1)
    tasks = source._create_tasks([requested, requested], ["atms"])
    assert [task.datetime_file for task in tasks] == [requested]

    source = NNJAObsSat(
        time_tolerance=(timedelta(hours=-21), timedelta(hours=3)),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    tasks = source._create_tasks([requested], ["atms"])
    assert [task.datetime_file for task in tasks] == [
        datetime(2023, 12, 31, 6),
        datetime(2023, 12, 31, 12),
        datetime(2023, 12, 31, 18),
        requested,
    ]


def test_nnja_obs_sat_fields_time_platform_and_adapter_validation():
    source = NNJAObsSat(cache=False, verbose=False, decode_workers=1)
    assert NNJAObsSat.available(datetime(2024, 1, 1, 6))
    assert NNJAObsSat.available(np.datetime64("2024-01-01T12:00"))
    assert NNJAObsSat.available(datetime(2024, 1, 1, 1))
    assert not NNJAObsSat.available(datetime(1997, 1, 1))
    assert NNJAObsSat.resolve_fields(["time", "observation"]).names == [
        "time",
        "observation",
    ]

    with pytest.raises(ValueError, match="Invalid satellite"):
        NNJAObsSat(satellites=["unknown"])
    with pytest.raises(ValueError, match="Invalid satellite"):
        NNJAObsSat(satellites=["n14"])
    assert NNJAObsSat.resolve_fields(["scan_angle"]).names == ["scan_angle"]
    with pytest.raises(KeyError):
        NNJAObsSat.resolve_fields(["unknown"])
    with pytest.raises(TypeError):
        NNJAObsSat.resolve_fields(pa.schema([pa.field("lat", pa.float64())]))
    # ATMS starts in 2012, AMSU-A in 1998: plan the one that exists
    tasks = source._create_tasks([datetime(2000, 1, 1)], ["atms", "amsua"])
    assert {task.sensor for task in tasks} == {"amsua"}
    assert source._create_tasks([datetime(2000, 1, 1)], ["atms", "cris"]) == []


def test_nnja_obs_sat_ir_tasks_use_verified_archive_routes():
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    cycle = datetime(2019, 1, 1)
    tasks = source._create_tasks([cycle], ["airs", "iasi", "cris"])

    assert len(tasks) == 3
    by_sensor = {task.sensor: task for task in tasks}
    assert by_sensor["airs"].uri.endswith(
        "airs/nasa/aqua/2019/01/bufr/airs_disc_final.20190101.t00z.bufr"
    )
    assert by_sensor["iasi"].uri.endswith(
        "iasi/mtiasi/2019/01/bufr/gdas.20190101.t00z.mtiasi.tm00.bufr_d"
    )
    assert by_sensor["cris"].uri.endswith(
        "cris/crisf4/2019/01/bufr/gdas.20190101.t00z.crisf4.tm00.bufr_d"
    )
    assert by_sensor["airs"].var_plan == {"airs": "TMBR"}
    assert by_sensor["iasi"].var_plan == {"iasi": "SCRA"}
    assert by_sensor["cris"].var_plan == {"cris": "SRAD"}


def test_nnja_obs_sat_cycles_outside_the_archive_period_are_skipped(monkeypatch):
    source = NNJAObsSat(
        time_tolerance=timedelta(0), layout="wide", cache=False, verbose=False
    )
    # Periods are exact to the cycle: crisf4 starts 2018-01-16T18, airs/nasa/aqua
    # ends 2023-01-18T06
    for sensor, outside, inside in [
        ("cris", datetime(2018, 1, 16, 12), datetime(2018, 1, 16, 18)),
        ("airs", datetime(2023, 1, 18, 12), datetime(2023, 1, 18, 6)),
    ]:
        assert source._create_tasks([outside], [sensor]) == []
        assert len(source._create_tasks([inside], [sensor])) == 1
        assert not NNJAObsSat.available(outside, sensor)
        assert NNJAObsSat.available(inside, sensor)
    assert NNJAObsSat.available(datetime(2023, 1, 18, 12))
    with pytest.raises(ValueError, match="Valid options"):
        NNJAObsSat.available(datetime(2023, 1, 18, 12), "ssmis")

    # A request with no cycle in the period reads nothing and returns an empty frame
    async def no_fetch(uris):
        assert not uris

    monkeypatch.setattr(source, "fetch_files", no_fetch)
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        frame = source(datetime(2025, 1, 3), ["airs"])
    finally:
        logger.remove(sink)
    assert frame.empty and channel_axis(frame, "airs").size == 281
    assert sum("outside the archive period" in w for w in warnings) == 1


def test_nnja_obs_sat_missing_files_give_an_empty_frame(monkeypatch):
    async def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(nnja, "obstore_fetch_to_cache", missing)
    source = NNJAObsSat(
        time_tolerance=(timedelta(hours=-4), timedelta(0)),
        layout="wide",
        cache=False,
        verbose=False,
    )
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        frame = source(datetime(2025, 1, 3), ["cris"])
    finally:
        logger.remove(sink)
    assert frame.empty
    assert sum("not found, skipping" in w for w in warnings) == 2


def test_nnja_obs_sat_lexicon_ir_routes():
    from earth2studio.lexicon import NNJAObsSatLexicon

    assert NNJAObsSatLexicon.VOCAB["airs"] == "airs::TMBR"
    assert NNJAObsSatLexicon.VOCAB["iasi"] == "iasi::SCRA"
    assert NNJAObsSatLexicon.VOCAB["cris"] == "cris::SRAD"


def test_nnja_obs_sat_sensor_indices_validation():
    # Empty selection would silently return nothing after a full download;
    # raise at construction instead, matching the JPSS/METOP sources
    with pytest.raises(ValueError, match="at least one"):
        NNJAObsSat(sensor_indices={"cris": []}, cache=False, verbose=False)
    # Channels the archive does not carry: off the instrument grid, or on it but
    # outside the NCEP subset
    for sensor, channel in [("cris", 0), ("cris", 2212), ("cris", 1), ("iasi", 1)]:
        with pytest.raises(ValueError, match="archive carries"):
            NNJAObsSat(sensor_indices={sensor: [channel]}, cache=False, verbose=False)
    # Non-integer and bool entries
    with pytest.raises(ValueError, match="integer"):
        NNJAObsSat(sensor_indices={"cris": [1.5]}, cache=False, verbose=False)
    with pytest.raises(ValueError, match="integer"):
        NNJAObsSat(sensor_indices={"cris": [True]}, cache=False, verbose=False)
    # Duplicates
    with pytest.raises(ValueError, match="unique"):
        NNJAObsSat(sensor_indices={"cris": [19, 19]}, cache=False, verbose=False)
    source = NNJAObsSat(
        sensor_indices={"cris": [19, 2182], "iasi": [16, 8078], "airs": [1, 2377]},
        cache=False,
        verbose=False,
    )
    assert source._sensor_indices["cris"] == frozenset({19, 2182})


def test_nnja_obs_sat_ir_tasks_skipped_when_satellites_exclude_sensor():
    # A platform filter naming no satellite of an IR sensor skips planning
    # instead of fetching and decoding the aggregate for zero rows
    source = NNJAObsSat(
        satellites=["n15"],
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    assert source._create_tasks([datetime(2019, 1, 1)], ["iasi"]) == []
    # Microwave planning is unaffected by the IR skip
    assert source._create_tasks([datetime(2019, 1, 1)], ["amsua"])


def test_nnja_obs_conv_async_timeout_bounds_the_download_not_the_decode(
    tmp_path, monkeypatch
):
    cached_file = tmp_path / "cached.bufr"
    cached_file.write_bytes(b"fixture")
    frame = pa.table(
        {
            "time": pa.array([np.datetime64("2024-01-01", "ns")]),
            "observation_t": pa.array([273.15], type=pa.float32()),
        }
    )
    source = NNJAObsConv(cache=True, verbose=False, async_timeout=1)

    async def quick_fetch(uris):
        return None

    def slow_decode(path, task):
        time.sleep(1.5)
        return frame

    monkeypatch.setattr(source, "fetch_files", quick_fetch)
    monkeypatch.setattr(source, "local_path", lambda uri: str(cached_file))
    monkeypatch.setattr(source, "_decode_file", slow_decode)
    fields = ["time", "observation", "variable"]
    assert len(source(datetime(2024, 1, 1), ["t"], fields=fields)) == 1

    async def slow_fetch(uris):
        await asyncio.sleep(1.5)

    monkeypatch.setattr(source, "fetch_files", slow_fetch)
    with pytest.raises(TimeoutError):
        source(datetime(2024, 1, 1), ["t"], fields=fields)


# Decoders on synthetic earth2bufr batches


T0 = datetime(2024, 1, 1, 0)


# Table D mnemonics of satwnd root descriptors in the fake files' DX tables
_SATWND_SEQUENCES = {"3-63-130": "NC005030", "3-63-185": "NC005024"}


def _satwnd_message(root: str) -> SimpleNamespace:
    return SimpleNamespace(source_message_index=0, descriptors=(root,), table_epoch=0)


class _FakeBufr:
    """Stands in for an earth2bufr file: yields the given batches, applying list filters."""

    def __init__(self, batches, messages=(), sequences=None, truncated=()):
        self.batches = batches
        self.sequences = _SATWND_SEQUENCES if sequences is None else sequences
        self.info = SimpleNamespace(
            messages=list(messages),
            data_messages=list(messages),
            table_epochs=[SimpleNamespace(embedded_table_sha256=None)],
        )
        self.options = {}
        self.truncated = truncated

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def source_structure(self, source_message_index):
        return {
            "active_sequences": [
                {"descriptor": d, "mnemonic": n} for d, n in self.sequences.items()
            ]
        }

    def iter_batches(self, **options):
        self.options = options
        batches = []
        for batch in self.batches:
            for flt in options.get("list_filters", []):
                batch = _filter_list(batch, flt.field, flt.key, set(flt.values))
            batches.append(batch)
        skipped = self.truncated if options.get("skip_truncated_messages") else ()
        return _FakeBatches(batches, skipped)


class _FakeBatches:
    """earth2bufr's BatchIterator: batches, then the truncated messages it skipped."""

    def __init__(self, batches, decode_failures):
        self._batches = iter(batches)
        self.decode_failures = tuple(decode_failures)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._batches)


def _filter_list(batch, field, key, values):
    lists = batch.column(field)
    keep = pc.is_in(
        lists.values.field(key), value_set=pa.array(sorted(values))
    ).to_numpy(zero_copy_only=False)
    parent = np.repeat(np.arange(len(lists)), np.diff(lists.offsets.to_numpy()))
    offsets = np.zeros(len(lists) + 1, dtype=np.int32)
    np.cumsum(np.bincount(parent[keep], minlength=len(lists)), out=offsets[1:])
    filtered = pa.ListArray.from_arrays(
        pa.array(offsets), lists.values.filter(pa.array(keep))
    )
    index = batch.schema.get_field_index(field)
    return batch.set_column(index, field, filtered)


@pytest.fixture
def fake_bufr(monkeypatch):
    files = {}

    def open_fake(path):
        return files[path]

    monkeypatch.setattr(utils_ncep, "open_bufr", open_fake)
    return files


def _batch(columns: dict) -> pa.RecordBatch:
    names, arrays = [], []
    for name, values in columns.items():
        names.append(name.split("#")[0])
        arrays.append(values if isinstance(values, pa.Array) else pa.array(values))
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _time_parts(stamps: list[datetime]) -> dict:
    return {
        "YEAR": [float(t.year) for t in stamps],
        "MNTH": [float(t.month) for t in stamps],
        "DAYS": [float(t.day) for t in stamps],
        "HOUR": [float(t.hour) for t in stamps],
        "MINU": [float(t.minute) for t in stamps],
        "SECO": [float(t.second) for t in stamps],
    }


def _struct_list(rows: list[list[tuple]], names: list[str]) -> pa.Array:
    """list<struct> from per-row lists of tuples; duplicate child names allowed."""
    flat = [entry for row in rows for entry in row]
    children = [
        pa.array([entry[i] for entry in flat], type=pa.float64())
        for i in range(len(names))
    ]
    values = pa.StructArray.from_arrays(children, names=names)
    offsets = np.concatenate([[0], np.cumsum([len(row) for row in rows])])
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)


def _gpsro_batch(stamps: list[datetime]) -> pa.RecordBatch:
    n = len(stamps)
    levels = _struct_list(
        [[(10.5, 20.5)]] * n, ["CLATH", "CLONH"]
    )  # one level per occultation
    frequencies = _struct_list(
        [[(0.0, 6.4e6, 0.01, 0.001), (1.5e9, 6.4e6, 0.02, 0.002)]] * n,
        ["MEFR", "IMPP", "BNDA", "BNDA"],
    )
    level_values = pa.StructArray.from_arrays(
        [levels.values.field(0), levels.values.field(1), frequencies],
        names=["CLATH", "CLONH", "DRF8BIT"],
    )
    return _batch(
        {
            "SAID": [44.0] * n,
            "PTID": [7.0] * n,
            "QFRO": [0.0] * n,
            "ELRC": [6.37e6] * n,
            "GEODU": [12.0] * n,
            "CLATH": [10.0] * n,
            "CLONH": [-20.0] * n,
            **_time_parts(stamps),
            "DRF16BIT": pa.ListArray.from_arrays(levels.offsets, level_values),
            "DRF16BIT_2": _struct_list(
                [[(1000.0, 300.0, 1.0), (2000.0, 250.0, 1.0)]] * n,
                ["HEIT", "ARFR", "ARFR"],
            ),
        }
    )


def _prepbufr_file(category: int = 102, families: dict | None = None) -> _FakeBufr:
    """ADPUPA reports under `category`, plus empty messages of `families`.

    Files before 2007 number the families 230-249 (Table A sequence 3-60-YYY),
    later files 102-121 (3-48-YYY).
    """
    scale = {b"bufr.scale": b"3"}
    events = {
        "P": _struct_list([[(850.0, 1.0)], [(1000.0, 2.0)]], ["POB", "PQM"]),
        "T": _struct_list([[(10.0, 1.0), (11.0, 9.0)], [(20.0, 2.0)]], ["TOB", "TQM"]),
        "W": _struct_list([[(1.0, 2.0, 3.0)], []], ["UOB", "VOB", "WQM"]),
    }
    level = pa.StructArray.from_arrays(
        [pa.array([1.0, 0.0]), pa.array([1500.0, 10.0]), *events.values()],
        names=["CAT", "ZOB", "P", "T", "W"],
    )
    levels = pa.ListArray.from_arrays(pa.array([0, 1, 2], type=pa.int32()), level)
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["STN1    ", "STN2    "]),
            pa.array([250.0, -10.0]),
            pa.array([40.0, 41.0]),
            pa.array([0.0, 3.0]),
            pa.array([100.0, 5.0]),
            pa.array([120.0, 181.0]),
            levels,
            pa.array([float(category)] * 2),
            pa.array([0, 0], type=pa.uint64()),
        ],
        schema=pa.schema(
            [
                ("SID", pa.string()),
                ("XOB", pa.float64()),
                ("YOB", pa.float64()),
                pa.field("DHR", pa.float64(), metadata=scale),
                ("ELV", pa.float64()),
                ("TYP", pa.float64()),
                ("DRF8BIT", levels.type),
                ("_data_category", pa.float64()),
                ("_source_message_index", pa.uint64()),
            ]
        ),
    )
    messages, sequences = [], {}
    for index, (number, name) in enumerate(
        {category: "ADPUPA", **(families or {})}.items()
    ):
        table_a = f"3-{60 if number >= 230 else 48}-{number:03d}"
        sequences[table_a] = name
        messages.append(
            SimpleNamespace(
                source_message_index=index,
                data_category=number,
                descriptors=("0-63-000", table_a),
                table_epoch=0,
                edition=4,
                year=2024,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
            )
        )
    return _FakeBufr([batch], messages, sequences)


def _satwnd_batch(stamps: list[datetime], fields: dict | None = None) -> pa.RecordBatch:
    n = len(stamps)
    return _batch(
        {
            **_time_parts(stamps),
            "SAID": [270.0] * n,
            "SWCM": [1.0] * n,
            "PRLC": [50000.0] * n,
            "PRLC#2": [3000.0] * n,
            "SAZA": [30.0] * n,
            "WDIR": [90.0] * n,
            "WSPD": [10.0] * n,
            "SWQM": [0.0] * n,
            "CLATH": [10.0] * n,
            "CLONH": [-30.0] * n,
            "GNAP": [1.0] * n,
            "PCCF": [90.0] * n,
            "GNAP#2": [2.0] * n,
            "PCCF#2": [80.0] * n,
            "GNAP#3": [1.0] * n,
            "PCCF#3": [70.0] * n,
            **(fields or {}),
            "_source_message_index": pa.array([0] * n, type=pa.uint64()),
        }
    )


def _sat_batch(
    sensor: str, stamps: list[datetime], channels, entries, fields=None
) -> pa.RecordBatch:
    n = len(stamps)
    names = {
        "atms": ["CHNM", "SCCF", "ATMSCHQ", "TMBR"],
        "iasi": ["CHNM", "SCRA"],
        "airs": ["CHNM", "LOGRCW", "ACQF", "TMBR"],
        "cris": ["CHNM", "SRAD"],
    }[sensor]
    if sensor == "atms":
        # SCCF as the files encode it: the frequency table's raw codes x 10**5 Hz
        def entries(c, entries=entries):
            return (utils_ncep._ATMS_SCCF_CODE[int(c) - 1] / 1e-5, *entries(c))

    rows = [[(c, *entries(c)) for c in channels]] * n
    extra = {
        "atms": {"ATMSGQ": [0.0] * n, "ATMSSQ": [0.0] * n},
        "iasi": {
            "QGFQ": [0.0] * n,
            "STCH": [1.0] * n,
            "ENCH": [8461.0] * n,
            "CHSF": [7.0] * n,
        },
        "airs": {},
        "cris": {"FORN": [1.0] * n, "HOLS": [0.0] * n, "NSQF": [0.0] * n},
    }[sensor]
    return _batch(
        {
            "SAID": [{"atms": 224.0, "iasi": 4.0, "airs": 784.0, "cris": 225.0}[sensor]]
            * n,
            **_time_parts(stamps),
            "SLNM": [1.0] * n,
            "FOVN": [5.0] * n,
            "CLATH": [10.0] * n,
            "CLONH": [-30.0] * n,
            "SAZA": [20.0] * n,
            "SOZA": [40.0] * n,
            "BEARAZ": [100.0] * n,
            "SOLAZI": [200.0] * n,
            **extra,
            **(fields or {}),
            "DRF16BIT": _struct_list(rows, names),
        }
    )


def test_observation_cycle_times_read_overlapped_cycles():
    t = datetime(2024, 1, 3, 0)
    hours = timedelta(hours=1)
    # File T spans [T - 3h, T + 3h]; a request reads the files whose span it
    # overlaps in the interior
    cases = [
        (-3 * hours, 3 * hours, [t]),
        (-4 * hours, 3 * hours, [t - 6 * hours, t]),
        (-9 * hours, 3 * hours, [t - 6 * hours, t]),
        (-45 * hours, 3 * hours, [t + k * 6 * hours for k in range(-7, 1)]),
        (hours, 4 * hours, [t, t + 6 * hours]),
        (3 * hours + timedelta(minutes=1), 9 * hours, [t + 6 * hours]),
        (-21 * hours, 3 * hours, [t + k * 6 * hours for k in range(-3, 1)]),
        (hours, hours, [t]),
        (3 * hours, 3 * hours, [t, t + 6 * hours]),
    ]
    for k in range(1, 7):
        cases.append((-45 * hours, (k - 3) * hours, cases[3][2]))
    for lower, upper, expected in cases:
        assert observation_cycle_times(t, lower, upper) == expected
    assert observation_cycle_times(
        t + hours, timedelta(0), timedelta(0), timedelta(hours=3)
    ) == [t]
    with pytest.raises(ValueError, match="cadence"):
        observation_cycle_times(t, timedelta(0), timedelta(0), timedelta(0))


def test_requests_sharing_a_file_keep_their_own_intervals():
    t = datetime(2024, 1, 3, 0)
    minutes = timedelta(minutes=1)
    windows = cycle_windows(
        [t - 60 * minutes, t + 120 * minutes], -30 * minutes, 30 * minutes
    )
    assert windows == {
        t: (
            (t - 90 * minutes, t - 30 * minutes),
            (t + 90 * minutes, t + 150 * minutes),
        )
    }
    # 00Z ±1h and 06Z ±1h read separate files
    assert cycle_windows([t, t + 360 * minutes], -60 * minutes, 60 * minutes) == {
        t: ((t - 60 * minutes, t + 60 * minutes),),
        t + 360 * minutes: ((t + 300 * minutes, t + 420 * minutes),),
    }


def test_gpsro_long_wide_and_closed_window(fake_bufr):
    end = T0 + timedelta(hours=3)
    stamps = [T0, end, end + timedelta(seconds=1)]
    fake_bufr["gpsro"] = _FakeBufr([_gpsro_batch(stamps)])
    variables = ["gps", "gps_refractivity"]
    wide = decode_gpsro("gpsro", variables, T0 - timedelta(hours=3), end, 1)
    # Bending angle (MEFR 0 only) and two refractivity levels per kept occultation;
    # the occultation at exactly the window end is kept
    assert wide.num_rows == 2 * 3
    assert pc.count(wide["observation_gps"]).as_py() == 2
    assert pc.count(wide["observation_gps_refractivity"]).as_py() == 4
    assert pc.max(wide["time"]).as_py() == end
    long = _to_long(wide, variables, NCEP_CONVENTIONAL_PUBLIC_SCHEMA)
    assert long.column_names == NCEP_CONVENTIONAL_PUBLIC_SCHEMA.names
    assert long["variable"].to_pylist() == ["gps"] * 2 + ["gps_refractivity"] * 4
    np.testing.assert_allclose(long["observation"].to_numpy()[:2], 0.01, rtol=1e-6)
    assert long["station"].to_pylist()[0] == "00440007"


@pytest.mark.parametrize("category", [102, 230])
@pytest.mark.parametrize(
    "original_event, temperature", [(False, 283.15), (True, 284.15)]
)
def test_prepbufr_units_events_and_pressure_subset(
    fake_bufr, category, original_event, temperature
):
    fake_bufr["prep"] = _prepbufr_file(category)
    variables = ["t", "u", "pres"]
    wide = decode_prepbufr(
        "prep",
        variables,
        T0 - timedelta(hours=3),
        T0 + timedelta(hours=3),
        decode_workers=1,
        original_event=original_event,
    )
    # One row per report level giving a variable
    assert wide.num_rows == 2
    assert wide["station"].cast(pa.string()).to_pylist() == ["STN1", "STN2"]
    assert wide["report_time"].to_pylist()[1] == T0 + timedelta(hours=3)
    assert wide["observation_t"].to_pylist()[0] == pytest.approx(temperature)
    # pres: CAT 0, report type 181, POB >= 500 hPa -> Pa; the CAT 1 level is not a
    # pressure observation
    assert wide["observation_pres"].to_pylist() == [None, 100000.0]
    assert wide["observation_u"].to_pylist() == [1.0, None]
    long = _to_long(wide, variables, NCEP_CONVENTIONAL_PUBLIC_SCHEMA)
    assert long["variable"].to_pylist() == ["t", "t", "u", "pres"]
    assert long["quality"].to_pylist()[2] == 3
    assert long["pres"].to_pylist() == [85000.0, 100000.0, 85000.0, 100000.0]


@pytest.mark.parametrize("adpupa, satwnd, adpsfc", [(102, 105, 109), (230, 233, 237)])
def test_prepbufr_excluded_families_are_not_decoded(fake_bufr, adpupa, satwnd, adpsfc):
    fake_bufr["prep"] = _prepbufr_file(
        adpupa, families={satwnd: "SATWND", adpsfc: "ADPSFC"}
    )
    fake_bufr["prep"].batches = []
    decode_prepbufr(
        "prep",
        ["t"],
        T0,
        T0,
        decode_workers=1,
        exclude_message_types=("SATWND", "ADPUPA"),
    )
    assert fake_bufr["prep"].options["data_category_filter"] == [adpsfc]


def test_prepbufr_null_dhr_raises(fake_bufr):
    source = _prepbufr_file()
    batch = source.batches[0]
    index = batch.schema.get_field_index("DHR")
    source.batches = [
        batch.set_column(index, batch.schema.field(index), pa.array([None, 3.0]))
    ]
    fake_bufr["prepbufr"] = source
    with pytest.raises(ValueError, match="1 ADPUPA reports with null DHR"):
        decode_prepbufr("prepbufr", ["t"], T0, T0, decode_workers=1)


def test_prepbufr_without_message_families_raises(fake_bufr):
    fake_bufr["no_dx"] = _prepbufr_file()
    fake_bufr["no_dx"].sequences = {}
    with pytest.raises(ValueError, match="no PrepBUFR message family"):
        decode_prepbufr("no_dx", ["t"], T0, T0, decode_workers=1)
    # A file whose families are all excluded is empty, not an error
    fake_bufr["excluded"] = _prepbufr_file()
    wide = decode_prepbufr(
        "excluded", ["t"], T0, T0, decode_workers=1, exclude_message_types=("ADPUPA",)
    )
    assert wide.num_rows == 0


def test_prepbufr_drift_levels_follow_their_report_time(fake_bufr):
    # A 00Z sonde reported at 02:54 whose first level drifted to 03:06
    prepbufr = _prepbufr_file()
    batch = prepbufr.batches[0]
    levels = batch.column("DRF8BIT")
    hrdr = pa.field("HRDR", pa.float64(), metadata={b"bufr.scale": b"5"})
    values = pa.StructArray.from_arrays(
        [*levels.values.flatten(), pa.array([3.1, None])],
        fields=[*levels.type.value_type, hrdr],
    )
    batch = batch.set_column(
        batch.schema.get_field_index("DRF8BIT"),
        "DRF8BIT",
        pa.ListArray.from_arrays(levels.offsets, values),
    )
    batch = batch.set_column(
        batch.schema.get_field_index("DHR"),
        batch.schema.field("DHR"),
        pa.array([2.9, 2.9]),
    )
    prepbufr.batches = [batch]
    fake_bufr["prep"] = prepbufr
    cycle = T0

    # 03:01-09:00 neither reads the 00Z file nor takes the 03:06 level from it
    later = (cycle + timedelta(hours=3, minutes=1), cycle + timedelta(hours=9))
    assert cycle not in observation_cycle_times(
        later[0], timedelta(0), later[1] - later[0]
    )
    assert decode_prepbufr("prep", ["t"], *later, decode_workers=1).num_rows == 0

    # 00:00-03:00 returns the level with its report, at its own time
    wide = decode_prepbufr(
        "prep", ["t"], cycle, cycle + timedelta(hours=3), decode_workers=1
    )
    assert wide["time"].to_pylist()[0] == datetime(2024, 1, 1, 3, 6)
    assert wide["report_time"].to_pylist()[0] == datetime(2024, 1, 1, 2, 54)


def test_satwnd_wide_long_and_quality_pairs(fake_bufr):
    stamps = [T0, T0 + timedelta(hours=3)]
    messages = [_satwnd_message("3-63-130")]
    fake_bufr["satwnd"] = _FakeBufr([_satwnd_batch(stamps)], messages)
    wide = decode_satwnd("satwnd", ["u", "v"], T0, T0 + timedelta(hours=3), 1)
    assert wide.num_rows == 2
    # Wind from the east (90 deg) at 10 m/s: u = -10, v = 0
    np.testing.assert_allclose(wide["observation_u"].to_numpy(), -10.0, rtol=1e-6)
    np.testing.assert_allclose(wide["observation_v"].to_numpy(), 0.0, atol=1e-5)
    # First top-level PRLC is the wind's pressure; a repeated application keeps its first
    assert wide["pres"].to_pylist() == [50000.0, 50000.0]
    assert wide["quality_indicators"].to_pylist()[0] == [
        {"application": 1, "confidence": 90.0},
        {"application": 2, "confidence": 80.0},
    ]
    assert wide["subset"].cast(pa.string()).to_pylist() == ["NC005030"] * 2
    long = _to_long(wide, ["u", "v"], NCEP_SATWND_PUBLIC_SCHEMA)
    assert long.num_rows == 4
    assert long["variable"].to_pylist() == ["u", "u", "v", "v"]
    assert (
        long["quality_indicators"].to_pylist()[2]
        == wide["quality_indicators"].to_pylist()[0]
    )


def test_satwnd_non_positive_pressure_is_null(fake_bufr):
    messages = [_satwnd_message("3-63-130")]
    batch = _satwnd_batch([T0] * 3, {"PRLC": [0.0, -100.0, 50000.0]})
    fake_bufr["satwnd"] = _FakeBufr([batch], messages)
    wide = decode_satwnd("satwnd", ["u"], T0, T0, 1)
    assert wide["pres"].to_pylist() == [None, None, 50000.0]


def test_missing_hour_drops_the_wind_and_the_occultation(fake_bufr):
    messages = [_satwnd_message("3-63-130")]
    hour = pa.array([None, 0.0])
    fake_bufr["satwnd"] = _FakeBufr([_satwnd_batch([T0, T0], {"HOUR": hour})], messages)
    assert decode_satwnd("satwnd", ["u"], T0, T0, 1).num_rows == 1
    gpsro = _gpsro_batch([T0, T0])
    index = gpsro.schema.get_field_index("HOUR")
    fake_bufr["gpsro"] = _FakeBufr([gpsro.set_column(index, "HOUR", hour)])
    wide = decode_gpsro("gpsro", ["gps"], T0, T0, 1)
    assert wide["station"].to_pylist() == ["00440007"]


def test_gpsro_reference_point_is_the_occultation_clath(fake_bufr):
    # The level CLATH/CLONH of the first occultation do not stand in for its own
    gpsro = _gpsro_batch([T0, T0])
    index = gpsro.schema.get_field_index("CLATH")
    gpsro = gpsro.set_column(index, "CLATH", pa.array([None, 10.0]))
    fake_bufr["gpsro"] = _FakeBufr([gpsro])
    wide = decode_gpsro("gpsro", ["gps_refractivity"], T0, T0, 1)
    assert wide["lat"].to_pylist() == [10.0, 10.0]


def test_gpsro_one_bending_angle_per_level(fake_bufr):
    # Level 1: two corrected entries, the last wins. Level 2: none, its last entry wins
    levels = _struct_list([[(10.5, 20.5), (11.5, 21.5)]], ["CLATH", "CLONH"])
    frequencies = _struct_list(
        [
            [
                (0.0, 6.4e6, 0.01, 0.0),
                (1.5e9, 6.4e6, 0.02, 0.0),
                (0.0, 6.4e6, 0.03, 0.0),
            ],
            [(1.5e9, 6.5e6, 0.04, 0.0), (1.2e9, 6.5e6, 0.05, 0.0)],
        ],
        ["MEFR", "IMPP", "BNDA", "BNDA"],
    )
    level_values = pa.StructArray.from_arrays(
        [levels.values.field(0), levels.values.field(1), frequencies],
        names=["CLATH", "CLONH", "DRF8BIT"],
    )
    gpsro = _gpsro_batch([T0])
    index = gpsro.schema.get_field_index("DRF16BIT")
    gpsro = gpsro.set_column(
        index, "DRF16BIT", pa.ListArray.from_arrays(levels.offsets, level_values)
    )
    fake_bufr["gpsro"] = _FakeBufr([gpsro])
    wide = decode_gpsro("gpsro", ["gps"], T0, T0, 1)
    np.testing.assert_allclose(wide["observation_gps"].to_numpy(), [0.03, 0.05])


def test_truncated_messages_are_skipped_and_counted(fake_bufr):
    truncated = [SimpleNamespace(source_message_index=4, reason="past Section 4")]
    messages = [_satwnd_message("3-63-130")]
    fake_bufr["satwnd"] = _FakeBufr(
        [_satwnd_batch([T0])], messages, truncated=truncated
    )
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    fake_bufr["atms"] = _FakeBufr(
        [_sat_batch("atms", [T0], [1.0], lambda c: (0.0, 250.0))],
        truncated=truncated,
    )
    try:
        wide = decode_satwnd("satwnd", ["u"], T0, T0, 1)
        sat = decode_satellite(
            "atms", "atms", {"atms": "TMBR"}, T0, T0, channels=[1], decode_workers=1
        )
    finally:
        logger.remove(sink)
    assert wide.num_rows == 1 and sat.num_rows == 1
    assert sum("skipped 1 truncated BUFR messages" in w for w in warnings) == 2


def test_satwnd_subset_is_read_from_the_dx_table(fake_bufr):
    fake_bufr["satwnd"] = _FakeBufr(
        [_satwnd_batch([T0])], [_satwnd_message("3-63-130")], {"3-63-130": "NC005099"}
    )
    wide = decode_satwnd("satwnd", ["u"], T0, T0, 1)
    assert wide["subset"].cast(pa.string()).to_pylist() == ["NC005099"]


def test_satwnd_scalars_read_the_first_occurrence_populated_in_the_file(fake_bufr):
    stamps = [T0] * 3
    empty_first = {"HAMD": pa.nulls(3, pa.float64()), "HAMD#2": [7.0, None, 9.0]}
    sparse_first = {"HAMD": [None, 2.0, None], "HAMD#2": [7.0, 8.0, 9.0]}
    messages = [_satwnd_message("3-63-185")]
    first = _satwnd_batch(stamps, empty_first)
    second = _satwnd_batch(stamps, sparse_first)
    # The second batch populates the first HAMD, so the file reads it for every wind
    fake_bufr["two"] = _FakeBufr([first, second], messages)
    wide = decode_satwnd("two", ["u"], T0, T0, 1)
    assert wide["height_method"].to_pylist() == [None, None, None, None, 2, None]
    # Without that batch the second HAMD is the first populated one
    fake_bufr["one"] = _FakeBufr([first], messages)
    wide = decode_satwnd("one", ["u"], T0, T0, 1)
    assert wide["height_method"].to_pylist() == [7, None, 9]
    # The same winds in one batch decode the same
    whole = pa.Table.from_batches([first, second]).combine_chunks().to_batches()[0]
    fake_bufr["whole"] = _FakeBufr([whole], messages)
    assert decode_satwnd("whole", ["u"], T0, T0, 1).equals(
        decode_satwnd("two", ["u"], T0, T0, 1)
    )


def test_microwave_wide_is_fixed_size_and_long_matches(fake_bufr):
    channels = [1.0, 2.0, 3.0]
    batch = _sat_batch("atms", [T0, T0], channels, lambda c: (0.0, 200.0 + c))
    fake_bufr["atms"] = _FakeBufr([batch])
    wide = decode_satellite(
        "atms", "atms", {"atms": "TMBR"}, T0, T0, channels=[1, 2, 3], decode_workers=1
    )
    assert pa.types.is_fixed_size_list(wide.schema.field("observation_atms").type)
    np.testing.assert_allclose(
        wide["observation_atms"].combine_chunks().values.to_numpy().reshape(2, 3),
        [[201.0, 202.0, 203.0]] * 2,
    )
    long = _to_long(wide, ["atms"], NCEP_MICROWAVE_OUTPUT_SCHEMA)
    assert long.num_rows == 6
    assert long["sensor_index"].to_pylist() == [1, 2, 3, 1, 2, 3]
    assert long["scan_angle"].to_pylist()[0] == pytest.approx(-52.725 + 4 * 1.11)
    assert long["satellite"].cast(pa.string()).to_pylist() == ["npp"] * 6
    # ATMS wavenumbers come from the per-channel SCCF table: 23.8, 31.4, 50.3 GHz
    assert "wavenumber_atms" not in wide.column_names
    np.testing.assert_allclose(
        long["wavenumber"].to_numpy(),
        np.tile([23.8e9, 31.4e9, 50.3e9], 2) / 2.99792458e10,
    )


def test_satellite_wide_is_one_matrix_over_the_channel_axis(fake_bufr):
    def entries(c):
        return (0.0, 200.0 + c)

    full = _sat_batch("atms", [T0], [1.0, 2.0, 3.0], entries)
    # The second batch's footprint lacks channel 2
    partial = _sat_batch("atms", [T0], [3.0, 1.0], entries)
    fake_bufr["atms"] = _FakeBufr([full, partial])
    wide = decode_satellite(
        "atms", "atms", {"atms": "TMBR"}, T0, T0, channels=[1, 2, 3], decode_workers=1
    )
    matrix = pa.list_(pa.float32(), 3)
    assert [chunk.type for chunk in wide["observation_atms"].chunks] == [matrix] * 2
    assert wide["observation_atms"].to_pylist() == [
        [201.0, 202.0, 203.0],
        [201.0, None, 203.0],
    ]


def test_satellite_footprint_with_no_channel_on_the_axis_is_dropped(fake_bufr):
    def entries(c):
        return (0.0, 200.0 + c)

    full = _sat_batch("atms", [T0], [1.0, 2.0, 3.0], entries)
    # The second footprint's one channel entry has no channel number
    bare = _sat_batch("atms", [T0], [1.0], entries)
    sccf = utils_ncep._ATMS_SCCF_CODE[0] / 1e-5
    bare = bare.set_column(
        bare.schema.get_field_index("DRF16BIT"),
        "DRF16BIT",
        _struct_list([[(None, sccf, 0.0, 201.0)]], ["CHNM", "SCCF", "ATMSCHQ", "TMBR"]),
    )
    batch = pa.Table.from_batches([full, bare]).combine_chunks().to_batches()[0]
    fake_bufr["atms"] = _FakeBufr([batch])
    # No channel filter: the axis is every ATMS channel
    wide = decode_satellite("atms", "atms", {"atms": "TMBR"}, T0, T0, decode_workers=1)
    assert wide.num_rows == 1
    assert wide["observation_atms"].to_pylist()[0][:4] == [201.0, 202.0, 203.0, None]


def test_nnja_obs_sat_wide_frame_carries_channel_axis(fake_bufr, monkeypatch, tmp_path):
    source = NNJAObsSat(
        time_tolerance=(timedelta(hours=-3), timedelta(hours=3)),
        sensor_indices={"iasi": [38, 16]},
        layout="wide",
        cache=False,
        verbose=False,
    )
    batch = _sat_batch("iasi", [T0], [16.0, 38.0, 49.0], lambda c: (1000.0 + c,))
    _source_with_files(source, fake_bufr, monkeypatch, tmp_path, [batch])
    frame = fetch_dataframe(
        source,
        time=np.array([T0], dtype="datetime64[ns]"),
        variable=np.array(["iasi"]),
    )
    assert channel_axis(frame, "iasi").tolist() == [16, 38]
    assert frame["observation_iasi"].dtype == pd.ArrowDtype(pa.list_(pa.float32(), 2))
    assert frame["quality"].dtype == pd.ArrowDtype(pa.list_(pa.uint32(), 2))
    # Only IASI's own footprint columns
    assert "footprint_quality" in frame.columns
    assert not {"detector", "granule_quality", "scan_angle"} & set(frame.columns)


def test_nnja_obs_sat_wide_takes_one_sensor_per_call():
    source = NNJAObsSat(layout="wide", cache=False, verbose=False)
    with pytest.raises(ValueError, match="one sensor per call"):
        source(T0, ["atms", "cris"])


def test_concurrent_calls_share_the_decode_slots(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"x")
    tasks = [SimpleNamespace(uri=str(path), datetime_file=T0, intervals=[(T0, T0)])] * 4
    active, peak = [0], [0]
    lock = threading.Lock()

    def decode(path, task):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return pa.table({"x": [1]})

    async def run():
        await asyncio.gather(
            *(
                asyncio.to_thread(compile_tables, tasks, "shared", str, decode)
                for _ in range(3)
            )
        )

    asyncio.run(run())
    assert peak[0] == utils_ncep._FILES_IN_FLIGHT


def test_ir_channel_filter_and_brightness_temperature(fake_bufr):
    channels = [16.0, 38.0, 49.0]
    batch = _sat_batch("iasi", [T0], channels, lambda c: (1000.0 + c,))
    fake_bufr["iasi"] = _FakeBufr([batch])
    wide = decode_satellite(
        "iasi",
        "iasi",
        {"iasi": "SCRA"},
        T0,
        T0,
        channels=[16, 49],
        decode_workers=1,
    )
    (flt,) = fake_bufr["iasi"].options["list_filters"]
    assert (flt.field, flt.key, flt.values) == ("DRF16BIT", "CHNM", (16.0, 49.0))
    long = _to_long(wide, ["iasi"], NCEP_MICROWAVE_OUTPUT_SCHEMA)
    assert long["sensor_index"].to_pylist() == [16, 49]
    radiance = np.array([1016.0, 1049.0]) * 10.0**-7 * 1e5
    wavenumber = wavenumber_cm_inverse("iasi", [16, 49])
    np.testing.assert_array_equal(
        long["observation"].to_numpy(),
        radiance_to_bt(radiance, wavenumber).astype(np.float32),
    )
    np.testing.assert_array_equal(long["wavenumber"].to_numpy(), wavenumber)


def test_cris_band_quality_is_read_by_slot(fake_bufr):
    # NFQF/NCQF slots are LW, MW, SW; the MW slot is missing
    flags = {
        "NFQF": [1.0],
        "NFQF#2": pa.array([None], type=pa.float64()),
        "NFQF#3": [4.0],
        "NCQF": [2.0],
        "NCQF#2": pa.array([None], type=pa.float64()),
        "NCQF#3": [3.0],
    }
    # One channel per band: 1 (LW), 714 (MW), 1579 (SW)
    batch = _sat_batch("cris", [T0], [1.0, 714.0, 1579.0], lambda c: (1e-4,), flags)
    fake_bufr["cris"] = _FakeBufr([batch])
    wide = decode_satellite(
        "cris",
        "cris",
        {"cris": "SRAD"},
        T0,
        T0,
        channels=[1, 714, 1579],
        decode_workers=1,
    )
    assert wide["quality"].to_pylist() == [[1 | 2 << 19, None, 4 | 3 << 19]]


def test_iasi_scale_groups_by_bounds(fake_bufr):
    def iasi(groups, scra):
        # Groups (STCH, ENCH, CHSF); a null CHSF is None
        fields = {}
        for k, (first, last, chsf) in enumerate(groups):
            suffix = f"#{k + 1}" if k else ""
            fields[f"STCH{suffix}"] = [first]
            fields[f"ENCH{suffix}"] = [last]
            fields[f"CHSF{suffix}"] = pa.array([chsf], type=pa.float64())
        batch = _sat_batch("iasi", [T0], [16.0, 38.0], lambda c: (scra[c],), fields)
        fake_bufr["iasi"] = _FakeBufr([batch])
        wide = decode_satellite(
            "iasi",
            "iasi",
            {"iasi": "SCRA"},
            T0,
            T0,
            channels=[16, 38],
            decode_workers=1,
        )
        return wide["observation_iasi"].to_pylist()[0]

    # Channel 38 lies in no group, then in two
    with pytest.raises(ValueError, match="each channel once"):
        iasi([(1, 20, 7.0)], {16.0: 1000.0, 38.0: 1000.0})
    with pytest.raises(ValueError, match="each channel once"):
        iasi([(1, 40, 7.0), (30, 50, 6.0)], {16.0: 1000.0, 38.0: 1000.0})
    # A null CHSF is accepted for a zero SCRA (no radiance) and raises on a nonzero one
    groups = [(1, 20, 7.0), (21, 50, None)]
    first, second = iasi(groups, {16.0: 1000.0, 38.0: 0.0})
    assert first is not None and second is None
    with pytest.raises(ValueError, match="CHSF is null"):
        iasi(groups, {16.0: 1000.0, 38.0: 1000.0})


def test_satellite_footprint_time_and_location_are_checked(fake_bufr):
    def decode(**fields):
        batch = _sat_batch("atms", [T0], [1.0], lambda c: (0.0, 250.0), fields=fields)
        fake_bufr["atms"] = _FakeBufr([batch])
        day = timedelta(days=1)
        return decode_satellite(
            "atms", "atms", {"atms": "TMBR"}, T0 - day, T0 + day, channels=[1]
        )

    # 24:00:00 is midnight of the next day
    previous = {
        name: [value[0]]
        for name, value in _time_parts([T0 - timedelta(days=1)]).items()
    }
    wide = decode(**{**previous, "HOUR": [24.0]})
    assert wide["time"].to_pylist() == [pd.Timestamp(T0)]
    missing = pa.array([None], type=pa.float64())
    for fields in ({"HOUR": [24.0], "MINU": [1.0]}, {"MINU": missing}):
        with pytest.raises(ValueError, match="time"):
            decode(**fields)
    for fields in ({"SAID": missing}, {"CLATH": [91.0]}, {"CLONH": missing}):
        with pytest.raises(ValueError, match="SAID|latitude"):
            decode(**fields)


def test_satellite_footprints_the_request_skips_are_not_checked(fake_bufr):
    # Outside the window: no SAID and latitude 91; on an unrequested platform: latitude 91
    late = T0 + timedelta(days=3)
    batch = _sat_batch(
        "atms",
        [T0, late, T0],
        [1.0],
        lambda c: (0.0, 250.0),
        fields={"SAID": [224.0, None, 225.0], "CLATH": [10.0, 91.0, 91.0]},
    )
    fake_bufr["atms"] = _FakeBufr([batch])
    wide = decode_satellite(
        "atms", "atms", {"atms": "TMBR"}, T0, T0, satellites={"npp"}, channels=[1]
    )
    assert wide.num_rows == 1
    assert wide["satellite"].cast(pa.string()).to_pylist() == ["npp"]


def test_airs_wavenumber_read_from_file(fake_bufr):
    # LOGRCW (log10 of the centre wavenumber in m⁻¹) of AIRS channels 1 and 2377
    log10_m = {1.0: 4.81265938, 2377.0: 5.42555671}
    batch = _sat_batch(
        "airs", [T0, T0], list(log10_m), lambda c: (log10_m[c], 0.0, 250.0)
    )
    fake_bufr["airs"] = _FakeBufr([batch])
    wide = decode_satellite(
        "airs", "airs", {"airs": "TMBR"}, T0, T0, channels=[1, 2377], decode_workers=1
    )
    assert "wavenumber_airs" not in wide.column_names
    long = _to_long(wide, ["airs"], NCEP_MICROWAVE_OUTPUT_SCHEMA)
    assert long["sensor_index"].to_pylist() == [1, 2377, 1, 2377]
    np.testing.assert_allclose(
        long["wavenumber"].to_numpy(), [649.62, 2664.138] * 2, atol=0.001
    )
    # No footprint with LOGRCW for every channel
    batch = _sat_batch("airs", [T0], [1.0], lambda c: (np.nan, 0.0, 250.0))
    fake_bufr["airs"] = _FakeBufr([batch])
    with pytest.raises(ValueError, match="LOGRCW"):
        decode_satellite("airs", "airs", {"airs": "TMBR"}, T0, T0, channels=[1])


def test_atms_sccf_must_match_the_frequency_table(fake_bufr, monkeypatch):
    batch = _sat_batch("atms", [T0], [1.0], lambda c: (0.0, 250.0))
    fake_bufr["atms"] = _FakeBufr([batch])
    monkeypatch.setattr(utils_ncep, "_ATMS_SCCF_CODE", utils_ncep._ATMS_SCCF_CODE + 1)
    with pytest.raises(ValueError, match="SCCF"):
        decode_satellite("atms", "atms", {"atms": "TMBR"}, T0, T0, channels=[1])


def test_satellite_dictionary_holds_the_kept_platforms(fake_bufr):
    # The second footprint (n20) is outside the window
    batch = _sat_batch(
        "atms",
        [T0, T0 + timedelta(hours=1)],
        [1.0],
        lambda c: (0.0, 250.0),
        fields={"SAID": [224.0, 225.0]},
    )
    fake_bufr["atms"] = _FakeBufr([batch])
    wide = decode_satellite("atms", "atms", {"atms": "TMBR"}, T0, T0, channels=[1])
    assert wide["satellite"].combine_chunks().dictionary.to_pylist() == ["npp"]


def test_satellite_filter_and_window(fake_bufr):
    batch = _sat_batch(
        "atms", [T0, T0 + timedelta(hours=1)], [1.0], lambda c: (0.0, 250.0)
    )
    fake_bufr["atms"] = _FakeBufr([batch])
    kept = decode_satellite("atms", "atms", {"atms": "TMBR"}, T0, T0, decode_workers=1)
    assert kept.num_rows == 1
    none = decode_satellite(
        "atms", "atms", {"atms": "TMBR"}, T0, T0, satellites=["n20"], decode_workers=1
    )
    assert none.num_rows == 0


def _source_with_files(
    source, fake_bufr, monkeypatch, tmp_path, batches, messages=(), sequences=None
):
    async def fetch_files(uris):
        return None

    def local_path(uri):
        path = tmp_path / uri.rsplit("/", 1)[-1]
        path.write_bytes(b"fake")
        fake_bufr[str(path)] = _FakeBufr(batches, messages, sequences)
        return str(path)

    monkeypatch.setattr(source, "fetch_files", fetch_files)
    monkeypatch.setattr(source, "local_path", local_path)
    return source


@pytest.mark.parametrize("layout", ["long", "wide"])
def test_nnja_sources_long_and_wide(fake_bufr, monkeypatch, tmp_path, layout):
    tolerance = (timedelta(hours=-3), timedelta(hours=3))
    end = T0 + timedelta(hours=3)
    cases = [
        (
            NNJAObsConv(
                time_tolerance=tolerance, cache=False, verbose=False, layout=layout
            ),
            ["gps"],
            [_gpsro_batch([T0, end])],
            (),
        ),
        (
            NNJAObsSatwnd(
                time_tolerance=tolerance, cache=False, verbose=False, layout=layout
            ),
            ["u", "v"],
            [_satwnd_batch([T0, end])],
            [_satwnd_message("3-63-130")],
        ),
        (
            NNJAObsSat(
                time_tolerance=tolerance, cache=False, verbose=False, layout=layout
            ),
            ["atms"],
            [_sat_batch("atms", [T0, end], [1.0, 2.0], lambda c: (0.0, 250.0))],
            (),
        ),
    ]
    for source, variables, batches, messages in cases:
        _source_with_files(source, fake_bufr, monkeypatch, tmp_path, batches, messages)
        frame = source(T0, variables)
        assert all(isinstance(dtype, pd.ArrowDtype) for dtype in frame.dtypes)
        assert frame.attrs["source"] == source.SOURCE_ID
        # The T0 file covers [T0 - 3h, T0 + 3h]; the observation at exactly T0 + 3h
        # is included
        assert frame["time"].max() == pd.Timestamp(end)
        schema = source._output_fields(None, variables)
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        if layout == "long":
            assert list(frame.columns) == source.SCHEMA.names
            assert set(frame["variable"]) == set(variables)
        else:
            assert "variable" not in frame.columns
            assert {f"observation_{v}" for v in variables} <= set(frame.columns)
            # The source converts file by file
            cycles = pc.unique(table["cycle_time"]).to_pylist()
            long = pa.concat_tables(
                [
                    _to_long(
                        table.filter(pc.equal(table["cycle_time"], cycle)),
                        variables,
                        source.SCHEMA,
                    )
                    for cycle in cycles
                ]
            )
            reference = source.__class__(
                time_tolerance=tolerance, cache=False, verbose=False, layout="long"
            )
            _source_with_files(
                reference, fake_bufr, monkeypatch, tmp_path, batches, messages
            )
            expected = pa.Table.from_pandas(
                reference(T0, variables), preserve_index=False
            )
            assert long.num_rows == expected.num_rows
            for name in ("time", "observation", "variable"):
                assert long[name].cast(expected[name].type).equals(expected[name])


@pytest.mark.parametrize("layout", ["long", "wide"])
def test_nnja_sources_keep_rows_inside_each_requested_interval(
    fake_bufr, monkeypatch, tmp_path, layout
):
    # 23Z ±30 min and 02Z ±30 min both read the 00Z file, which returns no rows
    # between 23:30 and 01:30
    minutes = timedelta(minutes=1)
    tolerance = (-30 * minutes, 30 * minutes)
    stamps = [T0 - 75 * minutes, T0, T0 + 120 * minutes]
    prepbufr = _prepbufr_file()
    batch = prepbufr.batches[0]
    prepbufr_batch = batch.set_column(
        batch.schema.get_field_index("DHR"),
        batch.schema.field("DHR"),
        pa.array([-1.25, 0.0]),
    )
    cases = [
        (NNJAObsConv, ["gps"], [_gpsro_batch(stamps)], (), None, [0, 2]),
        (
            NNJAObsConv,
            ["t"],
            [prepbufr_batch],
            prepbufr.info.data_messages,
            prepbufr.sequences,
            [0],
        ),
        (
            NNJAObsSatwnd,
            ["u", "v"],
            [_satwnd_batch(stamps)],
            [_satwnd_message("3-63-130")],
            None,
            [0, 2],
        ),
        (
            NNJAObsSat,
            ["atms"],
            [_sat_batch("atms", stamps, [1.0, 2.0], lambda c: (0.0, 250.0))],
            (),
            None,
            [0, 2],
        ),
    ]
    for cls, variables, batches, messages, sequences, kept in cases:
        source = cls(
            time_tolerance=tolerance, cache=False, verbose=False, layout=layout
        )
        _source_with_files(
            source, fake_bufr, monkeypatch, tmp_path, batches, messages, sequences
        )
        frame = source([T0 - 60 * minutes, T0 + 120 * minutes], variables)
        assert sorted(set(frame["time"])) == [pd.Timestamp(stamps[i]) for i in kept]
        assert set(frame["cycle_time"]) == {pd.Timestamp(T0)}


def test_satwnd_decodes_quality_indicators_only_when_requested(
    fake_bufr, monkeypatch, tmp_path
):
    source = NNJAObsSatwnd(layout="wide", cache=False, verbose=False)
    _source_with_files(
        source,
        fake_bufr,
        monkeypatch,
        tmp_path,
        [_satwnd_batch([T0])],
        [_satwnd_message("3-63-130")],
    )
    for fields, decoded in (
        (["time", "observation_u"], False),
        (["time", "observation_u", "quality_indicators"], True),
    ):
        source(T0, ["u"], fields)
        (fake,) = fake_bufr.values()
        assert ("PCCF" in fake.options["projection"]) is decoded
        fake_bufr.clear()


def test_nnja_layout_validation_and_wide_fields():
    with pytest.raises(ValueError, match="layout"):
        NNJAObsConv(layout="tall", cache=False, verbose=False)
    wide = NNJAObsSatwnd(layout="wide", cache=False, verbose=False)
    assert wide._output_fields(["observation_u", "pres"], ["u"]).names == [
        "observation_u",
        "pres",
    ]
    with pytest.raises(KeyError):
        wide._output_fields(["observation"], ["u"])
    # A wide conventional call returns one stream: PrepBUFR or GPS-RO
    conv = NNJAObsConv(layout="wide", cache=False, verbose=False)
    assert "radius_curvature" in conv._output_fields(None, ["gps"]).names
    assert "report_time" not in conv._output_fields(None, ["gps"]).names
    assert "radius_curvature" not in conv._output_fields(None, ["t", "u"]).names
    # One observation and quality column per requested variable
    names = conv._output_fields(None, ["t", "u"]).names
    assert {"observation_t", "observation_u", "quality_t", "quality_u"} <= set(names)
    assert not {"observation_q", "quality_q", "observation_v"} & set(names)
    with pytest.raises(ValueError, match="one stream per call"):
        conv(T0, ["t", "gps"])


def test_default_decode_threads_respects_env(monkeypatch):
    calls = []
    monkeypatch.setattr(utils_bufr.earth2bufr, "set_num_threads", calls.append)
    expected = min(8, len(os.sched_getaffinity(0)))

    monkeypatch.delenv("EARTH2BUFR_NUM_THREADS", raising=False)
    utils_bufr.default_decode_threads.cache_clear()
    utils_bufr.default_decode_threads()
    utils_bufr.default_decode_threads()
    assert calls == [expected]

    monkeypatch.setenv("EARTH2BUFR_NUM_THREADS", "3")
    utils_bufr.default_decode_threads.cache_clear()
    utils_bufr.default_decode_threads()
    assert calls == [expected]
    utils_bufr.default_decode_threads.cache_clear()
