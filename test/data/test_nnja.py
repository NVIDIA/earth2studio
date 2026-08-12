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

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import earth2studio.data.nnja as nnja
from earth2studio.data import NNJAObsConv, NNJAObsSat, utils_ncep

ncep_microwave = utils_ncep

pytest.importorskip("pybufrkit", reason="pybufrkit not installed")


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
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "level_cat": [0, 0],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "pressure_quality": [1, 1],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
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
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "level_cat": [0, 0],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "pressure_quality": [1, 1],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
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

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00"]),
            "observation": [273.15],
            "variable": ["t"],
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
    assert result.equals(frame)
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


# These are unit tests for BUFR decoding


def test_nnja_safe_int():
    """Test _safe_int helper function with various input types."""
    from earth2studio.data.utils_bufr import safe_int as _safe_int

    # int/float inputs
    assert _safe_int(42) == 42
    assert _safe_int(3.14) == 3
    assert _safe_int(-7.9) == -7

    # bytes input
    assert _safe_int(b"123") == 123
    assert _safe_int(b"  456  ") == 456
    assert _safe_int(b"") == 0
    assert _safe_int(b"abc") == 0  # non-numeric bytes

    # None input
    assert _safe_int(None) == 0

    # string input
    assert _safe_int("789") == 789
    assert _safe_int("  012  ") == 12
    assert _safe_int("") == 0
    assert _safe_int("not_a_number") == 0


def test_nnja_extract_dx_tables_empty():
    """Test _extract_dx_tables with empty/minimal inputs."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Empty flat list
    _extract_dx_tables([], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # Only n_a (no entries)
    _extract_dx_tables([0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # n_a=0, n_b=0 (skip to table D)
    _extract_dx_tables([0, 0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # n_a=0, n_b=0, n_d=0 (all empty)
    _extract_dx_tables([0, 0, 0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}


def test_nnja_extract_dx_tables_truncated():
    """Test _extract_dx_tables with truncated/partial data."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # n_a entries truncated (should return early)
    _extract_dx_tables([1], table_b, table_d)  # n_a=1 but no entries
    assert table_b == {}
    assert table_d == {}

    # Table B entry truncated (less than 11 fields)
    # n_a=0, n_b=1, then only partial entry
    table_b.clear()
    table_d.clear()
    _extract_dx_tables([0, 1, 0, 1, 2], table_b, table_d)  # truncated B entry
    assert table_b == {}

    # Table D entry truncated
    # n_a=0, n_b=0, n_d=1, then only partial entry
    table_b.clear()
    table_d.clear()
    _extract_dx_tables([0, 0, 1, 0, 1], table_b, table_d)  # truncated D entry
    assert table_d == {}


def test_nnja_extract_dx_tables_valid_entries():
    """Test _extract_dx_tables with valid Table B and D entries."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Create a valid Table B entry
    # Format: n_a, [a_entries...], n_b, [b_entries (11 fields each)...], n_d, [d_entries...]
    # B entry: f, x, y, mnemonic, desc(skip), unit, sign_scale, scale, sign_ref, ref, width
    flat = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        1,
        2,
        b"TOB",
        b"",
        b"K",
        b"+",
        b"2",
        b"+",
        b"0",
        b"16",  # 11 fields
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat, table_b, table_d)
    # desc_id = 0*100000 + 1*1000 + 2 = 1002
    assert 1002 in table_b
    assert table_b[1002][0] == "TOB"  # mnemonic

    # Test with zero descriptor (should be skipped)
    table_b.clear()
    table_d.clear()
    flat_zero = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        0,
        0,
        b"ZERO",
        b"",
        b"K",
        b"+",
        b"0",
        b"+",
        b"0",
        b"8",  # desc_id = 0
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat_zero, table_b, table_d)
    assert 0 not in table_b  # Zero descriptor skipped

    # Test with negative scale/reference
    table_b.clear()
    table_d.clear()
    flat_neg = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        12,
        1,
        b"LAT",
        b"",
        b"DEG",
        b"-",
        b"5",
        b"-",
        b"9000000",
        b"25",
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat_neg, table_b, table_d)
    # desc_id = 0*100000 + 12*1000 + 1 = 12001
    assert 12001 in table_b
    assert table_b[12001][2] == -5  # scale is negative
    assert table_b[12001][3] == -9000000  # reference is negative


def test_nnja_extract_dx_tables_table_d():
    """Test _extract_dx_tables Table D sequence entries."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Table D entry: f, x, y, seq_mnemonic, n_members, [member_mnemonics...]
    flat = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        3,
        1,
        1,
        b"HEADR",  # seq entry header (f=3, x=1, y=1)
        2,  # n_members = 2
        b"SID",
        b"XOB",  # member mnemonics
    ]
    _extract_dx_tables(flat, table_b, table_d)
    # seq_id = 3*100000 + 1*1000 + 1 = 301001
    assert 301001 in table_d
    assert table_d[301001][0] == "HEADR"
    assert table_d[301001][1] == ["SID", "XOB"]

    # Test zero seq_id (should be skipped)
    table_b.clear()
    table_d.clear()
    flat_zero = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        0,
        0,
        0,
        b"ZERO",  # seq_id = 0
        1,
        b"X",
    ]
    _extract_dx_tables(flat_zero, table_b, table_d)
    assert 0 not in table_d  # Zero seq_id skipped

    # Test truncated member list
    table_b.clear()
    table_d.clear()
    flat_trunc = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        3,
        1,
        2,
        b"SEQ",
        3,  # n_members = 3 but only 1 provided
        b"ONLY_ONE",
    ]
    _extract_dx_tables(flat_trunc, table_b, table_d)
    # Should still create entry with partial members
    assert 301002 in table_d
    assert table_d[301002][1] == ["ONLY_ONE"]


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

    # Non-cycle request times select the next cycle file containing past obs
    tasks_offset = ds._create_tasks([datetime(2024, 1, 1, 1)], ["t"])
    assert len(tasks_offset) == 1
    assert tasks_offset[0].datetime_file == datetime(2024, 1, 1, 6)

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
    assert tasks[0].var_plan["gps"][0] == str(utils_ncep.GPSRO_BNDA)


def test_nnja_obs_conv_create_tasks_mixed_prepbufr_and_gpsro():
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["gps", "t"])

    assert len(tasks) == 2
    conv_task = next(t for t in tasks if t.route == "prepbufr")
    gpsro_task = next(t for t in tasks if t.route == "gpsro")
    assert set(conv_task.var_plan) == {"t"}
    assert set(gpsro_task.var_plan) == {"gps"}
    assert gpsro_task.var_plan["gps"][0] == str(utils_ncep.GPSRO_BNDA)


def test_nnja_obs_conv_pres_modifier_keeps_station_pressure_only():
    from earth2studio.lexicon import NNJAObsConvLexicon

    _, modifier = NNJAObsConvLexicon["pres"]

    df = pd.DataFrame(
        {
            "observation": [1000.0, 850.0, 850.0, 850.0, 850.0, 400.0, 850.0],
            "type": [180, 181, 187, 120, 120, 180, 250],
            "class": [
                "SFCSHP",
                "ADPSFC",
                "ADPSFC",
                "ADPUPA",
                "ADPUPA",
                "SFCSHP",
                "SATWND",
            ],
            "level_cat": [0, 0, 0, 0, 1, 0, 0],
            "quality": [2, 2, 2, 2, 2, 2, 2],
        }
    )

    result = modifier(df)

    assert result["observation"].tolist() == [100000.0, 85000.0, 85000.0, 85000.0]
    assert result["type"].tolist() == [180, 181, 187, 120]


@dataclass(frozen=True)
class _MicrowaveDescriptor:
    id: int


def _decode_microwave_pairs(
    pairs: list[tuple[int, Any]],
    variable_fields: tuple[tuple[str, int], ...],
    sensor: str = "atms",
    satellites: frozenset[str] | None = None,
    datetime_min: datetime = datetime(2023, 12, 31, 21),
    datetime_max: datetime = datetime(2024, 1, 1, 3),
) -> list[dict[str, Any]]:
    return ncep_microwave._decode_microwave_subset(
        [_MicrowaveDescriptor(descriptor) for descriptor, _ in pairs],
        [value for _, value in pairs],
        sensor,
        variable_fields,
        datetime_min,
        datetime_max,
        satellites,
    )


def _atms_microwave_pairs() -> list[tuple[int, Any]]:
    return [
        (ncep_microwave._YEAR, 2023),
        (ncep_microwave._MONTH, 12),
        (ncep_microwave._DAY, 31),
        (ncep_microwave._HOUR, 21),
        (ncep_microwave._MINUTE, 2),
        (ncep_microwave._SECOND, 45.352),
        (ncep_microwave._LAT_COARSE, 12.3),
        (ncep_microwave._LON_COARSE, -45.7),
        (ncep_microwave._LAT_HIGH, 12.34567),
        (ncep_microwave._LON_HIGH, -45.67891),
        (ncep_microwave._SAID, 225),
        (ncep_microwave._SCAN_LINE, 8),
        (ncep_microwave._FOV_NUMBER, 7),
        (ncep_microwave._SURFACE_ELEVATION, 123.5),
        (ncep_microwave._SATELLITE_ZENITH, 55.25),
        (ncep_microwave._BEARING_OR_AZIMUTH, 269.17),
        (ncep_microwave._SOLAR_ZENITH, 99.47),
        (ncep_microwave._SOLAR_AZIMUTH, 153.88),
        (ncep_microwave._CHANNEL_NUMBER, 2),
        (ncep_microwave._CHANNEL_FREQUENCY, 31.4e9),
        (ncep_microwave._ANTENNA_TEMPERATURE, 201.25),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 202.5),
        (ncep_microwave._CHANNEL_QUALITY, 2),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._CHANNEL_FREQUENCY, 23.8e9),
        (ncep_microwave._ANTENNA_TEMPERATURE, 190.25),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 191.5),
        (ncep_microwave._CHANNEL_QUALITY, 1),
    ]


def _microwave_message(pairs: list[tuple[int, Any]]):
    return _microwave_message_from_subsets([pairs])


def _microwave_message_from_subsets(subsets: list[list[tuple[int, Any]]]):
    template = SimpleNamespace(
        decoded_descriptors_all_subsets=[
            [_MicrowaveDescriptor(descriptor) for descriptor, _ in pairs]
            for pairs in subsets
        ],
        decoded_values_all_subsets=[[value for _, value in pairs] for pairs in subsets],
    )
    return SimpleNamespace(template_data=SimpleNamespace(value=template))


def test_nnja_obs_sat_decode_preserves_encoded_atms_quantities_and_identity():
    rows = _decode_microwave_pairs(
        _atms_microwave_pairs(),
        (
            ("atms", ncep_microwave._BRIGHTNESS_TEMPERATURE),
            ("atms_antenna_temperature", ncep_microwave._ANTENNA_TEMPERATURE),
        ),
    )

    assert [row["sensor_index"] for row in rows] == [2, 2, 1, 1]
    assert [row["variable"] for row in rows] == [
        "atms",
        "atms_antenna_temperature",
        "atms",
        "atms_antenna_temperature",
    ]
    assert [row["observation"] for row in rows] == pytest.approx(
        [202.5, 201.25, 191.5, 190.25]
    )
    assert all(
        row["time"] == np.datetime64("2023-12-31T21:02:45.352000000") for row in rows
    )
    assert rows[0]["lat"] == pytest.approx(12.34567)
    assert rows[0]["lon"] == pytest.approx(314.32109)
    assert rows[0]["elev"] == pytest.approx(123.5)
    assert rows[0]["satellite"] == "n20"
    assert rows[0]["scan_angle"] == pytest.approx(-46.065)
    assert rows[0]["scan_position"] == 7
    assert rows[0]["scan_line"] == 8
    assert rows[0]["satellite_aza"] == pytest.approx(269.17)
    assert rows[0]["quality"] == 2
    assert rows[0]["wavenumber"] == pytest.approx(31.4e9 / ncep_microwave._C_CM_S)

    frame = ncep_microwave._rows_to_dataframe(rows)
    assert NNJAObsSat.SCHEMA.names == [
        "time",
        "class",
        "lat",
        "lon",
        "elev",
        "scan_angle",
        "scan_position",
        "scan_line",
        "sensor_index",
        "wavenumber",
        "solza",
        "solaza",
        "satellite_za",
        "satellite_aza",
        "quality",
        "satellite",
        "observation",
        "variable",
    ]
    assert list(frame.columns) == NNJAObsSat.SCHEMA.names
    assert str(frame["time"].dtype) == "datetime64[ns]"
    assert str(frame["sensor_index"].dtype) == "uint16[pyarrow]"
    assert frame["lat"].dtype == np.float32
    assert frame["observation"].dtype == np.float32


@pytest.mark.parametrize(
    "sensor,scan_position,expected",
    [
        ("atms", 1, -52.725),
        ("atms", 96, 52.725),
        ("amsua", 1, -145.0 / 3.0),
        ("amsua", 30, 145.0 / 3.0),
        ("amsub", 1, -48.95),
        ("amsub", 90, 48.95),
        ("mhs", 1, -445.0 / 9.0),
        ("mhs", 90, 445.0 / 9.0),
    ],
)
def test_ncep_microwave_nominal_scan_geometry(
    sensor: str, scan_position: int, expected: float
):
    assert ncep_microwave._nominal_microwave_scan_angle(
        sensor, scan_position
    ) == pytest.approx(expected)


def test_nnja_obs_sat_decode_preserves_amsub_channels_and_quantity():
    pairs = [
        (descriptor, 207 if descriptor == ncep_microwave._SAID else value)
        for descriptor, value in _atms_microwave_pairs()
        if descriptor
        not in {
            ncep_microwave._ANTENNA_TEMPERATURE,
            ncep_microwave._CHANNEL_FREQUENCY,
            ncep_microwave._CHANNEL_QUALITY,
        }
    ]
    rows = _decode_microwave_pairs(
        pairs,
        (("amsub", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
        sensor="amsub",
    )

    assert [row["sensor_index"] for row in rows] == [2, 1]
    assert [row["observation"] for row in rows] == pytest.approx([202.5, 191.5])
    assert {row["satellite"] for row in rows} == {"n16"}
    assert {row["variable"] for row in rows} == {"amsub"}
    assert [row["scan_angle"] for row in rows] == pytest.approx([-42.35, -42.35])


@pytest.mark.parametrize("decode_workers,message_count", [(1, 1), (2, 33)])
def test_ncep_microwave_adapter_serial_and_parallel_batch_paths(
    tmp_path, monkeypatch, decode_workers, message_count
):
    source_path = tmp_path / "atms.bufr"
    source_path.write_bytes(b"synthetic")
    decoder = SimpleNamespace(
        process=lambda _message: _microwave_message(_atms_microwave_pairs())
    )
    monkeypatch.setattr(ncep_microwave, "_worker_decoder", decoder)
    monkeypatch.setattr(
        ncep_microwave,
        "_parse_prepbufr_messages",
        lambda _data, silence_noise=True: (
            {1: ("B",)},
            {1: ("D",)},
            [(b"message", 0)] * message_count,
        ),
    )
    initialized = []
    monkeypatch.setattr(
        ncep_microwave,
        "_init_decode_worker",
        lambda table_b, table_d: initialized.append((table_b, table_d)),
    )

    class _Executor:
        def __init__(self, *, max_workers, initializer, initargs):
            assert max_workers == 2
            assert initializer is ncep_microwave._init_decode_worker
            assert initargs == ({1: ("B",)}, {1: ("D",)})

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def map(self, function, arguments):
            return map(function, arguments)

    monkeypatch.setattr(ncep_microwave, "ProcessPoolExecutor", _Executor)
    frame = ncep_microwave.decode_microwave(
        str(source_path),
        "atms",
        {"atms": "TMBR"},
        datetime(2023, 12, 31, 21),
        datetime(2024, 1, 1, 3),
        decode_workers=decode_workers,
    )

    assert len(frame) == 2 * message_count
    assert frame["sensor_index"].tolist() == [2, 1] * message_count
    assert initialized == ([({1: ("B",)}, {1: ("D",)})] if decode_workers == 1 else [])


def test_ncep_microwave_message_preserves_mixed_satellite_order(monkeypatch):
    def pairs_for(satellite_id):
        return [
            (
                descriptor,
                satellite_id if descriptor == ncep_microwave._SAID else value,
            )
            for descriptor, value in _atms_microwave_pairs()
        ]

    message = _microwave_message_from_subsets(
        [pairs_for(223), pairs_for(223), pairs_for(3), pairs_for(223)]
    )
    monkeypatch.setattr(
        ncep_microwave,
        "_worker_decoder",
        SimpleNamespace(process=lambda _message: message),
    )
    rows, failures = ncep_microwave._decode_message_batch(
        (
            "mhs",
            [(17, b"message")],
            (("mhs", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
            datetime(2023, 12, 31, 21),
            datetime(2024, 1, 1, 3),
            None,
        )
    )

    assert failures == 0
    assert [row["satellite"] for row in rows] == [
        "n19",
        "n19",
        "n19",
        "n19",
        "metop-b",
        "metop-b",
        "n19",
        "n19",
    ]


def test_ncep_microwave_adapter_rejects_missing_tables_and_failed_messages(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "atms.bufr"
    source_path.write_bytes(b"synthetic")

    monkeypatch.setattr(
        ncep_microwave,
        "_parse_prepbufr_messages",
        lambda _data, silence_noise=True: ({}, {}, []),
    )
    with pytest.raises(ValueError, match="tables are missing"):
        ncep_microwave.decode_microwave(
            str(source_path),
            "atms",
            {"atms": "TMBR"},
            datetime(2024, 1, 1),
            datetime(2024, 1, 1),
            decode_workers=1,
        )

    monkeypatch.setattr(
        ncep_microwave,
        "_parse_prepbufr_messages",
        lambda _data, silence_noise=True: (
            {1: ("B",)},
            {1: ("D",)},
            [(b"good-message", 0), (b"bad-message", 0)],
        ),
    )
    monkeypatch.setattr(ncep_microwave, "_init_decode_worker", lambda *_args: None)
    monkeypatch.setattr(
        ncep_microwave,
        "_worker_decoder",
        SimpleNamespace(
            process=lambda message: (
                _microwave_message(_atms_microwave_pairs())
                if message == b"good-message"
                else (_ for _ in ()).throw(ValueError("bad message"))
            )
        ),
    )
    with pytest.raises(ncep_microwave._NCEPMicrowaveDecodeError) as error:
        ncep_microwave.decode_microwave(
            str(source_path),
            "atms",
            {"atms": "TMBR"},
            datetime(2023, 12, 31, 21),
            datetime(2024, 1, 1, 3),
            decode_workers=1,
        )
    assert error.value.context == {
        "path": str(source_path),
        "decoded_messages": 1,
        "failed_messages": 1,
        "total_messages": 2,
    }


def test_nnja_obs_sat_decode_uses_coarse_location_and_preserves_missingness():
    pairs = [
        (ncep_microwave._YEAR, 2024),
        (ncep_microwave._MONTH, 1),
        (ncep_microwave._DAY, 1),
        (ncep_microwave._HOUR, 0),
        (ncep_microwave._MINUTE, 0),
        (ncep_microwave._SECOND, 1),
        (ncep_microwave._LAT_COARSE, -1.2286),
        (ncep_microwave._LON_COARSE, -2.8979),
        (ncep_microwave._SAID, 3),
        (ncep_microwave._FOV_NUMBER, 1),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 272.02),
        (ncep_microwave._CHANNEL_NUMBER, 2),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, None),
    ]
    rows = _decode_microwave_pairs(
        pairs,
        (("mhs", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
        sensor="mhs",
    )

    assert len(rows) == 1
    assert rows[0]["lat"] == pytest.approx(-1.2286)
    assert rows[0]["lon"] == pytest.approx(357.1021)
    assert rows[0]["satellite"] == "metop-b"
    assert rows[0]["observation"] == pytest.approx(272.02)
    assert rows[0]["quality"] is None
    assert np.isnan(rows[0]["wavenumber"])

    assert not _decode_microwave_pairs(
        pairs,
        (("mhs", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
        sensor="mhs",
        satellites=frozenset(["n19"]),
    )


@pytest.mark.asyncio
async def test_nnja_obs_sat_fetch_uses_foundation_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=True,
        verbose=False,
        decode_workers=1,
    )
    local_path = tmp_path / "atms.bufr"
    local_path.write_bytes(b"fixture")
    monkeypatch.setattr(source, "local_path", lambda _uri: str(local_path))

    rows = _decode_microwave_pairs(
        _atms_microwave_pairs(),
        (("atms", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
    )
    decoded = ncep_microwave._rows_to_dataframe(rows)

    async def fetch_files(_uris):
        return None

    monkeypatch.setattr(source, "fetch_files", fetch_files)
    monkeypatch.setattr(source, "_decode_file", lambda _path, _task: decoded)

    result = await source.fetch(
        datetime(2024, 1, 1),
        "atms",
        fields=["time", "observation", "variable"],
    )

    assert list(result.columns) == ["time", "observation", "variable"]
    assert len(result) == 2
    assert result.attrs == {"source": NNJAObsSat.SOURCE_ID}


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
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as missing_error:
        source._handle_missing_file(requested_uri)
    assert missing_error.value.context == {
        "reason": "remote_file_missing",
        "uri": requested_uri,
    }

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


def test_nnja_obs_sat_decode_rejects_incomplete_or_out_of_window_subsets():
    missing_fov = [
        pair
        for pair in _atms_microwave_pairs()
        if pair[0] != ncep_microwave._FOV_NUMBER
    ]
    assert not _decode_microwave_pairs(
        missing_fov,
        (("atms", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
    )

    out_of_window = [
        (descriptor, 12 if descriptor == ncep_microwave._HOUR else value)
        for descriptor, value in _atms_microwave_pairs()
    ]
    assert not _decode_microwave_pairs(
        out_of_window,
        (("atms", ncep_microwave._BRIGHTNESS_TEMPERATURE),),
    )


def test_nnja_obs_sat_tasks_group_fields_and_use_verified_archive_routes():
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    cycle = datetime(2024, 1, 1)
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
    assert atms.uri.endswith("gdas.20240101.t00z.atms.tm00.bufr_d")
    amsub = next(task for task in tasks if task.sensor == "amsub")
    assert amsub.var_plan == {"amsub": "TMBR"}
    assert amsub.uri.endswith("gdas.20240101.t00z.1bamub.tm00.bufr_d")


def test_nnja_obs_sat_cycle_windows_follow_nnja_cycle_selection():
    source = NNJAObsSat(cache=False, verbose=False, decode_workers=1)
    requested = datetime(2024, 1, 1)
    tasks = source._create_tasks([requested, requested], ["atms"])
    assert [task.datetime_file for task in tasks] == [
        requested,
        requested + timedelta(hours=6),
    ]

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
        datetime(2024, 1, 1, 6),
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
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as unavailable:
        source._create_tasks([datetime(2000, 1, 1)], ["atms", "amsua"])
    assert unavailable.value.context["reason"] == "archive_unavailable"
    with pytest.raises(KeyError):
        ncep_microwave.decode_microwave(
            "not-read.bufr",
            "atms",
            {"atms": "UNKNOWN"},
            datetime(2024, 1, 1),
            datetime(2024, 1, 1),
            decode_workers=1,
        )


# ---------------------------------------------------------------------------
# Hyperspectral IR sounders folded into NNJAObsSat (airs / iasi / cris)
# ---------------------------------------------------------------------------


def _decode_ir_pairs(
    pairs: list[tuple[int, Any]],
    sensor: str,
    channels_filter: frozenset[int] | None = None,
    satellites: frozenset[str] | None = None,
    datetime_min: datetime = datetime(2023, 12, 31, 21),
    datetime_max: datetime = datetime(2024, 1, 1, 3),
) -> list[dict[str, Any]]:
    return ncep_microwave._decode_ir_subset(
        [_MicrowaveDescriptor(descriptor) for descriptor, _ in pairs],
        [value for _, value in pairs],
        sensor,
        channels_filter,
        datetime_min,
        datetime_max,
        satellites,
    )


def _ir_scalar_pairs(said: int) -> list[tuple[int, Any]]:
    return [
        (ncep_microwave._YEAR, 2023),
        (ncep_microwave._MONTH, 12),
        (ncep_microwave._DAY, 31),
        (ncep_microwave._HOUR, 21),
        (ncep_microwave._MINUTE, 2),
        (ncep_microwave._SECOND, 45.352),
        (ncep_microwave._LAT_HIGH, 12.34567),
        (ncep_microwave._LON_HIGH, -45.67891),
        (ncep_microwave._SAID, said),
        (ncep_microwave._SCAN_LINE, 8),
        (ncep_microwave._FOV_NUMBER, 7),
        (ncep_microwave._SATELLITE_ZENITH, 55.25),
        (ncep_microwave._SOLAR_ZENITH, 99.47),
    ]


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


def test_nnja_obs_sat_ir_archive_unavailable_outside_coverage():
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    # Before the archive start (crisf4 begins 2018, mtiasi 2008)
    for sensor, cycle in [
        ("cris", datetime(2017, 1, 1)),
        ("iasi", datetime(2007, 6, 1)),
    ]:
        with pytest.raises(nnja._NNJAObsSatIncompleteError) as err:
            source._create_tasks([cycle], [sensor])
        assert err.value.context["reason"] == "archive_unavailable"
        assert err.value.context["sensor"] == sensor

    # After the archive end (airs/nasa/aqua stops in January 2023)
    with pytest.raises(nnja._NNJAObsSatIncompleteError) as err:
        source._create_tasks([datetime(2024, 1, 1)], ["airs"])
    assert err.value.context["reason"] == "archive_unavailable"
    assert err.value.context["last_year"] == 2023

    # Inside coverage plans normally
    assert source._create_tasks([datetime(2019, 1, 1)], ["airs", "cris"])


def test_nnja_obs_sat_decode_file_routes_ir_and_microwave(monkeypatch):
    source = NNJAObsSat(
        satellites=["aqua", "npp", "metop-b"],
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    empty = ncep_microwave._rows_to_dataframe([])
    calls: dict[str, dict[str, Any]] = {}

    def fake_ir(path, sensor, channels, dt_min, dt_max, satellites, decode_workers):
        calls["ir"] = {
            "sensor": sensor,
            "channels": channels,
            "satellites": satellites,
        }
        return empty

    def fake_mw(path, sensor, plan, dt_min, dt_max, satellites, decode_workers):
        calls["mw"] = {"sensor": sensor, "satellites": satellites}
        return empty

    monkeypatch.setattr(nnja, "decode_ir_sounder", fake_ir)
    monkeypatch.setattr(nnja, "decode_microwave", fake_mw)

    cycle = datetime(2024, 1, 1)
    cris_task = nnja._NNJASatTask(
        uri="uri",
        datetime_file=cycle,
        datetime_min=cycle,
        datetime_max=cycle,
        sensor="cris",
        var_plan={"cris": "SRAD"},
    )
    frame = source._decode_file("path", cris_task)
    assert list(frame.columns) == list(NNJAObsSat.SCHEMA.names)
    # Every published channel is decoded; narrowing happens downstream
    assert calls["ir"]["channels"] is None
    # The user filter is narrowed to the sensor's own platforms
    assert calls["ir"]["satellites"] == ("npp",)
    assert "mw" not in calls

    atms_task = nnja._NNJASatTask(
        uri="uri",
        datetime_file=cycle,
        datetime_min=cycle,
        datetime_max=cycle,
        sensor="atms",
        var_plan={"atms": "TMBR"},
    )
    source._decode_file("path", atms_task)
    assert calls["mw"]["sensor"] == "atms"
    assert calls["mw"]["satellites"] == ("aqua", "metop-b", "npp")


def test_nnja_obs_sat_lexicon_ir_routes():
    from earth2studio.lexicon import NNJAObsSatLexicon

    assert NNJAObsSatLexicon.VOCAB["airs"] == "airs::TMBR"
    assert NNJAObsSatLexicon.VOCAB["iasi"] == "iasi::SCRA"
    assert NNJAObsSatLexicon.VOCAB["cris"] == "cris::SRAD"


def _airs_ir_pairs() -> list[tuple[int, Any]]:
    # Mirrors the real airsev channel loop layout: CHNM, LOGRCW, ACQF, TMBR
    return _ir_scalar_pairs(said=784) + [
        (31002, 3),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._LOG10_CENTRAL_WAVENUMBER, np.log10(649.62 * 100.0)),
        (ncep_microwave._ACQF, 0),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 285.5),
        (ncep_microwave._CHANNEL_NUMBER, 6),
        (ncep_microwave._LOG10_CENTRAL_WAVENUMBER, np.log10(650.814 * 100.0)),
        (ncep_microwave._ACQF, 2),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 250.25),
        # A channel missing LOGRCW keeps its BT but has no wavenumber
        (ncep_microwave._CHANNEL_NUMBER, 92),
        (ncep_microwave._ACQF, 0),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 260.0),
    ]


def test_nnja_ir_decode_airs_brightness_temperature_passthrough():
    rows = _decode_ir_pairs(_airs_ir_pairs(), "airs")

    assert len(rows) == 3
    by_channel = {row["sensor_index"]: row for row in rows}
    assert by_channel[1]["observation"] == 285.5
    # Wavenumber decoded from the file's per-channel LOGRCW field
    assert by_channel[1]["wavenumber"] == pytest.approx(649.62)
    assert by_channel[1]["quality"] == 0
    assert by_channel[6]["observation"] == 250.25
    assert by_channel[6]["wavenumber"] == pytest.approx(650.814)
    assert by_channel[6]["quality"] == 2
    assert by_channel[92]["observation"] == 260.0
    assert np.isnan(by_channel[92]["wavenumber"])
    assert by_channel[1]["variable"] == "airs"
    assert by_channel[1]["satellite"] == "aqua"

    # The decode-layer channel filter still narrows rows (internal capability)
    only = _decode_ir_pairs(_airs_ir_pairs(), "airs", channels_filter=frozenset({6}))
    assert [row["sensor_index"] for row in only] == [6]

    # Platform filtering drops the whole footprint
    assert not _decode_ir_pairs(_airs_ir_pairs(), "airs", satellites=frozenset({"npp"}))


def test_nnja_ir_decode_iasi_scaled_radiance_planck():
    from earth2studio.data.utils import PLANCK_C1 as C1
    from earth2studio.data.utils import PLANCK_C2 as C2

    wavenumber = 645.0  # IASI channel 1
    target_bt = 280.0
    radiance_mw = C1 * wavenumber**3 / (np.exp(C2 * wavenumber / target_bt) - 1.0)
    chsf = 7
    scra = radiance_mw * 10.0**chsf / 1e5

    pairs = _ir_scalar_pairs(said=3) + [
        (ncep_microwave._STCH, 1),
        (ncep_microwave._ENCH, 8461),
        (ncep_microwave._CHSF, chsf),
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._SCRA, scra),
    ]
    rows = _decode_ir_pairs(pairs, "iasi")

    assert len(rows) == 1
    assert rows[0]["observation"] == pytest.approx(target_bt, abs=1e-6)
    assert rows[0]["wavenumber"] == pytest.approx(645.0)
    assert rows[0]["satellite"] == "metop-b"

    # A channel outside every CHSF band cannot be converted and is skipped
    no_band = _ir_scalar_pairs(said=3) + [
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._SCRA, scra),
    ]
    assert not _decode_ir_pairs(no_band, "iasi")


def test_nnja_ir_decode_cris_radiance_planck_band_wavenumbers():
    from earth2studio.data.utils import PLANCK_C1 as C1
    from earth2studio.data.utils import PLANCK_C2 as C2

    wavenumber = 1210.0  # CrIS channel 714, first mid-wave channel
    target_bt = 260.0
    radiance_mw = C1 * wavenumber**3 / (np.exp(C2 * wavenumber / target_bt) - 1.0)
    srad = radiance_mw / 1000.0

    pairs = _ir_scalar_pairs(said=224) + [
        (ncep_microwave._FORN, 15),
        # Three per-band NFQF/NCQF occurrences (LW, MW, SW) precede the
        # channel loop in crisf4; channel 714 is the first mid-wave channel
        (ncep_microwave._NFQF, 1),
        (ncep_microwave._NCQF, 0),
        (ncep_microwave._NFQF, 4),
        (ncep_microwave._NCQF, 2),
        (ncep_microwave._NFQF, 0),
        (ncep_microwave._NCQF, 0),
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 714),
        (ncep_microwave._SRAD, srad),
    ]
    rows = _decode_ir_pairs(pairs, "cris")

    assert len(rows) == 1
    assert rows[0]["observation"] == pytest.approx(target_bt, abs=1e-6)
    assert rows[0]["wavenumber"] == pytest.approx(1210.0)
    assert rows[0]["satellite"] == "npp"
    # CrIS scan_position is FORN (cross-track 1-30), not FOVN (detector 1-9)
    assert rows[0]["scan_position"] == 15
    # Mid-wave band quality: NFQF=4 | NCQF=2 << 19
    assert rows[0]["quality"] == 4 | (2 << 19)


def test_nnja_ir_decode_cris_quality_null_without_band_flags():
    # A crisf4 subset without NFQF/NCQF occurrences leaves quality null
    pairs = _ir_scalar_pairs(said=224) + [
        (ncep_microwave._FORN, 3),
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 714),
        (ncep_microwave._SRAD, 0.05),
    ]
    rows = _decode_ir_pairs(pairs, "cris")
    assert len(rows) == 1
    assert rows[0]["quality"] is None


def test_nnja_ir_decode_cris_guard_block_not_emitted():
    # crisf4's guard-spectrum block opens with DRF8BIT (0-31-001) and reuses
    # CHNM/SRAD; its slots must not be decoded as science channels
    pairs = _ir_scalar_pairs(said=224) + [
        (ncep_microwave._FORN, 3),
        # Budget deliberately larger than the declared channels so the
        # 31001 marker, not the budget, is what stops the guard block
        (31002, 13),
        (ncep_microwave._CHANNEL_NUMBER, 714),
        (ncep_microwave._SRAD, 0.05),
        # Guard block: replication marker then guard CHNM/SRAD slots
        (31001, 12),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._SRAD, 0.07),
    ]
    rows = _decode_ir_pairs(pairs, "cris")
    assert [row["sensor_index"] for row in rows] == [714]


def test_nnja_ir_decode_second_marker_break_is_airs_only():
    # A second DRF16BIT does not terminate IASI/CrIS decoding; the channel
    # budget is the terminator, so a nested replication block added to a
    # future product cannot silently truncate footprints
    pairs = _ir_scalar_pairs(said=224) + [
        (ncep_microwave._FORN, 3),
        (31002, 2),
        (ncep_microwave._CHANNEL_NUMBER, 714),
        (ncep_microwave._SRAD, 0.05),
        (31002, 5),  # unexpected nested marker mid-block
        (ncep_microwave._CHANNEL_NUMBER, 715),
        (ncep_microwave._SRAD, 0.05),
    ]
    rows = _decode_ir_pairs(pairs, "cris")
    assert sorted(row["sensor_index"] for row in rows) == [714, 715]


def test_nnja_ir_decode_footprint_without_scan_position_dropped():
    # scan_position is non-nullable uint16 in the output schema; a footprint
    # missing its cross-track position (FORN for CrIS, FOVN otherwise) is
    # dropped rather than emitting None into a non-nullable column
    pairs_no_forn = _ir_scalar_pairs(said=224) + [
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 714),
        (ncep_microwave._SRAD, 0.05),
    ]
    assert not _decode_ir_pairs(pairs_no_forn, "cris")

    pairs_no_fovn = [
        pair
        for pair in _ir_scalar_pairs(said=784)
        if pair[0] != ncep_microwave._FOV_NUMBER
    ] + [
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._LOG10_CENTRAL_WAVENUMBER, np.log10(649.62 * 100.0)),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 285.5),
    ]
    assert not _decode_ir_pairs(pairs_no_fovn, "airs")


def test_nnja_ir_decode_airs_stops_at_second_replication_block():
    # airsev appends AMSU-A/HSB channel blocks that reuse the TMBR
    # descriptor; their channels must not be emitted as AIRS rows
    pairs = _airs_ir_pairs() + [
        (31002, 2),
        (ncep_microwave._CHANNEL_NUMBER, 2),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 199.75),
        (ncep_microwave._CHANNEL_NUMBER, 3),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 201.5),
    ]
    rows = _decode_ir_pairs(pairs, "airs")
    assert sorted(row["sensor_index"] for row in rows) == [1, 6, 92]


def test_nnja_ir_decode_failure_raises(monkeypatch, tmp_path):
    # decode_ir_sounder honors the same strict completeness contract as
    # decode_microwave: failed messages raise instead of silently truncating
    bufr = tmp_path / "fake.bufr"
    bufr.write_bytes(b"")
    monkeypatch.setattr(
        ncep_microwave,
        "_parse_prepbufr_messages",
        lambda *a, **k: ({1: 1}, {1: 1}, [(b"m", None)]),
    )
    monkeypatch.setattr(ncep_microwave, "_init_decode_worker", lambda *a: None)
    monkeypatch.setattr(
        ncep_microwave, "_decode_ir_message_batch", lambda argument: ([], 1)
    )
    with pytest.raises(ncep_microwave._NCEPIRSounderDecodeError) as err:
        ncep_microwave.decode_ir_sounder(
            str(bufr),
            "cris",
            None,
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            decode_workers=1,
        )
    assert err.value.context["failed_messages"] == 1


def test_nnja_obs_sat_sensor_indices_narrow_ir_decode(monkeypatch):
    with pytest.raises(ValueError, match="sensor_indices"):
        NNJAObsSat(sensor_indices={"atms": [1]}, cache=False, verbose=False)

    source = NNJAObsSat(
        sensor_indices={"cris": [19, 24]},
        cache=False,
        verbose=False,
        decode_workers=1,
    )
    seen = {}

    def fake_ir(path, sensor, channels, dt_min, dt_max, satellites, decode_workers):
        seen[sensor] = channels
        return ncep_microwave._rows_to_dataframe([])

    monkeypatch.setattr(nnja, "decode_ir_sounder", fake_ir)
    cycle = datetime(2024, 1, 1)

    for sensor in ["cris", "iasi"]:
        task = nnja._NNJASatTask(
            uri="uri",
            datetime_file=cycle,
            datetime_min=cycle,
            datetime_max=cycle,
            sensor=sensor,
            var_plan={sensor: "X"},
        )
        source._decode_file("path", task)

    assert seen["cris"] == frozenset({19, 24})
    assert seen["iasi"] is None


def test_nnja_obs_sat_sensor_indices_validation():
    # Empty selection would silently return nothing after a full download;
    # raise at construction instead, matching the JPSS/METOP sources
    with pytest.raises(ValueError, match="at least one"):
        NNJAObsSat(sensor_indices={"cris": []}, cache=False, verbose=False)
    # Out of instrument-grid range
    with pytest.raises(ValueError, match="1 to 2211"):
        NNJAObsSat(sensor_indices={"cris": [0]}, cache=False, verbose=False)
    with pytest.raises(ValueError, match="1 to 2211"):
        NNJAObsSat(sensor_indices={"cris": [2212]}, cache=False, verbose=False)
    with pytest.raises(ValueError, match="1 to 8461"):
        NNJAObsSat(sensor_indices={"iasi": [8462]}, cache=False, verbose=False)
    with pytest.raises(ValueError, match="1 to 2378"):
        NNJAObsSat(sensor_indices={"airs": [2379]}, cache=False, verbose=False)
    # Non-integer and bool entries
    with pytest.raises(ValueError, match="integer"):
        NNJAObsSat(sensor_indices={"cris": [1.5]}, cache=False, verbose=False)
    with pytest.raises(ValueError, match="integer"):
        NNJAObsSat(sensor_indices={"cris": [True]}, cache=False, verbose=False)
    # Duplicates
    with pytest.raises(ValueError, match="unique"):
        NNJAObsSat(sensor_indices={"cris": [19, 19]}, cache=False, verbose=False)
    # Valid boundary selections construct fine
    source = NNJAObsSat(
        sensor_indices={"cris": [1, 2211], "iasi": [1, 8461], "airs": [1, 2378]},
        cache=False,
        verbose=False,
    )
    assert source._sensor_indices["cris"] == frozenset({1, 2211})


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


def test_nnja_obs_sat_fetch_mixed_microwave_and_ir(monkeypatch, tmp_path):
    # The folded single entry point returns one Arrow-typed frame for a
    # request spanning both sensor families
    source = NNJAObsSat(
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
        decode_workers=1,
    )

    async def fake_fetch_files(uris):
        return None

    def _row(variable, observation, wavenumber):
        return {
            "time": np.datetime64("2019-01-01T00:00:00", "ns"),
            "class": "rad",
            "lat": 10.0,
            "lon": 240.0,
            "elev": np.nan,
            "scan_angle": np.nan,
            "scan_position": 7,
            "scan_line": 8,
            "sensor_index": 19,
            "wavenumber": wavenumber,
            "solza": 90.0,
            "solaza": 100.0,
            "satellite_za": 50.0,
            "satellite_aza": 200.0,
            "quality": None,
            "satellite": "npp",
            "observation": observation,
            "variable": variable,
        }

    def fake_decode(path, task):
        if task.sensor in nnja._NNJA_IR_SENSORS:
            frame = ncep_microwave._rows_to_dataframe([_row("cris", 250.0, 661.25)])
        else:
            frame = ncep_microwave._rows_to_dataframe([_row("atms", 220.0, np.nan)])
        return frame[NNJAObsSat.SCHEMA.names]

    cached = tmp_path / "cached.bufr"
    cached.write_bytes(b"fake")
    monkeypatch.setattr(source, "fetch_files", fake_fetch_files)
    monkeypatch.setattr(source, "local_path", lambda uri: str(cached))
    monkeypatch.setattr(source, "_decode_file", fake_decode)

    df = source(datetime(2019, 1, 1), ["atms", "cris"])

    assert list(df.columns) == NNJAObsSat.SCHEMA.names
    assert set(df["variable"]) == {"atms", "cris"}
    assert len(df) == 2
    # One dtype contract across sensor families after concat
    assert str(df["scan_position"].dtype) == "uint16[pyarrow]"


def test_nnja_ir_decode_respects_channel_replication_count():
    # In real airsev subsets the AMSU-A/HSB blocks that follow the sounder
    # channels are not always preceded by another replication descriptor;
    # the declared count is the reliable block terminator
    pairs = _ir_scalar_pairs(said=784) + [
        (31002, 1),
        (ncep_microwave._CHANNEL_NUMBER, 1),
        (ncep_microwave._LOG10_CENTRAL_WAVENUMBER, np.log10(649.62 * 100.0)),
        (ncep_microwave._ACQF, 0),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 285.5),
        # AMSU-A style channels without a second 31002 marker
        (ncep_microwave._CHANNEL_NUMBER, 3),
        (ncep_microwave._LOG10_CENTRAL_WAVENUMBER, np.log10(0.8 * 100.0)),
        (ncep_microwave._BRIGHTNESS_TEMPERATURE, 210.0),
    ]
    rows = _decode_ir_pairs(pairs, "airs")
    assert [row["sensor_index"] for row in rows] == [1]
