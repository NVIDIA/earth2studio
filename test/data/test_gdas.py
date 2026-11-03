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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

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

    # Test resolve_fields with None (all fields)
    fields = ds.resolve_fields(None)
    assert fields is None

    # Test resolve_fields with subset
    subset = ["time", "lat", "lon", "observation", "variable"]
    fields = ds.resolve_fields(subset)
    assert fields == subset

    # Test resolve_fields with single string
    fields = ds.resolve_fields("observation")
    assert fields == ["observation"]

    # Test resolve_fields with invalid field
    with pytest.raises(KeyError):
        ds.resolve_fields("nonexistent_field")

    # Test resolve_fields with schema
    fields = ds.resolve_fields(ds.SCHEMA)
    assert fields == ds.SCHEMA.names


@pytest.mark.timeout(15)
def test_nomads_gdas_exceptions():
    ds = NomadsGDASObsConv()

    # Invalid variable – use a recent time so validation passes before KeyError
    recent = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(
        hours=6
    )
    with pytest.raises(KeyError):
        ds(recent, "nonexistent_var")

    # Future time
    with pytest.raises(ValueError):
        ds(datetime(2030, 1, 1, 0), "t")

    # Too old
    with pytest.raises(ValueError):
        ds(datetime(2020, 1, 1, 0), "t")


@pytest.mark.timeout(15)
def test_nomads_gdas_available():
    # Future time should not be available
    assert not NomadsGDASObsConv.available(datetime(2030, 1, 1))

    # Very old time should not be available
    assert not NomadsGDASObsConv.available(datetime(2020, 1, 1))

    # numpy datetime64
    assert not NomadsGDASObsConv.available(np.datetime64("2030-01-01"))

    # Recent time should be available (use relative time within 2-day window)
    recent = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(
        hours=6
    )
    assert NomadsGDASObsConv.available(recent)


@pytest.mark.timeout(15)
def test_nomads_gdas_url_builder():
    ds = NomadsGDASObsConv(cache=False, verbose=False)
    uri = ds._build_uri("prepbufr", datetime(2026, 4, 5, 12))
    expected = "pub/data/nccf/com/obsproc/prod/" "gdas.20260405/gdas.t12z.prepbufr.nr"
    assert uri == expected

    gpsro_uri = ds._build_uri("gpsro", datetime(2026, 4, 5, 12))
    assert gpsro_uri == (
        "pub/data/nccf/com/obsproc/prod/" "gdas.20260405/gdas.t12z.gpsro.tm00.bufr_d.nr"
    )


@pytest.mark.timeout(15)
def test_nomads_gdas_lexicon():
    # All expected variables should be in the lexicon
    expected_vars = ["u", "v", "t", "q", "pres", "gps"]
    for var in expected_vars:
        assert var in GDASObsConvLexicon.VOCAB

    # get_item should return (route_prefixed_key, modifier)
    key, mod = GDASObsConvLexicon.get_item("t")
    assert key == "prepbufr::TOB"
    assert callable(mod)

    # Wind vars should have decomposition modifiers
    key_u, mod_u = GDASObsConvLexicon.get_item("u")
    assert key_u == "prepbufr::wind::u"

    key_v, mod_v = GDASObsConvLexicon.get_item("v")
    assert key_v == "prepbufr::wind::v"

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

    path1 = ds.local_path(url1)
    path2 = ds.local_path(url2)

    # Different URLs should produce different cache paths
    assert path1 != path2

    # Same URL should produce same cache path
    assert ds.local_path(url1) == path1

    # Should be in the cache directory
    assert path1.startswith(ds.cache)


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
    base_time = datetime.utcnow().replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)
    obs_time1 = base_time + timedelta(minutes=30)
    obs_time2 = base_time + timedelta(hours=1)

    # Mock _decode_file output: wide PrepBUFR level rows in E2S units (t in K)
    mock_df = pa.table(
        {
            "time": pa.array(
                [np.datetime64(obs_time1, "ns"), np.datetime64(obs_time2, "ns")]
            ),
            "observation_t": pa.array([250.0, 288.0], type=pa.float32()),
            "quality_t": pa.array([0, 0], type=pa.uint16()),
        }
    )

    async def mock_fetch(uri):
        return str(tmp_path / "mock.bin")

    with (
        patch.object(
            NomadsGDASObsConv,
            "_fetch_remote_file",
            side_effect=mock_fetch,
        ) as mf,
        patch.object(
            NomadsGDASObsConv,
            "_decode_file",
            return_value=mock_df,
        ) as md,
        patch.object(
            NomadsGDASObsConv,
            "local_path",
            return_value=str(tmp_path / "mock.bin"),
        ),
    ):
        # Create a dummy file so os.path.exists check passes
        dummy_file = tmp_path / "mock.bin"
        dummy_file.write_bytes(b"dummy")

        ds = NomadsGDASObsConv(
            time_tolerance=timedelta(hours=3),
            cache=False,
        )

        df = ds(base_time, ["t"])

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ds.SCHEMA.names
        assert set(df["variable"].unique()) == {"t"}
        mf.assert_called()
        md.assert_called()


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

    # Task URIs should be for cycle times around midnight
    for task in tasks:
        assert "gdas." in task.uri

    mixed_ds = NomadsGDASObsConv(time_tolerance=timedelta(0))
    mixed_tasks = mixed_ds._create_tasks([datetime(2026, 4, 4, 0)], ["t", "gps"])
    assert len(mixed_tasks) == 2
    prepbufr_task = next(task for task in mixed_tasks if task.route == "prepbufr")
    gpsro_task = next(task for task in mixed_tasks if task.route == "gpsro")
    assert prepbufr_task.uri.endswith(".prepbufr.nr")
    assert gpsro_task.uri.endswith(".gpsro.tm00.bufr_d.nr")
    assert "t" in prepbufr_task.var_plan
    assert "gps" in gpsro_task.var_plan

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
def test_nomads_gdas_empty_result(tmp_path):
    dummy_file = tmp_path / "mock.bin"
    dummy_file.write_bytes(b"dummy")
    empty = utils_ncep._NCEP_PREPBUFR_WIDE_SCHEMA.empty_table()

    async def mock_fetch(uri):
        return str(dummy_file)

    with (
        patch.object(NomadsGDASObsConv, "_fetch_remote_file", side_effect=mock_fetch),
        patch.object(NomadsGDASObsConv, "_decode_file", return_value=empty),
        patch.object(NomadsGDASObsConv, "local_path", return_value=str(dummy_file)),
    ):
        ds = NomadsGDASObsConv(time_tolerance=timedelta(hours=3), cache=False)
        df = ds(datetime.utcnow().replace(minute=0, second=0, microsecond=0), ["t"])

    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names


@pytest.mark.timeout(15)
def test_nomads_gdas_lexicon_modifiers():
    # The decoders return Earth2Studio units, so the modifiers pass frames through
    frame = pd.DataFrame({"observation": np.array([0.0, 20.0], dtype=np.float32)})
    for var in GDASObsConvLexicon.VOCAB:
        _, modifier = GDASObsConvLexicon.get_item(var)
        assert modifier(frame) is frame
