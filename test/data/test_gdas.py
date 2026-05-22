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
import pytest

from earth2studio.data import NomadsGDASObsConv
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
    url = NomadsGDASObsConv._build_url(datetime(2026, 4, 5, 12))
    expected = (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/"
        "gdas.20260405/gdas.t12z.prepbufr.nr"
    )
    assert url == expected


@pytest.mark.timeout(15)
def test_nomads_gdas_lexicon():
    # All expected variables should be in the lexicon
    expected_vars = ["u", "v", "t", "q", "pres"]
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
    base_time = datetime.utcnow().replace(
        minute=0, second=0, microsecond=0
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
            "class": ["1", "1"],
            "lat": np.array([40.0, 35.0], dtype=np.float32),
            "lon": np.array([250.0, 280.0], dtype=np.float32),
            "station": ["72451", "72520"],
            "station_elev": np.array([300.0, 200.0], dtype=np.float32),
            "quality": np.array([0, 0], dtype=np.uint16),
            "observation": np.array([250.0, 288.0], dtype=np.float32),
            "variable": ["t", "t"],
        }
    )

    async def mock_fetch(url):
        return str(tmp_path / "mock.bin")

    with (
        patch.object(
            NomadsGDASObsConv,
            "_fetch_remote_file",
            side_effect=mock_fetch,
        ) as mf,
        patch.object(
            NomadsGDASObsConv,
            "_decode_prepbufr",
            return_value=mock_df,
        ) as md,
        patch.object(
            NomadsGDASObsConv,
            "_cache_path",
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
    assert all(t.url.endswith(".prepbufr.nr") for t in tasks)
    assert all("t" in t.variables for t in tasks)

    # Task URLs should be for cycle times around midnight
    for task in tasks:
        assert "gdas." in task.url


@pytest.mark.timeout(15)
def test_nomads_gdas_empty_result():
    ds = NomadsGDASObsConv()
    empty_df = ds._compile_dataframe([], ["t"])

    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty
    assert list(empty_df.columns) == ds.SCHEMA.names


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
