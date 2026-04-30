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
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from earth2studio.data import GHCNDaily
from earth2studio.lexicon.ghcn import GHCNLexicon


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2023, month=7, day=4),
        [
            datetime(year=2022, month=1, day=1),
            datetime(year=2023, month=6, day=15),
        ],
    ],
)
@pytest.mark.parametrize(
    "stations, variable, tol",
    [
        (
            ["USW00013722"],
            ["t2m_max"],
            timedelta(days=0),
        ),
        (
            ["USW00013722", "USW00023234"],
            ["t2m_max", "tp"],
            timedelta(days=1),
        ),
    ],
)
def test_ghcn_fetch(stations, time, variable, tol):
    ds = GHCNDaily(stations=stations, time_tolerance=tol, cache=False)
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
        time_union["time"] = time_union["time"] | (
            df_times.ge(min_time) & df_times.le(max_time)
        )

    assert time_union["time"].all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2023-01-01")]),
    ],
)
@pytest.mark.parametrize("variable", [["t2m_max", "tp"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ghcn_cache(time, variable, cache):
    ds = GHCNDaily(
        stations=["USW00013722"],
        time_tolerance=timedelta(days=0),
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
def test_ghcn_schema_fields():
    station = "USW00013722"
    time = np.array(["2023-01-01"], dtype=np.datetime64)

    ds = GHCNDaily(stations=[station], time_tolerance=timedelta(days=0))

    # Test with default schema (all fields)
    df_full = ds(time, ["t2m_max"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    # Test with subset of fields
    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t2m_max"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_ghcn_exceptions():
    # Throw key error for invalid variable
    ds = GHCNDaily(
        stations=["USW00013722"],
        time_tolerance=timedelta(days=0),
        cache=False,
        verbose=False,
    )
    with pytest.raises(KeyError):
        ds(np.array([np.datetime64("2023-01-01")]), ["invalid"])

    # For an invalid station should return empty
    ds = GHCNDaily(stations=["INVALID00000"], cache=False, verbose=False)
    df = ds(np.array(["2023-01-01"], dtype=np.datetime64), ["t2m_max"])
    assert df.empty

    # Time that there is no data for should return empty
    ds = GHCNDaily(stations=["USW00013722"], cache=False, verbose=False)
    df = ds(np.array(["2050-01-01"], dtype=np.datetime64), ["t2m_max"])
    assert df.empty
    assert list(df.columns) == ds.SCHEMA.names

    with pytest.raises(KeyError):
        ds(
            np.datetime64("2023-01-01"),
            ["t2m_max"],
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
        ds(np.datetime64("2023-01-01"), ["t2m_max"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),  # Should be timestamp, not string
        ]
    )
    with pytest.raises(TypeError):
        ds(np.datetime64("2023-01-01"), ["t2m_max"], fields=wrong_type_schema)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "bbox",
    [
        # [-180, 180)
        (35.0, -80.0, 36.0, -78.0),
        # [0, 360)
        (35.0, 280.0, 36.0, 282.0),
    ],
)
def test_ghcn_station_bbox(bbox):
    lat_min, lon_min, lat_max, lon_max = bbox

    stations = GHCNDaily.get_stations_bbox((lat_min, lon_min, lat_max, lon_max))
    assert len(stations) > 0

    # Load station metadata to verify coordinates
    df = GHCNDaily.get_station_metadata()
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])

    lon_180 = df["LON"]
    lon_360 = (df["LON"] + 360.0) % 360.0

    # Verify each returned station falls inside bbox
    for sid in stations[:10]:  # Check first 10 to keep test fast
        rows = df[df["ID"] == sid]
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


def test_ghcn_schema_structure():
    assert "time" in GHCNDaily.SCHEMA.names
    assert "lat" in GHCNDaily.SCHEMA.names
    assert "lon" in GHCNDaily.SCHEMA.names
    assert "station" in GHCNDaily.SCHEMA.names
    assert "observation" in GHCNDaily.SCHEMA.names
    assert "variable" in GHCNDaily.SCHEMA.names
    assert GHCNDaily.SCHEMA.field("time").type == pa.timestamp("ns")
    assert GHCNDaily.SCHEMA.field("observation").type == pa.float32()


def test_ghcn_column_map():
    cmap = GHCNDaily.column_map()
    assert cmap["DATE"] == "time"
    assert cmap["LAT"] == "lat"
    assert cmap["LON"] == "lon"
    assert cmap["ID"] == "station"
    assert cmap["ELEV"] == "elev"


def test_ghcn_available_invalid_time():
    # Invalid time (before 1750) — rejected by _validate_time without network
    assert not GHCNDaily.available(datetime(1700, 1, 1))
    assert not GHCNDaily.available(np.datetime64("1700-01-01"))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_ghcn_available():
    # Valid time — partition should exist on S3
    assert GHCNDaily.available(datetime(2020, 1, 1))
    assert GHCNDaily.available(np.datetime64("2020-01-01"))

    # Invalid time (before 1750)
    assert not GHCNDaily.available(datetime(1700, 1, 1))
    assert not GHCNDaily.available(np.datetime64("1700-01-01"))

    # Future time — partition unlikely to exist
    assert not GHCNDaily.available(datetime(2099, 1, 1))


class TestGHCNLexicon:
    @pytest.mark.parametrize(
        "var, element",
        [
            ("t2m_max", "TMAX"),
            ("t2m_min", "TMIN"),
            ("t2m", "TAVG"),
            ("d2m", "ADPT"),
            ("r2m", "RHAV"),
            ("tp", "PRCP"),
            ("sf", "WESF"),
            ("sd", "SNWD"),
            ("sde", "SNOW"),
            ("ws10m", "AWND"),
            ("fg10m", "WSF2"),
            ("tcc", "ACMH"),
        ],
    )
    def test_element_map(self, var, element):
        assert GHCNLexicon.VOCAB[var] == element

    def test_lexicon_keys(self):
        for var in GHCNLexicon.VOCAB:
            element, mod = GHCNLexicon[var]
            assert element == GHCNLexicon.VOCAB[var]
            assert callable(mod)

    @pytest.mark.parametrize(
        "var, raw, expected",
        [
            ("t2m_max", np.array([200.0]), np.array([293.15])),  # 20.0 C -> K
            ("t2m_min", np.array([-100.0]), np.array([263.15])),  # -10.0 C -> K
            ("t2m", np.array([200.0]), np.array([293.15])),  # 20.0 C -> K
            ("d2m", np.array([150.0]), np.array([288.15])),  # 15.0 C -> K
            ("r2m", np.array([75.0]), np.array([75.0])),  # percent (no conv)
            ("tp", np.array([100.0]), np.array([0.01])),  # 10.0 mm -> 0.01 m
            ("sf", np.array([100.0]), np.array([0.01])),  # 10.0 mm -> 0.01 m
            ("sd", np.array([500.0]), np.array([0.5])),  # 500 mm -> 0.5 m
            ("sde", np.array([250.0]), np.array([0.25])),  # 250 mm -> 0.25 m
            ("ws10m", np.array([50.0]), np.array([5.0])),  # 50 tenths -> 5 m/s
            ("fg10m", np.array([120.0]), np.array([12.0])),  # 120 tenths -> 12 m/s
            ("tcc", np.array([80.0]), np.array([0.8])),  # 80% -> 0.8 fraction
        ],
    )
    def test_unit_conversions(self, var, raw, expected):
        _, mod = GHCNLexicon[var]
        np.testing.assert_allclose(mod(raw), expected, atol=1e-6)

    def test_invalid_variable(self):
        with pytest.raises(KeyError):
            GHCNLexicon["nonexistent"]


class TestGHCNMock:
    """Mock tests that run without network access."""

    def _build_mock_parquet_bytes(self) -> bytes:
        """Create a fake parquet file with GHCN-like data."""
        df = pd.DataFrame(
            {
                "ID": ["USW00013722", "USW00013722", "USW00099999"],
                "DATE": ["20230101", "20230102", "20230101"],
                "DATA_VALUE": [250, 300, 100],
                "Q_FLAG": [None, None, None],
            }
        )
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        return buf.getvalue().to_pybytes()

    def _build_station_metadata(self) -> pd.DataFrame:
        """Create fake station metadata."""
        return pd.DataFrame(
            {
                "ID": ["USW00013722", "USW00099999"],
                "LAT": [33.63, 40.0],
                "LON": [-84.44, -75.0],
                "ELEV": [315.0, 100.0],
                "STATE": ["GA", "PA"],
                "NAME": ["ATLANTA", "TESTVILLE"],
                "GSN": [None, None],
                "HCN": [None, None],
                "WMO": [None, None],
            }
        )

    @pytest.mark.parametrize(
        "variable,element,raw_value,expected_obs",
        [
            ("t2m_max", "TMAX", 250, 298.15),  # 25.0 C -> K
            ("t2m_min", "TMIN", -100, 263.15),  # -10.0 C -> K
            ("t2m", "TAVG", 200, 293.15),  # 20.0 C -> K
            ("d2m", "ADPT", 150, 288.15),  # 15.0 C -> K
            ("tp", "PRCP", 100, 0.01),  # 10.0 mm -> 0.01 m
            ("sd", "SNWD", 500, 0.5),  # 500 mm -> 0.5 m
            ("ws10m", "AWND", 50, 5.0),  # 50 tenths m/s -> 5 m/s
            ("tcc", "ACMH", 80, 0.8),  # 80% -> 0.8 fraction
        ],
    )
    @patch("earth2studio.data.ghcn.GHCNDaily.get_station_metadata")
    def test_mock_fetch(
        self, mock_get_meta, variable, element, raw_value, expected_obs
    ):
        """Test GHCNDaily fetch with mocked S3 filesystem for various variables."""
        mock_get_meta.return_value = self._build_station_metadata()

        # Build parquet with the specified raw value
        df = pd.DataFrame(
            {
                "ID": ["USW00013722", "USW00099999"],
                "DATE": ["20230101", "20230101"],
                "DATA_VALUE": [raw_value, raw_value + 10],
                "Q_FLAG": [None, None],
            }
        )
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        parquet_bytes = buf.getvalue().to_pybytes()

        ds = GHCNDaily(
            stations=["USW00013722"],
            time_tolerance=timedelta(days=0),
            cache=False,
            verbose=False,
        )

        # Mock the async filesystem
        mock_fs = MagicMock()
        mock_fs.set_session = AsyncMock(return_value=MagicMock(close=AsyncMock()))
        mock_fs._ls = AsyncMock(
            return_value=[
                f"noaa-ghcn-pds/parquet/by_year/YEAR=2023/ELEMENT={element}/part.parquet"
            ]
        )
        mock_fs._cat_file = AsyncMock(return_value=parquet_bytes)
        ds.fs = mock_fs

        result = ds(datetime(2023, 1, 1), [variable])

        assert list(result.columns) == ds.SCHEMA.names
        assert set(result["variable"].unique()) == {variable}
        assert set(result["station"].unique()) == {"USW00013722"}
        # Verify unit conversion
        assert np.isclose(result["observation"].iloc[0], expected_obs, atol=1e-2)
        # Verify longitude normalization ([-84.44 + 360] % 360 = 275.56)
        assert all(result["lon"] >= 0)
        assert all(result["lon"] < 360)
