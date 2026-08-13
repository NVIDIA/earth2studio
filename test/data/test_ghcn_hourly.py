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

import io
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import GHCNHourly
from earth2studio.lexicon.ghcn import GHCNHourlyLexicon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sparse_parquet_bytes(
    station: str = "USW00013874",
    date: str = "2024-01-01T12:00:00",
    temperature: float = 15.0,
    lat: float = 33.63,
    lon: float = -84.44,
    elev: float = 315.0,
) -> bytes:
    """Parquet with only mandatory + temperature columns — no wind/precip/sky."""
    df = pd.DataFrame(
        {
            "STATION": [station],
            "DATE": [date],
            "LATITUDE": [lat],
            "LONGITUDE": [lon],
            "ELEVATION": [elev],
            "temperature": [temperature],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _build_mock_parquet_bytes(
    station: str = "USW00013874",
    date: str = "2024-01-01T12:00:00",
    temperature: float = 20.0,  # °C
    dew_point_temperature: float = 10.0,  # °C
    wind_speed: float = 10.0,  # m/s
    wind_direction: str = "090",  # degrees as string, 999=missing
    wind_gust: float = np.nan,  # m/s
    precipitation: float = np.nan,  # mm
    sky_cover_layer_1: str | None = "FEW:02",
    lat: float = 33.63,
    lon: float = -84.44,
    elev: float = 315.0,
) -> bytes:
    """Return a minimal GHCNh parquet file as bytes."""
    df = pd.DataFrame(
        {
            "STATION": [station],
            "DATE": [date],
            "LATITUDE": [lat],
            "LONGITUDE": [lon],
            "ELEVATION": [elev],
            "temperature": [temperature],
            "dew_point_temperature": [dew_point_temperature],
            "wind_speed": [wind_speed],
            "wind_direction": [wind_direction],
            "wind_gust": [wind_gust],
            "precipitation": [precipitation],
            "sky_cover_layer_1": [sky_cover_layer_1],
            "sky_cover_layer_2": [None],
            "sky_cover_layer_3": [None],
            "sky_cover_layer_4": [None],
        }
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Schema / column-map (no network)
# ---------------------------------------------------------------------------


def test_ghcnh_schema_structure():
    assert "time" in GHCNHourly.SCHEMA.names
    assert "lat" in GHCNHourly.SCHEMA.names
    assert "lon" in GHCNHourly.SCHEMA.names
    assert "station" in GHCNHourly.SCHEMA.names
    assert "observation" in GHCNHourly.SCHEMA.names
    assert "variable" in GHCNHourly.SCHEMA.names
    assert GHCNHourly.SCHEMA.field("time").type == pa.timestamp("ns")
    assert GHCNHourly.SCHEMA.field("observation").type == pa.float32()


def test_ghcnh_column_map():
    cmap = GHCNHourly.column_map()
    assert cmap["DATE"] == "time"
    assert cmap["LATITUDE"] == "lat"
    assert cmap["LONGITUDE"] == "lon"
    assert cmap["STATION"] == "station"


def test_ghcnh_available_invalid_time():
    assert not GHCNHourly.available(datetime(1800, 1, 1))
    assert not GHCNHourly.available(np.datetime64("1800-01-01"))


def test_ghcnh_available_valid_time():
    assert GHCNHourly.available(datetime(2024, 1, 1))
    assert GHCNHourly.available(np.datetime64("2024-01-01"))


# ---------------------------------------------------------------------------
# resolve_fields
# ---------------------------------------------------------------------------


def test_ghcnh_resolve_fields_none():
    schema = GHCNHourly.resolve_fields(None)
    assert schema == GHCNHourly.SCHEMA


def test_ghcnh_resolve_fields_str():
    schema = GHCNHourly.resolve_fields("time")
    assert schema.names == ["time"]


def test_ghcnh_resolve_fields_list():
    schema = GHCNHourly.resolve_fields(["time", "lat", "lon"])
    assert schema.names == ["time", "lat", "lon"]


def test_ghcnh_resolve_fields_invalid():
    with pytest.raises(KeyError):
        GHCNHourly.resolve_fields(["nonexistent"])


def test_ghcnh_resolve_fields_schema_type_mismatch():
    bad_schema = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        GHCNHourly.resolve_fields(bad_schema)


# ---------------------------------------------------------------------------
# Mock fetch tests
# ---------------------------------------------------------------------------


class TestGHCNHourlyMock:
    """Unit tests that mock the network layer."""

    def _ds(self, station: str = "USW00013874", cache: bool = False) -> GHCNHourly:
        return GHCNHourly(
            stations=[station],
            time_tolerance=np.timedelta64(10, "m"),
            cache=cache,
            verbose=False,
        )

    def _mock_fs(self, parquet_bytes: bytes) -> MagicMock:
        mock_fs = MagicMock()
        mock_fs.set_session = AsyncMock(return_value=MagicMock(close=AsyncMock()))
        mock_fs._cat_file = AsyncMock(return_value=parquet_bytes)
        return mock_fs

    def test_mock_ws10m(self):
        """wind_speed column returned as ws10m in m/s."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(wind_speed=10.0))

        result = ds(datetime(2024, 1, 1, 12), ["ws10m"])

        assert not result.empty
        assert set(result["variable"].unique()) == {"ws10m"}
        assert np.isclose(result["observation"].iloc[0], 10.0, atol=1e-3)

    def test_mock_u10m_v10m(self):
        """u10m / v10m derived from wind_direction=0° and wind_speed=10 m/s."""
        # dir=0 (wind from north) → u=0, v=-10
        ds = self._ds()
        ds.fs = self._mock_fs(
            _build_mock_parquet_bytes(wind_speed=10.0, wind_direction="000")
        )

        result = ds(datetime(2024, 1, 1, 12), ["u10m", "v10m"])
        assert set(result["variable"].unique()) == {"u10m", "v10m"}
        v = result.loc[result["variable"] == "v10m", "observation"].iloc[0]
        u = result.loc[result["variable"] == "u10m", "observation"].iloc[0]
        assert np.isclose(v, -10.0, atol=1e-3)
        assert np.isclose(u, 0.0, atol=1e-3)

    def test_mock_t2m(self):
        """temperature column converted from °C to K."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(temperature=20.0))

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 293.15, atol=1e-2)

    def test_mock_d2m(self):
        """dew_point_temperature column converted from °C to K."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(dew_point_temperature=10.0))

        result = ds(datetime(2024, 1, 1, 12), ["d2m"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 283.15, atol=1e-2)

    def test_mock_tp(self):
        """precipitation column converted from mm to m."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(precipitation=5.0))

        result = ds(datetime(2024, 1, 1, 12), ["tp"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 0.005, atol=1e-5)

    def test_mock_fg10m(self):
        """wind_gust column returned as fg10m in m/s."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(wind_gust=15.0))

        result = ds(datetime(2024, 1, 1, 12), ["fg10m"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 15.0, atol=1e-3)

    def test_mock_tcc_few(self):
        """sky_cover_layer_1='FEW' maps to ~0.1875 fraction."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(sky_cover_layer_1="FEW:02"))

        result = ds(datetime(2024, 1, 1, 12), ["tcc"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 0.1875, atol=1e-4)

    def test_mock_tcc_ovc(self):
        """sky_cover_layer_1='OVC' maps to 1.0 fraction."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(sky_cover_layer_1="OVC:000"))

        result = ds(datetime(2024, 1, 1, 12), ["tcc"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 1.0, atol=1e-4)

    def test_mock_tcc_clr(self):
        """sky_cover_layer_1='CLR' maps to 0.0 fraction."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(sky_cover_layer_1="CLR"))

        result = ds(datetime(2024, 1, 1, 12), ["tcc"])

        assert not result.empty
        assert np.isclose(result["observation"].iloc[0], 0.0, atol=1e-4)

    def test_schema_columns_present(self):
        """All SCHEMA columns appear in the output."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        for col in GHCNHourly.SCHEMA.names:
            assert col in result.columns, f"Missing column: {col}"

    def test_lon_normalized_to_360(self):
        """Longitude is converted to [0, 360) convention."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        assert (result["lon"] >= 0).all()
        assert (result["lon"] < 360).all()

    def test_no_data_in_window_returns_empty(self):
        """Returns empty DataFrame when no observations fall in the time window."""
        ds = self._ds()
        # parquet has a row at 12:00, request at 00:00 with ±10 min tolerance
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(date="2024-01-01T12:00:00"))

        result = ds(datetime(2024, 1, 1, 0), ["t2m"])

        assert result.empty
        assert list(result.columns) == GHCNHourly.SCHEMA.names

    def test_invalid_variable_raises(self):
        """KeyError raised for variable not in GHCNHourlyLexicon."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        with pytest.raises(KeyError):
            ds(datetime(2024, 1, 1, 12), ["not_a_variable"])

    def test_pre_1901_time_raises(self):
        """ValueError raised for times before the GHCNh record start."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        with pytest.raises(ValueError):
            ds(datetime(1800, 1, 1), ["t2m"])

    def test_sparse_parquet_missing_optional_columns(self):
        """Station parquet with no wind/precip/sky columns still returns t2m."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_sparse_parquet_bytes(temperature=15.0))

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        assert not result.empty
        assert set(result["variable"].unique()) == {"t2m"}
        assert np.isclose(result["observation"].iloc[0], 15.0 + 273.15, atol=1e-3)

    def test_missing_wind_gust_gives_nan(self):
        """NaN wind_gust column → fg10m rows dropped (no observations)."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(wind_gust=np.nan))

        result = ds(datetime(2024, 1, 1, 12), ["fg10m"])

        assert result.empty

    def test_cache_parquet_written(self, tmp_path, monkeypatch):
        """Parquet cache file is created when cache=True."""
        monkeypatch.setattr(
            "earth2studio.data.ghcn.datasource_cache_root", lambda: str(tmp_path)
        )
        ds = GHCNHourly(
            stations=["USW00013874"],
            time_tolerance=np.timedelta64(10, "m"),
            cache=True,
            verbose=False,
        )
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        ds(datetime(2024, 1, 1, 12), ["t2m"])

        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) == 1

    def test_station_attr_correct(self):
        """'station' column matches the requested station ID."""
        station = "USW00013874"
        ds = self._ds(station=station)
        ds.fs = self._mock_fs(_build_mock_parquet_bytes(station=station))

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        assert not result.empty
        assert set(result["station"].unique()) == {station}

    def test_source_attr_set(self):
        """DataFrame attrs['source'] is set to GHCNHourly.SOURCE_ID."""
        ds = self._ds()
        ds.fs = self._mock_fs(_build_mock_parquet_bytes())

        result = ds(datetime(2024, 1, 1, 12), ["t2m"])

        assert result.attrs.get("source") == GHCNHourly.SOURCE_ID

    def test_year_boundary_fetches_both_years(self, tmp_path, monkeypatch):
        """Tolerance window crossing Dec 31 → Jan 1 fetches both year files."""
        monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

        # Observation sits at 2023-12-31 23:50 — inside a 2h lower-tolerance
        # window around 2024-01-01 00:00.
        prev_year_bytes = _build_mock_parquet_bytes(
            date="2023-12-31T23:50:00", temperature=5.0
        )
        curr_year_bytes = _build_mock_parquet_bytes(
            date="2024-01-01T00:10:00", temperature=6.0
        )

        call_count = 0

        async def _cat_file(url: str) -> bytes:
            nonlocal call_count
            call_count += 1
            return prev_year_bytes if "2023" in url else curr_year_bytes

        mock_fs = MagicMock()
        mock_fs.set_session = AsyncMock(return_value=MagicMock(close=AsyncMock()))
        mock_fs._cat_file = _cat_file

        ds = GHCNHourly(
            stations=["USW00013874"],
            time_tolerance=(np.timedelta64(-2, "h"), np.timedelta64(30, "m")),
            cache=False,
            verbose=False,
        )
        ds.fs = mock_fs

        result = ds(datetime(2024, 1, 1, 0, 0), ["t2m"])

        # Both year files must have been fetched
        assert call_count == 2
        # Both observations should be in the result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Variable computation unit tests (no network, no async)
# ---------------------------------------------------------------------------


class TestGHCNHourlyVariables:
    """Direct tests of _add_variables on DataFrames."""

    def _ds(self) -> GHCNHourly:
        return GHCNHourly(stations=[], verbose=False)

    def _base_df(self, **kwargs) -> pd.DataFrame:
        defaults = {
            "STATION": ["S1"],
            "DATE": [pd.Timestamp("2024-01-01 12:00")],
            "LATITUDE": [10.0],
            "LONGITUDE": [20.0],
            "ELEVATION": [100.0],
            "temperature": [20.0],
            "dew_point_temperature": [10.0],
            "wind_speed": [5.0],
            "wind_direction": ["090"],
            "wind_gust": [np.nan],
            "precipitation": [np.nan],
            "sky_cover_layer_1": [None],
            "sky_cover_layer_2": [None],
            "sky_cover_layer_3": [None],
            "sky_cover_layer_4": [None],
        }
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_t2m_conversion(self):
        ds = self._ds()
        df = self._base_df(temperature=[0.0])
        result = ds._add_variables(df)
        assert np.isclose(result["t2m"].iloc[0], 273.15, atol=1e-2)

    def test_d2m_conversion(self):
        ds = self._ds()
        df = self._base_df(dew_point_temperature=[0.0])
        result = ds._add_variables(df)
        assert np.isclose(result["d2m"].iloc[0], 273.15, atol=1e-2)

    def test_ws10m_passthrough(self):
        ds = self._ds()
        df = self._base_df(wind_speed=[7.5])
        result = ds._add_variables(df)
        assert np.isclose(result["ws10m"].iloc[0], 7.5, atol=1e-5)

    def test_wind_missing_direction_gives_nan_uv(self):
        ds = self._ds()
        df = self._base_df(wind_direction=["999"])
        result = ds._add_variables(df)
        assert pd.isna(result["u10m"].iloc[0])
        assert pd.isna(result["v10m"].iloc[0])

    def test_uv_eastward_wind(self):
        # wind FROM east (090°) → u < 0 (wind moving westward)
        ds = self._ds()
        df = self._base_df(wind_speed=[10.0], wind_direction=["090"])
        result = ds._add_variables(df)
        assert np.isclose(result["u10m"].iloc[0], -10.0, atol=1e-3)
        assert np.isclose(result["v10m"].iloc[0], 0.0, atol=1e-3)

    def test_tp_mm_to_m(self):
        ds = self._ds()
        df = self._base_df(precipitation=[1000.0])
        result = ds._add_variables(df)
        assert np.isclose(result["tp"].iloc[0], 1.0, atol=1e-5)

    def test_fg10m_passthrough(self):
        ds = self._ds()
        df = self._base_df(wind_gust=[20.0])
        result = ds._add_variables(df)
        assert np.isclose(result["fg10m"].iloc[0], 20.0, atol=1e-5)

    def test_tcc_sct(self):
        ds = self._ds()
        df = self._base_df(sky_cover_layer_1=["SCT:030"])
        result = ds._add_variables(df)
        assert np.isclose(result["tcc"].iloc[0], 0.4375, atol=1e-4)

    def test_tcc_unknown_code_is_nan(self):
        ds = self._ds()
        df = self._base_df(sky_cover_layer_1=["UNKN:000"])
        result = ds._add_variables(df)
        assert pd.isna(result["tcc"].iloc[0])

    def test_tcc_max_across_layers(self):
        ds = self._ds()
        df = self._base_df(
            sky_cover_layer_1=["FEW:010"],  # 0.1875
            sky_cover_layer_2=["BKN:030"],  # 0.75
        )
        result = ds._add_variables(df)
        assert np.isclose(result["tcc"].iloc[0], 0.75, atol=1e-4)


# ---------------------------------------------------------------------------
# Lexicon tests (no network)
# ---------------------------------------------------------------------------


class TestGHCNHourlyLexicon:
    @pytest.mark.parametrize(
        "var",
        ["ws10m", "u10m", "v10m", "tp", "t2m", "fg10m", "d2m", "tcc"],
    )
    def test_vocab_keys_present(self, var):
        assert var in GHCNHourlyLexicon.VOCAB

    def test_invalid_variable_raises(self):
        with pytest.raises(KeyError):
            GHCNHourlyLexicon["nonexistent"]


# ---------------------------------------------------------------------------
# Slow / network tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time,variable",
    [
        (datetime(2024, 1, 1, 12), ["t2m", "ws10m"]),
        ([datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)], ["t2m"]),
    ],
)
def test_ghcnh_fetch(time, variable):
    """Fetch real data from NCEI GHCNh endpoint (requires network)."""
    # Atlanta Hartsfield-Jackson airport (USAF 722190, WBAN 13874)
    ds = GHCNHourly(
        stations=["USW00013874"],
        time_tolerance=np.timedelta64(30, "m"),
        cache=False,
        verbose=False,
    )
    result = ds(time, variable)

    assert list(result.columns) == ds.SCHEMA.names
    assert set(result["variable"].unique()).issubset(set(variable))
    assert not result.empty


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ghcnh_cache(tmp_path, monkeypatch):
    """Verify parquet cache avoids re-download."""
    monkeypatch.setattr(
        "earth2studio.data.ghcn.datasource_cache_root", lambda: str(tmp_path)
    )
    ds = GHCNHourly(
        stations=["USW00013874"],
        time_tolerance=np.timedelta64(30, "m"),
        cache=True,
        verbose=False,
    )
    time = datetime(2024, 1, 1, 12)
    ds(time, ["t2m"])
    parquet_count_after_first = len(list(tmp_path.rglob("*.parquet")))
    assert parquet_count_after_first > 0

    ds(time, ["t2m"])
    parquet_count_after_second = len(list(tmp_path.rglob("*.parquet")))
    assert parquet_count_after_second == parquet_count_after_first


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_ghcnh_station_bbox():
    """get_stations_bbox returns plausible results for a bay-area box."""
    stations = GHCNHourly.get_stations_bbox((37.0, -123.0, 38.0, -121.0))
    assert isinstance(stations, list)
    assert len(stations) > 0
    for sid in stations:
        assert len(sid) == 11
