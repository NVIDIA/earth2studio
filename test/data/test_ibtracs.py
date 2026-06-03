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

from earth2studio.data import IBTrACS
from earth2studio.lexicon.ibtracs import IBTrACSLexicon

# Use ACTIVE region for tests as it's the smallest file (~685KB)
_TEST_REGION = "ACTIVE"
# Use a recent time that should have data in ACTIVE
_TEST_TIME = datetime(year=2024, month=9, day=1, hour=0)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        _TEST_TIME,
        [_TEST_TIME, _TEST_TIME + timedelta(days=1)],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [
        ["tcwnd"],
        ["tcwnd", "mslp"],
    ],
)
def test_ibtracs_fetch(time, variable):
    ds = IBTrACS(region=_TEST_REGION, time_tolerance=timedelta(days=7), cache=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))

    if not isinstance(time, list):
        time = [time]

    # Check all rows are within requested time tolerance
    tol = timedelta(days=7)
    time_union = pd.Series([False] * len(df))
    for t in time:
        tmin = t - tol
        tmax = t + tol
        time_union = time_union | (df["time"].ge(tmin) & df["time"].le(tmax))
    assert time_union.all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_ibtracs_cache(cache):
    ds = IBTrACS(
        region=_TEST_REGION,
        time_tolerance=timedelta(days=7),
        cache=cache,
        verbose=False,
    )
    df = ds(_TEST_TIME, ["tcwnd"])

    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Second fetch exercises cache path
    df2 = ds(_TEST_TIME, ["tcwnd"])
    assert list(df2.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(10)
def test_ibtracs_exceptions():
    # Test invalid variable via lexicon
    with pytest.raises(KeyError):
        IBTrACSLexicon["nonexistent_variable"]

    # Test invalid region
    with pytest.raises(ValueError):
        IBTrACS(region="INVALID_REGION")

    # Test invalid time (before 1842)
    ds = IBTrACS(region="NA", cache=False)
    with pytest.raises(ValueError):
        ds(datetime(1800, 1, 1), ["tcwnd"])

    # Test invalid fields
    with pytest.raises(KeyError):
        IBTrACS.resolve_fields(["nonexistent_field"])


@pytest.mark.timeout(5)
def test_ibtracs_available():
    # Valid time should return True
    assert IBTrACS.available(datetime(2024, 6, 1, 0)) is True

    # Time before 1842 should return False
    assert IBTrACS.available(datetime(1800, 1, 1)) is False

    # Test with np.datetime64 input
    assert IBTrACS.available(np.datetime64("2024-06-01T00:00")) is True


@pytest.mark.timeout(5)
def test_ibtracs_schema_structure():
    assert "time" in IBTrACS.SCHEMA.names
    assert "lat" in IBTrACS.SCHEMA.names
    assert "lon" in IBTrACS.SCHEMA.names
    assert "track_id" in IBTrACS.SCHEMA.names
    assert "storm_name" in IBTrACS.SCHEMA.names
    assert "basin" in IBTrACS.SCHEMA.names
    assert "season" in IBTrACS.SCHEMA.names
    assert "observation" in IBTrACS.SCHEMA.names
    assert "variable" in IBTrACS.SCHEMA.names
    assert IBTrACS.SCHEMA.field("time").type == pa.timestamp("ns")
    assert IBTrACS.SCHEMA.field("observation").type == pa.float32()


@pytest.mark.timeout(5)
def test_ibtracs_list_regions():
    regions = IBTrACS.list_regions()
    assert "NA" in regions
    assert "EP" in regions
    assert "WP" in regions
    assert "ALL" in regions
    assert "ACTIVE" in regions
    assert "since1980" in regions


@pytest.mark.timeout(5)
def test_ibtracs_resolve_fields():
    # Test with None (returns full schema)
    schema = IBTrACS.resolve_fields(None)
    assert schema == IBTrACS.SCHEMA

    # Test with single string
    schema = IBTrACS.resolve_fields("time")
    assert len(schema) == 1
    assert schema.names == ["time"]

    # Test with list
    schema = IBTrACS.resolve_fields(["time", "lat", "lon"])
    assert len(schema) == 3
    assert schema.names == ["time", "lat", "lon"]

    # Test with PyArrow schema
    input_schema = pa.schema([pa.field("time", pa.timestamp("ns"))])
    schema = IBTrACS.resolve_fields(input_schema)
    assert schema == input_schema


@pytest.mark.timeout(5)
def test_ibtracs_multiple_regions():
    # Test that multiple regions can be specified
    ds = IBTrACS(region=["NA", "EP"], cache=False)
    assert ds.regions == ["NA", "EP"]


class TestIBTrACSLexicon:
    @pytest.mark.parametrize(
        "var, source_key",
        [
            ("tcwnd", "wmo_wind"),
            ("mslp", "wmo_pres"),
            ("tcustm", "storm_speed::u"),
            ("tcvstm", "storm_speed::v"),
            ("tcr34", "usa_r34"),
            ("tcr50", "usa_r50"),
            ("tcr64", "usa_r64"),
            ("tcsshs", "usa_sshs"),
            ("tcd2l", "dist2land"),
        ],
    )
    def test_vocab_map(self, var, source_key):
        assert IBTrACSLexicon.VOCAB[var] == source_key

    def test_lexicon_keys(self):
        for var in IBTrACSLexicon.VOCAB:
            source_key, mod = IBTrACSLexicon[var]
            assert source_key == IBTrACSLexicon.VOCAB[var]
            assert callable(mod)

    @pytest.mark.parametrize(
        "var, raw, expected",
        [
            # Wind speed: 100 knots -> 51.4444 m/s
            ("tcwnd", np.array([100.0]), np.array([51.4444])),
            # Pressure: 950 mb -> 95000 Pa
            ("mslp", np.array([950.0]), np.array([95000.0])),
            # Wind radii: 150 nmile -> 277.8 km
            ("tcr34", np.array([150.0]), np.array([277.8])),
            # Distance to land: identity (already km)
            ("tcd2l", np.array([100.0]), np.array([100.0])),
            # SSHS category: identity
            ("tcsshs", np.array([3.0]), np.array([3.0])),
        ],
    )
    def test_unit_conversions(self, var, raw, expected):
        _, mod = IBTrACSLexicon[var]
        np.testing.assert_allclose(mod(raw), expected, rtol=1e-3)

    def test_invalid_variable(self):
        with pytest.raises(KeyError):
            IBTrACSLexicon["nonexistent"]


class TestIBTrACSMock:
    """Mock tests that run without network access."""

    @pytest.mark.timeout(15)
    def test_mock_fetch(self, tmp_path, monkeypatch):
        """Test IBTrACS fetch with mocked filesystem."""
        monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

        # Create mock DataFrame result
        mock_df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-09-01 06:00:00", "2024-09-01 12:00:00"]),
                "lat": np.array([25.3, 26.1], dtype=np.float32),
                "lon": np.array([284.8, 285.2], dtype=np.float32),  # normalized
                "observation": np.array([51.44, 56.59], dtype=np.float32),
                "variable": ["tcwnd", "tcwnd"],
                "track_id": ["2024261N10138", "2024261N10138"],
                "storm_name": ["HELENE", "HELENE"],
                "basin": ["NA", "NA"],
                "season": np.array([2024, 2024], dtype=np.int32),
            }
        )

        # Patch _sync_async to return mock data
        with patch("earth2studio.data.ibtracs._sync_async") as mock_sync:
            mock_sync.return_value = mock_df
            ds = IBTrACS(region="NA", cache=False)
            df = ds(_TEST_TIME, ["tcwnd"])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == mock_df.columns.tolist()
        assert not df.empty
        assert set(df["variable"].unique()) == {"tcwnd"}
        assert df.attrs.get("source") is None  # attrs not preserved through mock

    @pytest.mark.timeout(15)
    def test_mock_compile_dataframe(self, tmp_path, monkeypatch):
        """Test _compile_dataframe with a mock NetCDF file."""
        monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

        # Create a minimal mock NetCDF file using netCDF4
        from netCDF4 import Dataset as NCDataset

        nc_path = tmp_path / "IBTrACS.MOCK.v04r01.nc"
        with NCDataset(str(nc_path), "w") as ds:
            # Create dimensions
            ds.createDimension("storm", 2)
            ds.createDimension("date_time", 3)
            ds.createDimension("char", 13)
            ds.createDimension("char4", 4)

            # Create variables
            time_var = ds.createVariable("time", "f8", ("storm", "date_time"))
            lat_var = ds.createVariable("lat", "f4", ("storm", "date_time"))
            lon_var = ds.createVariable("lon", "f4", ("storm", "date_time"))
            sid_var = ds.createVariable("sid", "S1", ("storm", "char"))
            name_var = ds.createVariable("name", "S1", ("storm", "char"))
            basin_var = ds.createVariable(
                "basin", "S1", ("storm", "date_time", "char4")
            )
            season_var = ds.createVariable("season", "i4", ("storm",))
            wmo_wind_var = ds.createVariable("wmo_wind", "f4", ("storm", "date_time"))

            # Fill with test data
            # Time is days since 1858-11-17
            # 2024-09-01 = ~60527 days since epoch
            base_days = (datetime(2024, 9, 1) - datetime(1858, 11, 17)).days
            time_var[0, :] = [base_days, base_days + 0.25, base_days + 0.5]
            time_var[1, :] = [base_days, np.nan, np.nan]

            lat_var[0, :] = [25.0, 26.0, 27.0]
            lat_var[1, :] = [15.0, np.nan, np.nan]

            lon_var[0, :] = [-75.0, -76.0, -77.0]
            lon_var[1, :] = [-120.0, np.nan, np.nan]

            # Storm IDs as char arrays
            sid1 = list("2024261N10138")
            sid2 = list("2024262N10120")
            for i, c in enumerate(sid1):
                sid_var[0, i] = c
            for i, c in enumerate(sid2):
                sid_var[1, i] = c

            # Storm names
            name1 = list("HELENE       ")
            name2 = list("TEST         ")
            for i, c in enumerate(name1[:13]):
                name_var[0, i] = c
            for i, c in enumerate(name2[:13]):
                name_var[1, i] = c

            # Basin (per time point)
            basin1 = list("NA  ")
            for t in range(3):
                for i, c in enumerate(basin1):
                    basin_var[0, t, i] = c
                    basin_var[1, t, i] = c

            season_var[:] = [2024, 2024]
            wmo_wind_var[0, :] = [100.0, 110.0, 120.0]  # knots
            wmo_wind_var[1, :] = [50.0, np.nan, np.nan]

        # Now test the data source
        ds = IBTrACS(region="NA", cache=True)
        ds._tmp_cache_hash = "test"

        # Create a fake task
        from earth2studio.data.ibtracs import IBTrACSAsyncTask

        task = IBTrACSAsyncTask(
            region="MOCK",
            remote_url="https://example.com/mock.nc",
            local_path=str(nc_path),
        )

        # Call _compile_dataframe
        time_list = [datetime(2024, 9, 1)]
        variable_list = ["tcwnd"]
        schema = ds.resolve_fields(None)

        df = ds._compile_dataframe([task], time_list, variable_list, schema)

        assert not df.empty
        assert "tcwnd" in df["variable"].values
        # Wind should be converted from knots to m/s
        # 100 knots * 0.514444 = 51.4444 m/s
        tcwnd_obs = df[df["variable"] == "tcwnd"]["observation"].values
        assert len(tcwnd_obs) > 0
        # All observations should be reasonable wind speeds in m/s
        assert all(tcwnd_obs > 0) and all(tcwnd_obs < 100)
