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

from earth2studio.data import GHCN
from earth2studio.lexicon.ghcn import GHCN_ELEMENT_MAP, GHCNLexicon


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
    ds = GHCN(stations=stations, time_tolerance=tol, cache=False)
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
    ds = GHCN(
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

    ds = GHCN(stations=[station], time_tolerance=timedelta(days=0))

    # Test with default schema (all fields)
    df_full = ds(time, ["t2m_max"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    # Test with subset of fields
    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t2m_max"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ghcn_exceptions():
    # Throw key error for invalid variable
    ds = GHCN(
        stations=["USW00013722"],
        time_tolerance=timedelta(days=0),
        cache=False,
        verbose=False,
    )
    with pytest.raises(KeyError):
        ds(np.array([np.datetime64("2023-01-01")]), ["invalid"])

    # For an invalid station should return empty
    ds = GHCN(stations=["INVALID00000"], cache=False, verbose=False)
    df = ds(np.array(["2023-01-01"], dtype=np.datetime64), ["t2m_max"])
    assert df.empty

    # Time that there is no data for should return empty
    ds = GHCN(stations=["USW00013722"], cache=False, verbose=False)
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

    stations = GHCN.get_stations_bbox((lat_min, lon_min, lat_max, lon_max))
    assert len(stations) > 0

    # Load station metadata to verify coordinates
    df = GHCN.get_station_metadata()
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


# -- Offline unit tests (no network needed) --


def test_ghcn_schema_structure():
    assert "time" in GHCN.SCHEMA.names
    assert "lat" in GHCN.SCHEMA.names
    assert "lon" in GHCN.SCHEMA.names
    assert "station" in GHCN.SCHEMA.names
    assert "observation" in GHCN.SCHEMA.names
    assert "variable" in GHCN.SCHEMA.names
    assert GHCN.SCHEMA.field("time").type == pa.timestamp("ns")
    assert GHCN.SCHEMA.field("observation").type == pa.float32()


def test_ghcn_column_map():
    cmap = GHCN.column_map()
    assert cmap["DATE"] == "time"
    assert cmap["LAT"] == "lat"
    assert cmap["LON"] == "lon"
    assert cmap["ID"] == "station"
    assert cmap["ELEV"] == "elev"


def test_ghcn_resolve_fields_none():
    schema = GHCN.resolve_fields(None)
    assert schema == GHCN.SCHEMA


def test_ghcn_resolve_fields_list():
    schema = GHCN.resolve_fields(["time", "observation", "variable"])
    assert len(schema) == 3
    assert schema.names == ["time", "observation", "variable"]


def test_ghcn_resolve_fields_string():
    schema = GHCN.resolve_fields("observation")
    assert len(schema) == 1
    assert schema.names == ["observation"]


class TestGHCNLexicon:
    @pytest.mark.parametrize(
        "var, element",
        [
            ("t2m_max", "TMAX"),
            ("t2m_min", "TMIN"),
            ("tp", "PRCP"),
            ("sd", "SNWD"),
            ("sde", "SNOW"),
        ],
    )
    def test_element_map(self, var, element):
        assert GHCN_ELEMENT_MAP[var] == element

    def test_lexicon_keys(self):
        for var in GHCN_ELEMENT_MAP:
            element, mod = GHCNLexicon[var]
            assert element == GHCN_ELEMENT_MAP[var]
            assert callable(mod)

    @pytest.mark.parametrize(
        "var, raw, expected",
        [
            ("t2m_max", np.array([200.0]), np.array([293.15])),  # 20.0 C -> K
            ("t2m_min", np.array([-100.0]), np.array([263.15])),  # -10.0 C -> K
            ("tp", np.array([100.0]), np.array([0.01])),  # 10.0 mm -> 0.01 m
            ("sd", np.array([500.0]), np.array([0.5])),  # 500 mm -> 0.5 m
            ("sde", np.array([250.0]), np.array([0.25])),  # 250 mm -> 0.25 m
        ],
    )
    def test_unit_conversions(self, var, raw, expected):
        _, mod = GHCNLexicon[var]
        np.testing.assert_allclose(mod(raw), expected, atol=1e-6)

    def test_invalid_variable(self):
        with pytest.raises(KeyError):
            GHCNLexicon["nonexistent"]
