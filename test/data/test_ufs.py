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

from earth2studio.data import UFSObsConv, UFSObsSat


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [datetime(year=2024, month=8, day=1, hour=12)],
    ],
)
@pytest.mark.parametrize(
    "variable, tol",
    [
        (["t2m"], timedelta(hours=1)),
        (["u10m", "v10m"], timedelta(hours=2)),
    ],
)
def test_ufsobsconv_fetch(time, variable, tol):
    ds = UFSObsConv(tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

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
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["t2m"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobsconv_cache(time, variable, cache):
    ds = UFSObsConv(
        tolerance=timedelta(hours=1),
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobsconv_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsConv(tolerance=tol, cache=False, verbose=False)

    df_full = ds(time, ["t2m"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t2m"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobsconv_exceptions():
    ds = UFSObsConv(
        tolerance=timedelta(hours=1),
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["t2m"],
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
        ds(datetime(2024, 1, 1), ["t2m"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["t2m"], fields=wrong_type_schema)


def test_ufsobsconv_tolerance_conversion():
    ds_timedelta = UFSObsConv(tolerance=timedelta(hours=1), cache=False, verbose=False)
    assert ds_timedelta.tolerance == timedelta(hours=1)

    ds_numpy = UFSObsConv(tolerance=np.timedelta64(1, "h"), cache=False, verbose=False)
    assert ds_numpy.tolerance == timedelta(hours=1)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [datetime(year=2024, month=1, day=1, hour=6)],
    ],
)
@pytest.mark.parametrize(
    "variable, satellites, tol",
    [
        (["atms"], ["npp"], timedelta(hours=1)),
        (["mhs"], ["metop-a", "metop-b"], timedelta(hours=2)),
    ],
)
def test_ufsobssat_fetch(time, variable, satellites, tol):
    ds = UFSObsSat(tolerance=tol, satellites=satellites, cache=False, verbose=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert "satellite" in df.columns

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    if not df.empty:
        time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
        for t in time:
            df_times = df["time"]
            min_time = t - tol
            max_time = t + tol
            time_union["time"] = time_union["time"] | (
                df_times.ge(min_time) & df_times.le(max_time)
            )
        assert time_union["time"].all()
        assert set(df["satellite"].unique()).issubset(set(satellites))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["atms"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobssat_cache(time, variable, cache):
    ds = UFSObsSat(
        tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobssat_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsSat(tolerance=tol, satellites=["npp"], cache=False, verbose=False)

    df_full = ds(time, ["atms"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "satellite", "observation", "variable"]
    df_subset = ds(time, ["atms"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobssat_exceptions():
    ds = UFSObsSat(
        tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["atms"],
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
        ds(datetime(2024, 1, 1), ["atms"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["atms"], fields=wrong_type_schema)

    # Test satellites
    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["invalid_sat"])

    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["npp", "invalid_sat"])

    ds = UFSObsSat(cache=False, verbose=False)
    assert set(ds.satellites) == ds.VALID_SATELLITES

    ds = UFSObsSat(satellites=["npp", "n20"], cache=False, verbose=False)
    assert ds.satellites == ["npp", "n20"]


def test_gsi_cache_path():
    ds = UFSObsConv(cache=True, verbose=False)
    path1 = ds.cache_path("s3://bucket/file.nc4")
    path2 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100)
    path3 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100, byte_length=200)

    assert path1 != path2
    assert path2 != path3
    assert all(p.startswith(ds.cache) for p in [path1, path2, path3])
