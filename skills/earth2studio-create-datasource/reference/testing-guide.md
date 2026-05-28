# Testing Guide

> Reference for Step 12 — writing pytest unit tests for data sources.
> Load this when creating the test file.

## Table of Contents

- [Canonical Test Names](#canonical-test-names)
- [DataSource Test Template](#datasource-test-template)
- [ForecastSource Test Additions](#forecastsource-test-additions)
- [DataFrameSource Test Additions](#dataframesource-test-additions)
- [Mock Test (Required)](#mock-test-required)
- [Test Guidelines](#test-guidelines)
- [Pytest Timeout Considerations](#pytest-timeout-considerations)

---

## Canonical Test Names

Every test name must use the prefix `test_<source_name>_` (lowercase, underscores).
Use these canonical suffixes, in this order:

| Section | Test name suffix | Purpose |
|---|---|---|
| Network (slow, xfail) | `*_fetch` | Real-network smoke test |
| Network (slow, xfail) | `*_cache` | `cache=True / False` toggle |
| Mock end-to-end | `*_call_mock` | Full `__call__` happy path, full schema |
| Unit — errors | `*_exceptions` | Single error/exception test, multiple `pytest.raises` |
| Unit — availability | `*_available` | `available()` classmethod |
| Unit — time check | `*_validate_time` | Internal time-window check (only if class has one) |

**Rules:**
1. Match the order above following existing data sources
2. Do NOT test trivial helpers in isolation
3. No docstrings on test functions

---

## DataSource Test Template

```python
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
import pytest

from earth2studio.data import SourceName


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=YYYY, month=M, day=D),
        [
            datetime(year=YYYY, month=M, day=D, hour=H),
            datetime(year=YYYY2, month=M2, day=D2, hour=H2),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["msl", "tp"]])
def test_source_fetch(time, variable):
    ds = SourceName(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == EXPECTED_LAT
    assert shape[3] == EXPECTED_LON
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("YYYY-MM-DDT00:00")])],
)
@pytest.mark.parametrize("variable", [["t2m", "msl"]])
@pytest.mark.parametrize("cache", [True, False])
def test_source_cache(time, variable, cache):
    ds = SourceName(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert not np.isnan(data.values).any()
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Reload from cache
    data = ds(time, variable[0])
    assert data.shape[1] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=OUT_OF_RANGE_YEAR, month=1, day=1),
        datetime(year=INVALID_HOUR_YEAR, month=1, day=1, hour=13),
    ],
)
@pytest.mark.parametrize("variable", ["nonexistent_var"])
def test_source_available(time, variable):
    assert not SourceName.available(time)
    with pytest.raises(ValueError):
        ds = SourceName()
        ds(time, variable)
```

---

## ForecastSource Test Additions

```python
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time,lead_time",
    [
        (datetime(...), timedelta(hours=6)),
        (datetime(...), [timedelta(hours=6), timedelta(hours=12)]),
        (
            np.array([np.datetime64("2024-01-01T00:00"),
                       np.datetime64("2024-02-01T00:00")]),
            np.array([np.timedelta64(0, "h")]),
        ),
    ],
)
def test_source_fx_fetch(time, lead_time):
    variable = "t2m"
    ds = SourceName_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]
    if isinstance(time, datetime):
        time = [time]

    # shape: [time, lead_time, variable, lat, lon]
    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert not np.isnan(data.values).any()
```

---

## DataFrameSource Test Additions

```python
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=YYYY, month=M, day=D),
        [datetime(year=YYYY, month=M, day=D, hour=H)],
    ],
)
@pytest.mark.parametrize(
    "source_params, variable, tol",
    [
        (["param1"], ["t2m"], timedelta(hours=1)),
        (["param2"], ["u10m", "v10m", "t2m"], timedelta(hours=4)),
    ],
)
def test_source_fetch(source_params, time, variable, tol):
    ds = SourceName(params=source_params, time_tolerance=tol, cache=False)
    df = ds(time, variable)

    # Schema validation
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))


def test_source_schema_fields():
    ds = SourceName(...)
    time = np.array(["YYYY-MM-DDT12:00:00"], dtype=np.datetime64)

    # Test with default schema (all fields)
    df_full = ds(time, ["t2m"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    # Test with subset of fields
    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t2m"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_source_exceptions():
    ds = SourceName(...)
    with pytest.raises(KeyError):
        ds(np.array([np.datetime64("2025-01-01T12:00:00")]), ["invalid"])
```

---

## Mock Test (Required)

Every data source **must** have at least one mock test that exercises the full
`__call__` path without network access, timeouts, or `xfail`.

```python
from unittest.mock import patch

def test_source_call_mock(tmp_path):
    # Create synthetic data in tmp_path
    # ...write minimal valid data...

    with patch(
        "earth2studio.data.<module>.<ClassName>._download_products"
    ) as mock_dl:
        mock_dl.return_value = [str(tmp_path / "mock_data.file")]
        ds = SourceName(cache=False)
        df = ds(datetime(2025, 1, 1), ["var1"])

        assert list(df.columns) == ds.SCHEMA.names
        assert not df.empty
        mock_dl.assert_called_once()
```

For DataSource/ForecastSource, write a synthetic NetCDF, GRIB, or Zarr file
and mock `fetch_array` or the filesystem layer.

---

## Test Guidelines

- **Keep tests low and concise with maximal coverage.** Each test should
  justify its existence.
- No docstrings on test functions
- Use **strategic parameterization** — minimize combinations, maximize coverage
- Markers: `@pytest.mark.slow` + `@pytest.mark.xfail` + `@pytest.mark.timeout(30)`
  for all network tests
- Test both single and list inputs for time/variable
- Test `cache=True` and `cache=False`
- Test availability / error handling
- At least one mock test per source (no network, no timeout, no xfail)
- **Target 90%+ line coverage** with `--slow`. Use `--cov-report=term-missing`.
- Run via: `make pytest TOX_ENV=test-data` or
  `pytest test/data/test_<filename>.py -v`

---

## Pytest Timeout Considerations

Tests using `asyncio.to_thread` (via `cancellable_to_thread`) can hang because
Python threads cannot be forcibly cancelled.

When pytest timeout fires:
1. `asyncio.wait_for` raises `TimeoutError`
2. The coroutine is abandoned
3. But the underlying thread keeps running
4. Test process may hang

**Mitigations:**
- Avoid `asyncio.to_thread` entirely — use pure async I/O
- Set test timeout **higher** than task timeouts (e.g., `timeout(60)` with 30s tasks)
- Use `@pytest.mark.xfail` for tests that might hang
- Always include at least one mock test for CI validation
