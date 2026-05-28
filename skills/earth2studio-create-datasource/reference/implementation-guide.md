# Implementation Guide

> Reference for Steps 3–8 of the data source creation workflow.
> Load this when implementing the source skeleton, async patterns, and registration.

## Table of Contents

- [Optional Dependency Imports](#optional-dependency-imports)
- [Lexicon Class](#lexicon-class)
- [E2STUDIO_VOCAB and SCHEMA Updates](#e2studio_vocab-and-schema-updates)
- [Skeleton File Structure](#skeleton-file-structure)
- [Async Utilities](#async-utilities)
- [Implementation Patterns](#implementation-patterns)
- [Registration](#registration)
- [Documentation](#documentation)
- [CHANGELOG](#changelog)

---

## Optional Dependency Imports

New dependencies in the `data` extras group are optional. Use the project's
`OptionalDependencyFailure` / `check_optional_dependencies` pattern:

```python
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import pybufkit
    from pybufkit import BUFRDecoder
except ImportError:
    OptionalDependencyFailure("data")
    pybufkit = None  # type: ignore[assignment]
    BUFRDecoder = None  # type: ignore[assignment,misc]
```

Then decorate the class:

```python
@check_optional_dependencies()
class SourceName:
    ...
```

**Rules:**
- One `try/except` per extras group — bundle all optional imports together
- Always assign `None` fallbacks with `# type: ignore`
- Never use lazy imports inside method bodies
- Never raise `ImportError` directly — use `OptionalDependencyFailure`

See `earth2studio/data/cams.py` or `earth2studio/data/hrrr.py` for examples.

---

## Lexicon Class

Create `earth2studio/lexicon/<source_name>.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (full Apache-2.0 header)

from collections.abc import Callable

import numpy as np

from earth2studio.lexicon.base import LexiconType


class SourceNameLexicon(metaclass=LexiconType):
    """Lexicon for <Source Name> data source.

    Note
    ----
    Variable documentation: <URL>
    """

    VOCAB = {
        "t2m": "<remote_key_for_2m_temp>",
        "u10m": "<remote_key_for_10m_u_wind>",
        "u500": "<remote_key>::<level_indicator>",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return remote key and modifier function for a variable."""
        remote_key = cls.VOCAB[val]
        return remote_key, lambda x: x
```

**Key separator `::` patterns:**

| Store format | Key pattern | Example |
|---|---|---|
| GRIB2 | `param::level_type::level` | `"u::pl::500"` |
| GFS GRIB | `param::level_desc` | `"UGRD::10 m above ground"` |
| CDS / API | `dataset::api_var::nc_key::level` | `"cams::aod550::aod550::0"` |
| Simple name | `remote_name` | `"2m_temperature"` |

For pressure-level variables with many levels, use a `build_vocab()` helper
(see `earth2studio/lexicon/ecmwf.py`).

---

## E2STUDIO_VOCAB and SCHEMA Updates

### New vocabulary entries

If variables are not in `E2STUDIO_VOCAB` (in `earth2studio/lexicon/base.py` lines 33-292):
- Surface: descriptive abbreviation (e.g., `"aod550"`)
- Pressure-level: `{short_name}{level_hPa}` (e.g., `"o3500"`)

### New schema fields (DataFrame sources only)

Check `E2STUDIO_SCHEMA` (lines 295-410) for needed columns. Core fields:
- `time`, `lat`, `lon`, `elev`, `observation`, `variable`
- Conventional: `pressure`, `type`, `class`, `source`, `station`
- Satellite: `satellite`, `scan_angle`, `channel_index`, `zenith_angle`, `solar_zenith_angle`

---

## Skeleton File Structure

### Naming conventions

- **Class name**: PascalCase or UPPER_CASE (e.g., `GFS`, `ARCO`, `CAMS_FX`)
- **Satellite sources**: `<Platform><Sensor>` (e.g., `HimawariAHI`, `MetOpAMSUA`)
- **File name**: lowercase with underscores (e.g., `gfs.py`, `cams.py`)
- **File path**: `earth2studio/data/<filename>.py`

### Canonical method ordering

Methods must appear in this exact relative order:

1. Class constants (lat/lon arrays, bucket names)
2. `SCHEMA` (DataFrame sources only)
3. `__init__`
4. `_async_init`
5. `__call__`
6. `fetch`
7. `_create_tasks`
8. `fetch_wrapper`
9. `fetch_array`
10. `_validate_time`
11. Private helpers
12. `cache` property
13. `available` classmethod
14. (DataFrame only) `resolve_fields`, `column_map`, etc.

### Async task dataclass

```python
@dataclass
class SourceNameAsyncTask:
    """Async task request for a single fetch operation."""
    data_array_indices: tuple[int, ...]
    remote_uri: str
    modifier: Callable
```

### Source type signatures

**DataSource:**
```python
def __call__(self, time, variable) -> xr.DataArray: ...
async def fetch(self, time, variable) -> xr.DataArray: ...
```

**ForecastSource:**
```python
def __call__(self, time, lead_time, variable) -> xr.DataArray: ...
async def fetch(self, time, lead_time, variable) -> xr.DataArray: ...
```

**DataFrameSource:**
```python
SCHEMA: pa.Schema = pa.schema([...])
def __init__(self, ..., time_tolerance: TimeTolerance = np.timedelta64(10, "m"), ...): ...
def __call__(self, time, variable, fields=None) -> pd.DataFrame: ...
async def fetch(self, time, variable, fields=None) -> pd.DataFrame: ...
```

**ForecastFrameSource:**
```python
SCHEMA: pa.Schema = pa.schema([...])
def __init__(self, ..., time_tolerance: TimeTolerance = np.timedelta64(10, "m"), ...): ...
def __call__(self, time, lead_time, variable, fields=None) -> pd.DataFrame: ...
async def fetch(self, time, lead_time, variable, fields=None) -> pd.DataFrame: ...
```

### DataFrame SCHEMA definition

```python
import pyarrow as pa
from earth2studio.lexicon.base import E2STUDIO_SCHEMA

class SourceName:
    SCHEMA = pa.schema([
        E2STUDIO_SCHEMA.field("time"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
    ])
```

---

## Async Utilities

All utilities are in `earth2studio/data/utils.py`. Import:

```python
from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    cancellable_to_thread,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
    prep_forecast_inputs,
)
```

### Input preprocessing

| Source Type | Function | Handles |
|---|---|---|
| DataSource / DataFrameSource | `prep_data_inputs` | `(time, variable)` |
| ForecastSource / ForecastFrameSource | `prep_forecast_inputs` | `(time, lead_time, variable)` |

These normalize single values to lists and convert timezone-aware datetimes to naive UTC.

### `_sync_async` — Run async from synchronous context

```python
xr_array = _sync_async(self.fetch, time, variable, timeout=self.async_timeout)
```

### `async_retry` — Retry with exponential backoff

```python
out = await async_retry(
    self.fetch_array, task,
    retries=self._retries, backoff=1.0,
    task_timeout=60.0,
    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
)
```

Scope `exceptions` to transient I/O errors only.

### `managed_session` — Guaranteed session cleanup

```python
async with managed_session(self.fs) as session:
    # ... fetch data here ...
```

### `gather_with_concurrency` — Bounded parallel execution

```python
await gather_with_concurrency(
    coros,
    max_workers=self._async_workers,
    task_timeout=60.0,
    desc="Fetching <Source> data",
    disable=(not self._verbose),
)
```

**NEVER** use bare `tqdm.gather(*tasks)`.

### Pure async is ALWAYS preferred

| Blocking operation | Pure async alternative |
|---|---|
| `fs.read_block()` | `await fs._cat_file(path, start=..., end=...)` |
| `zarr.open()` | Use async zarr or `AsyncCachingFileSystem` |
| `requests.get()` | `httpx.AsyncClient` |
| File I/O | `aiofiles` or cache then read |
| pandas `read_csv` | Fetch bytes async, then parse |

Only use `cancellable_to_thread` as absolute last resort (e.g., pygrib).

---

## Implementation Patterns

### Constructor (`__init__`)

```python
def __init__(
    self,
    # Source-specific params first
    cache: bool = True,
    verbose: bool = True,
    async_timeout: int = 600,
    async_workers: int = 16,
    retries: int = 3,
):
    self._cache = cache
    self._verbose = verbose
    self._async_workers = async_workers
    self._retries = retries
    self._tmp_cache_hash: str | None = None
    self.fs = None
    self.async_timeout = async_timeout
```

For DataFrame sources, add `time_tolerance: TimeTolerance = np.timedelta64(10, "m")`.

### Async init (`_async_init`)

```python
async def _async_init(self) -> None:
    self.fs = s3fs.S3FileSystem(
        anon=True, client_kwargs={},
        asynchronous=True, skip_instance_cache=True,
    )
```

### Synchronous `__call__`

```python
def __call__(self, time, variable) -> xr.DataArray:
    try:
        xr_array = _sync_async(self.fetch, time, variable, timeout=self.async_timeout)
    finally:
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)
    return xr_array
```

### Async `fetch`

```python
async def fetch(self, time, variable) -> xr.DataArray:
    if self.fs is None:
        await self._async_init()
    time, variable = prep_data_inputs(time, variable)
    pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
    self._validate_time(time)

    async with managed_session(self.fs) as session:
        xr_array = xr.DataArray(...)
        tasks = await self._create_tasks(time, variable)
        coros = [self.fetch_wrapper(task, xr_array=xr_array) for task in tasks]
        await gather_with_concurrency(
            coros, max_workers=self._async_workers,
            task_timeout=60.0, desc="Fetching data", disable=(not self._verbose),
        )
    return xr_array
```

### Task creation and execution

```python
async def _create_tasks(self, time, variable) -> list[SourceNameAsyncTask]:
    tasks = []
    for i, t in enumerate(time):
        for j, v in enumerate(variable):
            try:
                remote_key, modifier = SourceNameLexicon[v]
            except KeyError:
                logger.warning(f"variable id {v} not found in lexicon, skipping")
                continue
            tasks.append(SourceNameAsyncTask(
                data_array_indices=(i, j),
                remote_uri=self._build_uri(t, remote_key),
                modifier=modifier,
            ))
    return tasks

async def fetch_wrapper(self, task, xr_array) -> None:
    out = await async_retry(
        self.fetch_array, task,
        retries=self._retries, backoff=1.0, task_timeout=60.0,
        exceptions=(OSError, IOError, TimeoutError, ConnectionError),
    )
    i, j = task.data_array_indices
    xr_array[i, j] = out

async def fetch_array(self, task) -> np.ndarray:
    data = await self.fs._cat_file(task.remote_uri, start=offset, end=offset + length)
    return task.modifier(parsed_data)
```

### Time validation

```python
@classmethod
def _validate_time(cls, times: list[datetime]) -> None:
    for time in times:
        if not (time - datetime(1900, 1, 1)).total_seconds() % INTERVAL == 0:
            raise ValueError(f"Requested {time} needs {INTERVAL//3600}h interval")
        if time < MIN_DATE or time > MAX_DATE:
            raise ValueError(f"Requested {time} outside [{MIN_DATE}, {MAX_DATE}]")
```

### Cache property

```python
@property
def cache(self) -> str:
    cache_location = os.path.join(datasource_cache_root(), "source_name")
    if not self._cache:
        if self._tmp_cache_hash is None:
            self._tmp_cache_hash = uuid.uuid4().hex[:8]
        cache_location = os.path.join(cache_location, f"tmp_{self._tmp_cache_hash}")
    return cache_location
```

### Available classmethod

```python
@classmethod
def available(cls, time: datetime | np.datetime64) -> bool:
    if isinstance(time, np.datetime64):
        time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
    try:
        cls._validate_time([time])
    except ValueError:
        return False
    return True
```

---

## Registration

### `earth2studio/data/__init__.py`

Add import in **alphabetical order**: `from .<filename> import SourceName`

### `earth2studio/lexicon/__init__.py`

Add import in **alphabetical order**: `from .<filename> import SourceNameLexicon`

---

## Documentation

### RST files

| Source type | RST file |
|---|---|
| DataSource | `docs/modules/datasources_analysis.rst` |
| ForecastSource | `docs/modules/datasources_forecast.rst` |
| DataFrameSource / ForecastFrameSource | `docs/modules/datasources_dataframe.rst` |

Add within `.. autosummary::` directive in alphabetical order.

### Class docstring requirements

Must include:
- Brief + extended description
- All Parameters (NumPy-style)
- `Warning` section about remote data/download size
- `Note` section with reference URLs
- `Badges` section (last): region, dataclass, product badges

### Badge options

- **Region**: `region:global`, `region:na`, `region:eu`, `region:as`, etc.
- **Data class**: `dataclass:analysis`, `dataclass:reanalysis`, `dataclass:observation`
- **Product**: `product:wind`, `product:temp`, `product:atmos`, `product:sat`, etc.

---

## CHANGELOG

Add under current unreleased version:

```markdown
### Added

- Added <SourceName> <source_type> for <brief description> (`ClassName`)

### Dependencies

- Added `<package>` to `data` optional dependency group (if applicable)
```

One line per source. Do NOT add separate lexicon entries.
