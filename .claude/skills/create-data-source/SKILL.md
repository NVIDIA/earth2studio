---
name: create-data-source
description: Create a new Earth2Studio data source wrapper (DataSource, ForecastSource, DataFrameSource, or ForecastFrameSource) from a remote data store. Use this skill whenever the user mentions adding a data source, weather data API, observation feed, forecast archive, or any new remote/cloud data integration to Earth2Studio. Also trigger when the user asks about implementing DataSource, ForecastSource, DataFrameSource, or ForecastFrameSource protocols, connecting to S3/GCS/Azure/FTP stores, or wrapping a new weather/climate dataset.
argument-hint: URL or description of remote data store (optional — will be asked if not provided)
---

# Create Data Source Wrapper

Create a new Earth2Studio data source by following every step below **in order**.
Each confirmation gate marked by:

```markdown
### **[CONFIRM — <Title>]**
```

requires **explicit user approval** before proceeding.

> **One source type per invocation.** This skill covers DataSource,
> ForecastSource, DataFrameSource, and ForecastFrameSource but creates
> exactly one per run. Invoke the skill again for a companion type
> (e.g., `GFS` then `GFS_FX`).

---

## Step 0 — Obtain Remote Data Store Reference

If `$ARGUMENTS` is provided, use it as the reference.

- URL → use WebFetch to retrieve documentation / API reference.
- Local file path → read directly.

If `$ARGUMENTS` is empty, ask the user:

> Please provide a URL, API documentation link, or description of the
> remote data store you want to integrate. This will be used to
> understand the storage format (Zarr, GRIB, NetCDF, CSV, Parquet,
> etc.), access pattern (S3, GCS, FTP, HTTP API), variable inventory,
> temporal resolution, and spatial grid.
>
> If you have a link to the data documentation or variable listing,
> please share that too — it will help map variables to Earth2Studio
> conventions.

Store the reference content for subsequent steps.

---

## Step 1 — Determine Source Type

Based on the reference, recommend one of the four protocol types and
explain your reasoning to the user.

| Protocol | Returns | Has `lead_time`? | Typical use |
|---|---|---|---|
| **DataSource** | `xr.DataArray` `[time, variable, ...]` | No | Gridded analysis / reanalysis |
| **ForecastSource** | `xr.DataArray` `[time, lead_time, ...]` | Yes | Gridded forecast data |
| **DataFrameSource** | `pd.DataFrame` (rows = obs) | No | Sparse / station / satellite obs |
| **ForecastFrameSource** | `pd.DataFrame` (rows = obs) | Yes | Sparse forecast obs |

Key decision factors:

- **Gridded** (regular lat/lon) vs **sparse** (point observations) →
  DataArray vs DataFrame.
- **Analysis / reanalysis / observations at a single time** vs
  **forecast with lead times** → Source vs ForecastSource.

Ask the user if they know what variables are available in the store
and whether the data is gridded or point/station-based. This helps
confirm the source type recommendation.

### **[CONFIRM — Source Type]**

Present to the user:

1. The recommended source type with justification
2. The protocol signature it must satisfy (from `earth2studio/data/base.py`)
3. Ask for confirmation or correction

---

## Step 2 — Examine Remote Store & Propose Dependencies

### 2a. Analyze the remote store

Identify:

- **Storage backend** (S3, GCS, Azure Blob, FTP, HTTP REST API, etc.)
- **File format** (GRIB2, NetCDF4, Zarr, CSV/Parquet, HDF5)
- **Authentication** (anonymous, API key, OAuth)
- **Access pattern** (full file download, byte-range, streaming,
  REST query)
- **Temporal resolution** (hourly, 6-hourly, daily, monthly)
- **Spatial resolution** and grid type
- **Variable inventory** — list the variables available and how they
  are named in the remote store

### 2b. Prefer fsspec-compatible filesystems

For remote data access, **always prefer fsspec-compatible filesystem
implementations** over dedicated client libraries when possible:

| Backend | Preferred package | Avoid |
|---|---|---|
| AWS S3 | `s3fs` (already in core deps) | `boto3` directly |
| Google Cloud Storage | `gcsfs` (already in core deps) | `google-cloud-storage` directly |
| Azure Blob | `adlfs` | `azure-storage-blob` directly |
| HTTP/HTTPS | `fsspec` (already in core deps) | `requests` directly |
| FTP | `fsspec.implementations.ftp` | `ftplib` directly |
| HuggingFace Hub | `huggingface_hub` (already in core deps) | custom download scripts |

These are preferred because they integrate with async patterns used
throughout Earth2Studio, support caching, and provide a consistent
interface. Many are already in the project's core dependencies.

Only fall back to dedicated libraries (e.g., `cdsapi`, `ecmwf-opendata`)
when the data store requires a proprietary API that cannot be accessed
via fsspec.

### 2c. Propose dependencies

Check if the required packages already exist in `pyproject.toml`
under `[project.dependencies]` (core) or
`[project.optional-dependencies] data` group. Only propose
**new** packages that are not already present.

Current core deps include: `s3fs`, `gcsfs`, `fsspec`, `zarr`,
`netCDF4`, `h5py`, `h5netcdf`, `pygrib`, `huggingface-hub`,
`pandas`, `pyarrow`, `nest_asyncio`.

Current `data` group includes: `cdsapi`, `eccodes`,
`ecmwf-opendata`, `planetary-computer`, `pystac-client`,
`rasterio`, `rioxarray`.

Read `pyproject.toml` to confirm the current state.

If new packages are needed, propose additions to the `data` extra
group:

```toml
data = [
    # ... existing entries ...
    "new-package>=1.0",
]
```

### **[CONFIRM — Dependencies & Access Pattern]**

Present:

1. The storage backend and access pattern
2. The fsspec filesystem to use (or justification for a dedicated library)
3. Any new packages to add (or "none needed — all deps already present")
4. How authentication works (env vars, config file, anonymous)
5. Ask if the analysis looks correct

---

## Step 3 — Add Dependencies to pyproject.toml

After confirmation, if new packages are needed:

1. Use `uv add --extra data <package>` for each new dependency
2. Verify with `uv lock` that the lockfile resolves cleanly
3. Also add the new package to the `all` aggregate group if not
   automatically included

If no new dependencies are needed, skip this step.

### 3b. Register optional dependency imports

New dependencies added to the `data` extras group are **optional** —
they are not installed by default. The data source module **must not**
use bare `import` or lazy `try/except ImportError: raise ImportError`
for these packages. Instead, use the project's
`OptionalDependencyFailure` / `check_optional_dependencies` pattern
so that users get a clear, actionable error message pointing them to
the correct install command.

**At the top of the data source file** (module level), wrap all
optional imports in a single `try/except` block:

```python
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import eumdac
    from satpy import Scene
except ImportError:
    OptionalDependencyFailure("data")
    eumdac = None  # type: ignore[assignment]
    Scene = None  # type: ignore[assignment,misc]
```

Then decorate the data source class:

```python
@check_optional_dependencies()
class SourceName:
    ...
```

**Key rules:**

- **One `try/except` per extras group** — bundle all optional imports
  from the same group (e.g., `data`) into a single block.
- **Always assign `None` fallbacks** with `# type: ignore` so the
  module can still be imported for type checking and IDE navigation.
- **Never use lazy imports** (`try: import X` inside a method body).
  The `@check_optional_dependencies()` decorator on the class
  handles the error at instantiation time.
- **Never raise `ImportError` directly** — let
  `OptionalDependencyFailure` produce the standardized rich error
  with install instructions.

See `earth2studio/data/cams.py` or `earth2studio/data/hrrr.py` for
canonical examples.

---

## Step 4 — Create Lexicon Class

Every data source needs a lexicon that maps Earth2Studio variable
names (from `E2STUDIO_VOCAB`) to the remote store's native variable
identifiers.

### 4a. Gather variable information

Ask the user:

> Do you have a link to the variable documentation or a list of
> variables available in this data store? This helps ensure accurate
> mapping to Earth2Studio conventions.

If the user provides a URL, fetch it. If not, inspect the remote store
directly (e.g., list keys in a Zarr group, read GRIB index files, or
check API documentation).

### 4b. Understand lexicon structure

Read `earth2studio/lexicon/base.py` for the base vocabulary and
`LexiconType` metaclass. A lexicon class:

- Uses `metaclass=LexiconType` (enables `MyLexicon["t2m"]` syntax)
- Defines a `VOCAB: dict[str, str]` class attribute mapping
  `e2s_name → remote_key`
- Implements `@classmethod get_item(cls, val)` returning
  `tuple[str, Callable]` — the remote key and a modifier function

The `::` separator is the standard for structured keys. Examples:

| Store format | Key pattern | Example |
|---|---|---|
| GRIB2 | `param::level_type::level` | `"u::pl::500"` |
| GFS GRIB | `param::level_desc` | `"UGRD::10 m above ground"` |
| CDS / API | `dataset::api_var::nc_key::level` | `"cams::aod550::aod550::0"` |
| Simple name | `remote_name` | `"2m_temperature"` |

### 4c. Map variables

Compare the remote store's variable inventory against
`E2STUDIO_VOCAB` (282 entries in `earth2studio/lexicon/base.py`
lines 33-292). For each remote variable:

- Find the matching E2S name (e.g., remote `"2m_temperature"` →
  E2S `"t2m"`)
- If a modifier is needed (unit conversion, accumulation, etc.),
  define the lambda/function
- If a variable has **no** match in `E2STUDIO_VOCAB`, flag it — see
  Step 5

Surface variables: `t2m`, `d2m`, `u10m`, `v10m`, `sp`, `msl`,
`tcwv`, `tp`, `skt`, `sst`, etc.

Pressure-level variables: `{var}{level}` where var ∈
{u, v, w, z, t, r, q} and level ∈ {50, 100, 150, 200, 250, 300,
400, 500, 600, 700, 850, 925, 1000}.

### 4d. Write the lexicon file

Create `earth2studio/lexicon/<source_name>.py`:

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

from collections.abc import Callable

import numpy as np

from earth2studio.lexicon.base import LexiconType


class SourceNameLexicon(metaclass=LexiconType):
    """Lexicon for <Source Name> data source.

    <Brief description of the remote store and variable naming
    convention used.>

    Note
    ----
    Variable documentation: <URL to variable listing>
    """

    VOCAB = {
        # Surface variables
        "t2m": "<remote_key_for_2m_temp>",
        "u10m": "<remote_key_for_10m_u_wind>",
        # ... more mappings ...
        # Pressure-level variables
        "u500": "<remote_key>::<level_indicator>",
        # ... more mappings ...
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return remote key and modifier function for a variable.

        Parameters
        ----------
        val : str
            Variable name (E2S convention)

        Returns
        -------
        tuple[str, Callable]
            Remote key string and modifier function
        """
        remote_key = cls.VOCAB[val]
        # Define modifier if unit conversion is needed
        # e.g., geopotential: modifier = lambda x: x * 9.81
        # For identity: modifier = lambda x: x
        return remote_key, lambda x: x
```

If the remote store uses **pressure-level variables** with many
levels, consider a `build_vocab()` helper (see
`earth2studio/lexicon/ecmwf.py` for the pattern) rather than
manually listing all 91+ entries.

### **[CONFIRM — Lexicon & Variable Mapping]**

Present to the user:

1. The proposed lexicon class name
2. The key format / separator pattern
3. The full variable mapping table (E2S name → remote key)
4. Any variables that need modifier functions (unit conversions)
5. The **reference URL** used for variable documentation
6. Any variables not yet in `E2STUDIO_VOCAB` (to be addressed in Step 5)
7. Ask if the mapping looks correct

---

## Step 5 — Update E2STUDIO_VOCAB and E2STUDIO_SCHEMA (if needed)

### 5a. New vocabulary entries

If Step 4c flagged variables not in `E2STUDIO_VOCAB`:

1. Read `earth2studio/lexicon/base.py` lines 33-292
2. Propose new entries following the naming conventions:
   - Surface: descriptive abbreviation (e.g., `"aod550"` for aerosol
     optical depth at 550nm)
   - Pressure-level: `{short_name}{level_hPa}` (e.g., `"o3500"`)
3. Add entries in the appropriate category section of `E2STUDIO_VOCAB`

### 5b. New schema fields (DataFrameSource / ForecastFrameSource only)

If the source is a DataFrame source, check `E2STUDIO_SCHEMA`
(lines 295-410 in `earth2studio/lexicon/base.py`) for any new
schema fields needed. The schema defines the PyArrow column types
for DataFrame sources.

Core schema fields already defined:

- `time` (timestamp), `lat` (float32), `lon` (float32),
  `elev` (float32), `observation` (float32), `variable` (string)
- Conventional: `pressure`, `type`, `class`, `source`, `station`
- Satellite: `satellite`, `scan_angle`, `channel_index`,
  `zenith_angle`, `solar_zenith_angle`

If the new source needs columns not in `E2STUDIO_SCHEMA`, propose
additions.

### **[CONFIRM — Vocabulary & Schema Updates]**

Present:

1. Proposed new `E2STUDIO_VOCAB` entries with a **reference URL** for
   where the variable definition comes from (e.g., ECMWF parameter
   database, CF conventions, data provider docs)
2. Proposed new `E2STUDIO_SCHEMA` fields (if DataFrame source)
3. Ask for approval

If no new vocab or schema entries are needed, inform the user and
skip this gate.

---

## Step 6 — Create Skeleton Data Source File

### 6a. Determine names

- **Class name**: PascalCase or UPPER_CASE matching existing patterns
  (e.g., `GFS`, `ARCO`, `ISD`, `CAMS_FX`, `GEFS_FX_721x1440`)
- **File name**: lowercase with underscores (e.g., `gfs.py`,
  `cams.py`, `isd.py`)
- **File path**: `earth2studio/data/<filename>.py`

### 6b. Write skeleton with canonical method ordering

Every `.py` file in `earth2studio/` **must** start with the SPDX
Apache-2.0 license header:

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
```

**Canonical method ordering** — methods in the class must appear in
this exact order. Not every source needs every method, but the
**relative ordering must be preserved**:

1. **Class constants** (lat/lon arrays, bucket names, etc.)
2. **`SCHEMA`** (DataFrame sources only — `pa.Schema` class attribute)
3. **`__init__`** — constructor
4. **`_async_init`** — async filesystem initialization (if async)
5. **`__call__`** — synchronous entry point
6. **`fetch`** — async implementation
7. **`_create_tasks`** — build list of async task dataclass instances
8. **`fetch_wrapper`** — unpack task and call `fetch_array`
9. **`fetch_array`** — fetch a single datum from remote store
10. **`_validate_time`** — time input validation (classmethod)
11. **Private helpers** — `_fetch_index`, `_fetch_remote_file`,
    `_build_uri`, etc.
12. **`cache`** — `@property` returning cache path
13. **`available`** — `@classmethod` checking date availability
14. **(DataFrame only)** `resolve_fields`, `column_map`, and other
    classmethods/staticmethods

### 6c. Async task dataclass pattern

For async data sources, define a **task dataclass** above the class
that bundles all information needed for a single fetch operation.
This enables batched parallel execution via `tqdm.gather`:

```python
@dataclass
class SourceNameAsyncTask:
    """Async task request for a single fetch operation."""

    data_array_indices: tuple[int, ...]  # Where to place result in output array
    remote_uri: str                       # Full URI to fetch from
    # ... any additional fields needed (byte offsets, variable keys, etc.)
    modifier: Callable                    # Lexicon modifier function
```

The `_create_tasks` method builds a `list[SourceNameAsyncTask]`.
The `fetch_wrapper` unpacks each task and calls `fetch_array`.
The `fetch` method gathers all tasks with `tqdm.gather`.

This is the standard pattern used across GFS, ARCO, ISD, and other
async sources. See `earth2studio/data/gfs.py` for the canonical
reference (`GFSAsyncTask`).

### 6d. Source type-specific signatures

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
SCHEMA: pa.Schema = pa.schema([...])  # Class attribute
def __call__(self, time, variable, fields=None) -> pd.DataFrame: ...
async def fetch(self, time, variable, fields=None) -> pd.DataFrame: ...
```

**ForecastFrameSource:**

```python
SCHEMA: pa.Schema = pa.schema([...])  # Class attribute
def __call__(self, time, lead_time, variable, fields=None) -> pd.DataFrame: ...
async def fetch(self, time, lead_time, variable, fields=None) -> pd.DataFrame: ...
```

### 6e. DataFrame SCHEMA definition

For DataFrameSource / ForecastFrameSource, the SCHEMA class attribute
defines the output column structure. Build it from `E2STUDIO_SCHEMA`
fields (in `earth2studio/lexicon/base.py` lines 295-410):

```python
import pyarrow as pa
from earth2studio.lexicon.base import E2STUDIO_SCHEMA

class SourceName:
    SCHEMA = pa.schema([
        E2STUDIO_SCHEMA.field("time"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        E2STUDIO_SCHEMA.field("elev"),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
        E2STUDIO_SCHEMA.field("station"),     # if station-based
        # ... select fields relevant to this source
    ])
```

### **[CONFIRM — Skeleton]**

Present:

1. The proposed class name and file path
2. The skeleton code with TODO placeholders
3. The async task dataclass (if applicable)
4. The SCHEMA definition (if DataFrame source)
5. Ask if the structure looks correct

---

## Step 6.5 — Use Async Utilities from utils.py

Earth2Studio provides standardized async utilities in
`earth2studio/data/utils.py`. **All new data sources must use these
utilities** instead of implementing their own patterns. Import them:

```python
from earth2studio.data.utils import (
    async_retry,
    cancellable_to_thread,
    ensure_utc,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
    prep_forecast_inputs,
)
```

### 6.5a. `ensure_utc` — Timezone normalization

All time inputs should be treated as UTC. Use `ensure_utc` to convert
timezone-aware datetimes to naive UTC:

```python
from earth2studio.data.utils import ensure_utc

# Converts tz-aware to naive UTC, passes through naive unchanged
utc_time = ensure_utc(input_time)
```

### 6.5b. `async_retry` — Retry with exponential backoff

Individual `fetch_array` calls can fail due to transient network
errors. Wrap fetch operations with `async_retry` so each task is
retried independently — if 1 of 100 tasks fails, only that one retries:

```python
from earth2studio.data.utils import async_retry

# In fetch_wrapper:
out = await async_retry(
    self.fetch_array,
    task,
    retries=self._retries,
    backoff=1.0,
    task_timeout=60.0,  # Per-attempt timeout
    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
)
```

**Important:** Scope `exceptions` to transient I/O errors only. Using
broad `Exception` would mask programming errors.

### 6.5c. `managed_session` — Guaranteed session cleanup

Use this context manager instead of bare `set_session`/`close` to
prevent session leaks on error or timeout:

```python
from earth2studio.data.utils import managed_session

async def fetch(self, time, variable):
    # Session is ALWAYS closed, even if an exception is raised
    async with await managed_session(self.fs) as session:
        # ... fetch data here ...
```

### 6.5d. `gather_with_concurrency` — Bounded parallel execution

**Never** use bare `tqdm.gather(*tasks)` which launches ALL tasks at
once — this can exhaust connections or memory. Use
`gather_with_concurrency` which limits concurrency via semaphore:

```python
from earth2studio.data.utils import gather_with_concurrency

# Execute with bounded concurrency (max 16 concurrent tasks)
await gather_with_concurrency(
    coros,
    max_workers=self._async_workers,  # Default 16
    task_timeout=60.0,  # Optional per-task timeout
    desc="Fetching <Source> data",
    disable=(not self._verbose),
)
```

The `max_workers` parameter controls semaphore concurrency, NOT thread
pool size. It limits how many coroutines run simultaneously.

### 6.5e. Pure Async is ALWAYS Preferred — Avoid `asyncio.to_thread`

> **CRITICAL:** Pure async operations are ALWAYS preferred over
> `asyncio.to_thread`. Only use `to_thread` as an absolute last resort
> when no async alternative exists.

**Why avoid `asyncio.to_thread`:**

1. **Threads cannot be cancelled** — when timeout fires, the coroutine
   is abandoned but the thread continues running
2. **Causes pytest timeout issues** — tests hang because threads keep
   running after test timeout
3. **Thread pool exhaustion** — many concurrent tasks can exhaust the
   default thread pool

**Prefer these pure async alternatives:**

| Blocking operation | Pure async alternative |
|---|---|
| `fs.read_block()` | `await fs._cat_file(path, start=..., end=...)` |
| `zarr.open()` | Use async zarr or `AsyncCachingFileSystem` |
| `requests.get()` | `httpx.AsyncClient` |
| File I/O | `aiofiles` or cache then read |
| pandas `read_csv` | Fetch bytes async, then parse (see ISD pattern) |

**Only use `cancellable_to_thread` for unavoidable sync operations:**

```python
from earth2studio.data.utils import cancellable_to_thread

# LAST RESORT — e.g., pygrib has no async alternative
data = await cancellable_to_thread(
    pygrib.open,
    grib_file,
    timeout=30.0,  # Abandon coroutine after 30s (thread continues!)
)
```

**Never do these:**

```python
# BAD — mutates global state, conflicts with other sources
loop.set_default_executor(ThreadPoolExecutor(max_workers=8))

# BAD — no timeout, thread runs forever on hang
await asyncio.to_thread(slow_func, arg)

# BAD — bare tqdm.gather launches all tasks at once
await tqdm.gather(*[fetch(t) for t in tasks])  # Could be 1000+ tasks!
```

### 6.5f. Pytest Timeout Considerations

Tests with `@pytest.mark.timeout(30)` can hang if using
`asyncio.to_thread` because:

1. Pytest timeout fires after 30 seconds
2. `asyncio.wait_for` raises `TimeoutError`
3. But the underlying thread keeps running — Python cannot kill threads
4. Test process waits for thread to complete (or gets stuck)

**Mitigations:**

- Minimize use of `asyncio.to_thread` — prefer pure async
- Use `cancellable_to_thread` with conservative timeouts
- Set test timeouts higher than task timeouts
- Accept that tests using `to_thread` may occasionally hang (use `xfail`)

---

## Step 7 — Implement the Data Source

Now fill in the skeleton. Follow these patterns from existing
implementations, using the async utilities from Step 6.5.

### 7a. Constructor (`__init__`)

All new data sources must accept these standardized parameters:

```python
def __init__(
    self,
    # Source-specific params first (e.g., product, source, stations)
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
    self._tmp_cache_hash: str | None = None  # Lazy init for temp cache

    # For async sources: attempt sync init of filesystem
    try:
        nest_asyncio.apply()
        loop = asyncio.get_running_loop()
        loop.run_until_complete(self._async_init())
    except RuntimeError:
        self.fs = None  # Will be initialized in __call__/fetch

    self.async_timeout = async_timeout
```

| Parameter | Purpose | Default |
|---|---|---|
| `cache` | Cache fetched data locally | `True` |
| `verbose` | Show progress bars and log info | `True` |
| `async_timeout` | Total timeout (seconds) for the entire fetch | `600` |
| `async_workers` | Max concurrent async tasks (semaphore bound) | `16` |
| `retries` | Per-task retry count on transient failure | `3` |

Keep source-specific parameters before these common ones. Avoid
over-exposing internal configuration. Use `loguru.logger` for all
logging, never `print()`.

### 7b. Async init (`_async_init`)

Use fsspec-compatible filesystems. Pick the right one for the backend:

```python
async def _async_init(self) -> None:
    """Async initialization of filesystem.

    Note
    ----
    Async fsspec expects initialization inside the execution loop.
    """
    # For S3:
    self.fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={},
        asynchronous=True,
        skip_instance_cache=True,  # IMPORTANT: Always use for s3fs
    )

    # For GCS:
    # self.fs = gcsfs.GCSFileSystem(asynchronous=True)

    # For HTTP:
    # self.fs = fsspec.filesystem("https", asynchronous=True)
```

### 7c. Synchronous `__call__`

```python
def __call__(
    self,
    time: datetime | list[datetime] | TimeArray,
    variable: str | list[str] | VariableArray,
) -> xr.DataArray:
    """Function to get data.

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Timestamps to return data for (UTC). Timezone-aware
        datetimes are converted to UTC automatically.
    variable : str | list[str] | VariableArray
        Variables to return. Must be in the lexicon.

    Returns
    -------
    xr.DataArray
        Weather data array with dimensions [time, variable, ...].
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if self.fs is None:
        loop.run_until_complete(self._async_init())

    try:
        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable), timeout=self.async_timeout
            )
        )
    finally:
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    return xr_array
```

### 7d. Async `fetch`

The fetch method orchestrates parallel data retrieval using the
async task dataclass pattern, `managed_session` for safe cleanup,
and `gather_with_concurrency` for bounded parallelism:

```python
async def fetch(
    self,
    time: datetime | list[datetime] | TimeArray,
    variable: str | list[str] | VariableArray,
) -> xr.DataArray:
    """Async function to get data.
    ...
    """
    if self.fs is None:
        raise ValueError(
            "File store is not initialized! If you are calling this "
            "function directly make sure the data source is initialized "
            "inside the async loop!"
        )

    time, variable = prep_data_inputs(time, variable)
    # Normalize timezone-aware inputs to naive UTC
    time = [ensure_utc(t) for t in time]

    pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
    self._validate_time(time)

    # Use managed_session for guaranteed cleanup on error/timeout
    async with await managed_session(self.fs) as session:

        # Pre-allocate output
        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), N_LAT, N_LON)
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": SOURCE_LAT,
                "lon": SOURCE_LON,
            },
        )

        # Create batched async tasks (dataclass instances)
        tasks = await self._create_tasks(time, variable)

        # Map tasks through fetch_wrapper
        coros = [
            self.fetch_wrapper(task, xr_array=xr_array)
            for task in tasks
        ]

        # Execute with bounded concurrency — NEVER use bare tqdm.gather!
        await gather_with_concurrency(
            coros,
            max_workers=self._async_workers,
            task_timeout=60.0,  # Per-task timeout
            desc="Fetching <Source> data",
            disable=(not self._verbose),
        )

    return xr_array
```

Key patterns (all new data sources MUST follow these):

- **`managed_session`** replaces bare `set_session`/`close` — ensures
  the aiohttp session is closed even if an exception is raised
  during fetching.
- **`gather_with_concurrency`** replaces `tqdm.gather` — prevents
  resource exhaustion by limiting concurrent tasks to
  `self._async_workers`.
- **`ensure_utc`** — ensures all timezone-aware inputs are converted
  to naive UTC.

For **ForecastSource**, use `prep_forecast_inputs` instead and add
`lead_time` dimension.

For **DataFrameSource**, return `pd.DataFrame` instead and follow
the ISD pattern (see `earth2studio/data/isd.py`).

### 7e. Task creation and execution

The core pattern: build a list of request dataclass instances, then
execute them all in parallel with retry on each individual task:

```python
async def _create_tasks(
    self,
    time: list[datetime],
    variable: list[str],
) -> list[SourceNameAsyncTask]:
    """Build download tasks for parallel execution.

    Parameters
    ----------
    time : list[datetime]
        Timestamps to download
    variable : list[str]
        Variables to download

    Returns
    -------
    list[SourceNameAsyncTask]
        List of async task requests
    """
    tasks: list[SourceNameAsyncTask] = []
    for i, t in enumerate(time):
        for j, v in enumerate(variable):
            try:
                remote_key, modifier = SourceNameLexicon[v]
            except KeyError:
                logger.warning(
                    f"variable id {v} not found in lexicon, skipping"
                )
                continue

            tasks.append(
                SourceNameAsyncTask(
                    data_array_indices=(i, j),
                    remote_uri=self._build_uri(t, remote_key),
                    modifier=modifier,
                )
            )
    return tasks

async def fetch_wrapper(
    self,
    task: SourceNameAsyncTask,
    xr_array: xr.DataArray,
) -> None:
    """Unpack task, fetch with retry, and place result."""
    out = await async_retry(
        self.fetch_array,
        task,
        retries=self._retries,
        backoff=1.0,
        task_timeout=60.0,  # Per-attempt timeout
        exceptions=(OSError, IOError, TimeoutError, ConnectionError),
    )
    i, j = task.data_array_indices
    xr_array[i, j] = out

async def fetch_array(
    self, task: SourceNameAsyncTask
) -> np.ndarray:
    """Fetch a single variable/time slice from remote store.

    Parameters
    ----------
    task : SourceNameAsyncTask
        Task with all fetch metadata

    Returns
    -------
    np.ndarray
        Fetched data array
    """
    # ALWAYS prefer pure async fsspec operations:
    data = await self.fs._cat_file(
        task.remote_uri, start=offset, end=offset + length
    )
    # Parse and apply modifier
    return task.modifier(parsed_data)
```

The key points:

- **`async_retry` wraps `fetch_array`** in `fetch_wrapper`, so each
  individual task is retried independently. If 1 of 100 tasks hits a
  transient error, only that one retries — the other 99 are fine.
- **Retry exceptions** should be scoped to transient I/O errors
  (OSError, IOError, TimeoutError, ConnectionError), not broad
  `Exception` which would mask programming errors.
- **Pure async I/O is REQUIRED**. Use `self.fs._cat_file()` for
  byte-range reads from s3fs/gcsfs rather than downloading to disk.

> **CRITICAL:** Avoid `asyncio.to_thread` whenever possible.
> Pure async is ALWAYS preferred. Only use `cancellable_to_thread`
> as an absolute last resort for unavoidable sync operations.

**Last resort pattern (avoid if possible):**

```python
# ONLY for unavoidable sync operations like pygrib parsing
async def fetch_array(self, task: SourceNameAsyncTask) -> np.ndarray:
    # First: fetch data using pure async
    data = await self.fs._cat_file(task.remote_uri, start=..., end=...)

    # Write to cache file (pure async preferred, but file write is fast)
    cache_path = self._get_cache_path(task)
    with open(cache_path, "wb") as f:
        f.write(data)

    # LAST RESORT: parse with sync library (e.g., pygrib)
    # Only use when no async alternative exists!
    grbs = await cancellable_to_thread(
        pygrib.open,
        cache_path,
        timeout=30.0,  # Coroutine abandoned after 30s, but thread continues!
    )
    try:
        values = grbs[1].values
    finally:
        grbs.close()

    return task.modifier(values)
```

### 7f. Time validation

```python
@classmethod
def _validate_time(cls, times: list[datetime]) -> None:
    """Verify date times are valid for this source.

    Parameters
    ----------
    times : list[datetime]
        Date times to validate
    """
    for time in times:
        if not (time - datetime(1900, 1, 1)).total_seconds() % INTERVAL == 0:
            raise ValueError(
                f"Requested date time {time} needs to be a "
                f"{INTERVAL // 3600}h interval for <SourceName>"
            )
        if time < MIN_DATE or time > MAX_DATE:
            raise ValueError(
                f"Requested date time {time} is outside the available "
                f"range [{MIN_DATE}, {MAX_DATE}] for <SourceName>"
            )
```

### 7g. Cache property

When `cache=False`, each data source instance must use a **unique
temporary subdirectory** to avoid collisions between concurrent
instances or repeated calls. Use a lazily-initialized UUID hash:

```python
@property
def cache(self) -> str:
    """Get the appropriate cache location."""
    cache_location = os.path.join(datasource_cache_root(), "source_name")
    if not self._cache:
        if self._tmp_cache_hash is None:
            # First access: create a random suffix to avoid collisions
            self._tmp_cache_hash = uuid.uuid4().hex[:8]
        cache_location = os.path.join(
            cache_location, f"tmp_source_name_{self._tmp_cache_hash}"
        )
    return cache_location
```

The constructor must initialize `self._tmp_cache_hash: str | None = None`.
Import `uuid` at the top of the file.

**IMPORTANT:** The `__call__` method **must** wrap the fetch call in
`try/finally` so the temp cache is cleaned even on error, timeout, or
cancellation — never leave orphaned temp directories on disk:

```python
try:
    result = loop.run_until_complete(
        asyncio.wait_for(self.fetch(...), timeout=self.async_timeout)
    )
finally:
    if not self._cache:
        shutil.rmtree(self.cache, ignore_errors=True)
```

### 7h. Available classmethod

```python
@classmethod
def available(cls, time: datetime | np.datetime64) -> bool:
    """Check if given date time is available.

    Parameters
    ----------
    time : datetime | np.datetime64
        Date time to check

    Returns
    -------
    bool
        If date time is available
    """
    if isinstance(time, np.datetime64):
        time = (
            time.astype("datetime64[ns]")
            .astype("datetime64[us]")
            .item()
        )
    try:
        cls._validate_time([time])
    except ValueError:
        return False
    return True
```

### Important implementation notes

- **Pure async I/O is REQUIRED** — use `fs._cat_file()`, async zarr,
  etc. This is not optional. Avoid `asyncio.to_thread` at all costs.
- **`asyncio.to_thread` is a LAST RESORT ONLY** — only use
  `cancellable_to_thread` when no async alternative exists (e.g.,
  pygrib parsing). Threads cannot be cancelled and cause pytest
  timeout issues.
- **Prefer fsspec-compatible filesystems** (s3fs, gcsfs, adlfs, etc.)
  over dedicated client libraries
- **Always use `managed_session`** instead of bare
  `set_session`/`close` — prevents session leaks on error
- **Always use `gather_with_concurrency`** with
  `self._async_workers` — prevents resource exhaustion. NEVER use
  bare `tqdm.gather(*tasks)` which launches all tasks at once.
- **Always wrap `fetch_array` with `async_retry`** in
  `fetch_wrapper` using `self._retries` — handles transient errors
- **Always use `ensure_utc`** on time inputs — ensures correct
  timezone handling
- **Never call `loop.set_default_executor()`** — this mutates global
  state. Use `cancellable_to_thread` with per-call timeout instead.
- **AVOID** using xarray for loading data where possible — work with
  files directly (GRIB via pygrib, NetCDF via netCDF4, Zarr via zarr)
- **AVOID** downloading full files — use byte-range requests, index
  files, or slice queries when possible
- **AVOID** over-complicating constructor parameters
- For **s3fs**: always use `skip_instance_cache=True`
- Data must always be in **unnormalized physical units**
- Use `loguru.logger` for all logging, never `print()`
- **Always use `try/finally`** in `__call__` to clean up temp cache
  when `cache=False` — ensures no orphaned temp directories on
  error, timeout, or cancellation

---

## Step 8 — Register the Source

### 8a. Add to `earth2studio/data/__init__.py`

Add the import in **alphabetical order** among existing imports:

```python
from .<filename> import SourceName
```

### 8b. Add lexicon to `earth2studio/lexicon/__init__.py`

Add the import in **alphabetical order**:

```python
from .<filename> import SourceNameLexicon
```

### 8c. Verify pyproject.toml

Confirm any new dependencies from Step 3 are present in the `data`
extras group.

---

## Step 9 — Update Documentation

### 9a. Add to the correct RST file

Based on source type, add the class to the appropriate RST file in
`docs/modules/` in **alphabetical order**:

| Source type | RST file |
|---|---|
| DataSource | `docs/modules/datasources_analysis.rst` |
| ForecastSource | `docs/modules/datasources_forecast.rst` |
| DataFrameSource | `docs/modules/datasources_dataframe.rst` |
| ForecastFrameSource | `docs/modules/datasources_dataframe.rst` |

Add the entry within the `.. autosummary::` directive:

```rst
   .. autosummary::
      :nosignatures:
      :toctree: generated/data/
      :template: datasource.rst

      data.ExistingSource
      data.NewSourceName    <-- alphabetical order
      data.OtherSource
```

### 9b. Add Badges to class docstring

The class docstring must include a `Badges` section as the **last**
section. Badges enable filtering in the documentation.

Available badges:

**Region**: `region:global`, `region:na`, `region:sa`, `region:eu`,
`region:as`, `region:af`, `region:au`

**Data class**: `dataclass:analysis`, `dataclass:reanalysis`,
`dataclass:observation`, `dataclass:simulation`

**Product**: `product:wind`, `product:precip`, `product:temp`,
`product:atmos`, `product:ocean`, `product:land`, `product:veg`,
`product:solar`, `product:radar`, `product:sat`, `product:insitu`

### 9c. Docstring must include reference URL

The class docstring **must** include a `Note` section with the
reference URL to the data source documentation. This is how users
discover the upstream data and its license:

```python
class SourceName:
    """Brief description.

    Extended description...

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks,
        by default 16
    retries : int, optional
        Number of retry attempts per failed fetch task with
        exponential backoff, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a
    large amount of data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository:

    - https://link-to-data-documentation
    - https://link-to-variable-reference

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp
    """
```

### 9d. Verify all public method docstrings

All public methods must have NumPy-style docstrings with
`Parameters`, `Returns`, and `Raises` sections. The class docstring
goes under the `class` definition, NOT under `__init__`.

---

## Step 10 — Update CHANGELOG.md

Edit `CHANGELOG.md` to record the new source under the **current
unreleased version** section (the topmost version block).

Add entries under the appropriate subsections following the
Keep-a-Changelog format:

```markdown
### Added

- Added <SourceName> <source_type> for <brief description> (`ClassName`)
- Added `SourceNameLexicon` for <source> variable mappings

### Dependencies

- Added `<package>` to `data` optional dependency group (if applicable)
```

Look at existing entries for style reference. For example, recent
additions look like:

```markdown
- Added CAMS Global atmospheric composition forecast data source (`CAMS_FX`)
- Added `CAMSGlobalLexicon` for CAMS variable mappings (AOD, total column gases)
```

If no new dependencies were added, omit the Dependencies subsection
entry.

---

## Step 11 — Verify Style, Format & Lint

### 11a. Run formatting

```bash
make format
```

### 11b. Run linting

```bash
make lint
```

Common issues:

- Missing type annotations on public functions
- Unused imports
- Import ordering
- Type errors from incorrect return types

### 11c. Check license headers

```bash
make license
```

Verify that **all new files** (data source, lexicon, test file) have
the correct SPDX Apache-2.0 license header.

Fix all errors before proceeding. Re-run until clean.

---

## Step 12 — Write Pytest Unit Tests

Create `test/data/test_<filename>.py` following the existing patterns.

### 12a. Test file structure for DataSource

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
    assert np.array_equal(
        data.coords["variable"].values, np.array(variable)
    )


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

### 12b. ForecastSource test additions

For ForecastSource, add lead_time tests:

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

### 12c. DataFrameSource test additions

For DataFrameSource, add schema and column validation:

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

### 12d. Mock test (REQUIRED)

Every data source **must** have at least one mock test that exercises
the full `__call__` path without requiring network access, timeouts,
or `xfail`. This ensures the data source can be tested in CI
environments without credentials or network connectivity.

Mock the download/fetch layer (e.g., `_download_products`) using
`unittest.mock.patch` to return synthetic data files, then verify
the DataFrame/DataArray output has the correct schema, shape, and
content.

Example pattern (for DataFrameSource):

```python
from unittest.mock import patch

def test_source_call_mock(tmp_path):
    """Mock the download layer to test __call__ without network."""
    # Create a synthetic data file in tmp_path
    # ...write minimal valid data to tmp_path / "mock_data.file"...

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

For DataSource / ForecastSource, write a synthetic NetCDF, GRIB, or
Zarr file and mock `fetch_array` or the filesystem layer.

### 12e. Test guidelines

- **No docstrings** on test functions
- Use **strategic parameterization** — minimize combinations
  while maximizing coverage
- Markers: `@pytest.mark.slow` + `@pytest.mark.xfail` +
  `@pytest.mark.timeout(30)` for all network tests
- Test both **single and list inputs** for time/variable
- Test **cache=True and cache=False**
- Test **availability / error handling**
- **At least one mock test** per source (see 12d) — no network,
  no timeout, no xfail
- **Target 90%+ line coverage** on the source module when running
  with `--slow` (see Step 13b). Use `--cov-report=term-missing`
  to identify uncovered lines.
- Run via: `make pytest TOX_ENV=test-data` or
  `pytest test/data/test_<filename>.py -v`

### 12e. Pytest timeout considerations

> **Warning:** Tests using `asyncio.to_thread` (via
> `cancellable_to_thread`) can hang even with `@pytest.mark.timeout`
> because Python threads cannot be forcibly cancelled.

When pytest timeout fires:

1. `asyncio.wait_for` raises `TimeoutError`
2. The coroutine is abandoned
3. But the underlying thread keeps running
4. Test process may hang waiting for the thread

**Mitigations:**

- **Avoid `asyncio.to_thread` entirely** — use pure async I/O
- If unavoidable, set test timeout **higher** than task timeouts
  (e.g., `@pytest.mark.timeout(60)` with 30s task timeout)
- Use `@pytest.mark.xfail` for tests that might hang
- **Always include at least one mock test** (see 12d) that exercises
  `__call__` without network, ensuring CI can validate the source

### **[CONFIRM — Tests]**

Present:

1. The test file path
2. The test functions and what they cover
3. Any test fixtures needed
4. Coverage percentage (must be >= 90% with `--slow`)
5. Ask if the tests look comprehensive enough

---

## Step 13 — Run Tests

### 13a. Run the new test file

```bash
uv run python -m pytest test/data/test_<filename>.py -v --timeout=60
```

All tests must pass (or `xfail` for network tests). Fix failures
and re-run until green.

### 13b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include network tests. The new data source file must achieve
**at least 90% line coverage**:

```bash
uv run python -m pytest test/data/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/data/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables network tests (marked `@pytest.mark.slow`)
- `--cov=earth2studio/data/<filename>` scopes coverage to the
  new source module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to
cover the missing lines. Common gaps:

- Error handling branches (e.g., empty products, invalid data)
- Edge cases in parsing (e.g., missing fields, corrupt records)
- Cache property paths (`cache=True` vs `cache=False`)
- `resolve_fields` with different input types

Re-run until coverage is at or above 90%.

### 13c. Run the full data test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-data
```

Confirm no regressions.

---

## Step 14 — Validate Variables, Summarize, and Sanity-Check Plots

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo.

### 14a. Script templates

**For DataSource / ForecastSource** (gridded data):

```python
"""Sanity-check plot for <SourceName> data source.

This script is for PR review only — do NOT commit to the repo.
Run it to produce a quick visualization confirming the source works.
"""
import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import SourceName

# Fetch a sample
ds = SourceName(cache=False)
time = ...  # pick a recent valid time
variables = ["t2m", "msl", "u10m"]  # 1-3 representative variables
data = ds(time, variables)

# Plot contours for each variable
fig, axes = plt.subplots(1, len(variables), figsize=(6 * len(variables), 5))
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    im = data.sel(variable=var).isel(time=0).plot(ax=ax, cmap="turbo")
    ax.set_title(f"{var}")

plt.suptitle(f"<SourceName> — {time}", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<source_name>.png")
```

**For DataFrameSource / ForecastFrameSource** (sparse data):

```python
"""Sanity-check plot for <SourceName> data source.

This script is for PR review only — do NOT commit to the repo.
Run it to produce a quick visualization confirming the source works.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import SourceName

# Fetch a sample
ds = SourceName(cache=False)
time = ...  # pick a recent valid time
variables = ["var1", "var2"]  # 1-3 representative variables
df = ds(time, variables)

# Convert lon from [0,360] to [-180,180] for cartopy
df["lon_plt"] = df["lon"].where(df["lon"] <= 180, df["lon"] - 360)

# Plot scatter on globe for each variable (cartopy projection)
fig, axes = plt.subplots(
    1, len(variables),
    figsize=(8 * len(variables), 5),
    subplot_kw={"projection": ccrs.Robinson()},
)
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    subset = df[df["variable"] == var]
    obs = subset["observation"].values
    # Use percentile clipping to avoid outliers blowing out the colorbar
    vmin, vmax = np.percentile(obs[np.isfinite(obs)], [2, 98])

    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

    sc = ax.scatter(
        subset["lon_plt"], subset["lat"],
        c=obs, s=2, cmap="turbo", alpha=0.8,
        vmin=vmin, vmax=vmax, edgecolors="none",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{var} ({len(subset)} obs)\nrange: {vmin:.0f}–{vmax:.0f} (p2–p98)")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Observation",
                 orientation="horizontal", pad=0.05)

plt.suptitle(f"<SourceName> — {time}", y=1.0)
plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<source_name>.png")
```

> **Tip:** Always use `cartopy` with a proper map projection
> (Robinson, PlateCarree, etc.) for geospatial scatter plots.
> Without a projection, satellite swath data looks distorted and
> hard to interpret. Use percentile-clipped `vmin`/`vmax` and
> `s >= 2` marker size — without clipping, outliers compress the
> colorbar and make the plot appear blank.

### 14b. Validate all lexicon variables

**Every variable in the lexicon must be validated against real data.**
A variable that exists in the lexicon but consistently produces missing
or invalid data should be **removed** from the lexicon and
`E2STUDIO_VOCAB`.

Run a validation script that fetches a representative sample and
checks every variable:

```python
"""Variable validation for <SourceName>.

Checks every lexicon variable against real data to confirm it
returns valid observations. Variables with very low valid-data
rates should be removed from the lexicon.
"""
from datetime import datetime

from earth2studio.data import SourceName
from earth2studio.lexicon import SourceNameLexicon

ds = SourceName(cache=True)
time = ...  # pick a recent valid time
all_vars = list(SourceNameLexicon.VOCAB.keys())
df = ds(time, all_vars)

print(f"{'Variable':<16} {'Obs Count':>10} {'Valid %':>8} {'Min':>10} {'Max':>10}")
print("-" * 60)
for var in sorted(all_vars):
    sub = df[df["variable"] == var]
    n_total = len(sub)
    n_valid = sub["observation"].notna().sum()
    pct = (n_valid / n_total * 100) if n_total > 0 else 0
    vmin = sub["observation"].min() if n_valid > 0 else float("nan")
    vmax = sub["observation"].max() if n_valid > 0 else float("nan")
    flag = " *** REMOVE" if pct < 10 else ""
    print(f"{var:<16} {n_total:>10} {pct:>7.1f}% {vmin:>10.2f} {vmax:>10.2f}{flag}")
```

**Action required:**

- Variables with **< 10% valid data** must be removed from:
  1. The lexicon class `VOCAB` dict
  2. `E2STUDIO_VOCAB` in `earth2studio/lexicon/base.py`
  3. The class docstring (note the removal and reason)
- Variables with **10-50% valid data** should be kept but documented
  with a note explaining the low coverage (e.g., day/night switching,
  quality filtering)
- After removing variables, re-run `make lint` and tests

### 14c. Summarize variables and time range to user

Before generating sanity-check plots, **present a summary table** to
the user covering:

1. **All valid variables** — name, description, observation count,
   value range
2. **Removed variables** — name, reason for removal (e.g., "> 97%
   missing data")
3. **Valid time range** — earliest and latest data available from the
   remote store
4. **Typical data density** — observations per orbit/file,
   approximate global coverage cadence

Example summary format:

> **Variable Summary for MetOpAMSUA (14 channels):**
>
> | Variable | Freq (GHz) | Layer | Obs/orbit | BT Range (K) |
> |----------|-----------|-------|-----------|---------------|
> | amsua01 | 23.8 | Surface | 22,950 | 142–291 |
> | amsua02 | 31.4 | Surface | 22,950 | 145–290 |
> | ... | | | | |
>
> **Removed:** `amsua15` (89.0 GHz) — L1B product marks ~97% of
> measurements as missing due to quality filtering.
>
> **Time range:** 2007-10-22 (Metop-A launch) to present.
> Metop-B (2012-09-17 to present), Metop-C (2018-11-07 to present).
>
> **Data density:** ~767 scan lines × 30 FOVs = 23,010 obs per orbit,
> ~14 orbits/day, global coverage twice daily.

This summary helps the user understand what they are getting from the
data source and verify it matches their expectations.

### 14d. Create sanity-check plot script

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo. See 14a above for templates.

### 14e. Run the script

Execute the script and **verify the output images exist and look
reasonable**:

```bash
python sanity_check_<source_name>.py
```

Check that:

- The script runs without errors
- Output PNGs are generated
- Data points appear in expected geographic regions
- Values are in physically reasonable ranges (e.g., temperatures
  200–320 K, not 0 or NaN)
- For DataFrame sources: observations form recognizable swath or
  station patterns

### 14f. **[CONFIRM — Sanity-Check Plots]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the script ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
ask them to open and inspect the output:

> The sanity-check script ran successfully and saved the following
> plot:
>
> `/absolute/path/to/sanity_check_<source_name>.png`
>
> **Please open this image and confirm it looks correct.** Check:
>
> 1. Data points are visible on the axes (not blank/empty)
> 2. Geographic coverage matches expectations (global swaths,
>    regional stations, etc.)
> 3. Colorbar values are in physically reasonable ranges
> 4. No obvious artifacts (all-NaN regions, garbled coordinates)
>
> Does the plot look correct?

**Do not proceed to Step 15 until the user explicitly confirms the
plots look correct.** If the user reports problems (empty plots,
wrong ranges, missing coverage), debug and fix the issue, re-run
the script, and ask the user to inspect again.

The output images will be uploaded to the PR description in
Step 15.

---

## Step 15 — Branch, Commit & Open PR

### **[CONFIRM — Ready to Submit]**

Before proceeding, confirm with the user:

 > All implementation steps are complete:
 >
 > - Data source class implemented with correct method ordering
 > - Lexicon class created with variable mappings
 > - **All lexicon variables validated against real data** (low-validity
 >   variables removed)
 > - **Variable summary and time range presented to user**
 > - E2STUDIO_VOCAB / E2STUDIO_SCHEMA updated (if needed)
 > - Registered in `__init__.py` files
 > - Documentation RST files updated with badges
 > - Reference URLs included in docstrings
 > - CHANGELOG.md updated
 > - Format, lint, and license checks pass
 > - Unit tests written and passing
 > - Dependencies in `data` extras group confirmed
 > - Sanity-check plots generated and confirmed by user
 >
 > Ready to create a branch, commit, and prepare a PR?

### 15a. Create branch and commit

```bash
git checkout -b feat/data-source-<name>
git add earth2studio/data/<filename>.py \
        earth2studio/data/__init__.py \
        earth2studio/data/utils.py \
        earth2studio/lexicon/<filename>.py \
        earth2studio/lexicon/__init__.py \
        earth2studio/lexicon/base.py \
        test/data/test_<filename>.py \
        docs/modules/datasources_*.rst \
        pyproject.toml \
        CHANGELOG.md
git commit -m "feat: add <SourceName> data source

Add <SourceName> <source_type> for <brief description>.
Includes lexicon, unit tests, and documentation."
```

Do **NOT** add the sanity-check script or its output image.

### 15b. Identify the fork remote and push branch

The working repository is typically a **fork** of
`NVIDIA/earth2studio`. Before pushing, confirm which git remote
points to the user's fork:

```bash
git remote -v
```

Ask the user:

> Which git remote is your fork of `NVIDIA/earth2studio`?
> (Usually `origin` — e.g., `git@github.com:<user>/earth2studio.git`)

Then push the feature branch to the **fork** remote:

```bash
git push -u <fork-remote> feat/data-source-<name>
```

### 15c. Open Pull Request (fork → NVIDIA/earth2studio)

> **Important:** PRs must be opened **from the fork** to the
> **upstream source repository** `NVIDIA/earth2studio`. The branch
> lives on the fork; the PR targets `main` (or the appropriate
> base branch) on the upstream repo.

Use `gh pr create` with explicit `--repo` and `--head` flags:

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/data-source-<name> \
  --title "feat: add <SourceName> data source" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork
(e.g., `NickGeneva`).

The PR should follow the repo's template and
include the data licensing section, the sanity-check script inside
a `<details>` block, and the sanity-check images uploaded to the
PR body:

````markdown
## Description

Add `<ClassName>` <source_type> for <brief description of what
the data source provides>.

Closes #<issue_number> (if applicable)

### Data source details

| Property | Value |
|---|---|
| **Source type** | DataSource / ForecastSource / DataFrameSource / ForecastFrameSource |
| **Remote store** | <URL to data documentation> |
| **Format** | GRIB2 / NetCDF / Zarr / CSV / etc. |
| **Spatial resolution** | X deg x Y deg (NxM grid) |
| **Temporal resolution** | Hourly / 6-hourly / daily / etc. |
| **Date range** | YYYY-MM-DD to present |
| **Region** | Global / Regional |
| **Authentication** | Anonymous / API key / etc. |

### Data licensing

> **License**: <Name of the data license>
> **URL**: <Link to the license terms>
>
> <Brief summary of key restrictions or permissions, e.g.,
> "Open data, freely available for commercial and
> non-commercial use" or "Requires attribution, non-commercial
> use only">

### Dependencies added

- `<package>=<version>` — <brief reason> (added to `data` extras)
- *(or "No new dependencies needed")*

### Reference plot

<details>
<summary>Sanity-check visualization (click to expand)</summary>

![sanity_check_<source_name>](sanity_check_<source_name>.png)

*<Caption describing what the plot shows, e.g., "2m temperature
from <SourceName> at 2024-01-01T00:00 UTC">*

</details>

### Sanity-check script

Include the sanity-check Python script inside a `<details>` block
so reviewers can reproduce the plot:

````markdown
<details>
<summary>Sanity-check script (click to expand)</summary>

```python
<paste the full sanity-check script here>
```

</details>
````

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues

## Dependencies

<List any new packages added to pyproject.toml, or "None">

```` <!-- markdownlint-disable-line MD040 -->

Drag-and-drop or attach the sanity-check PNG image into the PR body
so the `<details>` spoiler renders correctly.

---

## Step 16 — Automated Code Review (Greptile)

After the PR is created and pushed, an automated code review from
**greptile-apps** (Greptile) will be posted as PR review comments.
Wait for this review, then process the feedback.

### 16a. Wait for Greptile review

Poll for review comments from `greptile-apps[bot]` every 30 seconds
for up to **5 minutes**. Time out gracefully if no review arrives:

```bash
# Poll loop — check every 30s, timeout after 5 minutes (10 attempts)
for i in $(seq 1 10); do
  REVIEW_ID=$(gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
    --jq '.[] | select(.user.login == "greptile-apps[bot]") | .id' 2>/dev/null)
  if [ -n "$REVIEW_ID" ]; then
    echo "Greptile review found: $REVIEW_ID"
    break
  fi
  echo "Attempt $i/10 — no review yet, waiting 30s..."
  sleep 30
done
```

If no review after 5 minutes, inform the user:

> Greptile hasn't posted a review after 5 minutes. This can happen if
> the review bot is busy or the PR hasn't triggered it. You can:
> 1. Ask me to check again later
> 2. Skip this step and proceed without automated review
> 3. Manually request a review from Greptile on the PR page

### 16b. Pull and parse review comments

Once the review is posted, fetch all comments:

```bash
# Get all review comments on the PR
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") |
    {path: .path, line: .diff_hunk, body: .body}'
```

Also fetch the top-level review body:

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | .body'
```

### 16c. Categorize and present to user

Parse each comment and categorize it:

| Category | Description | Default action |
|---|---|---|
| **Bug / correctness** | Logic errors, wrong behavior | Fix |
| **Style / convention** | Naming, formatting, patterns | Fix if valid |
| **Performance** | Inefficiency, resource waste | Evaluate |
| **Documentation** | Missing/wrong docs, docstrings | Fix |
| **Suggestion** | Alternative approach, nice-to-have | User decides |
| **False positive** | Incorrect or irrelevant feedback | Dismiss |

### **[CONFIRM — Review Triage]**

Present each comment to the user in a summary table:

```markdown
| # | File | Line | Category | Summary | Proposed Action |
|---|------|------|----------|---------|-----------------|
| 1 | metop_amsua.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | metop_avhrr.py | 305 | Style | Use f-string | Fix: convert |
| 3 | metop.py | 45 | Suggestion | Add type alias | Skip: not needed |
| ... | ... | ... | ... | ... | ... |
```

For each comment, briefly explain:
- What Greptile flagged
- Whether you agree or disagree (with reasoning)
- Your proposed fix (or why to skip)

Ask the user to confirm which comments to address. The user may:
- Accept all proposed fixes
- Select specific fixes
- Override your recommendation on any comment
- Add their own fixes

### 16d. Implement fixes

For each accepted fix:

1. Make the code change
2. Run `make format && make lint` after all fixes
3. Run the relevant tests
4. Commit with a message like:
   `fix: address code review feedback (Greptile)`

### 16e. Respond to review comments

For each Greptile comment, post a reply on the PR:

**For fixed comments:**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <brief description of fix>"
```

**For dismissed comments (false positives / won't fix):**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <brief justification>"
```

### 16f. Push and resolve

```bash
git push origin <branch>
```

After pushing, resolve all addressed review threads if possible.

Inform the user of the final state:
- How many comments were fixed
- How many were dismissed (with reasons)
- Any remaining open threads

---

## Reminders

- **DO NOT** commit the sanity-check script or image to the repo
- **DO** use `loguru.logger` for logging, never `print()`, inside
  `earth2studio/`
- **DO** ensure all public functions have full type hints (mypy-clean)
- **DO** maintain alphabetical order in `__init__.py` exports,
  RST file entries, and CHANGELOG entries
- **DO** follow the canonical method ordering within the class
- **DO** use the async task dataclass pattern for parallel fetching
- **DO** use `prep_data_inputs` / `prep_forecast_inputs` to normalize
  inputs
- **DO** use `nest_asyncio.apply()` in `__init__` for notebook compat
- **DO** use `datasource_cache_root()` for cache paths
- **DO** delete temporary cache when `cache=False`
- **DO** add all new dependencies to the `data` extras group in
  `pyproject.toml` using `uv add --extra data <package>`
- **DO** include reference URLs in class docstrings and lexicon docs
- **DO** always update CHANGELOG.md under the current unreleased version
- **DO** use async utilities from `earth2studio/data/utils.py`:
  `managed_session`, `gather_with_concurrency`, `async_retry`,
  `ensure_utc`
- **DO** use pure async I/O (`fs._cat_file()`, async zarr, etc.)
- **PREFER** fsspec-compatible filesystems (s3fs, gcsfs, adlfs, etc.)
  over dedicated client libraries
- **AVOID** `asyncio.to_thread` — pure async is ALWAYS preferred.
  Only use `cancellable_to_thread` as a last resort when no async
  alternative exists.
- **AVOID** bare `tqdm.gather(*tasks)` — use `gather_with_concurrency`
- **AVOID** using xarray for loading data — prefer direct file I/O
- **AVOID** downloading full files — use byte-range / slicing
- **AVOID** over-complicating constructor parameters
- **NEVER** call `loop.set_default_executor()` — mutates global state
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git. Always read
  credentials from environment variables at runtime. If the user
  provides test credentials during the session, use them only in
  ephemeral shell commands — never persist them.
