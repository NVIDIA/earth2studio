---
name: earth2studio-data-fetch
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - data-fetch
    - weather-data
    - xarray
description: >
  Fetch weather/climate data via Earth2Studio data sources for specific variables
  and times. Do NOT use for inference pipelines, model discovery, or installation.
---

# Earth2Studio Data Fetch Skill

## Purpose

Guide a user through fetching weather/climate data with Earth2Studio data
sources: pick a source that provides the requested variables and times, verify
support from the lexicon, and produce a working script that returns an xarray
DataArray (or a pandas DataFrame for observations).

## Prerequisites

- Earth2Studio installed (see `earth2studio-install`; it owns the Python
  version, extras, and environment setup)
- Network access to the remote data store
- Some sources need credentials or an extra (for example CDS needs
  `~/.cdsapirc`); the source's docstring says what it needs

## Core principle: read the repo and live docs, do not memorize

Sources, variables, and lexicons change every release. Any list in this skill
would go stale, so look things up and cite what you used.

| Need | Where to look (repo path first, live page as a fallback) |
|------|------------------------------------------------|
| Available sources (analysis, forecast, dataframe) | `docs/modules/datasources_analysis.md`, `datasources_forecast.md`, `datasources_dataframe.md` |
| Source constructor args, auth, and caveats | the class in `earth2studio/data/` and its docstring |
| Variables a source supports | its lexicon in `earth2studio/lexicon/` (`VOCAB` keys) |
| Temporal statistics a source supports | `TIME_STATISTICS` in its lexicon, and the "Temporal statistics" section of the class docstring |
| Variable naming | `docs/userguide/advanced/lexicon.md`, `E2STUDIO_VOCAB` in `earth2studio/lexicon/base.py` |
| Fetch helpers (`fetch_data`, `fetch_dataframe`, regridding, local stores) | `earth2studio/data/utils.py` and `docs/userguide/components/datasources.md` |
| Source interfaces (call signatures) | `earth2studio/data/base.py` |
| Cache locations and env vars | `docs/userguide/about/install.md` |
| Changes from 0.x | `MIGRATION.md` ("Data sources and fetching") |

Live pages (note the `/main/` segment; URLs without it return 404):

```text
https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/
https://nvidia.github.io/earth2studio/main/modules/datasources_forecast/
https://nvidia.github.io/earth2studio/main/modules/datasources_dataframe/
https://nvidia.github.io/earth2studio/main/userguide/components/datasources/
https://nvidia.github.io/earth2studio/main/userguide/advanced/lexicon/
```

If working in a local clone, read the files directly. Otherwise fetch from
`https://github.com/NVIDIA/earth2studio` or the live pages. Fetch only what the
request needs.

## Instructions

### Step 1. Understand the request

Ask at most 3 questions, skipping what is already known:

- **Variables.** Use Earth2Studio names (`t2m`, `u500`, `z850`, `tp`, `msl`).
  Map plain language ("500 hPa geopotential height") using `E2STUDIO_VOCAB`.
- **Time.** One timestamp, a range, or several discrete times (UTC).
- **Data type.** Analysis/reanalysis (state at a time), forecast (init time plus
  lead times), or point observations (a DataFrame, not a gridded array).
- **Lead time and init time** (forecast only).
- **Region and grid.** Global or regional; does the data need to match a model's
  grid (see regridding below)?
- **Statistics.** Does the user want an accumulation or aggregate over a window
  (for example 24 h mean temperature) instead of an instantaneous field?
- **Output.** In-memory DataArray (default), or a local NetCDF/Zarr file.

### Step 2. Choose candidate sources

Match the data type to the right docs page (analysis, forecast, or dataframe),
then filter by temporal coverage (operational sources keep limited history,
reanalyses go back decades), region and resolution, and access (some need keys or
extras). Check each candidate's lexicon: a variable that is not a `VOCAB` key
cannot be fetched from that source.

Present viable options and tradeoffs briefly and let the user pick. If one choice
is clearly best, recommend it and ask for confirmation.

### Step 3. Write the fetch script

Use the source's own docstring for constructor arguments. Match the source type:

| Source type | Call | Returns |
|-------------|------|---------|
| Analysis | `ds(time, variable)` | DataArray `[time, variable, ...]` |
| Forecast | `ds(time, lead_time, variable)` | DataArray `[time, lead_time, variable, ...]` |
| DataFrame | `ds(time, variable, fields=None)` | pandas DataFrame |
| Forecast DataFrame | `ds(time, lead_time, variable, fields=None)` | pandas DataFrame |

For gridded data, prefer `fetch_data` from `earth2studio.data`. It works with
both analysis and forecast sources and is what inference pipelines use:

```python
import numpy as np
from earth2studio.data import fetch_data, <SourceClass>

ds = <SourceClass>()

x = fetch_data(
    ds,
    time=np.array([np.datetime64("YYYY-MM-DDTHH:00")]),
    variable=np.array(["t2m", "u10m"]),
    lead_time=np.array([np.timedelta64(0, "h")]),  # forecast sources: the lead times
)
# x is one xr.DataArray with dims [time, lead_time, variable, ...]
print(x.dims, x.shape)
print(x.coords)
```

Rules that `fetch_data` enforces, and the reasons behind them:

- `time` and `lead_time` are non-empty 1D NumPy `datetime64` / `timedelta64`
  arrays without NaT, and `variable` is a non-empty 1D array. Plain
  `datetime.datetime` lists work when calling a source directly, but not here.
- The result is always a single DataArray (the old `tensor, coords` tuple and
  the `legacy=` option are gone from 1.0).
- Pass `device="cuda"` to get CuPy-backed data; the default is NumPy on CPU.

Calling a source directly (`ds(time, variable)`) is fine for a simple one-off and
gives the same kind of DataArray without the extra handling.

#### Optional features, use only when the request needs them

- **Temporal statistics.** Qualify a variable with a statistic and window, for
  example `"t2m:mean:24h"` (methods such as `mean`, `sum`, `min`, `max`). Only
  sources whose lexicon declares `TIME_STATISTICS` support it; others raise
  `ValueError`. Windows include the left endpoint and exclude the right. For
  forecast sources the requested lead time must cover the window.
- **Regridding.** Pass `target_grid` (for example
  `infer_grid(model.input_coords())`) to return data on a given grid. Without a
  `regridder` it passes through, crops exactly, or uses nearest neighbor, and it
  raises if the target extends beyond the source. For a chosen method, pass a
  regridder from `earth2studio.grids` (see `list_regridders`).
- **Observations.** Use `fetch_dataframe(source, time, variable, fields=...)`,
  or `fetch_dataframes` to pull several sources concurrently. DataFrame sources
  expose a `SCHEMA`; use `fields` to select columns. Some require constructor
  arguments (for example `ISD` needs `stations=[...]`) and have a
  `time_tolerance` for matching observations to requested times.
- **Combining sources.** `CompositeSource` / `CompositeForecastSource` map
  variables to different sources behind one interface. A source can only be
  one type in a call; do not mix analysis and forecast in one `fetch_data`.
- **Saving locally.** `datasource_to_file(...)` writes NetCDF or Zarr, which can
  be read back with `DataArrayFile`, `DataSetFile`, or `DataArrayDirectory`.
  Prefer this over repeated network fetches in later runs.

Include in the script: imports, short comments, and how to inspect the result
(`print(x)`, `x.shape`, `x.coords`).

### Step 4. Offer next steps

- How to change variables/times without rewriting the script.
- Data is cached locally after the first fetch. `EARTH2STUDIO_CACHE` sets the
  cache root, `EARTH2STUDIO_DATA_CACHE` overrides it for data sources only.
- If they want to run a model on the data, point them to `earth2studio-discover`
  and `earth2studio-deterministic-forecast`.

## Ownership and out-of-scope

**Owns:** choosing data sources for a variable/time request, verifying variable
support via the lexicon, and generating fetch scripts.

**Does not own:** installation (`earth2studio-install`), model selection
(`earth2studio-discover`), inference pipelines, writing a new data source
(see `earth2studio-create-datasource`), or credential setup beyond what the
docs describe.

## Example

> "I need 500 hPa geopotential height and 2m temperature from ERA5 for
> January 1, 2020 at 00Z."

1. Map to `z500`, `t2m`.
2. Check the ERA5-based sources' lexicons in `earth2studio/lexicon/`.
3. Recommend the one that needs no API key if the user has none, otherwise the
   official one; confirm.
4. Generate a `fetch_data` script for the chosen source.

## Limitations

- Network is required; sources read from remote stores.
- Variable availability and time coverage differ per source; always verify.
- Only as current as the branch or docs you read.
- Rate limits and queues (notably CDS) are outside Earth2Studio's control.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: '<var>'` | Not in the source's lexicon | Check `VOCAB`; try another source |
| `ValueError` mentioning a statistic | Source lacks that temporal statistic | Check `TIME_STATISTICS`; use a supporting source or compute it yourself |
| `ValueError` on `time` / `lead_time` | Not a 1D `datetime64` / `timedelta64` array | Build the arrays with `np.array([np.datetime64(...)])` |
| `FileNotFoundError` / 404 | Time outside the source's coverage | Check the source's temporal coverage |
| Timeout on a CDS source | Queue congestion | Retry later or use a source without a queue |
| `ModuleNotFoundError` | Missing extra or install | See `earth2studio-install` |
| Empty result | Time/variable mismatch | Recheck timestamp (UTC) and variable name |
| Target grid error | `target_grid` extends beyond the source | Choose a source covering the region or a smaller grid |
