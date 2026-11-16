# Temporal Statistics

Temporal statistics describe quantities over a time window, such as a daily mean
temperature or six-hour precipitation total. Request them through qualified variable
labels. The data source supplies the reduced field; `fetch_data` forwards the request
and preserves its metadata before device placement and regridding.

## Labels and Windows

Use `variable:method:duration` for a trailing window or
`variable:method:start:end` for explicit offsets from the target valid time `T`.
Built-in methods are `mean`, `sum`, `min`, and `max`.

| Label | Window | Hourly samples |
| --- | --- | --- |
| `t2m:mean:24h` | `(T - 24h, T]` | `T-23h` through `T` |
| `tp:sum:6h` | `(T - 6h, T]` | `T-5h` through `T` |
| `t2m:mean:-12h:+12h` | `(T - 12h, T + 12h]` | `T-11h` through `T+12h` |
| `t2m:max:-25h:-1h` | `(T - 25h, T - 1h]` | `T-24h` through `T-1h` |

Windows are **left-open and right-closed**: the starting timestamp is excluded and
the ending timestamp is included. At six-hour cadence, `mean:24h` reads four samples
at `T-18h`, `T-12h`, `T-6h`, and `T`.

Durations use integer units from nanoseconds through weeks, including `6h`, `1day`,
and `2weeks`. A week is seven days. Months and years are unsupported; use explicit
days or hours. Uppercase `M` and `Y` are rejected; lowercase `m` means minutes.

See [E2S-03: Time Statistics][time-spec] for window planning, source aggregation,
missing-sample handling, and metadata requirements.

[time-spec]: ../../specs/E2S-03_TIME_STATISTICS_SPEC.md

## Request Reduced Fields

Mix ordinary and qualified variables in one request:

```python
import numpy as np

from earth2studio.data import ARCO_ERA5, fetch_data

field = fetch_data(
    ARCO_ERA5(),
    time=np.array(["2024-01-02T00"], dtype="datetime64[ns]"),
    variable=np.array(["u10m", "t2m:mean:24h", "tp:sum:6h"]),
    device="cpu",
)
metadata = field.attrs["earth2studio_statistics"]
```

The output keeps the requested variable spelling and target timestamps, with
dimensions `time, lead_time, variable, ...`. The source reads the extra samples
needed for each window internally. Inspect the source's `Temporal statistics:`
docstring note and its lexicon's `TIME_STATISTICS` for support.

These built-in sources support all four methods:

| Source | Sampling cadence | Window axis |
| --- | --- | --- |
| `ARCO_ERA5` | 1 hour | Valid time |
| `WB2ERA5`, `WB2ERA5_121x240`, `WB2ERA5_32x64` | 6 hours | Valid time |
| `GFS` | 6 hours | Valid time |
| `GFS_FX` | 1 hour | Lead within initialization |
| `HRRR` | 1 hour | Valid time |
| `HRRR_FX` | 1 hour | Lead within initialization |

Availability constraints still apply at those cadences. `WB2Climatology` does not
declare native temporal-statistics support.

### Accumulated Fields and Calendar Days

Hourly ARCO-ERA5 `tp` represents precipitation over `(t - 1h, t]`, so
`tp:sum:6h` gives the six-hour total ending at `T`. GFS raw `tp` instead uses resetting,
overlapping buckets: summing them double counts precipitation. Use its `tp06`
quantity for six-hour totals, and check that any further sum uses disjoint periods.
WB2 `tp12` and `tp24` can likewise overlap at the source's sampling cadence.

A calendar day labeled by its **starting midnight** needs explicit offsets.
FuXi-S2S uses `t2m:mean:-1h:23h` for instantaneous samples at 00–23 UTC and
`tp:mean:0h:24h` for hourly accumulations stamped 01 UTC through next-day 00 UTC.
The precipitation mean retains one-hour accumulation units; multiply it by 24 for
a daily total. Replacing either label with `mean:24h` selects a different day window.

## Work with Local Arrays

Use the same window planner and reducer for data already in memory:

```python
import numpy as np
import xarray as xr

from earth2studio.utils.time_statistics import (
    apply_time_statistic,
    source_times,
    time_statistic_metadata,
)

target = np.datetime64("2024-01-02T00", "ns")
cadence = np.timedelta64(6, "h")
times = source_times("mean:24h", target, cadence)
block = xr.DataArray(
    np.array([[280.0], [282.0], [284.0], [286.0]]),
    dims=("time", "variable"),
    coords={"time": times, "variable": ["t2m"]},
)
result = apply_time_statistic(block, "mean:24h", target, cadence)
# result has dimension "variable" and value 283 K.
result = result.assign_coords(variable=["t2m:mean:24h"])
result.attrs["earth2studio_statistics"] = {
    "t2m:mean:24h": time_statistic_metadata("mean:24h")
}
```

`apply_time_statistic` reduces one scalar target, preserving non-time dimensions.
It selects `lead_time` when present, otherwise `time`; pass `dimension` to override.
Missing or duplicate required coordinates are rejected. For forecast planning,
use `source_lead_times` with timedelta targets.

Model coordinate constructors derive statistics metadata from qualified labels;
there is no separate `statistics=` declaration. When reducing local arrays directly,
label the resulting quantity and attach metadata as above before passing it onward.
For spatial reductions and forecast-verification metrics, see
[Statistics](../components/statistics.md).
