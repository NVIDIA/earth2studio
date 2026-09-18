# Earth2Studio Time Statistics

Describe temporal source requirements and reduce Xarray data. Fetch integration
and metadata accessors are separate work.

## Declarations and Ownership

Qualified variable labels identify temporal quantities unambiguously:

```python
variables = ["u10m", "t2m:mean:1day", "t2m:mean:1week", "t2m:max:1month"]
```

The first `:` separates the source name from its modifier; base names cannot
contain `:`. Unqualified labels are instantaneous. Qualified labels are canonical
for model and output coordinates. When each source variable occurs once, a mapping
such as `{"u10m": "mean:24h"}` is equivalent shorthand; a single modifier can also
apply to all requested variables.

`fetch_data` or coordinate-system objects own label splitting, shorthand conversion,
duplicate/conflict checks, and grouping. `time_statistics` accepts modifiers only
and owns normalization, source-window planning, metadata, and reductions.

## Windows

For valid time `T`, intervals are half-open:

| Modifier | Selected interval |
| --- | --- |
| `mean:24h` | `[T - 24h, T)` |
| `mean:-12h:+12h` | `[T - 12h, T + 12h)` |

Built-in methods are `mean`, `sum`, `min`, and `max`. Source cadence must divide
the window exactly. Durations are integers with units `ns`, `us`, `ms`,
`s/sec/second`, `m/min/minute`, `h/hr/hour`, `d/day`, `w/wk/week`, or
`mo/mon/month`; plural names are accepted. Weeks are seven days, months fixed
30-day windows—not calendar-relative months.

### Calendar-Day Means: FuXi-S2S

FuXi-S2S labels days by **starting midnight**, consumes two consecutive daily means,
and predicts the next. All 76 channels declare means with these hourly windows:

| Quantity | Qualified label | Samples relative to day start |
| --- | --- | --- |
| Ordinary fields | `t2m:mean:0h:24h` | 00–23 UTC |
| Hourly precipitation accumulations | `tp:mean:1h:25h` | 01 UTC through next 00 UTC |
| Hourly thermal radiation accumulations | `ttr:mean:1h:25h` | 01 UTC through next 00 UTC |

Only `tp` and `ttr` shift the window because their source timestamps mark interval
ends. `[T + 1h, T + 25h)` includes 24 hourly samples, excluding next-day 01 UTC.
These means retain one-hour accumulation units; multiply predicted `tp` by 24 for
a daily total. Channel order and model-unit conversions are unchanged.

Preserve start-of-day timestamps and one-day lead increments. Substituting
`mean:24h` would select the preceding day. Fetch at hourly cadence; label already
prepared daily means directly without reducing them again.

## Planning and Reduction

```python
delta_t = np.timedelta64(6, "h")
times = source_times("mean:24h", valid_time, delta_t)
leads = source_lead_times("mean:24h", lead_time, delta_t)
result = apply_time_statistic(block, "mean:24h", valid_time, delta_t)
```

Planning accepts scalar or array targets. Forecast sources retain initialization
time and request lead times, including negative leads when required.

`apply_time_statistic` reduces a variable block for one scalar target, excluding
values outside the window and rejecting missing or duplicate required coordinates.
It selects `lead_time` when present, otherwise `time`; callers may override the
dimension. Xarray preserves NumPy, CuPy, or Dask execution without implicit transfers.

The fetch layer groups variables by normalized modifier, fetches base names, and
reduces once per group. Outputs retain requested qualified labels. Cache parsed
modifiers, avoid host transfers and repeated concatenation, and assemble results
once. Fetch-plan caching, contiguous variable ordering, and streaming are future
optimizations.

Custom reductions share the window syntax and must preserve non-time dimensions,
including `variable`:

```python
register_time_statistic("range", lambda x, dim: x.max(dim) - x.min(dim))
```

## Metadata and Scope

`time_statistic_metadata()` returns normalized modifier, method, window, offsets,
and interval closure. Durations remain `np.timedelta64` until serialization.
Results may mirror qualified labels in `earth2studio_statistics`:

```python
array.attrs["earth2studio_statistics"] = {"t2m:mean:1day": "mean:1day"}
```

Labels carry quantity identity through handshakes; attributes support lookup and
serialization. Record CF-compatible time bounds and `cell_methods` when supported.

This utility excludes fetch implementation, irregular-cadence inference,
cross-variable diagnostics, and initial streaming/fused GPU kernels.
