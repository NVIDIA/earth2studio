# Earth2Studio Time Statistics

## Goal

Describe temporal source requirements and apply reductions to Xarray data. Fetch
integration and metadata accessors are separate work.

## Declaration

Variable coordinates remain strings. A statistic-qualified label appends its
modifier to the source variable name:

```python
variables = ["u10m", "t2m:mean:1day", "t2m:mean:1week", "t2m:mean:1month"]
```

This makes each quantity unambiguous when one model requests multiple statistics of
the same source variable. The first `:` separates the source name from the modifier;
base variable names must not contain `:`. Unqualified labels remain instantaneous.

For requests containing each source variable at most once, a mapping is equivalent
shorthand:

```python
statistics = {"u10m": "mean:24h", "v10m": "mean:24h", "t2m": "max:24h"}
```

A single modifier may also apply to every requested variable. Qualified labels are
the canonical model-coordinate and output-coordinate representation.

## Modifiers

Two compact forms are supported:

- `mean:24h` describes `T - 24h <= t < T`
- `mean:-12h:+12h` describes `T - 12h <= t < T + 12h`

`T` is valid time. Intervals are half-open, and the source cadence must divide the
window exactly. Built-in methods are `mean`, `sum`, `min`, and `max`.

Durations are integer values with these units:

- nanoseconds: `ns`
- microseconds: `us`
- milliseconds: `ms`
- seconds: `s`, `sec`, `second`
- minutes: `m`, `min`, `minute`
- hours: `h`, `hr`, `hour`
- days: `d`, `day`
- weeks: `w`, `wk`, `week`
- months: `mo`, `mon`, `month`

Plural names are accepted. Weeks are seven days and months are fixed 30-day windows,
so declarations behave identically for analysis timestamps and forecast lead times.
Calendar-relative months are not supported.

## Source Planning

Analysis sources request timestamps:

```python
delta_t = np.timedelta64(6, "h")
times = source_times("mean:24h", valid_time, delta_t=delta_t)
```

Forecast sources keep initialization time fixed and request lead times:

```python
leads = source_lead_times("mean:24h", lead_time, delta_t=delta_t)
```

Both functions accept scalar or array targets. Negative lead times remain valid
requirements when the window extends before initialization.

## Grouped Execution

The fetch implementation splits qualified labels into source names and modifiers,
then privately groups variables that can share one fetch and one reduction:

```python
variables = ["u10m:mean:1day", "v10m:mean:1day", "t2m:max:1day"]
groups = _group_time_statistics(variables, None)
# {"mean:24h": ("u10m", "v10m"), "max:24h": ("t2m",)}
```

The fetch layer iterates over unique groups, not individual variables. It fetches
each variable block over the required coordinates and applies the statistic once:

```python
for modifier, group in groups.items():
    times = source_times(modifier, valid_time, delta_t)
    block = source(times, group)
    result = apply_time_statistic(block, modifier, valid_time, delta_t)
```

This loop belongs inside `fetch_data`. The assembled output keeps the requested
qualified labels in its `variable` coordinate even though the source receives base
variable names.

## Reduction

`apply_time_statistic()` derives the exact half-open window from a scalar target and
source cadence before reducing an entire variable block. Values outside the window
are excluded. Missing or duplicate required coordinates raise an error rather than
producing a partial statistic. It uses `lead_time` when present, otherwise `time`;
callers may override the dimension for custom datetime or timedelta coordinates.
Xarray dispatch preserves NumPy, CuPy, or Dask execution without an implicit device
transfer.

```python
result = apply_time_statistic(
    block,
    "mean:24h",
    target=valid_time,
    delta_t=np.timedelta64(6, "h"),
)
```

Custom block reductions use the same window syntax:

```python
register_time_statistic("range", lambda x, dim: x.max(dim) - x.min(dim))
```

Custom reductions must preserve all non-time dimensions, including `variable`.
Cross-variable derived quantities are outside this utility.

## Required Optimizations

- Cache parsed modifiers
- Fetch variables with identical modifiers together
- Reduce once per unique variable block, not once per variable
- Avoid host transfers and repeated concatenation
- Assemble the final result once

Future implementations may cache complete fetch plans, order variables for contiguous
GPU selection, or stream associative reductions without changing the public syntax.

## Metadata

`time_statistic_metadata()` normalizes a modifier into method, window, offset, and
interval-closure fields. Durations remain `np.timedelta64`; conversion to a storage
format belongs at the eventual serialization boundary. Generated results should
additionally record CF-compatible time bounds and `cell_methods` when supported.

An Xarray result may mirror its qualified coordinate labels in one concise attribute:

```python
array.attrs["earth2studio_statistics"] = {
    "t2m:mean:1day": "mean:1day",
    "t2m:mean:1week": "mean:1week",
    "t2m:mean:1month": "mean:1month",
}
```

The coordinate label carries quantity identity through model handshakes; the
attribute provides convenient structured lookup and serialization metadata.

## Non-goals

- Modifying `fetch_data`
- Inferring cadence from irregular coordinates
- Defining cross-variable diagnostics
- Implementing streaming or fused GPU kernels in the first version
