# Earth2Studio Time Statistics

## Goal

Describe temporal source requirements and apply reductions to Xarray data without
changing the variable coordinate. Fetch integration and metadata accessors are
separate work.

## Declaration

Variables remain strings. A mapping adds a temporal modifier to selected variables:

```python
statistics = {
    "u10m": "mean:24h",
    "v10m": "mean:24h",
    "t2m": "max:24h",
}
```

A single modifier applies to every requested variable. Variables absent from a
mapping remain instantaneous.

## Modifiers

Two compact forms are supported:

- `mean:24h` describes `T - 24h <= t < T`
- `mean:-12h:+12h` describes `T - 12h <= t < T + 12h`

`T` is valid time. Intervals are half-open, and the source cadence must divide the
window exactly. Built-in methods are `mean`, `sum`, `min`, and `max`.

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

The fetch implementation normalizes modifiers and privately groups variables that
can share one fetch and one reduction:

```python
groups = _group_time_statistics(variables, statistics)
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

This loop belongs inside `fetch_data`; callers continue to provide only the simple
statistics declaration.

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

## Non-goals

- Modifying `fetch_data`
- Inferring cadence from irregular coordinates
- Defining cross-variable diagnostics
- Implementing streaming or fused GPU kernels in the first version
