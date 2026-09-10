# Time Statistics Research Notes

**Status:** Non-normative background for temporal-statistic design
**Last updated:** September 2026

## Purpose

This note records the references, tradeoffs, and performance considerations behind
Earth2Studio's temporal-statistic utilities. The contract is defined in
`dev/spec/TIME_STATISTICS_SPEC.md`.

Earth2Studio must request enough source samples to compute temporal products for
analysis and forecast sources, then reduce NumPy-, CuPy-, or Dask-backed Xarray data.
Variable names must remain plain strings, and common operations must execute over
variable blocks rather than through Python loops over individual fields.

## References

### CF Conventions

[CF cell methods](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.pdf)
distinguish instantaneous values from means, sums, minima, and maxima. They describe
the reduction with `cell_methods`, the represented interval with coordinate bounds,
and the original sample spacing with an optional `interval` clause.

Earth2Studio follows the same conceptual separation:

- The method identifies the operation
- Start and end offsets identify the represented interval
- `delta_t` identifies required source sampling
- Generated metadata can be translated to CF bounds and `cell_methods`

The compact Earth2Studio string is an input request syntax, not a replacement for
CF metadata on serialized results.

### Xarray

[Xarray reductions](https://docs.xarray.dev/en/stable/api/dataarray.html) operate by
named dimension and preserve every other dimension. This permits one reduction over
an array shaped like `(time, variable, y, x)` rather than one call per variable.

[Xarray duck arrays](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html)
allow the same labeled operation to dispatch to NumPy, CuPy, or another compatible
backend. Accessing `.data` preserves the backend, while `.values` converts to NumPy.

[Xarray resampling](https://docs.xarray.dev/en/stable/time-series.html) returns an
intermediate object containing grouping state before a reduction is selected. That
pattern is useful for repeated general resampling, but Earth2Studio already receives
the complete operation in a compact modifier. Exposing another per-variable object
would duplicate the declaration without simplifying the fetch handshake.

### Pandas

[Pandas resampling](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html)
defines explicit interval closure, labels, origins, and offsets. Its aggregation API
accepts column-to-operation mappings, showing that a simple declarative mapping can
represent heterogeneous reductions while the implementation groups work internally.

### CuPy

[CuPy reductions](https://docs.cupy.dev/en/stable/reference/generated/cupy.mean.html)
accept an axis and reduce all remaining dimensions in one operation. The important
optimization boundary is therefore a complete variable block, not a single field.
Separate kernels are generally required for different reduction methods, but fields
sharing a method and window can execute together.

### Dask

[Dask reductions](https://docs.dask.org/en/stable/generated/dask.array.reduction.html)
support chunk, combine, and aggregate phases. Tree depth can be controlled with
`split_every`, trading task overhead against intermediate memory and communication.

[Xarray's Dask guidance](https://docs.xarray.dev/en/stable/user-guide/dask.html)
emphasizes choosing chunks for downstream operations and avoiding unnecessary
rechunking. Temporal windows should therefore use Xarray's existing backend-aware
reductions first, leaving chunk tuning to the source and execution environment.

## Original Design Review

### Strengths

- One parser defined half-open windows consistently
- Analysis timestamps and forecast lead times shared valid-time semantics
- A registry allowed custom reduction functions
- Xarray reductions preserved NumPy and CuPy backends
- Serializable metadata retained the method and exact offsets

### Weaknesses

- `resolve_time_statistics()` returned one object per variable
- Callers had to understand and iterate those objects
- The object repeated information already present in the modifier string
- `reduce(array, dimension)` required boilerplate even for standard arrays
- The API encouraged per-variable reduction instead of grouped execution
- Public parsing objects constrained future internal planning changes

## Refined Design

The refined API keeps strings and mappings at the public boundary:

```python
statistics = {"u10m": "mean:24h", "v10m": "mean:24h", "t2m": "max:24h"}
```

It exposes direct primitives for the three required operations:

```python
times = source_times(modifier, valid_time, delta_t)
leads = source_lead_times(modifier, lead_time, delta_t)
result = apply_time_statistic(block, modifier)
```

Parsing and variable grouping are private implementation details. Parsed windows are
cached. `apply_time_statistic()` infers `lead_time` or `time` and applies a registered
function to an entire block. The future fetch layer owns the loop over unique groups,
so callers do not manage a plan or a collection of statistic objects.

## Performance Path

### Initial Implementation

- Group fields by normalized modifier
- Fetch each unique field/window group once
- Reduce all fields in that group with one Xarray operation
- Reassemble groups once after reduction
- Keep execution on the existing array backend

This changes Python dispatch and backend launches from approximately one per variable
to one per unique modifier. When all fields share a statistic, one operation reduces
the complete array.

### Future Optimizations

- Cache grouped plans for repeated forecast steps
- Order fetched variables by group to keep selections contiguous
- Preallocate final output and write each group into its destination slice
- Deduplicate overlapping source requests across windows
- Add streaming accumulators for sum, mean, minimum, and maximum
- Add specialized fused kernels when profiling justifies them
- Pass backend-specific reduction options without changing modifier syntax

Streaming can reduce peak memory from the full temporal window to one source slice
plus accumulator state. Sum, count, minimum, and maximum are naturally composable;
mean can be finalized from sum and count. Exact behavior for missing values and
precision must be part of any streaming contract.

## Risks and Open Questions

- Fetching a union of all timestamps may over-fetch short-window variables; grouped
  requests avoid this but may duplicate reads for overlapping windows
- Noncontiguous variable selection can allocate a gather buffer on GPUs; request
  ordering should be optimized before adding lower-level kernels
- A DataArray cannot naturally give different variables different time-coordinate
  lengths, so mixed windows are best processed as separate blocks
- Dask performance depends on source chunking; forcing one time chunk may increase
  memory and should not be the default
- Sums may change physical interpretation or units and require metadata policy
- Weighted, irregular, and calendar-relative windows need separate specifications
- Cross-variable diagnostics should not be disguised as temporal statistics

## Decision

Use a declarative variable-to-modifier mapping, direct source-coordinate helpers, and
block-wise reductions. Keep parsed windows and future compiled plans internal. This
is simpler for callers, compatible with CF-style output metadata, and leaves clear
paths to grouped GPU execution, Dask trees, caching, and streaming.
