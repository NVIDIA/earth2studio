# Earth2Studio Fetch Data Utility

## Goal

`earth2studio.data.fetch_data` adapts analysis and forecast sources to a field
`xr.DataArray` with consistent time, lead-time, and variable dimensions. It owns
source dispatch, temporal-statistic requests, output assembly, and device placement.

This contract reflects the API in [PR #1170](https://github.com/NVIDIA/earth2studio/pull/1170).
Temporal windows follow [TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md); spatial
geometry follows [GRID_SPEC.md](GRID_SPEC.md).

## Interface

```python
def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device | str = "cpu",
    target_grid: CoordinateSystem | GridDefinition | str | None = None,
    regridder: str = "nearest",
    *,
    delta_t: np.timedelta64 | None = None,
) -> xr.DataArray:
    ...
```

- `time` contains initialization timestamps; `lead_time` contains offsets from
  each timestamp. Both must be nonempty one-dimensional NumPy arrays of datetime
  and timedelta values respectively, without `NaT`.
- `variable` is a nonempty one-dimensional array of requested labels.
- `delta_t` specifies the source sampling cadence for statistics. When omitted,
  use `source.time_step`; statistical requests fail if neither is available.
  Instantaneous requests do not require a cadence.
- `device` accepts CPU or CUDA destinations. CUDA output requires CuPy.
- `target_grid` and `regridder` are reserved spatial arguments and currently have
  no effect.

Grid objects own CRS and spatial subset semantics through `GridDefinition.crs`
and `GridDefinition.subset_indexers()`. Bounds and their CRS belong to that grid
selection interface, not to `fetch_data`. Callers describe the desired output
geometry through `target_grid`.

There is no separate `metadata` or `statistics` input. Temporal quantities are
declared in variable labels. `target_grid` replaces `interp_to`; `regridder`
replaces `interp_method`.

## Source Dispatch

A source whose `__call__` signature contains `lead_time` is a forecast source:

| Source | Call | Temporal interpretation |
| --- | --- | --- |
| `DataSource` | `source(valid_time, variable)` | Absolute valid timestamps |
| `ForecastSource` | `source(time, lead_time, variable)` | Leads within each initialization |

For instantaneous analysis requests, fetch at `time + lead_time` for each lead,
then restore the requested initialization timestamps and add a `lead_time`
dimension. For instantaneous forecasts, request initialization times and leads
directly. Select returned coordinates in request order.

## Temporal Statistics

Split each label at its first `:` into a base variable and optional modifier:

```python
variables = np.array(["t2m", "t2m:mean:24h", "tp:sum:0h:6h"])
```

Unqualified requests pass through source values. Qualified requests fetch the base
variable and apply the specified reduction. Returned labels preserve the exact
requested spelling; normalized modifiers determine grouping and duplicate checks.

1. Group base variables by normalized modifier, with one group for unqualified
   requests. Reject empty base names and duplicate normalized quantities, including
   aliases such as `t2m:mean:24h` and `t2m:mean:1day`.
2. Validate every modifier, cadence, and window before making source calls.
3. For each statistical group, fetch the unique union of required samples in one
   source call. For analysis sources, plan absolute timestamps across all requested
   time/lead pairs. For forecasts, plan lead times across the requested leads and
   fetch them for every initialization.
4. Reduce each requested window over `time` for analyses or `lead_time` for
   forecasts. Never average across forecast initializations. Missing or duplicate
   reduction coordinates are errors; negative leads are passed to the source when
   required by a window.
5. Assemble groups in the original variable order. Reject reduction of a base
   variable already marked as aggregated in source `earth2studio_statistics`.

Windows are left-closed and right-open. For example, `mean:24h` at a six-hour
cadence uses offsets -24, -18, -12, and -6 hours. `mean:0h:24h` instead uses
0, 6, 12, and 18 hours. Source availability and support for those timestamps or
leads remain the source's responsibility.

## Output Contract

- Return one `xr.DataArray`, ordered as `[time, lead_time, variable, ...]`, with
  remaining source dimensions retained. Lead coordinates use nanosecond precision.
- Preserve source spatial coordinates, auxiliary geometry, array name, and source
  attributes. Do not infer a grid, reconstruct grid metadata, or relabel geometry
  to match a target. Groups must agree on shared coordinates when concatenated.
- Record each performed reduction under its requested variable label in
  `earth2studio_statistics`, using the dictionary returned by
  `time_statistic_metadata(modifier)`. Preserve existing statistical metadata for
  unqualified source variables that pass through.
- Remove coordinate-signature markers `earth2studio_kind`,
  `earth2studio_schema_version`, and `earth2studio_dynamic_dims` from the field.
- Return NumPy-backed values on CPU and CuPy-backed values on the selected CUDA
  device. Device conversion occurs after fetching and temporal reduction.

The former `(tensor, coords)` return and `legacy` switch are removed. Consumers
access `field.data` and `field.coords`; tensor-based consumers can explicitly use
`prep_data_array`.

## Example

Given an analysis or forecast source providing instantaneous `t2m`:

```python
field = fetch_data(
    source,
    time=np.array([np.datetime64("2024-01-02T00")]),
    variable=np.array(["t2m", "t2m:mean:24h"]),
    lead_time=np.array([np.timedelta64(0, "h")]),
    delta_t=np.timedelta64(6, "h"),
)
```

The output contains the instantaneous field and preceding-day mean on the source
grid. An analysis source supplies absolute timestamps; a forecast source supplies
leads -24 through -6 hours for the mean at initialization time.

## Deferred Work and Verification

Spatial mapping, interpolation, bounds selection, and target-grid validation are
deferred. The regridding helper is a pass-through. A single coordinate-system
request object, fetch-plan caching, and streaming are also outside this contract.

Tests must cover both source types, multiple initializations and leads, mixed
instantaneous/statistical labels, grouped source requests, output ordering and
metadata, invalid or incomplete windows, native spatial geometry, and CPU/CUDA
placement. The fetch regressions live in `test/data/test_fetch_data_xarray.py` and
`test/data/test_data_utils.py`, with window semantics in
`test/utils/test_time_statistics.py`.
