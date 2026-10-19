# Earth2Studio Fetch Data Utility

## Goal

`earth2studio.data.fetch_data` adapts analysis and forecast sources to a field
`xr.DataArray` with consistent time, lead-time, and variable dimensions. It owns
source dispatch, temporal-statistic requests, output assembly, device placement, and
the order in which regridding is applied.

This contract reflects the API in [PR #1170](https://github.com/NVIDIA/earth2studio/pull/1170).
Temporal windows follow [TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md); spatial
geometry follows [GRID_SPEC.md](GRID_SPEC.md) and regridding follows
[REGRID_SPEC.md](REGRID_SPEC.md).

## Interface

```python
def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device | str = "cpu",
    target_grid: GridDefinition | str | None = None,
    regridder: Regridder | str | None = None,
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
- `target_grid` is the grid to return fields on, typically
  `infer_grid(model.input_coords())`. Without it, fields stay on the source grid.
- `regridder` is a `Regridder` to apply, or a registered method name resolved for
  the source and target grids (see [REGRID_SPEC.md](REGRID_SPEC.md)).

There is no separate `metadata` or `statistics` input. Temporal quantities are
declared in variable labels. `target_grid` replaces the former `interp_to`, and
`regridder` replaces `interp_method`.

### Regridding

`fetch_data` maps the field the source returns; it never inspects how the source
produced it, so a `RegriddedSource` is fetched like any other source.

| Arguments | Behavior |
| --- | --- |
| Neither | Fields stay on the source grid |
| `regridder` object | Applied; with `target_grid`, it must produce that grid |
| `target_grid` | Inferred source grid mapped onto it, cautiously, as below |
| `target_grid` and a method name | As above, but with the named method |

Mapping onto `target_grid` infers the source grid from the fetched field with
`infer_grid` and raises `ValueError` if it cannot. Then:

1. Fields already on the target pass through.
2. A target that is an exact crop of the source, on rectilinear or projected grids
   with the same CRS, is selected by label without interpolation.
3. Otherwise the method's regridder is constructed with `resolve_regridder` and
   applied, with an info log. Without a method name, `fetch_data` uses
   `"nearest"`, limited to the source grid's resolution
   (`max_dist_km="auto"`), and raises `ValueError` if any target point lies
   beyond the source. Naming a method opts in to filling such points.

Resolving a method constructs a regridder on every call. Callers fetching
repeatedly on one pair of grids should build a `Regridder` once and pass it.

### Order of Operations

1. Fetch from the source on its native grid.
2. Apply temporal reductions on the backing the source returned.
3. Move the result to `device` if it is not already there.
4. Map onto the target grid on `device`.

A `RegriddedSource` regrids each fetched sample before step 2. Nearest and
bilinear regridding commute with `mean` and `sum`, but bilinear `max` and `min` do
not.

### Source Backing and Device

Data sources accept a `device` keyword, default `"cpu"`, and expose it as a `device`
attribute holding a `torch.device`. On a CUDA device, field sources return
CuPy-backed DataArrays and dataframe sources return cuDF frames. The source
protocol is unchanged, and a source without the attribute is a CPU source.
`RegriddedSource` and `CompositeSource` forward the attribute. `fetch_data` accepts
either backing and moves data only in step 3.

## Source Dispatch

A source whose `__call__` signature contains `lead_time` is a forecast source.
Wrappers that forward requests through `*args`, such as `RegriddedSource` and
`CompositeSource`, declare this instead with a boolean `is_forecast` property:

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
- Without a regridder, preserve source spatial coordinates, auxiliary geometry,
  array name, and source attributes. Do not infer a grid, reconstruct grid
  metadata, or relabel geometry. Groups must agree on shared coordinates when
  concatenated.
- With a regridder, spatial coordinates and grid metadata follow its target grid
  under `R3` and `R4`; other coordinates and attributes are preserved.
- Record each performed reduction under its requested variable label in
  `earth2studio_statistics`, using the dictionary returned by
  `time_statistic_metadata(modifier)`. Preserve existing statistical metadata for
  unqualified source variables that pass through.
- Remove coordinate-signature markers `earth2studio_kind`,
  `earth2studio_schema_version`, and `earth2studio_dynamic_dims` from the field.
- Return NumPy-backed values on CPU and CuPy-backed values on the selected CUDA
  device. Device conversion occurs after fetching and temporal reduction, and
  before regridding.

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

Bounds selection, a single coordinate-system request object, fetch-plan caching,
and streaming are outside this contract.

Tests must cover both source types, multiple initializations and leads, mixed
instantaneous/statistical labels, grouped source requests, output ordering and
metadata, invalid or incomplete windows, native spatial geometry, CPU/CUDA
placement, regridding after reduction, deferred and opaque compositions, and
sources returning device arrays. The fetch regressions live in
`test/data/test_fetch_data_xarray.py` and `test/data/test_data_utils.py`, with
regridders in `test/grids/test_regrid.py` and window semantics in
`test/utils/test_time_statistics.py`.
