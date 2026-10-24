# Earth2Studio Fetch Data Utility

## Goal

`earth2studio.data.fetch_data` adapts analysis and forecast sources to a field
`xr.DataArray` with consistent time, lead-time, and variable dimensions. It owns
source dispatch, output assembly, device placement, and the order in which
regridding is applied. Native data sources own temporal aggregation; `fetch_data`
forwards supported labels and preserves their metadata. Explicit `delta_t` selects
generic reduction instead.

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
    target_grid: GridDefinition | None = None,
    regridder: Regridder | None = None,
    delta_t: np.timedelta64 | None = None,
) -> xr.DataArray:
    ...
```

- `time` contains initialization timestamps; `lead_time` contains offsets from
  each timestamp. Both must be nonempty one-dimensional NumPy arrays of datetime
  and timedelta values respectively, without `NaT`.
- `variable` is a nonempty one-dimensional array of requested labels, optionally
  qualified with a temporal statistic.
- `delta_t` is the sampling cadence for generic temporal reductions. Omit it to
  use native source statistics; provide it to override native handling.
- `device` accepts CPU or CUDA destinations. CUDA output requires CuPy.
- `target_grid` is the grid to return fields on, typically
  `infer_grid(model.input_coords())`; use `resolve_grid` for a registered name.
  Without it, fields stay on the source grid.
- `regridder` is a `Regridder` to apply (see [REGRID_SPEC.md](REGRID_SPEC.md)).
  Choosing a method, or a registered one with `resolve_regridder`, is the
  caller's job.

There is no separate `metadata` or `statistics` input. Temporal quantities are
declared in variable labels. `target_grid` replaces the former `interp_to`, and
`regridder` replaces `interp_method`. Native sampling cadence belongs to the source;
`delta_t` remains available for the generic fallback.

### Regridding

`fetch_data` maps the field the source returns; it never inspects how the source
produced it, so a `RegriddedSource` is fetched like any other source.

| Arguments | Behavior |
| --- | --- |
| Neither | Fields stay on the source grid |
| `regridder` | Applied; with `target_grid`, it must produce that grid |
| `target_grid` only | Inferred source grid mapped onto it, cautiously, as below |

Mapping onto `target_grid` infers the source grid from the fetched field with
`infer_grid` and raises `ValueError` if it cannot. Then:

1. Fields already on the target pass through.
2. A target that is an exact crop of the source, on rectilinear or projected grids
   with the same CRS, is selected by label without interpolation.
3. Otherwise a nearest-neighbor regridder is resolved with
   `max_dist_km="auto"` and `fill_value=None`, and applied with an info log. The
   regridder raises `ValueError` if any target point lies beyond the source's
   resolution, and `fetch_data` re-raises it rather than fill; a caller wanting
   fills passes a `Regridder` configured for them.

Step 3 constructs a regridder on every call. Callers fetching repeatedly on one
pair of grids, or wanting another method, should build a `Regridder` once and pass
it. The helpers behind steps 1 and 2, `same_geometry` and `select_subset`, are
public in `earth2studio.grids`.

### Order of Operations

1. Fetch native statistics or fetch base fields and apply generic temporal reductions.
2. Assemble requested labels and their statistics metadata on the source grid.
3. Move the result to `device` if it is not already there.
4. Map onto the target grid on `device`.

Regridding requested through `fetch_data` follows reduction. A `RegriddedSource`
regrids whatever its wrapped source returns: native statistics are reduced first,
but generic fallback calls the wrapper for base samples and reduces them afterwards.

### Source Backing and Device

Data sources accept a `device` keyword, default `"cpu"`, and expose it as a `device`
attribute holding a `torch.device`. On a CUDA device, field sources return
CuPy-backed DataArrays and dataframe sources return cuDF frames. The source
protocol is unchanged, and a source without the attribute is a CPU source.
`RegriddedSource` and the composites forward the attribute; composites compare
devices with the CUDA index made explicit, so `cuda` matches the current device. `fetch_data` accepts
either backing and moves data only in step 3. Reduction backing is implementation
specific: the native helpers currently reduce NumPy fields before source
device conversion, while generic block reductions preserve their input backing.

## Source Dispatch

A source whose `__call__` signature contains `lead_time` is a forecast source.
Wrappers that forward requests through `*args`, such as `RegriddedSource`, declare
this instead with a boolean `is_forecast` property. `CompositeSource` and
`CompositeForecastSource` have explicit signatures:

| Source | Call | Temporal interpretation |
| --- | --- | --- |
| `DataSource` | `source(valid_time, variable)` | Absolute valid timestamps |
| `ForecastSource` | `source(time, lead_time, variable)` | Leads within each initialization |

For instantaneous analysis requests, fetch at `time + lead_time` for each lead,
then restore the requested initialization timestamps and add a `lead_time`
dimension. For instantaneous forecasts, request initialization times and leads
directly. Select returned coordinates in request order.

## Temporal Statistics

Without `delta_t`, `fetch_data` discovers methods through
`source.lexicon.TIME_STATISTICS` and passes supported qualified labels unchanged.
The source aggregates as described in
[TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md#source-responsibilities):

```python
variables = np.array(["t2m", "t2m:mean:24h", "tp:sum:0h:6h"])
```

For analysis sources, the source receives valid timestamps `time + lead_time` and
anchors windows to them. For forecast sources, it receives initializations and
leads and reduces over leads within each initialization. A source that does not
support a requested method requires an explicit `delta_t`. Invalid source windows
raise before field downloads; `fetch_data` propagates source errors.

With explicit `delta_t`, native dispatch is disabled. `fetch_data` groups base
variables by normalized modifier, plans the union of required coordinates with
`source_times` or `source_lead_times`, fetches base fields, and reduces each target
using `apply_time_statistic`. Generic reduction rejects base fields already marked
as aggregated in `earth2studio_statistics`. It materializes the requested samples;
native sources are preferable for long windows and large grids.

Native adapters use `plan_statistics` to parse labels and plan sample coordinates,
and `reduce_statistics` to fetch them through a source coroutine and reduce each
target's transient sample stack through the registered statistic into the output.
Plain variables keep the source's ordinary path. Sources must validate coordinates
before fetching and propagate failures. Shared reduction imposes no sampling cadence.

Source results must record native reductions in `earth2studio_statistics` as
specified by the source contract. `fetch_data` preserves these entries and creates
metadata for generic reductions; it currently trusts native capability declarations
rather than independently validating each returned metadata entry. Duplicate
normalized quantities are rejected before any source calls. Raw backend identifiers
containing `::` pass through without statistic parsing.

Wrappers forward qualified labels. `RegriddedSource` passes them to its source and
regrids the reduced field. The composites route a label by its base name, the text
before the first `:`, so a component providing `t2m` also receives
`t2m:mean:24h`. Native capability discovery unwraps `RegriddedSource` and takes the
intersection of component methods for composites. Heterogeneous composites may
therefore need an explicit `delta_t` even when one component supports a method.

## Output Contract

- Return one `xr.DataArray`, ordered as `[time, lead_time, variable, ...]`, with
  remaining source dimensions retained. Lead coordinates use nanosecond precision.
- Without a regridder, preserve source spatial coordinates, auxiliary geometry,
  array name, and source attributes. Do not infer a grid, reconstruct grid
  metadata, or relabel geometry. Groups must agree on shared coordinates when
  concatenated.
- With a regridder, spatial coordinates and grid metadata follow its target grid
  under `R3` and `R4`; other coordinates and attributes are preserved.
- Preserve the `earth2studio_statistics` the source records, including metadata
  for unqualified source variables that are already aggregated.
- Remove coordinate-signature markers `earth2studio_kind`,
  `earth2studio_schema_version`, and `earth2studio_dynamic_dims` from the field.
- Return NumPy-backed values on CPU and CuPy-backed values on the selected CUDA
  device. Device conversion occurs after fetching, and before regridding.

The former `(tensor, coords)` return and `legacy` switch are removed. Consumers
access `field.data` and `field.coords`; tensor-based consumers can explicitly use
`prep_data_array`.

## Example

Given an analysis source supporting `t2m:mean:24h`:

```python
field = fetch_data(
    source,
    time=np.array([np.datetime64("2024-01-02T00")]),
    variable=np.array(["t2m", "t2m:mean:24h"]),
    lead_time=np.array([np.timedelta64(0, "h")]),
)
```

The output contains the instantaneous field and preceding-day mean on the source
grid. A six-hourly analysis source averages valid times -24 through -6 hours.
For `GFS_FX`, request a lead of at least 24 hours for this trailing window: at
lead 24h, hourly samples are leads 0 through 23h of the same initialization.
A trailing window at lead zero is rejected because its required leads are negative.

## Deferred Work and Verification

Bounds selection, a single coordinate-system request object, and fetch-plan caching
are outside this contract. Native reduction is described in the time-statistics spec;
generic fallback still materializes sample arrays.

Tests must cover both source types, multiple initializations and leads, mixed
instantaneous/statistical labels forwarded unchanged, generic cadence override,
rejection of duplicate quantities and generic re-reduction, output metadata, native spatial
geometry, CPU/CUDA placement, regridding after reduction, compositions forwarding
qualified labels, and sources returning device arrays. Source aggregation is tested
with the shared helper and with each source that supports statistics. The fetch regressions live in
`test/data/test_fetch_data_xarray.py` and `test/data/test_data_utils.py`, with
regridders in `test/grids/test_regrid.py` and window semantics in
`test/utils/test_time_statistics.py`.
