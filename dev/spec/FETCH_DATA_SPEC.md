# Earth2Studio Fetch Data Utility

## Goal

`earth2studio.data.fetch_data` adapts analysis and forecast sources to a field
`xr.DataArray` with consistent time, lead-time, and variable dimensions. It owns
source dispatch, output assembly, device placement, and the order in which
regridding is applied. Native data sources own temporal aggregation; `fetch_data`
forwards supported labels and preserves their metadata.

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
    member: int | str | Sequence | MemberArray | None = None,
) -> xr.DataArray:
    ...
```

- `time` contains initialization timestamps; `lead_time` contains offsets from
  each timestamp. Both must be nonempty one-dimensional NumPy arrays of datetime
  and timedelta values respectively, without `NaT`.
- `variable` is a nonempty one-dimensional array of requested labels, optionally
  qualified with a temporal statistic.
- `device` accepts CPU or CUDA destinations. CUDA output requires CuPy.
- `target_grid` is the grid to return fields on, typically
  `infer_grid(model.input_coords())`; use `resolve_grid` for a registered name.
  Without it, fields stay on the source grid.
- `regridder` is a `Regridder` to apply (see [REGRID_SPEC.md](REGRID_SPEC.md)).
  Choosing a method, or a registered one with `resolve_regridder`, is the
  caller's job.
- `member` names one member or distinct members to fetch from an ensemble
  source, in the source's own labels (see [Ensemble Members](#ensemble-members)).
  Other sources reject it with `ValueError`.

There is no separate `metadata` or `statistics` input. Temporal quantities are
declared in variable labels. `target_grid` replaces the former `interp_to`, and
`regridder` replaces `interp_method`. Sampling cadence belongs to the source.

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

1. Fetch requested fields, including source-computed temporal statistics.
2. Assemble requested labels and their statistics metadata on the source grid.
   Cast floating fields to float32; other dtypes pass through.
3. Move the result to `device` if it is not already there.
4. Map onto the target grid on `device`.

Models consume float32, and sources may build float64 arrays (for example with
a bare `np.empty`), so `fetch_data` owns the cast. A DataArray from a source called
directly keeps the source's dtype.

Regridding requested through `fetch_data` follows reduction. A `RegriddedSource`
regrids whatever its wrapped source returns, after any temporal reduction.

### Source Backing and Device

Data sources accept a `device` keyword, default `"cpu"`, and expose it as a `device`
attribute holding a `torch.device`. On a CUDA device, field sources return
CuPy-backed DataArrays and dataframe sources return cuDF frames. The source
protocol is unchanged, and a source without the attribute is a CPU source.
`RegriddedSource` and the composites forward the attribute; composites compare
devices with the CUDA index made explicit, so `cuda` matches the current device. `fetch_data` accepts
either backing and moves data only in step 3. Reduction backing is implementation
specific: the native helpers currently reduce NumPy fields before source
device conversion.

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

## Ensemble Members

Ensemble forecasts and stored ensemble runs supply one field per member, and an
ensemble run may start each member from a different one: for example four seeded
branches from each of 50 IFS ENS members, member `m` starting from forecast
member `m // 4 + 1`. Members are therefore a request coordinate, like
`lead_time`, on the sources that have them:

```python
IFS_ENS()(time, variable)                     # unchanged: the configured member
IFS_ENS()(time, variable, member=[3, 4])      # leading ensemble dimension: 3, 4
IFS_ENS()(time, variable, member=3)           # leading ensemble dimension: 3
fetch_data(IFS_ENS(), time, variable, member=[3, 4])
```

- Ensemble sources follow `EnsembleDataSource` or `EnsembleForecastSource` in
  `earth2studio/data/base.py`: a `DataSource` or `ForecastSource` whose
  `__call__` and `fetch` also take an optional `member` keyword after their
  other arguments. `member` takes a label, a list of labels or a `MemberArray`,
  as `time` and `variable` take a value, list or array. Given, even as one
  label, the result has a leading `ensemble` dimension holding those members in
  request order; omitted, the source returns the member it was built with,
  without the dimension, so an ensemble source still serves every `DataSource`
  or `ForecastSource` caller. `prep_member_inputs` normalizes the keyword.
- The keyword is `member` and the dimension `ensemble`, the dimension name
  runner outputs, IO backends and `InferenceOutputSource` already use: `member`
  selects along `ensemble` as `time` and `variable` select along theirs.
- A source is an ensemble source when its `__call__` signature contains `member`,
  mirroring forecast detection. Wrappers that forward requests through `*args`
  declare it with a boolean `is_ensemble` property; `RegriddedSource` reports
  its source's. The protocols document the contract and serve type checking; a
  runtime-checkable protocol checks method names only, so `isinstance` cannot
  tell ensemble sources from other sources and detection does not use it.
- `CompositeSource` and `CompositeForecastSource` are ensemble sources when any
  component is, through `is_ensemble`. They pass `member` to those components,
  which must share member labels, and repeat other components' fields for every
  member. Without such a component, they reject `member`.
- Labels are the source's own: `IFS_ENS`, `IFS_ENS_FX` and `AIFS_ENS_FX` use `0`
  for the control and `1` and up for perturbed members, `GEFS_FX` uses `gec00`
  and `gep01` to `gep30`, and `InferenceOutputSource` uses the stored `ensemble`
  coordinate. Any subset, including all members, is requested by listing its
  labels. Constructor members, such as `IFS_ENS(member=k)`, remain the default
  for calls without the keyword.
- `fetch_data` forwards `member` to every source call, with qualified labels
  alike, so a source reduces each member's own samples. It raises if the source
  returns no `ensemble` dimension, and selects members in request order. Labels
  must be distinct; callers repeat members after fetching.

## Temporal Statistics

`fetch_data` discovers methods through
`source.lexicon.TIME_STATISTICS` and passes supported qualified labels unchanged.
The source aggregates as described in
[TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md#source-responsibilities):

```python
variables = np.array(["t2m", "t2m:mean:24h", "tp:sum:0h:6h"])
```

For analysis sources, the source receives valid timestamps `time + lead_time` and
anchors windows to them. For forecast sources, it receives initializations and
leads and reduces over leads within each initialization. A source that does not
support a requested method raises `ValueError` before any source calls. Invalid source windows
raise before field downloads; `fetch_data` propagates source errors.

Native adapters use `plan_statistics` to parse labels and plan sample coordinates,
and `reduce_statistics` to fetch them through a source coroutine and reduce each
target's transient sample stack through the registered statistic into the output.
Plain variables keep the source's ordinary path. Sources must validate coordinates
before fetching and propagate failures. Shared reduction imposes no sampling cadence.

Source results must record native reductions in `earth2studio_statistics` as
specified by the source contract. `fetch_data` preserves these entries and trusts
native capability declarations
rather than independently validating each returned metadata entry. Duplicate
normalized quantities are rejected before any source calls. Raw backend identifiers
containing `::` pass through without statistic parsing. Metadata does not record
`delta_t`: source sampling may vary within a window.

Wrappers forward qualified labels. `RegriddedSource` passes them to its source and
regrids the reduced field. The composites route a label by its base name, the text
before the first `:`, so a component providing `t2m` also receives
`t2m:mean:24h`. Native capability discovery unwraps `RegriddedSource` and takes the
intersection of component methods for composites. A method must therefore be
supported by every component to be requested through `fetch_data`.

## Output Contract

- Return one `xr.DataArray`, ordered as `[time, lead_time, variable, ...]`, with
  remaining source dimensions retained. Lead coordinates use nanosecond precision.
  With `member`, a leading `ensemble` dimension comes first, labeled by the
  requested members in order.
- Without a regridder, preserve source spatial coordinates, auxiliary geometry,
  array name, and source attributes. Do not infer a grid, reconstruct grid
  metadata, or relabel geometry.
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
are outside this contract. Native reduction is described in the time-statistics spec.

Tests must cover both source types, multiple initializations and leads, mixed
instantaneous/statistical labels forwarded unchanged, rejection of unsupported
statistics and duplicate quantities, output metadata, native spatial
geometry, CPU/CUDA placement, regridding after reduction, compositions forwarding
qualified labels, sources returning device arrays, and ensemble members through both
source types, temporal statistics, `RegriddedSource` and the composites. Source
aggregation is tested with the shared helper and with each source that supports
statistics. The fetch regressions live in `test/data/test_fetch_data_xarray.py` and
`test/data/test_data_utils.py`, member selection by each ensemble source in
its own test file, with regridders in `test/grids/test_regrid.py` and window
semantics in `test/utils/test_time_statistics.py`.
