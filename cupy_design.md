# Earth2Studio CuPy DataArray API Migration

**Status:** Draft
**Target window:** September-November 2026
**Implementation model:** Incremental changes on `main`

## Summary

Earth2Studio currently passes model and workflow data as a pair containing a
`torch.Tensor` and a coordinate dictionary. This design proposes making
`xarray.DataArray` the canonical data container, backed by NumPy on CPU and CuPy
on CUDA devices.

The migration must not interrupt ongoing development or require a long-lived
migration branch. Existing models, workflows, and IO backends will continue to
work through adapters while native DataArray support is introduced in stages.
Public workflow signatures, including `run.deterministic`, remain unchanged.

## Why Change

The migration is motivated by four outcomes: an easier user API, better
machine-readable context for agents, simpler and more general protocols, and a
higher-performance path from data providers to GPUs.

### 1. Easier for users

Xarray is widely used across the weather, climate, and scientific Python
communities. Users can work with familiar dimensions, coordinates, selection,
interpolation, and metadata APIs instead of learning Earth2Studio's separate
tensor and coordinate conventions.

A single DataArray is also harder to misuse. Operations that select, reorder, or
reshape data carry the associated coordinates with them rather than requiring
the caller to update a second object in lockstep.

### 2. Easier for agents

A DataArray can package dimensions, coordinates, attributes, grid descriptions,
and statistical metadata with the data it describes. These fields can be
serialized independently from the array values and exposed through an extended
backend as a compact, LLM-consumable description.

An agent can therefore reason about variables, units, grids, valid times, lead
times, ranges, and provenance without inspecting a tensor or reconstructing
context from a separate coordinate dictionary. This creates a consistent
foundation for agent-authored workflows and tool calls.

### 3. Simpler, more general protocols

The current tensor-coordinate pair requires every protocol to accept, validate,
transform, and return two coupled objects. Moving both into one DataArray reduces
maintenance complexity and enables similar protocols across data sources,
prognostic models, diagnostic models, workflow utilities, and IO backends.

Components can specialize their computation while sharing the same container
contract. Torch models remain supported through boundary adapters rather than
forcing the core orchestration API to be Torch-specific.

### 4. Higher-performance data movement

CuPy-backed DataArrays open a direct path for data streams and storage backends
to load data onto GPUs without first materializing a Torch tensor on the CPU.
Once data is GPU-resident, CuPy and Torch can share memory through DLPack where
the layout permits it.

This reduces unnecessary host staging and repeated conversions while allowing
data loading, preprocessing, model execution, and output handling to remain on
the GPU. The design does not require model kernels to be rewritten in CuPy.

## Core Data Structure

The core object is an `xarray.DataArray`. Xarray provides the labeled container;
NumPy or CuPy provides the underlying array implementation.

```text
xr.DataArray
├── data       NumPy/CuPy runtime data or a private shape-only signature backend
├── dims       Ordered dimension names
├── coords     Index and auxiliary coordinates
├── attrs      Global and Earth2Studio metadata
└── name       Optional field name
```

This distinction is important: Earth2Studio is not introducing a new array
class. It is defining a consistent DataArray contract and extending it with a
small `.e2s` accessor for operations that require Earth2Studio semantics.

### Existing `.e2s` utilities

The current implementation in `earth2studio.utils.cupy` already provides the
foundation:

- `from_torch` packs a Torch tensor and `CoordSystem` into one DataArray
- `is_cupy` reports whether the backing data is GPU-resident
- `as_cupy` moves or exposes the backing data as CuPy
- `as_numpy` moves or exposes the backing data as NumPy
- `to_torch` returns the legacy tensor and coordinate representation
- `batch` packs selected dimensions into one batch dimension
- `unbatch` restores dimensions and coordinates recorded by `batch`

```python
array = from_torch(tensor, coords)
array = array.e2s.as_cupy()

packed = array.e2s.batch(("time", "ensemble"))
restored = packed.e2s.unbatch()

tensor, coords = restored.e2s.to_torch()
```

These methods centralize conversion and structural bookkeeping instead of
duplicating it across data sources, models, workflows, and IO backends.

### Coordinate declaration

Model authors should not allocate or initialize empty NumPy arrays to declare a
coordinate contract. Earth2Studio should expose a top-level constructor that accepts
the xarray dimensions and coordinates directly:

```python
import earth2studio as e2s

signature = e2s.coord_array(
    dims=(
        "batch",
        "time",
        "lead_time",
        "variable",
        "hrrr_y",
        "hrrr_x",
    ),
    coords={
        "lead_time": [np.timedelta64(0, "h")],
        "variable": variables,
    },
    dynamic=("batch", "time"),
    grid="hrrr-conus-3km",
)

assert isinstance(signature, xr.DataArray)
assert signature.data.nbytes == 0
assert signature.e2s.dynamic_dims == ("batch", "time")
```

`coord_array()` infers fixed sizes from dimension coordinates, a known grid key, or an
explicit `sizes` mapping. It uses a private xarray-compatible duck array that stores
only shape and dtype metadata and raises on attempted materialization. There is no
field-data allocation; only the small DataArray and coordinate metadata objects
exist.

The optional `statistics` argument accepts a mapping from existing variable labels to
compact statistic strings. It uses the same resolver as `set_statistic()` and stores
the normalized descriptions in the DataArray attributes.

The `grid` resolver checks Earth2Studio's known grids first and then attempts to parse
the value as a PyProj CRS. Known keys provide complete CRS and geometry metadata;
explicit CRS inputs require the caller's coordinates, sizes, transform, or topology.

[Xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html)
attach to DataArray instances, not the `xarray` module. Its [backend plugin
interface](https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html) is
for file IO through `open_dataset(..., engine=...)`, not for adding constructors.
Therefore `xr.e2s.coord_array()` would require monkeypatching xarray. The single
supported constructor should be `earth2studio.coord_array()`; after construction,
all Earth2Studio operations use the normal `array.e2s` accessor.

### Proposed metadata utilities

The existing `.e2s` accessor should own validated Earth2Studio metadata. Grid
declarations use ordinary xarray coordinates and attributes. Statistic modifiers use
one namespaced DataArray attribute. The accessor provides validation and convenience
without introducing `GridMetadata` or `StatisticMetadata` classes.

```python
array = array.e2s.set_grid("hrrr-conus-3km")
grid = array.e2s.get_grid()
crs = array.e2s.crs

array = array.e2s.set_statistic(
    "u10m",
    method="mean",
    window="24h",
)
statistic = array.e2s.get_statistic("u10m")

description = array.e2s.describe(include_values=False)
```

Setter-style methods return a new DataArray rather than mutating the caller.
Metadata getters return validated, JSON-serializable structures. `describe`
produces a compact representation for logs, agents, APIs, and backend catalogs
without serializing the full data values.

The description should include:

- Shape, dtype, dimensions, and device
- Coordinate names, dimensions, ranges, and units
- Grid type, spatial dimensions, CRS, and latitude/longitude coordinate names
- Variable names, units, standard names, and provenance
- Optional temporal statistic modifier for each variable
- Time range, lead-time range, and sample cadence
- DataArray and coordinate attributes that are safe to serialize

Computing statistics from the array values is separate and opt-in. Reading
statistic metadata must not trigger a GPU reduction or move data to the CPU.

### Metadata schema

Earth2Studio-specific keys should be exported constants whose values begin with
`earth2studio_`. Global facts such as schema version and grid identity are flat
DataArray attributes. Metadata that participates in xarray alignment uses auxiliary
coordinates. The small statistic modifier map is the deliberate exception: it lives
in one namespaced attribute keyed by the plain string-valued `variable` coordinate and
is managed entirely by the accessor.

Earth2Studio metadata adds a machine-friendly structure without adding implicit
coordinates to a model signature. Standard fields such as `units`, `standard_name`,
coordinate bounds, and `cell_methods` remain valid when present. CF export adapters
may add a grid-mapping coordinate when a target format requires one.

## Temporal Statistic Modifiers

A statistic is a small modifier on an existing variable. It does not create a new
variable name: `u10m` remains `u10m` whether it is instantaneous, a daily mean, or a
daily maximum. The `variable` coordinate always remains a plain array of strings.
Statistics are independent of the spatial grid.

The public API needs only the variable, method, and trailing window:

```python
array = array.e2s.set_statistic("u10m", method="mean", window="24h")
modifier = array.e2s.get_statistic("u10m")
```

A model can declare the same modifier directly in its coordinate signature:

```python
def input_coords(self) -> xr.DataArray:
    return e2s.coord_array(
        dims=("batch", "time", "variable", "lat", "lon"),
        coords={"variable": np.array(["u10m", "t2m"])},
        dynamic=("batch", "time"),
        grid="latlon-0.25deg",
        statistics={"u10m": "mean:24h"},
    )
```

Like `grid=`, this keeps the declaration short while packing a normalized description
into one DataArray attribute:

```python
array.coords["variable"].values
# np.array(["u10m", "t2m"])

array.attrs["earth2studio_statistics"]
# {
#     "u10m": {
#         "modifier": "mean:24h",
#         "method": "mean",
#         "window": "PT24H",
#         "start_offset": "-PT24H",
#         "end_offset": "PT0S",
#         "closed": "left",
#     }
# }
```

Variables absent from the mapping are instantaneous. Canonical modifiers use the
compact form `method:window`, for example `mean:24h`, `max:24h`, or `sum:6h`. The
constructor and accessor validate the variable, parse the modifier, normalize the
duration, and define the interval in valid-time space. Users never construct the
expanded mapping directly.

An optional three-part form declares custom offsets relative to the target valid time:

```python
centered = e2s.coord_array(
    dims=("batch", "time", "variable", "lat", "lon"),
    coords={"variable": ["u10m"]},
    dynamic=("batch", "time"),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)
```

Here `mean:-12h:+12h` means a centered 24-hour mean over
`T - 12h <= t < T + 12h`. It expands to `start_offset="-PT12H"`,
`end_offset="PT12H"`, and `window="PT24H"`. The two-part `mean:24h` form remains
shorthand for `mean:-24h:0h`.

The statistic string defines offsets from `T`, where `T` is always `valid_time`. The
coordinate declaration determines how valid time is represented:

```python
# Data or analysis: valid_time = time
e2s.coord_array(
    dims=("time", "variable", "lat", "lon"),
    coords={"variable": ["u10m"]},
    dynamic=("time",),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)

# Forecast: valid_time = time + lead_time
e2s.coord_array(
    dims=("time", "lead_time", "variable", "lat", "lon"),
    coords={"lead_time": [np.timedelta64(6, "h")], "variable": ["u10m"]},
    dynamic=("time",),
    grid="latlon-0.25deg",
    statistics={"u10m": "mean:-12h:+12h"},
)
```

Thus the same compact string works for both data and forecast sources without
duplicating storage-specific dimension names in the string.

If every variable is instantaneous, the attribute is omitted entirely.
`get_statistic()` returns `None` for a variable without a modifier.

The window is trailing and half-open: `mean:24h` at target valid time `T` means
`mean(u10m[t] for T - 24h <= t < T)`. `T` is the valid timestamp for which the
statistic is requested; it is not another array dimension.

For forecasts, `time` is the initialization or reference time and `lead_time` is the
forecast offset. `valid_time` is the physical time represented by a forecast value:

```python
time = np.datetime64("2026-08-28T00:00")
lead_time = np.timedelta64(6, "h")
valid_time = time + lead_time
# np.datetime64("2026-08-28T06:00")
```

Observations and analyses normally have no `lead_time`; their `time` coordinate is
already the valid time.

The statistic resolver selects source values by their valid times, then chooses the
physical reduction dimension from the concrete request:

- Varying `time` with no `lead_time`, or one fixed lead, reduces over `time`
- One fixed initialization with varying `lead_time` reduces over `lead_time`
- Varying `time` and `lead_time` is ambiguous and raises until the caller selects one
  initialization or one lead

The static descriptor stores offsets in valid-time space, but neither a redundant
anchor field nor a physical reduction dimension. Once the target valid time and
request shape are known, the accessor resolves the source dimension and actual bounds:

```python
resolved = request.e2s.resolve_statistic(
    "u10m",
    valid_time=np.datetime64("2026-08-28T06:00"),
)

resolved["start"]
# np.datetime64("2026-08-27T06:00")

resolved["end"]
# np.datetime64("2026-08-28T06:00")
```

For a request containing one initialization and multiple leads,
`resolved["source_dimension"]` is `"lead_time"`; for analysis data sampled across
times, it is `"time"`. This keeps model declarations compact while making the selected
dimension, target valid time, offsets, closure, and concrete bounds inspectable.

Downstream fetching remains direct:

```python
modifier = request.e2s.get_statistic("u10m")
source_times = request.e2s.statistic_times("u10m", valid_time)
source = fetch_data(data_source, source_times, "u10m")
result = source.e2s.apply_statistic("u10m", modifier)
```

The accessor only exposes entries for strings currently present in the `variable`
coordinate. Earth2Studio validation and IO prune stale attribute keys after generic
xarray selection or coordinate replacement.

When unpacking for IO, Earth2Studio converts the modifier to CF `cell_methods` and
concrete time bounds. The initial design supports one statistic per variable label in
a packed DataArray. Requesting instantaneous and aggregated forms of the same variable
together should use separate arrays until there is a demonstrated need for another
representation.

## Grid Metadata and Coordinate Systems

A DataArray separates ordered tensor dimensions from physical coordinates. Every
spatial DataArray must carry a CRS that PyProj can normalize. EPSG authority strings,
WKT2, PROJJSON, and other user inputs use `CRS.from_user_input()`; CF mappings use
`CRS.from_cf()`. Earth2Studio must not invent projection names such as
`"hrrr-conus-3km"` and pass them as a CRS.

`pyproj` is the appropriate CRS parser and transformation library. Its [CF
support](https://pyproj4.github.io/pyproj/stable/build_crs_cf.html) can read user input
through `CRS.from_user_input`, export a CF grid mapping through `CRS.to_cf`, export
coordinate-system axis metadata through `CRS.cs_to_cf`, and reconstruct a CRS through
`CRS.from_cf`. [PROJ recommends WKT2 or authority
identifiers](https://proj.org/en/stable/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems)
over PROJ strings because conversion to PROJ strings can lose information.

The accessor normalizes the CRS to `pyproj.CRS` and writes a scalar CF grid-mapping
coordinate containing `crs_wkt` and available CF projection parameters. Coordinate
transforms use `always_xy=True`; `DataArray.dims` remains the authoritative tensor
order regardless of CRS axis order.

### Grid resolver and registry

`coord_array(grid=...)` and `array.e2s.set_grid(...)` use the same resolver:

1. Normalize aliases and check Earth2Studio's exact custom grids.
2. Check parameterized Earth2Studio grid families such as regular lat/lon and
   HEALPix.
3. Attempt `pyproj.CRS.from_user_input()` or `CRS.from_cf()`.
4. Raise with close known-key suggestions if neither path resolves.

Initial canonical keys should be concise but unambiguous:

| Canonical key | Short alias | Resolved definition |
| --- | --- | --- |
| `hrrr-conus-3km` | `hrrr` | HRRR Lambert CRS, 1059 by 1799 shape, transform |
| `latlon-0.25deg` | `latlon025` | EPSG:4326, north-to-south latitude, 0-360 longitude |
| `fcn-global-0.25deg` | `fcn` | EPSG:4326, 720 by 1440, south pole excluded |
| `healpix-l6-nested` | `hpx6` | Spherical geographic CRS, `nside=64`, 49,152 pixels |
| `healpix-l10-nested` | `hpx10` | Spherical geographic CRS, `nside=1024`, 12,582,912 pixels |

The registry should also support canonical families such as
`latlon-{resolution}deg` and `healpix-l{level}-{ring|nested}` rather than enumerating
every possible resolution. Short aliases always resolve to a documented canonical
key; for example, `hpx6` means NESTED ordering.

Registry entries are private mappings or resolver functions, not public grid classes.
They contain the normalized CRS, ordered spatial dimensions, shape, transform or
topology, units, and a lazy coordinate generator. Registration must reject names that
PyProj already recognizes so a custom key cannot shadow an EPSG or other authority
identifier.

After resolution, Earth2Studio writes the complete CRS and geometry or topology onto
the DataArray. Serialization and handshakes therefore remain self-describing and do
not depend on the registry being available later.

For `coord_array()`, a known key fills missing fixed spatial sizes and generated
coordinates. For an existing runtime DataArray, `set_grid()` only validates and
annotates matching dimensions and shape; it never reshapes or regrids data.

A CRS still does not identify a discrete grid. Exact identity also requires ordered
spatial dimensions, shape, axis units, and coordinates, a transform, or topology
metadata. `grid_id` is therefore optional and only provides a registry/cache key for
known geometry.

### Regular latitude-longitude

A regular global grid uses its latitude and longitude dimensions directly and an
EPSG-supported geographic CRS:

```python
latlon = e2s.coord_array(
    dims=("batch", "time", "variable", "lat", "lon"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    grid="latlon-0.25deg",
)
```

The tensor order is exactly `(..., "lat", "lon")`; EPSG axis conventions do not
reorder it.

### Lambert conformal

HRRR uses a custom Lambert conformal CRS that PyProj supports even though it has no
EPSG authority code. Its registry entry hides the long projection definition and
stores shape and transform without allocating coordinate arrays:

```python
hrrr = e2s.coord_array(
    dims=("batch", "time", "variable", "hrrr_y", "hrrr_x"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    grid="hrrr-conus-3km",
)
```

An unregistered grid uses the same API with explicit PyProj input and geometry:

```python
regional_lcc = pyproj.CRS.from_user_input(
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=-97 "
    "+datum=WGS84 +units=m +type=crs"
)

regional = e2s.coord_array(
    dims=("batch", "time", "variable", "y", "x"),
    coords={"variable": variables},
    dynamic=("batch", "time"),
    sizes={"y": ny, "x": nx},
).e2s.set_grid(
    regional_lcc,
    spatial_dims=("y", "x"),
    transform=regional_transform,
)
```

At runtime, one-dimensional projected x/y coordinates and optional two-dimensional
latitude/longitude coordinates can be materialized as normal xarray coordinates.

### HEALPix

HEALPix model tensors can use one spatial dimension containing a pixel index. The CRS
defines the geographic frame, while HEALPix topology requires `nside` and the RING or
NESTED ordering. The official pixel count is `12 * nside**2`.

```python
def healpix_signature(level: int) -> xr.DataArray:
    return e2s.coord_array(
        dims=("batch", "time", "variable", "hpx"),
        coords={"variable": variables},
        dynamic=("batch", "time"),
        grid=f"healpix-l{level}-nested",
    )


hpx_level_6 = healpix_signature(6)  # 49,152 pixels
hpx_level_10 = healpix_signature(10)  # 12,582,912 pixels
```

The signature omits the large `hpx` index coordinate because its values are implied
by `range(12 * nside**2)`. Runtime arrays may attach `hpx`, `lat(hpx)`, and `lon(hpx)`
coordinates when selection or visualization needs them. PROJ also implements a
`+proj=healpix` map projection, but that projection does not encode the discrete
`nside`, pixel index, or ordering and therefore cannot replace the topology metadata.

The grid hash should be computed from canonical WKT2 or PROJJSON, ordered spatial
dimensions, shape, units, transform or topology parameters, and coordinate digests
where needed. The optional ID is excluded from the hash. Embedded CRS and geometry
metadata remain authoritative when reading serialized arrays.

Because grid interpretation is part of the core DataArray protocol, Stage 1 should
either promote the existing `pyproj` dependency from the `data` extra to core or make
only CRS parsing lazy and raise a focused installation error. Promoting it to core is
the simpler long-term behavior.

The same structure supports regular latitude/longitude grids, projected grids,
rotated-pole grids, unstructured meshes, and station dimensions while preserving
the coordinate relationships needed for selection, interpolation, serialization,
and agent inspection.

## Goals

1. Make `xarray.DataArray` the canonical internal data representation.
2. Use NumPy-backed DataArrays on CPU and CuPy-backed DataArrays on CUDA.
3. Preserve existing user-facing workflow signatures.
4. Continue supporting legacy `(torch.Tensor, CoordSystem)` components during a
   multi-release transition.
5. Support both `DataSource` and `ForecastSource`, including lead times.
6. Preserve coordinate values, dimensions, ordering, names, and attributes.
7. Use zero-copy CuPy-Torch conversions where the backing layout permits it.
8. Keep `main` continuously usable throughout the migration.
9. Warn users before changing default return behavior.
10. Maintain accurate test coverage in both API modes.

## Non-goals

- Removing Torch or rewriting model kernels in CuPy
- Removing the legacy API during this project
- Changing the public signatures of built-in workflows
- Replacing model coordinate requirements such as `input_coords()` in the first
  migration
- Adding gradient-preserving CuPy-Torch conversion; `requires_grad=True` remains
  unsupported until it has a separate design
- Migrating DataFrame-based sparse data sources in the first data stage
- Supporting every xarray operation on CuPy-backed arrays
- Introducing a long-lived legacy or DataArray development branch

## Design Principles

### DataArray is canonical internally

Data should be fetched, mapped, batched, passed between workflow components, and
prepared for IO as a DataArray. Conversion should occur only at a boundary with
a legacy component.

### Compatibility belongs at boundaries

Legacy models and IO backends should not be rewritten solely to participate in
the transition. Small dispatch helpers convert at their boundaries and return to
the canonical DataArray representation immediately afterward.

### Public workflows remain stable

Users continue calling workflows as they do today:

```python
run.deterministic(time, nsteps, model, data, io)
```

The workflow selects its internal representation from centralized runtime
configuration. No `array_api` argument is added to each workflow.

### Capabilities are explicit

Native DataArray components declare their supported API. Dispatch must not infer
support by catching `TypeError` or inspecting parameter names.

### The migration stays on `main`

Each stage is delivered through normal, short-lived pull requests. Every merged
change preserves the production legacy path and expands the DataArray path. This
design-document branch is not an implementation or support branch.

## Current State

- `DataSource` and `ForecastSource` already return `xr.DataArray`
- `fetch_data(..., legacy=True)` returns `(torch.Tensor, CoordSystem)`
- `fetch_data(..., legacy=False)` returns a NumPy- or CuPy-backed DataArray but
  does not yet support interpolation
- Prognostic and diagnostic protocols accept and return tensors plus coordinates
- IO backends accept tensors plus coordinates
- `DataArray.e2s` supports NumPy/CuPy conversion, Torch conversion, batching, and
  unbatching
- `earth2studio.utils.cupy.from_torch` wraps legacy model output as a DataArray

## Runtime Configuration

One environment variable controls public compatibility behavior:

```bash
EARTH2STUDIO_ARRAY_API=legacy
EARTH2STUDIO_ARRAY_API=xarray
```

The value is resolved at affected public API boundaries, not at import time.
Private helpers receive an explicit resolved mode or operate on DataArrays only.

### Transition behavior

| Configuration | Before default switch | After default switch |
| --- | --- | --- |
| Unset | Legacy behavior and one warning | DataArray behavior |
| `legacy` | Legacy behavior | Legacy behavior |
| `xarray` | DataArray behavior | DataArray behavior |
| Other value | Configuration error | Configuration error |

The default must not change until all five migration stages are complete. Legacy
removal requires a separate proposal and deprecation period.

### Resolution pseudocode

```python
ArrayAPI = Literal["legacy", "xarray"]

_DEFAULT_ARRAY_API: ArrayAPI = "legacy"
_warned_about_array_api = False


class Earth2StudioArrayAPIFutureWarning(FutureWarning):
    pass


def resolve_array_api(*, warn: bool = True) -> ArrayAPI:
    value = os.getenv("EARTH2STUDIO_ARRAY_API")
    if value is None:
        if warn:
            warn_array_api_once()
        return _DEFAULT_ARRAY_API
    if value not in {"legacy", "xarray"}:
        raise ValueError(
            "EARTH2STUDIO_ARRAY_API must be 'legacy' or 'xarray'"
        )
    return cast(ArrayAPI, value)
```

## Migration Warnings

The first affected public call with an unset environment variable emits one
`Earth2StudioArrayAPIFutureWarning` per process. The warning is not emitted at
import time, from private conversion helpers, or during each forecast step.

The warning should state:

```text
Earth2Studio currently returns torch.Tensor and CoordSystem objects by default.
The default will change to xarray.DataArray in Earth2Studio <target release>.
Set EARTH2STUDIO_ARRAY_API=xarray to test the new behavior, or set it to legacy
to retain the current behavior during the transition. See <migration URL>.
```

The final warning must name a target release, link to a migration guide, and use
an appropriate `stacklevel` to identify the user's call site. Explicitly setting
`legacy` suppresses the pre-switch warning. After the default changes, legacy
mode should warn only when a separate removal release has been scheduled.

This follows established transition patterns:

- [SQLAlchemy 2.0 migration warnings](https://docs.sqlalchemy.org/en/20/changelog/migration_20.html)
- [pandas Copy-on-Write migration](https://pandas.pydata.org/docs/user_guide/copy_on_write.html)
- [Zarr 3 migration and runtime format selection](https://zarr.readthedocs.io/en/stable/user-guide/v3_migration/)
- [Python warning categories](https://docs.python.org/3/library/warnings.html)

## Target Architecture

```text
DataSource / ForecastSource
            |
            v
    xr.DataArray on CPU
            |
      coordinate mapping
            |
      device placement
            |
            v
 xr.DataArray[NumPy or CuPy]
            |
   +--------+---------+
   |                  |
   v                  v
native component   legacy adapter
DataArray API      DataArray -> Torch + CoordSystem
   |                  |
   |               legacy component
   |                  |
   |              Torch + CoordSystem -> DataArray
   +--------+---------+
            |
            v
      xr.DataArray
            |
      native or legacy IO adapter
```

## Stage 1: Data Fetching

Data sources are already DataArray-native. The first stage makes that format
canonical inside `fetch_data` and brings the DataArray path to feature parity
with the legacy path.

### Public API

The current `legacy` parameter remains a temporary explicit override so existing
callers continue working. Changing its default to `None` allows the environment
variable to select behavior without adding another public parameter.

```python
@overload
def fetch_data(..., legacy: Literal[True]) -> tuple[torch.Tensor, CoordSystem]: ...


@overload
def fetch_data(..., legacy: Literal[False]) -> xr.DataArray: ...


@overload
def fetch_data(
    ...,
    legacy: None = None,
) -> tuple[torch.Tensor, CoordSystem] | xr.DataArray: ...


def fetch_data(
    source: DataSource | ForecastSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = ZERO_LEAD_TIME,
    device: torch.device = "cpu",
    interp_to: CoordSystem | None = None,
    interp_method: str = "nearest",
    legacy: bool | None = None,
) -> tuple[torch.Tensor, CoordSystem] | xr.DataArray:
    array = _fetch_data_array(
        source, time, variable, lead_time, interp_to, interp_method
    )
    array = _place_data_array(array, device)

    mode = resolve_array_api() if legacy is None else (
        "legacy" if legacy else "xarray"
    )
    return array.e2s.to_torch() if mode == "legacy" else array
```

### Canonical fetch pipeline

```python
def _fetch_data_array(
    source,
    time,
    variable,
    lead_time,
    interp_to,
    interp_method,
) -> xr.DataArray:
    if isinstance_forecast_source(source):
        array = source(time, lead_time, variable)
    else:
        arrays = []
        for lead in lead_time:
            value = source(offset_time(time, lead), variable)
            arrays.append(assign_lead_time(value, time, lead))
        array = xr.concat(arrays, dim="lead_time")

    array = validate_data_array(array)
    if interp_to is not None:
        array = interpolate_data_array(array, interp_to, interp_method)
    return array


def _place_data_array(array: xr.DataArray, device) -> xr.DataArray:
    device = torch.device(device)
    if device.type == "cuda":
        return array.e2s.as_cupy(device.index)
    return array.e2s.as_numpy()
```

Interpolation initially occurs before CUDA placement unless a GPU-native
implementation is explicitly supported and tested. Stage 1 must support current
regular-grid and curvilinear-grid behavior, nearest and linear interpolation,
and one- and two-dimensional target coordinates.

### Stage 1 completion criteria

- DataSource and ForecastSource produce equivalent results
- Lead-time assembly is identical in both modes
- DataArray interpolation matches the legacy path
- CPU returns NumPy-backed data and CUDA returns CuPy-backed data
- Legacy conversion preserves values and coordinate ordering
- `coord_array()` allocates no field data and supports required structural operations
- Existing callers that omit `legacy` retain legacy behavior
- The warning is emitted once when mode is implicit

## Stage 2: Prognostic Models

Existing prognostic models retain the current protocol. Native models may opt in
to a DataArray execution protocol through an explicit capability marker.

Coordinate requirement methods remain unchanged during this migration. The
execution payload changes, not how a model describes required coordinates.

### Native prognostic pseudocode

```python
class NativePrognostic:
    array_api = "xarray"

    def input_coords(self) -> CoordSystem:
        ...

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        ...

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        tensor, coords = x.e2s.to_torch()
        output = self.model(tensor)
        return from_torch(output, self.output_coords(coords))

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        yield x
        while True:
            x = self(x)
            yield x
```

Models implemented with Torch may still be DataArray-native at their public
boundary. The internal model forward pass can use zero-copy Torch conversion.

### Legacy prognostic adapter

```python
def prognostic_iterator(model, x: xr.DataArray) -> Iterator[xr.DataArray]:
    if getattr(model, "array_api", "legacy") == "xarray":
        yield from model.create_iterator(x)
        return

    tensor, coords = x.e2s.to_torch()
    for output, output_coords in model.create_iterator(tensor, coords):
        yield from_torch(output, output_coords, attrs=x.attrs)
```

The built-in deterministic and ensemble workflow signatures do not change. Only
their internal orchestration moves to DataArrays.

## Stage 3: Diagnostic Models

Diagnostic models use the same explicit native capability and legacy adapter
pattern as prognostic models.

### Native diagnostic pseudocode

```python
class NativeDiagnostic:
    array_api = "xarray"

    def input_coords(self) -> CoordSystem:
        ...

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        ...

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        tensor, coords = x.e2s.to_torch()
        output = self.model(tensor)
        return from_torch(output, self.output_coords(coords))
```

### Diagnostic dispatch

```python
def call_diagnostic(model, x: xr.DataArray) -> xr.DataArray:
    if getattr(model, "array_api", "legacy") == "xarray":
        return model(x)

    tensor, coords = x.e2s.to_torch()
    output, output_coords = model(tensor, coords)
    return from_torch(output, output_coords, attrs=x.attrs)
```

Prognostic-to-diagnostic workflows remain DataArray-native between adapters, so
mixed native and legacy components can be composed.

## Stage 4: IO and Supporting Utilities

IO and workflow utilities should accept DataArrays internally while preserving
legacy backends through dispatch.

### Native IO pseudocode

```python
class NativeIOBackend(Protocol):
    array_api: Literal["xarray"]

    def add_array(self, template: xr.DataArray, array_name: str | list[str]) -> None:
        ...

    def write(self, x: xr.DataArray, array_name: str | list[str]) -> None:
        ...
```

### IO dispatch

```python
def write_array(io, x: xr.DataArray, array_name) -> None:
    if getattr(io, "array_api", "legacy") == "xarray":
        io.write(x, array_name)
        return

    tensor, coords = x.e2s.to_torch()
    io.write(tensor, coords, array_name)
```

Coordinate selection and ordering should use DataArray operations where they are
supported. Earth2Studio utilities remain responsible for operations that xarray
cannot perform safely on CuPy data.

Utilities included in this stage:

- Coordinate mapping and selection
- Batching and unbatching
- Perturbations
- Workflow transforms
- Device placement
- Statistics used directly by workflows
- IO allocation and writes

Stage 4 must be complete by mid-October 2026.

## Stage 5: Documentation and Rollout

Documentation must include:

- DataArray fetching examples
- CuPy device-placement examples
- Native prognostic and diagnostic model authoring
- Legacy component compatibility
- IO backend authoring
- Environment-variable behavior
- Migration warning interpretation
- Zero-copy guarantees and copy-producing operations
- Troubleshooting for missing CuPy or unsupported xarray operations

Documentation and rollout readiness must be complete by the end of November
2026. The default changes only in a later scheduled release after both modes have
passed CI for at least two normal releases.

## Continuous Delivery Plan

No stage is developed in isolation from `main`.

1. Add internal DataArray behavior behind the resolver.
2. Keep legacy behavior as the default.
3. Merge small changes with tests for both active modes.
4. Expand DataArray CI coverage as each subsystem migrates.
5. Fix forward rather than maintaining a parallel implementation branch.
6. Do not change the default until implementation and documentation are complete.

Normal features may continue landing during the migration. New code should avoid
introducing additional tensor-coordinate coupling when a DataArray path is
available.

## Test Strategy

Every migrated subsystem is tested in both modes.

### Required parity

- Values and dtypes
- Dimension names and order
- Coordinate values and attributes
- DataArray name and attributes where applicable
- CPU and CUDA placement
- DataSource and ForecastSource lead-time behavior
- Regular and curvilinear coordinate mapping
- Legacy and native component composition
- Copy versus shared-memory expectations
- Warning count, category, message, and call-site stack level

### CI progression

1. Run the complete suite with legacy behavior.
2. Run Stage 1 data tests with `EARTH2STUDIO_ARRAY_API=xarray`.
3. Add prognostic, diagnostic, and IO tests as their stages complete.
4. Run the complete core workflow suite in both modes by the end of Stage 4.
5. Treat unexpected migration warnings as test failures.

## Timeline

| Stage | Target | Deliverable |
| --- | --- | --- |
| 1. Data fetching | Early September 2026 | Canonical fetch pipeline and warnings |
| 2. Prognostics | Late September 2026 | Native protocol and legacy adapter |
| 3. Diagnostics | Early October 2026 | Native protocol and mixed composition |
| 4. IO and utilities | Mid-October 2026 | End-to-end DataArray core workflows |
| 5. Documentation | End of November 2026 | Migration guide and rollout readiness |

## Risks

### Xarray operations may materialize host arrays

Not every xarray operation preserves CuPy residency. Supported operations must be
tested explicitly. Utilities should fail clearly rather than silently copying to
CPU.

### Coordinate attributes can be lost at legacy boundaries

`CoordSystem` does not represent coordinate attributes. Adapters must preserve
attributes when dimensions survive a legacy call and document when preservation
is impossible.

### Capability detection can become ambiguous

The `array_api` marker is required for native components. Runtime exception
handling is not a valid detection strategy.

### Environment variables are process-wide

Private helpers must not mutate the environment. Tests should use isolated
configuration and restore environment state.

### Warnings can overwhelm iterative workflows

Warnings are emitted once at public boundaries, never per model step or per data
request inside a workflow.

## Open Questions

1. Should `array_api` become a shared typed capability enum or remain a string
   class attribute during the transition?
2. Should native IO be introduced as a public protocol in Stage 4 or remain an
   internal capability until the default changes?
3. Which release will switch the default after the November documentation
   milestone?
4. How long will explicit legacy mode be supported after the default changes?
5. Which coordinate attributes must be guaranteed across legacy adapters?

## Decision Summary

- DataArray is the canonical internal representation
- NumPy backs CPU data and CuPy backs CUDA data
- Existing workflow signatures remain unchanged
- Existing component protocols remain supported through adapters
- Native components opt in explicitly with `array_api = "xarray"`
- One environment variable controls public compatibility behavior
- Users receive a single visible warning before the default changes
- Implementation lands continuously on `main`
- Legacy removal is outside this three-month project