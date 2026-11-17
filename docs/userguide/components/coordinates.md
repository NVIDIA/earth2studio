# Data and Metadata { #coordinates_userguide }

Creating meaningful data products is a foundation of Earth2Studio. Components
produce data, and rich metadata gives those outputs the context that humans and
agents need to interpret and use them: what quantities they represent, where and
when they apply, and how they were produced.

Earth2Studio uses coordinate systems and metadata structures both to describe
existing data and to declare what each component needs and produces. These shared
descriptions let people and agents inspect data products, prepare inputs, and
connect components into pipelines.

## Interface

### Fields and Signatures { #data_userguide }

Gridded components exchange `xarray.DataArray` fields with values, named dimensions,
coordinate labels, and attributes together. CPU fields typically use NumPy storage;
CUDA fields use CuPy. Values cross component boundaries in physical units, with
normalization and model-specific preprocessing handled inside the model wrapper.

A `CoordinateSystem` is a metadata-only DataArray constructed with `coord_array`.
It describes a field without allocating its values; coordinate labels still occupy
memory. Models return these signatures from `input_coords()`, `forcing_coords()`,
and `output_coords(...)` so callers can plan inputs and output storage before inference.

| Member | What to inspect |
| --- | --- |
| `.dims` | Names and order of the field axes. |
| `.sizes` | Length of each axis. |
| `.coords` | Dimension labels and auxiliary coordinates. |
| `.dtype` | Field-value dtype, separate from coordinate dtypes. |
| `.attrs` | Grid, temporal-statistics, and descriptive metadata. |

Inspect signature metadata rather than `.values`, which is unsupported for these
allocation-free objects. An execution field contains real values accessible through
`.data`. The legacy `CoordSystem` mapping does not capture the full metadata interface.

### Dimensions and Time

Coordinate dtypes describe labels independently of the field-value dtype.

| Name | Coordinate dtype | Meaning |
| --- | --- | --- |
| `time` | NumPy `datetime64` | Absolute time; forecast reference time when paired with leads. |
| `lead_time` | NumPy `timedelta64` | Offset from reference time, including input history. |
| `valid_time` | NumPy `datetime64` | Auxiliary valid time; `time + lead_time` for forecasts. |
| `variable` | String labels | Physical quantity identifiers, including temporal qualifiers. |
| `ensemble` | Declared dtype, often integer or string | Member identity, not array position. |
| `batch` | Declared label dtype, if labelled | Independent samples; may be unlabelled. |
| `lat`, `lon` | Floating-point | Geographic coordinates in degrees, as axes or auxiliaries. |
| `y`, `x` | Numeric or integer indexes | Native spatial coordinates or grid indexes. |
| `hpx`, `face`, `height`, `width` | Integer indexes | HEALPix pixel or layout indexes. |

Use NumPy `datetime64` for absolute times and `timedelta64` for offsets. For example,
an initialization at 00 UTC with a six-hour lead is valid at 06 UTC. A history window
with leads `[-6h, 0h]` describes two input frames relative to the reference time.

Dynamic dimensions are an explicitly declared, zero-sized leading prefix in a
signature. A model can accept concrete leading axes such as `time` and `ensemble`
through that prefix; its remaining dimensions still describe the fixed input layout.
See [Batch Dimension](../advanced/batch.md) for batching patterns.

Time labels locate a field, while a temporal qualifier describes the quantity over
a window. For example, `tp:sum:6h` identifies a six-hour accumulation ending at the
target time. Coordinate constructors derive `earth2studio_statistics` metadata from
qualified labels. See [E2S-03: Time Statistics][time-spec] for window syntax,
source support, and calendar-day semantics.

### Spatial Grids

A `GridDefinition` describes geometry independently of field values or interpolation.
Its `dims` and `shape` describe spatial axes, `coords()` supplies coordinates, and
`crs` identifies the native coordinate reference system where applicable. `topology`,
serializable `attrs`, and `fingerprint()` describe the grid family and identity.
`subset_indexers()` supports spatial selection; `cell_bounds()` supplies cell geometry
when available.

| Definition | Typical dimensions | Geometry |
| --- | --- | --- |
| `LatLonGrid` | `lat, lon` | Rectilinear geographic axes. |
| `ProjectedGrid` | `y, x` | Native projected coordinates with a CRS. |
| `CurvilinearGrid` | `y, x` | Two-dimensional latitude and longitude auxiliaries. |
| `PointGrid` | `x` | Individual geographic sample locations. |
| `HEALPixGrid` | `hpx` or `face, height, width` | Explicit pixel ordering and layout. |

Pass a grid object or registered name to `coord_array` to attach its geometry.
`infer_grid(signature)` recovers a target grid for fetching or regridding. Matching
array shapes alone does not establish matching geography: axis order, CRS, and
HEALPix ordering can differ.

### Attributes and Tabular Schemas

Grid metadata includes geometry descriptions, `earth2studio_crs`, and, for registered
grids, `earth2studio_grid_id`. Descriptive attributes can record units and provenance.
For a mixed-variable field, describe units per variable rather than assigning one
field-level unit to quantities with different meanings.

Sparse observations use pandas or cuDF DataFrames. A model's `FrameSchema` maps
column names to NumPy arrays describing dtype and allowed values: an empty array
leaves values unconstrained, while a nonempty array lists allowed values. These are
column declarations, not example rows. A source's PyArrow `SCHEMA` instead describes
available column types, nullability, and field metadata. See
[Tabular Observations](datasources.md#tabular-observations) for fetching them.

#### Standard DataFrame Fields

These fields form a shared vocabulary, not a requirement to include every column.
The source's `SCHEMA` and the consuming model's `FrameSchema` determine which fields
are available and required. Types below are the suggested PyArrow types from
`E2STUDIO_SCHEMA`; check the source schema for its exact types and nullability.
The units of `observation` depend on `variable`.

| Name | Suggested Arrow type | Meaning |
| --- | --- | --- |
| `time` | `timestamp[ns]` | Observation time or forecast reference time, in UTC. |
| `lat` | `float32` | Latitude in degrees, within [-90, 90]. |
| `lon` | `float32` | Longitude in degrees, within [0, 360]. |
| `elev` | `float32` | Observation elevation above the surface, metres. |
| `observation` | `float32` | Measurement or predicted value of the identified variable. |
| `variable` | `string` | Earth2Studio quantity identifier. |
| `pres` | `float32` | Observation pressure level, Pa. |
| `type`, `class` | `string` | Observation type identifier and class. |
| `source`, `station` | `string` | Per-row source and station identifiers. |
| `station_elev` | `float32` | Station elevation, metres. |
| `quality` | `uint16` | Quality-control marker; 0 is best, 15 is missing. |
| `satellite` | `string` | Satellite platform identifier. |
| `scan_angle` | `float32` | Satellite scan angle, degrees. |
| `sensor_index` | `uint16` | Sensor channel index. |
| `solza`, `solaza` | `float32` | Solar zenith and azimuth angles, degrees. |
| `satellite_za`, `satellite_aza` | `float32` | Satellite zenith and azimuth angles, degrees. |
| `wavenumber` | `float64` | Channel wavenumber, cm^-1. |
| `track_id`, `storm_name`, `basin` | `string` | Track identifier, storm name, and basin code. |
| `season` | `int32` | Tropical cyclone season year. |

Forecast frames can also declare `lead_time` as an Arrow duration and `valid_time`
as a timestamp equal to `time + lead_time`. A DataFrame's row index does not replace
its time, location, or variable columns.

Full contracts are defined in [E2S-01: Core Metadata][metadata-spec],
[E2S-02: Grids][grid-spec], and [E2S-03: Time Statistics][time-spec].

## Usage

### Declare and Inspect a Field

Create a small grid-backed signature with two history frames:

```python
import numpy as np

from earth2studio.grids import LatLonGrid, infer_grid
from earth2studio.utils import coord_array

grid = LatLonGrid(
    np.array([50.0, 49.0, 48.0]),
    np.array([0.0, 1.0, 2.0, 3.0]),
)
signature = coord_array(
    ("batch", "lead_time", "variable", "lat", "lon"),
    {
        "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
        "variable": ["t2m"],
    },
    dynamic=("batch",),
    grid=grid,
)
print(signature.dims)
print(signature.sizes)
target_grid = infer_grid(signature)
```

For a loaded model, inspect `model.input_coords()` the same way. Pass the recovered
grid to `fetch_data(..., target_grid=target_grid)` to prepare matching inputs;
[Data Sources](datasources.md#match-a-model-grid) shows the complete request.

A concrete field on this grid carries actual values:

```python
import xarray as xr

field = xr.DataArray(
    np.full(grid.shape, 280.0, dtype=np.float32),
    dims=grid.dims,
    coords=grid.coords(),
    name="t2m",
    attrs={**grid.attrs, "units": "K", "source": "local-example"},
)
```

### Preserve Meaning Across Transformations

Use `coord_array_like` to derive an output signature when retaining an input's
geometry. For a new spatial domain, build a new grid-backed signature so the
metadata describes the actual crop or transformed grid. After reordering spatial
axes on registered-grid data, remove the stale `earth2studio_grid_id` so grid
inference uses the actual coordinates; see the grid specification's current
[reconstruction guidance](../../specs/E2S-02_GRID_SPEC.md#model-coordinates-to-target-grid).

PyTorch wrappers convert at their computation boundary with `field.e2s.to_torch()`
and restore labelled fields with `from_torch(tensor, output_signature)`. Public
component handoffs remain DataArrays. Signature-only markers are not field metadata;
the conversion helpers handle that distinction.

### Shared CUDA Memory

On Linux CUDA systems with RMM installed, the `auto` and `cupy` backends share memory
pools between Torch, CuPy, and cuDF while preserving existing custom RMM resources.
Import Earth2Studio or select the CuPy backend before initializing Torch CUDA.
Set `EARTH2STUDIO_DISABLE_RMM=1` before import to retain existing allocators.
Non-Linux systems, missing RMM, CPU-only imports, and initial `numpy` or `torch`
backends skip this setup.

RMM's Torch adapter does not honor `Tensor.record_stream()`: keep tensors alive until
their streams finish, or disable RMM. Use RMM instrumentation for pool statistics;
Torch's native cache controls do not manage these shared pools.

## Describe Your Own Component

Start with your inference inputs and outputs: name the axes, variables and units,
history and forecast times, and spatial grid. Declare these with `coord_array` and
use them in the component's coordinate methods. Document any custom labels or axes
so other components and agents can interpret them.

Reuse a built-in grid definition where possible. For a new geometry, implement the
`GridDefinition` interface and optionally register it with `register_grid` for reuse.
For tabular inputs, declare the needed columns and value constraints in a
`FrameSchema`. Check your declarations against a representative field or observation
batch before connecting the component to a pipeline.

The [custom prognostic wrapper guide](prognostic.md#wrap-your-own-prognostic-model)
shows how to give these details to an agent along with your inference code and
reference predictions.

[metadata-spec]: ../../specs/E2S-01_CORE_METADATA_SPEC.md
[grid-spec]: ../../specs/E2S-02_GRID_SPEC.md
[time-spec]: ../../specs/E2S-03_TIME_STATISTICS_SPEC.md
