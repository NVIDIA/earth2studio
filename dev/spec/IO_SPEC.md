# Earth2Studio IO Backend Protocol

## Goal

Write model and data-source DataArrays to storage without converting them to
tensors and coordinate dictionaries, preserving the coordinates and metadata that
make stored fields self-describing. Models, sources and regridders already exchange
`xr.DataArray` (see [MODEL_CONTRACT_SPEC.md](MODEL_CONTRACT_SPEC.md)); this contract
moves IO onto the same payload so drivers write what models yield:

```python
io.add_array(schema)
for y in islice(model.create_iterator(x), nsteps):
    io.write(y)
io.close()
```

This revision also adds lifecycle methods, so the protocol changes once for
DataArray writes and finalization together.

## Interface

```python
@runtime_checkable
class IOBackend(Protocol):
    def add_array(self, schema: xr.DataArray) -> None: ...
    def write(self, x: xr.DataArray) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

- `add_array` creates storage for the arrays described by `schema`: dimension and
auxiliary coordinates, then one array per variable.
- `write` stores `x` at the positions its coordinate labels identify.
- `flush` blocks until every earlier write is visible to readers of the store. The
backend stays writable.
- `close` flushes and releases resources. Later writes raise; repeated `close` calls
are no-ops.

Reading is optional (see Reading). Other backend-specific methods (`__getitem__`,
commits, `to_xarray`) remain on the implementations and are not part of the
protocol. Any store can also be opened with Xarray (see Round Trip).

## Schemas

A schema is a DataArray whose field values are never read: normally an
allocation-free `coord_array` signature, though any DataArray works. `add_array`
reads only its dimensions, coordinates, dtype, name and attributes.

```python
signature = model.output_coords(model.input_coords())  # (batch, lead_time, ...)
schema = output_schema(
    signature,
    leading={"ensemble": members, "time": times},  # replaces the dynamic prefix
    coords={"lead_time": leads},                   # run extent of fixed dims
)
io.add_array(schema)  # (ensemble, time, lead_time, variable, lat, lon)
```

- **Concrete.** Every dimension is nonempty and none is dynamic. Planning
signatures must be concretized first (see `output_schema` under Shared Helpers);
`add_array` raises `ValueError` otherwise.
- **Naming.** With a `variable` dimension, each label becomes one stored array over
the remaining dimensions, in schema order. Without one, the schema's `name` names a
single array; a schema with neither raises `ValueError`. Labels are used
verbatim, including temporal-statistic qualifiers such as `tp:sum:6h` (see Array
Names).
- **dtype.** Arrays take the schema's dtype; `coord_array` defaults to `float32`.
Unwritten positions hold the backend's fill value, NaN for floating dtypes.
- **One extent per dimension.** Arrays in one backend share each dimension's
coordinates. Adding an array whose shared dimension or auxiliary coordinates
differ from the store raises `ValueError`. Re-adding an existing array with the
same dimensions is a no-op, so reopened stores and resumed runs can call
`add_array` unconditionally.
- **Fixed extent.** A store's dimensions do not grow after creation. Appending along
a dimension is out of scope (see Open Questions).

Outputs whose coordinates differ go to separate backends. A multi-slot model's `y` is a tuple of
DataArrays with differing coordinates, so drivers write each slot to its own backend; one backend
may hold several slots only when their shared dimensions agree.

## Writes

`write(x)` locates `x` in the store by coordinate label, never by position.

- `x` has the dimensions of its target arrays in the same order. Missing,
additional or reordered dimensions raise `ValueError`.
- Along each dimension, `x` holds any nonempty subset of the store's labels, in any
order and without duplicates. An unknown label raises `ValueError` before anything
is written. Several `lead_time` entries in one write are allowed, so chunked model
steps write directly.
- `x` writes one array per `variable` label, or the array named `x.name` when it has
no `variable` dimension. Writing to an array that `add_array` did not create
raises `ValueError`; writes never create arrays implicitly.
- Auxiliary coordinates and attributes on `x` are not compared with the store or
written again. `add_array` wrote them once.
- Backends may restrict which subsets they accept and must document it, raising
`ValueError` for unsupported writes. `AsyncZarrBackend`, for example, requires
dimensions outside its parallel dimensions to be written in full.

### Payloads and ownership

`x` may be NumPy-, CuPy- or Torch-adapter-backed on any device. Moving data to the
storage device is the backend's job; callers never convert first.

A backend borrows `x` and never modifies its values, coordinates or attributes. A
backend that returns before the data is stored copies whatever it needs first, so
callers may reuse or release `x` immediately. This mirrors the model ownership rules
and protects rollouts from backends that would otherwise convert units or dtypes in
place.

## Array Names

Stored array names are the `variable` labels, verbatim. Statistic qualifiers carry
quantity identity (see [TIME_STATISTICS_SPEC.md](TIME_STATISTICS_SPEC.md)), so the
name is enough to rebuild `earth2studio_statistics` with `coord_array` or
`coord_array_like` on read. No statistics attribute is stored.

Zarr and netCDF accept `:` in names. Some environments do not: local Zarr stores
create one directory per array, and Windows forbids `:` in file names. A backend
may offer an opt-in name mapping for these cases. A mapped store records the original
label in each array's `earth2studio_variable` attribute so readers can restore it.
Mapping is never the default.

## Metadata

`add_array` persists:

- dimension coordinates with their attributes;
- auxiliary coordinates (curvilinear and projected latitude/longitude, point
geometry) as arrays over their own dimensions. Each data array lists them in a
CF `coordinates` attribute so Xarray opens them as coordinates;
- the schema's attributes on every data array, including grid metadata
(`earth2studio_grid_id`, `earth2studio_crs` and the definition's attributes) and
user metadata such as `units`.

It drops the signature markers `earth2studio_kind`, `earth2studio_schema_version`
and `earth2studio_dynamic_dims`, and `earth2studio_statistics`, which the labels
determine. A format that cannot store an attribute value natively, such as a
dictionary or list in netCDF, stores it as a JSON string.

## Round Trip

After `flush`, opening the store with Xarray (`xr.open_zarr`, `xr.open_dataset`, or
the in-memory dataset) yields, for each array:

- the written values, with fill values at unwritten positions;
- dimension coordinates equal to the schema's: `time` as `datetime64[ns]`,
`lead_time` as `timedelta64[ns]`, and other labels unchanged;
- the schema's auxiliary coordinates and coordinate attributes;
- the persisted attributes above, so `infer_grid` recovers the schema's grid.

## Reading

Backends that can read implement the inverse of `write`:

```python
def read(
    self,
    selection: xr.DataArray | Mapping[Hashable, ArrayLike],
    device: torch.device | str = "cpu",
    dtype: DTypeLike | None = None,
) -> xr.DataArray: ...
```

- `selection` is a schema or field, or an ordered mapping from every dimension to
  its labels. `variable` labels select arrays; without a `variable` dimension, a
  DataArray's name does. Field values are never read.
- Labels may be any subset, in any order, and dimensions may be in any order. The
  result follows the selection's dimension and label order.
- Unknown arrays, labels or dimensions raise `ValueError`.
- The result is NumPy-backed on CPU and CuPy-backed on CUDA, cast to `dtype` when
  given. It carries the stored coordinates and the attributes shared by the
  selected arrays.

Reading is not part of the protocol because some backends are write-only, such as
`AsyncZarrBackend`.

## Rules

| Rule  | Requirement                                                                          |
| ----- | ------------------------------------------------------------------------------------ |
| `I1`  | `add_array` never reads field values from the schema                                 |
| `I2`  | `add_array` rejects dynamic or zero-sized dimensions with `ValueError`               |
| `I3`  | Each `variable` label, verbatim, names one array; otherwise the schema's `name` does |
| `I4`  | A backend's arrays share coordinates; re-adding an identical array is a no-op        |
| `I5`  | `write` locates by label; unknown labels, arrays or dims raise `ValueError` first    |
| `I6`  | `write` accepts any label subset, including several `lead_time` entries              |
| `I7`  | `write` never modifies `x`; non-blocking backends copy before returning              |
| `I8`  | `write` accepts NumPy-, CuPy- and Torch-adapter-backed arrays                        |
| `I9`  | After `flush`, the store round-trips values, coordinates and persisted attributes    |
| `I10` | Signature markers and `earth2studio_statistics` are not persisted                    |
| `I11` | `close` flushes; later writes raise and repeated closes are no-ops                   |

A shared test suite in `test/io/` checks these rules for every migrated backend.
There is no runtime conformance checker: backends are few and rarely added.

## Shared Helpers

`earth2studio/io/utils.py` holds the logic previously duplicated in each backend:
schema validation and array planning, label-to-position location (contiguous runs
become slices), host transfer for NumPy, CuPy and Torch-adapter payloads, and
attribute filtering and encoding. Backends implement only storage.

It also plans schemas for drivers, replacing per-driver and per-recipe output
coordinate assembly:

```python
def output_schema(
    signature: xr.DataArray,
    leading: Mapping[str, ArrayLike],
    coords: Mapping[str, ArrayLike] | None = None,
) -> CoordinateSystem: ...
```

- `leading` replaces the signature's dynamic prefix with concrete dimensions, in
the given order. It may hold more or fewer dimensions than the prefix it replaces.
- `coords` replaces labels of fixed dimensions, such as the run's `lead_time`
extent. Spatial replacements raise, as in `coord_array_like`.
- Grid metadata, auxiliary coordinates, dtype, name and user attributes carry over;
the result is allocation-free and has no dynamic dimensions.

## Migration

The legacy tensor protocol is replaced, not deprecated alongside. Development will
proceed in two steps:

1. **Protocol and foundation.** This spec, the protocol, shared helpers, the shared
  test suite and `XarrayBackend`. `run.py` plans its schema with a DataArray helper
   replacing `_output_dimensions`. Unmigrated backends keep the legacy signature, and
   `run.py` annotates `io` with a private copy of the legacy protocol so drivers and
   CI keep working.
2. **Backends and drivers.** Migrate `ZarrBackend`, `IceChunkBackend`,
  `NetCDF4Backend`, `KVBackend` and `AsyncZarrBackend`. Move `run.deterministic`,
   `run.diagnostic` and `run.ensemble` to `io.write(y)`, and serve's
   `BackendProgress`, examples, recipes and tests with them. Remove the private
   legacy protocol and `split_coords` from IO paths.

Breaking changes for users:

- `add_array(coords, array_name, data=...)` becomes `add_array(schema)`; initialize
values with a `write` instead of `data=`.
- `write(tensor, coords, array_name)` becomes `write(x)`.
- Writes no longer create arrays implicitly.
- Curvilinear stores hold native auxiliary latitude/longitude instead of
`ilat`/`ilon` dimensions, so readers of existing stores change.

## Open Questions

- Should stores grow along a dimension (`time` for operational cycling, or runs
whose extent is discovered rather than declared)?
- Should the protocol answer whether data is already written (`exists`), and at
what granularity? Resume is item-granular and owned by the planned `Pipeline`
progress store, so this is deferred to that work.
- `AsyncZarrBackend` takes its parallel dimensions and their full values at
construction. Should it derive them from the first `add_array` schema instead?
- Chunking options differ by backend (`chunks`, `chunked_coords`, `shard_coords`).
Should they share one name and meaning?
- Should backends export statistics as CF `cell_methods` and time bounds, alongside
the qualified labels?
- Should `read` join the protocol, requiring every backend to read, for predownload
caches and post-processing?
