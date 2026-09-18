# Earth2Studio Grid Protocol

## Goal

Provide one lightweight contract for describing spatial grids without coupling grid
geometry to field data or regridding engines. This protocol describes grids only; it
does not select interpolation methods, construct weights, or transform field data.

## Interface

`GridDefinition` is a runtime-checkable structural protocol. Implementations satisfy
the interface directly; inheritance is not required.

```python
@runtime_checkable
class GridDefinition(Protocol):
    @property
    def dims(self) -> tuple[str, ...]: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def topology(self) -> GridTopology: ...

    @property
    def crs(self) -> pyproj.CRS | None: ...

    def coords(self, indexes=None, *, only_index=False) -> xr.Coordinates: ...
    def subset_indexers(
        self,
        coordinates,
        *,
        bounds=None,
        bounds_crs=None,
    ) -> dict[str, object]: ...
    def cell_bounds(self, indexes) -> xr.Coordinates | None: ...
    @property
    def attrs(self) -> dict[str, object]: ...
    def fingerprint(self) -> str: ...
```

- `dims` defines authoritative spatial dimension order
- `shape` sizes each dimension in the same order
- `topology` identifies the geometry family
- `crs` describes native coordinates when PyProj can represent them
- `coords()` returns complete Xarray coordinates, or dimension indexes with
  `only_index=True`
- `subset_indexers()` translates supported selections into Xarray indexers
- `cell_bounds()` returns optional geographic cell geometry
- `attrs` returns a JSON-serializable description for Xarray and agents
- `fingerprint()` returns stable geometry identity for validation and caching

## Package

Grid definitions live in `earth2studio.grids`, with one implementation per file. The
package initializer exports definitions and owns the process-local registry.

| Definition | Dimensions | Topology | Native CRS |
| --- | --- | --- | --- |
| `LatLonGrid` | `lat, lon` | Rectilinear | Geographic |
| `ProjectedGrid` | `y, x` | Projected | Required |
| `CurvilinearGrid` | `y, x` | Curvilinear | None |
| `PointGrid` | `x` | Points | None |
| `HEALPixGrid` | `hpx` or `face, height, width` | HEALPix | None |

Users may implement the protocol directly for other grid families.

## Populating Xarray Coordinates

A definition provides complete coordinates in one call:

```python
array = xr.DataArray(data, dims=definition.dims, coords=definition.coords())
```

Dimension coordinates preserve `definition.dims`. Projected, curvilinear, point, and
HEALPix definitions add latitude and longitude as auxiliary coordinates. Use
`only_index=True` when only inexpensive dimension indexes are needed:

```python
indexes = definition.coords(only_index=True)
```

Selected index values avoid generating geographic coordinates for the full grid:

```python
coordinates = definition.coords({"y": selected_y, "x": selected_x})
```

### Allocation-free model signatures

`earth2studio.utils.coord_array` consumes the same definitions without allocating
field values. It supplies grid sizes, index coordinates, serializable grid attrs,
and `earth2studio_crs` from `definition.crs`. Curvilinear and point grids also
supply geographic coordinates. A registered string adds `earth2studio_grid_id`.

```python
signature = coord_array(
    ("batch", "variable", "lat", "lon"),
    {"variable": ["t2m"]},
    dynamic=("batch",),
    grid="latlon-0.25deg-south-pole-excluded",
)
```

Models using projected grids use the standard `y, x` dimensions, preserving
native coordinate values in the CRS's units. The grid supplies these axes and
geographic coordinates; `infer_grid()` can recover the geometry without renaming.

`ProjectedGrid` lazily caches full-grid latitude/longitude as read-only arrays.
Repeated `coords()` calls share those arrays in fresh coordinate containers.
`coord_array(grid=...)` also avoids repeat projections, although xarray may copy
the coordinates when constructing a DataArray. Index-only requests skip projection;
explicit custom indexes are projected independently without populating the full-grid
cache.

Optional `grid_dims` maps standard grid axes to a model's
dimension names, including grid-coordinate dimensions and the `dims` metadata.
Explicit coordinates use the mapped names. This mapping does not rename the grid
definition itself: rename model axes back before passing such an array to
`infer_grid()` or grid selection methods.

A crop has different axes and shape from its full parent grid. Construct a new
definition with the selected native axes and the same CRS, then pass that definition
to `coord_array`. For example, for a projected `parent` grid:

```python
crop = ProjectedGrid(parent.y[10:20], parent.x[30:50], parent.crs)
signature = coord_array(("y", "x"), grid=crop)
```

This produces the cropped axes and their geographic coordinates without attaching
the full parent's registered ID. Using `grid="hrrr"`, for example, requests the
entire registered HRRR geometry, not a crop; attaching that ID to cropped data
would incorrectly claim that its geometry matches the full registered grid.

Coordinate handshakes compare fixed axes and auxiliary coordinates and require
matching declared grid ID and CRS metadata. `coord_array_like()` creates output
signatures from validated inputs without recreating or reprojecting their grid.

### Model coordinates to target grid

The grid contract must support the round trip
`GridDefinition -> coord_array -> infer_grid -> GridDefinition`. Coordinate arrays
store spatial coordinates and serializable grid metadata, not live grid objects.
Non-spatial model dimensions, including empty dynamic dimensions, must not affect
reconstruction.

The reconstructed target grid owns its geometry, extent, and CRS at construction.
Consumers such as `fetch_data` can obtain spatial requirements from that object
without asking callers for separate bounds or CRS arguments. Extent computation
belongs to the grid implementation and must respect topology and longitude wrap;
a bounding rectangle alone is not a complete grid description.

Reconstruction must preserve actual coordinate values, crops, axis order, and CRS.
A registered grid ID must not cause changed coordinates to be replaced by the
parent geometry merely because their shapes match. Renamed model axes must be
recoverable from coordinate metadata without caller-side renaming.

Standard-layout reconstruction exists today. Metadata-driven reconstruction of
renamed axes, validation of registered geometry against coordinate values, and an
explicit grid-owned extent interface are required follow-up work. The manual
renaming guidance above describes the current implementation until that support
lands. Fetch-side spatial processing remains deferred; see
[FETCH_DATA_SPEC.md](FETCH_DATA_SPEC.md).

## HEALPix Representations

HEALPix ordering and storage layout are explicit. `nested` and `ring` use a flat
`hpx` dimension. `xy` supports flat `hpx` storage or an expanded
`(face, height, width)` layout. XY also records its face origin and winding so layouts
such as Earth2Grid's `HEALPIX_PAD_XY` are unambiguous:

```python
dlesym = grids.HEALPixGrid(
    level=6,
    ordering="xy",
    layout="face",
    xy_origin="north",
    xy_clockwise=True,
)
```

## Registry

The process-local registry maps stable names and aliases to complete definitions:

```python
grids.register_grid("regional-lcc", definition, aliases=("regional",))
definition = grids.resolve_grid("regional")
names = grids.list_grids()
```

Registration validates protocol conformance, dimensions, shape, index coordinates,
serializable metadata, and a nonempty fingerprint. Re-registering an identical
definition is a no-op; conflicting names and aliases raise an error. Names recognized
as CRS input by PyProj are reserved.

## Inferring Grid Type

`infer_grid()` follows an explicit-to-general chain:

1. Resolve and validate `earth2studio_grid_id`
2. Infer a projected grid from `y`, `x`, and `earth2studio_crs`
3. Infer rectilinear, point, or curvilinear layouts from latitude and longitude
4. Raise when no supported layout is unambiguous

`PointGrid` is the fallback for arbitrary samples represented as `lat(x)` and
`lon(x)`. Generic `lat(hpx)` and `lon(hpx)` are not enough to distinguish HEALPix
ordering, layout, or level, so HEALPix requires a registered grid identifier.

## Selection

Selection returns positional Xarray indexers and never receives field data. Every
definition supports geographic `bounds` and optional `bounds_crs`. HEALPix also
supports face selection when its ordering identifies faces.

```python
indexers = definition.subset_indexers(
    coordinates,
    bounds=(-125, 25, -65, 50),
    bounds_crs="EPSG:4326",
)
subset = array.isel(indexers)
```
