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
    grid="fcn1",
)
```

`ProjectedGrid` lazily caches full-grid latitude/longitude as read-only arrays.
Repeated `coords()` calls share those arrays in fresh coordinate containers.
`coord_array(grid=...)` also avoids repeat projections, although xarray may copy
the coordinates when constructing a DataArray. Index-only requests skip projection;
explicit custom indexes are projected independently without populating the full-grid
cache.

`grid_dims={"y": "hrrr_y", "x": "hrrr_x"}` maps standard grid axes to a model's
dimension names, including grid-coordinate dimensions and the `dims` metadata.
Explicit coordinates use the mapped names. This mapping does not rename the grid
definition itself: rename model axes back before passing such an array to
`infer_grid()` or grid selection methods. A cropped grid should use a definition
constructed from the cropped axes, not the identifier of its full parent grid.

Coordinate handshakes compare fixed axes and auxiliary coordinates and require
matching declared grid ID and CRS metadata. `coord_array_like()` creates output
signatures from validated inputs without recreating or reprojecting their grid.

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
