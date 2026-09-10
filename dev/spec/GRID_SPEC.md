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
    def subset_indexers(self, coordinates, **selection) -> dict[str, object]: ...
    def cell_bounds(self, indexes) -> xr.Coordinates | None: ...
    def to_metadata(self) -> dict[str, object]: ...
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
- `to_metadata()` returns a JSON-serializable description for attributes and agents
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
