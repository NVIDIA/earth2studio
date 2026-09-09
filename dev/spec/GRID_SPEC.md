# Earth2Studio Grid Protocol

**Status:** Draft

## Goal

Provide one explicit, lightweight contract for describing spatial grids without
coupling grid geometry to weather data, model tensors, or regridding engines.

## Interface

`GridDefinition` is a runtime-checkable structural protocol. A grid object satisfies
the protocol by implementing its members; inheritance is neither required nor used.

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

    def index_coordinates(self) -> xr.Coordinates: ...
    def geographic_coordinates(self, indexes) -> xr.Coordinates: ...
    def subset_indexers(self, coordinates, **selection) -> dict[str, object]: ...
    def cell_bounds(self, indexes) -> xr.Coordinates | None: ...
    def to_metadata(self) -> dict[str, object]: ...
    def fingerprint(self) -> str: ...
```

The members have the following meanings:

- `dims` defines authoritative spatial dimension order
- `shape` sizes each spatial dimension in the same order
- `topology` identifies the geometry family
- `crs` describes native coordinates when PyProj can represent them
- `index_coordinates()` returns one-dimensional indexes for every spatial dimension
- `geographic_coordinates()` maps selected indexes to latitude and longitude
- `subset_indexers()` translates supported selections into Xarray indexers
- `cell_bounds()` returns optional geographic cell geometry
- `to_metadata()` returns a JSON-serializable description
- `fingerprint()` returns stable geometry identity for validation and caching

## Definitions

Earth2Studio provides these implementations:

| Definition | Dimensions | Topology | Native CRS |
| --- | --- | --- | --- |
| `LatLonGrid` | `lat, lon` | Rectilinear | Geographic |
| `ProjectedGrid` | `y, x` | Projected | Required |
| `CurvilinearGrid` | `y, x` | Curvilinear | None |
| `PointGrid` | `x` | Points | None |
| `HEALPixGrid` | `hpx` | HEALPix | None |

Users may implement the protocol directly for other grid families.

## Registry

The process-local registry maps stable names and aliases to complete definitions:

```python
e2s.register_grid("regional-lcc", definition, aliases=("regional",))
definition = e2s.resolve_grid("regional")
names = e2s.list_grids()
```

Registration validates protocol conformance, dimensions, shape, index coordinates,
serializable metadata, and a nonempty fingerprint. Re-registering an identical
definition is a no-op. Conflicting names and aliases raise an error. Names already
recognized as CRS input by PyProj are reserved.

Built-in registry entries include global 0.25-degree latitude-longitude grids, the
HRRR CONUS 3-km Lambert grid, and nested HEALPix level 6.

## Inference

`infer_grid()` recognizes common Xarray coordinate layouts without registration:

- independent one-dimensional `lat` and `lon` coordinates
- two-dimensional `lat` and `lon` coordinates on `y, x`
- one-dimensional point `lat` and `lon` coordinates on `x`
- projected `y, x` coordinates with an `earth2studio_crs` attribute
- a registered `earth2studio_grid_id` attribute

Ambiguous layouts raise rather than guessing.

## Selection

Grid selection returns positional Xarray indexers and never receives field data.
Every definition supports geographic `bounds` and optional `bounds_crs`. HEALPix
also supports face selection for nested ordering.

```python
indexers = definition.subset_indexers(
    coordinates,
    bounds=(-125, 25, -65, 50),
    bounds_crs="EPSG:4326",
)
subset = array.isel(indexers)
```

## Regridding Boundary

The protocol describes geometry but does not choose interpolation methods or engines.
A regridder can use topology, CRS, geographic centers, optional cell bounds, and the
fingerprint to select an implementation and cache weights. Unsupported grid pairs or
methods must raise explicitly.
