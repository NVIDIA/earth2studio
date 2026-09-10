# Grid Research Notes

**Status:** Non-normative background for future grid work
**Last updated:** September 2026

## Purpose

This note records design lessons from Xarray and geospatial packages that informed
Earth2Studio's grid protocol. The concise contract remains in `dev/spec/GRID_SPEC.md`.

Earth2Studio needs to describe a model's spatial requirements without allocating
field data. A grid definition must communicate dimension order, shape, geometry,
selection behavior, and enough identity for validation and future regridding.

## Package Findings

### Xarray

[Xarray distinguishes dimension coordinates from auxiliary coordinates](https://docs.xarray.dev/en/stable/user-guide/terminology.html).
This maps well to projected and irregular weather grids:

- `DataArray.dims` is the authoritative array-axis order
- One-dimensional `x`, `y`, or cell identifiers are dimension coordinates
- Latitude and longitude may be multidimensional auxiliary coordinates
- Named dimensions permit selection without relying on positional axis numbers

Xarray's custom `Index` API can support geospatial selection later, but requiring a
custom index for the initial protocol would add complexity. Normal coordinates and
`isel()` indexers provide the simpler baseline.

### CF Conventions and PyProj

[CF grid mappings](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.pdf)
separate projected coordinates, auxiliary latitude and longitude, and CRS metadata.
This supports keeping geometry separate from field data while retaining a familiar
serialized representation.

[PyProj accepts authority identifiers, WKT, PROJ strings, and existing CRS objects](https://pyproj4.github.io/pyproj/stable/api/crs/index.html).
A CRS defines a coordinate reference system, not a finite grid: it does not provide
dimension names, resolution, shape, extent, ordering, or cell identifiers. Therefore
a CRS may be part of a grid definition but cannot replace one.

### Pyresample

[Pyresample separates fixed areas, geographic grids, and swaths into geometry objects](https://pyresample.readthedocs.io/en/latest/api/pyresample.geometry.html).
Its `AreaDefinition` combines projection, extent, width, and height, while swaths use
explicit longitude and latitude. This validates having distinct projected, point,
and curvilinear definitions behind one common protocol.

The main difference is ownership: Earth2Studio should expose Xarray coordinates and
metadata rather than introduce another required data container.

### Earth2Grid

[Earth2Grid uses explicit source and target grid objects](https://github.com/NVlabs/earth2grid)
and dispatches to supported regridders. It currently focuses on regular latitude-
longitude and HEALPix grids. This suggests a future Earth2Studio adapter rather than
embedding Earth2Grid behavior in each grid definition.

Grid definitions should supply geometry; a separate regridder should decide whether
Earth2Grid, ESMF, Pyresample, or another engine supports a requested pair and method.

### xESMF

[xESMF constructs a regridder from source geometry, target geometry, and a method](https://xesmf.readthedocs.io/en/stable/user_api.html).
It accepts rectilinear or curvilinear latitude and longitude, requires cell bounds
for conservative methods, and supports point-like location streams.

[Weight generation is intentionally separate from weight application](https://xesmf.readthedocs.io/en/latest/notebooks/Reuse_regridder.html).
Earth2Studio's grid fingerprint can serve as part of a cache key, together with the
engine, method, options, and source and target definitions.

### xgcm

[xgcm groups coordinates by physical axis and cell position](https://xgcm.readthedocs.io/en/latest/grids/).
Its center, face, inner, and outer positions are useful precedent for staggered
grids. Earth2Studio does not need this in the first grid protocol, but metadata or a
future protocol extension may be needed before supporting staggered model states.

### UXarray and xdggs

[UXarray maintains standalone topology for unstructured grids](https://uxarray.readthedocs.io/en/latest/userguide.html),
including connectivity and multiple supported conventions. This is more machinery
than the initial Earth2Studio scope but identifies what would be required for true
mesh support.

[xdggs represents discrete global grids with a one-dimensional cell coordinate](https://github.com/xarray-contrib/xdggs/blob/main/design_doc.md)
plus grid parameters and specialized selection behavior. HEALPix resolution, RING,
NESTED, or XY ordering, and flat versus face-oriented layout must remain explicit
because they change the meaning of identical array positions. XY definitions must
also record origin and winding; Earth2Grid's `HEALPIX_PAD_XY` convention supports
DLESyM-style `(face, height, width)` arrays.

## Implications for Earth2Studio

### Coordinate Population

A definition returns complete Xarray coordinates in one call:

```python
coordinates = grid.coords()
indexes = grid.coords(only_index=True)
```

Field data can reuse `coordinates` with `dims=grid.dims`. This keeps `y, x` as the
dimensions of projected or curvilinear arrays while exposing two-dimensional `lat`
and `lon` for inspection and downstream tools. Arbitrary locations use `x` as the
dimension with one-dimensional `lat(x)` and `lon(x)` auxiliary coordinates. The
index-only form avoids generating geographic coordinates for allocation-free model
contracts.

### Registry

The registry should remain a small, process-local naming layer:

- `register_grid()` validates and stores one complete definition
- `resolve_grid()` accepts a canonical name or alias
- `list_grids()` returns canonical names
- Built-ins cover common Earth2Studio grids; applications may register local grids
- CRS-like names are reserved to avoid confusing a reference system with a grid

The registry should not select interpolation methods, import optional regridding
engines, discover plugins, or persist state. Those responsibilities belong to a
future regridding layer or application setup.

### Selection

Selection belongs to geometry because each topology interprets a subdomain
differently. `subset_indexers()` returns positional Xarray indexers and never receives
field values. Built-in grids can share geographic bounds while specialized grids may
accept additional options such as HEALPix faces.

Selections need not be rectangular in geographic space. A structured result may be
the smallest enclosing index window, while point and HEALPix grids may return arrays
of selected positions.

### Regridding Boundary

A future regridder should consume source and target `GridDefinition` instances. A
capability registry can match topology pair, method, device, and installed engine:

```text
(source topology, target topology, method, device) -> engine adapter
```

An adapter would validate required geometry before processing data. Examples:

- Bilinear interpolation requires cell centers
- Conservative interpolation requires cell bounds
- Projected transforms require a CRS or explicit geographic coordinates
- HEALPix adapters require level and ordering
- Point interpolation requires explicit geographic locations

The regridder, not the grid or DataArray metadata, should own method choice and engine
options. Unsupported combinations should fail during planning rather than midway
through field processing.

## Current Risks and Follow-up

- `cell_bounds()` is currently optional, so conservative methods remain unavailable
  for definitions that return `None`
- Fingerprints must include every property that changes geometry, including CRS,
  coordinate values, HEALPix level, and ordering
- Registry validation currently checks the structural contract but could later verify
  geographic coordinate dimensions, finite values, and metadata completeness
- The built-in HEALPix coordinate implementation should eventually be compared with
  a maintained implementation such as Earth2Grid or healpy
- Curvilinear and point grids may require spatial indexes for fast repeated selection
- Staggered grids and unstructured connectivity are intentionally deferred
- Serialized CF compatibility needs a separate decision from the in-memory protocol

## References

- [Xarray terminology](https://docs.xarray.dev/en/stable/user-guide/terminology.html)
- [Xarray indexing](https://docs.xarray.dev/en/stable/indexing.html)
- [CF Conventions 1.12](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.pdf)
- [PyProj CRS API](https://pyproj4.github.io/pyproj/stable/api/crs/index.html)
- [Pyresample geometry API](https://pyresample.readthedocs.io/en/latest/api/pyresample.geometry.html)
- [Earth2Grid](https://github.com/NVlabs/earth2grid)
- [xESMF user API](https://xesmf.readthedocs.io/en/stable/user_api.html)
- [xgcm grids](https://xgcm.readthedocs.io/en/latest/grids/)
- [UXarray user guide](https://uxarray.readthedocs.io/en/latest/userguide.html)
- [xdggs design](https://github.com/xarray-contrib/xdggs/blob/main/design_doc.md)
