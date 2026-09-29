# Spatial Grids

Use `earth2studio.grids.list_grids()` to discover built-in grids and
`resolve_grid(name)` to retrieve their geometry. Pass a registered name to
`coord_array(..., grid=name)` to include that identity in a model signature.

Input validation first checks `earth2studio_grid_id`. Matching nonempty IDs are
trusted to identify spatial coordinates, skipping their value comparisons.
Different nonempty IDs are rejected, even when coordinates match. If either ID is
missing, validation falls back to explicit coordinate and specification checks.
Dimensions, sizes, non-spatial labels, and required metadata are always checked.
Clear the grid ID when changing spatial coordinates to enable explicit validation.
Latitude–longitude inputs may also omit the default EPSG:4326 CRS metadata;
projected inputs still require their CRS, and HEALPix inputs require matching
level, ordering, layout, and applicable XY orientation metadata.

## CF descriptions

Grid coordinates carry CF `standard_name`, `long_name`, and `units` attributes.
Spatial dimension coordinates also carry `axis` where applicable. Projected
coordinate descriptions and units come from the CRS. `coord_array` preserves
these descriptions when dimensions are renamed or coordinates are supplied explicitly.

For grids with a CRS, CF projection parameters are included directly in the
DataArray attributes. These are descriptive metadata: initialization adds no
grid-mapping variable, bounds, or other coordinate arrays. HEALPix signatures
remain index-only; latitude/longitude descriptions accompany geographic
coordinates when those are explicitly requested from the grid.

## Global latitude–longitude grids

| Name | Shape (latitude, longitude) | Convention / examples |
| --- | --- | --- |
| `latlon-0.25deg` | 721 × 1440 | North to south, both poles; most global models |
| `latlon-0.25deg-south-pole-excluded` | 720 × 1440 | N→S, excludes south pole; FCN, Aurora |
| `latlon-1deg` | 181 × 360 | North to south, both poles; GraphCastSmall |
| `latlon-1.5deg` | 121 × 240 | North to south, both poles; FuXiS2S, UCast |
| `gaussian-f90` | 180 × 360 | South to north Gaussian latitudes; ACE2 |

Equiangular grids start longitude at 0° and exclude 360°. The regular Gaussian
F90 grid uses the arcsine of the 180 Legendre roots for latitude and longitude
centers at 0.5°, 1.5°, …, 359.5°. Its alias is `ace2`. Latitude spacing is not
uniform, and this grid must not be substituted with `latlon-1deg`.

## HEALPix grids

Levels **3, 6, and 10** are registered with each of these suffixes:

| Name pattern | Ordering | Dimensions |
| --- | --- | --- |
| `healpix-l{level}-nested` | NESTED | `hpx` |
| `healpix-l{level}-ring` | RING | `hpx` |
| `healpix-l{level}-xy-north-clockwise` | XY, north origin, clockwise | `hpx` |
| `healpix-l{level}-xy-north-clockwise-face` | XY, north, clockwise | `face`, `height`, `width` |

At level L, `nside = 2**L`, the flat layout has `12 * nside**2` pixels, and the
face layout has shape `(12, nside, nside)`. Aliases `hpx3`, `hpx6`, and `hpx10`
resolve to NESTED flat grids. Orders and layouts are distinct geometries: the
same integer index need not represent the same location.

CBottle uses level 6 NESTED, CBottle SR outputs level 10 NESTED, TC guidance
uses level 3 XY flat, and DLESyM uses level 6 XY face. Custom model resolutions
can still use explicit `HEALPixGrid` definitions.

## HRRR domains

`hrrr-conus-3km` (alias `hrrr`) is the parent Lambert conformal grid, with shape
`(1059, 1799)`. Regional models select their domain from that grid rather than
registering a new grid for each crop:

```python
from earth2studio.grids import resolve_grid

hrrr = resolve_grid("hrrr")
# StormCast's default domain; index slices are stop-exclusive.
domain = hrrr.coords(only_index=True).to_dataset().isel(
    y=slice(273, 785), x=slice(579, 1219)
)
coordinates = hrrr.coords({dim: domain[dim].values for dim in hrrr.dims})
```

StormCastCONUS uses `y=slice(17, 1041)`, `x=slice(3, 1795)` for its full model
domain. Both models retain configurable domain selection. Geographic bounds
can also be converted to positional selections through `subset_indexers`.
