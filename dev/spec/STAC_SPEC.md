# Earth2Studio STAC Access Layer

## Goal

Provide one shared layer for reading rasters indexed by SpatioTemporal Asset
Catalogs (STAC): catalog search, asset href signing and lazy, windowed reads of
cloud-optimized GeoTIFFs. Collection-specific data sources (Landsat Collection 2,
Sentinel-2 L2A, NAIP, ...) build on it instead of each carrying its own COG reader,
and users can call it directly to explore a catalog before a source exists.

The layer is not a `DataSource`. It returns fields in the asset's native projected
grid; packing onto a fixed grid, vocabulary mapping and the `DataSource` contract
belong to the sources built on top (see [Collection sources](#collection-sources)).

Status: implemented in `earth2studio/data/stac.py` with providers for Microsoft
Planetary Computer and Element 84 Earth Search. No collection source exists yet.

## Interface

```python
@dataclass(frozen=True)
class StacProvider:
    name: str
    api_url: str
    sign: Callable[[str, Mapping[str, Any]], str] = _sign_generic

PLANETARY_COMPUTER: StacProvider
EARTH_SEARCH: StacProvider

def search_items(collection: str, *, provider=PLANETARY_COMPUTER, bbox=None,
                 intersects=None, time_range=None, query=None, cql2_filter=None,
                 ids=None, sortby=None, max_items=100) -> list[pystac.Item]
def open_asset(item: pystac.Item, asset: str, *, provider=None, chunks=1024,
               rescale=True, mask_nodata=True) -> xr.DataArray
def search(collection: str, assets=None, *, provider=PLANETARY_COMPUTER, bbox=None,
           intersects=None, time_range=None, query=None, cql2_filter=None,
           ids=None, sortby=None, max_items=10, chunks=1024, rescale=True,
           mask_nodata=True, lazy=True) -> xr.DataArray
def collections(provider=PLANETARY_COMPUTER, *, bbox=None, text=None,
                sample_assets=True) -> list[dict[str, Any]]
```

- `search_items` runs the STAC query and returns the matching items.
- `open_asset` opens one asset of one item as a dask-backed array in its native grid.
- `search` composes the two: it opens the requested assets of every matching item,
  clips them to `bbox` and stacks them into `(time, variable, y, x)`.
- `collections` lists a catalog's collections with asset keys and overall extents.
- `StacProvider.sign` turns an asset href plus the merged item/asset properties
  (and the item's `collection` id) into a path GDAL can read. Signing is the only
  provider-specific code; everything else is catalog agnostic.

Each argument takes one form: `collection` is one id, `assets` a sequence of asset
keys, `time_range` an ISO 8601 interval string, `provider` a `StacProvider`. The
STAC filter extension is exposed as `cql2_filter` (CQL2-JSON) and `ids`; arbitrary
keyword passthrough to the client is not supported.

## Rules

| Rule | Requirement |
| --- | --- |
| `S1` | Opening reads the file header only; pixels are read on compute, tile by tile |
| `S2` | Arrays keep the asset's native CRS and affine transform |
| `S3` | `raster:bands` nodata, scale and offset are applied per band |
| `S4` | Masked nodata is NaN; unmasked data keeps the file dtype |
| `S5` | A nodata not shared by all stacked bands or variables is dropped, not misreported |
| `S6` | Signing never writes to the process environment |
| `S7` | Planetary Computer tokens are fetched and renewed by GDAL, one per collection |
| `S8` | `search` returns `(time, variable, y, x)` sorted by time on the first asset's grid |
| `S9` | Layers on another grid are resampled onto it from a padded clip |
| `S10` | Item time is `datetime`, else `start_datetime`, as UTC `datetime64` |
| `S11` | Items that miss `bbox` are skipped; a sub-pixel `bbox` raises `ValueError` |
| `S12` | Multi-band assets become one variable per band, labelled `<asset>_<band>` |
| `S13` | `earth2studio.data` imports `rasterio` before any ECMWF library |

- `S1`: only the COG tiles a computation touches are fetched. `search` has one
  exception, `S9`: resampling is eager for the clipped window of the layers that
  need it.
- `S2`: `da.rio.crs` and `da.rio.transform()` hold the geometry, carried by
  `spatial_ref` and the `y`, `x` dimension coordinates. This is what
  `ProjectedGrid(y, x, crs)` describes ([GRID_SPEC.md](GRID_SPEC.md));
  `earth2studio_crs` is not written by this layer.
- `S3`: a band without an entry falls back to the file's nodata and scale/offset
  tags; once applied, the scale tags are removed from attrs so CF decoders do not
  apply them again.
- `S4`: with `mask_nodata=False` the declared nodata is rescaled alongside the
  values. Integer inputs promote to float32 (8/16-bit) or float64 (wider) only when
  masking or scaling applies.
- `S6`: GDAL options travel in the path (`/vsicurl?opt=val&url=...`); credentials
  and process-wide tuning are read from the environment the user set.
  Requester-pays buckets return `/vsis3/` paths and need
  `AWS_REQUEST_PAYER=requester` plus AWS credentials; a warning is logged when the
  variable is unset.
- `S7`: GDAL's `pc_url_signing` with `pc_collection`; the module keeps no token
  cache, so lazy arrays stay readable past a token's lifetime.
- `S8`: `attrs["stac_items"]` lists the item ids in time order; per-asset attrs
  (`stac_item`, `stac_asset`, `href`) are not carried over. List the
  finest-resolution asset first to keep full resolution.
- `S9`: nearest neighbour from a clip padded by two pixels, so pixels straddling
  the window edge survive. Uncovered pixels are the declared nodata, or NaN
  (promoting to float) when none is declared.
- `S10`: a naive datetime is UTC. Items with neither field get `NaT` and sort last.
- `S11`: "miss" means no pixel of an asset intersects `bbox`; "sub-pixel" means the
  box covers fewer than two pixels of an asset.
- `S13`: loading `eccodes` first makes every GDAL `/vsicurl` read segfault on Linux.

## Collection sources

A source for a particular collection is a thin class over this layer that owns what
the layer deliberately does not:

- Grid: declare a `ProjectedGrid` (or lat/lon `LatLonGrid`) and resample or mosaic
  the per-scene windows from `search` onto it, one field per requested time. Where
  several scene footprints cover the request, the source decides how to combine
  them; this layer returns one window per scene.
- Request: translate a `Region` ([GRID_SPEC.md](GRID_SPEC.md)) into `bbox`, and the
  `DataSource` `time` array into a `time_range` plus item selection (nearest scene,
  cloud-cover sort, time tolerance).
- Vocabulary: map `E2STUDIO_VOCAB` ids to asset keys (and bands of multi-band assets)
  through a lexicon.
- Contract: meet `DataSource` ([FETCH_DATA_SPEC.md](FETCH_DATA_SPEC.md)), including
  the `device` convention: compute the dask array and return NumPy on CPU or CuPy on
  CUDA. `RegriddedSource` ([REGRID_SPEC.md](REGRID_SPEC.md)) then maps the source
  onto a model grid without further code here.
- Coupling: a collection source is bound to a model slot through
  `Coupler(sources=...)` ([COUPLING_SPEC.md](COUPLING_SPEC.md)); the coupler stages
  it onto the model device and regrids it like any source, so the layer needs
  nothing coupler-specific.

## Deferred Work and Verification

- First collection source (candidate: Landsat Collection 2 Level-2) and its mosaic
  rule; nothing in this layer prejudges whether the fixed grid is projected or
  lat/lon.
- Resampling methods other than nearest neighbour for `S9`.
- Verification: `test/data/test_stac.py` covers signing, calibration, per-band
  metadata, grid alignment (incl. `S9` edge pixels and fill), time handling and
  catalog listing offline (local GeoTIFFs, a local HTTP server for the `/vsicurl`
  path form, a fake STAC client); two `slow` tests read the live catalogs.
