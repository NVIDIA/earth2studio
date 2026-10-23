# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for lazy, cloud-native access to STAC-cataloged rasters.

This module is not a :class:`~earth2studio.data.DataSource`. It is the common
layer underneath collection-specific data sources: catalog search, asset href
signing and lazy, windowed reads of cloud-optimized GeoTIFFs. A data source
for a particular catalog (Landsat Collection 2, Sentinel-2 L2A, ...) builds
on these functions and adds the fixed grid, lexicon and ``DataSource``
interface that models and :func:`~earth2studio.data.fetch_data` need; see
:mod:`earth2studio.data.planetary_computer` for that pattern. Power users can
call this module directly to explore a catalog.

:func:`search` queries a catalog and returns the matching assets as one lazy,
dask-backed ``(time, variable, y, x)`` :class:`xarray.DataArray` in its native
grid and CRS. :func:`open_asset` opens a single item asset and
:func:`collections` lists what a catalog offers. Only the byte ranges a
computation touches are read from the remote COGs.

Examples
--------
>>> from earth2studio.data import stac
>>> da = stac.search(
...     "landsat-c2-l2",
...     assets=["red", "nir08"],
...     bbox=[-122.55, 37.70, -122.35, 37.85],
...     time_range="2024-10-01/2024-12-31",
...     query={"eo:cloud_cover": {"lt": 5}},
...     max_items=3,
... )                                          # lazy (time, variable, y, x)
>>> da.attrs["stac_items"][0]
'LC09_L2SP_045034_20241209_02_T1'
>>> ndvi = (da.sel(variable="nir08") - da.sel(variable="red")) / (
...     da.sel(variable="nir08") + da.sel(variable="red")
... )
>>> ndvi.compute()                             # reads only the clipped tiles

One asset at a time; the array keeps the asset's projected grid, so window
in lon/lat by passing the ``crs``:

>>> item = stac.search_items("landsat-c2-l2", ids=da.attrs["stac_items"][:1])[0]
>>> red = stac.open_asset(item, "red")         # lazy; nothing read yet
>>> red.rio.clip_box(-122.5, 37.7, -122.4, 37.8, crs="EPSG:4326").compute()

GDAL reads the remote files. Its behaviour is controlled through GDAL
configuration options which this module never writes; set them in the
environment if needed, e.g. ``GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`` and
``VSI_CACHE=TRUE`` for fewer, larger range requests, or ``AWS_REQUEST_PAYER``
and AWS credentials for requester-pays buckets.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timezone
from typing import Any, TypeVar
from urllib.parse import quote, urlparse

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from pystac import Item
    from pystac_client import Client
except ImportError:
    OptionalDependencyFailure("data")
    Client = None
    rioxarray = None
    Item = TypeVar("Item")  # type: ignore


# ---------------------------------------------------------------------------
# Providers: STAC API endpoint + how to turn an asset href into a GDAL path
# ---------------------------------------------------------------------------
def _vsicurl(url: str, **options: str) -> str:
    """Build a ``/vsicurl?opt=val&url=...`` path.

    GDAL applies options given in the path to this file only, so nothing is
    written to the process environment. ``empty_dir=yes`` skips the remote
    directory listing and side-car probing when the COG is opened and
    ``max_retry``/``retry_delay`` retry HTTP 429/5xx responses, which the
    Planetary Computer token service returns when throttling.
    """
    options.setdefault("empty_dir", "yes")
    options.setdefault("max_retry", "3")
    options.setdefault("retry_delay", "2")
    query = "&".join(f"{k}={v}" for k, v in options.items())
    return f"/vsicurl?{query}&url={quote(url, safe='')}"


def _sign_generic(href: str, _properties: Mapping[str, Any]) -> str:
    if href.startswith(("http://", "https://")):
        return _vsicurl(href)
    return href


def _sign_planetary_computer(href: str, _properties: Mapping[str, Any]) -> str:
    # GDAL (>= 3.5.2) fetches and renews the short-lived SAS token itself, so
    # there is no token cache here and lazy arrays stay readable past an hour
    if not urlparse(href).netloc.endswith(".blob.core.windows.net"):
        return _sign_generic(href, _properties)
    return _vsicurl(href, pc_url_signing="yes")


def _sign_earth_search(href: str, properties: Mapping[str, Any]) -> str:
    parsed = urlparse(href)
    if parsed.scheme != "s3":
        return _sign_generic(href, properties)
    if properties.get("storage:requester_pays"):
        # Requester-pays buckets need signed requests: go through GDAL's
        # /vsis3/ so its AWS credential chain signs them
        if os.environ.get("AWS_REQUEST_PAYER") != "requester":
            logger.warning(
                f"{parsed.netloc} is a requester-pays bucket; set "
                "AWS_REQUEST_PAYER=requester and AWS credentials to read it"
            )
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    region = properties.get("storage:region", "us-west-2")
    return _vsicurl(f"https://{parsed.netloc}.s3.{region}.amazonaws.com{parsed.path}")


@dataclass(frozen=True, slots=True)
class StacProvider:
    """A STAC API endpoint and how to turn its asset hrefs into GDAL paths.

    Parameters
    ----------
    name : str
        Short provider name, e.g. ``"planetary-computer"``.
    api_url : str
        STAC API root URL.
    sign : Callable[[str, Mapping[str, Any]], str], optional
        Maps an asset href (plus the merged item/asset properties) to a path
        GDAL can read: Planetary Computer blobs are read with GDAL's built-in
        SAS signing, Earth Search ``s3://`` hrefs over public HTTPS. The
        default wraps http(s) hrefs in ``/vsicurl`` and leaves other hrefs
        unchanged.
    """

    name: str
    api_url: str
    sign: Callable[[str, Mapping[str, Any]], str] = _sign_generic


PLANETARY_COMPUTER = StacProvider(
    "planetary-computer",
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    _sign_planetary_computer,
)
EARTH_SEARCH = StacProvider(
    "earth-search",
    "https://earth-search.aws.element84.com/v1",
    _sign_earth_search,
)


def _infer_provider(href: str) -> StacProvider:
    parsed = urlparse(href)
    if parsed.netloc.endswith(".blob.core.windows.net"):
        return PLANETARY_COMPUTER
    if parsed.scheme == "s3" or parsed.netloc.endswith(".amazonaws.com"):
        return EARTH_SEARCH
    return StacProvider("generic", "")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@check_optional_dependencies()
def search_items(
    collection: str,
    *,
    provider: StacProvider = PLANETARY_COMPUTER,
    bbox: Sequence[float] | None = None,
    intersects: Mapping[str, Any] | None = None,
    time_range: str | None = None,
    query: Mapping[str, Any] | None = None,
    cql2_filter: Mapping[str, Any] | None = None,
    ids: Sequence[str] | None = None,
    sortby: Sequence[Mapping[str, str]] | None = None,
    max_items: int | None = 100,
) -> list[Item]:
    """Search a STAC API and return the matching items.

    Parameters
    ----------
    collection : str
        Collection id, e.g. ``"landsat-c2-l2"``.
    provider : StacProvider, optional
        Catalog to query (:data:`PLANETARY_COMPUTER`, :data:`EARTH_SEARCH` or
        a custom :class:`StacProvider`), by default PLANETARY_COMPUTER
    bbox : Sequence[float] | None, optional
        ``[min_lon, min_lat, max_lon, max_lat]``, by default None
    intersects : Mapping[str, Any] | None, optional
        GeoJSON geometry to intersect, by default None
    time_range : str | None, optional
        ISO 8601 ``"start/end"`` interval (``".."`` for an open end) or a
        single instant, by default None
    query : Mapping[str, Any] | None, optional
        STAC query-extension filter, e.g. ``{"eo:cloud_cover": {"lt": 5}}``,
        by default None
    cql2_filter : Mapping[str, Any] | None, optional
        CQL2-JSON filter (STAC filter extension), by default None
    ids : Sequence[str] | None, optional
        Restrict the search to these item ids, by default None
    sortby : Sequence[Mapping[str, str]] | None, optional
        Sort spec, e.g. ``[{"field": "properties.eo:cloud_cover",
        "direction": "asc"}]``, by default None
    max_items : int | None, optional
        Maximum number of items to return, by default 100

    Returns
    -------
    list[pystac.Item]
    """
    client = Client.open(provider.api_url)
    result = client.search(
        collections=[collection],
        bbox=list(bbox) if bbox is not None else None,
        intersects=intersects,
        datetime=time_range,
        query=dict(query) if query else None,
        filter=dict(cql2_filter) if cql2_filter else None,
        ids=list(ids) if ids else None,
        sortby=list(sortby) if sortby else None,
        max_items=max_items,
    )
    return list(result.items())


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------
def _calibrate(
    da: xr.DataArray,
    band: Mapping[str, Any],
    *,
    rescale: bool,
    mask_nodata: bool,
    skip_offset: bool,
) -> xr.DataArray:
    """Apply one ``raster:bands`` entry (nodata, scale, offset) to a band."""
    nodata = band.get("nodata", da.rio.nodata)
    scale = band.get("scale")
    offset = None if skip_offset else band.get("offset")
    # The raster extension allows string nodata ("nan"); a str compared to a
    # uint16 array would silently mask nothing
    if nodata is not None:
        nodata = float(nodata)
    if mask_nodata or rescale:
        da = da.astype("float32")
    if mask_nodata and nodata is not None:
        valid = da.notnull() if np.isnan(nodata) else da != nodata
        da = da.where(valid)
    if rescale and (scale is not None or offset is not None):
        factor = np.float32(scale if scale is not None else 1.0)
        shift = np.float32(offset if offset is not None else 0.0)
        da = da * factor + shift
        if not mask_nodata and nodata is not None and not np.isnan(nodata):
            # Keep the declared nodata consistent with the rescaled values
            nodata = float(nodata * factor + shift)
    # Arithmetic drops attrs, so (re)declare nodata last
    if nodata is not None:
        da.rio.write_nodata(np.nan if mask_nodata else nodata, inplace=True)
    return da


@check_optional_dependencies()
def open_asset(
    item: Item,
    asset: str,
    *,
    provider: StacProvider | None = None,
    chunks: int | Mapping[str, int] | None = 1024,
    rescale: bool = True,
    mask_nodata: bool = True,
) -> xr.DataArray:
    """Open one item asset as a lazy, dask-backed :class:`xarray.DataArray`.

    Nothing beyond the file header is read until values are requested, so
    windowing (``.rio.clip_box``, ``.isel``) before ``.compute()`` fetches only
    the COG tiles covering that window. The array keeps the asset's native
    grid and CRS (``da.rio.crs``, ``da.rio.transform()``).

    Parameters
    ----------
    item : pystac.Item
        Item returned by :func:`search_items`.
    asset : str
        Asset key, e.g. ``"red"``.
    provider : StacProvider | None, optional
        Provider whose ``sign`` turns the href into a GDAL path; inferred
        from the href host when None, by default None
    chunks : int | Mapping[str, int] | None, optional
        Dask chunk size; an int is used for both ``x`` and ``y``. Match the
        COG's internal block size for best throughput. None loads eagerly,
        by default 1024
    rescale : bool, optional
        Apply the STAC ``raster:bands`` ``scale``/``offset`` so values are in
        physical units, by default True
    mask_nodata : bool, optional
        Replace ``raster:bands``/file nodata with NaN, by default True

    Returns
    -------
    xr.DataArray
        ``(time, y, x)`` for single-band assets, else ``(time, band, y, x)``
        (no ``time`` when the item has no datetime); float32 when rescaled or
        masked, otherwise the file dtype.
    """
    if asset not in item.assets:
        raise KeyError(
            f"Asset {asset!r} not in item {item.id}; available: {sorted(item.assets)}"
        )
    stac_asset = item.assets[asset]
    properties: dict[str, Any] = {**item.properties, **stac_asset.extra_fields}
    prov = provider if provider is not None else _infer_provider(stac_asset.href)
    path = prov.sign(stac_asset.href, properties)

    if isinstance(chunks, int):
        chunks = {"band": 1, "x": chunks, "y": chunks}
    da = rioxarray.open_rasterio(path, chunks=chunks, lock=False, masked=False)

    # One raster:bands entry per band; a missing entry falls back to file nodata
    bands = stac_asset.extra_fields.get("raster:bands") or []
    options = dict(
        rescale=rescale,
        mask_nodata=mask_nodata,
        # Earth Search bakes the Sentinel-2 BOA offset into its COGs while the
        # metadata still reports it; applying it again would bias reflectance
        skip_offset=bool(properties.get("earthsearch:boa_offset_applied")),
    )
    if da.sizes["band"] == 1:
        da = _calibrate(
            da.squeeze("band", drop=True), bands[0] if bands else {}, **options
        )
    else:
        da = xr.concat(
            [
                _calibrate(
                    da.isel(band=i), bands[i] if i < len(bands) else {}, **options
                )
                for i in range(da.sizes["band"])
            ],
            dim="band",
        )

    da.attrs.update(
        stac_collection=item.collection_id,
        stac_item=item.id,
        stac_asset=asset,
        href=stac_asset.href,
    )
    if item.datetime is not None:
        utc = item.datetime.astimezone(timezone.utc).replace(tzinfo=None)
        da = da.expand_dims(time=[np.datetime64(utc)])
    return da


def _same_grid(a: xr.DataArray, b: xr.DataArray) -> bool:
    return (
        a.rio.crs == b.rio.crs
        and a.rio.transform() == b.rio.transform()
        and a.shape[-2:] == b.shape[-2:]
    )


def _match_grid(da: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Resample a ``(time=1, variable, y, x)`` array onto ``reference``'s grid.

    ``rio.reproject_match`` is eager and handles at most 3D, so this reads
    ``da`` (already clipped to the window) and returns a numpy-backed array
    aligned to ``reference``'s coordinates.
    """
    resampled = da.isel(time=0).rio.reproject_match(reference.isel(time=0))
    return resampled.expand_dims(time=da["time"]).assign_coords(
        x=reference["x"], y=reference["y"]
    )


@check_optional_dependencies()
def search(
    collection: str,
    assets: Sequence[str] | None = None,
    *,
    provider: StacProvider = PLANETARY_COMPUTER,
    bbox: Sequence[float] | None = None,
    intersects: Mapping[str, Any] | None = None,
    time_range: str | None = None,
    query: Mapping[str, Any] | None = None,
    cql2_filter: Mapping[str, Any] | None = None,
    ids: Sequence[str] | None = None,
    sortby: Sequence[Mapping[str, str]] | None = None,
    max_items: int | None = 10,
    chunks: int | Mapping[str, int] | None = 1024,
    rescale: bool = True,
    mask_nodata: bool = True,
    lazy: bool = True,
) -> xr.DataArray:
    """Search a STAC catalog and open the results as one lazy array.

    Runs :func:`search_items`, opens the requested assets of every matching
    item (see :func:`open_asset`), clips them to ``bbox`` and stacks them into
    Earth2Studio's ``(time, variable, y, x)`` layout in the first item's
    grid. Nothing beyond file headers is read until values are computed, with
    one exception: assets or items on a different grid than the first (other
    resolution, scene or UTM zone) are resampled onto it while ``search`` runs,
    which reads their clipped window eagerly.

    Parameters
    ----------
    collection : str
        Collection id, e.g. ``"landsat-c2-l2"``.
    assets : Sequence[str] | None, optional
        Asset keys to open; these become the ``variable`` coordinate. None
        opens every raster (GeoTIFF/COG) asset of the first item, by default None
    provider : StacProvider, optional
        Catalog to query, by default PLANETARY_COMPUTER
    bbox : Sequence[float] | None, optional
        ``[min_lon, min_lat, max_lon, max_lat]``; used for the search and to
        clip each asset, by default None (whole scenes)
    intersects : Mapping[str, Any] | None, optional
        GeoJSON geometry to search with, by default None
    time_range : str | None, optional
        ISO 8601 ``"start/end"`` interval or single instant, by default None
    query : Mapping[str, Any] | None, optional
        STAC query-extension filter, e.g. ``{"eo:cloud_cover": {"lt": 5}}``,
        by default None
    cql2_filter : Mapping[str, Any] | None, optional
        CQL2-JSON filter, by default None
    ids : Sequence[str] | None, optional
        Restrict the search to these item ids, by default None
    sortby : Sequence[Mapping[str, str]] | None, optional
        Sort spec, e.g. ``[{"field": "properties.eo:cloud_cover",
        "direction": "asc"}]``, by default None
    max_items : int | None, optional
        Maximum number of items, by default 10
    chunks : int | Mapping[str, int] | None, optional
        Dask chunk size per asset (see :func:`open_asset`), by default 1024
    rescale : bool, optional
        Apply ``raster:bands`` scale/offset, by default True
    mask_nodata : bool, optional
        Replace nodata with NaN, by default True
    lazy : bool, optional
        Return a dask-backed array; False computes before returning, by
        default True

    Returns
    -------
    xr.DataArray
        Lazy ``(time, variable, y, x)`` array sorted by time, with
        ``.rio.crs``/``.rio.transform()`` describing the grid and the STAC
        item ids in ``attrs["stac_items"]``.

    Raises
    ------
    FileNotFoundError
        If the search returns no items.
    """
    items = search_items(
        collection,
        provider=provider,
        bbox=bbox,
        intersects=intersects,
        time_range=time_range,
        query=query,
        cql2_filter=cql2_filter,
        ids=ids,
        sortby=sortby,
        max_items=max_items,
    )
    if not items:
        raise FileNotFoundError(
            f"No items found in {collection!r} for bbox={bbox} time_range={time_range}"
        )
    if assets is None:
        assets = [
            key
            for key, a in items[0].assets.items()
            if "tiff" in (a.media_type or "").lower()
            or a.href.lower().endswith((".tif", ".tiff"))
        ]
        if not assets:
            raise ValueError(
                f"No raster assets found on {items[0].id}; pass assets= explicitly "
                f"(available: {sorted(items[0].assets)})"
            )

    per_item: list[xr.DataArray] = []
    reference: xr.DataArray | None = None
    for item in items:
        layers: list[xr.DataArray] = []
        for key in assets:
            da = open_asset(
                item,
                key,
                provider=provider,
                chunks=chunks,
                rescale=rescale,
                mask_nodata=mask_nodata,
            )
            if "time" not in da.dims:
                da = da.expand_dims(time=[np.datetime64("NaT", "ns")])
            if bbox is not None:
                da = da.rio.clip_box(*bbox, crs="EPSG:4326")
            if "band" in da.dims:
                # Multi-band asset (e.g. an RGB "visual" COG): one variable per band
                labels = [f"{key}_{b}" for b in da["band"].values]
                da = da.rename(band="variable").assign_coords(variable=labels)
            else:
                da = da.expand_dims(variable=[key])
            da = da.transpose("time", "variable", "y", "x")
            if layers and not _same_grid(da, layers[0]):
                # Assets at different resolutions (Sentinel-2 10 m vs 20 m):
                # resample onto the first asset's grid instead of letting concat
                # outer-join the coordinates into a NaN-riddled union grid
                da = _match_grid(da, layers[0])
            layers.append(da)
        # "override" keeps the first layer's attrs (incl. spatial_ref) rather
        # than dropping the CRS because per-asset attrs differ
        stacked = xr.concat(layers, dim="variable", combine_attrs="override")
        if reference is None:
            reference = stacked
        elif not _same_grid(stacked, reference):
            stacked = _match_grid(stacked, reference)
        per_item.append(stacked)

    out = xr.concat(per_item, dim="time", combine_attrs="override")
    order = np.argsort(out.time.values, kind="stable")
    out = out.isel(time=order)
    if not lazy:
        out = out.compute()
    out.attrs["stac_collection"] = collection
    out.attrs["stac_items"] = [items[i].id for i in order]
    return out


# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------
@check_optional_dependencies()
def collections(
    provider: StacProvider = PLANETARY_COMPUTER,
    *,
    bbox: Sequence[float] | None = None,
    text: str | None = None,
    sample_assets: bool = True,
) -> list[dict[str, Any]]:
    """List a catalog's collections with their asset keys and extents.

    Filtering is client-side over ``/collections`` (neither Planetary
    Computer nor Earth Search serve a collection-search extension).

    Parameters
    ----------
    provider : StacProvider, optional
        Catalog to list, by default PLANETARY_COMPUTER
    bbox : Sequence[float] | None, optional
        Keep collections whose overall spatial extent intersects this
        ``[min_lon, min_lat, max_lon, max_lat]`` box, by default None
    text : str | None, optional
        Case-insensitive substring matched on id and title, by default None
    sample_assets : bool, optional
        Fetch one item per matching collection to list its asset keys (one
        extra request per collection), by default True

    Returns
    -------
    list[dict[str, Any]]
        One dict per collection with ``id``, ``title``, ``assets`` (sorted
        asset keys), ``bbox`` (overall lon/lat extent or None) and
        ``time_range`` (overall ``(start, end)`` datetimes, None for open).
    """
    client = Client.open(provider.api_url)
    needle = text.lower() if text else None

    infos: list[dict[str, Any]] = []
    for collection in client.get_collections():
        title = collection.title or collection.id
        if needle and needle not in f"{collection.id} {title}".lower():
            continue

        # The spec defines the first bbox/interval as the overall extent
        bboxes = collection.extent.spatial.bboxes or [None]
        extent = bboxes[0]
        if extent is not None and len(extent) == 6:
            # 3D bboxes carry min/max elevation at indices 2 and 5
            extent = [extent[0], extent[1], extent[3], extent[4]]
        if (
            bbox is not None
            and extent is not None
            and (
                bbox[2] < extent[0]
                or bbox[0] > extent[2]
                or bbox[3] < extent[1]
                or bbox[1] > extent[3]
            )
        ):
            continue
        intervals = collection.extent.temporal.intervals or [[None, None]]

        asset_keys: list[str] = []
        if sample_assets:
            sample = next(
                client.search(collections=[collection.id], max_items=1).items(), None
            )
            if sample is not None:
                asset_keys = sorted(sample.assets)

        infos.append(
            {
                "id": collection.id,
                "title": title,
                "assets": asset_keys,
                "bbox": tuple(extent) if extent is not None else None,
                "time_range": tuple(intervals[0]),
            }
        )
    return infos
