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

"""Lazy, cloud-native access to STAC-cataloged geospatial data.

Discovery and access without a data source per collection. :func:`search`
queries a catalog and returns the matching data as one lazy, dask-backed
``(time, variable, y, x)`` :class:`xarray.DataArray` in its native grid and
CRS, with the STAC items behind it on ``.items``. :func:`open` is the
one-asset building block and :func:`collections` browses a catalog. Only
the byte ranges a computation touches are read from the remote COGs.

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
>>> da.items[0].id
'LC09_L2SP_045034_20241209_02_T1'
>>> ndvi = (da.sel(variable="nir08") - da.sel(variable="red")) / (
...     da.sel(variable="nir08") + da.sel(variable="red")
... )
>>> ndvi.compute()                             # reads only the clipped tiles

One asset at a time:

>>> red = stac.open(da.items[0], "red")        # lazy; nothing read yet
>>> red.rio.clip_box(-122.5, 37.7, -122.4, 37.8).compute()
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar
from urllib.parse import urlparse

import numpy as np
import xarray as xr

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


def _passthrough(href: str, _properties: Mapping[str, Any]) -> str:
    return href


@dataclass(frozen=True, slots=True)
class StacProvider:
    """A STAC API endpoint and how to turn its asset hrefs into readable URLs.

    Parameters
    ----------
    name : str
        Short provider name, e.g. ``"planetary-computer"``.
    api_url : str
        STAC API root URL.
    sign : Callable[[str, Mapping[str, Any]], str], optional
        Maps an asset href (plus the merged item/asset properties) to a URL
        GDAL can read directly: appends a SAS token for Planetary Computer,
        rewrites ``s3://`` to public HTTPS for Earth Search, by default the
        href is returned unchanged.
    """

    name: str
    api_url: str
    sign: Callable[[str, Mapping[str, Any]], str] = _passthrough


# ---------------------------------------------------------------------------
# Planetary Computer: short-lived SAS tokens per (storage account, container)
# ---------------------------------------------------------------------------
_PC_SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_PC_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)
_pc_tokens: dict[tuple[str, str], tuple[str, datetime]] = {}
_pc_tokens_lock = threading.Lock()


def _planetary_computer_token(account: str, container: str) -> str:
    key = (account, container)
    now = datetime.now(timezone.utc)
    with _pc_tokens_lock:
        cached = _pc_tokens.get(key)
        if cached is not None and cached[1] - now > _PC_TOKEN_REFRESH_MARGIN:
            return cached[0]
        url = f"{_PC_SAS_URL}/{account}/{container}"
        if not url.startswith("https://"):
            raise ValueError(f"Refusing non-https SAS endpoint: {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            payload = json.load(resp)
        expiry = datetime.fromisoformat(payload["msft:expiry"].replace("Z", "+00:00"))
        _pc_tokens[key] = (payload["token"], expiry)
        return payload["token"]


def _sign_planetary_computer(href: str, _properties: Mapping[str, Any]) -> str:
    parsed = urlparse(href)
    if not parsed.netloc.endswith(".blob.core.windows.net") or parsed.query:
        return href
    account = parsed.netloc.split(".")[0]
    container = parsed.path.lstrip("/").split("/", 1)[0]
    return f"{href}?{_planetary_computer_token(account, container)}"


# ---------------------------------------------------------------------------
# Earth Search: public S3, read anonymously over HTTPS
# ---------------------------------------------------------------------------
_S3_DEFAULT_REGION = "us-west-2"


def _sign_public_s3(href: str, properties: Mapping[str, Any]) -> str:
    parsed = urlparse(href)
    if parsed.scheme != "s3":
        return href
    if properties.get("storage:requester_pays"):
        # Requester-pays buckets need signed requests; route through GDAL's
        # /vsis3/ so its AWS credential chain signs them (fails clearly
        # without credentials instead of an anonymous 403)
        os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    region = properties.get("storage:region", _S3_DEFAULT_REGION)
    return f"https://{parsed.netloc}.s3.{region}.amazonaws.com{parsed.path}"


PROVIDERS: dict[str, StacProvider] = {
    "planetary-computer": StacProvider(
        "planetary-computer",
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        _sign_planetary_computer,
    ),
    "earth-search": StacProvider(
        "earth-search",
        "https://earth-search.aws.element84.com/v1",
        _sign_public_s3,
    ),
}


def get_provider(provider: str | StacProvider) -> StacProvider:
    """Resolve a provider name, STAC API URL, or :class:`StacProvider`."""
    if isinstance(provider, StacProvider):
        return provider
    if provider in PROVIDERS:
        return PROVIDERS[provider]
    if provider.startswith(("http://", "https://")):
        return StacProvider(provider, provider)
    raise ValueError(
        f"Unknown STAC provider {provider!r}; use one of {sorted(PROVIDERS)} "
        "or a STAC API URL"
    )


def _infer_provider(href: str) -> StacProvider:
    parsed = urlparse(href)
    if parsed.netloc.endswith(".blob.core.windows.net"):
        return PROVIDERS["planetary-computer"]
    if parsed.scheme == "s3" or parsed.netloc.endswith(".amazonaws.com"):
        return PROVIDERS["earth-search"]
    return StacProvider("generic", "")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _search_items(
    collection: str | Sequence[str],
    *,
    provider: str | StacProvider = "planetary-computer",
    bbox: Sequence[float] | None = None,
    intersects: Mapping[str, Any] | None = None,
    time_range: str | datetime | Sequence[datetime] | None = None,
    query: Mapping[str, Any] | None = None,
    sortby: Sequence[Mapping[str, str]] | None = None,
    max_items: int | None = 100,
    **search_kwargs: Any,
) -> list[Item]:
    """Search a STAC API and return matching items (see :func:`search`).

    Parameters
    ----------
    collection : str | Sequence[str]
        Collection id(s) to search, e.g. ``"landsat-c2-l2"``.
    provider : str | StacProvider, optional
        Provider name (``"planetary-computer"``, ``"earth-search"``), STAC
        API URL, or :class:`StacProvider`, by default "planetary-computer".
    bbox : Sequence[float] | None, optional
        ``[min_lon, min_lat, max_lon, max_lat]``, by default None
    intersects : Mapping[str, Any] | None, optional
        GeoJSON geometry to intersect, by default None
    time_range : str | datetime | Sequence[datetime] | None, optional
        ISO ``"start/end"`` string, a single datetime, or a (start, end)
        pair, by default None
    query : Mapping[str, Any] | None, optional
        STAC query-extension filter, e.g. ``{"eo:cloud_cover": {"lt": 5}}``,
        by default None
    sortby : Sequence[Mapping[str, str]] | None, optional
        Sort spec, e.g. ``[{"field": "properties.eo:cloud_cover",
        "direction": "asc"}]``, by default None
    max_items : int | None, optional
        Maximum number of items to return, by default 100
    **search_kwargs : Any
        Forwarded to :meth:`pystac_client.Client.search` (e.g. ``filter``,
        ``filter_lang``, ``ids``).

    Returns
    -------
    list[pystac.Item]
    """
    prov = get_provider(provider)
    client = Client.open(prov.api_url)
    collections = [collection] if isinstance(collection, str) else list(collection)
    result = client.search(
        collections=collections,
        bbox=list(bbox) if bbox is not None else None,
        intersects=intersects,
        datetime=time_range,
        query=dict(query) if query else None,
        sortby=list(sortby) if sortby else None,
        max_items=max_items,
        **search_kwargs,
    )
    return list(result.items())


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------
_GDAL_DEFAULTS = {
    # Do not list the remote "directory" when opening a single COG
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Fewer, larger range requests when neighbouring tiles are read
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
}


def _configure_gdal() -> None:
    for key, value in _GDAL_DEFAULTS.items():
        os.environ.setdefault(key, value)


@check_optional_dependencies()
def open(  # noqa: A001
    item: Item,
    asset: str,
    *,
    provider: str | StacProvider | None = None,
    chunks: int | Mapping[str, int] | None = 1024,
    rescale: bool = True,
    mask_nodata: bool = True,
) -> xr.DataArray:
    """Open one item asset as a lazy, dask-backed :class:`xarray.DataArray`.

    Nothing beyond the file header is read until values are requested, so
    windowing (``.rio.clip_box``, ``.isel``) before ``.compute()`` fetches only
    the COG tiles covering that window. The array keeps the asset's native
    grid and CRS (``da.rio.crs``, ``da.rio.transform()``).

    Planetary Computer hrefs are signed with a SAS token that is valid for
    roughly an hour and is embedded in the lazy array's URL, so compute
    within that window (or call ``open`` again for a fresh token).
    Earth Search requester-pays buckets are read through GDAL's ``/vsis3/``
    with ``AWS_REQUEST_PAYER=requester`` and need AWS credentials.

    Parameters
    ----------
    item : pystac.Item
        Item returned by :func:`search`.
    asset : str
        Asset key, e.g. ``"red"``.
    provider : str | StacProvider | None, optional
        Provider used to sign the href; inferred from the href host when
        None, by default None
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
        ``(y, x)`` for single-band assets, else ``(band, y, x)``; float32 when
        rescaled or masked, otherwise the file dtype.
    """
    if asset not in item.assets:
        raise KeyError(
            f"Asset {asset!r} not in item {item.id}; available: {sorted(item.assets)}"
        )
    stac_asset = item.assets[asset]
    properties: dict[str, Any] = {**item.properties, **stac_asset.extra_fields}
    prov = (
        get_provider(provider)
        if provider is not None
        else _infer_provider(stac_asset.href)
    )
    url = prov.sign(stac_asset.href, properties)

    _configure_gdal()
    if isinstance(chunks, int):
        chunks = {"band": 1, "x": chunks, "y": chunks}
    da = rioxarray.open_rasterio(url, chunks=chunks, lock=False, masked=False)
    if da.sizes.get("band") == 1:
        da = da.squeeze("band", drop=True)

    bands = stac_asset.extra_fields.get("raster:bands") or []
    band: Mapping[str, Any] = bands[0] if bands else {}
    nodata = band.get("nodata", da.rio.nodata)
    scale = band.get("scale")
    offset = band.get("offset")
    # Earth Search bakes the Sentinel-2 BOA offset into its COGs while the
    # metadata still reports it; applying it again would bias reflectance
    if properties.get("earthsearch:boa_offset_applied"):
        offset = None

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

    da.attrs.update(
        stac_collection=item.collection_id,
        stac_item=item.id,
        stac_asset=asset,
        href=stac_asset.href,
    )
    if item.datetime is not None:
        da = da.expand_dims(time=[_to_datetime64(item.datetime)])
    return da


def _to_datetime64(value: datetime) -> np.datetime64:
    utc = value.astimezone(timezone.utc) if value.tzinfo else value
    return np.datetime64(utc.replace(tzinfo=None))


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


class SearchResult(xr.DataArray):
    """Lazy ``(time, variable, y, x)`` :class:`xarray.DataArray` returned by
    :func:`search`, with the STAC items behind it available as ``.items``.

    Behaves exactly like a DataArray. Derived arrays (``.sel``, arithmetic,
    ``.compute()``) are plain results and report empty ``.items``.
    """

    __slots__ = ("_items",)

    @property
    def items(self) -> tuple[Item, ...]:
        """STAC items the array was assembled from, in ``time`` order."""
        return getattr(self, "_items", ())


def _raster_assets(item: Item) -> list[str]:
    keys = []
    for key, asset in item.assets.items():
        media = (asset.media_type or "").lower()
        if "tiff" in media or asset.href.lower().endswith((".tif", ".tiff")):
            keys.append(key)
    return keys


@check_optional_dependencies()
def search(
    collection: str | Sequence[str],
    assets: str | Sequence[str] | None = None,
    *,
    provider: str | StacProvider = "planetary-computer",
    bbox: Sequence[float] | None = None,
    intersects: Mapping[str, Any] | None = None,
    time_range: str | datetime | Sequence[datetime] | None = None,
    query: Mapping[str, Any] | None = None,
    sortby: Sequence[Mapping[str, str]] | None = None,
    max_items: int | None = 10,
    chunks: int | Mapping[str, int] | None = 1024,
    rescale: bool = True,
    mask_nodata: bool = True,
    lazy: bool = True,
    **search_kwargs: Any,
) -> SearchResult:
    """Search a STAC catalog and open the results as one lazy array.

    Runs the STAC query, opens the requested assets of every matching item
    (see :func:`open`), clips them to ``bbox`` and stacks them into
    Earth2Studio's ``(time, variable, y, x)`` layout in the first item's
    grid. Nothing beyond file headers is read until values are computed, with
    one exception: assets or items on a different grid than the first (other
    resolution, scene or UTM zone) are resampled onto it while ``search`` runs,
    which reads their clipped window eagerly. The matching
    :class:`pystac.Item` objects are available as ``result.items``.

    Parameters
    ----------
    collection : str | Sequence[str]
        Collection id(s), e.g. ``"landsat-c2-l2"``.
    assets : str | Sequence[str] | None, optional
        Asset key(s) to open; these become the ``variable`` coordinate. None
        opens every raster (GeoTIFF/COG) asset of the first item, by default None
    provider : str | StacProvider, optional
        Provider name (``"planetary-computer"``, ``"earth-search"``), STAC
        API URL, or :class:`StacProvider`, by default "planetary-computer"
    bbox : Sequence[float] | None, optional
        ``[min_lon, min_lat, max_lon, max_lat]``; used for the search and to
        clip each asset, by default None (whole scenes)
    intersects : Mapping[str, Any] | None, optional
        GeoJSON geometry to search with, by default None
    time_range : str | datetime | Sequence[datetime] | None, optional
        ISO ``"start/end"`` string, a single datetime, or a (start, end)
        pair, by default None
    query : Mapping[str, Any] | None, optional
        STAC query-extension filter, e.g. ``{"eo:cloud_cover": {"lt": 5}}``,
        by default None
    sortby : Sequence[Mapping[str, str]] | None, optional
        Sort spec, e.g. ``[{"field": "properties.eo:cloud_cover",
        "direction": "asc"}]``, by default None
    max_items : int | None, optional
        Maximum number of items, by default 10
    chunks : int | Mapping[str, int] | None, optional
        Dask chunk size per asset (see :func:`open`), by default 1024
    rescale : bool, optional
        Apply ``raster:bands`` scale/offset, by default True
    mask_nodata : bool, optional
        Replace nodata with NaN, by default True
    lazy : bool, optional
        Return a dask-backed array; False computes before returning, by
        default True
    **search_kwargs : Any
        Forwarded to :meth:`pystac_client.Client.search` (e.g. ``filter``,
        ``filter_lang``, ``ids``).

    Returns
    -------
    SearchResult
        Lazy ``(time, variable, y, x)`` array sorted by time, with
        ``.rio.crs``/``.rio.transform()`` describing the grid and ``.items``
        holding the STAC items.

    Raises
    ------
    FileNotFoundError
        If the search returns no items.
    """
    items = _search_items(
        collection,
        provider=provider,
        bbox=bbox,
        intersects=intersects,
        time_range=time_range,
        query=query,
        sortby=sortby,
        max_items=max_items,
        **search_kwargs,
    )
    if not items:
        raise FileNotFoundError(
            f"No items found in {collection!r} for bbox={bbox} time_range={time_range}"
        )
    if assets is None:
        asset_keys = _raster_assets(items[0])
        if not asset_keys:
            raise ValueError(
                f"No raster assets found on {items[0].id}; pass assets= explicitly "
                f"(available: {sorted(items[0].assets)})"
            )
    else:
        asset_keys = [assets] if isinstance(assets, str) else list(assets)

    per_item: list[xr.DataArray] = []
    reference: xr.DataArray | None = None
    for item in items:
        layers: list[xr.DataArray] = []
        for key in asset_keys:
            da = open(
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
    out.attrs["stac_collection"] = (
        collection if isinstance(collection, str) else ",".join(collection)
    )
    out.attrs["stac_items"] = [items[i].id for i in order]
    result = SearchResult(out)
    result._items = tuple(items[i] for i in order)
    return result


# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StacCollectionInfo:
    """Summary of a STAC collection, as returned by :func:`collections`."""

    collection_id: str
    title: str
    asset_keys: tuple[str, ...]
    temporal_extent: tuple[datetime | None, datetime | None]
    spatial_extent: tuple[float, float, float, float] | None


def _bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _parse_time_range(
    time_range: str | datetime | Sequence[datetime | None],
) -> tuple[datetime | None, datetime | None]:
    """Parse the ``time_range`` forms :func:`search` accepts into UTC bounds.

    Open ends (``".."`` or empty) become None; a date-only end bound is
    extended to the end of that day so ``"2024-01-01/2024-12-31"`` includes
    the 31st.
    """
    if isinstance(time_range, datetime):
        bound = _as_utc(time_range)
        return bound, bound

    def parse(text: str, *, end: bool) -> datetime | None:
        if text in ("", ".."):
            return None
        value = _as_utc(datetime.fromisoformat(text))
        if end and "T" not in text and " " not in text:
            value = value + timedelta(days=1) - timedelta(microseconds=1)
        return value

    if isinstance(time_range, str):
        start_text, sep, end_text = time_range.partition("/")
        if not sep:
            end_text = start_text
        return parse(start_text, end=False), parse(end_text, end=True)
    start, end = time_range
    return (
        _as_utc(start) if start is not None else None,
        _as_utc(end) if end is not None else None,
    )


@check_optional_dependencies()
def collections(
    provider: str | StacProvider = "planetary-computer",
    *,
    bbox: Sequence[float] | None = None,
    time_range: str | Sequence[datetime] | None = None,
    text: str | None = None,
    sample_assets: bool = True,
) -> list[StacCollectionInfo]:
    """Browse a provider's catalog to find collection ids and asset keys.

    Filtering is client-side over ``/collections`` (neither Planetary
    Computer nor Earth Search serve a collection-search extension).

    Parameters
    ----------
    provider : str | StacProvider, optional
        Provider name, STAC API URL, or :class:`StacProvider`, by default
        "planetary-computer"
    bbox : Sequence[float] | None, optional
        Keep collections whose spatial extent intersects this box, by default None
    time_range : str | Sequence[datetime] | None, optional
        ISO ``"start/end"`` or (start, end); keep collections whose temporal
        extent overlaps, by default None
    text : str | None, optional
        Case-insensitive substring matched on id and title, by default None
    sample_assets : bool, optional
        Fetch one item per matching collection to list its asset keys (one
        extra request per collection), by default True

    Returns
    -------
    list[StacCollectionInfo]
    """
    prov = get_provider(provider)
    client = Client.open(prov.api_url)
    query_range = _parse_time_range(time_range) if time_range is not None else None
    needle = text.lower() if text else None

    infos: list[StacCollectionInfo] = []
    for collection in client.get_collections():
        title = collection.title or collection.id
        if (
            needle
            and needle not in collection.id.lower()
            and needle not in title.lower()
        ):
            continue

        bboxes = collection.extent.spatial.bboxes or []
        spatial = None
        for b in bboxes:
            # 3D bboxes carry min/max elevation at indices 2 and 5
            lonlat = (b[0], b[1], b[3], b[4]) if len(b) == 6 else tuple(b[:4])
            spatial = (
                lonlat
                if spatial is None
                else (
                    min(spatial[0], lonlat[0]),
                    min(spatial[1], lonlat[1]),
                    max(spatial[2], lonlat[2]),
                    max(spatial[3], lonlat[3]),
                )
            )
        if (
            bbox is not None
            and spatial is not None
            and not _bbox_intersects(bbox, spatial)
        ):
            continue

        intervals = collection.extent.temporal.intervals or [[None, None]]
        start, end = intervals[0]
        if query_range is not None:
            q_start, q_end = query_range
            if q_end is not None and start is not None and _as_utc(start) > q_end:
                continue
            if q_start is not None and end is not None and _as_utc(end) < q_start:
                continue

        asset_keys: tuple[str, ...] = ()
        if sample_assets:
            sample = next(
                client.search(collections=[collection.id], max_items=1).items(), None
            )
            if sample is not None:
                asset_keys = tuple(sorted(sample.assets))

        infos.append(
            StacCollectionInfo(
                collection_id=collection.id,
                title=title,
                asset_keys=asset_keys,
                temporal_extent=(start, end),
                spatial_extent=spatial,  # type: ignore[arg-type]
            )
        )
    return infos
