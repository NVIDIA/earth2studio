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

import io
import json
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import stac

# ---------------------------------------------------------------------------
# Fakes standing in for pystac / pystac_client objects
# ---------------------------------------------------------------------------


class _FakeAsset:
    def __init__(self, href, extra_fields=None, media_type="image/tiff"):
        self.href = href
        self.media_type = media_type
        self.extra_fields = extra_fields or {}


class _FakeItem:
    def __init__(self, id, assets, properties=None, collection_id="c", when=None):
        self.id = id
        self.assets = assets
        self.properties = properties or {}
        self.collection_id = collection_id
        self.datetime = when


class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeCollection:
    def __init__(self, id, title, bboxes, intervals):
        self.id = id
        self.title = title
        self.extent = type("E", (), {})()
        self.extent.spatial = type("S", (), {"bboxes": bboxes})()
        self.extent.temporal = type("T", (), {"intervals": intervals})()


class _FakeClient:
    instance = None

    def __init__(self, collections=(), items_by_collection=None):
        self._collections = list(collections)
        self._items = items_by_collection or {}
        self.opened_url = None
        self.search_calls = []

    @classmethod
    def open(cls, url):
        cls.instance.opened_url = url
        return cls.instance

    def get_collections(self):
        return iter(self._collections)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return _FakeSearch(self._items.get(kwargs["collections"][0], []))


@pytest.fixture
def fake_client(monkeypatch):
    utc = timezone.utc
    collections = [
        _FakeCollection(
            "landsat-c2-l2",
            "Landsat Collection 2 Level-2",
            [[-180, -90, 180, 90]],
            [[datetime(1982, 8, 22, tzinfo=utc), None]],
        ),
        _FakeCollection(
            "naip",
            "NAIP",
            # Two bboxes: CONUS and Hawaii; union must be used
            [[-125, 24, -66, 50], [-160, 18, -154, 23]],
            [[datetime(2010, 1, 1, tzinfo=utc), datetime(2022, 12, 31, tzinfo=utc)]],
        ),
        _FakeCollection(
            "cop-dem-glo-30",
            None,
            [[-180, -90, -500, 180, 90, 9000]],
            [[None, None]],
        ),
    ]
    items = {
        "landsat-c2-l2": [
            _FakeItem("L1", {"red": _FakeAsset("x"), "nir08": _FakeAsset("y")})
        ],
        "naip": [_FakeItem("N1", {"image": _FakeAsset("z")})],
    }
    client = _FakeClient(collections, items)
    _FakeClient.instance = client
    monkeypatch.setattr(stac, "Client", _FakeClient)
    return client


# ---------------------------------------------------------------------------
# Providers and signing
# ---------------------------------------------------------------------------


def test_get_provider() -> None:
    assert stac.get_provider("planetary-computer").api_url.endswith("/stac/v1")
    assert stac.get_provider("earth-search").name == "earth-search"
    custom = stac.get_provider("https://example.com/stac/v1")
    assert custom.api_url == "https://example.com/stac/v1"
    assert custom.sign("href", {}) == "href"
    assert stac.get_provider(custom) is custom
    with pytest.raises(ValueError, match="Unknown STAC provider"):
        stac.get_provider("nope")


def test_infer_provider() -> None:
    assert stac._infer_provider("https://acct.blob.core.windows.net/c/k.tif").name == (
        "planetary-computer"
    )
    assert stac._infer_provider("s3://bucket/k.tif").name == "earth-search"
    assert stac._infer_provider("https://b.s3.us-west-2.amazonaws.com/k.tif").name == (
        "earth-search"
    )
    assert stac._infer_provider("/local/file.tif").name == "generic"


def test_sign_public_s3() -> None:
    assert stac._sign_public_s3("s3://sentinel-cogs/a/b.tif", {}) == (
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/a/b.tif"
    )
    assert (
        stac._sign_public_s3(
            "s3://copernicus-dem-30m/x.tif", {"storage:region": "eu-central-1"}
        )
        == "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/x.tif"
    )
    https = "https://b.s3.us-west-2.amazonaws.com/k.tif"
    assert stac._sign_public_s3(https, {}) == https


def test_sign_planetary_computer_caches_token(monkeypatch) -> None:
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        body = json.dumps(
            {"token": "sig=abc", "msft:expiry": "2999-01-01T00:00:00Z"}
        ).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(stac.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(stac, "_pc_tokens", {})

    href = "https://landsateuwest.blob.core.windows.net/landsat-c2/a/b.tif"
    signed = stac._sign_planetary_computer(href, {})
    assert signed == href + "?sig=abc"
    assert calls == [f"{stac._PC_SAS_URL}/landsateuwest/landsat-c2"]

    # Same container -> cached; different container -> new request
    stac._sign_planetary_computer(href.replace("b.tif", "c.tif"), {})
    stac._sign_planetary_computer(
        "https://landsateuwest.blob.core.windows.net/other/x.tif", {}
    )
    assert len(calls) == 2

    # Already signed or non-blob hrefs pass through untouched
    assert stac._sign_planetary_computer(signed, {}) == signed
    assert stac._sign_planetary_computer("s3://b/k", {}) == "s3://b/k"


def test_sign_planetary_computer_refreshes_expiring_token(monkeypatch) -> None:
    def fake_urlopen(url, timeout):
        return io.BytesIO(
            json.dumps({"token": "new", "msft:expiry": "2999-01-01T00:00:00Z"}).encode()
        )

    monkeypatch.setattr(stac.urllib.request, "urlopen", fake_urlopen)
    soon = datetime.now(timezone.utc) + stac._PC_TOKEN_REFRESH_MARGIN / 2
    monkeypatch.setattr(stac, "_pc_tokens", {("a", "c"): ("old", soon)})
    assert stac._planetary_computer_token("a", "c") == "new"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_items_forwards_arguments(fake_client) -> None:
    items = stac._search_items(
        "landsat-c2-l2",
        bbox=(-122.6, 37.6, -122.2, 37.9),
        time_range="2024-10-27/2024-12-31",
        query={"eo:cloud_cover": {"lt": 1}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=10,
        filter_lang="cql2-json",
    )
    assert [i.id for i in items] == ["L1"]
    assert fake_client.opened_url == stac.PROVIDERS["planetary-computer"].api_url
    call = fake_client.search_calls[0]
    assert call["collections"] == ["landsat-c2-l2"]
    assert call["bbox"] == [-122.6, 37.6, -122.2, 37.9]
    assert call["datetime"] == "2024-10-27/2024-12-31"
    assert call["query"] == {"eo:cloud_cover": {"lt": 1}}
    assert call["sortby"][0]["field"] == "properties.eo:cloud_cover"
    assert call["max_items"] == 10
    assert call["filter_lang"] == "cql2-json"


def test_search_items_other_provider_and_multiple_collections(fake_client) -> None:
    stac._search_items(
        ["naip", "landsat-c2-l2"], provider="earth-search", max_items=None
    )
    assert fake_client.opened_url == stac.PROVIDERS["earth-search"].api_url
    assert fake_client.search_calls[0]["collections"] == ["naip", "landsat-c2-l2"]
    assert fake_client.search_calls[0]["query"] is None


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


@pytest.fixture
def cog(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "red.tif"
    dn = np.array([[0, 1000, 2000], [10000, 40000, 65535]], dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32610",
        transform=from_origin(500000, 4200000, 30, 30),
    ) as dst:
        dst.write(dn, 1)
    return path, dn


def _landsat_item(path, bands=None, properties=None):
    return _FakeItem(
        "LC09_L2SP_045034_20241209_02_T1",
        {"red": _FakeAsset(str(path), {"raster:bands": bands} if bands else {})},
        properties=properties,
        collection_id="landsat-c2-l2",
        when=datetime(2024, 12, 9, 18, 45, tzinfo=timezone.utc),
    )


def test_open_is_lazy_and_keeps_native_grid(cog) -> None:
    path, dn = cog
    da = stac.open(_landsat_item(path), "red", rescale=False, mask_nodata=False)
    assert da.chunks is not None  # dask-backed, nothing read yet
    assert da.dims == ("time", "y", "x")
    assert da.dtype == np.uint16
    assert da.rio.crs.to_epsg() == 32610
    assert da.rio.transform().a == 30
    assert da.attrs["stac_item"] == "LC09_L2SP_045034_20241209_02_T1"
    assert da.attrs["stac_asset"] == "red"
    assert da.time.values[0] == np.datetime64("2024-12-09T18:45:00")
    np.testing.assert_array_equal(da.isel(time=0).values, dn)


def test_open_applies_scale_offset_and_masks_nodata(cog) -> None:
    path, dn = cog
    bands = [{"nodata": 0, "scale": 2.75e-05, "offset": -0.2, "data_type": "uint16"}]
    da = stac.open(_landsat_item(path, bands), "red").isel(time=0)
    assert da.dtype == np.float32
    values = da.values
    assert np.isnan(values[0, 0])
    np.testing.assert_allclose(values[0, 1], 1000 * 2.75e-05 - 0.2, rtol=1e-5)
    np.testing.assert_allclose(values[1, 0], 10000 * 2.75e-05 - 0.2, rtol=1e-5)
    assert np.isnan(da.rio.nodata)


def test_open_falls_back_to_file_nodata(cog) -> None:
    path, _ = cog
    da = stac.open(_landsat_item(path), "red", rescale=False).isel(time=0)
    assert np.isnan(da.values[0, 0]) and da.values[0, 1] == 1000


def test_open_skips_offset_when_earth_search_already_applied(cog) -> None:
    path, _ = cog
    bands = [{"nodata": 0, "scale": 1e-4, "offset": -0.1}]
    item = _landsat_item(path, bands, {"earthsearch:boa_offset_applied": True})
    da = stac.open(item, "red").isel(time=0)
    np.testing.assert_allclose(da.values[0, 1], 0.1, rtol=1e-5)


def test_open_chunks_and_missing_asset(cog) -> None:
    path, _ = cog
    da = stac.open(_landsat_item(path), "red", chunks={"x": 2, "y": 1})
    assert da.chunks[-1] == (2, 1)
    eager = stac.open(_landsat_item(path), "red", chunks=None)
    assert eager.chunks is None
    with pytest.raises(KeyError, match="available: \\['red'\\]"):
        stac.open(_landsat_item(path), "nir08")


def test_open_uses_provider_signer(cog, monkeypatch) -> None:
    path, _ = cog
    seen = []

    def sign(href, props):
        seen.append((href, props.get("storage:region")))
        return href

    prov = stac.StacProvider("custom", "https://x", sign)
    item = _landsat_item(path, properties={"storage:region": "eu-central-1"})
    stac.open(item, "red", provider=prov, rescale=False, mask_nodata=False)
    assert seen == [(str(path), "eu-central-1")]


# ---------------------------------------------------------------------------
# search (stacked result)
# ---------------------------------------------------------------------------


def _write_tif(path, values, *, origin=(500000, 4200000), res=30, crs="EPSG:32610"):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    bands = values if values.ndim == 3 else values[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=str(values.dtype),
        nodata=0,
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(bands)
    return str(path)


def _scene(tmp_path, tag, when, red, nir, **grid):
    return _FakeItem(
        f"scene-{tag}",
        {
            "red": _FakeAsset(_write_tif(tmp_path / f"{tag}_red.tif", red, **grid)),
            "nir08": _FakeAsset(_write_tif(tmp_path / f"{tag}_nir.tif", nir, **grid)),
        },
        collection_id="landsat-c2-l2",
        when=when,
    )


def test_search_stacks_time_and_variable_lazily(tmp_path, fake_client) -> None:
    utc = timezone.utc
    a = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
    later = _scene(tmp_path, "b", datetime(2024, 12, 9, tzinfo=utc), a * 2, a * 3)
    earlier = _scene(tmp_path, "a", datetime(2024, 11, 7, tzinfo=utc), a, a + 1)
    fake_client._items["landsat-c2-l2"] = [later, earlier]

    da = stac.search("landsat-c2-l2", ["red", "nir08"], rescale=False)

    assert isinstance(da, stac.SearchResult) and isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "y", "x")
    assert da.shape == (2, 2, 3, 4)
    assert da.chunks is not None
    assert list(da.coords["variable"].values) == ["red", "nir08"]
    # Sorted by time regardless of item order; .items follows the same order
    assert da.time.values[0] == np.datetime64("2024-11-07")
    assert [i.id for i in da.items] == ["scene-a", "scene-b"]
    assert da.attrs["stac_items"] == ["scene-a", "scene-b"]
    assert da.rio.crs.to_epsg() == 32610
    assert da.rio.transform().a == 30
    np.testing.assert_array_equal(da.sel(variable="red").isel(time=0).values, a)
    np.testing.assert_array_equal(da.sel(variable="nir08").isel(time=1).values, a * 3)
    # Derived arrays are ordinary results without items
    assert da.sel(variable="red").items == ()
    assert (da * 2).items == ()


def test_search_default_assets_and_eager(tmp_path, fake_client) -> None:
    a = np.ones((2, 2), dtype=np.uint16)
    item = _scene(tmp_path, "s", datetime(2024, 12, 9, tzinfo=timezone.utc), a, a)
    item.assets["qa_json"] = _FakeAsset("meta.json", media_type="application/json")
    fake_client._items["landsat-c2-l2"] = [item]

    da = stac.search("landsat-c2-l2", rescale=False, lazy=False)
    assert list(da.coords["variable"].values) == ["red", "nir08"]  # json skipped
    assert da.chunks is None  # computed
    assert [i.id for i in da.items] == ["scene-s"]


def test_search_clips_to_bbox(tmp_path, fake_client) -> None:
    utc = timezone.utc
    a = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    item = _scene(tmp_path, "s", datetime(2024, 12, 9, tzinfo=utc), a, a)
    fake_client._items["landsat-c2-l2"] = [item]

    # A lon/lat box covering the 3x3 pixel block at the scene's top-left
    from rasterio.warp import transform_bounds

    bbox = list(
        transform_bounds(
            "EPSG:32610",
            "EPSG:4326",
            500000,
            4200000 - 3 * 30,
            500000 + 3 * 30,
            4200000,
        )
    )
    da = stac.search(
        "landsat-c2-l2",
        "red",
        bbox=bbox,
        time_range="2024-12-01/2024-12-31",
        max_items=1,
        rescale=False,
    )
    assert fake_client.search_calls[0]["bbox"] == bbox
    assert fake_client.search_calls[0]["datetime"] == "2024-12-01/2024-12-31"
    assert da.shape[0] == 1 and da.shape[1] == 1
    assert da.shape[2] < 10 and da.shape[3] < 10


def test_search_aligns_items_on_different_grids(tmp_path, fake_client) -> None:
    utc = timezone.utc
    a = np.full((4, 4), 7, dtype=np.uint16)
    ref = _scene(tmp_path, "ref", datetime(2024, 11, 7, tzinfo=utc), a, a)
    # Same CRS, shifted origin: must be resampled onto the first grid
    shifted = _scene(
        tmp_path,
        "shift",
        datetime(2024, 12, 9, tzinfo=utc),
        a * 2,
        a * 2,
        origin=(500015, 4200015),
    )
    fake_client._items["landsat-c2-l2"] = [ref, shifted]
    da = stac.search("landsat-c2-l2", "red", rescale=False)
    assert da.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(da.x.values, stac.open(ref, "red").x.values)
    assert float(da.isel(time=1).max()) == 14


def test_search_no_items(fake_client) -> None:
    fake_client._items["landsat-c2-l2"] = []
    with pytest.raises(FileNotFoundError, match="No items found"):
        stac.search("landsat-c2-l2", "red", bbox=[0, 0, 1, 1])


# ---------------------------------------------------------------------------
# collections
# ---------------------------------------------------------------------------


def test_collections_all(fake_client) -> None:
    infos = stac.collections()
    by_id = {i.collection_id: i for i in infos}
    assert set(by_id) == {"landsat-c2-l2", "naip", "cop-dem-glo-30"}
    assert by_id["landsat-c2-l2"].asset_keys == ("nir08", "red")
    assert by_id["landsat-c2-l2"].temporal_extent[1] is None
    # Multiple bboxes are unioned; 3D bbox reduced to lon/lat
    assert by_id["naip"].spatial_extent == (-160, 18, -66, 50)
    assert by_id["cop-dem-glo-30"].spatial_extent == (-180, -90, 180, 90)
    # Title falls back to id; no items -> no asset keys
    assert by_id["cop-dem-glo-30"].title == "cop-dem-glo-30"
    assert by_id["cop-dem-glo-30"].asset_keys == ()


def test_collections_filters(fake_client) -> None:
    assert [
        i.collection_id for i in stac.collections(text="LANDSAT", sample_assets=False)
    ] == ["landsat-c2-l2"]
    # Hawaii falls in NAIP's second bbox
    ids = {
        i.collection_id
        for i in stac.collections(bbox=(-158, 20, -157, 21), sample_assets=False)
    }
    assert "naip" in ids
    # Europe excludes NAIP
    ids = {
        i.collection_id
        for i in stac.collections(bbox=(0, 40, 20, 55), sample_assets=False)
    }
    assert "naip" not in ids and "landsat-c2-l2" in ids
    assert fake_client.search_calls == []


@pytest.mark.parametrize(
    "time_range",
    [
        "2023-01-01/2023-12-31",
        (datetime(2023, 1, 1), datetime(2023, 12, 31)),
        (
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        ),
    ],
)
def test_collections_time_range_naive_and_aware(fake_client, time_range) -> None:
    ids = {
        i.collection_id
        for i in stac.collections(time_range=time_range, sample_assets=False)
    }
    # NAIP ended 2022; open-ended Landsat and unbounded DEM remain
    assert ids == {"landsat-c2-l2", "cop-dem-glo-30"}


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def test_search_multiband_asset_becomes_variables(tmp_path, fake_client) -> None:
    rgb = np.stack([np.full((2, 2), v, dtype=np.uint8) for v in (10, 20, 30)])
    item = _FakeItem(
        "tci",
        {"visual": _FakeAsset(_write_tif(tmp_path / "visual.tif", rgb))},
        when=datetime(2025, 6, 2, tzinfo=timezone.utc),
    )
    fake_client._items["landsat-c2-l2"] = [item]
    da = stac.search("landsat-c2-l2", rescale=False, mask_nodata=False)
    assert da.dims == ("time", "variable", "y", "x")
    assert list(da.coords["variable"].values) == ["visual_1", "visual_2", "visual_3"]
    assert float(da.sel(variable="visual_3").max()) == 30


def test_search_item_without_datetime(tmp_path, fake_client) -> None:
    a = np.ones((2, 2), dtype=np.uint16)
    item = _scene(tmp_path, "dem", None, a, a)
    fake_client._items["landsat-c2-l2"] = [item]
    da = stac.search("landsat-c2-l2", "red", rescale=False)
    assert da.shape == (1, 1, 2, 2)
    assert np.isnat(da.time.values[0])


def test_search_forwards_provider_signer(tmp_path, fake_client) -> None:
    a = np.ones((2, 2), dtype=np.uint16)
    item = _scene(tmp_path, "p", datetime(2025, 1, 1, tzinfo=timezone.utc), a, a)
    fake_client._items["landsat-c2-l2"] = [item]
    seen = []

    def sign(href, props):
        seen.append(href)
        return href

    prov = stac.StacProvider("custom", "https://my-api/v1", sign)
    stac.search("landsat-c2-l2", "red", provider=prov, rescale=False)
    assert seen == [item.assets["red"].href]
    assert fake_client.opened_url == "https://my-api/v1"


def test_search_resamples_assets_at_different_resolution(tmp_path, fake_client) -> None:
    fine = np.full((4, 4), 7, dtype=np.uint16)
    coarse = np.full((2, 2), 9, dtype=np.uint16)
    item = _FakeItem(
        "mixed",
        {
            "red": _FakeAsset(_write_tif(tmp_path / "red.tif", fine, res=30)),
            "swir16": _FakeAsset(_write_tif(tmp_path / "swir.tif", coarse, res=60)),
        },
        when=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    fake_client._items["landsat-c2-l2"] = [item]
    da = stac.search("landsat-c2-l2", ["red", "swir16"], rescale=False)
    # On the first asset's 4x4 grid, no NaN holes from coordinate outer-join
    assert da.shape == (1, 2, 4, 4)
    assert not np.isnan(da.values).any()
    assert float(da.sel(variable="swir16").min()) == 9


def test_sign_public_s3_requester_pays(monkeypatch) -> None:
    monkeypatch.delenv("AWS_REQUEST_PAYER", raising=False)
    href = "s3://usgs-landsat/collection02/x.tif"
    assert stac._sign_public_s3(href, {"storage:requester_pays": True}) == (
        "/vsis3/usgs-landsat/collection02/x.tif"
    )
    import os

    assert os.environ["AWS_REQUEST_PAYER"] == "requester"


def test_open_rescaled_nodata_stays_consistent(cog) -> None:
    path, _ = cog
    bands = [{"nodata": 0, "scale": 2.75e-05, "offset": -0.2}]
    da = stac.open(_landsat_item(path, bands), "red", mask_nodata=False).isel(time=0)
    # Fill pixels now hold -0.2, and the declared nodata says so
    np.testing.assert_allclose(da.values[0, 0], -0.2, rtol=1e-6)
    np.testing.assert_allclose(da.rio.nodata, -0.2, rtol=1e-6)


def test_parse_time_range_forms() -> None:
    utc = timezone.utc
    day_end = datetime(2020, 1, 1, 23, 59, 59, 999999, tzinfo=utc)
    assert stac._parse_time_range("2020-01-01/..") == (
        datetime(2020, 1, 1, tzinfo=utc),
        None,
    )
    assert stac._parse_time_range("../2020-01-01") == (None, day_end)
    assert stac._parse_time_range("2020-01-01") == (
        datetime(2020, 1, 1, tzinfo=utc),
        day_end,
    )
    assert stac._parse_time_range("2020-01-01T06:00/2020-01-02T06:00") == (
        datetime(2020, 1, 1, 6, tzinfo=utc),
        datetime(2020, 1, 2, 6, tzinfo=utc),
    )
    assert stac._parse_time_range(datetime(2020, 1, 1)) == (
        datetime(2020, 1, 1, tzinfo=utc),
        datetime(2020, 1, 1, tzinfo=utc),
    )


def test_collections_open_ended_time_range(fake_client) -> None:
    ids = {
        i.collection_id
        for i in stac.collections(time_range="2023-01-01/..", sample_assets=False)
    }
    assert ids == {"landsat-c2-l2", "cop-dem-glo-30"}
    # Single day inside NAIP's extent keeps NAIP
    ids = {
        i.collection_id
        for i in stac.collections(time_range="2022-12-31", sample_assets=False)
    }
    assert "naip" in ids


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(120)
def test_live_planetary_computer_landsat_window() -> None:
    da = stac.search(
        "landsat-c2-l2",
        ["red", "nir08"],
        bbox=[-122.55, 37.70, -122.35, 37.85],
        time_range="2024-10-27/2024-12-31",
        query={"eo:cloud_cover": {"lt": 5}},
        max_items=2,
    )
    assert len(da.items) >= 1 and da.chunks is not None
    assert da.dims == ("time", "variable", "y", "x")
    assert da.rio.crs.to_epsg() == 32610
    red = da.sel(variable="red").isel(time=0).compute()
    valid = red.values[np.isfinite(red.values)]
    assert valid.size > 0 and valid.min() >= -0.2 and valid.max() <= 1.6
    # Single-asset path on a real item, raw dtype preserved
    qa = stac.open(da.items[0], "qa_pixel", rescale=False, mask_nodata=False)
    assert qa.dtype == np.uint16 and qa.chunks is not None


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(120)
def test_live_earth_search_sentinel2_window() -> None:
    da = stac.search(
        "sentinel-2-l2a",
        "red",
        provider="earth-search",
        bbox=[-122.55, 37.70, -122.35, 37.85],
        query={"grid:code": {"eq": "MGRS-10SEG"}},
        time_range="2025-06-01/2025-06-04",
        sortby=[{"field": "properties.s2:nodata_pixel_percentage", "direction": "asc"}],
        max_items=1,
    )
    assert da.rio.crs.to_epsg() == 32610 and len(da.items) == 1
    red = da.isel(time=0, variable=0).compute()
    valid = red.values[np.isfinite(red.values)]
    assert valid.size > 0 and valid.min() >= 0.0 and np.nanmedian(valid) < 0.5
