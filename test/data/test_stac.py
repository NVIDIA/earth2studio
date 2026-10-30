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

import functools
import http.server
import os
import socketserver
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

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
            # First bbox is the overall extent (CONUS), the rest sub-extents
            [[-125, 18, -66, 50], [-125, 24, -66, 50], [-160, 18, -154, 23]],
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


def _write_tif(
    path,
    values,
    *,
    origin=(500000, 4200000),
    res=30,
    crs="EPSG:32610",
    nodata=0,
    scale=None,
    offset=None,
):
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
        nodata=nodata,
        crs=crs,
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(bands)
        if scale is not None:
            dst.scales = [scale] * bands.shape[0]
            dst.offsets = [offset or 0.0] * bands.shape[0]
    return str(path)


def _landsat_item(path, bands=None, properties=None):
    return _FakeItem(
        "LC09_L2SP_045034_20241209_02_T1",
        {"red": _FakeAsset(str(path), {"raster:bands": bands} if bands else {})},
        properties=properties,
        collection_id="landsat-c2-l2",
        when=datetime(2024, 12, 9, 18, 45, tzinfo=timezone.utc),
    )


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


@pytest.fixture
def cog(tmp_path):
    dn = np.array([[0, 1000, 2000], [10000, 40000, 65535]], dtype=np.uint16)
    return _write_tif(tmp_path / "red.tif", dn), dn


# ---------------------------------------------------------------------------
# Providers and href signing
# ---------------------------------------------------------------------------


def test_providers_sign_hrefs(monkeypatch) -> None:
    pc_href = "https://landsateuwest.blob.core.windows.net/landsat-c2/a/b.tif"
    assert stac._infer_provider(pc_href) is stac.PLANETARY_COMPUTER
    assert stac._infer_provider("s3://bucket/k.tif") is stac.EARTH_SEARCH
    assert stac._infer_provider("https://b.s3.us-west-2.amazonaws.com/k.tif") is (
        stac.EARTH_SEARCH
    )
    assert stac._infer_provider("/local/file.tif").name == "generic"

    # Planetary Computer: GDAL signs and renews the SAS token itself, one
    # token per collection when the collection is known
    assert stac.PLANETARY_COMPUTER.sign(pc_href, {"collection": "landsat-c2-l2"}) == (
        "/vsicurl?pc_url_signing=yes&pc_collection=landsat-c2-l2&empty_dir=yes"
        "&max_retry=3&retry_delay=2&url="
        "https%3A%2F%2Flandsateuwest.blob.core.windows.net%2Flandsat-c2%2Fa%2Fb.tif"
    )
    assert "pc_collection" not in stac.PLANETARY_COMPUTER.sign(pc_href, {})
    # Earth Search: public s3:// read anonymously over HTTPS in the bucket region
    assert stac.EARTH_SEARCH.sign("s3://sentinel-cogs/a/b.tif", {}) == (
        "/vsicurl?empty_dir=yes&max_retry=3&retry_delay=2&url="
        "https%3A%2F%2Fsentinel-cogs.s3.us-west-2.amazonaws.com%2Fa%2Fb.tif"
    )
    assert "copernicus-dem-30m.s3.eu-central-1.amazonaws.com" in stac.EARTH_SEARCH.sign(
        "s3://copernicus-dem-30m/x.tif", {"storage:region": "eu-central-1"}
    )
    # Requester-pays buckets go through /vsis3/ and never touch the environment
    monkeypatch.delenv("AWS_REQUEST_PAYER", raising=False)
    assert stac.EARTH_SEARCH.sign(
        "s3://usgs-landsat/collection02/x.tif", {"storage:requester_pays": True}
    ) == ("/vsis3/usgs-landsat/collection02/x.tif")
    assert "AWS_REQUEST_PAYER" not in os.environ
    # Non-matching hrefs: http(s) is wrapped in /vsicurl, anything else untouched
    https = "https://example.com/k.tif"
    assert stac.EARTH_SEARCH.sign(https, {}) == stac.PLANETARY_COMPUTER.sign(https, {})
    assert stac.EARTH_SEARCH.sign(https, {}).startswith(
        "/vsicurl?empty_dir=yes&max_retry=3&retry_delay=2&url="
    )
    assert stac.StacProvider("custom", "https://x").sign("/local/k.tif", {}) == (
        "/local/k.tif"
    )


# ---------------------------------------------------------------------------
# open_asset
# ---------------------------------------------------------------------------


def test_open_asset_is_lazy_and_keeps_native_grid(cog) -> None:
    path, dn = cog
    da = stac.open_asset(_landsat_item(path), "red", rescale=False, mask_nodata=False)
    assert da.chunks is not None  # dask-backed, nothing read yet
    assert da.dims == ("time", "y", "x")
    assert da.dtype == np.uint16
    assert da.rio.crs.to_epsg() == 32610
    assert da.rio.transform().a == 30
    assert da.attrs["stac_item"] == "LC09_L2SP_045034_20241209_02_T1"
    assert da.attrs["stac_asset"] == "red"
    assert da.time.values[0] == np.datetime64("2024-12-09T18:45:00")
    np.testing.assert_array_equal(da.isel(time=0).values, dn)

    # Chunk spec, eager load, missing asset and a custom provider signer
    assert stac.open_asset(_landsat_item(path), "red", chunks={"x": 2, "y": 1}).chunks[
        -1
    ] == (2, 1)
    assert stac.open_asset(_landsat_item(path), "red", chunks=None).chunks is None
    with pytest.raises(KeyError, match="available: \\['red'\\]"):
        stac.open_asset(_landsat_item(path), "nir08")
    seen = []
    prov = stac.StacProvider(
        "custom", "https://x", lambda href, props: seen.append(props) or href
    )
    item = _landsat_item(path, properties={"storage:region": "eu-central-1"})
    stac.open_asset(item, "red", provider=prov, rescale=False, mask_nodata=False)
    assert seen[0]["storage:region"] == "eu-central-1"


def test_open_asset_applies_scale_offset_and_nodata(cog) -> None:
    path, _ = cog
    bands = [{"nodata": 0, "scale": 2.75e-05, "offset": -0.2, "data_type": "uint16"}]
    da = stac.open_asset(_landsat_item(path, bands), "red").isel(time=0)
    assert da.dtype == np.float32
    assert np.isnan(da.values[0, 0])
    np.testing.assert_allclose(da.values[0, 1], 1000 * 2.75e-05 - 0.2, rtol=1e-5)
    np.testing.assert_allclose(da.values[1, 0], 10000 * 2.75e-05 - 0.2, rtol=1e-5)
    assert np.isnan(da.rio.nodata)

    # Without raster:bands the file nodata is masked
    da = stac.open_asset(_landsat_item(path), "red", rescale=False).isel(time=0)
    assert np.isnan(da.values[0, 0]) and da.values[0, 1] == 1000

    # Unmasked: fill pixels hold the rescaled value and the declared nodata agrees
    da = stac.open_asset(_landsat_item(path, bands), "red", mask_nodata=False).isel(
        time=0
    )
    np.testing.assert_allclose(da.values[0, 0], -0.2, rtol=1e-6)
    np.testing.assert_allclose(da.rio.nodata, -0.2, rtol=1e-6)

    # Earth Search already applied the BOA offset in the file
    item = _landsat_item(path, bands, {"earthsearch:boa_offset_applied": True})
    da = stac.open_asset(item, "red").isel(time=0)
    np.testing.assert_allclose(da.values[0, 1], 1000 * 2.75e-05, rtol=1e-5)

    # Without raster:bands the file's own scale/offset tags are applied and
    # removed from attrs so CF decoders do not apply them again
    tagged = _write_tif(
        os.path.join(os.path.dirname(path), "tagged.tif"),
        np.array([[0, 1000]], dtype=np.uint16),
        scale=0.01,
        offset=-1.0,
    )
    da = stac.open_asset(_landsat_item(tagged), "red").isel(time=0)
    np.testing.assert_allclose(da.values[0, 1], 9.0, rtol=1e-6)
    assert "scale_factor" not in da.attrs and "add_offset" not in da.attrs
    raw = stac.open_asset(_landsat_item(tagged), "red", rescale=False).isel(time=0)
    assert raw.values[0, 1] == 1000 and raw.attrs["scale_factor"] == 0.01

    # 32-bit integers are promoted to float64 so masking keeps them exact
    wide = _write_tif(
        os.path.join(os.path.dirname(path), "wide.tif"),
        np.array([[0, 16777217]], dtype=np.int32),
    )
    da = stac.open_asset(_landsat_item(wide), "red").isel(time=0)
    assert da.dtype == np.float64 and da.values[0, 1] == 16777217


def test_open_asset_multiband_uses_per_band_metadata(tmp_path) -> None:
    rgb = np.stack([np.full((2, 2), v, dtype=np.uint16) for v in (10, 20, 30)])
    rgb[2, 0, 0] = 7
    bands = [
        {"nodata": 10, "scale": 1.0},
        {"nodata": 0, "scale": 0.5},
        {"nodata": 7, "scale": 2.0, "offset": 1.0},
    ]
    item = _FakeItem(
        "tci",
        {
            "visual": _FakeAsset(
                _write_tif(tmp_path / "visual.tif", rgb), {"raster:bands": bands}
            )
        },
        when=datetime(2025, 6, 2, tzinfo=timezone.utc),
    )
    da = stac.open_asset(item, "visual")
    assert da.dims == ("time", "band", "y", "x") and da.chunks is not None
    values = da.isel(time=0).values
    assert np.isnan(values[0]).all()  # band 1 is entirely its own nodata
    np.testing.assert_array_equal(values[1], 10.0)
    assert np.isnan(values[2, 0, 0]) and values[2, 0, 1] == 61.0
    assert np.isnan(da.rio.nodata)
    # Unmasked, the bands keep different nodata values: none is declared for
    # the stack rather than band 1's for all
    da = stac.open_asset(item, "visual", mask_nodata=False, rescale=False)
    assert da.rio.nodata is None and da.dtype == np.uint16


def test_open_asset_reads_over_http(cog) -> None:
    """The /vsicurl?empty_dir=yes&url=... path form is accepted by GDAL."""
    path, dn = cog
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(path)
    )
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        href = f"http://127.0.0.1:{server.server_address[1]}/red.tif"
        item = _landsat_item(href)
        try:
            da = stac.open_asset(item, "red", rescale=False, mask_nodata=False)
            assert da.attrs["href"] == href
            np.testing.assert_array_equal(da.isel(time=0).values, dn)
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_stacks_time_and_variable_lazily(tmp_path, fake_client) -> None:
    utc = timezone.utc
    a = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
    later = _scene(tmp_path, "b", datetime(2024, 12, 9, tzinfo=utc), a * 2, a * 3)
    earlier = _scene(tmp_path, "a", datetime(2024, 11, 7, tzinfo=utc), a, a + 1)
    fake_client._items["landsat-c2-l2"] = [later, earlier]

    da = stac.search("landsat-c2-l2", ["red", "nir08"], rescale=False)

    assert da.dims == ("time", "variable", "y", "x")
    assert da.shape == (2, 2, 3, 4)
    assert da.chunks is not None
    assert list(da.coords["variable"].values) == ["red", "nir08"]
    # Sorted by time regardless of item order; item ids follow the same order
    assert da.time.values[0] == np.datetime64("2024-11-07")
    assert da.attrs["stac_items"] == ["scene-a", "scene-b"]
    assert da.attrs["stac_collection"] == "landsat-c2-l2"
    # Per-asset attrs of the first layer do not describe the stack
    assert not {"stac_item", "stac_asset", "href"} & set(da.attrs)
    assert da.rio.crs.to_epsg() == 32610
    assert da.rio.transform().a == 30
    np.testing.assert_array_equal(da.sel(variable="red").isel(time=0).values, a)
    np.testing.assert_array_equal(da.sel(variable="nir08").isel(time=1).values, a * 3)


def test_search_default_assets_and_eager(tmp_path, fake_client) -> None:
    a = np.ones((2, 2), dtype=np.uint16)
    item = _scene(tmp_path, "s", datetime(2024, 12, 9, tzinfo=timezone.utc), a, a)
    item.assets["qa_json"] = _FakeAsset("meta.json", media_type="application/json")
    fake_client._items["landsat-c2-l2"] = [item]

    da = stac.search("landsat-c2-l2", rescale=False, lazy=False)
    assert list(da.coords["variable"].values) == ["red", "nir08"]  # json skipped
    assert da.chunks is None  # computed
    assert da.attrs["stac_items"] == ["scene-s"]


def test_search_clips_to_bbox(tmp_path, fake_client) -> None:
    utc = timezone.utc
    a = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    item = _scene(tmp_path, "s", datetime(2024, 12, 9, tzinfo=utc), a, a)
    # Same scene a day later on a grid shifted by 25 m: the pixels holding the
    # window's right/bottom pixel centres lie wholly outside the bbox
    shifted = _scene(
        tmp_path,
        "t",
        datetime(2024, 12, 10, tzinfo=utc),
        a,
        a,
        origin=(499975, 4200025),
    )
    # And one that does not intersect the window at all
    far = _scene(
        tmp_path,
        "far",
        datetime(2024, 12, 11, tzinfo=utc),
        a,
        a,
        origin=(600000, 4100000),
    )
    fake_client._items["landsat-c2-l2"] = [item, shifted, far]

    # A lon/lat box that cuts through the 3x3 pixel block at the scene's
    # top-left: pixels are kept when their extent touches the box, so the
    # window's last column/row have centres outside it
    from rasterio.warp import transform_bounds

    bbox = list(
        transform_bounds("EPSG:32610", "EPSG:4326", 500005, 4199938, 500062, 4199995)
    )
    da = stac.search(
        "landsat-c2-l2",
        ["red"],
        bbox=bbox,
        time_range="2024-12-01/2024-12-31",
        rescale=False,
    )
    assert fake_client.search_calls[0]["bbox"] == bbox
    assert fake_client.search_calls[0]["datetime"] == "2024-12-01/2024-12-31"
    # Non-intersecting item skipped; the window is the 3x3 block
    assert da.attrs["stac_items"] == ["scene-s", "scene-t"]
    assert da.shape == (2, 1, 3, 3)
    np.testing.assert_array_equal(da.isel(time=0, variable=0).values, a[:3, :3])
    # The shifted scene is resampled onto the window grid without losing the
    # pixels that straddle the window edge
    assert not np.isnan(da.isel(time=1).values).any()
    assert float(da.isel(time=1).max()) <= a[3, 3]

    # A window narrower than one pixel is a user error, not a rioxarray trace
    tiny = list(
        transform_bounds("EPSG:32610", "EPSG:4326", 500001, 4199999, 500002, 4199998)
    )
    with pytest.raises(ValueError, match="fewer than two pixels"):
        stac.search("landsat-c2-l2", ["red"], bbox=tiny, rescale=False)


def test_search_aligns_grids(tmp_path, fake_client) -> None:
    utc = timezone.utc
    fine = np.full((4, 4), 7, dtype=np.uint16)
    coarse = np.full((2, 2), 9, dtype=np.uint16)
    # Assets at different resolutions (Sentinel-2 10 m vs 20 m) within an item
    ref = _FakeItem(
        "mixed",
        {
            "red": _FakeAsset(_write_tif(tmp_path / "red.tif", fine, res=30)),
            "swir16": _FakeAsset(_write_tif(tmp_path / "swir.tif", coarse, res=60)),
        },
        when=datetime(2024, 11, 7, tzinfo=utc),
    )
    # Same CRS, origin shifted by 20 m (not a half pixel, so no reference
    # pixel centre sits on a source pixel edge), files without a nodata value:
    # resampled onto the first grid
    shifted = _scene(
        tmp_path,
        "shift",
        datetime(2024, 12, 9, tzinfo=utc),
        fine * 2,
        fine * 2,
        origin=(500020, 4200020),
        nodata=None,
    )
    shifted.assets["swir16"] = shifted.assets.pop("nir08")
    fake_client._items["landsat-c2-l2"] = [ref, shifted]

    da = stac.search(
        "landsat-c2-l2", ["red", "swir16"], rescale=False, mask_nodata=False
    )
    # On the first asset's 4x4 grid, no NaN holes from coordinate outer-join
    assert da.shape == (2, 2, 4, 4)
    assert not np.isnan(da.isel(time=0).values).any()
    np.testing.assert_array_equal(da.x.values, stac.open_asset(ref, "red").x.values)
    assert float(da.isel(time=0).sel(variable="swir16").min()) == 9
    # Shifted scene lands on the reference grid; where it does not cover the
    # grid the fill is NaN, not the uint16 maximum, and no misleading nodata
    # is declared for the mixed stack
    assert float(da.isel(time=1).max()) == 14
    assert np.isnan(da.isel(time=1, variable=0).values[0, 0])
    assert da.rio.nodata is None


def test_search_multiband_asset_and_missing_datetime(tmp_path, fake_client) -> None:
    rgb = np.stack([np.full((2, 2), v, dtype=np.uint8) for v in (10, 20, 30)])
    path = _write_tif(tmp_path / "visual.tif", rgb)
    # datetime: null with a start/end range, as the STAC spec requires
    ranged = _FakeItem(
        "tci-range",
        {"visual": _FakeAsset(path)},
        properties={"start_datetime": "2025-06-02T00:00:00Z"},
    )
    # Naive datetimes are UTC, never local time
    naive = _FakeItem(
        "tci-naive", {"visual": _FakeAsset(path)}, when=datetime(2025, 6, 1, 18)
    )
    undated = _FakeItem("tci-none", {"visual": _FakeAsset(path)})
    fake_client._items["landsat-c2-l2"] = [undated, ranged, naive]
    da = stac.search("landsat-c2-l2", rescale=False, mask_nodata=False)
    assert da.dims == ("time", "variable", "y", "x")
    assert list(da.coords["variable"].values) == ["visual_1", "visual_2", "visual_3"]
    assert float(da.sel(variable="visual_3").max()) == 30
    # Sorted by time with the undated item (NaT) last
    assert da.attrs["stac_items"] == ["tci-naive", "tci-range", "tci-none"]
    assert da.time.values[0] == np.datetime64("2025-06-01T18:00:00")
    assert da.time.values[1] == np.datetime64("2025-06-02T00:00:00")
    assert np.isnat(da.time.values[2])


def test_search_no_items(fake_client) -> None:
    fake_client._items["landsat-c2-l2"] = []
    with pytest.raises(FileNotFoundError, match="No items found"):
        stac.search("landsat-c2-l2", ["red"], bbox=[0, 0, 1, 1])


# ---------------------------------------------------------------------------
# collections
# ---------------------------------------------------------------------------


def test_collections(fake_client) -> None:
    by_id = {c["id"]: c for c in stac.collections()}
    assert set(by_id) == {"landsat-c2-l2", "naip", "cop-dem-glo-30"}
    assert by_id["landsat-c2-l2"]["assets"] == ["nir08", "red"]
    assert by_id["landsat-c2-l2"]["time_range"][1] is None
    # First bbox is the overall extent; 3D bbox reduced to lon/lat
    assert by_id["naip"]["bbox"] == (-125, 18, -66, 50)
    assert by_id["cop-dem-glo-30"]["bbox"] == (-180, -90, 180, 90)
    # Title falls back to id; no items -> no asset keys
    assert by_id["cop-dem-glo-30"]["title"] == "cop-dem-glo-30"
    assert by_id["cop-dem-glo-30"]["assets"] == []
    assert by_id["cop-dem-glo-30"]["time_range"] == (None, None)

    fake_client.search_calls.clear()
    assert [c["id"] for c in stac.collections(text="LANDSAT", sample_assets=False)] == [
        "landsat-c2-l2"
    ]
    ids = {c["id"] for c in stac.collections(bbox=(0, 40, 20, 55), sample_assets=False)}
    assert "naip" not in ids and "landsat-c2-l2" in ids
    assert fake_client.search_calls == []


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
    assert len(da.attrs["stac_items"]) >= 1 and da.chunks is not None
    assert da.dims == ("time", "variable", "y", "x")
    assert da.rio.crs.to_epsg() == 32610
    red = da.sel(variable="red").isel(time=0).compute()
    valid = red.values[np.isfinite(red.values)]
    assert valid.size > 0 and valid.min() >= -0.2 and valid.max() <= 1.6
    # Single-asset path on a real item, raw dtype preserved
    item = stac.search_items("landsat-c2-l2", ids=da.attrs["stac_items"][:1])[0]
    qa = stac.open_asset(item, "qa_pixel", rescale=False, mask_nodata=False)
    assert qa.dtype == np.uint16 and qa.chunks is not None


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(120)
def test_live_earth_search_sentinel2_window() -> None:
    da = stac.search(
        "sentinel-2-l2a",
        ["red"],
        provider=stac.EARTH_SEARCH,
        bbox=[-122.55, 37.70, -122.35, 37.85],
        query={"grid:code": {"eq": "MGRS-10SEG"}},
        time_range="2025-06-01/2025-06-04",
        sortby=[{"field": "properties.s2:nodata_pixel_percentage", "direction": "asc"}],
        max_items=1,
    )
    assert da.rio.crs.to_epsg() == 32610 and len(da.attrs["stac_items"]) == 1
    red = da.isel(time=0, variable=0).compute()
    valid = red.values[np.isfinite(red.values)]
    assert valid.size > 0 and valid.min() >= 0.0 and np.nanmedian(valid) < 0.5
