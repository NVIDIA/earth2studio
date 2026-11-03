# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
Lazy Access to STAC Catalogs
============================

Explore a STAC catalog and read only the satellite pixels you need.

Earth observation archives such as Landsat and Sentinel-2 are published as
cloud-optimized GeoTIFFs (COGs) indexed by SpatioTemporal Asset Catalogs (STAC).
A single scene is hundreds of megabytes, so downloading whole files to look at
a city-sized area is wasteful. [`earth2studio.data.stac`][earth2studio.data.stac]
is the shared layer that collection-specific data sources build on: it searches
a catalog and opens the matching assets as one lazy, dask-backed
``(time, variable, y, x)`` array in the scene's native grid, reading only the
COG tiles a computation touches.

In this example you will learn:

- How to browse a catalog's collections with
  [`stac.collections`][earth2studio.data.stac.collections]
- How to turn a search into a lazy time stack with
  [`stac.search`][earth2studio.data.stac.search] and compute NDVI from it
- How to open a single asset with
  [`stac.open_asset`][earth2studio.data.stac.open_asset] and use it as a mask
- How the same call works against a second provider (Earth Search)
"""

# /// script
# dependencies = [
#   "earth2studio[data] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# No model is involved; everything here is data access. Two public catalogs are
# used, neither needs credentials:
#
# - Microsoft Planetary Computer ([`stac.PLANETARY_COMPUTER`][earth2studio.data.stac.PLANETARY_COMPUTER]),
#   whose Azure blobs are read through GDAL's built-in SAS signing.
# - Element 84 Earth Search ([`stac.EARTH_SEARCH`][earth2studio.data.stac.EARTH_SEARCH]),
#   whose public S3 buckets are read anonymously over HTTPS.
#
# The area of interest is San Francisco, given as a lon/lat bounding box.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

from earth2studio.data import stac

bbox = [-122.55, 37.70, -122.35, 37.85]  # [min_lon, min_lat, max_lon, max_lat]

# %%
# Browse the Catalog
# ------------------
# Collection ids and asset keys are what every later call needs.
# [`stac.collections`][earth2studio.data.stac.collections] lists them with
# the overall spatial and temporal extent; ``sample_assets=True`` would fetch
# one item per collection to list its asset keys too.

# %%
for info in stac.collections(text="landsat", sample_assets=False):
    start, end = info["time_range"]
    end = f"{end:%Y}" if end else "present"
    print(f"{info['id']:<16} {info['title']:<60} {start:%Y} - {end}")

# %%
# Search and Stack
# ----------------
# [`stac.search`][earth2studio.data.stac.search] runs the STAC query, opens
# the requested assets of every matching item, clips them to ``bbox`` and
# stacks them into a ``(time, variable, y, x)`` array. Only the COG headers are
# read at this point. The ``raster:bands`` scale/offset is applied so the values
# are surface reflectance, and nodata is NaN.

# %%
da = stac.search(
    "landsat-c2-l2",
    ["red", "nir08"],
    bbox=bbox,
    time_range="2024-10-01/2024-12-31",
    query={"eo:cloud_cover": {"lt": 5}},
    max_items=3,
)
print(da)
print("CRS:", da.rio.crs, "| resolution (m):", da.rio.resolution())
print("items:", da.attrs["stac_items"])

# %%
# The array behaves like any xarray/rioxarray object. Computing NDVI pulls only
# the clipped tiles of the four COGs.

# %%
red = da.sel(variable="red")
nir = da.sel(variable="nir08")
ndvi = ((nir - red) / (nir + red)).compute()

# %%
# Mask Clouds
# -----------
# Landsat's ``qa_pixel`` band encodes cloud flags in bits. Searching the same
# item ids for that asset, without rescaling or masking, returns the raw
# ``uint16`` flags on the same grid and in the same time order as the NDVI
# stack, so the two line up directly.

# %%
import numpy as np

qa = stac.search(
    "landsat-c2-l2",
    ["qa_pixel"],
    bbox=bbox,
    ids=da.attrs["stac_items"],
    rescale=False,
    mask_nodata=False,
    lazy=False,
)
# Bit 3 = cloud, bit 4 = cloud shadow (Landsat Collection 2 QA_PIXEL)
cloud = (qa.sel(variable="qa_pixel").values.astype(np.uint16) & 0b11000) > 0
ndvi_clear = ndvi.where(~cloud)

# %%
# Open a Single Asset
# -------------------
# [`stac.open_asset`][earth2studio.data.stac.open_asset] is the building block
# underneath ``search``: one asset of one item, lazily, in its raw dtype and
# full extent. Window it yourself with rioxarray before computing.

# %%
item = stac.search_items("landsat-c2-l2", ids=da.attrs["stac_items"][:1])[0]
qa_scene = stac.open_asset(item, "qa_pixel", rescale=False, mask_nodata=False)
print(qa_scene)
print("window:", qa_scene.rio.clip_box(*bbox, crs="EPSG:4326").shape)

# %%
# A Second Provider
# -----------------
# The same call against Earth Search reads Sentinel-2 L2A. The ``visual``
# asset is an 8-bit RGB COG, so each band becomes a variable
# (``visual_1``..``visual_3``); here it is read as-is without rescaling.

# %%
rgb = stac.search(
    "sentinel-2-l2a",
    ["visual"],
    provider=stac.EARTH_SEARCH,
    bbox=bbox,
    query={"grid:code": {"eq": "MGRS-10SEG"}},
    time_range="2025-06-01/2025-06-10",
    sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    max_items=1,
    rescale=False,
    mask_nodata=False,
)
print(rgb)
rgb = rgb.isel(time=0).compute()

# %%
# Post Processing
# ---------------
# The arrays keep the scene's UTM grid, so plot them in that projection and let
# cartopy draw coastlines. The NDVI panels show one Landsat pass each; a pass
# that only clips the corner of the bbox is mostly NaN, which is expected from a
# per-scene search and is exactly what a collection-specific data source would
# resolve by mosaicking.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
utm = ccrs.epsg(da.rio.crs.to_epsg())
extent = [float(da.x.min()), float(da.x.max()), float(da.y.min()), float(da.y.max())]

fig, axes = plt.subplots(
    1,
    ndvi.sizes["time"] + 1,
    figsize=(5 * (ndvi.sizes["time"] + 1), 5),
    subplot_kw={"projection": utm},
)
for ax, t in zip(axes[:-1], ndvi_clear.time.values):
    im = ax.imshow(
        ndvi_clear.sel(time=t).values,
        extent=extent,
        transform=utm,
        origin="upper",
        vmin=-0.2,
        vmax=0.8,
        cmap="RdYlGn",
    )
    ax.coastlines(resolution="10m")
    ax.set_title(f"Landsat NDVI {str(t)[:10]}")
fig.colorbar(im, ax=axes[:-1].tolist(), shrink=0.7, label="NDVI (clouds masked)")

ax = axes[-1]
ax.imshow(
    np.clip(rgb.values.transpose(1, 2, 0) / 255.0, 0, 1),
    extent=[
        float(rgb.x.min()),
        float(rgb.x.max()),
        float(rgb.y.min()),
        float(rgb.y.max()),
    ],
    transform=utm,
    origin="upper",
)
ax.coastlines(resolution="10m")
ax.set_title(f"Sentinel-2 visual {str(rgb.time.values)[:10]}")
plt.savefig("outputs/06_stac_landsat_ndvi_sentinel2_rgb.jpg", bbox_inches="tight")
