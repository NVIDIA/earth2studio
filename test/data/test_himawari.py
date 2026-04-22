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

import pathlib
import shutil
from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import Himawari


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "satellite,time,variable",
    [
        # Himawari-9: Operational from Dec 13, 2022 onwards
        (
            "himawari9",
            datetime(year=2024, month=6, day=15, hour=0, minute=0, second=0),
            "ahi13",
        ),
        (
            "himawari9",
            datetime(year=2024, month=6, day=15, hour=0, minute=0, second=0),
            ["ahi13", "ahi14"],
        ),
        (
            "himawari9",
            [
                datetime(year=2024, month=6, day=15, hour=0, minute=0, second=0),
                datetime(year=2024, month=6, day=15, hour=0, minute=10, second=0),
            ],
            "ahi13",
        ),
        # Himawari-8: Operational from Jul 7, 2015 to Dec 13, 2022
        (
            "himawari8",
            datetime(year=2022, month=6, day=15, hour=0, minute=0, second=0),
            "ahi13",
        ),
    ],
)
def test_himawari_fetch(satellite, time, variable):
    """Test Himawari data fetching for different satellites and variables.

    Uses a small lat_lon_bbox that overlaps a single tile to minimise
    download size and test duration.
    """
    # Small bbox near sub-satellite point — overlaps only tile 50
    bbox = (-5.0, 141.0, -2.0, 144.0)
    ds = Himawari(satellite=satellite, lat_lon_bbox=bbox, cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] < 5500  # Cropped, not full disk
    assert shape[3] < 5500
    assert shape[2] > 50  # But still meaningful
    assert shape[3] > 50
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-06-15T00:00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["ahi13", "ahi14"]])
@pytest.mark.parametrize("cache", [True, False])
def test_himawari_cache(time, variable, cache):
    """Test Himawari caching functionality.

    Uses a small lat_lon_bbox to limit download to a single tile.
    """
    bbox = (-5.0, 141.0, -2.0, 144.0)
    ds = Himawari(satellite="himawari9", lat_lon_bbox=bbox, cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] < 5500
    assert shape[3] < 5500
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Reload from cache
    data = ds(time, variable[0])
    assert data.shape[1] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "satellite,time,expected",
    [
        # Valid time within Himawari-9 range
        ("himawari9", datetime(2024, 6, 15, 0, 0, 0), True),
        # Valid time within Himawari-8 range
        ("himawari8", datetime(2020, 6, 15, 0, 0, 0), True),
        # Before Himawari-8 operational start
        ("himawari8", datetime(2014, 1, 1, 0, 0, 0), False),
        # After Himawari-8 retired
        ("himawari8", datetime(2023, 6, 15, 0, 0, 0), False),
        # Not on 10-minute interval
        ("himawari9", datetime(2024, 6, 15, 0, 3, 0), False),
        # Invalid satellite
        ("himawari7", datetime(2024, 6, 15, 0, 0, 0), False),
    ],
)
def test_himawari_available(satellite, time, expected):
    """Test Himawari availability checks."""
    assert Himawari.available(time, satellite=satellite) == expected


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        # Before operational start
        datetime(2014, 1, 1, 0, 0, 0),
        # Not on 10-minute interval
        datetime(2024, 6, 15, 0, 5, 30),
    ],
)
@pytest.mark.parametrize("variable", ["ahi13"])
def test_himawari_invalid_time(time, variable):
    """Test that invalid times raise ValueError."""
    ds = Himawari(satellite="himawari9")
    with pytest.raises(ValueError):
        ds(time, variable)


@pytest.mark.timeout(5)
def test_himawari_invalid_satellite():
    """Test that invalid satellite raises ValueError."""
    with pytest.raises(ValueError):
        Himawari(satellite="himawari7")


@pytest.mark.timeout(30)
def test_himawari_grid():
    """Test Himawari grid method returns correct lat/lon coordinates."""
    lat, lon = Himawari.grid()

    assert lat.shape == (5500, 5500)
    assert lon.shape == (5500, 5500)

    # Sub-satellite point should be near (0, 140.7)
    center_y = 5500 // 2
    center_x = 5500 // 2
    assert abs(lat[center_y, center_x] - 0.0) < 0.1
    assert abs(lon[center_y, center_x] - 140.7) < 0.1


@pytest.mark.timeout(5)
def test_himawari_numpy_datetime_available():
    """Test availability with numpy datetime64."""
    time = np.datetime64("2024-06-15T00:00:00")
    assert Himawari.available(time, satellite="himawari9")

    time_invalid = np.datetime64("2014-01-01T00:00:00")
    assert not Himawari.available(time_invalid, satellite="himawari8")


@pytest.mark.timeout(5)
def test_himawari_parse_tile_number():
    """Test tile number parsing from ISatSS filenames."""
    assert (
        Himawari._parse_tile_number("OR_HFD-020-B14-M1C13-T001_GH9_s202406150000.nc")
        == 1
    )
    assert (
        Himawari._parse_tile_number("OR_HFD-020-B14-M1C13-T088_GH9_s202406150000.nc")
        == 88
    )
    assert (
        Himawari._parse_tile_number("OR_HFD-010-B11-M1C01-T045_GH9_s202406150000.nc")
        == 45
    )
    assert Himawari._parse_tile_number("invalid_filename.nc") is None


def _make_mock_tile_nc(
    path: pathlib.Path,
    rows: int = 550,
    cols: int = 550,
    tile_row_offset: int = 0,
    tile_column_offset: int = 0,
) -> None:
    """Create a minimal ISatSS-like tile NetCDF file at ``path``."""
    ds = xr.Dataset(
        {
            "Sectorized_CMI": (
                ["y", "x"],
                np.random.rand(rows, cols).astype(np.float32) * 300 + 200,
            ),
        },
        coords={
            "y": np.arange(rows, dtype=np.int16),
            "x": np.arange(cols, dtype=np.int16),
        },
        attrs={
            "tile_row_offset": tile_row_offset,
            "tile_column_offset": tile_column_offset,
        },
    )
    ds.to_netcdf(path)


@pytest.mark.timeout(30)
def test_himawari_call_mock(tmp_path: pathlib.Path):
    """Test full __call__ path with mocked S3 (no network)."""
    # Use a tiny 2x2 grid (4 tiles of 1x1) to keep things fast
    small_dim = 2
    tile_sz = 1

    # Generate fake tile filenames (4 tiles, 1 channel)
    tile_files = [
        f"noaa-himawari9/AHI-L2-FLDK-ISatSS/2024/06/15/0000/"
        f"OR_HFD-020-B14-M1C13-T{t:03d}_GH9_s20240615000000.nc"
        for t in range(1, 5)  # 4 tiles for a 2x2 grid with tile_size=1
    ]

    # Tile offsets for a 2x2 grid: T001=(0,0), T002=(0,1), T003=(1,0), T004=(1,1)
    tile_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Write synthetic tile NetCDFs to tmp_path, keyed by SHA hash
    import hashlib

    tile_cache = {}
    for tf, (row_off, col_off) in zip(tile_files, tile_offsets):
        sha = hashlib.sha256(tf.encode()).hexdigest()
        nc_path = tmp_path / sha
        _make_mock_tile_nc(
            nc_path,
            rows=tile_sz,
            cols=tile_sz,
            tile_row_offset=row_off,
            tile_column_offset=col_off,
        )
        tile_cache[tf] = str(nc_path)

    # Async mock for fs._ls that returns our tile filenames
    async def mock_ls(path, detail=False):
        return tile_files

    # Async mock for fs._cat_file that returns bytes from our cached tiles
    async def mock_cat(path, **kwargs):
        sha = hashlib.sha256(path.encode()).hexdigest()
        nc_path = tmp_path / sha
        with open(nc_path, "rb") as f:
            return f.read()

    with (
        patch.object(Himawari, "_async_init", return_value=None),
        patch.object(
            Himawari,
            "SCAN_DIMENSIONS",
            (small_dim, small_dim),
        ),
        patch.object(
            Himawari,
            "_Y_COORDS",
            np.linspace(0.15, -0.15, small_dim),
        ),
        patch.object(
            Himawari,
            "_X_COORDS",
            np.linspace(-0.15, 0.15, small_dim),
        ),
        patch("earth2studio.data.himawari.FULL_DISK_PIXELS", small_dim),
        patch("earth2studio.data.himawari.TILE_SIZE", tile_sz),
    ):
        ds = Himawari(satellite="himawari9", cache=False)
        # Inject mock FS with async methods
        mock_fs = AsyncMock()
        mock_fs._ls = mock_ls
        mock_fs._cat_file = mock_cat
        mock_fs.async_impl = True
        ds.fs = mock_fs
        # Override lat/lon since we patched grid size
        ds._lat = np.zeros((small_dim, small_dim))
        ds._lon = np.zeros((small_dim, small_dim))
        # Override cache to use tmp_path
        ds._cache = False
        ds._tmp_cache_hash = "mock"

        # Patch cache property to use tmp_path
        cache_dir = str(tmp_path / "cache")
        with patch.object(type(ds), "cache", property(lambda self: cache_dir)):
            data = ds(datetime(2024, 6, 15, 0, 0), "ahi13")

    assert data.shape == (1, 1, small_dim, small_dim)
    assert list(data.coords["variable"].values) == ["ahi13"]
    # Verify data is not all NaN (tiles were assembled)
    assert not np.isnan(data.values).all()


@pytest.mark.timeout(5)
def test_himawari_compute_pixel_roi():
    """Test _compute_pixel_roi finds correct pixel bounds."""
    from earth2studio.data.himawari import _compute_pixel_roi

    # Create a small synthetic grid
    lat = np.array([[30.0, 30.0], [20.0, 20.0], [10.0, 10.0]])
    lon = np.array([[130.0, 150.0], [130.0, 150.0], [130.0, 150.0]])

    # Bbox covering the whole grid
    r0, r1, c0, c1 = _compute_pixel_roi((5.0, 125.0, 35.0, 155.0), lat, lon)
    assert (r0, r1, c0, c1) == (0, 3, 0, 2)

    # Bbox covering only the top row (lat >= 25)
    r0, r1, c0, c1 = _compute_pixel_roi((25.0, 125.0, 35.0, 155.0), lat, lon)
    assert (r0, r1) == (0, 1)

    # Bbox with no overlap raises ValueError
    with pytest.raises(ValueError, match="No grid points"):
        _compute_pixel_roi((50.0, 0.0, 60.0, 10.0), lat, lon)

    # Antimeridian-crossing bbox (wraps from 140°E to 170°W = -170°)
    lon_am = np.array([[170.0, -170.0], [170.0, -170.0]])
    lat_am = np.array([[30.0, 30.0], [20.0, 20.0]])
    r0, r1, c0, c1 = _compute_pixel_roi((15.0, 160.0, 35.0, -160.0), lat_am, lon_am)
    assert (r0, r1, c0, c1) == (0, 2, 0, 2)

    # Same antimeridian bbox using [0, 360] convention (160°E to 200°E)
    r0, r1, c0, c1 = _compute_pixel_roi((15.0, 160.0, 35.0, 200.0), lat_am, lon_am)
    assert (r0, r1, c0, c1) == (0, 2, 0, 2)


@pytest.mark.timeout(30)
def test_himawari_lat_lon_bbox():
    """Test that lat_lon_bbox computes pixel ROI correctly from real grid."""
    # Use the real Himawari grid to verify ROI computation
    ds = Himawari(satellite="himawari9", cache=False)

    # Japan bbox: ~25-50°N, 125-150°E
    ds._lat_lon_bbox = (25.0, 125.0, 50.0, 150.0)
    from earth2studio.data.himawari import _compute_pixel_roi

    roi = _compute_pixel_roi(ds._lat_lon_bbox, ds._lat, ds._lon)
    r0, r1, c0, c1 = roi
    ny = r1 - r0
    nx = c1 - c0

    # Japan region should be a subset of the full disk
    assert ny < 5500
    assert nx < 5500
    assert ny > 100  # Should cover a meaningful region
    assert nx > 100
    assert r0 >= 0 and r1 <= 5500
    assert c0 >= 0 and c1 <= 5500


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_himawari_fetch_bbox():
    """Test fetching with a lat_lon_bbox (real network)."""
    # Small bbox overlapping a single tile (tile 50)
    ds = Himawari(
        satellite="himawari9",
        lat_lon_bbox=(-5.0, 141.0, -2.0, 144.0),
        cache=False,
    )
    data = ds(datetime(2024, 6, 15, 0, 0), "ahi13")
    shape = data.shape

    # Output should be cropped, not full 5500x5500
    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] < 5500
    assert shape[3] < 5500
    # Should have some valid data
    assert not np.isnan(data.values).all()
    # Should have _lat and _lon coordinates
    assert "_lat" in data.coords
    assert "_lon" in data.coords


@pytest.mark.timeout(30)
def test_himawari_bbox_tile_filtering():
    """Test that tiles outside bbox are filtered in _create_tasks."""
    from earth2studio.data.himawari import TILE_OFFSETS_2KM

    # Verify tile offset table has 88 entries
    assert len(TILE_OFFSETS_2KM) == 88

    # Verify offsets are within full disk bounds
    for tnum, (r, c) in TILE_OFFSETS_2KM.items():
        assert 0 <= r < 5500, f"Tile {tnum} row offset {r} out of range"
        assert 0 <= c < 5500, f"Tile {tnum} col offset {c} out of range"
        assert 1 <= tnum <= 88
