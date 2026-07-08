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
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest

from earth2studio.data import MeteosatFCI

# ---------------------------------------------------------------------------
# Slow integration tests (require EUMETSAT credentials and network access)
#
# All slow tests crop to a small Europe bounding box so only the relevant
# BODY segments are read from the downloaded product, keeping I/O manageable.
# ---------------------------------------------------------------------------

# lat_lon_bbox format: ((lat_min, lat_max), (lon_min, lon_max))
_EUROPE_BBOX = ((35.0, 60.0), (-10.0, 30.0))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time,variable",
    [
        (datetime(2025, 3, 1, 12, 0), "fci_ir_87"),
        (
            [datetime(2025, 3, 1, 12, 0), datetime(2025, 3, 1, 12, 10)],
            ["fci_ir_87", "fci_wv_63"],
        ),
    ],
)
def test_meteosat_fci_fetch(time, variable):
    ds = MeteosatFCI(resolution="2km", lat_lon_bbox=_EUROPE_BBOX, cache=False)
    data = ds(time, variable)
    if isinstance(time, datetime):
        time = [time]
    if isinstance(variable, str):
        variable = [variable]
    assert data.shape[:2] == (len(time), len(variable))
    assert 0 < data.shape[2] < 5568
    assert 0 < data.shape[3] < 5568
    assert not np.all(np.isnan(data.values))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_meteosat_fci_cache():
    ds = MeteosatFCI(resolution="2km", lat_lon_bbox=_EUROPE_BBOX, cache=True)
    time = datetime(2025, 3, 1, 12, 0)
    data1 = ds(time, "fci_ir_87")
    data2 = ds(time, "fci_ir_87")  # served from cache — no re-download
    assert data1.shape == data2.shape
    assert pathlib.Path(ds.cache).is_dir()
    shutil.rmtree(ds.cache, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_meteosat_fci_hrfi_fetch():
    # HRFI vis_06 at 500 m.  pixel_bbox restricts which BODY segments are read.
    # pixel_bbox format: ((row_start, row_end), (col_start, col_end))
    row_bounds = (10000, 11000)
    col_bounds = (10000, 11000)
    ds = MeteosatFCI(
        resolution="500m", pixel_bbox=(row_bounds, col_bounds), cache=False
    )
    data = ds(datetime(2025, 3, 1, 12, 0), "fci_vis_06")
    ny = row_bounds[1] - row_bounds[0]
    nx = col_bounds[1] - col_bounds[0]
    assert data.shape == (1, 1, ny, nx)
    assert not np.all(np.isnan(data.values))


# ---------------------------------------------------------------------------
# Helper: build synthetic BODY NetCDF4 segment files
# ---------------------------------------------------------------------------


def _build_body_nc(
    path: pathlib.Path,
    channel: str,
    rows: int,
    cols: int,
    hdr: bool = False,
) -> None:
    """Write a minimal FCI BODY segment NetCDF4 file."""
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    grp = ds.createGroup("data").createGroup(channel).createGroup("measured")
    grp.createDimension("y", rows)
    grp.createDimension("x", cols)

    if hdr:
        var = grp.createVariable("effective_radiance", "u2", ("y", "x"))
        raw = np.zeros((rows, cols), dtype=np.uint16)
        raw[0, 0] = 2000  # normal pixel  → 2000 × 0.01 + 0.0   = 20.0
        raw[0, 1] = 5000  # HDR pixel     → 5000 × 0.001 + 100.0 = 105.0
        var[:] = raw
        var.setncattr("scale_factor", np.float32(0.01))
        var.setncattr("add_offset", np.float32(0.0))
        var.setncattr("warm_scale_factor", np.float32(0.001))
        var.setncattr("warm_add_offset", np.float32(100.0))
    else:
        var = grp.createVariable("effective_radiance", "f4", ("y", "x"))
        var[:] = np.random.default_rng(42).random((rows, cols), dtype=np.float32)

    ds.close()


# ---------------------------------------------------------------------------
# Mock-based unit tests (no network required)
# ---------------------------------------------------------------------------

_ROWS, _COLS = 32, 32

# Small _GRID_SIZE dict to replace the real one during mocking.
_SMALL_GRID_SIZE = {
    "2km": (_ROWS, _COLS),
    "1km": (_ROWS * 2, _COLS * 2),
    "500m": (_ROWS * 4, _COLS * 4),
}

# Small FCI_X / FCI_Y class-attribute replacements.
# FCI_X: decreasing (west = positive), FCI_Y: increasing (south = negative).
_SMALL_FCI_X = {
    res: np.linspace(0.05, -0.05, _SMALL_GRID_SIZE[res][1]) for res in _SMALL_GRID_SIZE
}
_SMALL_FCI_Y = {
    res: np.linspace(-0.05, 0.05, _SMALL_GRID_SIZE[res][0]) for res in _SMALL_GRID_SIZE
}


def _mock_lat_lon(rows: int = _ROWS, cols: int = _COLS):
    lat = np.linspace(-10, 10, rows * cols).reshape(rows, cols).astype(np.float32)
    lon = np.linspace(-10, 10, rows * cols).reshape(rows, cols).astype(np.float32)
    return lat, lon


def test_meteosat_fci_call_mock(tmp_path):
    product_dir = tmp_path / "mtg_mock"
    product_dir.mkdir()
    _build_body_nc(product_dir / "BODY_001.nc", "ir_87", _ROWS, _COLS)

    with (
        patch.object(MeteosatFCI, "_fetch_product", return_value=str(product_dir)),
        patch.object(MeteosatFCI, "_ensure_grid", return_value=_mock_lat_lon()),
        patch("earth2studio.data.meteosat_fci._GRID_SIZE", _SMALL_GRID_SIZE),
        patch.object(MeteosatFCI, "FCI_X", _SMALL_FCI_X),
        patch.object(MeteosatFCI, "FCI_Y", _SMALL_FCI_Y),
    ):
        ds = MeteosatFCI(resolution="2km", cache=False, verbose=False)
        data = ds(datetime(2025, 3, 1, 12, 0), "fci_ir_87")

    assert data.shape == (1, 1, _ROWS, _COLS)
    assert not np.all(np.isnan(data.values))
    assert "_lat" in data.coords and "_lon" in data.coords
    # y-coords should be FCI_Y["2km"] (south-to-north, increasing)
    assert data.coords["y"].values[0] < data.coords["y"].values[-1]
    # x-coords should be FCI_X["2km"][0:_COLS] (first element positive = west)
    assert data.coords["x"].values[0] > data.coords["x"].values[-1]


def test_meteosat_fci_call_mock_flipped(tmp_path):
    """flip_north_south=True and False produce exactly flipped data and coords."""
    product_dir = tmp_path / "mtg_mock_flip"
    product_dir.mkdir()
    _build_body_nc(product_dir / "BODY_001.nc", "ir_87", _ROWS, _COLS)

    patches = (
        patch.object(MeteosatFCI, "_fetch_product", return_value=str(product_dir)),
        patch.object(MeteosatFCI, "_ensure_grid", return_value=_mock_lat_lon()),
        patch("earth2studio.data.meteosat_fci._GRID_SIZE", _SMALL_GRID_SIZE),
        patch.object(MeteosatFCI, "FCI_X", _SMALL_FCI_X),
        patch.object(MeteosatFCI, "FCI_Y", _SMALL_FCI_Y),
    )

    t = datetime(2025, 3, 1, 12, 0)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        data_native = MeteosatFCI(
            resolution="2km", flip_north_south=False, cache=False, verbose=False
        )(t, "fci_ir_87")
        data_flipped = MeteosatFCI(
            resolution="2km", flip_north_south=True, cache=False, verbose=False
        )(t, "fci_ir_87")

    # --- shape ---
    assert data_native.shape == (1, 1, _ROWS, _COLS)
    assert data_flipped.shape == (1, 1, _ROWS, _COLS)

    # --- y-coordinate direction ---
    # native: south-to-north (increasing y)
    assert data_native.coords["y"].values[0] < data_native.coords["y"].values[-1]
    # flipped: north-to-south (decreasing y)
    assert data_flipped.coords["y"].values[0] > data_flipped.coords["y"].values[-1]

    # --- y-coordinates are reversed copies of each other ---
    np.testing.assert_array_equal(
        data_native.coords["y"].values,
        data_flipped.coords["y"].values[::-1],
    )

    # --- data values are row-flipped versions of each other ---
    np.testing.assert_array_equal(
        data_native.values,
        data_flipped.values[:, :, ::-1, :],
    )

    # --- lat/lon are also row-flipped ---
    np.testing.assert_array_equal(
        data_native.coords["_lat"].values,
        data_flipped.coords["_lat"].values[::-1, :],
    )
    np.testing.assert_array_equal(
        data_native.coords["_lon"].values,
        data_flipped.coords["_lon"].values[::-1, :],
    )


def test_meteosat_fci_pixel_bbox_mock(tmp_path):
    # pixel_bbox format: ((row_start, row_end), (col_start, col_end))
    row_bounds = (8, 24)  # 16-row crop of a 32-row grid
    col_bounds = (8, 24)  # 16-col crop of a 32-col grid
    product_dir = tmp_path / "mtg_mock"
    product_dir.mkdir()
    _build_body_nc(product_dir / "BODY_001.nc", "ir_87", _ROWS, _COLS)

    with (
        patch.object(MeteosatFCI, "_fetch_product", return_value=str(product_dir)),
        patch.object(MeteosatFCI, "_ensure_grid", return_value=_mock_lat_lon()),
        patch("earth2studio.data.meteosat_fci._GRID_SIZE", _SMALL_GRID_SIZE),
        patch.object(MeteosatFCI, "FCI_X", _SMALL_FCI_X),
        patch.object(MeteosatFCI, "FCI_Y", _SMALL_FCI_Y),
    ):
        ds = MeteosatFCI(
            resolution="2km",
            pixel_bbox=(row_bounds, col_bounds),
            cache=False,
            verbose=False,
        )
        data = ds(datetime(2025, 3, 1, 12, 0), "fci_ir_87")

    ny = row_bounds[1] - row_bounds[0]
    nx = col_bounds[1] - col_bounds[0]
    assert data.shape == (1, 1, ny, nx)
    assert "_lat" in data.coords and "_lon" in data.coords


def test_meteosat_fci_ir38_hdr_mock(tmp_path):
    # Pixels with raw > 4095 must use the warm coefficients, not standard ones.
    product_dir = tmp_path / "mtg_mock_hdr"
    product_dir.mkdir()
    _build_body_nc(product_dir / "BODY_001.nc", "ir_38", _ROWS, _COLS, hdr=True)

    with (
        patch.object(MeteosatFCI, "_fetch_product", return_value=str(product_dir)),
        patch.object(MeteosatFCI, "_ensure_grid", return_value=_mock_lat_lon()),
        patch("earth2studio.data.meteosat_fci._GRID_SIZE", _SMALL_GRID_SIZE),
        patch.object(MeteosatFCI, "FCI_X", _SMALL_FCI_X),
        patch.object(MeteosatFCI, "FCI_Y", _SMALL_FCI_Y),
    ):
        ds = MeteosatFCI(resolution="2km", cache=False, verbose=False)
        data = ds(datetime(2025, 3, 1, 12, 0), "fci_ir_38")

    values = data.values
    assert np.any(np.isclose(values, 20.0, atol=0.1))  # normal: 2000×0.01
    assert np.any(np.isclose(values, 105.0, atol=0.1))  # HDR:    5000×0.001+100
    assert not np.any(np.isclose(values, 50.0, atol=0.1))  # wrong if HDR skipped


# ---------------------------------------------------------------------------
# Grid and availability tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "resolution,expected_shape",
    [
        ("2km", (5568, 5568)),
        ("1km", (11136, 11136)),
    ],
)
def test_meteosat_fci_grid(resolution, expected_shape):
    lat, lon = MeteosatFCI.grid(resolution=resolution)
    assert lat.shape == expected_shape == lon.shape
    assert not np.all(np.isnan(lat))
    # Sub-satellite point (0°N, 0°E) should be near the centre of the grid
    mid_row = expected_shape[0] // 2
    mid_col = expected_shape[1] // 2
    assert abs(float(lat[mid_row, mid_col])) < 1.0
    assert abs(float(lon[mid_row, mid_col])) < 1.0


def test_meteosat_fci_grid_orientation():
    """FCI grid is south-to-north (lat increases with row) and
    west-to-east (lon increases with column) in native FCI order."""
    lat, lon = MeteosatFCI.grid(resolution="2km")
    # Use the central quarter of the disk to avoid NaN at the Earth's limb
    n = 5568
    q1, q3 = n // 4, 3 * n // 4
    mid = n // 2
    # Southern rows have lower latitudes than northern rows
    assert float(lat[q1, mid]) < float(lat[q3, mid])
    # Western columns have lower (more negative) longitudes than eastern ones
    assert float(lon[mid, q1]) < float(lon[mid, q3])


def test_meteosat_fci_available():
    assert MeteosatFCI.available(datetime(2024, 1, 16, 12, 0))
    assert MeteosatFCI.available(datetime(2025, 3, 1, 0, 0))
    assert not MeteosatFCI.available(datetime(2023, 12, 31, 0, 0))


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_meteosat_fci_invalid_resolution():
    with pytest.raises(ValueError, match="resolution must be one of"):
        MeteosatFCI(resolution="999km")  # type: ignore[arg-type]


def test_meteosat_fci_bbox_conflict():
    with pytest.raises(ValueError, match="At most one"):
        MeteosatFCI(
            lat_lon_bbox=((35.0, 60.0), (-10.0, 30.0)),
            pixel_bbox=((0, 100), (0, 100)),
        )


def test_meteosat_fci_mixed_resolution_error():
    # _check_resolution_consistency fires before any network call, so no
    # credentials or slow markers are needed.
    ds = MeteosatFCI(resolution="2km")
    with pytest.raises(ValueError, match="resolution"):
        # fci_ir_87 is 2 km (FDHSI); fci_vis_06 is 1 km/500 m only → mismatch
        ds(datetime(2025, 1, 1, 0, 0), ["fci_ir_87", "fci_vis_06"])
