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

import os
import pathlib
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from earth2studio.data import MTG
from earth2studio.data.mtg import _VARIABLE_RESOLUTION
from earth2studio.lexicon import MTGLexicon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recent_aligned_time(hours_back: int = 1) -> datetime:
    """Return a recent UTC time aligned to the 10-minute MTG scan boundary.

    Uses *now - hours_back* so the test always targets data within EUMETSAT's
    rolling ~30-day archive and the product is already fully processed.
    """
    now = datetime.now(tz=timezone.utc)
    t = now.replace(second=0, microsecond=0)
    t = t.replace(minute=(t.minute // 10) * 10)
    return t - timedelta(hours=hours_back)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mtg_credentials():
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        pytest.skip("EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET not set")
    return key, secret


# ---------------------------------------------------------------------------
# Fast tests – no credentials required
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolution,expected_shape",
    [
        ("2km", (5568, 5568)),
        ("1km", (11136, 11136)),
    ],
)
def test_mtg_grid(resolution, expected_shape):
    """MTG.grid() must return lat/lon arrays of the correct shape."""
    lat, lon = MTG.grid(resolution)
    assert lat.shape == expected_shape
    assert lon.shape == expected_shape
    # Off-disk pixels (corners) should be NaN; on-disk pixels should be finite
    assert np.any(np.isnan(lat))
    assert np.any(np.isfinite(lat))


def test_mtg_grid_invalid_resolution():
    """MTG.grid() must reject unrecognised resolution strings."""
    with pytest.raises(ValueError, match="resolution"):
        MTG.grid("500m")


@pytest.mark.parametrize(
    "time,expected",
    [
        (datetime(2024, 1, 16, tzinfo=timezone.utc), True),
        (datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc), True),
        (datetime(2023, 12, 31, 23, 59, tzinfo=timezone.utc), False),
        (datetime(2020, 1, 1, tzinfo=timezone.utc), False),
        (np.datetime64("2025-01-01T00:00:00"), True),
        (np.datetime64("2023-06-01T00:00:00"), False),
    ],
)
def test_mtg_available(time, expected):
    """MTG.available() should reflect the MTG-I1 operational start date."""
    assert MTG.available(time) == expected


def test_mtg_lexicon():
    """All 12 MTG FCI non-HR channels must be present in MTGLexicon."""
    expected_vars = {
        "mtg_vis_04", "mtg_vis_05", "mtg_vis_08", "mtg_vis_09",
        "mtg_nir_13", "mtg_nir_16",
        "mtg_wv_63",  "mtg_wv_73",  "mtg_ir_87",  "mtg_ir_97",
        "mtg_ir_123", "mtg_ir_133",
    }
    for v in expected_vars:
        channel, modifier = MTGLexicon[v]
        assert isinstance(channel, str)
        assert callable(modifier)
        # modifier is identity – value should pass through unchanged
        assert modifier(1.0) == 1.0


def test_mtg_mixed_resolution_error():
    """Mixing 1 km and 2 km variables in one call must raise ValueError."""
    # Error is raised before authentication, so dummy credentials are fine
    ds = MTG("dummy_key", "dummy_secret", cache=False)
    with pytest.raises(ValueError, match="mixed resolutions"):
        ds(
            _recent_aligned_time(),
            ["mtg_vis_04", "mtg_ir_87"],  # 1 km + 2 km
        )


# ---------------------------------------------------------------------------
# Slow tests – require EUMETSAT credentials via env vars
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "variable",
    [
        # Single 2 km channel
        "mtg_ir_87",
        # Multiple 2 km channels
        ["mtg_wv_63", "mtg_ir_87", "mtg_ir_133"],
        # Single 1 km channel
        "mtg_vis_04",
    ],
)
def test_mtg_fetch(mtg_credentials, variable):
    """Full-disk fetch at native resolution (single timestep)."""
    key, secret = mtg_credentials
    time = _recent_aligned_time()
    ds = MTG(key, secret, cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]

    assert data.dims == ("time", "variable", "y", "x")
    assert data.shape[0] == 1
    assert data.shape[1] == len(variable)

    res_m = _VARIABLE_RESOLUTION[MTGLexicon[variable[0]][0]]
    expected_hw = MTG.GRID_SIZE_1KM[0] if res_m == 1000 else MTG.GRID_SIZE_2KM[0]
    assert data.shape[2] == expected_hw
    assert data.shape[3] == expected_hw

    assert "_lat" in data.coords
    assert "_lon" in data.coords
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.all(np.isnan(data.values))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(600)
def test_mtg_fetch_multistep(mtg_credentials):
    """Fetch two consecutive timesteps and check output shape."""
    key, secret = mtg_credentials
    t0 = _recent_aligned_time(hours_back=2)
    t1 = t0 + timedelta(minutes=10)
    ds = MTG(key, secret, cache=False)
    data = ds([t0, t1], "mtg_ir_133")

    assert data.shape[0] == 2
    assert data.shape[1] == 1
    assert data.shape[2] == MTG.GRID_SIZE_2KM[0]
    assert data.shape[3] == MTG.GRID_SIZE_2KM[1]


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "roi,variable",
    [
        # Europe (2 km channels)
        ((35.0, 60.0, -10.0, 30.0), "mtg_ir_87"),
        # Western Europe (1 km channels)
        ((40.0, 55.0, -5.0, 15.0), "mtg_vis_04"),
        # Multiple 2 km channels with same ROI
        ((35.0, 60.0, -10.0, 30.0), ["mtg_wv_63", "mtg_ir_133"]),
    ],
)
def test_mtg_roi_fetch(mtg_credentials, roi, variable):
    """ROI fetch must return a cropped array smaller than the full disk."""
    key, secret = mtg_credentials
    time = _recent_aligned_time()
    ds = MTG(key, secret, roi=roi, cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]

    res_m = _VARIABLE_RESOLUTION[MTGLexicon[variable[0]][0]]
    full_size = MTG.GRID_SIZE_1KM[0] if res_m == 1000 else MTG.GRID_SIZE_2KM[0]

    # Output must be strictly smaller than the full disk
    assert data.shape[2] < full_size
    assert data.shape[3] < full_size
    assert not np.all(np.isnan(data.values))

    # The pixel bounding box is rectangular in geostationary pixel space, not lat/lon space.
    # Corner pixels can fall slightly outside the requested geographic ROI due to projection
    # distortion. Verify only that the median lat/lon of finite pixels is inside the ROI.
    lat_min, lat_max, lon_min, lon_max = roi
    lat = data.coords["_lat"].values
    lon = data.coords["_lon"].values
    finite = np.isfinite(lat) & np.isfinite(lon)
    assert lat_min <= np.median(lat[finite]) <= lat_max
    assert lon_min <= np.median(lon[finite]) <= lon_max


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(600)
@pytest.mark.parametrize("cache", [True, False])
def test_mtg_cache(mtg_credentials, cache):
    """Caching: second call should hit the cache; non-cache mode cleans up."""
    key, secret = mtg_credentials
    time = _recent_aligned_time()
    variable = "mtg_ir_87"

    ds = MTG(key, secret, cache=cache, verbose=False)
    _ = ds(time, variable)

    cache_path = pathlib.Path(ds.cache)
    assert cache_path.is_dir() == cache

    if cache:
        # Second call must succeed from cache without re-downloading
        data2 = ds(time, variable)
        assert data2.shape[2] == MTG.GRID_SIZE_2KM[0]
        assert data2.shape[3] == MTG.GRID_SIZE_2KM[1]
        try:
            shutil.rmtree(ds.cache)
        except FileNotFoundError:
            pass
