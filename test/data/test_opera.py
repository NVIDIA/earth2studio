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

from __future__ import annotations

import pathlib
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from earth2studio.data import OPERA
from earth2studio.lexicon import OPERALexicon

# ODYSSEY era (pre-2024-07): all products use 1900 × 2200 @ 2 km
_TEST_TIME_ODYSSEY = datetime(2024, 6, 1, 0, 0)
_ODYSSEY_YX = (2200, 1900)

# CIRRUS era (post-2024-07): DBZH is 3800 × 4400 @ 1 km;
# RATE and ACRR stay at 1900 × 2200 @ 2 km
_TEST_TIME_CIRRUS = datetime(2024, 9, 1, 0, 0)
_CIRRUS_DBZH_YX = (4400, 3800)
_CIRRUS_RATE_YX = (2200, 1900)


# ==========================================================================
# 1. Network fetch tests (slow, xfail)
# ==========================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "time,variable,expected_yx",
    [
        (_TEST_TIME_ODYSSEY, "refc", _ODYSSEY_YX),
        (_TEST_TIME_ODYSSEY, "tprate", _ODYSSEY_YX),
        (_TEST_TIME_ODYSSEY, "tp01", _ODYSSEY_YX),
        (_TEST_TIME_CIRRUS, "refc", _CIRRUS_DBZH_YX),
        (_TEST_TIME_CIRRUS, "tprate", _CIRRUS_RATE_YX),
        (_TEST_TIME_CIRRUS, "tp01", _CIRRUS_RATE_YX),
        (
            [_TEST_TIME_ODYSSEY, _TEST_TIME_ODYSSEY + timedelta(minutes=15)],
            "refc",
            _ODYSSEY_YX,
        ),
        # CIRRUS era: 5-minute boundary
        (_TEST_TIME_CIRRUS + timedelta(minutes=5), "refc", _CIRRUS_DBZH_YX),
    ],
)
def test_opera_fetch(time, variable, expected_yx):
    ds = OPERA(cache=False)
    data = ds(time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    ny, nx = expected_yx
    assert data.shape == (len(time), len(variable), ny, nx)
    assert data.dims == ("time", "variable", "y", "x")
    assert "_lat" in data.coords
    assert "_lon" in data.coords
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_opera_mixed_grid_raises():
    """CIRRUS era DBZH + RATE/ACRR in one call raises due to differing grids."""
    ds = OPERA(cache=False)
    with pytest.raises(ValueError, match="different pixel grids"):
        ds(_TEST_TIME_CIRRUS, ["refc", "tprate"])


# ==========================================================================
# 2. Cache test (slow, xfail)
# ==========================================================================


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize("cache", [True, False])
def test_opera_cache(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    ds = OPERA(cache=cache)
    data = ds(_TEST_TIME_ODYSSEY, ["refc"])

    ny, nx = _ODYSSEY_YX
    assert data.shape == (1, 1, ny, nx)
    assert pathlib.Path(ds.cache).is_dir() == cache

    data2 = ds(_TEST_TIME_ODYSSEY, ["refc"])
    assert data2.shape == data.shape

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


# ==========================================================================
# 3. Mock tests — no network, ODYSSEY era grid
# ==========================================================================


@pytest.mark.timeout(30)
def test_opera_call_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    ny, nx = _ODYSSEY_YX
    fake_grid = np.full((ny, nx), 20.0, dtype=np.float32)

    async def _fake_fetch_array(task):
        return fake_grid

    ds = OPERA(cache=True)
    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "fetch_array", side_effect=_fake_fetch_array),
    ):
        ds.fs = object()
        data = ds(_TEST_TIME_ODYSSEY, ["refc", "tprate"])

    assert data.shape == (1, 2, ny, nx)
    assert data.dims == ("time", "variable", "y", "x")
    assert "_lat" in data.coords
    assert "_lon" in data.coords
    np.testing.assert_allclose(data.sel(variable="refc").values[0], fake_grid)
    np.testing.assert_allclose(data.sel(variable="tprate").values[0], fake_grid)


@pytest.mark.timeout(30)
def test_opera_call_mock_multi_time(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    ny, nx = _ODYSSEY_YX
    fake_grid = np.zeros((ny, nx), dtype=np.float32)

    async def _fake_fetch_array(task):
        return fake_grid + task.time_idx  # distinct value per time step

    times = [_TEST_TIME_ODYSSEY, _TEST_TIME_ODYSSEY + timedelta(minutes=15)]
    ds = OPERA(cache=False)

    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "fetch_array", side_effect=_fake_fetch_array),
    ):
        ds.fs = object()
        data = ds(times, ["refc"])

    assert data.shape == (2, 1, ny, nx)
    np.testing.assert_allclose(data[0, 0].values, 0.0)
    np.testing.assert_allclose(data[1, 0].values, 1.0)


# ==========================================================================
# 4. Mock test — mixed grid raises ValueError
# ==========================================================================


@pytest.mark.timeout(30)
def test_opera_call_mock_mixed_grid_raises(tmp_path, monkeypatch):
    """If mocked fetch_array returns different shapes, fetch raises ValueError."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    ny_dbzh, nx_dbzh = _CIRRUS_DBZH_YX
    ny_rate, nx_rate = _CIRRUS_RATE_YX

    shapes = {
        "refc": (ny_dbzh, nx_dbzh),
        "tprate": (ny_rate, nx_rate),
    }

    async def _fake_fetch_array(task):
        from earth2studio.lexicon import OPERALexicon

        qty, _ = OPERALexicon[["refc", "tprate"][task.var_idx]]
        ny, nx = shapes[["refc", "tprate"][task.var_idx]]
        return np.zeros((ny, nx), dtype=np.float32)

    ds = OPERA(cache=False)
    with (
        patch.object(ds, "_async_init", new=AsyncMock(return_value=None)),
        patch.object(ds, "fetch_array", side_effect=_fake_fetch_array),
    ):
        ds.fs = object()
        with pytest.raises(ValueError, match="different pixel grids"):
            ds(_TEST_TIME_CIRRUS, ["refc", "tprate"])


# ==========================================================================
# 5. Exception / error-path tests
# ==========================================================================


@pytest.mark.timeout(10)
def test_opera_exceptions():
    with pytest.raises(KeyError):
        OPERALexicon["nonexistent_variable"]

    ds = OPERA(cache=False)

    with pytest.raises(ValueError):
        ds._validate_time([datetime(2000, 1, 1, 0)])  # before archive start

    with pytest.raises(ValueError):
        ds._validate_time([datetime(2024, 6, 1, 0, 7)])  # not on ODYSSEY 15-min grid

    with pytest.raises(ValueError):
        ds._validate_time([datetime(2024, 9, 1, 0, 7)])  # not on CIRRUS 5-min grid

    # 5-min boundaries accepted in CIRRUS era
    ds._validate_time([datetime(2024, 9, 1, 0, 5)])
    ds._validate_time([datetime(2024, 9, 1, 0, 10)])


# ==========================================================================
# 6. Available classmethod
# ==========================================================================


@pytest.mark.timeout(5)
def test_opera_available():
    # ODYSSEY era: 15-minute grid
    assert OPERA.available(datetime(2024, 6, 1, 0)) is True
    assert OPERA.available(datetime(2024, 6, 1, 0, 15)) is True
    assert OPERA.available(datetime(2024, 6, 1, 0, 7)) is False  # off ODYSSEY grid
    assert OPERA.available(datetime(2000, 1, 1, 0)) is False  # before archive
    assert OPERA.available(np.datetime64("2024-06-01T00:00")) is True
    # CIRRUS era: 5-minute grid
    assert OPERA.available(datetime(2024, 9, 1, 0)) is True
    assert OPERA.available(datetime(2024, 9, 1, 0, 5)) is True
    assert (
        OPERA.available(datetime(2024, 9, 1, 0, 15)) is True
    )  # 15 is also a 5-min boundary
    assert OPERA.available(datetime(2024, 9, 1, 0, 7)) is False  # off CIRRUS grid too


# ==========================================================================
# 7. URL construction
# ==========================================================================


@pytest.mark.timeout(5)
def test_opera_build_url():
    # CIRRUS era (post-2024-07): plain quantity name
    url = OPERA._build_url(datetime(2025, 1, 1, 12, 0), "DBZH")
    assert url.endswith("OPERA@20250101T1200@0@DBZH.h5")
    assert "openradar-archive" in url

    # ODYSSEY era (pre-2024-07): legacy filename parameters
    url_legacy = OPERA._build_url(datetime(2022, 3, 15, 6, 30), "DBZH")
    assert url_legacy.endswith("OPERA@20220315T0630@0@DBZH_QIND.h5")

    url_rate = OPERA._build_url(datetime(2022, 3, 15, 6, 30), "RATE")
    assert url_rate.endswith("OPERA@20220315T0630@0@QIND_RATE.h5")

    url_acrr = OPERA._build_url(datetime(2022, 3, 15, 6, 30), "ACRR")
    assert url_acrr.endswith("OPERA@20220315T0630@0@ACRR_QIND.h5")


# ==========================================================================
# 8. Lexicon
# ==========================================================================


@pytest.mark.timeout(5)
def test_opera_lexicon():
    quantity, mod = OPERALexicon["refc"]
    assert quantity == "DBZH"

    quantity, mod = OPERALexicon["tprate"]
    assert quantity == "RATE"

    quantity, mod = OPERALexicon["tp01"]
    assert quantity == "ACRR"

    arr = np.array([1000.0, 2500.0])
    np.testing.assert_array_equal(mod(arr), np.array([1.0, 2.5]))
