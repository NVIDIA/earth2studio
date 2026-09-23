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


"""Tests for site extraction on curvilinear and regular grids."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.site_extraction import (
    CurvilinearSiteExtractor,
    RegularLatLonSiteExtractor,
)


def _grid_5x5() -> tuple[np.ndarray, np.ndarray]:
    """A synthetic regular 2-D lat/lon grid with lon in [0,360) (as HRRR.grid emits)."""
    lats = np.linspace(40.0, 44.0, 5)
    lons = np.linspace(260.0, 264.0, 5)
    lon2d, lat2d = np.meshgrid(lons, lats)  # (5, 5)
    return lat2d, lon2d


@pytest.mark.parametrize(
    ("grid_offset", "query_lon"),
    [
        pytest.param(0.0, 262.0, id="unsigned-grid-and-query"),
        pytest.param(0.0, -98.0, id="signed-query"),
        pytest.param(-360.0, -98.0, id="signed-grid-and-query"),
    ],
)
def test_curvilinear_extractor_maps_and_normalizes_longitude(
    grid_offset: float, query_lon: float
) -> None:
    lat, lon = _grid_5x5()
    lon += grid_offset
    ex = CurvilinearSiteExtractor(
        lat, lon, [{"id": "site", "lat": 42.0, "lon": query_lon}]
    )

    assert ex._site_indices["site"] == (2, 2)
    assert ex.site_lon["site"] == pytest.approx(262.0)
    assert ex.extract(lat)["site"] == pytest.approx(42.0)


def test_site_extractor_rejects_off_grid_site() -> None:
    lat, lon = _grid_5x5()
    with pytest.raises(ValueError, match="outside the model grid"):
        CurvilinearSiteExtractor(lat, lon, [{"id": "bad", "lat": 0.0, "lon": 0.0}])


@pytest.mark.parametrize(
    ("lat", "lon", "message"),
    [
        (np.array([1.0]), np.array([1.0]), "must be 2-D"),
        (np.ones((2, 2)), np.ones((2, 3)), "shapes must match"),
        (np.empty((0, 0)), np.empty((0, 0)), "must not be empty"),
        (
            np.array([[float("nan")]]),
            np.array([[0.0]]),
            "only finite values",
        ),
    ],
)
def test_site_extractor_rejects_invalid_grids(
    lat: np.ndarray, lon: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CurvilinearSiteExtractor(lat, lon, [{"id": "site", "lat": 0.0, "lon": 0.0}])


def _regular_global_axes() -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(90.0, -90.0, 721), np.linspace(0.0, 360.0, 1440, endpoint=False)


def test_regular_nearest_index_and_extract() -> None:
    lat, lon = _regular_global_axes()
    extractor = RegularLatLonSiteExtractor(
        lat, lon, [{"id": "a", "lat": 40.0, "lon": 262.0}]
    )
    field = np.arange(721 * 1440, dtype=float).reshape(721, 1440)

    assert extractor._site_indices["a"] == (200, 1048)
    assert extractor.site_lon["a"] == pytest.approx(262.0)
    assert extractor.extract(field) == {"a": float(200 * 1440 + 1048)}


def test_regular_longitude_is_cyclic() -> None:
    lat, lon = _regular_global_axes()
    extractor = RegularLatLonSiteExtractor(
        lat, lon, [{"id": "a", "lat": 0.0, "lon": -0.1}]
    )

    assert extractor._site_indices["a"][1] == 0


def test_regular_rejects_out_of_domain_on_regional_grid() -> None:
    lat = np.linspace(50.0, 30.0, 81)
    lon = np.linspace(250.0, 290.0, 161)
    with pytest.raises(ValueError, match="from the nearest grid cell"):
        RegularLatLonSiteExtractor(lat, lon, [{"id": "x", "lat": 40.0, "lon": 245.0}])

    extractor = RegularLatLonSiteExtractor(
        lat, lon, [{"id": "x", "lat": 40.0, "lon": 270.0}]
    )
    assert extractor._site_indices["x"] == (40, 80)


def test_regular_rejects_non_monotonic_axis() -> None:
    _, lon = _regular_global_axes()
    with pytest.raises(ValueError, match="strictly monotonic"):
        RegularLatLonSiteExtractor(
            np.array([0.0, 1.0, 0.5]),
            lon,
            [{"id": "a", "lat": 0.5, "lon": 10.0}],
        )


@pytest.mark.parametrize(
    ("lat", "message"),
    [
        pytest.param(np.array([]), "empty", id="empty"),
        pytest.param(np.zeros((3, 3)), "1-D", id="non-1d"),
    ],
)
def test_regular_rejects_invalid_axis(lat: np.ndarray, message: str) -> None:
    _, lon = _regular_global_axes()
    with pytest.raises(ValueError, match=message):
        RegularLatLonSiteExtractor(
            lat,
            lon,
            [{"id": "a", "lat": 0.0, "lon": 10.0}],
        )


@pytest.mark.parametrize("shape", [(10, 10), (1, 721, 1440)])
def test_regular_extract_rejects_wrong_field_shape(shape: tuple[int, ...]) -> None:
    lat, lon = _regular_global_axes()
    extractor = RegularLatLonSiteExtractor(
        lat, lon, [{"id": "a", "lat": 40.0, "lon": 262.0}]
    )
    with pytest.raises(ValueError, match="does not match the 2-D grid"):
        extractor.extract(np.zeros(shape))
