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

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import E2S_GRID_ID, HEALPixGrid, infer_grid, register_grid

NSIDE1_LAT = [41.8103149] * 4 + [0.0] * 4 + [-41.8103149] * 4
NSIDE1_LON = [45, 135, 225, 315, 0, 90, 180, 270, 45, 135, 225, 315]
GOLDENS = (
    pytest.param(
        HEALPixGrid(0, "nested"),
        np.arange(12),
        NSIDE1_LAT,
        NSIDE1_LON,
        id="nside1-nested",
    ),
    pytest.param(
        HEALPixGrid(0, "ring"), np.arange(12), NSIDE1_LAT, NSIDE1_LON, id="nside1-ring"
    ),
    pytest.param(
        HEALPixGrid(2, "nested"),
        np.arange(4),
        [9.59406823, 19.47122063, 19.47122063, 30],
        [45, 56.25, 33.75, 45],
        id="nside4-nested",
    ),
    pytest.param(
        HEALPixGrid(2, "ring"),
        np.arange(4),
        [78.28414761] * 4,
        [45, 135, 225, 315],
        id="nside4-ring",
    ),
    pytest.param(
        HEALPixGrid(2, "xy", xy_origin="north", xy_clockwise=True),
        [0, 1, 4, 5],
        [78.28414761, 66.44353569, 66.44353569, 54.3409123],
        [45, 67.5, 22.5, 45],
        id="earth2grid-pad-xy",
    ),
)


@pytest.mark.parametrize(
    ("grid", "dims", "shape"),
    (
        (HEALPixGrid(1, "nested"), ("hpx",), (48,)),
        (HEALPixGrid(1, "ring"), ("hpx",), (48,)),
        (HEALPixGrid(1, "xy"), ("hpx",), (48,)),
        (
            HEALPixGrid(1, "xy", "face", "north", True),
            ("face", "height", "width"),
            (12, 2, 2),
        ),
    ),
)
def test_grid_contract(check_grid_contract, grid, dims, shape):
    check_grid_contract(grid, dims, shape, "healpix")
    assert grid.crs is None
    assert grid.attrs["nside"] == 2


@pytest.mark.parametrize(("grid", "pixels", "latitude", "longitude"), GOLDENS)
def test_grid_coordinates(grid, pixels, latitude, longitude):
    coordinates = grid.coords({"hpx": np.asarray(pixels)})
    np.testing.assert_allclose(coordinates["lat"], latitude, atol=1e-8)
    np.testing.assert_allclose(coordinates["lon"], longitude, atol=1e-12)


def test_grid_selection():
    grid = HEALPixGrid(2, "nested")
    indexes = grid.coords(only_index=True)
    positions = grid.subset_indexers(indexes, faces=(1, 3))["hpx"]
    np.testing.assert_array_equal(
        np.unique(np.asarray(indexes["hpx"])[positions] // 16), [1, 3]
    )

    ring = HEALPixGrid(1, "ring")
    with pytest.raises(NotImplementedError, match="RING"):
        ring.subset_indexers(ring.coords(only_index=True), faces=(0,))
    with pytest.raises(ValueError, match="0 through 11"):
        grid.subset_indexers(indexes, faces=())


def test_grid_validation():
    cases = (
        (lambda: HEALPixGrid(-1), "nonnegative"),
        (lambda: HEALPixGrid(1, "bad"), "ordering"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, layout="bad"), "layout"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, layout="face"), "requires XY"),
        (lambda: HEALPixGrid(1, "xy", xy_origin="bad"), "origin"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, "ring", xy_clockwise=True), "orientation"),
    )
    for operation, message in cases:
        with pytest.raises(ValueError, match=message):
            operation()


def test_grid_inference():
    grid = HEALPixGrid(0)
    register_grid("test-healpix-grid", grid)
    array = xr.DataArray(
        np.ones(12),
        dims="hpx",
        coords=grid.coords(),
        attrs={E2S_GRID_ID: "test-healpix-grid"},
    )
    assert infer_grid(array) is grid
    with pytest.raises(ValueError, match="unsupported layout"):
        infer_grid(array.drop_attrs())


@pytest.mark.parametrize("origin", ["south", "east", "north", "west"])
@pytest.mark.parametrize("clockwise", [False, True])
def test_grid_xy_orientations(origin, clockwise):
    expected = HEALPixGrid(1, "nested").coords()
    actual = HEALPixGrid(
        1, "xy", xy_origin=origin, xy_clockwise=clockwise  # type: ignore[arg-type]
    ).coords()
    expected = np.column_stack((expected["lat"], expected["lon"]))
    actual = np.column_stack((actual["lat"], actual["lon"]))
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_grid_face_layout():
    flat = HEALPixGrid(2, "xy", xy_origin="north", xy_clockwise=True).coords()
    grid = HEALPixGrid(2, "xy", "face", "north", True)
    coordinates = grid.coords()
    np.testing.assert_allclose(coordinates["lat"].values.ravel(), flat["lat"])
    np.testing.assert_allclose(coordinates["lon"].values.ravel(), flat["lon"])

    latitude = float(coordinates["lat"].sel(face=1, height=0, width=0))
    longitude = float(coordinates["lon"].sel(face=1, height=0, width=0))
    indexers = grid.subset_indexers(
        coordinates,
        faces=(1, 3),
        bounds=(longitude - 1e-6, latitude - 1e-6, longitude + 1e-6, latitude + 1e-6),
    )
    np.testing.assert_array_equal(indexers.pop("face"), [1])
    assert indexers == {"height": slice(0, 1), "width": slice(0, 1)}
    assert (grid.attrs["origin"], grid.attrs["clockwise"]) == ("north", True)
