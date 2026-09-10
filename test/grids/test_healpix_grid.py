# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import E2S_GRID_ID, HEALPixGrid, infer_grid, register_grid

NSIDE_1_LATITUDE = np.array([41.8103149] * 4 + [0.0] * 4 + [-41.8103149] * 4)
NSIDE_1_LONGITUDE = np.array(
    [45.0, 135.0, 225.0, 315.0, 0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
)
NSIDE_4_NESTED_LATITUDE = np.array(
    [
        9.59406823,
        19.47122063,
        19.47122063,
        30.0,
        30.0,
        41.8103149,
        41.8103149,
        54.3409123,
    ]
)
NSIDE_4_NESTED_LONGITUDE = np.array(
    [45.0, 56.25, 33.75, 45.0, 67.5, 78.75, 56.25, 75.0]
)
NSIDE_4_RING_LATITUDE = np.array([78.28414761] * 4 + [66.44353569] * 4)
NSIDE_4_RING_LONGITUDE = np.array([45.0, 135.0, 225.0, 315.0, 22.5, 67.5, 112.5, 157.5])
PAD_XY_PIXELS = np.array([0, 1, 4, 5, 15, 16])
PAD_XY_LATITUDE = np.array(
    [78.28414761, 66.44353569, 66.44353569, 54.3409123, 9.59406823, 78.28414761]
)
PAD_XY_LONGITUDE = np.array([45.0, 67.5, 22.5, 45.0, 45.0, 135.0])


@pytest.mark.parametrize(
    ("grid", "dims", "shape"),
    [
        (HEALPixGrid(1, "nested"), ("hpx",), (48,)),
        (HEALPixGrid(1, "ring"), ("hpx",), (48,)),
        (HEALPixGrid(1, "xy"), ("hpx",), (48,)),
        (
            HEALPixGrid(1, "xy", "face", "north", True),
            ("face", "height", "width"),
            (12, 2, 2),
        ),
    ],
)
def test_healpix_grid_contract(
    check_grid: Callable[..., None],
    grid: HEALPixGrid,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
):
    check_grid(grid, dims, shape, "healpix")
    assert grid.crs is None
    assert grid.attrs["nside"] == 2


@pytest.mark.parametrize("ordering", ["nested", "ring"])
def test_healpix_nside_one_golden_coordinates(ordering: str):
    coordinates = HEALPixGrid(0, ordering).coords()
    np.testing.assert_allclose(coordinates["lat"], NSIDE_1_LATITUDE, atol=1e-8)
    np.testing.assert_allclose(coordinates["lon"], NSIDE_1_LONGITUDE, atol=1e-12)


@pytest.mark.parametrize(
    ("ordering", "latitude", "longitude"),
    [
        ("nested", NSIDE_4_NESTED_LATITUDE, NSIDE_4_NESTED_LONGITUDE),
        ("ring", NSIDE_4_RING_LATITUDE, NSIDE_4_RING_LONGITUDE),
    ],
)
def test_healpix_ordering_golden_coordinates(
    ordering: str, latitude: np.ndarray, longitude: np.ndarray
):
    coordinates = HEALPixGrid(2, ordering).coords({"hpx": np.arange(8)})
    np.testing.assert_allclose(coordinates["lat"], latitude, atol=1e-8)
    np.testing.assert_allclose(coordinates["lon"], longitude, atol=1e-12)


def test_healpix_pad_xy_golden_coordinates():
    grid = HEALPixGrid(2, "xy", xy_origin="north", xy_clockwise=True)
    coordinates = grid.coords({"hpx": PAD_XY_PIXELS})
    np.testing.assert_allclose(coordinates["lat"], PAD_XY_LATITUDE, atol=1e-8)
    np.testing.assert_allclose(coordinates["lon"], PAD_XY_LONGITUDE, atol=1e-12)


@pytest.mark.parametrize("origin", ["south", "east", "north", "west"])
@pytest.mark.parametrize("clockwise", [False, True])
def test_healpix_xy_orientations_preserve_pixel_centers(origin: str, clockwise: bool):
    expected = HEALPixGrid(1, "nested").coords()
    actual = HEALPixGrid(
        1,
        "xy",
        xy_origin=origin,  # type: ignore[arg-type]
        xy_clockwise=clockwise,
    ).coords()

    expected_pairs = np.column_stack((expected["lat"], expected["lon"]))
    actual_pairs = np.column_stack((actual["lat"], actual["lon"]))
    expected_order = np.lexsort((expected_pairs[:, 1], expected_pairs[:, 0]))
    actual_order = np.lexsort((actual_pairs[:, 1], actual_pairs[:, 0]))
    np.testing.assert_allclose(
        actual_pairs[actual_order], expected_pairs[expected_order], atol=1e-12
    )


def test_healpix_face_layout_matches_flat_xy():
    flat = HEALPixGrid(2, "xy", xy_origin="north", xy_clockwise=True)
    faced = HEALPixGrid(2, "xy", "face", "north", True)
    flat_coordinates = flat.coords()
    face_coordinates = faced.coords()
    np.testing.assert_allclose(
        face_coordinates["lat"].values.ravel(), flat_coordinates["lat"]
    )
    np.testing.assert_allclose(
        face_coordinates["lon"].values.ravel(), flat_coordinates["lon"]
    )
    assert faced.subset_indexers(face_coordinates, faces=(1, 3))["face"].tolist() == [
        1,
        3,
    ]
    assert faced.attrs["origin"] == "north"
    assert faced.attrs["clockwise"] is True


def test_healpix_face_layout_combines_face_and_bounds_selection():
    grid = HEALPixGrid(2, "xy", "face", "north", True)
    coordinates = grid.coords()
    latitude = float(coordinates["lat"].sel(face=1, height=0, width=0))
    longitude = float(coordinates["lon"].sel(face=1, height=0, width=0))
    indexers = grid.subset_indexers(
        coordinates,
        faces=(1, 3),
        bounds=(longitude - 1e-6, latitude - 1e-6, longitude + 1e-6, latitude + 1e-6),
    )
    assert indexers["face"].tolist() == [1]
    assert indexers["height"] == slice(0, 1)
    assert indexers["width"] == slice(0, 1)


def test_healpix_flat_face_selection():
    grid = HEALPixGrid(2, "nested")
    subset = grid.subset_indexers(grid.coords(only_index=True), faces=(1, 3))
    pixels = np.asarray(grid.coords(only_index=True)["hpx"])[subset["hpx"]]
    np.testing.assert_array_equal(
        np.unique(pixels // grid.nside**2),
        [1, 3],
    )

    with pytest.raises(NotImplementedError, match="RING"):
        HEALPixGrid(1, "ring").subset_indexers(
            HEALPixGrid(1, "ring").coords(only_index=True), faces=(0,)
        )
    with pytest.raises(ValueError, match="0 through 11"):
        grid.subset_indexers(grid.coords(only_index=True), faces=())
    with pytest.raises(ValueError, match="Unsupported"):
        grid.subset_indexers(grid.coords(only_index=True), radius=1)


def test_healpix_validation():
    for operation, message in (
        (lambda: HEALPixGrid(-1), "nonnegative"),
        (lambda: HEALPixGrid(1, "bad"), "ordering"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, layout="bad"), "layout"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, layout="face"), "requires XY"),
        (lambda: HEALPixGrid(1, "xy", xy_origin="bad"), "origin"),  # type: ignore[arg-type]
        (lambda: HEALPixGrid(1, "ring", xy_clockwise=True), "orientation"),
    ):
        with pytest.raises(ValueError, match=message):
            operation()


def test_healpix_inference_requires_explicit_metadata():
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
