# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

import earth2studio.grids as e2s


def test_builtin_grid_registry_and_protocol():
    assert {
        "latlon-0.25deg",
        "latlon-0.25deg-south-pole-excluded",
        "hrrr-conus-3km",
        "healpix-l6-nested",
    } <= set(e2s.list_grids())

    hrrr = e2s.resolve_grid("hrrr")
    assert isinstance(hrrr, e2s.GridDefinition)
    assert e2s.resolve_grid("fcn1").shape == (720, 1440)
    assert e2s.GridDefinition not in type(hrrr).__mro__
    assert (hrrr.dims, hrrr.shape, hrrr.topology) == (
        ("y", "x"),
        (1059, 1799),
        "projected",
    )
    assert hrrr.crs.coordinate_operation is not None

    indexes = hrrr.coords(only_index=True)
    geographic = hrrr.coords(
        {dimension: np.asarray(indexes[dimension][:2]) for dimension in hrrr.dims}
    )
    assert geographic["lat"].shape == geographic["lon"].shape == (2, 2)
    assert hrrr.to_metadata() == {
        "type": "ProjectedGrid",
        "dims": ["y", "x"],
        "shape": [1059, 1799],
        "topology": "projected",
        "crs": hrrr.crs.to_string(),
    }
    assert hrrr.fingerprint()
    with pytest.raises(ValueError, match="read-only"):
        hrrr.x[0] = 0


def test_grid_definitions_and_selection():
    latlon = e2s.LatLonGrid([40.0, 39.0], [250.0, 251.0, 252.0])
    assert latlon.dims == ("lat", "lon") and latlon.shape == (2, 3)
    assert latlon.crs.to_epsg() == 4326
    assert tuple(latlon.coords()) == ("lat", "lon")
    assert latlon.cell_bounds(latlon.coords(only_index=True)) is None
    assert latlon.subset_indexers(
        latlon.coords(only_index=True), bounds=(-110, 38, -90, 41)
    ) == {"lat": slice(0, 2), "lon": slice(0, 3)}
    dateline = e2s.LatLonGrid([0], [300, 350, 0, 10, 20])
    assert dateline.subset_indexers(
        dateline.coords(only_index=True), bounds=(350, -1, 10, 1)
    ) == {"lat": slice(0, 1), "lon": slice(1, 4)}
    assert (
        latlon.fingerprint()
        != e2s.LatLonGrid(latlon.latitude, latlon.longitude, "OGC:CRS84").fingerprint()
    )

    latitude = np.array([[40.0, 40.1], [41.0, 41.1]])
    longitude = np.array([[-100.0, -99.0], [-100.1, -99.1]])
    curvilinear = e2s.CurvilinearGrid(latitude, longitude)
    assert curvilinear.crs is None and curvilinear.coords()["lat"].shape == (2, 2)
    with pytest.raises(ValueError, match="read-only"):
        curvilinear.x[0] = 1
    assert curvilinear.subset_indexers(
        curvilinear.coords(only_index=True), bounds=(-99.2, 39.5, -98.5, 41.5)
    ) == {"y": slice(0, 2), "x": slice(1, 2)}

    points = e2s.PointGrid([35.2, 40.8, 51.0], [-97.4, -74.0, 0.1])
    point_indexes = points.coords(only_index=True)
    assert points.dims == ("x",) and points.coords()["lat"].shape == (3,)
    with pytest.raises(ValueError, match="read-only"):
        points.x[0] = 1
    assert points.subset_indexers(point_indexes, bounds=(-100, 30, -90, 40)) == {
        "x": slice(0, 1)
    }


def test_healpix_orderings_and_layouts():
    nested = e2s.resolve_grid("hpx6")
    assert nested.shape == (49_152,) and nested.to_metadata()["nside"] == 64
    subset = nested.subset_indexers(nested.coords(only_index=True), faces=(1, 3))
    assert len(subset["hpx"]) == 2 * 64**2
    geographic = nested.coords({"hpx": np.arange(12)})
    assert np.isfinite(geographic["lat"]).all()

    ring = e2s.HEALPixGrid(level=1, ordering="ring")
    assert np.isfinite(ring.coords()["lat"]).all()
    with pytest.raises(NotImplementedError, match="RING"):
        ring.subset_indexers(ring.coords(only_index=True), faces=(0,))

    xy = e2s.HEALPixGrid(
        level=1, ordering="xy", xy_origin="north", xy_clockwise=True
    )
    expected = e2s.HEALPixGrid(level=1).coords({"hpx": np.array([3])})
    actual = xy.coords({"hpx": np.array([0])})
    np.testing.assert_allclose(actual["lat"], expected["lat"])
    np.testing.assert_allclose(actual["lon"], expected["lon"])

    faced = e2s.HEALPixGrid(
        level=1,
        ordering="xy",
        layout="face",
        xy_origin="north",
        xy_clockwise=True,
    )
    coordinates = faced.coords()
    assert faced.dims == ("face", "height", "width")
    assert coordinates["lat"].dims == faced.dims
    assert coordinates["lat"].shape == (12, 2, 2)
    assert faced.subset_indexers(coordinates, faces=(1, 3))["face"].tolist() == [1, 3]
    assert faced.to_metadata()["origin"] == "north"


def test_grid_registration_inference_and_validation():
    projected = e2s.ProjectedGrid(
        y=np.arange(2) * 3000.0,
        x=np.arange(3) * 3000.0,
        coordinate_reference_system="EPSG:3857",
    )
    e2s.register_grid("test-grid-protocol", projected, aliases=("test-grid",))
    e2s.register_grid("test-grid-protocol", projected, aliases=("test-grid",))
    assert e2s.resolve_grid("test-grid") is projected

    with pytest.raises(ValueError, match="conflicts"):
        e2s.register_grid("test-grid", projected)
    with pytest.raises(ValueError, match="valid CRS"):
        e2s.register_grid("EPSG:4326", projected)
    with pytest.raises(TypeError, match="implement GridDefinition"):
        e2s.register_grid("invalid-grid", object())  # type: ignore[arg-type]

    rectilinear = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [40.0, 39.0], "lon": [250.0, 251.0, 252.0]},
    )
    assert isinstance(e2s.infer_grid(rectilinear), e2s.LatLonGrid)

    curvilinear = xr.Dataset(
        coords={
            "lat": (("y", "x"), [[40.0, 40.1], [41.0, 41.1]]),
            "lon": (("y", "x"), [[-100.0, -99.0], [-100.1, -99.1]]),
        }
    )
    assert isinstance(e2s.infer_grid(curvilinear), e2s.CurvilinearGrid)

    points = xr.Dataset(
        coords={
            "x": np.arange(2),
            "lat": ("x", [35.2, 40.8]),
            "lon": ("x", [-97.4, -74.0]),
        }
    )
    assert isinstance(e2s.infer_grid(points), e2s.PointGrid)

    projected_array = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": np.arange(2), "x": np.arange(3)},
        attrs={e2s.E2S_CRS: "EPSG:3857"},
    )
    assert isinstance(e2s.infer_grid(projected_array), e2s.ProjectedGrid)
    registered = projected_array.assign_attrs(
        {e2s.E2S_GRID_ID: "test-grid-protocol"}
    )
    assert e2s.infer_grid(registered) is projected
    assert e2s.infer_grid(registered.expand_dims(time=[0])) is projected
    assert e2s.infer_grid(registered.isel(x=slice(2))).shape == (2, 2)
    assert isinstance(
        e2s.infer_grid(
            projected_array.assign_coords(
                lat=(("y", "x"), np.ones((2, 3))),
                lon=(("y", "x"), np.ones((2, 3))),
            )
        ),
        e2s.ProjectedGrid,
    )

    healpix = e2s.HEALPixGrid(level=0)
    e2s.register_grid("test-healpix", healpix)
    healpix_array = xr.DataArray(
        np.ones(12),
        dims="hpx",
        coords=healpix.coords(),
        attrs={e2s.E2S_GRID_ID: "test-healpix"},
    )
    assert e2s.infer_grid(healpix_array) is healpix
    with pytest.raises(ValueError, match="unsupported layout"):
        e2s.infer_grid(healpix_array.drop_attrs())

    for operation, message in (
        (lambda: e2s.resolve_grid("missing"), "Unknown"),
        (lambda: e2s.resolve_grid("EPSG:4326"), "does not define"),
        (lambda: e2s.LatLonGrid([], [0]), "nonempty"),
        (lambda: e2s.CurvilinearGrid([[0]], [[0, 1]]), "shapes must match"),
        (lambda: e2s.PointGrid([0], [0, 1]), "shapes must match"),
        (lambda: e2s.HEALPixGrid(-1), "nonnegative"),
        (lambda: e2s.HEALPixGrid(1, "bad"), "ordering"),  # type: ignore[arg-type]
        (lambda: e2s.HEALPixGrid(1, layout="face"), "requires XY"),
        (
            lambda: e2s.infer_grid(xr.DataArray(np.ones(1), dims="x")),
            "Cannot infer",
        ),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            operation()
