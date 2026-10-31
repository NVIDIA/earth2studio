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
import torch
import xarray as xr

from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    BilinearRegridder,
    CurvilinearGrid,
    LatLonGrid,
    NearestRegridder,
    PointGrid,
    ProjectedGrid,
    Regridder,
    list_regridders,
    register_regridder,
    resolve_grid,
    resolve_regridder,
)
from earth2studio.utils.coords import coord_array, handshake_dataarray

GLOBAL = LatLonGrid(np.linspace(90, -90, 19), np.arange(0.0, 360.0, 10.0))
REGION = LatLonGrid(np.array([40.0, 30.0, 20.0]), np.array([250.0, 260.0, 270.0]))
HRRR = resolve_grid("hrrr")
HRRR_CROP = ProjectedGrid(HRRR.y[500:520:4], HRRR.x[800:830:5], HRRR.crs)


def field(grid, function, leading=("time", "variable")):
    """A field evaluating ``function(lat, lon)`` with two leading frames."""
    coords = grid.coords()
    lat, lon = xr.broadcast(coords["lat"], coords["lon"])
    values = function(
        lat.transpose(*grid.dims).values, lon.transpose(*grid.dims).values
    )
    data = np.stack([values, 2 * values])[:, None]
    return xr.DataArray(
        data,
        dims=(*leading, *grid.dims),
        coords={
            leading[0]: [0, 1],
            leading[1]: ["a"],
            **coords,
            "label": (leading[0], ["x", "y"]),
        },
        name="weather",
        attrs={"source": "test", **grid.attrs},
    )


def linear(lat, lon):
    return lat + 2 * np.mod(lon, 360)


def target_values(grid, function):
    coords = grid.coords()
    lat, lon = xr.broadcast(coords["lat"], coords["lon"])
    return function(lat.transpose(*grid.dims).values, lon.transpose(*grid.dims).values)


@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
def test_protocol_and_output_contract(cls):
    target = LatLonGrid(np.array([35.0, 25.0]), np.array([255.0, 265.0]))
    regridder = cls(REGION, target)
    assert isinstance(regridder, Regridder)
    assert regridder.source_grid is REGION and regridder.target_grid is target
    assert regridder.to("cpu") is regridder

    x = field(REGION, linear)
    y = regridder(x)
    # R3: non-spatial dimensions and coordinates pass through, in order.
    assert y.dims == ("time", "variable", "lat", "lon")
    np.testing.assert_array_equal(y["label"], ["x", "y"])
    # R4: spatial coordinates and grid metadata follow the target.
    for name, values in target.coords().items():
        np.testing.assert_array_equal(y[name], values)
    assert y.attrs["source"] == "test"
    assert y.attrs["dims"] == ["lat", "lon"] and y.attrs["shape"] == [2, 2]
    assert y.name == "weather"
    signature = coord_array(
        ("time", "variable", "lat", "lon"),
        {"variable": ["a"]},
        dynamic=("time",),
        grid=target,
    )
    handshake_dataarray(y, signature)
    # R5: NumPy in, NumPy out; R6: calls do not change results.
    assert isinstance(y.data, np.ndarray)
    xr.testing.assert_identical(regridder(x), y)


def test_spatial_dimensions_move_last():
    x = field(REGION, linear).transpose("lat", "time", "lon", "variable")
    y = BilinearRegridder(REGION, REGION)(x)
    assert y.dims == ("time", "variable", "lat", "lon")
    xr.testing.assert_allclose(y, field(REGION, linear), check_dim_order=False)


@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
def test_source_grid_mismatch_raises(cls):
    regridder = cls(REGION, GLOBAL)
    with pytest.raises(ValueError, match="source grid"):
        regridder(field(GLOBAL, linear))
    with pytest.raises(ValueError, match="source grid"):
        regridder(field(REGION, linear).rename(lat="y"))
    # Same shape, flipped latitude labels.
    with pytest.raises(ValueError, match="does not match the source grid"):
        regridder(field(REGION, linear).isel(lat=slice(None, None, -1)))
    with pytest.raises(TypeError):
        regridder(np.zeros(REGION.shape))


def test_named_target_records_grid_id():
    regridder = NearestRegridder(GLOBAL, "latlon-1deg")
    x = field(GLOBAL, linear)
    x.attrs[E2S_GRID_ID] = "some-source-grid"
    y = regridder(x)
    assert y.attrs[E2S_GRID_ID] == "latlon-1deg"
    assert E2S_GRID_ID not in NearestRegridder(GLOBAL, GLOBAL)(x).attrs


def test_projected_target_metadata():
    y = BilinearRegridder(GLOBAL, HRRR_CROP)(field(GLOBAL, linear))
    assert y.dims[-2:] == ("y", "x")
    assert y.attrs[E2S_CRS] == HRRR.crs.to_string()
    assert y.attrs["grid_mapping_name"] == "lambert_conformal_conic"
    np.testing.assert_array_equal(y["lat"], HRRR_CROP.coords()["lat"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
def test_dtype(cls, dtype):
    x = field(REGION, linear).astype(dtype)
    y = cls(REGION, REGION)(x)
    assert y.dtype == (np.float32 if dtype == np.int32 else dtype)
    np.testing.assert_allclose(y, x)


def test_bilinear_exact_for_linear_fields():
    target = LatLonGrid(np.array([37.5, 21.0]), np.array([251.0, 266.5, 269.0]))
    y = BilinearRegridder(REGION, target)(field(REGION, linear))
    np.testing.assert_allclose(y[0, 0], target_values(target, linear))
    np.testing.assert_allclose(y[1, 0], 2 * target_values(target, linear))


def test_bilinear_descending_and_ascending_latitude_agree():
    ascending = LatLonGrid(REGION.latitude[::-1], REGION.longitude)
    target = LatLonGrid(np.array([37.5, 21.0]), np.array([251.0, 266.5]))
    a = BilinearRegridder(REGION, target)(field(REGION, linear))
    b = BilinearRegridder(ascending, target)(field(ascending, linear))
    xr.testing.assert_allclose(a, b)


def test_bilinear_wraps_global_longitude():
    target = LatLonGrid(np.array([0.0]), np.array([355.0, -5.0, 5.0]))
    x = field(GLOBAL, lambda lat, lon: np.where(lon == 350, 1.0, 0.0) + lat * 0)
    y = BilinearRegridder(GLOBAL, target)(x)
    # 355 and -5 sit halfway between 350 and the seam point 0.
    np.testing.assert_allclose(y[0, 0, 0], [0.5, 0.5, 0.0])
    assert BilinearRegridder(GLOBAL, target).coverage.all()


def test_regional_longitude_convention():
    # The region spans 250-270 east; the same points in degrees west.
    target = LatLonGrid(np.array([30.0]), np.array([-105.0, -95.0, -80.0]))
    regridder = BilinearRegridder(REGION, target)
    y = regridder(field(REGION, linear))
    np.testing.assert_allclose(y[0, 0, 0, :2], 30 + 2 * np.array([255.0, 265.0]))
    np.testing.assert_array_equal(regridder.coverage, [[True, True, False]])
    assert np.isnan(y[0, 0, 0, 2])


@pytest.mark.parametrize("fill_value", [np.nan, 0.0])
def test_fill_value_only_outside_coverage(fill_value):
    target = LatLonGrid(np.array([30.0, 60.0]), np.array([260.0]))
    x = field(REGION, linear)
    x[:, :, 0, 0] = np.nan  # A source NaN away from the covered target.
    y = BilinearRegridder(REGION, target, fill_value=fill_value)(x)
    np.testing.assert_allclose(y[0, 0, 0, 0], 30 + 2 * 260)
    np.testing.assert_equal(y[0, 0, 1, 0].values, fill_value)


def test_zero_weight_neighbors_do_not_leak_nan():
    x = field(REGION, linear)
    x[:, :, 1, 1] = np.nan
    y = BilinearRegridder(REGION, REGION)(x)
    assert np.isnan(y[:, :, 1, 1]).all()
    assert np.isnan(y).sum() == 2


def test_bilinear_projected_source_is_exact():
    lcc = ProjectedGrid(HRRR.y[480:560:10], HRRR.x[780:860:10], HRRR.crs.to_string())

    def projected(lat, lon):
        # A field linear in the source's native y and x.
        from pyproj import Transformer

        x, y = Transformer.from_crs(4326, HRRR.crs, always_xy=True).transform(lon, lat)
        return x / 1000 + 3 * y / 1000

    target = CurvilinearGrid(
        HRRR_CROP.coords()["lat"].values, HRRR_CROP.coords()["lon"].values
    )
    y = BilinearRegridder(lcc, target)(field(lcc, projected))
    np.testing.assert_allclose(y[0, 0], target_values(target, projected), rtol=1e-9)


def test_bilinear_same_crs_projected_target_is_selection():
    parent = ProjectedGrid(HRRR.y[500:530], HRRR.x[800:840], HRRR.crs)
    x = field(parent, linear)
    y = BilinearRegridder(parent, HRRR_CROP)(x)
    expected = x.sel(y=HRRR_CROP.y, x=HRRR_CROP.x)
    np.testing.assert_allclose(y.values, expected.values, rtol=1e-12)


def test_bilinear_curvilinear_source():
    pytest.importorskip("scipy")
    lat, lon = np.meshgrid(REGION.latitude, REGION.longitude, indexing="ij")
    curvilinear = CurvilinearGrid(lat, lon)
    target = LatLonGrid(np.array([37.5, 21.0]), np.array([251.0, 266.5]))
    y = BilinearRegridder(curvilinear, target)(field(curvilinear, linear))
    np.testing.assert_allclose(y[0, 0], target_values(target, linear))


def test_bilinear_rejects_unstructured_source():
    points = PointGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="NearestRegridder"):
        BilinearRegridder(points, REGION)
    with pytest.raises(ValueError, match="two points"):
        BilinearRegridder(LatLonGrid(np.array([10.0]), np.array([20.0])), REGION)


def test_nearest_latlon_selects_nearest_points():
    target = LatLonGrid(np.array([34.0, 26.0]), np.array([254.0, 266.0]))
    y = NearestRegridder(REGION, target)(field(REGION, linear))
    np.testing.assert_allclose(y[0, 0], [[30 + 500, 30 + 540], [30 + 500, 30 + 540]])


def test_nearest_max_distance():
    target = LatLonGrid(np.array([30.0, 60.0]), np.array([260.0]))
    regridder = NearestRegridder(REGION, target, max_dist_km=100.0)
    np.testing.assert_array_equal(regridder.coverage, [[True], [False]])
    y = regridder(field(REGION, linear))
    assert np.isnan(y[0, 0, 1, 0]) and not np.isnan(y[0, 0, 0, 0])
    filled = NearestRegridder(REGION, target, max_dist_km=100.0, fill_value=-1.0)
    assert filled(field(REGION, linear))[0, 0, 1, 0] == -1
    # Without a fill value, targets outside coverage raise at construction.
    with pytest.raises(ValueError, match="1 of 2 target points"):
        NearestRegridder(REGION, target, max_dist_km=100.0, fill_value=None)
    covered = LatLonGrid(np.array([30.0]), np.array([260.0]))
    y = NearestRegridder(REGION, covered, max_dist_km=100.0, fill_value=None)(
        field(REGION, linear)
    )
    assert not np.isnan(y).any()


def test_nearest_tree_skips_invalid_source_points():
    pytest.importorskip("scipy")
    lat, lon = np.meshgrid(REGION.latitude, REGION.longitude, indexing="ij")
    lat[1, 1] = np.nan  # e.g. an off-disk satellite pixel
    curvilinear = CurvilinearGrid(lat, lon)
    target = PointGrid(np.array([30.0, 40.0]), np.array([260.0, 250.0]))
    x = field(curvilinear, lambda lat, lon: np.arange(9.0).reshape(3, 3))
    y = NearestRegridder(curvilinear, target)(x)
    assert not np.isnan(y).any()
    assert y[0, 0, 0] != 4 and y[0, 0, 1] == 0


def test_nearest_source_mask_fills_from_valid_points():
    pytest.importorskip("scipy")
    # Mask the centre point; targets on it take the nearest unmasked value.
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    x = field(REGION, lambda lat, lon: np.arange(9.0).reshape(3, 3))
    target = PointGrid(np.array([REGION.latitude[1]]), np.array([REGION.longitude[1]]))
    unmasked = NearestRegridder(REGION, target)(x)
    masked = NearestRegridder(REGION, target, source_mask=mask)(x)
    assert unmasked[0, 0, 0] == 4 and masked[0, 0, 0] != 4
    assert not np.isnan(masked).any()
    # An xarray mask is accepted in any dimension order.
    as_array = xr.DataArray(mask.T, dims=("lon", "lat"))
    regridder = NearestRegridder(REGION, target, source_mask=as_array)
    assert regridder.source_mask is not None
    np.testing.assert_array_equal(regridder.source_mask, mask)
    assert NearestRegridder(REGION, target).source_mask is None
    with pytest.raises(ValueError, match="does not match the source grid"):
        NearestRegridder(REGION, target, source_mask=mask[:2])
    with pytest.raises(ValueError, match="no valid source points"):
        NearestRegridder(REGION, target, source_mask=np.zeros((3, 3), dtype=bool))


_RANDOM = np.random.default_rng(0)


@pytest.mark.parametrize(
    "source, target",
    [
        # Random targets over the globe, including longitudes outside [0, 360).
        (
            GLOBAL,
            PointGrid(_RANDOM.uniform(-90, 90, 500), _RANDOM.uniform(-180, 360, 500)),
        ),
        # Latitude 60 is nearer in degrees, but 80 is nearer on the sphere.
        (
            LatLonGrid(np.array([60.0, 80.0]), np.array([0.0, 60.0])),
            PointGrid(np.array([69.0]), np.array([25.0])),
        ),
        # The meridian passes closest on the far side, so the end row is nearest.
        (
            LatLonGrid(np.array([-80.0, -60.0, -40.0]), np.array([0.0])),
            PointGrid(np.array([10.0]), np.array([180.0])),
        ),
        # A longitude axis crossing the seam.
        (
            LatLonGrid(np.array([1.0, 0.0]), np.array([300.0, 350.0, 0.0, 10.0, 20.0])),
            PointGrid(np.array([0.5, 0.5]), np.array([-8.0, 15.5])),
        ),
        # Single-point axes.
        (
            LatLonGrid(np.array([10.0]), np.array([20.0])),
            PointGrid(np.array([0.0, 5.0]), np.array([0.0, 30.0])),
        ),
        (
            LatLonGrid(np.array([10.0]), np.array([20.0, 30.0])),
            PointGrid(np.array([0.0, 5.0]), np.array([0.0, 30.0])),
        ),
        (
            LatLonGrid(np.array([10.0, 0.0]), np.array([20.0])),
            PointGrid(np.array([0.0, 5.0]), np.array([0.0, 30.0])),
        ),
    ],
)
def test_nearest_matches_brute_force(source, target):
    pytest.importorskip("scipy")
    lat, lon = target.latitude[:, None], target.longitude[:, None]

    def great_circle(a, b):
        a, b, p, q = map(np.radians, (a, b, lat, lon))
        cosine = np.sin(a) * np.sin(p) + np.cos(a) * np.cos(p) * np.cos(b - q)
        return np.arccos(np.clip(cosine, -1, 1))

    # Regrid the source's own coordinates to recover each selected point.
    regridder = NearestRegridder(source, target)
    selected = (
        regridder(field(source, lambda lat, lon: lat + 0 * lon))[0, 0].values,
        regridder(field(source, lambda lat, lon: lon + 0 * lat))[0, 0].values,
    )
    points = np.meshgrid(source.latitude, source.longitude, indexing="ij")
    nearest = great_circle(*(axis.ravel()[None] for axis in points)).min(axis=1)
    # Pole rows tie across longitudes, so compare distances rather than indices.
    np.testing.assert_allclose(
        great_circle(*(np.asarray(axis)[:, None] for axis in selected))[:, 0],
        nearest,
        atol=1e-12,
    )


def test_bilinear_across_the_seam():
    source = LatLonGrid(np.array([1.0, 0.0]), np.array([300.0, 350.0, 0.0, 10.0, 20.0]))
    target = PointGrid(np.array([0.5, 0.5]), np.array([-8.0, 15.5]))
    x = field(source, lambda lat, lon: np.mod(lon + 180, 360) + 0 * lat)
    # Values are continuous across the seam here, so bilinear is exact.
    np.testing.assert_allclose(
        BilinearRegridder(source, target)(x)[0, 0], [172.0, 195.5]
    )


@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
@pytest.mark.parametrize("longitude", [[0.0, 200.0, 300.0], [0.0, 270.0]])
def test_coarse_longitude_axes_keep_coverage(cls, longitude):
    source = LatLonGrid(np.array([1.0, 0.0]), np.array(longitude))
    target = PointGrid(np.array([0.5, 0.5]), np.array([100.0, 250.0]))
    y = cls(source, target)(field(source, linear))
    assert cls(source, target).coverage.all()
    if cls is BilinearRegridder:
        np.testing.assert_allclose(y[0, 0], 0.5 + 2 * np.array([100.0, 250.0]))


def test_nearest_non_monotonic_axes():
    pytest.importorskip("scipy")
    source = LatLonGrid(np.array([0.0, 20.0, 10.0]), np.array([0.0, 10.0]))
    target = PointGrid(np.array([18.0]), np.array([1.0]))
    x = field(source, lambda lat, lon: lat + 0 * lon)
    np.testing.assert_allclose(NearestRegridder(source, target)(x)[0, 0], [20.0])


def test_parity_with_interp_kernels():
    pytest.importorskip("scipy")
    pytest.importorskip("earth2grid")
    from earth2studio.utils.interp import (
        LatLonInterpolation,
        NearestNeighborInterpolator,
    )

    lat, lon = (HRRR_CROP.coords()[name].values for name in ("lat", "lon"))
    x = field(
        GLOBAL, lambda lat, lon: np.sin(np.radians(lat)) * np.cos(np.radians(lon))
    )
    values = torch.tensor(x.values, dtype=torch.float32)

    bilinear = BilinearRegridder(GLOBAL, HRRR_CROP)(x.astype(np.float32))
    source_lat, source_lon = np.meshgrid(
        GLOBAL.latitude, GLOBAL.longitude, indexing="ij"
    )
    reference = LatLonInterpolation(source_lat, source_lon, lat, lon)(values)
    np.testing.assert_allclose(bilinear.values, reference.numpy(), atol=1e-6)

    nearest = NearestRegridder(GLOBAL, HRRR_CROP, max_dist_km=500.0)(x)
    reference = NearestNeighborInterpolator(
        GLOBAL.latitude, GLOBAL.longitude, lat, lon, max_dist_km=500.0
    )(values)
    np.testing.assert_allclose(nearest.values, reference.numpy(), atol=1e-6)


def test_list_and_resolve_regridders(monkeypatch):
    import earth2studio.grids as grids

    points = PointGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert list_regridders(REGION, "hrrr") == ("nearest", "bilinear")
    assert list_regridders(points, REGION) == ("nearest",)
    assert NearestRegridder.supports(points, REGION)
    assert not BilinearRegridder.supports(points, REGION)
    assert not BilinearRegridder.supports(LatLonGrid([0.0], [0.0, 1.0]), REGION)

    regridder = resolve_regridder(REGION, "hrrr", "bilinear", fill_value=0.0)
    assert isinstance(regridder, BilinearRegridder) and regridder.fill_value == 0
    assert isinstance(resolve_regridder(points, REGION), NearestRegridder)
    with pytest.raises(ValueError, match="available methods: \\['nearest'\\]"):
        resolve_regridder(points, REGION, "bilinear")
    with pytest.raises(ValueError, match="No 'conservative' regridder"):
        resolve_regridder(REGION, GLOBAL, "conservative")

    # Later registrations take precedence for the grids they support.
    monkeypatch.setattr(
        grids, "_REGRIDDERS", {k: list(v) for k, v in grids._REGRIDDERS.items()}
    )

    class _RegionOnly(NearestRegridder):
        @classmethod
        def supports(cls, source_grid, target_grid):
            return source_grid.fingerprint() == REGION.fingerprint()

    register_regridder("nearest", _RegionOnly)
    assert isinstance(resolve_regridder(REGION, GLOBAL), _RegionOnly)
    assert type(resolve_regridder(GLOBAL, REGION)) is NearestRegridder
    with pytest.raises(TypeError, match="supports"):
        register_regridder("nearest", object)


@pytest.mark.parametrize("curvilinear", [False, True])
def test_nearest_auto_distance_follows_source_resolution(curvilinear):
    if curvilinear:
        pytest.importorskip("scipy")
    source = REGION
    if curvilinear:
        source = CurvilinearGrid(
            *np.meshgrid(REGION.latitude, REGION.longitude, indexing="ij")
        )
    # Inside the region, just past its edge, and far beyond it.
    target = PointGrid(np.array([35.0, 45.0, 80.0]), np.array([255.0, 260.0, 260.0]))
    np.testing.assert_array_equal(
        NearestRegridder(source, target, max_dist_km="auto").coverage,
        [True, True, False],
    )
    assert NearestRegridder(source, target).coverage.all()


def test_to_rejects_unsupported_device():
    with pytest.raises(ValueError, match="CPU and CUDA"):
        NearestRegridder(REGION, REGION).to("meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
def test_cupy_fields_stay_on_device(cls):
    cp = pytest.importorskip("cupy")
    regridder = cls(REGION, GLOBAL)
    x = field(REGION, linear)
    expected = regridder(x)
    gpu = regridder(x.copy(data=cp.asarray(x.data)))
    assert isinstance(gpu.data, cp.ndarray)
    assert 0 in regridder._devices
    np.testing.assert_allclose(cp.asnumpy(gpu.data), expected.values, equal_nan=True)
    assert regridder.to("cuda:0") is regridder


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs",
)
@pytest.mark.parametrize("cls", [NearestRegridder, BilinearRegridder])
def test_cupy_fields_stay_on_their_device(cls):
    cp = pytest.importorskip("cupy")
    regridder = cls(REGION, GLOBAL)
    x = field(REGION, linear)
    expected = regridder(x)
    with cp.cuda.Device(1):
        data = cp.asarray(x.data)
    with cp.cuda.Device(0):
        y = regridder(x.copy(data=data))
    assert y.data.device.id == 1
    np.testing.assert_allclose(cp.asnumpy(y.data), expected.values, equal_nan=True)
