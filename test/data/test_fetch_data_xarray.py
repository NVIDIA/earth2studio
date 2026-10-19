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
from loguru import logger

from earth2studio.data import fetch_data
from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    BilinearRegridder,
    CurvilinearGrid,
    LatLonGrid,
    NearestRegridder,
    PointGrid,
    ProjectedGrid,
    resolve_grid,
)
from earth2studio.utils.coords import (
    E2S_DYNAMIC_DIMS,
    E2S_KIND,
    E2S_SCHEMA_VERSION,
    E2S_STATISTICS,
    coord_array,
    handshake_dataarray,
)
from earth2studio.utils.time_statistics import time_statistic_metadata

TIME = np.array([np.datetime64("2024-01-02T00", "s")])
GRID = LatLonGrid(np.array([2.0, 1.0, 0.0]), np.array([0.0, 1.0, 2.0]))


class AnalysisSource:
    time_step = np.timedelta64(1, "h")

    def __init__(self, grid=GRID, attrs=None):
        self.grid = grid
        self.attrs = attrs or {}
        self.requests = []

    def __call__(self, time, variable):
        self.requests.append((np.array(time), np.array(variable)))
        coords = self.grid.coords()
        lat, lon = xr.broadcast(coords["lat"], coords["lon"])
        hours = (time - TIME[0]) / np.timedelta64(1, "h")
        values = hours.reshape((-1, 1) + (1,) * len(self.grid.dims))
        values = values + (lat.values + 2 * lon.values)[None, None]
        return xr.DataArray(
            np.broadcast_to(values, (len(time), len(variable), *self.grid.shape)),
            dims=("time", "variable", *self.grid.dims),
            coords={"time": time, "variable": variable, **dict(coords)},
            attrs={"source": "offline", **self.attrs},
            name="weather",
        )


class ForecastSource(AnalysisSource):
    def __call__(self, time, lead_time, variable):
        self.requests.append((np.array(time), np.array(lead_time), np.array(variable)))
        arrays = []
        for lead in lead_time:
            array = super().__call__(time + lead, variable).assign_coords(time=time)
            arrays.append(array.expand_dims(lead_time=[lead], axis=1))
        return xr.concat(arrays, "lead_time")


def request(variables=("a",), grid=GRID, **kwargs):
    dims = resolve_grid(grid).dims if isinstance(grid, str) else grid.dims
    return coord_array(
        ("batch", "time", "lead_time", "variable", *dims),
        {"lead_time": [np.timedelta64(0, "h")], "variable": list(variables)},
        dynamic=("batch", "time"),
        grid=grid,
        **kwargs,
    )


def test_default_field_array():
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]))
    assert isinstance(array, xr.DataArray)
    assert array.dims == ("time", "lead_time", "variable", "lat", "lon")
    assert array.name == "weather"
    assert array.attrs["source"] == "offline"
    assert array.attrs == {"source": "offline"}
    np.testing.assert_allclose(array[0, 0, 0], [[2, 4, 6], [1, 3, 5], [0, 2, 4]])


def test_source_variable_order_is_respected():
    class ReorderedSource(AnalysisSource):
        def __call__(self, time, variable):
            array = super().__call__(time, variable)
            array = array + xr.DataArray(
                np.arange(len(variable)), dims="variable", coords={"variable": variable}
            )
            return array.isel(variable=slice(None, None, -1))

    array = fetch_data(ReorderedSource(), TIME, np.array(["a", "b"]))
    np.testing.assert_allclose(array.sel(variable="b") - array.sel(variable="a"), 1)


@pytest.mark.parametrize(
    "grid",
    [
        CurvilinearGrid(
            np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([[0.0, 1.0], [0.0, 1.0]])
        ),
        ProjectedGrid(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), "EPSG:4326"
        ),
        PointGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    ],
)
def test_native_grid_preserves_auxiliaries(grid):
    signature = request(grid=grid)
    source = AnalysisSource(grid, attrs=signature.attrs)
    array = fetch_data(source, TIME, np.array(["a"]))
    handshake_dataarray(array, signature)
    assert array.attrs["source"] == "offline"


def test_projected_source_bilinear_regridding():
    grid = ProjectedGrid(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), "EPSG:4326"
    )
    target = LatLonGrid(np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    array = fetch_data(
        AnalysisSource(grid, attrs=request(grid=grid).attrs),
        TIME,
        np.array(["a"]),
        regridder=BilinearRegridder(grid, target),
    )
    assert array.dims == ("time", "lead_time", "variable", "lat", "lon")
    handshake_dataarray(array, request(grid=target))
    # The source field is linear in latitude and longitude, so bilinear is exact.
    np.testing.assert_allclose(array[0, 0, 0], [[1.5, 3.5], [2.5, 4.5]])


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
def test_signature_statistics_are_computed(source_cls):
    signature = request(("a:sum:3h", "b"))
    array = fetch_data(
        source_cls(attrs=request().attrs),
        TIME,
        np.array(["a:sum:3h", "b"]),
    )
    np.testing.assert_allclose(
        array.sel(variable="a:sum:3h")[0, 0], [[0, 6, 12], [-3, 3, 9], [-6, 0, 6]]
    )
    np.testing.assert_allclose(
        array.sel(variable="b")[0, 0], [[2, 4, 6], [1, 3, 5], [0, 2, 4]]
    )
    handshake_dataarray(array, signature)
    assert not {E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS} & array.attrs.keys()


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
def test_qualified_statistics_preserve_order_and_metadata(source_cls):
    labels = np.array(["a:mean:3h", "a", "a:max:0h:+2h"])
    array = fetch_data(source_cls(), TIME, labels)
    np.testing.assert_array_equal(array.coords["variable"], labels)
    np.testing.assert_allclose(
        array.sel(variable=labels[0])[0, 0], [[0, 2, 4], [-1, 1, 3], [-2, 0, 2]]
    )
    np.testing.assert_allclose(
        array.sel(variable=labels[2])[0, 0], [[3, 5, 7], [2, 4, 6], [1, 3, 5]]
    )
    assert array.attrs[E2S_STATISTICS] == {
        labels[0]: time_statistic_metadata("mean:3h"),
        labels[2]: time_statistic_metadata("max:0h:+2h"),
    }


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
def test_grouped_time_windows(source_cls):
    times = TIME + np.array([0, 72], dtype="timedelta64[h]")
    leads = np.array([-24, 0], dtype="timedelta64[h]")
    labels = np.array(["a:mean:0h:24h", "b:mean:1h:25h", "c:mean:0h:24h"])
    target_grid = LatLonGrid(np.array([0.0]), np.array([0.0]))
    source = source_cls(target_grid, attrs=request(grid=target_grid).attrs)
    signature = coord_array(
        ("time", "lead_time", "variable", "lat", "lon"),
        {"lead_time": leads, "variable": labels},
        dynamic=("time",),
        grid=target_grid,
    )
    array = fetch_data(source, times, labels, leads)
    handshake_dataarray(array, signature)
    np.testing.assert_array_equal(array.time, times)
    np.testing.assert_array_equal(array.lead_time, leads)
    np.testing.assert_array_equal(array.coords["variable"], labels)
    offsets = np.array([11.5, 12.5, 11.5])
    expected = (
        np.array([0, 72])[:, None, None]
        + np.array([-24, 0])[None, :, None]
        + offsets[None, None, :]
    )
    np.testing.assert_allclose(array.values[..., 0, 0], expected)
    assert array.attrs[E2S_STATISTICS] == {
        label: time_statistic_metadata(label.partition(":")[2]) for label in labels
    }

    # ForecastSource also records its internal per-lead analysis calls.
    calls = [
        call
        for call in source.requests
        if len(call) == (3 if source_cls is ForecastSource else 2)
    ]
    assert len(calls) == 2
    for call, start in zip(calls, [0, 1], strict=True):
        expected_variables = ["a", "c"] if start == 0 else ["b"]
        np.testing.assert_array_equal(call[-1], expected_variables)
        hours = np.arange(-24 + start, 24 + start).astype("timedelta64[h]")
        if source_cls is ForecastSource:
            np.testing.assert_array_equal(call[0], times)
            np.testing.assert_array_equal(call[1], hours)
        else:
            expected_times = np.unique(times[:, None] + hours[None, :])
            np.testing.assert_array_equal(call[0], expected_times)


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
@pytest.mark.parametrize("duplicate", [False, True])
def test_daily_window_rejects_incomplete_or_duplicate_samples(source_cls, duplicate):
    def corrupt(array, dimension):
        indexes = np.arange(array.sizes[dimension] - 1)
        if duplicate:
            indexes = np.append(np.arange(array.sizes[dimension]), 0)
        return array.isel({dimension: indexes})

    class IncompleteAnalysis(AnalysisSource):
        def __call__(self, time, variable):
            return corrupt(super().__call__(time, variable), "time")

    class IncompleteForecast(ForecastSource):
        def __call__(self, time, lead_time, variable):
            return corrupt(super().__call__(time, lead_time, variable), "lead_time")

    source = (
        IncompleteForecast() if source_cls is ForecastSource else IncompleteAnalysis()
    )
    with pytest.raises(ValueError, match="[Mm]issing|[Dd]uplicate"):
        fetch_data(source, TIME, np.array(["tp:mean:1h:25h"]))


def test_duplicate_normalized_quantities_fail_before_fetch():
    source = AnalysisSource()
    with pytest.raises(ValueError, match="duplicate"):
        fetch_data(source, TIME, np.array(["a:mean:3h", "a:mean:180m"]))
    assert not source.requests


@pytest.mark.parametrize(
    "regridder, expected",
    [
        (BilinearRegridder, [[1.0, 3.0], [0.0, 2.0]]),
        # Midpoints round to the upper neighbor along each axis.
        (NearestRegridder, [[2.5, 4.5], [1.5, 3.5]]),
    ],
)
def test_regridding_follows_temporal_reduction(regridder, expected):
    target = LatLonGrid(np.array([1.5, 0.5]), np.array([0.5, 1.5]))
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a:mean:2h"]),
        regridder=regridder(GRID, target),
    )
    np.testing.assert_array_equal(array.lat, target.latitude)
    np.testing.assert_array_equal(array.lon, target.longitude)
    np.testing.assert_allclose(array[0, 0, 0], expected)
    assert array.attrs[E2S_STATISTICS] == {
        "a:mean:2h": time_statistic_metadata("mean:2h")
    }
    assert array.attrs["source"] == "offline"


@pytest.mark.parametrize("named", [False, True])
def test_regridded_metadata_follows_target_grid(named):
    target = resolve_grid("latlon-1deg")
    regridder = NearestRegridder(GRID, "latlon-1deg" if named else target)
    array = fetch_data(
        AnalysisSource(attrs={E2S_CRS: "EPSG:4326", "crs": "EPSG:4326"}),
        TIME,
        np.array(["a"]),
        regridder=regridder,
    )
    assert array.sizes["lat"] == 181 and array.sizes["lon"] == 360
    assert array.attrs["type"] == "LatLonGrid"
    assert array.attrs["dims"] == ["lat", "lon"]
    assert array.attrs["shape"] == [181, 360]
    assert (E2S_GRID_ID in array.attrs) is named
    handshake_dataarray(array, request(grid="latlon-1deg" if named else target))


@pytest.mark.parametrize(
    "target",
    [
        PointGrid(np.array([0.5, 1.5]), np.array([1.0, 0.5])),
        CurvilinearGrid(np.array([[0.5, 1.5]]), np.array([[1.0, 0.5]])),
        ProjectedGrid(np.array([0.5, 1.5]), np.array([0.5, 1.5]), "EPSG:4326"),
    ],
)
def test_bilinear_onto_other_target_topologies(target):
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a"]),
        regridder=BilinearRegridder(GRID, target),
    )
    handshake_dataarray(array, request(grid=target))
    lat, lon = xr.broadcast(target.coords()["lat"], target.coords()["lon"])
    np.testing.assert_allclose(array[0, 0, 0], lat + 2 * lon)


def _logged(level="INFO"):
    messages = []
    handler = logger.add(lambda message: messages.append(str(message)), level=level)
    return messages, handler


@pytest.mark.parametrize(
    "target",
    [GRID, LatLonGrid(np.array([2, 1, 0]), np.array([0, 1, 2]))],
)
def test_target_grid_matching_source_passes_through(target):
    messages, handler = _logged()
    try:
        array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), target_grid=target)
    finally:
        logger.remove(handler)
    xr.testing.assert_identical(
        array, fetch_data(AnalysisSource(), TIME, np.array(["a"]))
    )
    assert not any("Regridding" in message for message in messages)


def test_target_grid_crop_is_selected():
    target = LatLonGrid(np.array([1.0, 0.0]), np.array([1.0, 2.0]))
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), target_grid=target)
    handshake_dataarray(array, request(grid=target))
    np.testing.assert_allclose(array[0, 0, 0], [[3, 5], [2, 4]])
    assert array.attrs["shape"] == [2, 2] and array.attrs["source"] == "offline"


def test_projected_target_grid_crop_is_selected():
    grid = ProjectedGrid(
        np.array([0.0, 1.0, 2.0]) * 1000, np.array([0.0, 1.0, 2.0]) * 1000, "EPSG:3857"
    )
    crop = ProjectedGrid(grid.y[1:], grid.x[:2], grid.crs)
    source = AnalysisSource(grid, attrs=request(grid=grid).attrs)
    array = fetch_data(source, TIME, np.array(["a"]), target_grid=crop)
    handshake_dataarray(array, request(grid=crop))
    full = fetch_data(source, TIME, np.array(["a"]))
    np.testing.assert_allclose(array, full.isel(y=slice(1, None), x=slice(None, 2)))


def test_target_grid_regrids_by_nearest_and_logs():
    target = LatLonGrid(np.array([1.6, 0.4]), np.array([0.4, 1.6]))
    messages, handler = _logged()
    try:
        array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), target_grid=target)
    finally:
        logger.remove(handler)
    handshake_dataarray(array, request(grid=target))
    # Field values are lat + 2 * lon at the nearest source points.
    np.testing.assert_allclose(array[0, 0, 0], [[2, 6], [0, 4]])
    assert any("by nearest" in message for message in messages)


def test_target_grid_beyond_source_raises():
    target = LatLonGrid(np.array([20.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="extends beyond"):
        fetch_data(AnalysisSource(), TIME, np.array(["a"]), target_grid=target)
    # Naming a method opts in: points without coverage are filled.
    array = fetch_data(
        AnalysisSource(), TIME, np.array(["a"]), target_grid=target, regridder="nearest"
    )
    assert np.isnan(array[0, 0, 0, 0, 0]) and array[0, 0, 0, 1, 0] == 3


def test_target_grid_with_method_name():
    target = LatLonGrid(np.array([1.5, 0.5]), np.array([0.5, 1.5]))
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a"]),
        target_grid=target,
        regridder="bilinear",
    )
    np.testing.assert_allclose(array[0, 0, 0], [[2.5, 4.5], [1.5, 3.5]])
    with pytest.raises(ValueError, match="needs a target_grid"):
        fetch_data(AnalysisSource(), TIME, np.array(["a"]), regridder="bilinear")
    with pytest.raises(ValueError, match="No 'bilinear' regridder"):
        fetch_data(
            AnalysisSource(PointGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))),
            TIME,
            np.array(["a"]),
            target_grid=target,
            regridder="bilinear",
        )


def test_regridder_must_produce_target_grid():
    target = LatLonGrid(np.array([1.5, 0.5]), np.array([0.5, 1.5]))
    regridder = BilinearRegridder(GRID, target)
    array = fetch_data(
        AnalysisSource(), TIME, np.array(["a"]), target_grid=target, regridder=regridder
    )
    np.testing.assert_allclose(array[0, 0, 0], [[2.5, 4.5], [1.5, 3.5]])
    with pytest.raises(ValueError, match="does not match target_grid"):
        fetch_data(
            AnalysisSource(),
            TIME,
            np.array(["a"]),
            target_grid=GRID,
            regridder=regridder,
        )


def test_target_grid_needs_an_inferable_source_grid():
    class Unlabelled(AnalysisSource):
        def __call__(self, time, variable):
            array = super().__call__(time, variable)
            return array.drop_vars(["lat", "lon"]).rename(lat="a", lon="b")

    with pytest.raises(ValueError, match="Cannot determine the grid"):
        fetch_data(Unlabelled(), TIME, np.array(["a"]), target_grid=GRID)
    # Without a target the field passes through on its own coordinates.
    assert fetch_data(Unlabelled(), TIME, np.array(["a"])).dims[-2:] == ("a", "b")


def test_source_metadata_mismatch_is_not_relabelled():
    source = AnalysisSource(attrs={E2S_CRS: "EPSG:3857"})
    array = fetch_data(source, TIME, np.array(["a"]))
    assert array.attrs[E2S_CRS] == "EPSG:3857"


def test_already_aggregated_source_not_reduced_twice():
    source = AnalysisSource(
        attrs={E2S_STATISTICS: {"a": time_statistic_metadata("sum:6h")}}
    )
    with pytest.raises(ValueError, match="statistic"):
        fetch_data(source, TIME, np.array(["a:sum:3h"]))


@pytest.mark.parametrize(
    "lead", [[0], [np.datetime64("2024-01-01")], [np.timedelta64("NaT")]]
)
def test_invalid_lead_time(lead):
    with pytest.raises((TypeError, ValueError), match="lead_time"):
        fetch_data(AnalysisSource(), TIME, np.array(["a"]), np.array(lead))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_field_array():
    cp = pytest.importorskip("cupy")
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a:mean:2h"]),
        device="cuda:0",
    )
    assert isinstance(array.data, cp.ndarray)
    assert array.data.device.id == 0
    np.testing.assert_allclose(
        cp.asnumpy(array.data[0, 0, 0]),
        [[0.5, 2.5, 4.5], [-0.5, 1.5, 3.5], [-1.5, 0.5, 2.5]],
    )
