# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import fetch_data
from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    CurvilinearGrid,
    LatLonGrid,
    PointGrid,
    ProjectedGrid,
)
from earth2studio.models.px.fuxi_s2s import DAILY_VARIABLES, VARIABLES, FuXiS2S
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
    return coord_array(
        ("batch", "time", "lead_time", "variable", *grid.dims),
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
    array = fetch_data(source, TIME, np.array(["a"]), metadata=signature)
    handshake_dataarray(array, signature)
    assert array.attrs["source"] == "offline"


def test_renamed_target_spatial_dimensions():
    signature = coord_array(
        ("time", "lead_time", "variable", "height", "width"),
        {"lead_time": [np.timedelta64(0, "h")], "variable": ["a"]},
        dynamic=("time",),
        grid=ProjectedGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]), "EPSG:4326"),
        grid_dims={"y": "height", "x": "width"},
    )
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), metadata=signature)
    assert array.dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(array.lat, GRID.coords()["lat"])


def test_projected_source_regridding_is_passthrough():
    grid = ProjectedGrid(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), "EPSG:4326"
    )
    target = LatLonGrid(np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    array = fetch_data(
        AnalysisSource(grid, attrs=request(grid=grid).attrs),
        TIME,
        np.array(["a"]),
        interp_to=target,
        interp_method="linear",
    )
    assert array.dims == ("time", "lead_time", "variable", "y", "x")
    np.testing.assert_allclose(array[0, 0, 0], [[0, 2, 4], [1, 3, 5], [2, 4, 6]])


def test_target_grid_validation_is_deferred():
    signature = request(grid=GRID)
    signature.attrs[E2S_GRID_ID] = "fetch-test-full"
    signature = signature.assign_coords(lat=[20.0, 10.0, 0.0])
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), interp_to=signature)
    np.testing.assert_array_equal(array.lat, GRID.coords()["lat"])


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
def test_signature_statistics_are_computed(source_cls):
    signature = request(("a:sum:3h", "b"))
    array = fetch_data(
        source_cls(attrs=request().attrs),
        TIME,
        np.array(["a:sum:3h", "b"]),
        metadata=signature,
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
def test_fuxi_s2s_daily_windows(source_cls):
    times = TIME + np.array([0, 72], dtype="timedelta64[h]")
    leads = np.array([-24, 0], dtype="timedelta64[h]")
    labels = np.array(DAILY_VARIABLES)
    target_grid = LatLonGrid(np.array([0.0]), np.array([0.0]))
    source = source_cls(target_grid, attrs=request(grid=target_grid).attrs)
    signature = coord_array(
        ("time", "lead_time", "variable", "lat", "lon"),
        {"lead_time": leads, "variable": labels},
        dynamic=("time",),
        grid=target_grid,
    )
    array = fetch_data(source, times, labels, leads, metadata=signature)
    handshake_dataarray(array, signature)
    np.testing.assert_array_equal(array.time, times)
    np.testing.assert_array_equal(array.lead_time, leads)
    np.testing.assert_array_equal(array.coords["variable"], labels)
    offsets = np.array([12.5 if v in {"tp", "ttr"} else 11.5 for v in VARIABLES])
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
        expected_variables = [
            v for v in VARIABLES if (v in {"tp", "ttr"}) == bool(start)
        ]
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


@pytest.mark.parametrize("source_cls", [AnalysisSource, ForecastSource])
def test_fetch_fuxi_model_signature_handshake(source_cls, monkeypatch):
    model = FuXiS2S.__new__(FuXiS2S)
    torch.nn.Module.__init__(model)
    model._time_step = np.timedelta64(1, "D")
    # Keep the real model's temporal declarations and crop only its spatial grid.
    signature = model.input_coords().isel(lat=slice(60, 61), lon=slice(0, 1))
    monkeypatch.setattr(model, "input_coords", lambda: signature)
    source_grid = LatLonGrid(signature.lat.values, signature.lon.values)
    array = fetch_data(
        source_cls(source_grid, attrs=request(grid=source_grid).attrs),
        TIME,
        signature.coords["variable"].values,
        signature.lead_time.values,
        metadata=signature,
    )
    handshake_dataarray(array, signature)
    output = model.output_coords(array)
    np.testing.assert_array_equal(output.lead_time, [np.timedelta64(1, "D")])
    np.testing.assert_array_equal(output.time, TIME)
    np.testing.assert_array_equal(output.coords["variable"], DAILY_VARIABLES)
    assert output.attrs[E2S_STATISTICS] == signature.attrs[E2S_STATISTICS]
    np.testing.assert_allclose(
        array.sel(variable="t2m:mean:0h:24h").values.ravel(), [-12.5, 11.5]
    )
    np.testing.assert_allclose(
        array.sel(variable="tp:mean:1h:25h").values.ravel(), [-11.5, 12.5]
    )


def test_duplicate_normalized_quantities_fail_before_fetch():
    source = AnalysisSource()
    with pytest.raises(ValueError, match="duplicate"):
        fetch_data(source, TIME, np.array(["a:mean:3h", "a:mean:180m"]))
    assert not source.requests


def test_conflicting_statistics_fail_before_fetch():
    source = AnalysisSource()
    signature = request(("a:mean:3h",))
    signature.attrs[E2S_STATISTICS] = {"a:mean:3h": time_statistic_metadata("sum:3h")}
    with pytest.raises(ValueError, match="[Cc]onflict"):
        fetch_data(source, TIME, np.array(["a:mean:3h"]), metadata=signature)
    assert not source.requests


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_grid_signature_regridding_is_passthrough(method):
    target = LatLonGrid(np.array([1.5, 0.5]), np.array([0.5, 1.5]))
    signature = request(("a:mean:2h",), grid=target)
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a:mean:2h"]),
        metadata=signature,
        interp_method=method,
    )
    np.testing.assert_array_equal(array.lat, GRID.coords()["lat"])
    np.testing.assert_array_equal(array.lon, GRID.coords()["lon"])
    np.testing.assert_allclose(
        array[0, 0, 0], [[0.5, 2.5, 4.5], [-0.5, 1.5, 3.5], [-1.5, 0.5, 2.5]]
    )


@pytest.mark.parametrize("kind", ["definition", "name", "signature"])
def test_grid_target_is_passthrough(kind):
    target = LatLonGrid(np.array([0.0, 2.0]), np.array([2.0, 0.0]))
    interp_to = {
        "definition": target,
        "name": "fetch-test-subset",
        "signature": request(grid=target),
    }[kind]
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), interp_to=interp_to)
    np.testing.assert_allclose(array[0, 0, 0], [[2, 4, 6], [1, 3, 5], [0, 2, 4]])
    assert E2S_GRID_ID not in array.attrs


def test_geographic_bounds_are_deferred():
    array = fetch_data(AnalysisSource(), TIME, np.array(["a"]), bounds=(0.5, 0.5, 2, 2))
    np.testing.assert_array_equal(array.lat, [2, 1, 0])
    np.testing.assert_array_equal(array.lon, [0, 1, 2])


@pytest.mark.parametrize(
    "target",
    [
        PointGrid(np.array([0.5, 1.5]), np.array([1.0, 0.5])),
        CurvilinearGrid(np.array([[0.5, 1.5]]), np.array([[1.0, 0.5]])),
        ProjectedGrid(np.array([0.5, 1.5]), np.array([0.5, 1.5]), "EPSG:4326"),
    ],
)
def test_other_target_topologies_are_passthrough(target):
    signature = request(grid=target)
    array = fetch_data(
        AnalysisSource(),
        TIME,
        np.array(["a"]),
        metadata=signature,
        interp_method="linear",
    )
    lat, lon = xr.broadcast(GRID.coords()["lat"], GRID.coords()["lon"])
    np.testing.assert_allclose(array[0, 0, 0], lat + 2 * lon)


def test_source_metadata_mismatch_is_not_relabelled():
    source = AnalysisSource(attrs={E2S_CRS: "EPSG:3857"})
    array = fetch_data(source, TIME, np.array(["a"]), metadata=request())
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
        metadata=request(("a:mean:2h",)),
    )
    assert isinstance(array.data, cp.ndarray)
    assert array.data.device.id == 0
    np.testing.assert_allclose(
        cp.asnumpy(array.data[0, 0, 0]),
        [[0.5, 2.5, 4.5], [-0.5, 1.5, 3.5], [-1.5, 0.5, 2.5]],
    )
