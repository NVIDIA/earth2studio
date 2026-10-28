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

import asyncio
import threading

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import (
    CompositeForecastSource,
    CompositeSource,
    DataSource,
    ForecastSource,
    RegriddedSource,
    fetch_data,
)
from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.utils.coords import E2S_STATISTICS
from earth2studio.utils.time_statistics import time_statistic_metadata

GRID = LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 10.0, 20.0]))
FINE = LatLonGrid(np.array([10.0, 5.0, 0.0]), np.array([0.0, 5.0, 10.0, 15.0, 20.0]))
TIME = np.array([np.datetime64("2024-01-01T12")])


class _Source:
    """Fills each variable with its offset; records requested variables."""

    def __init__(self, offsets, grid=GRID, wrap=np.asarray):
        self.offsets, self.grid, self.wrap = offsets, grid, wrap
        self.requests, self.times = [], []

    def _field(self, time, variable, lead_time=None):
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.requests.append(list(variable))
        self.times.append(list(time))
        dims, coords = ["time", "variable"], {"time": time, "variable": variable}
        if lead_time is not None:
            dims.insert(1, "lead_time")
            coords["lead_time"] = np.atleast_1d(lead_time)
        shape = [len(coords[dim]) for dim in dims] + list(self.grid.shape)
        values = np.zeros(shape) + np.array(
            [self.offsets[str(name)] for name in variable]
        ).reshape([-1 if dim == "variable" else 1 for dim in dims] + [1, 1])
        return xr.DataArray(
            self.wrap(values),
            dims=(*dims, *self.grid.dims),
            coords={**coords, **self.grid.coords()},
        )

    def __call__(self, time, variable):
        return self._field(time, variable)


class _AsyncSource(_Source):
    def __init__(self, offsets, barrier, **kwargs):
        super().__init__(offsets, **kwargs)
        self.barrier = barrier

    async def fetch(self, time, variable):
        # Both components must be in flight together to pass the barrier.
        await asyncio.wait_for(self.barrier.wait(), timeout=5)
        return self._field(time, variable)


class _ForecastSource(_Source):
    def __call__(self, time, lead_time, variable):
        return self._field(time, variable, lead_time)


def _composite(**kwargs):
    radar = _Source({"refc": 1.0, "refd": 2.0}, **kwargs)
    satellite = _Source({"abi01c": 10.0}, **kwargs)
    return (
        CompositeSource(
            {"radar": radar, "satellite": satellite},
            {"radar": ["refc", "refd"], "satellite": ["abi01c"]},
        ),
        radar,
        satellite,
    )


def test_routes_variables_in_request_order():
    source, radar, satellite = _composite()
    assert isinstance(source, DataSource)
    array = source(TIME, np.array(["refd", "abi01c", "refc"]))
    np.testing.assert_array_equal(array["variable"], ["refd", "abi01c", "refc"])
    np.testing.assert_allclose(array.isel(time=0, lat=0, lon=0), [2.0, 10.0, 1.0])
    assert radar.requests == [["refd", "refc"]] and satellite.requests == [["abi01c"]]

    # Only the components providing requested variables are called.
    source(TIME, "refc")
    assert len(satellite.requests) == 1
    assert dict(source.variables) == {
        "radar": ("refc", "refd"),
        "satellite": ("abi01c",),
    }
    assert set(source.sources) == {"radar", "satellite"}


def test_construction_errors():
    with pytest.raises(ValueError, match="unknown"):
        CompositeSource({"a": _Source({})}, {"b": ["x"]})
    with pytest.raises(ValueError, match="both"):
        CompositeSource(
            {"a": _Source({}), "b": _Source({})}, {"a": ["x"], "b": ["x", "y"]}
        )
    with pytest.raises(ValueError, match="qualified"):
        CompositeSource({"a": _Source({})}, {"a": ["x:mean:24h"]})
    with pytest.raises(TypeError, match="CompositeForecastSource"):
        CompositeSource(
            {"a": _Source({}), "b": _ForecastSource({})}, {"a": ["x"], "b": ["y"]}
        )


def test_qualified_labels_route_by_base_name():
    radar = _Source({"refc": 1.0, "refc:max:1h": 3.0})
    satellite = _Source({"abi01c:mean:24h": 10.0})
    source = CompositeSource(
        {"radar": radar, "satellite": satellite},
        {"radar": ["refc"], "satellite": ["abi01c"]},
    )
    variables = ["abi01c:mean:24h", "refc", "refc:max:1h"]
    array = source(TIME, np.array(variables))
    np.testing.assert_array_equal(array["variable"], variables)
    np.testing.assert_allclose(array.isel(time=0, lat=0, lon=0), [10.0, 1.0, 3.0])
    # Labels reach the providing component unchanged.
    assert radar.requests == [["refc", "refc:max:1h"]]
    assert satellite.requests == [["abi01c:mean:24h"]]
    with pytest.raises(KeyError, match="not provided"):
        source(TIME, ["t2m:mean:24h"])

    # Only statistics supported natively by every component are advertised, so
    # fetch_data forwards those labels and reduces anything else itself.
    from earth2studio.data.utils import _source_statistics

    assert not hasattr(source, "time_statistics")
    assert _source_statistics(source) == ()
    radar.lexicon = type(
        "RadarLexicon", (), {"TIME_STATISTICS": ("mean", "max", "sum")}
    )
    assert _source_statistics(source) == ()
    satellite.lexicon = type(
        "SatelliteLexicon", (), {"TIME_STATISTICS": ("mean", "sum")}
    )
    assert _source_statistics(source) == ("mean", "sum")
    array = fetch_data(source, TIME, np.array(["abi01c:mean:24h", "refc"]))
    np.testing.assert_allclose(array[0, 0, :, 0, 0], [10.0, 1.0])
    assert satellite.requests[-1] == ["abi01c:mean:24h"]


def test_unknown_variable_raises():
    source, _, _ = _composite()
    with pytest.raises(KeyError, match="not provided"):
        source(TIME, ["refc", "t2m"])


def test_mismatched_grids_raise():
    radar = _Source({"refc": 1.0})
    satellite = _Source({"abi01c": 10.0}, grid=FINE)
    source = CompositeSource(
        {"radar": radar, "satellite": satellite},
        {"radar": ["refc"], "satellite": ["abi01c"]},
    )
    with pytest.raises(ValueError, match="common grid"):
        source(TIME, ["refc", "abi01c"])

    # Regridding the fine component onto the shared grid resolves the mismatch.
    source = CompositeSource(
        {
            "radar": radar,
            "satellite": RegriddedSource(satellite, BilinearRegridder(FINE, GRID)),
        },
        {"radar": ["refc"], "satellite": ["abi01c"]},
    )
    np.testing.assert_allclose(source(TIME, ["refc", "abi01c"])[0, :, 0, 0], [1, 10])


def test_forecast_components():
    source = CompositeForecastSource(
        {"a": _ForecastSource({"x": 1.0}), "b": _ForecastSource({"y": 2.0})},
        {"a": ["x"], "b": ["y"]},
    )
    assert isinstance(source, ForecastSource)
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    array = source(TIME, lead_time, np.array(["y", "x"]))
    assert array.dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_allclose(array[0, 1, :, 0, 0], [2.0, 1.0])
    array = fetch_data(source, TIME, np.array(["x", "y"]), lead_time)
    np.testing.assert_array_equal(array["lead_time"], lead_time)


def test_mixed_components_present_forecast_interface():
    analysis = _Source({"x": 1.0})
    forecast = _ForecastSource({"y": 2.0})
    source = CompositeForecastSource(
        {"a": analysis, "b": forecast}, {"a": ["x"], "b": ["y"]}
    )
    assert isinstance(source, ForecastSource)
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    array = source(TIME, lead_time, np.array(["x", "y"]))
    assert array.dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(array["time"], TIME)
    np.testing.assert_array_equal(array["lead_time"], lead_time)
    np.testing.assert_allclose(array[0, :, :, 0, 0], [[1.0, 2.0], [1.0, 2.0]])
    # The analysis component is fetched at each valid time.
    assert analysis.times == [list(TIME), list(TIME + lead_time[1])]
    array = asyncio.run(source.fetch(TIME, lead_time, np.array(["y", "x"])))
    np.testing.assert_allclose(array[0, 1, :, 0, 0], [2.0, 1.0])
    array = fetch_data(source, TIME, np.array(["x", "y"]), lead_time)
    np.testing.assert_array_equal(array["lead_time"], lead_time)


def test_fetch_awaits_components_concurrently():
    async def run():
        barrier = asyncio.Barrier(2)
        radar = _AsyncSource({"refc": 1.0}, barrier)
        satellite = _AsyncSource({"abi01c": 10.0}, barrier)
        source = CompositeSource(
            {"radar": radar, "satellite": satellite},
            {"radar": ["refc"], "satellite": ["abi01c"]},
        )
        return await source.fetch(TIME, variable=["abi01c", "refc"])

    array = asyncio.run(run())
    np.testing.assert_allclose(array[0, :, 0, 0], [10.0, 1.0])


def test_fetch_runs_sync_components_in_threads():
    threads = set()

    class _Recording(_Source):
        def __call__(self, time, variable):
            threads.add(threading.get_ident())
            return super().__call__(time, variable)

    source = CompositeSource(
        {"a": _Recording({"x": 1.0}), "b": _Recording({"y": 2.0})},
        {"a": ["x"], "b": ["y"]},
    )
    array = asyncio.run(source.fetch(TIME, ["x", "y"]))
    np.testing.assert_allclose(array[0, :, 0, 0], [1.0, 2.0])
    assert threading.get_ident() not in threads


def test_components_share_one_device():
    source, _, _ = _composite()
    assert source.device == torch.device("cpu")

    on_gpu = _Source({"y": 2.0})
    on_gpu.device = "cuda:1"
    with pytest.raises(ValueError, match="one device"):
        CompositeSource(
            {"a": _Source({"x": 1.0}), "b": on_gpu}, {"a": ["x"], "b": ["y"]}
        )
    other = _Source({"x": 1.0})
    other.device = torch.device("cuda", 1)
    shared = CompositeSource({"a": other, "b": on_gpu}, {"a": ["x"], "b": ["y"]})
    assert shared.device == torch.device("cuda", 1)

    # An unindexed CUDA device is the current device, index 0 without CUDA here.
    current = _Source({"x": 1.0})
    current.device = "cuda"
    indexed = _Source({"y": 2.0})
    indexed.device = "cuda:0"
    if not torch.cuda.is_available() or torch.cuda.current_device() == 0:
        shared = CompositeSource({"a": current, "b": indexed}, {"a": ["x"], "b": ["y"]})
        assert shared.device == torch.device("cuda", 0)


def test_statistics_metadata_merged_from_every_component():
    def _aggregated(source, label):
        def call(time, variable):
            array = _Source.__call__(source, time, variable)
            array.attrs[E2S_STATISTICS] = {label: time_statistic_metadata("sum:6h")}
            return array

        return call

    radar, satellite = _Source({"refc": 1.0}), _Source({"tp": 2.0})
    radar.__call__ = _aggregated(radar, "refc")
    satellite.__call__ = _aggregated(satellite, "tp")

    class _Wrapped:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, time, variable):
            return self.inner.__call__(time, variable)

    source = CompositeSource(
        {"radar": _Wrapped(radar), "satellite": _Wrapped(satellite)},
        {"radar": ["refc"], "satellite": ["tp"]},
    )
    array = source(TIME, ["refc", "tp"])
    assert set(array.attrs[E2S_STATISTICS]) == {"refc", "tp"}
    assert set(source(TIME, ["tp"]).attrs[E2S_STATISTICS]) == {"tp"}
    # Already aggregated variables from any component cannot be reduced again.
    with pytest.raises(ValueError, match="statistic"):
        fetch_data(
            source,
            TIME,
            np.array(["refc:sum:12h", "tp:sum:12h"]),
            delta_t=np.timedelta64(6, "h"),
        )


def test_temporal_statistics_through_regridded_components():
    fine = _Source({"abi01c": 10.0}, grid=FINE)
    source = CompositeSource(
        {
            "radar": _Source({"refc": 1.0}),
            "satellite": RegriddedSource(fine, BilinearRegridder(FINE, GRID)),
        },
        {"radar": ["refc"], "satellite": ["abi01c"]},
    )
    array = fetch_data(
        source,
        TIME,
        np.array(["refc:mean:12h", "abi01c"]),
        delta_t=np.timedelta64(6, "h"),
    )
    np.testing.assert_allclose(array[0, 0, :, 0, 0], [1.0, 10.0])
    assert array.sizes["lon"] == GRID.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_device_components_join_on_device():
    cp = pytest.importorskip("cupy")
    a = _Source({"x": 1.0}, wrap=cp.asarray)
    b = _Source({"y": 2.0}, wrap=cp.asarray)
    a.device = b.device = torch.device("cuda", 0)
    source = CompositeSource({"a": a, "b": b}, {"a": ["x"], "b": ["y"]})
    array = source(TIME, ["y", "x"])
    assert isinstance(array.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(array.data[0, :, 0, 0]), [2.0, 1.0])
