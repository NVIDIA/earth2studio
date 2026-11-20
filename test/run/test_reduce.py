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

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import KVBackend
from earth2studio.run import LeadWindow, Pipeline, WorkItem, work_items
from earth2studio.run.work import ProgressStore
from earth2studio.utils.coords import E2S_STATISTICS, coord_array
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")
T1 = np.datetime64("2026-01-02T00")
STEP = np.timedelta64(6, "h")
LAT = np.array([10.0, 0.0])
LON = np.array([0.0, 10.0, 20.0])


def _h(*hours: int) -> np.ndarray:
    return np.array(hours, dtype="timedelta64[h]").astype("timedelta64[ns]")


class _Steps:
    """Yield the value ``k`` at lead ``6k h`` on every stream, plus 10 per member ID.

    ``held`` maps a lead in hours to the variables published there, so an
    output can lack some.
    """

    supports_member_batching = True
    seed = 0

    def __init__(
        self,
        streams: Sequence[str] = ("forecast",),
        held: Mapping[int, list[str]] | None = None,
    ) -> None:
        self.streams = tuple(streams)
        self.held = dict(held or {})
        self.ran: list[np.datetime64] = []

    def to(self, device: torch.device) -> _Steps:
        return self

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        leads = np.arange(0, final_lead_time + STEP, STEP).astype("timedelta64[ns]")
        signature = coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {"lead_time": leads, "variable": ["u10m", "v10m"], "lat": LAT, "lon": LON},
            dynamic=("batch",),
        )
        return {stream: signature for stream in self.streams}

    def data_requests(self, item: WorkItem) -> None:
        return None

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        self.ran.append(item.time)
        members = list(item.member_ids)
        for k in range(int(item.final_lead_time // STEP) + 1):
            variables = self.held.get(6 * k, ["u10m", "v10m"])
            shape = (len(variables), LAT.size, LON.size)
            dims: tuple[str, ...] = ("time", "lead_time", "variable", "lat", "lon")
            coords = {
                "time": [item.time],
                "lead_time": _h(6 * k),
                "variable": variables,
                "lat": LAT,
                "lon": LON,
            }
            values = np.full((1, 1, *shape), float(k), dtype=np.float32)
            if members:
                dims = ("ensemble", *dims)
                coords["ensemble"] = members
                offset = 10.0 * np.array(members, dtype=np.float32)
                values = values[None] + offset[:, None, None, None, None, None]
            x = xr.DataArray(values, dims=dims, coords=coords)
            yield {stream: x for stream in self.streams}


class _Scorer:
    def __init__(self) -> None:
        self.updates: list[tuple[str, float]] = []

    def begin_item(self, item: WorkItem) -> None:
        pass

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        self.updates.append((stream, float(x.values.max())))

    def finish_item(self, item: WorkItem) -> None:
        pass


def _run(
    reduction: LeadWindow, items: list[WorkItem] | None = None, **kwargs: object
) -> KVBackend:
    store = KVBackend()
    Pipeline(
        _Steps(),
        output={"forecast": store},
        reduction={"forecast": reduction},
        device="cpu",
        **kwargs,  # type: ignore[arg-type]
    ).run(items or work_items([T0], np.timedelta64(24, "h")))
    return store


# -- through Pipeline -----------------------------------------------------------


def test_window_means_relabel_and_reach_the_scorer() -> None:
    scorer = _Scorer()
    forecast, copy = KVBackend(), KVBackend()
    Pipeline(
        _Steps(streams=("forecast", "copy")),
        output={"forecast": forecast, "copy": copy},
        scorer=scorer,
        output_variables={"forecast": ["u10m"]},
        reduction={"forecast": LeadWindow("12h")},
        device="cpu",
    ).run(work_items([T0], np.timedelta64(24, "h")))

    assert list(forecast) == ["u10m:mean:12h"]
    np.testing.assert_array_equal(forecast.coords["lead_time"], _h(12, 24))
    # Leads 6 and 12 h hold 1 and 2; 18 and 24 h hold 3 and 4. Lead 0 is outside.
    np.testing.assert_allclose(forecast["u10m:mean:12h"][0, :, 0, 0], [1.5, 3.5])
    assert copy["u10m"].shape == (1, 5, 2, 3)
    # The scorer sees each window as it closes, and every step of the copy.
    assert [u for u in scorer.updates if u[0] == "forecast"] == [
        ("forecast", 1.5),
        ("forecast", 3.5),
    ]
    assert len([u for u in scorer.updates if u[0] == "copy"]) == 5


@pytest.mark.parametrize(
    "method, expected",
    [("mean", [1.5, 3.5]), ("sum", [3, 7]), ("min", [1, 3]), ("max", [2, 4])],
)
def test_methods(method: str, expected: list[float]) -> None:
    store = _run(LeadWindow("12h", method=method))  # type: ignore[arg-type]
    np.testing.assert_allclose(store[f"v10m:{method}:12h"][0, :, 1, 2], expected)


def test_explicit_ends_overlap_and_drop_leads_outside_every_window() -> None:
    store = _run(LeadWindow("12h", ends=["12h", "18h", "24h"]))
    np.testing.assert_array_equal(store.coords["lead_time"], _h(12, 18, 24))
    np.testing.assert_allclose(store["u10m:mean:12h"][0, :, 0, 0], [1.5, 2.5, 3.5])

    store = _run(LeadWindow(np.timedelta64(6, "h"), method="max", ends=["24h"]))
    np.testing.assert_allclose(store["u10m:max:6h"][0, :, 0, 0], [4.0])


def test_one_reduction_serves_several_streams() -> None:
    forecast, copy = KVBackend(), KVBackend()
    window = LeadWindow("12h")
    pipeline = Pipeline(
        _Steps(streams=("forecast", "copy", "unwritten")),
        output={"forecast": forecast, "copy": copy},
        reduction={"forecast": window, "copy": window, "unwritten": window},
        device="cpu",
    )
    # Each stream gets its own copy; an unwritten, unscored stream is not reduced.
    assert pipeline.reduction["forecast"] is not pipeline.reduction["copy"]
    pipeline.run(work_items([T0, T1], np.timedelta64(24, "h")))
    for store in (forecast, copy):
        np.testing.assert_allclose(store["u10m:mean:12h"][:, :, 0, 0], [[1.5, 3.5]] * 2)


def test_reduce_before_regridding() -> None:
    target = LatLonGrid(np.array([10.0, 5.0, 0.0]), np.arange(0.0, 21.0, 5.0))
    regridder = BilinearRegridder(LatLonGrid(LAT, LON), target)
    store = _run(LeadWindow("24h"), regridder={"forecast": regridder})
    assert store["u10m:mean:24h"].shape == (1, 1, 3, 5)
    np.testing.assert_allclose(store["u10m:mean:24h"], 2.5, rtol=1e-6)


def test_member_batches_reduce_per_member() -> None:
    items = work_items([T0], np.timedelta64(24, "h"), members=[3, 5], member_batch=2)
    store = _run(LeadWindow("12h"), items)
    assert store["u10m:mean:12h"].shape == (2, 1, 2, 2, 3)
    np.testing.assert_allclose(
        store["u10m:mean:12h"][:, 0, :, 0, 0], [[31.5, 33.5], [51.5, 53.5]]
    )


def test_missing_samples_and_unreached_windows_are_nan() -> None:
    store = KVBackend()
    Pipeline(
        # Lead 18 h holds none of the selected variables.
        _Steps(held={18: ["v10m"]}),
        output={"forecast": store},
        output_variables={"forecast": ["u10m"]},
        reduction={"forecast": LeadWindow("12h")},
        device="cpu",
    ).run(
        [WorkItem(T0, np.timedelta64(24, "h")), WorkItem(T1, np.timedelta64(12, "h"))]
    )
    values = store["u10m:mean:12h"][:, :, 0, 0]
    np.testing.assert_allclose(values[:, 0], [1.5, 1.5])
    # The first item misses a sample; the shorter second never reaches 24 h.
    assert np.isnan(values[:, 1]).all()


def test_resume_rejects_another_method(tmp_path: Path) -> None:
    def run(method: str) -> None:
        _run(LeadWindow("12h", method=method), progress=ProgressStore(tmp_path))  # type: ignore[arg-type]

    run("mean")
    run("mean")
    with pytest.raises(ValueError, match="does not match"):
        run("sum")


@pytest.mark.parametrize(
    "reduction, items, match",
    [
        (LeadWindow("9h"), [WorkItem(T0, np.timedelta64(18, "h"))], "whole number"),
        (LeadWindow("12h"), [WorkItem(T0, np.timedelta64(18, "h"))], "whole, nonzero"),
        (
            LeadWindow("12h"),
            [
                WorkItem(T0, np.timedelta64(24, "h")),
                WorkItem(T1, np.timedelta64(6, "h")),
            ],
            "final lead time",
        ),
        (
            LeadWindow("12h", ends=["36h"]),
            [WorkItem(T0, np.timedelta64(24, "h"))],
            "beyond",
        ),
        (
            {"forcast": LeadWindow("12h")},
            [WorkItem(T0, np.timedelta64(24, "h"))],
            "unknown streams",
        ),
    ],
    ids=["cadence", "partial-window", "short-item", "end-beyond", "unknown-stream"],
)
def test_unreducible_runs_fail_before_running(
    reduction: LeadWindow | dict, items: list[WorkItem], match: str
) -> None:
    runner = _Steps()
    pipeline = Pipeline(
        runner,
        output={"forecast": KVBackend()},
        reduction=reduction if isinstance(reduction, dict) else {"forecast": reduction},
        device="cpu",
    )
    with pytest.raises(ValueError, match=match):
        pipeline.run(items)
    assert runner.ran == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"window": "-6h"}, "positive"),
        ({"window": "1M"}, "Calendar"),
        ({"window": "6h", "method": "median"}, "method must be one of"),
        ({"window": "12h", "ends": []}, "no windows"),
        ({"window": "12h", "ends": ["12h", "12h"]}, "repeats"),
        ({"window": "12h", "ends": ["6h"]}, "at least"),
    ],
)
def test_invalid_windows_are_rejected(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        LeadWindow(**kwargs)


# -- the reduction alone --------------------------------------------------------

NAMES = ["t2m", "tp"]
# Daily means, as a source publishing them labels them.
DAILY = ["t2m:mean:-1h:23h", "tp"]


def _signature(
    leads: np.ndarray, dtype: str = "float32", variables: list[str] = NAMES
) -> CoordinateSystem:
    return coord_array(
        ("time", "lead_time", "variable", "lat"),
        {"lead_time": leads, "variable": variables, "lat": [0.0, 1.0]},
        dynamic=("time",),
        dtype=np.dtype(dtype),
    )


def _output(
    leads: np.ndarray,
    values: list[float],
    variables: list[str] = NAMES,
    dtype: str = "float32",
) -> xr.DataArray:
    data = np.array(values, dtype=dtype)[:, None, None] * np.ones(
        (len(leads), len(variables), 2), dtype=dtype
    )
    return xr.DataArray(
        data[None],
        dims=("time", "lead_time", "variable", "lat"),
        coords={
            "time": [T0],
            "lead_time": leads,
            "variable": variables,
            "lat": [0.0, 1.0],
        },
    )


def test_daily_means_compose_and_missing_variables_are_nan() -> None:
    leads = _h(0, 24, 48, 72, 96)
    signature = _signature(leads, "float16", DAILY)
    reduction = LeadWindow("2d")
    labels = ["t2m:mean:-25h:+23h", "tp:mean:48h"]
    planned = reduction.output_coords(signature)
    assert list(planned["variable"].values) == labels
    assert planned.dtype == np.float16

    reduction.begin_item(signature, np.timedelta64(96, "h"))
    # One output may carry several leads; another lacks tp.
    assert reduction.update(_output(leads[:1], [9.0], DAILY, "float16")) == []
    (first,) = reduction.update(_output(leads[1:3], [1.0, 2.0], DAILY, "float16"))
    assert reduction.update(_output(leads[3:4], [3.0], DAILY[:1], "float16")) == []
    (second,) = reduction.update(_output(leads[4:], [4.0], DAILY, "float16"))
    reduction.finish_item()

    assert first.dtype == np.float16
    assert list(first["variable"].values) == labels
    assert set(first.attrs[E2S_STATISTICS]) == set(labels)
    np.testing.assert_array_equal(first["lead_time"].values, _h(48))
    np.testing.assert_allclose(first.values, 1.5)
    np.testing.assert_allclose(second.sel(variable=labels[0]).values, 3.5)
    assert np.isnan(second.sel(variable=labels[1]).values).all()


def test_offset_streams_use_the_gaps_between_their_leads() -> None:
    leads = _h(3, 9, 15, 21)
    signature = _signature(leads)
    reduction = LeadWindow("12h", ends=["15h"])
    np.testing.assert_array_equal(
        reduction.output_coords(signature)["lead_time"].values, _h(15)
    )
    reduction.begin_item(signature, np.timedelta64(21, "h"))
    assert reduction.update(_output(leads[:1], [1.0])) == []
    (window,) = reduction.update(_output(leads[1:3], [2.0, 3.0]))
    assert reduction.update(_output(leads[3:], [4.0])) == []
    reduction.finish_item()
    np.testing.assert_allclose(window.values, 2.5)


@pytest.mark.parametrize(
    "reduction, leads, variables, match",
    [
        # Default windows end at multiples of 12 h, which needs lead 6 h.
        (LeadWindow("12h"), _h(3, 9, 15, 21, 24), NAMES, "never publishes lead 6h"),
        (
            LeadWindow("18h", ends=["24h"]),
            _h(0, 6, 18, 24),
            NAMES,
            "never publishes lead 12h",
        ),
        (LeadWindow("6h"), _h(0), NAMES, "no positive lead"),
        # A sum of daily means is not one statistic of the samples.
        (LeadWindow("2d", method="sum"), _h(0, 24, 48), DAILY, "do not compose"),
    ],
)
def test_unplannable_streams_raise(
    reduction: LeadWindow, leads: np.ndarray, variables: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        reduction.output_coords(_signature(leads, variables=variables))


def test_streams_without_variables_or_lead_time() -> None:
    # One positive lead sets the cadence; without variables nothing is relabelled.
    signature = coord_array(
        ("time", "lead_time", "lat"),
        {"lead_time": _h(0, 12), "lat": [0.0, 1.0]},
        dynamic=("time",),
    )
    reduction = LeadWindow("12h", method="max")
    assert reduction.output_coords(signature).dims == signature.dims
    reduction.begin_item(signature, np.timedelta64(12, "h"))
    x = xr.DataArray(
        np.full((1, 1, 2), 5.0, dtype=np.float32),
        dims=("time", "lead_time", "lat"),
        coords={"time": [T0], "lead_time": _h(12), "lat": [0.0, 1.0]},
    )
    (window,) = reduction.update(x)
    np.testing.assert_allclose(window.values, 5.0)

    with pytest.raises(ValueError, match="needs a lead_time"):
        reduction.output_coords(signature.isel(lead_time=0, drop=True))


def test_lifecycle_errors() -> None:
    leads = _h(0, 6, 12)
    signature = _signature(leads)
    reduction = LeadWindow("12h")

    with pytest.raises(RuntimeError, match="begin_item was not called"):
        reduction.update(_output(leads[1:2], [1.0]))

    reduction.begin_item(signature, np.timedelta64(12, "h"))
    reduction.update(_output(leads[1:2], [1.0]))
    with pytest.raises(RuntimeError, match="ended before"):
        reduction.finish_item()
    with pytest.raises(RuntimeError, match="begin_item was not called"):
        reduction.finish_item()

    reduction.begin_item(signature, np.timedelta64(12, "h"))
    with pytest.raises(RuntimeError, match="received 1 outputs, not 2"):
        reduction.update(_output(leads[2:], [2.0]))

    # begin_item discards the failed item's partial results.
    reduction.begin_item(signature, np.timedelta64(12, "h"))
    assert len(reduction.update(_output(leads[1:], [1.0, 2.0]))) == 1
    with pytest.raises(RuntimeError, match="after the window"):
        reduction.update(_output(leads[1:2], [1.0]))
    with pytest.raises(ValueError, match="lead_time"):
        reduction.update(_output(leads[1:2], [1.0]).isel(lead_time=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reduction_stays_on_the_device() -> None:
    cupy = pytest.importorskip("cupy")
    leads = _h(0, 6, 12)
    reduction = LeadWindow("12h")
    reduction.begin_item(_signature(leads), np.timedelta64(12, "h"))
    outputs = [_output(leads[k : k + 1], [float(k)]).e2s.as_cupy() for k in (1, 2)]
    assert reduction.update(outputs[0]) == []
    (window,) = reduction.update(outputs[1])
    assert isinstance(window.data, cupy.ndarray)
    np.testing.assert_allclose(window.data.get(), 1.5)
