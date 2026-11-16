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

"""Lead-time reductions in Pipeline.

The model adds one per 6 h step and the source returns zeros, so lead ``6k h``
holds the value ``k`` everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
import xarray as xr
from test_pipeline import _Recording, _Scorer
from test_runner import _model, _stochastic_model, _Zeros

from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import KVBackend
from earth2studio.models.dx import Identity
from earth2studio.run import (
    LeadWindow,
    Pipeline,
    PrognosticRunner,
    WorkItem,
    work_items,
)
from earth2studio.utils.coords import E2S_STATISTICS, coord_array
from earth2studio.utils.time_statistics import (
    _TIME_STATISTICS,
    register_time_statistic,
)

T0 = np.datetime64("2026-01-01T00")
T1 = np.datetime64("2026-01-02T00")


def _h(*hours: int) -> np.ndarray:
    return np.array(hours, dtype="timedelta64[h]").astype("timedelta64[ns]")


def _span(array: xr.DataArray, dimension: object) -> xr.DataArray:
    return array.max(dim=dimension) - array.min(dim=dimension)


@pytest.fixture
def span() -> Iterator[None]:
    register_time_statistic("span", _span)
    try:
        yield
    finally:
        _TIME_STATISTICS.pop("span", None)


def _runner(**kwargs: object) -> PrognosticRunner:
    return PrognosticRunner(_model(), _Zeros(), **kwargs)  # type: ignore[arg-type]


def _run(reduction: LeadWindow, hours: int = 24, **kwargs: object) -> KVBackend:
    store = KVBackend()
    Pipeline(
        _runner(),
        output={"forecast": store},
        reduction={"forecast": reduction},
        device="cpu",
        **kwargs,  # type: ignore[arg-type]
    ).run(work_items([T0], np.timedelta64(hours, "h")))
    return store


def test_window_means_skip_the_initial_condition_and_relabel() -> None:
    scorer = _Scorer()
    forecast, copy = KVBackend(), KVBackend()
    Pipeline(
        _runner(diagnostics={"copy": Identity()}),
        output={"forecast": forecast, "copy": copy},
        scorer=scorer,
        output_variables={"forecast": ["u10m"]},
        reduction={"forecast": LeadWindow("12h")},
        device="cpu",
    ).run(work_items([T0], np.timedelta64(24, "h")))

    assert "u10m:mean:12h" in forecast and "u10m" not in forecast
    np.testing.assert_array_equal(forecast.coords["lead_time"], _h(12, 24))
    # Leads 6 and 12 h hold 1 and 2; 18 and 24 h hold 3 and 4.
    np.testing.assert_allclose(forecast["u10m:mean:12h"][0, :, 0, 0], [1.5, 3.5])
    # The unreduced stream keeps every step.
    assert copy["u10m"].shape == (1, 5, 2, 3)
    # The scorer sees each window once it closes, and every step of the copy.
    updates = [call[1:] for call in scorer.calls if call[0] == "update"]
    assert [u[0] for u in updates].count("forecast") == 2
    assert [u[0] for u in updates].count("copy") == 5
    assert ("forecast", ("time", "lead_time", "variable", "lat", "lon"), 3.5) in (
        updates
    )


@pytest.mark.parametrize(
    "method, expected",
    [("sum", [3.0, 7.0]), ("min", [1.0, 3.0]), ("max", [2.0, 4.0]), ("span", [1, 1])],
)
@pytest.mark.usefixtures("span")
def test_methods(method: str, expected: list[float]) -> None:
    store = _run(LeadWindow("12h", method=method))
    np.testing.assert_allclose(store[f"v10m:{method}:12h"][0, :, 1, 2], expected)


def test_explicit_ends_overlap_and_drop_leads_outside_every_window() -> None:
    store = _run(LeadWindow("12h", ends=["12h", "18h", "24h"]))
    np.testing.assert_array_equal(store.coords["lead_time"], _h(12, 18, 24))
    np.testing.assert_allclose(store["u10m:mean:12h"][0, :, 0, 0], [1.5, 2.5, 3.5])

    store = _run(LeadWindow(np.timedelta64(6, "h"), ends=["24h"], relabel=False))
    np.testing.assert_array_equal(store.coords["lead_time"], _h(24))
    np.testing.assert_allclose(store["u10m"][0, :, 0, 0], [4.0])


def test_reduce_before_regridding() -> None:
    source = LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 10.0, 20.0]))
    target = LatLonGrid(np.array([10.0, 5.0, 0.0]), np.arange(0.0, 21.0, 5.0))
    store = _run(
        LeadWindow("24h"), regridder={"forecast": BilinearRegridder(source, target)}
    )
    assert store["u10m:mean:1d"].shape == (1, 1, 3, 5)
    np.testing.assert_allclose(store["u10m:mean:1d"], 2.5, rtol=1e-6)


def test_member_batches_reduce_per_member() -> None:
    store = KVBackend()
    runner = PrognosticRunner(_stochastic_model(), _Zeros())  # type: ignore[arg-type]
    Pipeline(
        runner,
        output={"forecast": store},
        reduction={"forecast": LeadWindow("12h")},
        device="cpu",
    ).run(work_items([T0], np.timedelta64(24, "h"), members=[3, 5], member_batch=2))
    assert store["u10m:mean:12h"].shape == (2, 1, 2, 2, 3)
    np.testing.assert_allclose(store["u10m:mean:12h"][1, 0, :, 0, 0], [1.5, 3.5])


def test_shorter_items_write_only_the_windows_they_reach() -> None:
    store = KVBackend()
    Pipeline(
        _runner(),
        output={"forecast": store},
        reduction={"forecast": LeadWindow("12h")},
        device="cpu",
    ).run(
        [WorkItem(T0, np.timedelta64(24, "h")), WorkItem(T1, np.timedelta64(12, "h"))]
    )
    np.testing.assert_allclose(store["u10m:mean:12h"][1, 0, 0, 0], 1.5)
    assert np.isnan(store["u10m:mean:12h"][1, 1, 0, 0])


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
    ],
    ids=["cadence", "partial-window", "short-item", "end-beyond"],
)
def test_unreducible_runs_fail_before_running(
    reduction: LeadWindow, items: list[WorkItem], match: str
) -> None:
    runner = _Recording(_runner())
    pipeline = Pipeline(
        runner,
        output={"forecast": KVBackend()},
        reduction={"forecast": reduction},
        device="cpu",
    )
    with pytest.raises(ValueError, match=match):
        pipeline.run(items)
    assert runner.ran == []


def test_unknown_reduction_stream_is_rejected() -> None:
    pipeline = Pipeline(
        _runner(),
        output={"forecast": KVBackend()},
        reduction={"forcast": LeadWindow("12h")},
        device="cpu",
    )
    with pytest.raises(ValueError, match="unknown streams"):
        pipeline.run(work_items([T0], np.timedelta64(24, "h")))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"window": "-6h"}, "positive"),
        ({"window": "1M"}, "Calendar"),
        ({"window": "6h", "method": "mode"}, "Unknown"),
        ({"window": "12h", "ends": []}, "no windows"),
        ({"window": "12h", "ends": ["12h", "12h"]}, "repeats"),
        ({"window": "12h", "ends": ["6h"]}, "at least"),
    ],
)
def test_invalid_windows_are_rejected(kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        LeadWindow(**kwargs)


# -- state, without a pipeline ------------------------------------------------


def _signature(leads: np.ndarray, dtype: str = "float32") -> xr.DataArray:
    return coord_array(
        ("time", "lead_time", "variable", "lat"),
        {
            "lead_time": leads,
            "variable": ["t2m:mean:-1h:23h", "tp"],
            "lat": [0.0, 1.0],
        },
        dynamic=("time",),
        dtype=np.dtype(dtype),
    )


def _output(
    leads: np.ndarray, values: list[float], variables: list[str], dtype: str
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
    day = np.timedelta64(24, "h")
    leads = (np.arange(0, 5) * day).astype("timedelta64[ns]")
    signature = _signature(leads, "float16")
    reduction = LeadWindow("2d", method="mean")
    planned = reduction.output_coords(signature)
    labels = ["t2m:mean:-25h:+23h", "tp:mean:2d"]
    assert list(planned["variable"].values) == labels
    assert planned.dtype == np.float16

    state = reduction.begin(signature, np.timedelta64(96, "h"))
    both = ["t2m:mean:-1h:23h", "tp"]
    # One output may carry several leads; another lacks tp.
    assert state.update(_output(leads[:1], [9.0], both, "float16")) == []
    first = state.update(_output(leads[1:3], [1.0, 2.0], both, "float16"))
    lacking = state.update(_output(leads[3:4], [3.0], both[:1], "float16"))
    second = state.update(_output(leads[4:], [4.0], both, "float16"))
    assert lacking == []
    state.finish()

    (window,) = first
    assert window.dtype == np.float16
    assert list(window["variable"].values) == labels
    assert set(window.attrs[E2S_STATISTICS]) == set(labels)
    np.testing.assert_array_equal(window["lead_time"].values, _h(48))
    np.testing.assert_allclose(window.values, 1.5)
    (window,) = second
    np.testing.assert_allclose(window.sel(variable=labels[0]).values, 3.5)
    assert np.isnan(window.sel(variable=labels[1]).values).all()


def test_noncomposing_labels_need_relabel_off() -> None:
    signature = _signature(_h(0, 24, 48))
    with pytest.raises(ValueError, match="compose"):
        LeadWindow("2d", method="sum").output_coords(signature)
    planned = LeadWindow("2d", method="sum", relabel=False).output_coords(signature)
    assert list(planned["variable"].values) == ["t2m:mean:-1h:23h", "tp"]


def test_unfinished_and_out_of_order_windows_raise() -> None:
    leads = _h(0, 6, 12)
    signature = _signature(leads)
    names = ["t2m:mean:-1h:23h", "tp"]
    reduction = LeadWindow("12h", relabel=False)

    state = reduction.begin(signature, np.timedelta64(12, "h"))
    state.update(_output(leads[1:2], [1.0], names, "float32"))
    with pytest.raises(RuntimeError, match="ended before"):
        state.finish()

    state = reduction.begin(signature, np.timedelta64(12, "h"))
    with pytest.raises(RuntimeError, match="received 1 outputs, not 2"):
        state.update(_output(leads[2:], [2.0], names, "float32"))

    state = reduction.begin(signature, np.timedelta64(12, "h"))
    assert len(state.update(_output(leads[1:], [1.0, 2.0], names, "float32"))) == 1
    with pytest.raises(RuntimeError, match="after the window"):
        state.update(_output(leads[1:2], [1.0], names, "float32"))

    with pytest.raises(ValueError, match="lead_time"):
        state.update(_output(leads[1:2], [1.0], names, "float32").isel(lead_time=0))


def test_streams_must_publish_every_lead_a_window_needs() -> None:
    with pytest.raises(ValueError, match="never publishes lead 12h"):
        LeadWindow("18h", ends=["24h"]).output_coords(_signature(_h(0, 6, 18, 24)))
    with pytest.raises(ValueError, match="no positive lead"):
        LeadWindow("6h").output_coords(_signature(_h(0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reduction_stays_on_the_device() -> None:
    cupy = pytest.importorskip("cupy")
    leads = _h(0, 6, 12)
    names = ["t2m:mean:-1h:23h", "tp"]
    state = LeadWindow("12h", relabel=False).begin(
        _signature(leads), np.timedelta64(12, "h")
    )
    outputs = [
        _output(leads[k : k + 1], [float(k)], names, "float32").e2s.as_cupy()
        for k in (1, 2)
    ]
    assert state.update(outputs[0]) == []
    (window,) = state.update(outputs[1])
    assert isinstance(window.data, cupy.ndarray)
    np.testing.assert_allclose(window.data.get(), 1.5)
