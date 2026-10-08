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

"""The built-in single-model runner.

The model is FCN's real DataArray execution path with an add-one core, and the
source returns zeros, so step ``k`` holds the value ``k`` everywhere.
"""

from collections.abc import Iterator

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.run as run
from earth2studio.io import KVBackend
from earth2studio.models.dx import Identity
from earth2studio.models.px.fcn import FCN
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import DiagnosticRunner, PrognosticRunner, Runner, WorkItem
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")


class _AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _TinyFCN(FCN):
    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": ["u10m", "v10m"],
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
            dynamic=("batch",),
        )


class _Zeros:
    fill = 0.0

    def __init__(self) -> None:
        self.times: list[np.datetime64] = []

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.times.extend(time)
        return xr.DataArray(
            np.full((len(time), len(variable), 2, 3), self.fill, dtype=np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
        )


class _Ones(_Zeros):
    fill = 1.0


class _WindSpeed:
    """Diagnostic computing 10 m wind speed from its components."""

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": ["u10m", "v10m"], "lat": [10, 0], "lon": [0, 10, 20]},
            dynamic=("batch",),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(input_coords, {"variable": ["ws10m"]})

    def to(self, device: torch.device) -> "_WindSpeed":
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        speed = np.hypot(x.sel(variable="u10m"), x.sel(variable="v10m"))
        return speed.expand_dims(variable=["ws10m"], axis=x.get_axis_num("variable"))


class _Forced(PrognosticMixin):
    """Migrated model adding its forcing; records the forcing leads it receives."""

    def __init__(self) -> None:
        self.seen: list[list[np.timedelta64]] = []

    def input_coords(self) -> CoordinateSystem:
        return _model().input_coords()

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(self.input_coords(), {"variable": ["sst"]})

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords, {"lead_time": np.array([6], dtype="timedelta64[h]")}
        )

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        self.seen.append(list(forcing["lead_time"].values))
        y = x + 1 + float(forcing.values.mean())
        return y.assign_coords(lead_time=x["lead_time"] + np.timedelta64(6, "h"))

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        return self._advance(x, forcing), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, forcing), None

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, forcing)


def _model() -> _TinyFCN:
    return _TinyFCN(_AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))


def _item(hours: int = 18) -> WorkItem:
    return WorkItem(T0, np.timedelta64(hours, "h"))


def test_steps_through_horizon() -> None:
    runner: Runner = PrognosticRunner(_model(), _Zeros())
    steps = list(runner.run_item(_item()))
    assert [set(s) for s in steps] == [{"forecast"}] * 4
    for k, step in enumerate(steps):
        np.testing.assert_array_equal(step["forecast"].values, k)
    leads = [s["forecast"]["lead_time"].values[-1] for s in steps]
    np.testing.assert_array_equal(
        leads, runner.output_coords(_item().horizon)["forecast"]["lead_time"]
    )


def test_matches_deterministic_workflow() -> None:
    model, source = _model(), _Zeros()
    # Pin CPU: deterministic moves the shared model to CUDA when available, and the
    # runner then follows the model's device.
    io = run.deterministic(
        [T0], 3, model, source, KVBackend(), device=torch.device("cpu"), verbose=False
    )
    assert PrognosticRunner(model, source).device == torch.device("cpu")
    steps = list(PrognosticRunner(model, source).run_item(_item()))
    written = io["u10m"][0]  # time, lead_time, lat, lon
    for k, step in enumerate(steps):
        np.testing.assert_array_equal(
            written[k], step["forecast"].sel(variable="u10m").values.squeeze()
        )


def test_diagnostics_publish_their_own_streams() -> None:
    runner = PrognosticRunner(_model(), _Zeros(), diagnostics={"copy": Identity()})
    coords = runner.output_coords(_item().horizon)
    assert set(coords) == {"forecast", "copy"}
    np.testing.assert_array_equal(
        coords["copy"]["lead_time"], coords["forecast"]["lead_time"]
    )
    last = list(runner.run_item(_item()))[-1]
    xr.testing.assert_equal(last["copy"], last["forecast"])


def test_requests_are_per_item_and_match_fetch() -> None:
    later = WorkItem(np.datetime64("2026-02-01"), np.timedelta64(6, "h"))
    (request,) = PrognosticRunner(_model(), _Zeros()).data_requests(later)
    np.testing.assert_array_equal(request.time, [later.time])
    np.testing.assert_array_equal(request.variable, ["u10m", "v10m"])


@pytest.mark.parametrize(
    "item",
    [
        WorkItem(T0, np.timedelta64(7, "h")),
        WorkItem(T0, np.timedelta64(6, "h"), member_ids=(0, 1)),
    ],
)
def test_rejects_unsupported_items(item: WorkItem) -> None:
    with pytest.raises(ValueError):
        list(PrognosticRunner(_model(), _Zeros()).run_item(item))


def test_forecast_stream_name_is_reserved() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_model(), _Zeros(), diagnostics={"forecast": Identity()})


def test_initial_condition_only_for_zero_horizon() -> None:
    steps = list(PrognosticRunner(_model(), _Zeros()).run_item(_item(0)))
    assert len(steps) == 1
    np.testing.assert_array_equal(steps[0]["forecast"].values, 0)


def test_forcing_is_fetched_requested_and_sent_each_step() -> None:
    model = _Forced()
    runner = PrognosticRunner(model, _Zeros(), forcing=(_Zeros(),))
    steps = list(runner.run_item(_item(18)))
    assert len(steps) == 4
    hours = [[int(v / np.timedelta64(1, "h")) for v in leads] for leads in model.seen]
    assert hours == [[0], [6], [12]]
    requests = runner.data_requests(_item(18))
    requested = [list(r.lead_time // np.timedelta64(1, "h")) for r in requests[1:]]
    assert requested == hours


def test_missing_forcing_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_Forced(), _Zeros())


def test_negative_horizon_is_rejected_before_fetching() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_model(), _Zeros()).nsteps(np.timedelta64(-6, "h"))


def test_device_follows_the_model() -> None:
    assert PrognosticRunner(_model(), _Zeros()).device == torch.device("cpu")


def test_diagnostic_runner_steps_through_source_times() -> None:
    source = _Ones()
    runner: Runner = DiagnosticRunner(
        {"wind": _WindSpeed()}, source, step=np.timedelta64(6, "h")
    )
    steps = list(runner.run_item(_item(12)))
    assert [set(s) for s in steps] == [{"wind"}] * 3
    np.testing.assert_allclose(steps[-1]["wind"].values, np.hypot(1, 1))
    np.testing.assert_array_equal(steps[-1]["wind"]["variable"], ["ws10m"])
    hours = [int((t - T0) / np.timedelta64(1, "h")) for t in source.times]
    assert hours == [0, 6, 12]
    coords = runner.output_coords(_item(12).horizon)["wind"]
    np.testing.assert_array_equal(
        coords["lead_time"], np.array([0, 6, 12], dtype="timedelta64[h]")
    )
    assert list(coords["variable"]) == ["ws10m"]
    requests = runner.data_requests(_item(12))
    assert [r.lead_time.tolist() for r in requests] == [
        [lead] for lead in coords["lead_time"].tolist()
    ]


def test_diagnostic_runner_without_step_runs_one_step() -> None:
    runner = DiagnosticRunner({"wind": _WindSpeed()}, _Ones())
    assert len(list(runner.run_item(_item(0)))) == 1
    with pytest.raises(ValueError):
        list(runner.run_item(_item(6)))


def test_diagnostic_runner_needs_a_source() -> None:
    with pytest.raises(ValueError):
        DiagnosticRunner({"wind": _WindSpeed()})


def test_diagnostic_runner_needs_declared_input_variables() -> None:
    with pytest.raises(ValueError):
        DiagnosticRunner({"copy": Identity()}, _Ones())
