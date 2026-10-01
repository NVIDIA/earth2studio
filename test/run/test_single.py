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

"""The default single-model plan and session.

The model is FCN's real DataArray execution path with an add-one core, and the
source returns zeros, so published step ``k`` holds the value ``k`` everywhere.
That makes state feedback, resume position, and output post-processing each
directly observable.
"""

from collections.abc import Mapping

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import fetch_data
from earth2studio.models.px.fcn import FCN
from earth2studio.run.component import PrognosticComponent
from earth2studio.run.schedules import FixedCadence
from earth2studio.run.session import (
    CheckpointCapability,
    ExecutionPlan,
    OutputPort,
    PortRef,
    WorkItem,
)
from earth2studio.run.single import OutputTransform, SingleModelPlan
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")
SIX_HOURS = np.timedelta64(6, "h")


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


class _TwoLeadStub:
    """Declaration-only model with a two-step input history, like DLESyM.

    FCN rejects multi-lead input, so this stub supplies only the coordinate
    contract that plan construction reads; it is never rolled out.
    """

    def input_coords(self) -> CoordinateSystem:
        return coord_array_like(
            _model().input_coords(),
            {"lead_time": np.array([-6, 0], dtype="timedelta64[h]")},
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            _model().input_coords(),
            {"lead_time": np.array([6], dtype="timedelta64[h]")},
        )


class _Zeros:
    """Deterministic source; counts calls so tests can see refetches."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        self.calls += 1
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        return xr.DataArray(
            np.zeros((len(time), len(variable), 2, 3)),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
        )


def _model() -> _TinyFCN:
    return _TinyFCN(_AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))


def _item(hours: int = 18, time: np.datetime64 = T0) -> WorkItem:
    return WorkItem(time, np.timedelta64(hours, "h"))


def _mask(outputs: Mapping[str, xr.DataArray]) -> Mapping[str, xr.DataArray]:
    return {"forecast": outputs["forecast"].where(outputs["forecast"]["lat"] > 0, 0.0)}


def _wind_ports(ports: tuple[OutputPort, ...]) -> tuple[OutputPort, ...]:
    (forecast,) = ports
    signature = coord_array_like(forecast.signature, {"variable": ["ws10m"]})
    return forecast, OutputPort("wind_speed", ("ws10m",), signature, forecast.schedule)


def _wind(outputs: Mapping[str, xr.DataArray]) -> Mapping[str, xr.DataArray]:
    x = outputs["forecast"]
    u = x.sel(variable="u10m", drop=True)
    v = x.sel(variable="v10m", drop=True)
    speed = np.hypot(u, v).expand_dims(
        variable=["ws10m"], axis=x.get_axis_num("variable")
    )
    return {**outputs, "wind_speed": speed}


MASK = OutputTransform(_mask, "north-mask-v1")
WIND = OutputTransform(_wind, "wind-speed-v1", ports=_wind_ports)


def test_publishes_initial_condition_then_each_step_to_the_horizon() -> None:
    plan = SingleModelPlan(_model(), _Zeros())
    events = list(plan.open(_item(18)).run())

    assert [event.produced_at for event in events] == [
        T0 + k * SIX_HOURS for k in range(4)
    ]
    assert {(event.component, event.port) for event in events} == {
        ("model", "forecast")
    }
    for k, event in enumerate(events):
        np.testing.assert_array_equal(event.data.values, k)
        assert event.data["lead_time"].values[-1] == k * SIX_HOURS


def test_plan_is_reusable_across_work_items() -> None:
    """Built once, opened per item; requests and times follow the item."""
    plan: ExecutionPlan = SingleModelPlan(_model(), _Zeros())
    later = T0 + np.timedelta64(1, "D")

    requests = plan.external_requests(_item(6, later))
    assert requests is not None
    (request,) = requests
    assert request.valid_time == later
    assert request.variables == ("u10m", "v10m")

    first = [event.produced_at for event in plan.open(_item(6)).run()]
    second = [event.produced_at for event in plan.open(_item(6, later)).run()]
    assert first == [T0, T0 + SIX_HOURS]
    assert second == [later, later + SIX_HOURS]


def test_plan_derives_ports_cadence_and_capabilities_from_the_model() -> None:
    plan = SingleModelPlan(_model(), _Zeros())

    assert plan.cadence.step == SIX_HOURS
    assert set(plan.output_ports) == {PortRef("model", "forecast")}
    assert plan.output_ports[PortRef("model", "forecast")].variables == (
        "u10m",
        "v10m",
    )
    assert plan.checkpoint_capability is CheckpointCapability.SNAPSHOTTABLE
    assert not plan.supports_member_batching
    assert "every" in plan.describe()


def test_plan_declarations_come_from_the_component_spec() -> None:
    """What a model requires and publishes is derived once, by its adapter."""
    model = _model()
    spec = PrognosticComponent(model).spec
    plan = SingleModelPlan(model, _Zeros())

    assert [port.name for port in plan.output_ports.values()] == [
        port.name for port in spec.outputs
    ]
    assert plan.cadence.fingerprint() == spec.schedule.fingerprint()
    assert plan.checkpoint_capability is spec.checkpoint_capability
    (requirement,) = spec.inputs
    (request,) = plan.external_requests(_item(6))
    assert requirement.phase == "initialize"
    assert request.variables == requirement.variables


def test_component_steps_and_resumes_without_a_plan() -> None:
    """The unit a graph will schedule works on its own."""
    component = PrognosticComponent(_model())
    initial = fetch_data(
        _Zeros(),
        time=np.array([T0]),
        variable=np.array(["u10m", "v10m"]),
        lead_time=np.array([np.timedelta64(0, "h")]),
    )
    session = component.open({"initial_condition": initial})
    for k in range(3):
        step = session.step(T0 + k * SIX_HOURS, {})
        np.testing.assert_array_equal(step["forecast"].values, k)
    snapshot = session.snapshot()
    session.close()

    resumed = component.open({})
    resumed.restore(snapshot)
    step = resumed.step(T0 + 3 * SIX_HOURS, {})
    np.testing.assert_array_equal(step["forecast"].values, 3)
    with pytest.raises(ValueError):
        resumed.step(T0 + 4 * SIX_HOURS, {"forcing": initial})


def test_resume_matches_an_uninterrupted_run_without_refetching() -> None:
    source = _Zeros()
    plan = SingleModelPlan(_model(), source)
    uninterrupted = list(plan.open(_item(18)).run())

    session = plan.open(_item(18))
    events = session.run()
    head = [next(events), next(events)]
    raw = plan.encode_snapshot(session.snapshot())
    calls_before_resume = source.calls

    snapshot = plan.decode_snapshot(raw)
    assert snapshot.output_cursor[PortRef("model", "forecast")] == 2
    tail = list(plan.open(_item(18), snapshot).run())

    assert source.calls == calls_before_resume
    assert len(head) + len(tail) == len(uninterrupted)
    for got, expected in zip(head + tail, uninterrupted):
        assert got.produced_at == expected.produced_at
        xr.testing.assert_identical(got.data, expected.data)


def test_output_post_processing_never_feeds_back_into_state() -> None:
    plan = SingleModelPlan(_model(), _Zeros(), transforms=(MASK,))
    session = plan.open(_item(12))
    events = list(session.run())

    for k, event in enumerate(events):
        np.testing.assert_array_equal(event.data.sel(lat=0).values, 0)
        np.testing.assert_array_equal(event.data.sel(lat=10).values, k)
    state = session.snapshot().payload.model_state["state"]
    np.testing.assert_array_equal(state.values, 2)


def test_changed_outputs_are_declared_by_the_transform() -> None:
    plan = SingleModelPlan(_model(), _Zeros(), transforms=(WIND,))
    assert set(plan.output_ports) == {
        PortRef("model", "forecast"),
        PortRef("model", "wind_speed"),
    }

    session = plan.open(_item(6))
    events = session.run()
    forecast = next(events)
    assert forecast.port == "forecast"
    assert not session.checkpoint_boundary
    with pytest.raises(RuntimeError):
        session.snapshot()

    speed = next(events)
    assert speed.port == "wind_speed"
    assert session.checkpoint_boundary
    np.testing.assert_array_equal(speed.data["variable"].values, ["ws10m"])


def test_transform_identity_prevents_incompatible_resume() -> None:
    """A snapshot from one transform configuration cannot resume another."""
    model, source = _model(), _Zeros()
    plain = SingleModelPlan(model, source)
    masked = SingleModelPlan(model, source, transforms=(MASK,))
    assert plain.identity == SingleModelPlan(model, source).identity
    assert plain.identity != masked.identity

    session = plain.open(_item(6))
    next(session.run())
    with pytest.raises(ValueError):
        masked.open(_item(6), session.snapshot())


def test_multi_lead_input_models_cannot_resume_by_reseeding() -> None:
    plan = SingleModelPlan(_TwoLeadStub(), _Zeros())  # type: ignore[arg-type]
    assert plan.checkpoint_capability is CheckpointCapability.UNSUPPORTED

    requests = plan.external_requests(_item(6))
    assert [request.valid_time for request in requests] == [T0 - SIX_HOURS, T0]

    session = plan.open(_item(6))
    assert not session.checkpoint_boundary
    with pytest.raises(RuntimeError):
        session.snapshot()


def test_horizon_must_be_a_whole_number_of_steps() -> None:
    plan = SingleModelPlan(_model(), _Zeros())
    with pytest.raises(ValueError):
        plan.open(_item(7))


def test_fixed_cadence_is_lazy_and_offset_aware() -> None:
    cadence = FixedCadence(SIX_HOURS, offset=np.timedelta64(3, "h"))
    times = list(
        cadence.iter_between(
            T0, T0 + np.timedelta64(4, "h"), T0 + np.timedelta64(1, "D")
        )
    )

    assert times == [T0 + np.timedelta64(h, "h") for h in (9, 15, 21)]
    assert cadence.fingerprint() == "fixed:21600s+10800s"
    with pytest.raises(ValueError):
        FixedCadence(np.timedelta64(0, "h"))


def test_composed_transforms_resume_all_ports_without_changing_model_state() -> None:
    plan = SingleModelPlan(_model(), _Zeros(), transforms=(MASK, WIND))
    expected = list(plan.open(_item()).run())
    session = plan.open(_item())
    stream = session.run()
    head = [next(stream) for _ in range(4)]
    snapshot = plan.decode_snapshot(plan.encode_snapshot(session.snapshot()))
    stream.close()
    tail = list(plan.open(_item(), snapshot).run())
    assert len(head + tail) == len(expected)
    for actual, wanted in zip(head + tail, expected):
        assert actual.port == wanted.port
        assert actual.produced_at == wanted.produced_at
        xr.testing.assert_identical(actual.data, wanted.data)
    speed = tail[-1].data
    np.testing.assert_array_equal(speed.sel(lat=0), 0)
    np.testing.assert_allclose(speed.sel(lat=10), np.hypot(3, 3))


def test_transform_must_publish_declared_ports() -> None:
    plan = SingleModelPlan(
        _model(), _Zeros(), transforms=(OutputTransform(_wind, "missing-declaration"),)
    )
    session = plan.open(_item())
    with pytest.raises(ValueError, match="declarations"):
        list(session.run())
    assert not session.checkpoint_boundary
    with pytest.raises(RuntimeError):
        session.snapshot()


def test_output_declarations_reject_duplicate_names() -> None:
    transform = OutputTransform(_mask, "duplicate", ports=lambda ports: ports + ports)
    with pytest.raises(ValueError, match="uniquely named"):
        SingleModelPlan(_model(), _Zeros(), transforms=(transform,))
