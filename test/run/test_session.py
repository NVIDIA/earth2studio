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

"""The shared boundary is usable, and usable on its own.

The fixtures implement resumable sessions directly and ordinary adaptive loops
through ``LoopPlan``. Both use one execution boundary without component graphs
or supervision subclasses.
"""

import pickle
import subprocess
import sys
from collections.abc import Iterator, Mapping

import numpy as np
import pytest
import xarray as xr

import earth2studio.run as run
from earth2studio.run.schedules import Schedule
from earth2studio.run.session import (
    CheckpointCapability,
    ExecutionPlan,
    LoopPlan,
    OutputEvent,
    OutputPort,
    PortRef,
    ResolvedRequest,
    RunSession,
    RunSnapshot,
    SnapshotCompatibility,
    WorkItem,
)
from earth2studio.utils.coords import coord_array

COMPATIBILITY = SnapshotCompatibility(
    schema_version=1,
    earth2studio_version="1.0.0a0",
    plan_identity="fixture-plan",
    component_versions={"atmos": "1"},
)


class _OneActivation:
    def iter_between(
        self,
        reference_time: np.datetime64,
        start: np.datetime64,
        stop: np.datetime64,
    ) -> Iterator[np.datetime64]:
        if start <= reference_time < stop:
            yield reference_time

    def fingerprint(self) -> str:
        return "at-reference-time"


class _Session:
    """A hand-written session: no adapters, no bindings, no graph."""

    def __init__(self, item: WorkItem, ports: tuple[PortRef, ...]) -> None:
        self.item = item
        self.ports = ports
        self.cursor = 0

    @property
    def checkpoint_boundary(self) -> bool:
        return True

    def run(self) -> Iterator[OutputEvent]:
        for index in range(self.cursor, len(self.ports)):
            port = self.ports[index]
            data = xr.DataArray(
                np.array([index], dtype=np.float32),
                dims=("variable",),
                coords={"variable": ["t2m"]},
            )
            self.cursor = index + 1
            yield OutputEvent(port.component, port.port, self.item.time, data)

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            compatibility=COMPATIBILITY,
            output_cursor={port: self.cursor for port in self.ports},
            payload={"cursor": self.cursor},
        )


class _Plan:
    identity = "fixture-plan"
    checkpoint_capability = CheckpointCapability.SNAPSHOTTABLE
    supports_member_batching = True

    def __init__(self, ports: tuple[PortRef, ...], schedule: Schedule) -> None:
        self.ports = ports
        signature = coord_array(("variable",), {"variable": ["t2m"]})
        self.output_ports = {
            port: OutputPort(port.port, ("t2m",), signature, schedule) for port in ports
        }

    def external_requests(self, item: WorkItem) -> tuple[ResolvedRequest, ...]:
        return ()

    def describe(self) -> str:
        return ", ".join(port.component for port in self.ports)

    def open(
        self,
        item: WorkItem,
        snapshot: RunSnapshot | None = None,
    ) -> RunSession:
        session = _Session(item, self.ports)
        if snapshot is not None:
            if not snapshot.compatibility.is_compatible_with(COMPATIBILITY):
                raise ValueError("incompatible snapshot")
            session.cursor = snapshot.payload["cursor"]  # type: ignore[index]
        return session

    def encode_snapshot(self, snapshot: RunSnapshot) -> bytes:
        return pickle.dumps(snapshot)

    def decode_snapshot(self, raw: bytes) -> RunSnapshot:
        return pickle.loads(raw)  # noqa: S301


def _drive(plan: ExecutionPlan, item: WorkItem) -> list[OutputEvent]:
    return list(plan.open(item).run())


def test_run_namespace_stays_small() -> None:
    """Supervision exports workflows and the work unit, nothing else."""
    assert sorted(run.__all__) == [
        "WorkItem",
        "deterministic",
        "diagnostic",
        "ensemble",
    ]
    assert callable(run.deterministic)
    assert callable(run.diagnostic)
    assert callable(run.ensemble)


def test_run_does_not_import_coupling() -> None:
    """The execution engine never depends on the graph authoring layer.

    Checked in a fresh interpreter because an in-process import elsewhere in the
    test session would mask the regression this guards against.
    """
    source = (
        "import sys; import earth2studio.run.component, earth2studio.run.single; "
        "assert not [m for m in sys.modules if m.startswith('earth2studio.coupling')]"
    )
    assert subprocess.run([sys.executable, "-c", source]).returncode == 0  # noqa: S603


def test_one_and_many_components_use_same_plan_session_boundary() -> None:
    item = WorkItem(np.datetime64("2024-01-01"), np.timedelta64(2, "D"))
    schedule = _OneActivation()
    single: ExecutionPlan = _Plan((PortRef("atmos", "forecast"),), schedule)
    coupled: ExecutionPlan = _Plan(
        (PortRef("atmos", "forecast"), PortRef("ocean", "forecast")),
        schedule,
    )

    assert [event.component for event in _drive(single, item)] == ["atmos"]
    assert [event.component for event in _drive(coupled, item)] == ["atmos", "ocean"]


def test_snapshot_round_trips_through_plan_owned_codec() -> None:
    """Supervision moves bytes and a cursor; it never reads the payload."""
    item = WorkItem(np.datetime64("2024-01-01"), np.timedelta64(2, "D"))
    plan: ExecutionPlan = _Plan(
        (PortRef("atmos", "forecast"), PortRef("ocean", "forecast")),
        _OneActivation(),
    )

    session = plan.open(item)
    first = next(session.run())
    assert first.component == "atmos"
    assert session.checkpoint_boundary

    restored = plan.decode_snapshot(plan.encode_snapshot(session.snapshot()))
    assert restored.output_cursor[PortRef("atmos", "forecast")] == 1

    resumed = plan.open(item, restored)
    assert [event.component for event in resumed.run()] == ["ocean"]


def test_restore_rejects_an_incompatible_snapshot() -> None:
    """Resume fails closed rather than making a best-effort attempt."""
    item = WorkItem(np.datetime64("2024-01-01"), np.timedelta64(2, "D"))
    plan: ExecutionPlan = _Plan((PortRef("atmos", "forecast"),), _OneActivation())
    stale = RunSnapshot(
        compatibility=SnapshotCompatibility(
            schema_version=1,
            earth2studio_version="1.0.0a0",
            plan_identity="fixture-plan",
            component_versions={"atmos": "2"},
        ),
        output_cursor={},
        payload={"cursor": 0},
    )

    with pytest.raises(ValueError):
        plan.open(item, stale)


def test_plan_reports_capabilities_without_component_metadata() -> None:
    """Member sizing and resume policy read off the plan, not the graph."""
    plan: ExecutionPlan = _Plan((PortRef("atmos", "forecast"),), _OneActivation())
    ports: Mapping[PortRef, OutputPort] = plan.output_ports

    assert plan.supports_member_batching
    assert plan.checkpoint_capability is CheckpointCapability.SNAPSHOTTABLE
    assert ports[PortRef("atmos", "forecast")].variables == ("t2m",)
    item = WorkItem(np.datetime64("2024-01-01"), np.timedelta64(2, "D"))
    assert plan.external_requests(item) == ()


def test_adaptive_loop_runs_with_unknown_requests_and_stops_early() -> None:
    calls: list[int] = []
    closed: list[bool] = []

    def adaptive(item: WorkItem) -> Iterator[OutputEvent]:
        try:
            for step in range(10):
                # Stand-in for a conditional observation fetch.
                if step == 1:
                    calls.append(step)
                data = xr.DataArray(
                    [step], dims="variable", coords={"variable": ["t2m"]}
                )
                yield OutputEvent("custom", "forecast", item.time, data)
                if step == 2:
                    break
        finally:
            closed.append(True)

    ref = PortRef("custom", "forecast")
    signature = coord_array(("variable",), {"variable": ["t2m"]})
    ports = {ref: OutputPort(ref.port, ("t2m",), signature, _OneActivation())}
    plan: ExecutionPlan = LoopPlan(adaptive, output_ports=ports, identity="adaptive-v1")
    item = WorkItem(np.datetime64("2024-01-01"), np.timedelta64(10, "D"))
    assert plan.external_requests(item) is None
    assert calls == []  # Planning must not execute the loop.
    assert not plan.supports_member_batching
    assert plan.checkpoint_capability is CheckpointCapability.UNSUPPORTED
    assert len(_drive(plan, item)) == 3
    assert calls == [1]
    assert closed == [True]
    assert len(_drive(plan, item)) == 3  # State is local to each work item.

    session = plan.open(item)
    stream = session.run()
    next(stream)
    stream.close()
    assert len(closed) == 3
    assert not session.checkpoint_boundary
    with pytest.raises(NotImplementedError):
        session.snapshot()
    snapshot = RunSnapshot(COMPATIBILITY, {}, None)
    with pytest.raises(ValueError, match="restoring"):
        plan.open(item, snapshot)
    with pytest.raises(NotImplementedError):
        plan.encode_snapshot(snapshot)
    with pytest.raises(NotImplementedError):
        plan.decode_snapshot(b"")

    no_inputs = LoopPlan(
        adaptive, output_ports=ports, identity="known-v1", requests=lambda item: ()
    )
    assert no_inputs.external_requests(item) == ()
