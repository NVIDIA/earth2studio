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

"""Single-model execution using shared component metadata and stepping primitives.

The direct loop stays independent of graph scheduling. Stateless output transforms
compose without subclassing; bespoke control flow belongs in a custom run session
or ``LoopPlan``.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

import earth2studio
from earth2studio.data import DataSource, fetch_data
from earth2studio.models.px.base import PrognosticModel
from earth2studio.run.component import (
    ComponentSnapshot,
    PrognosticComponent,
    PrognosticComponentSession,
    spec_fingerprint,
)
from earth2studio.run.session import (
    CheckpointCapability,
    OutputEvent,
    OutputPort,
    PortRef,
    ResolvedRequest,
    RunSnapshot,
    SnapshotCompatibility,
    WorkItem,
)

SNAPSHOT_SCHEMA_VERSION = 0


@dataclass(frozen=True)
class OutputTransform:
    """A stateless transformation of published outputs, independent of execution.

    Transforms compose in order. They must not mutate borrowed input arrays or
    retain mutable simulation state. Returned values never feed back into the
    model. A graph executor may reuse the same transform as a diagnostic.

    Parameters
    ----------
    apply : Callable[[Mapping[str, xr.DataArray]], Mapping[str, xr.DataArray]]
        Transform port-keyed values, returning every declared output port once.
    identity : str
        Stable identity including configuration; change it when behavior changes.
    ports : Callable[[tuple[OutputPort, ...]], tuple[OutputPort, ...]], optional
        Transform declarations when outputs change, by default unchanged
    """

    apply: Callable[[Mapping[str, xr.DataArray]], Mapping[str, xr.DataArray]]
    identity: str
    ports: Callable[[tuple[OutputPort, ...]], tuple[OutputPort, ...]] | None = None


class SingleModelSession:
    """Roll one prognostic component forward for one work item.

    Publishes the initial condition and then one step per activation, up to and
    including the work item's horizon -- the same steps ``run.deterministic``
    writes. The snapshot payload is the component's
    :class:`~earth2studio.run.component.ComponentSnapshot`.

    Parameters
    ----------
    plan : SingleModelPlan
        Plan that opened this session; supplies the component and source.
    item : WorkItem
        Work item this session simulates.
    """

    def __init__(self, plan: SingleModelPlan, item: WorkItem) -> None:
        self.plan = plan
        self.item = item
        self.published = 0
        self.component: PrognosticComponentSession | None = None
        self._mid_step = False
        horizon: np.timedelta64 = item.horizon.astype("timedelta64[s]")
        if horizon % plan.cadence.step != np.timedelta64(0, "s"):
            raise ValueError(
                f"Horizon {item.horizon} is not a multiple of the model step "
                f"{plan.cadence.step}"
            )
        self.nsteps = int(horizon // plan.cadence.step)

    def initial_condition(self) -> xr.DataArray:
        """Fetch this work item's initial condition from the plan's source."""
        requirement = self.plan.component.initial_condition
        return fetch_data(
            self.plan.source,
            time=np.array([self.item.time]),
            variable=np.array(requirement.variables),
            lead_time=np.array(requirement.lead_offsets),
        )

    @property
    def checkpoint_boundary(self) -> bool:
        """Whether a snapshot taken now would be restorable."""
        return (
            self.plan.checkpoint_capability is CheckpointCapability.SNAPSHOTTABLE
            and self.component is not None
            and not self._mid_step
        )

    def run(self) -> Generator[OutputEvent, None, None]:
        """Publish each step until the work item's horizon is reached."""
        if self.component is None:
            self.component = self.plan.component.open(
                {"initial_condition": self.initial_condition()}
            )
        cadence = self.plan.cadence
        start = self.item.time + self.published * cadence.step
        stop = self.item.time + (self.nsteps + 1) * cadence.step
        try:
            for activation in cadence.iter_between(self.item.time, start, stop):
                self._mid_step = True
                step = self.component.step(activation, {})
                yield from self._publish(activation, step["forecast"])
        finally:
            self.component.close()

    def _publish(
        self, activation: np.datetime64, x: xr.DataArray
    ) -> Iterator[OutputEvent]:
        self.published += 1
        values: Mapping[str, xr.DataArray] = {"forecast": x}
        for transform in self.plan.transforms:
            values = transform.apply(values)
        expected = {ref.port for ref in self.plan.output_ports}
        if set(values) != expected:
            raise ValueError("Output transform ports do not match their declarations")
        outputs = list(values.items())
        for index, (port, data) in enumerate(outputs):
            self._mid_step = index < len(outputs) - 1
            yield OutputEvent(self.plan.name, port, activation, data)
        self._mid_step = False

    def snapshot(self) -> RunSnapshot:
        """Capture the published-step cursor and the component's state."""
        if not self.checkpoint_boundary or self.component is None:
            raise RuntimeError("Snapshot requested outside a checkpoint boundary")
        return RunSnapshot(
            compatibility=self.plan.compatibility,
            output_cursor={ref: self.published for ref in self.plan.output_ports},
            payload=self.component.snapshot(),
        )

    def restore(self, payload: object) -> None:
        """Resume from a payload produced by :meth:`snapshot`, without refetching."""
        if not isinstance(payload, ComponentSnapshot):
            raise TypeError("Snapshot payload was not produced by this session")
        self.component = self.plan.component.open({})
        self.component.restore(payload)
        self.published = payload.step_index


class SingleModelPlan:
    """Execution plan for one prognostic model fed by one data source.

    Item-agnostic: build once, open per work item. Identity, snapshot
    compatibility, cadence, output ports, source requests, and capabilities come
    from the component's spec and output transforms.

    Parameters
    ----------
    model : PrognosticModel
        Model to roll out.
    source : DataSource
        Source of initial conditions. Live, cached, or predownloaded; the plan
        does not distinguish.
    transforms : tuple[OutputTransform, ...], optional
        Stateless output transforms applied in order, by default ()
    name : str, optional
        Component name used on published events, by default ``"model"``.
    """

    def __init__(
        self,
        model: PrognosticModel,
        source: DataSource,
        *,
        transforms: tuple[OutputTransform, ...] = (),
        name: str = "model",
    ) -> None:
        self.model = model
        self.source = source
        self.transforms = transforms
        self.name = name
        self.component = PrognosticComponent(model, name)
        self.cadence = self.component.cadence

        spec = self.component.spec
        ports = spec.outputs
        for transform in transforms:
            if transform.ports is not None:
                ports = transform.ports(ports)
        if not ports or len({port.name for port in ports}) != len(ports):
            raise ValueError("Output ports must be nonempty and uniquely named")
        self._output_ports = {PortRef(name, port.name): port for port in ports}
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        declaration = "|".join(
            [
                model_type,
                repr(tuple(transform.identity for transform in transforms)),
                spec_fingerprint(spec),
                ";".join(
                    f"{port.name}:{','.join(port.variables)}"
                    for port in self._output_ports.values()
                ),
            ]
        )
        self._identity = hashlib.sha256(declaration.encode()).hexdigest()[:16]
        self.compatibility = SnapshotCompatibility(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            earth2studio_version=earth2studio.__version__,
            plan_identity=self._identity,
            component_versions={name: model_type},
        )

    @property
    def identity(self) -> str:
        """Return stable identity for progress and snapshot compatibility."""
        return self._identity

    @property
    def output_ports(self) -> Mapping[PortRef, OutputPort]:
        """Return declared output streams for routing and shard ownership."""
        return self._output_ports

    @property
    def checkpoint_capability(self) -> CheckpointCapability:
        """Return whether sessions from this plan can resume."""
        return self.component.spec.checkpoint_capability

    @property
    def supports_member_batching(self) -> bool:
        """Whether one session may carry several ensemble members."""
        return False  # Member perturbation is not implemented by the default loop.

    def external_requests(self, item: WorkItem) -> tuple[ResolvedRequest, ...]:
        """Return the concrete source requests one work item will make."""
        return tuple(
            ResolvedRequest(
                requirement.variables, requirement.signature, item.time + lead
            )
            for requirement in self.component.spec.inputs
            if requirement.phase == "initialize"
            for lead in requirement.lead_offsets
        )

    def describe(self) -> str:
        """Render the plan without executing it."""
        ports = ", ".join(
            f"{ref.port}[{','.join(port.variables)}]"
            for ref, port in self._output_ports.items()
        )
        return (
            f"{self.name}: {type(self.model).__qualname__} every "
            f"{self.cadence.step} -> {ports} "
            f"({self.checkpoint_capability.value})"
        )

    def open(
        self, item: WorkItem, snapshot: RunSnapshot | None = None
    ) -> SingleModelSession:
        """Open one simulation, optionally restored from a snapshot."""
        session = SingleModelSession(self, item)
        if snapshot is not None:
            if not snapshot.compatibility.is_compatible_with(self.compatibility):
                raise ValueError(
                    f"Snapshot for plan {snapshot.compatibility.plan_identity} "
                    f"cannot resume plan {self.identity}"
                )
            session.restore(snapshot.payload)
        return session

    def encode_snapshot(self, snapshot: RunSnapshot) -> bytes:
        """Serialize a snapshot, including its payload."""
        return pickle.dumps(snapshot)

    def decode_snapshot(self, raw: bytes) -> RunSnapshot:
        """Deserialize a snapshot produced by :meth:`encode_snapshot`.

        Snapshots are local artifacts this process wrote, under the same trust
        model as model checkpoints; never decode one from an untrusted origin.
        """
        return pickle.loads(raw)  # noqa: S301
