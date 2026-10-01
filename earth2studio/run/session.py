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

"""The shared boundary between work supervision and one simulation's execution.

This module is the whole contract a custom execution strategy needs. Four
types define it -- :class:`ExecutionPlan`, :class:`RunSession`,
:class:`OutputEvent`, and :class:`RunSnapshot` -- plus the small types they
reference. Supervision (work distribution, member grouping, retries, resume,
IO routing and commit) is built on top of these; it never needs to know what
produced a plan.

A :class:`RunSession` is hand-implementable. A caller with an execution pattern
that the declarative component graph cannot express writes these two methods
directly and still gets distribution, resume, and output handling for free.
``LoopPlan`` wraps an ordinary generator when mid-run resume is not needed.
Implementing a supervision subclass should never be the way to add an execution
pattern.

Like all of :mod:`earth2studio.run`, this module is execution engine and must
not import the authoring layer, :mod:`earth2studio.coupling`. See ``dev/spec/EXECUTION_CONTRACT_SPEC.md`` for the behavioral contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import xarray as xr

from earth2studio.run.schedules import Schedule
from earth2studio.utils.type import CoordinateSystem


@dataclass(frozen=True)
class WorkItem:
    """One forecast reference time, horizon, and optional ensemble member group."""

    time: np.datetime64
    horizon: np.timedelta64
    member_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PortRef:
    """Identify one named output stream on one named component."""

    component: str
    port: str


@dataclass(frozen=True)
class OutputPort:
    """Describe one named stream of semantically valid component outputs.

    Every position published on a port is valid; a port never carries structural
    filler added only to reconcile unlike schedules. Outputs that differ in grid,
    cadence, or valid-time pattern belong on separate ports. The schedule lists
    possible production times, not a required event count; a run may stop early.
    """

    name: str
    variables: tuple[str, ...]
    signature: CoordinateSystem
    schedule: Schedule


@dataclass(frozen=True)
class ResolvedRequest:
    """A concrete external field request after binding and time expansion.

    Predownload and cache-planning tooling consumes these without opening a
    session or loading model weights.
    """

    variables: tuple[str, ...]
    signature: CoordinateSystem
    valid_time: np.datetime64


@dataclass(frozen=True)
class OutputEvent:
    """A component product available to the graph at ``produced_at``.

    ``produced_at`` is graph availability time. Forecast reference time, lead
    time, and valid time remain coordinates on ``data``. The buffer behind
    ``data`` must stay unchanged while supervision or an IO backend still owns
    the event; freezing this dataclass does not make its DataArray immutable.
    """

    component: str
    port: str
    produced_at: np.datetime64
    data: xr.DataArray


class CheckpointCapability(Enum):
    """Whether execution can be interrupted and resumed from a snapshot."""

    STATELESS = "stateless"
    SNAPSHOTTABLE = "snapshottable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SnapshotCompatibility:
    """Identity a snapshot must match before it may be restored.

    Supervision reads this without decoding the snapshot payload, because it is
    what decides whether a stored snapshot may be handed back at all. The first
    guarantee is same-version continuation, not cross-version migration.
    """

    schema_version: int
    earth2studio_version: str
    plan_identity: str
    component_versions: Mapping[str, str]

    def is_compatible_with(self, other: SnapshotCompatibility) -> bool:
        """Return whether a snapshot recorded as ``self`` may resume ``other``."""
        return (
            self.schema_version == other.schema_version
            and self.earth2studio_version == other.earth2studio_version
            and self.plan_identity == other.plan_identity
            and dict(self.component_versions) == dict(other.component_versions)
        )


@dataclass(frozen=True)
class RunSnapshot:
    """Resume state for one work item: a compatibility key, a cursor, a payload.

    ``payload`` is opaque to supervision. Scientific state -- component state,
    RNG state, connector history, graph event position -- lives inside it and is
    owned entirely by whatever produced the session. Supervision persists and
    returns the payload; it never inspects or constructs one.

    ``output_cursor`` is the one part with cross-boundary meaning, because
    supervision owns IO and commit. After a restore, a session must re-emit no
    event already committed at that cursor and skip none that was not.
    """

    compatibility: SnapshotCompatibility
    output_cursor: Mapping[PortRef, int]
    payload: object


class RunSession(Protocol):
    """Execute one complete simulation for one work item.

    The hand-implementable extension point. A single-model forecast, a coupled
    graph, and a bespoke rollout are all just implementations of these members.
    """

    @property
    def checkpoint_boundary(self) -> bool:
        """Whether a snapshot taken now would be restorable.

        Read between yielded events. Supervision chooses checkpoint *frequency*,
        because it knows retry cost and wall-clock budget; the session declares
        only where a boundary is *legal*. A session whose plan reports
        ``CheckpointCapability.UNSUPPORTED`` always returns ``False``.
        """
        ...

    def run(self) -> Iterator[OutputEvent]:
        """Yield outputs in production order until the work item completes."""
        ...

    def snapshot(self) -> RunSnapshot:
        """Capture complete resume state. Only valid at a checkpoint boundary."""
        ...


class ExecutionPlan(Protocol):
    """A validated, inspectable plan, reusable across work items.

    A plan is item-agnostic: it is built once and opened once per work item, so
    supervision can compile before distributing work rather than per item. Data
    sources are bound when the plan is built; field values are fetched by the
    session, never handed in by supervision. Anything that depends on a specific
    item -- concrete requests, the simulation itself -- takes the item as an
    argument.

    Plans declare output schemas and identity before execution. Declarative
    plans also validate static inputs and schedules; custom loops may resolve
    data-dependent inputs at runtime. Neither requires a component graph.

    The capability properties exist so that supervision can size member groups
    and decide resume policy without walking component metadata or branching on
    graph topology.
    """

    @property
    def identity(self) -> str:
        """Return stable identity for progress and snapshot compatibility."""
        ...

    @property
    def output_ports(self) -> Mapping[PortRef, OutputPort]:
        """Return declared output streams for routing and shard ownership."""
        ...

    def external_requests(self, item: WorkItem) -> tuple[ResolvedRequest, ...] | None:
        """Return a complete request set, or ``None`` when not knowable upfront.

        Includes initial conditions as well as step-time forcing. Predownload and
        cache planning call this per item without opening a session. An empty
        tuple means no requests; ``None`` disables complete predownload, not
        execution. A predownload-only driver must reject ``None``.
        """
        ...

    @property
    def checkpoint_capability(self) -> CheckpointCapability:
        """Return the weakest capability across everything this plan executes."""
        ...

    @property
    def supports_member_batching(self) -> bool:
        """Whether one session may carry several ensemble members.

        ``False`` forces supervision to build member groups of size one.
        """
        ...

    def describe(self) -> str:
        """Render the compiled plan without executing it."""
        ...

    def open(
        self,
        item: WorkItem,
        snapshot: RunSnapshot | None = None,
    ) -> RunSession:
        """Open one simulation, optionally restored from a snapshot.

        Must reject an incompatible snapshot rather than make a best-effort
        attempt; see :meth:`SnapshotCompatibility.is_compatible_with`.
        """
        ...

    def encode_snapshot(self, snapshot: RunSnapshot) -> bytes:
        """Serialize a snapshot, including its opaque payload.

        The plan owns the codec because the plan owns the payload. Supervision
        stores and retrieves bytes and chooses when to do so.
        """
        ...

    def decode_snapshot(self, raw: bytes) -> RunSnapshot:
        """Deserialize a snapshot previously produced by :meth:`encode_snapshot`."""
        ...


class _LoopSession:
    checkpoint_boundary = False

    def __init__(
        self, run: Callable[[WorkItem], Iterator[OutputEvent]], item: WorkItem
    ) -> None:
        self._run = run
        self._item = item

    def run(self) -> Iterator[OutputEvent]:
        return self._run(self._item)

    def snapshot(self) -> RunSnapshot:
        raise NotImplementedError("LoopPlan does not support checkpointing")


class LoopPlan:
    """Wrap an ordinary generator as an execution plan.

    No component or graph declarations are needed. Each session invokes ``run``
    with one work item. Inputs may be fetched conditionally and the loop may
    stop early. Mid-run resume and member batching are deliberately unsupported;
    implement ``ExecutionPlan`` and ``RunSession`` when those are needed.

    Parameters
    ----------
    run : Callable[[WorkItem], Iterator[OutputEvent]]
        Generator function that owns one simulation and releases its resources
        on completion or close. Mutable run state belongs inside this function.
    output_ports : Mapping[PortRef, OutputPort]
        Declared output schemas. Events must match these declarations.
    identity : str
        Stable caller-owned identity. Change it when code, configuration, or
        declarations change; Python callables cannot be fingerprinted reliably.
    requests : Callable[[WorkItem], tuple[ResolvedRequest, ...] | None], optional
        Complete external request set per item, by default unknown (``None``)
    """

    checkpoint_capability = CheckpointCapability.UNSUPPORTED
    supports_member_batching = False

    def __init__(
        self,
        run: Callable[[WorkItem], Iterator[OutputEvent]],
        *,
        output_ports: Mapping[PortRef, OutputPort],
        identity: str,
        requests: (
            Callable[[WorkItem], tuple[ResolvedRequest, ...] | None] | None
        ) = None,
    ) -> None:
        self._run = run
        self.output_ports = dict(output_ports)
        self.identity = identity
        self._requests = requests

    def external_requests(self, item: WorkItem) -> tuple[ResolvedRequest, ...] | None:
        """Return complete requests, or ``None`` if they depend on execution."""
        return None if self._requests is None else self._requests(item)

    def describe(self) -> str:
        """Describe the loop without invoking it."""
        return f"LoopPlan({self.identity})"

    def open(self, item: WorkItem, snapshot: RunSnapshot | None = None) -> RunSession:
        """Open a fresh loop; reject snapshots because resume is unsupported."""
        if snapshot is not None:
            raise ValueError("LoopPlan does not support restoring snapshots")
        return _LoopSession(self._run, item)

    def encode_snapshot(self, snapshot: RunSnapshot) -> bytes:
        """Reject checkpoint encoding; this plan cannot resume."""
        raise NotImplementedError("LoopPlan does not support checkpointing")

    def decode_snapshot(self, raw: bytes) -> RunSnapshot:
        """Reject checkpoint decoding; this plan cannot resume."""
        raise NotImplementedError("LoopPlan does not support checkpointing")
