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

"""Dataflow scheduler shared by planning and execution.

Time is lead time from the item's initial time, in integer nanoseconds. Ticks are
the lead times at which some stateful component's output chunk is due. At each
tick, components run in declaration order whenever every slot they need can be
resolved, until nothing more can run. A component whose slot waits on a frame
not yet published runs later, once it is; one that never can is an error.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import xarray as xr

from earth2studio.coupling._buffer import _Buffer
from earth2studio.coupling._nodes import _Node, _Slot
from earth2studio.coupling.connector import Connector, _split_label
from earth2studio.coupling.errors import CouplingError
from earth2studio.utils.time_statistics import source_lead_times


def _fmt(lead: int) -> str:
    """Format nanoseconds as a signed compact duration such as ``+6h``."""
    sign = "-" if lead < 0 else "+"
    value = abs(int(lead))
    for suffix, scale in (
        ("h", 3_600_000_000_000),
        ("m", 60_000_000_000),
        ("s", 10**9),
    ):
        if value % scale == 0:
            return f"{sign}{value // scale}{suffix}"
    return f"{sign}{value}ns"


def _window(modifier: str, lead: int, spacing: int) -> tuple[int, ...]:
    """Producer lead times a qualified label at ``lead`` reduces over."""
    leads = source_lead_times(
        modifier, np.timedelta64(lead, "ns"), np.timedelta64(spacing, "ns")
    )
    return tuple(int(v) for v in leads.astype("timedelta64[ns]").astype(np.int64))


@dataclass(frozen=True)
class _Read:
    """How one slot label at one lead time is obtained."""

    label: str
    lead: int
    variable: str  # producer variable
    modifier: str | None
    frames: tuple[int, ...]  # producer lead times read; empty when fetched


@dataclass(frozen=True)
class _SlotPlan:
    """What one slot receives for one call."""

    slot: _Slot
    leads: np.ndarray
    connector: Connector | None  # None: fetched from the slot's source
    reads: tuple[_Read, ...] = ()

    @property
    def fetched(self) -> tuple[_Read, ...]:
        """Reads of a connected slot that fall back to the slot's source."""
        return tuple(read for read in self.reads if not read.frames)


@dataclass
class _Event:
    """One component action within a work item."""

    ordinal: int  # 0 is initialization, then one per tick
    tick: int
    node: _Node
    action: str  # "initial condition", "initialize", "step" or "call"
    index: int  # chunk number, or the valid time of a diagnostic call
    leads: np.ndarray  # lead times published
    slots: tuple[int, ...]  # output slots published
    plans: tuple[_SlotPlan, ...] = ()
    states: tuple[_SlotPlan, ...] = ()  # state overwrites, one per connector
    frames: tuple[xr.DataArray, ...] | None = field(default=None, repr=False)

    @property
    def streams(self) -> tuple[str, ...]:
        """Streams this event publishes to."""
        return tuple(self.node.streams[i] for i in self.slots)


class _Executor(Protocol):
    def initial(self, node: _Node) -> tuple[Any, ...] | None:
        """Fetch every input slot of ``node``, in slot order; ``None`` when planning."""
        ...

    def fire(
        self, event: _Event, buffers: Mapping[str, _Buffer]
    ) -> tuple[xr.DataArray, ...] | None:
        """Run the call ``event`` describes; ``None`` when planning."""
        ...


class _Planner:
    """Executor that runs nothing, for planning and validation."""

    def initial(self, node: _Node) -> None:
        return None

    def fire(self, event: _Event, buffers: Mapping[str, _Buffer]) -> None:
        return None


class _Scheduler:
    """Schedule components over one work item, driving an executor."""

    def __init__(
        self,
        nodes: Mapping[str, _Node],
        connectors: Mapping[tuple[str, int], Connector],
        spacing: Mapping[str, int],
        states: Mapping[str, tuple[Connector, ...]] | None = None,
    ) -> None:
        self.nodes = nodes
        self.connectors = connectors
        self.spacing = spacing
        self.states = dict(states or {})
        self.triggered: dict[str, list[str]] = {name: [] for name in nodes}
        for (dst, _), connector in connectors.items():
            if not nodes[dst].stateful and connector.time == "exact":
                if dst not in self.triggered[connector.src]:
                    self.triggered[connector.src].append(dst)

    def ticks(self, final: int) -> list[int]:
        """Lead times at which some stateful component's chunk is due."""
        ticks: set[int] = set()
        for node in self.nodes.values():
            if node.stateful:
                k = 0
                while node.due(k) <= final:
                    ticks.add(node.due(k))
                    k += 1
        return sorted(ticks)

    def run(
        self,
        executor: _Executor,
        final: int,
        slack: int = 0,
        retention: Mapping[str, int] | None = None,
        ages: dict[str, int] | None = None,
    ) -> Iterator[_Event]:
        """Yield every event of one item through lead time ``final``.

        ``slack`` excuses chunks due within that much of ``final`` from the
        completeness check, for planning over a truncated horizon. ``retention``
        bounds how far back each producer's frames are kept; ``ages`` collects the
        oldest frame each producer had to supply, to derive it.
        """
        run = _Run(self, executor, final, ages)
        yield from run.initialize()
        yield from run.drain(0, 0)
        for ordinal, tick in enumerate(self.ticks(final), start=1):
            yield from run.drain(ordinal, tick)
            if retention is not None:
                for name, buffer in run.buffers.items():
                    buffer.prune(tick - retention.get(name, 0))
        run.check_complete(final - slack)


class _Run:
    """Mutable state of one scheduled item."""

    def __init__(
        self,
        scheduler: _Scheduler,
        executor: _Executor,
        final: int,
        ages: dict[str, int] | None,
    ) -> None:
        self.s = scheduler
        self.executor = executor
        self.final = final
        self.ages = ages
        self.buffers = {name: _Buffer() for name in scheduler.nodes}
        self.next_chunk = {
            name: 0 for name, node in scheduler.nodes.items() if node.stateful
        }
        self.pending: dict[str, list[int]] = {
            name: [] for name, node in scheduler.nodes.items() if not node.stateful
        }
        self.queued: dict[str, set[int]] = {name: set() for name in self.pending}

    # -- publication -------------------------------------------------------

    def _trigger(self, name: str, leads: np.ndarray) -> None:
        for consumer in self.s.triggered[name]:
            for lead in leads:
                lead = int(lead)
                if lead <= self.final and lead not in self.queued[consumer]:
                    self.queued[consumer].add(lead)
                    bisect.insort(self.pending[consumer], lead)

    def _publish(self, event: _Event) -> None:
        node, frames = event.node, event.frames
        if frames is not None and len(frames) != len(event.slots):
            raise CouplingError(
                f"Component {node.name!r} returned {len(frames)} outputs, expected "
                f"{len(event.slots)}"
            )
        for position, slot in enumerate(event.slots):
            frame = None if frames is None else frames[position]
            if frame is not None:
                actual = frame.coords["lead_time"].values.astype("timedelta64[ns]")
                if not np.array_equal(actual.astype(np.int64), event.leads):
                    raise CouplingError(
                        f"Component {node.name!r} output slot {slot} has lead times "
                        f"{actual}, expected {event.leads.astype('timedelta64[ns]')}"
                    )
            variables = frozenset(node.exports(slot))
            for i, lead in enumerate(event.leads):
                part = None if frame is None else frame.isel(lead_time=[i])
                self.buffers[node.name].add(int(lead), variables, event.ordinal, part)
        self._trigger(node.name, event.leads)

    # -- initialization ----------------------------------------------------

    def initialize(self) -> Iterator[_Event]:
        """Fetch every stateful component's inputs and publish initial conditions."""
        for node in self.s.nodes.values():
            if not node.stateful:
                continue
            arrays = self.executor.initial(node)
            buffer = self.buffers[node.name]
            for i, slot in enumerate(node.inputs):
                if slot.frame or slot.static:
                    continue
                for j, lead in enumerate(slot.leads):
                    frame = None if arrays is None else arrays[i].isel(lead_time=[j])
                    buffer.add(int(lead), frozenset(slot.labels), 0, frame)
            leads = np.array([node.ic_lead], dtype=np.int64)
            event = _Event(0, 0, node, "initial condition", 0, leads, node.ic_slots)
            if arrays is not None:
                event.frames = tuple(
                    arrays[i].isel(lead_time=[-1]) for i in node.ic_slots
                )
            if node.ic_slots:
                self._trigger(node.name, leads)
            yield event

    # -- planning ----------------------------------------------------------

    def _resolve(
        self,
        connector: Connector,
        labels: tuple[str, ...],
        leads: np.ndarray,
        ordinal: int,
        fallback: bool,
    ) -> tuple[_Read, ...] | None:
        """Plan reads of ``labels`` at ``leads``, or ``None`` if one must wait.

        Frames valid at or before the item time that the producer does not hold
        fall back to the slot's source when ``fallback`` is set.
        """
        buffer = self.buffers[connector.src]
        reads = []
        for lead in leads:
            lead = int(lead)
            for label in labels:
                base, modifier = _split_label(label)
                variable = connector.source_variable(base)
                frames: tuple[int, ...] | None = None
                if modifier is not None:
                    window = _window(modifier, lead, self.s.spacing[connector.src])
                    if all(buffer.has(variable, f) for f in window):
                        frames = window
                elif connector.time == "exact":
                    if buffer.has(variable, lead):
                        frames = (lead,)
                else:
                    held = buffer.latest(variable, lead, ordinal)
                    if held is not None:
                        frames = (held,)
                if frames is None:
                    if not (fallback and lead <= 0):
                        return None
                    frames = ()
                reads.append(_Read(label, lead, variable, modifier, frames))
        return tuple(reads)

    def _plans(
        self, node: _Node, slots: tuple[_Slot, ...], index: int, ordinal: int
    ) -> tuple[_SlotPlan, ...] | None:
        plans = []
        for slot in slots:
            leads = node.required(slot, index)
            connector = None
            if not slot.frame and not slot.static:
                connector = self.s.connectors.get((node.name, slot.index))
            if connector is None:
                plans.append(_SlotPlan(slot, leads, None))
                continue
            reads = self._resolve(connector, slot.labels, leads, ordinal, node.stateful)
            if reads is None:
                return None
            plans.append(_SlotPlan(slot, leads, connector, reads))
        return tuple(plans)

    def _state_plans(
        self, node: _Node, k: int, ordinal: int
    ) -> tuple[_SlotPlan, ...] | None:
        """Overwrites of ``node``'s state before chunk ``k``; unsupplied values stay."""
        plans = []
        for connector in self.s.states.get(node.name, ()):
            slot = node.inputs[connector.slot]
            leads = node.state_leads(slot, k)
            labels = tuple(connector.variables or {})
            reads = self._resolve(connector, labels, leads, ordinal, True)
            if reads is None:
                return None
            plans.append(_SlotPlan(slot, leads, connector, reads))
        return tuple(plans)

    def _fire(self, event: _Event) -> _Event:
        if self.ages is not None:
            for plan in event.plans + event.states:
                for read in plan.reads:
                    if read.frames and plan.connector is not None:
                        src = plan.connector.src
                        age = event.tick - min(read.frames)
                        self.ages[src] = max(self.ages.get(src, 0), age)
        event.frames = self.executor.fire(event, self.buffers)
        self._publish(event)
        return event

    def drain(self, ordinal: int, tick: int) -> Iterator[_Event]:
        """Run everything that can run at ``tick`` until nothing more can."""
        progress = True
        while progress:
            progress = False
            for name, node in self.s.nodes.items():
                if node.stateful:
                    while node.due(self.next_chunk[name]) <= min(tick, self.final):
                        k = self.next_chunk[name]
                        plans = self._plans(node, node.slots_for(k), k, ordinal)
                        states = self._state_plans(node, k, ordinal)
                        if plans is None or states is None:
                            break
                        action = "initialize" if k == 0 else "step"
                        event = _Event(
                            ordinal,
                            tick,
                            node,
                            action,
                            k,
                            node.chunk(k),
                            tuple(range(len(node.outputs))),
                            plans,
                            states,
                        )
                        yield self._fire(event)
                        self.next_chunk[name] = k + 1
                        progress = True
                else:
                    for valid in list(self.pending[name]):
                        plans = self._plans(node, node.inputs, valid, ordinal)
                        if plans is None:
                            continue
                        self.pending[name].remove(valid)
                        leads = np.array([valid], dtype=np.int64)
                        event = _Event(
                            ordinal,
                            tick,
                            node,
                            "call",
                            valid,
                            leads,
                            tuple(range(len(node.outputs))),
                            plans,
                        )
                        yield self._fire(event)
                        progress = True

    # -- completeness --------------------------------------------------------

    def _blocker(self, name: str) -> tuple[str, str, int] | None:
        """First ``(producer, label, lead)`` component ``name`` is waiting for."""
        node = self.s.nodes[name]
        if node.stateful:
            k = self.next_chunk[name]
            slots, index = node.slots_for(k), k
            for state in self.s.states.get(name, ()):
                slot = node.inputs[state.slot]
                for lead in node.state_leads(slot, k):
                    single = np.array([lead], dtype=np.int64)
                    for label in state.variables or {}:
                        if self._resolve(state, (label,), single, 1 << 62, True):
                            continue
                        return state.src, label, int(lead)
        else:
            if not self.pending[name]:
                return None
            slots, index = node.inputs, self.pending[name][0]
        for slot in slots:
            connector = self.s.connectors.get((name, slot.index))
            if connector is None or slot.frame or slot.static:
                continue
            for lead in node.required(slot, index):
                single = np.array([lead], dtype=np.int64)
                for label in slot.labels:
                    reads = self._resolve(
                        connector, (label,), single, 1 << 62, node.stateful
                    )
                    if reads is None:
                        return connector.src, label, int(lead)
        return None

    def check_complete(self, limit: int) -> None:
        """Raise if a stateful component never produced a chunk due by ``limit``."""
        stuck = [
            name
            for name, node in self.s.nodes.items()
            if node.stateful and node.due(self.next_chunk[name]) <= limit
        ]
        if not stuck:
            return
        name = stuck[0]
        node = self.s.nodes[name]
        due = node.due(self.next_chunk[name])
        path = [name]
        current = name
        for _ in range(len(self.s.nodes) + 1):
            blocked = self._blocker(current)
            if blocked is None:
                break
            current = blocked[0]
            if self.s.nodes[current].stateful and current not in stuck:
                break  # waiting on a component that simply has not run yet
            if current in path:
                cycle = path[path.index(current) :] + [current]
                raise CouplingError(
                    f"Coupling cycle {' -> '.join(cycle)} at {_fmt(due)}: each "
                    "component waits for a frame the next has not produced. Set "
                    "time='hold' on one of these connectors, or declare that "
                    "forcing at an earlier lead time (C11)"
                )
            path.append(current)
        blocked = self._blocker(name)
        detail = ""
        if blocked is not None:
            producer, label, lead = blocked
            detail = (
                f": {producer!r} publishes no frames for {label!r} valid at "
                f"{_fmt(lead)}"
            )
        raise CouplingError(
            f"Component {name!r} cannot produce its output due at {_fmt(due)}"
            f"{detail}. Use time='hold' to take the latest earlier frame, or "
            "check the final lead time and the producer's cadence (C11)"
        )
