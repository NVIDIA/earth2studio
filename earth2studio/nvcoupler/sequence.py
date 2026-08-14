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

"""Run sequence: the ordered schedule of component runs and exchanges.

The NUOPC runSeq analog. A RunSequence is a list of Slots, each with an
interval and an ordered action list; the Driver executes a slot's actions,
in order, at every clock time aligned with the slot interval. Coupling
semantics are pure ordering: a ConnectAction placed *before* the destination
component's RunAction in the same slot means the destination sees the
source's previous state (lagged / NUOPC-explicit coupling); placed *after*,
it sees the state just produced (sequential coupling).

Sequences are derived, not required: :func:`derive_sequence` lays out the
canonical schedule from the coupling graph (components + connections) alone,
and the Driver calls it when no sequence is given. Hand-written sequences —
via the string DSL mirroring NUOPC's runSeq — are the override for schedules
the graph cannot express::

    @6h
      atmos -> med          # ConnectAction (exports of atmos -> imports of med)
      ocean -> atmos        # lagged: before atmos runs
      atmos                 # RunAction
    @48h
      med.compute           # MediateAction
      med -> ocean
      ocean
    @
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .clock import DeltaLike, as_timedelta, fmt_timedelta, is_multiple
from .errors import CadenceError, SequenceError, suggest

if TYPE_CHECKING:
    from .component import Component
    from .connector import Connector


@dataclass(frozen=True)
class RunAction:
    component: str

    def __str__(self) -> str:
        return self.component


@dataclass(frozen=True)
class ConnectAction:
    src: str
    dst: str

    def __str__(self) -> str:
        return f"{self.src} -> {self.dst}"


@dataclass(frozen=True)
class MediateAction:
    mediator: str
    phase: str = "compute"

    def __str__(self) -> str:
        return f"{self.mediator}.{self.phase}"


Action = RunAction | ConnectAction | MediateAction


def _fmt_interval(d: np.timedelta64) -> str:
    """Format a slot interval for the DSL: whole hours as NUOPC-style '@48h',
    otherwise fall back to fmt_timedelta so sub-hour slots ('@90m', '@30m')
    round-trip exactly instead of truncating to whole hours."""
    ns = int(d.astype("timedelta64[ns]").astype(np.int64))
    hour = 3_600_000_000_000
    if ns % hour == 0:
        return f"{ns // hour}h"
    return fmt_timedelta(d)


@dataclass
class Slot:
    interval: np.timedelta64
    actions: list[Action] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.interval = as_timedelta(self.interval)


@dataclass
class RunSequence:
    slots: list[Slot]

    def components_run(self) -> set[str]:
        return {
            a.component
            for s in self.slots
            for a in s.actions
            if isinstance(a, RunAction)
        } | {
            a.mediator
            for s in self.slots
            for a in s.actions
            if isinstance(a, MediateAction)
        }

    def connections(self) -> list[ConnectAction]:
        return [
            a for s in self.slots for a in s.actions if isinstance(a, ConnectAction)
        ]

    def validate(self, components: dict, dt: DeltaLike) -> None:
        """Check name resolution, cadence alignment, and completeness."""
        dt = as_timedelta(dt)
        names = set(components)

        def check_name(name: str, what: str) -> None:
            if name not in names:
                raise SequenceError(
                    f"Run sequence references unknown {what} {name!r}."
                    + suggest(name, names)
                    + f" Known components: {sorted(names)}"
                )

        for slot in self.slots:
            if not is_multiple(slot.interval, dt):
                raise CadenceError("Run-sequence slot", str(slot.interval), str(dt))
            for action in slot.actions:
                if isinstance(action, RunAction):
                    check_name(action.component, "component")
                    comp = components[action.component]
                    if comp.timestep != slot.interval:
                        raise CadenceError(
                            f"Component {action.component!r} (timestep "
                            f"{comp.timestep}) scheduled in a slot of",
                            str(slot.interval),
                            str(comp.timestep),
                        )
                elif isinstance(action, ConnectAction):
                    check_name(action.src, "connector source")
                    check_name(action.dst, "connector destination")
                else:
                    check_name(action.mediator, "mediator")
                    med = components[action.mediator]
                    if med.timestep != slot.interval:
                        raise CadenceError(
                            f"Mediator {action.mediator!r} (timestep "
                            f"{med.timestep}) scheduled in a slot of",
                            str(slot.interval),
                            str(med.timestep),
                        )
        # every component must run somewhere
        idle = names - self.components_run()
        if idle:
            raise SequenceError(
                f"Components never run by the sequence: {sorted(idle)} — add a "
                "RunAction (bare component name) to a slot matching their timestep"
            )

    def __str__(self) -> str:
        lines = []
        for slot in self.slots:
            lines.append(f"@{_fmt_interval(slot.interval)}")
            lines.extend(f"  {a}" for a in slot.actions)
        lines.append("@")
        return "\n".join(lines)


_SLOT_RE = re.compile(r"^@(\S+)?$")
_CONNECT_RE = re.compile(r"^(\w[\w.-]*)\s*->\s*(\w[\w.-]*)$")
_MEDIATE_RE = re.compile(r"^(\w[\w-]*)\.(\w+)$")
_RUN_RE = re.compile(r"^(\w[\w-]*)$")


def parse_run_sequence(text: str) -> RunSequence:
    """Parse the NUOPC-flavored DSL into a RunSequence."""
    slots: list[Slot] = []
    current: Slot | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if m := _SLOT_RE.match(line):
            if m.group(1):  # "@6h" opens a slot; bare "@" closes the sequence
                try:
                    current = Slot(as_timedelta(m.group(1)))
                except ValueError as e:
                    raise SequenceError(f"Line {lineno}: {e}") from None
                slots.append(current)
            else:
                current = None
            continue
        if current is None:
            raise SequenceError(
                f"Line {lineno}: action {line!r} outside any @interval slot"
            )
        if m := _CONNECT_RE.match(line):
            current.actions.append(ConnectAction(m.group(1), m.group(2)))
        elif m := _MEDIATE_RE.match(line):
            current.actions.append(MediateAction(m.group(1), m.group(2)))
        elif m := _RUN_RE.match(line):
            current.actions.append(RunAction(m.group(1)))
        else:
            raise SequenceError(
                f"Line {lineno}: cannot parse {line!r} — expected 'name', "
                "'src -> dst', or 'mediator.compute'"
            )
    if not slots:
        raise SequenceError("Run sequence is empty")
    return RunSequence(slots)


# ---------------------------------------------------------------------------
# derive_sequence(): the schedule implied by the coupling graph
# ---------------------------------------------------------------------------
def _toposort(
    nodes: list[str],
    edges: list[tuple[str, str]],
    interval: np.timedelta64,
) -> list[str]:
    """Order `nodes` so every sequential edge's source runs before its
    destination, keeping declaration order among unconstrained nodes."""
    node_set = set(nodes)
    deps: dict[str, set[str]] = {n: set() for n in nodes}
    for src, dst in edges:
        if src in node_set and dst in node_set:
            deps[dst].add(src)
    out: list[str] = []
    placed: set[str] = set()
    pending = list(nodes)
    while pending:
        ready = [n for n in pending if deps[n] <= placed]
        if not ready:
            cycle_edges = [
                (s, d) for s, d in edges if s in pending and d in pending
            ]
            raise SequenceError(
                f"Sequential coupling cycle among components {sorted(pending)} "
                f"at cadence {fmt_timedelta(interval)}: connections "
                f"{cycle_edges} require each destination to see state its "
                "source produces in the same step, so no run order exists. "
                f"Mark one edge lagged (e.g. lagged={{{cycle_edges[0]!r}}}) "
                "or pass an explicit run sequence"
            )
        out.extend(ready)
        placed.update(ready)
        pending = [n for n in pending if n not in placed]
    return out


def derive_sequence(
    components: "dict[str, Component]",
    connectors: "Iterable[Connector | tuple[str, str]] | None" = None,
    lagged: "set[tuple[str, str]] | Literal['all']" = "all",
) -> RunSequence:
    """Derive the canonical run sequence from the coupling graph.

    One slot per distinct component cadence, fast to slow. Within a slot:

    1. lagged connects delivered at this cadence (the faster endpoint of
       each connection) — destinations see the sources' previous exports;
    2. each mediator's compute followed by its outgoing connects (mediator
       exports only exist after compute, so these are always sequential);
    3. component runs, ordered by the sequential (non-lagged) connections
       among them, each run followed by its outgoing sequential connects.

    Parameters
    ----------
    components : dict[str, Component]
        All participants, mediators included.
    connectors : iterable of Connector or (src, dst) name tuples, optional
        The coupling graph's edges.
    lagged : set[tuple[str, str]] | "all"
        Connections whose destination consumes the source's *previous* state
        (NUOPC-explicit coupling). Default "all" — the canonical shape.
        Connections not in the set are sequential; a cycle of sequential
        connections among same-cadence components raises SequenceError.
    """
    from .mediator import Mediator

    names = set(components)
    pairs: list[tuple[str, str]] = []
    for item in connectors or []:
        pair = (
            (item.src.name, item.dst.name)
            if hasattr(item, "src")
            else (str(item[0]), str(item[1]))
        )
        if pair not in pairs:
            pairs.append(pair)
    for src, dst in pairs:
        for name, what in ((src, "connection source"), (dst, "connection destination")):
            if name not in names:
                raise SequenceError(
                    f"Coupling graph references unknown {what} {name!r}."
                    + suggest(name, names)
                    + f" Known components: {sorted(names)}"
                )

    def is_lagged(pair: tuple[str, str]) -> bool:
        return lagged == "all" or pair in lagged

    def ns(td: np.timedelta64) -> int:
        return int(td.astype("timedelta64[ns]").astype(np.int64))

    order = {name: i for i, name in enumerate(components)}
    cadence = {name: comp.timestep for name, comp in components.items()}
    is_mediator = {
        name: isinstance(comp, Mediator) for name, comp in components.items()
    }
    seq_edges = [
        p for p in pairs if not is_lagged(p) and not is_mediator[p[0]]
    ]

    slots: list[Slot] = []
    for interval in sorted({cadence[n] for n in components}, key=ns):
        here = [n for n in components if cadence[n] == interval]
        actions: list[Action] = []
        # 1. lagged connects delivered at this cadence (the faster endpoint),
        #    excluding mediator-sourced ones (delivered after compute below)
        block = [
            p
            for p in pairs
            if is_lagged(p)
            and not is_mediator[p[0]]
            and min(ns(cadence[p[0]]), ns(cadence[p[1]])) == ns(interval)
        ]
        block.sort(key=lambda p: (order[p[0]], order[p[1]]))
        actions.extend(ConnectAction(s, d) for s, d in block)
        # 2. mediators of this cadence: compute, then deliver
        for med in (n for n in here if is_mediator[n]):
            actions.append(MediateAction(med))
            actions.extend(ConnectAction(s, d) for s, d in pairs if s == med)
        # 3. runs in dependency order over the sequential connections, each
        #    followed by its outgoing sequential connects
        runs = [n for n in here if not is_mediator[n]]
        for name in _toposort(runs, seq_edges, interval):
            actions.append(RunAction(name))
            actions.extend(
                ConnectAction(s, d) for s, d in seq_edges if s == name
            )
        if actions:
            slots.append(Slot(interval, actions))
    if not slots:
        raise SequenceError(
            "Cannot derive a run sequence from an empty component dict"
        )
    return RunSequence(slots)
