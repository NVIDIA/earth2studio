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

The string DSL mirrors NUOPC's runSeq::

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
from dataclasses import dataclass, field

import numpy as np

from .clock import DeltaLike, as_timedelta, fmt_timedelta, is_multiple
from .errors import CadenceError, SequenceError, suggest


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
