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

"""Run sequence: the order models advance at each coupling step.

The NUOPC ``runSeq`` analog, reduced to what the coupler needs. A
:class:`RunSequence` is a list of :class:`Slot` entries, each with an interval
and an ordered tuple of model names; at every clock time aligned with a slot's
interval the coupler advances that slot's models in order. Exchanges are not
scheduled separately: a model pulls the frames its declaration requires from
its sources' buffers when it advances, so order within a slot decides whether a
model sees the frame its source produced *this* step (the source runs first) or
the previous one (it runs after).

:func:`derive_sequence` lays out the canonical sequence from the models'
cadences and the same-step dependencies the coupler finds in their
declarations. Pass a :class:`RunSequence` to the coupler only to override it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from earth2studio.coupling._time import (
    DeltaLike,
    as_timedelta,
    fmt_timedelta,
    is_multiple,
    nanoseconds,
)
from earth2studio.coupling.errors import CouplingError, _suggest


@dataclass(frozen=True)
class Slot:
    """Models advanced, in order, at every multiple of ``interval``."""

    interval: np.timedelta64
    models: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "interval", as_timedelta(self.interval))
        object.__setattr__(self, "models", tuple(str(name) for name in self.models))
        if not self.models:
            raise CouplingError(
                f"Run-sequence slot @{fmt_timedelta(self.interval)} names no models"
            )


@dataclass(frozen=True)
class RunSequence:
    """Ordered slots, fastest cadence first."""

    slots: tuple[Slot, ...]

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        if not slots:
            raise CouplingError("Run sequence has no slots")
        object.__setattr__(self, "slots", slots)

    def models(self) -> set[str]:
        """Return every model the sequence advances."""
        return {name for slot in self.slots for name in slot.models}

    def validate(
        self,
        cadences: Mapping[str, np.timedelta64],
        dt: DeltaLike,
        before: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Check names, cadence alignment, completeness and same-step ordering.

        Parameters
        ----------
        cadences : Mapping[str, np.timedelta64]
            Cadence of every model in the coupled system.
        dt : DeltaLike
            The coupler's clock step; every slot interval must be a multiple.
        before : Iterable[tuple[str, str]], optional
            ``(src, dst)`` pairs where ``dst`` needs the frame ``src`` produces
            in the same step, so ``src`` must advance first.

        Raises
        ------
        CouplingError
            If the sequence names unknown models, a model's slot interval differs
            from its cadence, a model is never advanced or is advanced twice at
            one step, or a same-step dependency is violated.
        """
        dt = as_timedelta(dt)
        names = set(cadences)
        position: dict[str, tuple[int, int]] = {}
        for index, slot in enumerate(self.slots):
            if not is_multiple(slot.interval, dt):
                raise CouplingError(
                    f"Run-sequence slot @{fmt_timedelta(slot.interval)} is not a "
                    f"multiple of the clock step {fmt_timedelta(dt)}"
                )
            for order, name in enumerate(slot.models):
                if name not in names:
                    raise CouplingError(
                        f"Run sequence names unknown model {name!r}."
                        + _suggest(name, names)
                        + f" Known models: {sorted(names)}"
                    )
                if name in position:
                    raise CouplingError(
                        f"Run sequence advances {name!r} more than once; each "
                        "model belongs in exactly one slot"
                    )
                if cadences[name] != slot.interval:
                    raise CouplingError(
                        f"Model {name!r} has cadence {fmt_timedelta(cadences[name])} "
                        f"but is scheduled in a @{fmt_timedelta(slot.interval)} slot"
                    )
                position[name] = (index, order)
        idle = names - set(position)
        if idle:
            raise CouplingError(
                f"Run sequence never advances {sorted(idle)}; add each to the "
                "slot matching its cadence"
            )
        for src, dst in before:
            if src not in position or dst not in position:
                continue
            if nanoseconds(cadences[src]) > nanoseconds(cadences[dst]):
                raise CouplingError(
                    f"{dst!r} needs the frame {src!r} produces in the same step, "
                    f"but {src!r} advances every {fmt_timedelta(cadences[src])} and "
                    f"{dst!r} every {fmt_timedelta(cadences[dst])}; a slower source "
                    "cannot run ahead of a faster destination"
                )
            if position[src] > position[dst]:
                raise CouplingError(
                    f"{dst!r} needs the frame {src!r} produces in the same step, so "
                    f"{src!r} must advance before {dst!r} in the run sequence"
                )

    def __str__(self) -> str:
        lines = []
        for slot in self.slots:
            lines.append(f"@{fmt_timedelta(slot.interval)}")
            lines.extend(f"  {name}" for name in slot.models)
        return "\n".join(lines)


def _toposort(
    nodes: list[str], edges: list[tuple[str, str]], interval: np.timedelta64
) -> list[str]:
    """Order ``nodes`` so each edge's source precedes its destination.

    Declaration order is kept among unconstrained nodes.
    """
    members = set(nodes)
    deps: dict[str, set[str]] = {name: set() for name in nodes}
    for src, dst in edges:
        if src in members and dst in members:
            deps[dst].add(src)
    ordered: list[str] = []
    placed: set[str] = set()
    pending = list(nodes)
    while pending:
        ready = [name for name in pending if deps[name] <= placed]
        if not ready:
            cycle = [(s, d) for s, d in edges if s in pending and d in pending]
            raise CouplingError(
                f"Same-step coupling cycle among {sorted(pending)} at cadence "
                f"{fmt_timedelta(interval)}: {cycle} each need the frame their "
                "source produces in this step, so no run order exists. Have one "
                "destination declare its forcing at the previous step instead"
            )
        ordered.extend(ready)
        placed.update(ready)
        pending = [name for name in pending if name not in placed]
    return ordered


def derive_sequence(
    cadences: Mapping[str, DeltaLike],
    before: Iterable[tuple[str, str]] = (),
) -> RunSequence:
    """Derive the canonical run sequence from cadences and same-step dependencies.

    One slot per distinct cadence, fastest first, so at a shared step a faster
    model always advances before a slower one. Within a slot, models are ordered
    so every ``(src, dst)`` in ``before`` has ``src`` first, keeping declaration
    order otherwise.

    Parameters
    ----------
    cadences : Mapping[str, DeltaLike]
        Cadence of every model, in declaration order.
    before : Iterable[tuple[str, str]], optional
        ``(src, dst)`` pairs where ``dst`` needs the frame ``src`` produces in
        the same step. The coupler derives these from the models' declarations.

    Returns
    -------
    RunSequence
        The derived sequence.

    Raises
    ------
    CouplingError
        If ``cadences`` is empty, a dependency names an unknown model or a
        slower source, or same-cadence dependencies form a cycle.
    """
    if not cadences:
        raise CouplingError("Cannot derive a run sequence without models")
    resolved = {name: as_timedelta(value) for name, value in cadences.items()}
    edges = list(before)
    for src, dst in edges:
        for name in (src, dst):
            if name not in resolved:
                raise CouplingError(
                    f"Dependency {src!r}->{dst!r} names unknown model {name!r}."
                    + _suggest(name, resolved)
                )
    slots = []
    for interval in sorted(set(resolved.values()), key=nanoseconds):
        here = [name for name, cadence in resolved.items() if cadence == interval]
        slots.append(Slot(interval, tuple(_toposort(here, edges, interval))))
    sequence = RunSequence(tuple(slots))
    sequence.validate(resolved, slots[0].interval, edges)
    return sequence
