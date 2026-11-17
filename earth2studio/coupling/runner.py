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

"""CoupledRunner: run coupled components over one work item."""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from itertools import chain
from typing import Any, TypeAlias

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.coupling._buffer import _Buffer
from earth2studio.coupling._nodes import _Node, _node, _ns, _Slot
from earth2studio.coupling._schedule import (
    _Event,
    _fmt,
    _Planner,
    _Read,
    _Scheduler,
    _SlotPlan,
    _window,
)
from earth2studio.coupling.connector import Connector, _split_label
from earth2studio.coupling.errors import CouplingError, _suggest
from earth2studio.data import (
    DataSource,
    ForecastSource,
    fetch_dataframe,
)
from earth2studio.data.base import DataFrameSource, ForecastFrameSource
from earth2studio.grids import infer_grid, same_geometry
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import recommended_sources
from earth2studio.perturbation import Perturbation
from earth2studio.run._fields import _with_lead_time, map_field
from earth2studio.run.runner import (
    DataRequest,
    _check_perturbation,
    _fetch,
    _module_device,
    _perturb,
    _seed_model,
    _signature_for_stream,
    _source_members,
)
from earth2studio.run.work import WorkItem
from earth2studio.utils.coords import E2S_STATISTICS, coord_array_like
from earth2studio.utils.time_statistics import apply_time_statistic
from earth2studio.utils.type import CoordinateSystem

# A source for any model slot: gridded fields, or DataFrames for observation slots.
SlotSource: TypeAlias = (
    DataSource | ForecastSource | DataFrameSource | ForecastFrameSource
)

_PROBE_TIME = np.datetime64("2000-01-01T00", "ns")


class CoupledRunner:
    """Run models coupled through connectors, one work item at a time.

    Each component is a prognostic, stateful assimilation or diagnostic model,
    used as is. A stateful component advances at its own cadence, derived from
    its signatures; a diagnostic runs at each valid time its connected inputs
    arrive. Before a component runs, each of its connected slots is filled from
    what the producer has published, selected or reduced in time, regridded and
    mapped onto the slot; every other slot is fetched from its source. Ordering
    follows from the data each slot needs, so feedback loops need one
    ``time="hold"`` edge.

    Each output slot yields its own stream: named after the component when it
    has one output slot, else ``name.0``, ``name.1`` and so on. A stateful
    component publishes each output slot's initial condition, when that slot
    declares the same-position input slot's variables on its grid, and every
    output chunk;
    a diagnostic publishes every call.

    A work item's final lead time must be a lead time at which some stateful
    component's output is due. Components' ``front_hook`` and ``rear_hook`` are
    not run (a warning is logged when one is set); use ``perturbation=`` or a
    connector instead. Each producer keeps the history a short dry run of the
    schedule reads, plus its longest step, so a reduction window much longer
    than its consumer's step (about 30 h for a 6 h consumer) can fail partway
    through a run.

    Items with several members run as one batch on a leading ``ensemble``
    dimension, observation frames shared by every member, seeded once from the
    first member, so stochastic members depend on their batch; ``member_batch=1``
    seeds each member alone. A perturbation, by contrast, draws each member alone
    under its own seed.

    Parameters
    ----------
    components : Mapping[str, Any]
        Models keyed by component name; names key streams and connectors.
        Declaration order breaks ties between components ready at once.
    connectors : Sequence[Connector], optional
        Edges between components, by default none. State connectors
        (``state=True``) into one slot must overwrite different variables.
    sources : Mapping[str, SlotSource | tuple[SlotSource | None, ...]], optional
        Sources per component in input-then-forcing slot order, like
        ``default_sources()``; a single source covers the first input slot. Slots
        left out use the model's recommendation, by default None. If neither
        source is available, a DataFrame slot receives None; gridded inputs
        still need a source. Connected slots use their source only for frames
        valid at or before the item time that the producer does not hold.
    seed : int, optional
        Base seed for per-member seeding of stochastic components, by default 0.
        Each component offsets it by a hash of its name, so components draw
        independent noise
    devices : Mapping[str, torch.device | str], optional
        Devices of components that :meth:`to` leaves in place, such as CPU-only
        impact models, by default none. Every other component runs where its
        parameters or buffers are, or on the runner's device without any. Fetches
        and connected fields are staged onto the receiving component's device.
    source_member : Callable[[int], Hashable], optional
        Maps a work item's member ID to the member that ensemble sources (those
        taking a ``member`` keyword) supply for it, in the source's own labels,
        by default None. Every fetched slot whose source is an ensemble source
        reads that member, initial conditions and forcing alike; other slots are
        shared by every member.
    perturbation : Mapping[str, Perturbation], optional
        Initial-condition perturbation per stateful component, applied to its
        first input slot as :class:`~earth2studio.run.PrognosticRunner` applies
        it, from the component's offset seed, by default none.

    Raises
    ------
    CouplingError
        If a declaration is invalid or the components cannot be scheduled.
    """

    supports_member_batching = True

    def __init__(
        self,
        components: Mapping[str, Any],
        connectors: Sequence[Connector] = (),
        sources: Mapping[str, SlotSource | tuple[SlotSource | None, ...]] | None = None,
        seed: int = 0,
        devices: Mapping[str, torch.device | str] | None = None,
        source_member: Callable[[int], Hashable] | None = None,
        perturbation: Mapping[str, Perturbation] | None = None,
    ) -> None:
        if not components:
            raise CouplingError("CoupledRunner needs at least one component")
        self.nodes: dict[str, _Node] = {
            str(name): _node(str(name), model) for name, model in components.items()
        }
        streams = [stream for node in self.nodes.values() for stream in node.streams]
        clashes = sorted({stream for stream in streams if streams.count(stream) > 1})
        if clashes:
            raise CouplingError(f"Stream names {clashes} are used twice (C12)")
        if not any(node.stateful for node in self.nodes.values()):
            raise CouplingError("CoupledRunner needs at least one stateful component")
        self.connectors: dict[tuple[str, int], tuple[Connector, ...]] = {}
        self.states: dict[str, tuple[Connector, ...]] = {}
        for connector in connectors:
            if connector.state:
                self._add_state(connector)
            else:
                self._add(connector)
        devices = dict(devices or {})
        for name in devices:
            if name not in self.nodes:
                raise CouplingError(
                    f"Device given for unknown component {name!r}"
                    + _suggest(name, self.nodes)
                )
        self.devices = {name: torch.device(d) for name, d in devices.items()}
        sources = dict(sources or {})
        for name in sources:
            if name not in self.nodes:
                raise CouplingError(
                    f"Sources given for unknown component {name!r}"
                    + _suggest(name, self.nodes)
                )
        self.sources = {
            name: _slot_sources(node, sources.get(name))
            for name, node in self.nodes.items()
        }
        self._recommended: dict[str, tuple[Any, ...]] = {}
        self.seed = seed
        self.source_member = source_member
        self.perturbation = dict(perturbation or {})
        for name, perturb in self.perturbation.items():
            if name not in self.nodes:
                raise CouplingError(
                    f"Perturbation given for unknown component {name!r}"
                    + _suggest(name, self.nodes)
                )
            if not self.nodes[name].stateful:
                raise CouplingError(
                    f"Perturbation given for component {name!r}, which has no "
                    "initial condition"
                )
            _check_perturbation(perturb)
        self.spacing = self._spacing()
        self._check_windows()
        self._scheduler = _Scheduler(
            self.nodes, self.connectors, self.spacing, self.states
        )
        self._retention = self._plan_retention()
        for name, node in self.nodes.items():
            hooks = [h for h in ("front_hook", "rear_hook") if _has_hook(node.model, h)]
            if node.stateful and hooks:
                logger.warning(
                    f"CoupledRunner does not run {' or '.join(hooks)} on component "
                    f"{name!r}; use perturbation= or a connector instead"
                )
        self.regridders = {
            c: c.regridder for c in self._all_connectors() if c.regridder is not None
        }
        self.device = next(
            (
                _module_device(node.model)
                for node in self.nodes.values()
                if isinstance(node.model, torch.nn.Module)
            ),
            torch.device("cpu"),
        )

    # -- validation ----------------------------------------------------------

    def _check_names(self, connector: Connector) -> None:
        for name in (connector.src, connector.dst):
            if name not in self.nodes:
                raise CouplingError(
                    f"Connector {connector} names unknown component {name!r}"
                    + _suggest(name, self.nodes)
                )

    def _producer(self, connector: Connector, label: str, variable: str) -> int:
        """The ``src`` output slot publishing ``variable`` for ``label`` (``C4``)."""
        src = self.nodes[connector.src]
        producers = [i for i in range(len(src.outputs)) if variable in src.exports(i)]
        if not producers:
            raise CouplingError(
                f"Connector {connector}: {connector.src!r} does not publish "
                f"{variable!r} for {label!r} (C4)" + _suggest(variable, src.variables)
            )
        if len(producers) > 1:
            raise CouplingError(
                f"Connector {connector}: {connector.src!r} publishes {variable!r} in "
                f"output slots {producers}; rename one (C4)"
            )
        return producers[0]

    def _add_state(self, connector: Connector) -> None:
        self._check_names(connector)
        src, dst = self.nodes[connector.src], self.nodes[connector.dst]
        if not dst.stateful or connector.slot not in dst.state_slots:
            raise CouplingError(
                f"Connector {connector}: {connector.dst!r} has no state slot "
                f"{connector.slot}, an input slot whose output slot holds the same "
                "variables (C3)"
            )
        slot = dst.inputs[connector.slot]
        labels = set(connector.variables or {})
        unknown = sorted(labels - set(slot.labels))
        if unknown:
            raise CouplingError(
                f"Connector {connector}: {connector.dst!r} {slot} holds no state "
                f"variables {unknown} (C4)" + _suggest(unknown[0], slot.labels)
            )
        for other in self.states.get(connector.dst, ()):
            shared = labels & set(other.variables or {})
            if other.slot == connector.slot and shared:
                raise CouplingError(
                    f"Connectors {other} and {connector} both overwrite "
                    f"{sorted(shared)} (C3)"
                )
        used = {
            self._producer(connector, label, connector.source_variable(label))
            for label in labels
        }
        for i in sorted(used):
            _check_geometry(connector, src.outputs[i], slot.signature)
        self.states[connector.dst] = self.states.get(connector.dst, ()) + (connector,)

    def _add(self, connector: Connector) -> None:
        self._check_names(connector)
        key = (connector.dst, connector.slot)
        src, dst = self.nodes[connector.src], self.nodes[connector.dst]
        slots = {slot.index: slot for slot in dst.imports}
        if connector.slot not in slots:
            kind = "forcing" if dst.stateful else "input"
            raise CouplingError(
                f"Connector {connector}: {connector.dst!r} has no time-varying "
                f"gridded {kind} slot {connector.slot} (C3)"
            )
        slot = slots[connector.slot]
        bases = {_split_label(label)[0] for label in slot.labels}
        for base in connector.variables or {}:
            if base not in bases:
                raise CouplingError(
                    f"Connector {connector} names {base!r}, which slot "
                    f"{connector.slot} does not declare (C4)" + _suggest(base, bases)
                )
        fed = [label for label in slot.labels if connector.feeds(label)]
        for other in self.connectors.get(key, ()):
            shared = [label for label in fed if other.feeds(label)]
            if shared:
                raise CouplingError(
                    f"Connectors {other} and {connector} both feed {shared}; name "
                    "each connector's labels in variables (C3)"
                )
        used: set[int] = set()
        for label in fed:
            base, modifier = _split_label(label)
            used.add(self._producer(connector, label, connector.source_variable(base)))
            if modifier is not None and connector.time == "hold":
                raise CouplingError(
                    f"Connector {connector}: time='hold' applies to plain labels, "
                    f"not the reduction {label!r} (C7)"
                )
        for i in sorted(used):
            _check_geometry(connector, src.outputs[i], slot.signature)
        self.connectors[key] = self.connectors.get(key, ()) + (connector,)

    def _spacing(self) -> dict[str, int]:
        """Lead-time spacing of each component's published frames."""
        spacing = {
            name: node.spacing for name, node in self.nodes.items() if node.stateful
        }
        for name, node in self.nodes.items():
            if node.stateful:
                continue
            exact = [
                c
                for (dst, _), feeding in self.connectors.items()
                for c in feeding
                if dst == name and c.time == "exact"
            ]
            if not exact:
                raise CouplingError(
                    f"Diagnostic {name!r} needs at least one connector with "
                    "time='exact' to tell it when to run (C2)"
                )
        # A diagnostic runs where every exact input has a frame, so its frames
        # are spaced by the least common multiple of its producers' spacings.
        for _ in self.nodes:  # resolve diagnostic chains
            for name in self.nodes:
                if name in spacing:
                    continue
                sources = [
                    c.src
                    for (dst, _), feeding in self.connectors.items()
                    for c in feeding
                    if dst == name and c.time == "exact"
                ]
                if all(src in spacing for src in sources):
                    spacing[name] = math.lcm(*(spacing[src] for src in sources))
        return spacing

    def _check_windows(self) -> None:
        for connector in self._all_connectors():
            if connector.state:
                continue
            slot = next(
                s
                for s in self.nodes[connector.dst].imports
                if s.index == connector.slot
            )
            for label in slot.labels:
                _, modifier = _split_label(label)
                if not connector.feeds(label):
                    continue
                if modifier is None or connector.src not in self.spacing:
                    continue
                try:
                    _window(modifier, 0, self.spacing[connector.src])
                except ValueError as error:
                    raise CouplingError(
                        f"Connector {connector}: {label!r} cannot be reduced from "
                        f"frames every {_fmt(self.spacing[connector.src])[1:]} (C8): "
                        f"{error}"
                    ) from error

    def _plan_retention(self) -> dict[str, int]:
        """Dry-run a few coupling periods, raising on cycles and missing sources."""
        stateful = [n for n in self.nodes.values() if n.stateful]
        period = math.lcm(*(n.cadence for n in stateful))
        horizon = max(n.due(0) for n in stateful) + 2 * period
        ages: dict[str, int] = {}
        requests = _Requests(self)
        for _ in self._scheduler.run(requests, horizon, slack=period, ages=ages):
            pass
        longest = max(n.cadence for n in stateful)
        return {name: ages.get(name, 0) + longest for name in self.nodes}

    # -- Runner protocol -------------------------------------------------------

    def to(self, device: torch.device | str) -> CoupledRunner:
        """Move every component without a ``devices`` entry to ``device``.

        Regridders move to the device of the component they feed.
        """
        self.device = torch.device(device)
        for name, node in self.nodes.items():
            to = getattr(node.model, "to", None)
            if name not in self.devices and callable(to):
                moved = to(self.device)
                node.model = moved if moved is not None else node.model
        self.regridders = {
            c: r.to(self.device_of(c.dst)) for c, r in self.regridders.items()
        }
        return self

    def device_of(self, name: str) -> torch.device:
        """Return the device component ``name`` runs and receives fields on.

        A component with parameters or buffers runs where they are; one without,
        such as a derived diagnostic, follows the runner's device.
        """
        if name in self.devices:
            return self.devices[name]
        model = self.nodes[name].model
        if isinstance(model, torch.nn.Module) and any(
            True for _ in chain(model.parameters(), model.buffers())
        ):
            return _module_device(model)
        return self.device

    def _final(self, final_lead_time: np.timedelta64) -> int:
        """``final_lead_time`` in nanoseconds, checked against the schedule.

        A run ends where some stateful component's output is due, as a
        :class:`~earth2studio.run.PrognosticRunner` ends on a step; any other
        final lead time would end the run early without an error.
        """
        final = int(_ns(final_lead_time)[0])
        if final < 0:
            raise ValueError(f"Final lead time {final_lead_time} must be non-negative")
        ticks = self._scheduler.ticks(final)
        if final != 0 and (not ticks or ticks[-1] != final):
            finest = min(n.cadence for n in self.nodes.values() if n.stateful)
            last = _fmt(ticks[-1])[1:] if ticks else "0h"
            raise ValueError(
                f"Final lead time {final_lead_time} is not a lead time at which any "
                f"component's output is due (the finest component step is "
                f"{_fmt(finest)[1:]}); the run would end early, at {last}"
            )
        return final

    def _all_connectors(self) -> list[Connector]:
        states = [c for connectors in self.states.values() for c in connectors]
        feeding = [c for connectors in self.connectors.values() for c in connectors]
        return feeding + states

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each component's stream signature through ``final_lead_time``."""
        leads: dict[tuple[str, int], list[int]] = {}
        for event in self._scheduler.run(_Planner(), self._final(final_lead_time)):
            for slot in event.slots:
                key = (event.node.name, slot)
                leads.setdefault(key, []).extend(int(v) for v in event.leads)
        coords: dict[str, CoordinateSystem] = {}
        for name, node in self.nodes.items():
            for slot, (stream, output) in enumerate(zip(node.streams, node.outputs)):
                if (name, slot) not in leads:
                    continue
                values = np.unique(np.array(leads[(name, slot)], dtype=np.int64))
                dtype = (
                    output["lead_time"].dtype
                    if "lead_time" in output.coords
                    else np.dtype("timedelta64[ns]")
                )
                stream_leads = values.astype("timedelta64[ns]").astype(dtype)
                coords[stream] = _signature_for_stream(output, stream_leads)
        return coords

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe every ``fetch_data`` call ``item`` makes, in run order."""
        requests = _Requests(self, item)
        for _ in self._scheduler.run(requests, self._final(item.final_lead_time)):
            pass
        return tuple(requests.requests)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield each component's outputs as it publishes them."""
        # A batch is seeded once, from its first member: stochastic members depend
        # on their batch, and member_batch=1 seeds each member alone.
        member_id = item.member_ids[0] if item.member_ids else 0
        for name, node in self.nodes.items():
            _seed_model(
                _component_seed(self.seed, name), item.time, member_id, node.model
            )
        execution = _Execution(self, item)
        for event in self._scheduler.run(
            execution, self._final(item.final_lead_time), retention=self._retention
        ):
            if event.slots and event.frames is not None:
                yield dict(zip(event.streams, event.frames))

    def describe(self, final_lead_time: np.timedelta64) -> str:
        """Return the components, connectors and schedule through ``final_lead_time``."""
        width = max(len(name) for name in self.nodes)
        lines = ["Components:"]
        for name, node in self.nodes.items():
            if node.stateful:
                kind, when = "stateful", f"every {_fmt(node.cadence)[1:]}"
            else:
                kind, when = "diagnostic", "when its inputs arrive"
            if len(node.streams) > 1:
                when += f"; streams {', '.join(node.streams)}"
            lines.append(f"  {name:<{width}}  {kind:<10}  {when}")
        lines.append("Connectors:")
        for connector in self._all_connectors():
            dst = self.nodes[connector.dst]
            if connector.state:
                slot, labels = dst.inputs[connector.slot], list(
                    connector.variables or {}
                )
                target = f"state {slot}"
            else:
                slot = next(s for s in dst.imports if s.index == connector.slot)
                labels = [label for label in slot.labels if connector.feeds(label)]
                target = str(slot)
            extra = ", regridded" if connector.regridder is not None else ""
            lines.append(
                f"  {connector.src} -> {connector.dst} {target} ({connector.time}"
                f"{extra}): {', '.join(labels)}"
            )
        if not self._all_connectors():
            lines.append("  none")
        lines.append(f"Schedule through {_fmt(self._final(final_lead_time))}:")
        for event in self._scheduler.run(_Planner(), self._final(final_lead_time)):
            published = ", ".join(_fmt(v) for v in event.leads)
            reads = "; ".join(
                [_describe_plan(plan) for plan in event.plans]
                + [_describe_plan(plan, state=True) for plan in event.states]
            )
            line = (
                f"  {_fmt(event.tick):>7}  {event.node.name:<{width}}  "
                f"{event.action:<17}  -> {published}"
            )
            lines.append(line + (f"  [{reads}]" if reads else ""))
        return "\n".join(lines)


# -- execution -----------------------------------------------------------------


class _Requests:
    """Planning executor recording every ``fetch_data`` call a run would make.

    DataFrame slots are not recorded: ``DataRequest`` describes gridded fetches.
    """

    def __init__(self, runner: CoupledRunner, item: WorkItem | None = None) -> None:
        # Without an item, a dry run at a probe time that reads no members.
        self.runner = runner
        self.item = item
        self.time = np.array([_PROBE_TIME if item is None else item.time])
        self.requests: list[DataRequest] = []

    def _add(self, node: _Node, slot: _Slot, labels: Sequence[str], leads: Any) -> None:
        if slot.frame:
            return
        source = _source(self.runner, node, slot)
        self.requests.append(
            DataRequest(
                source,
                self.time,
                np.array(list(labels)),
                np.asarray(leads, dtype=np.int64).astype("timedelta64[ns]"),
                (
                    None
                    if self.item is None
                    else _source_members(self.runner.source_member, source, self.item)
                ),
            )
        )

    def initial(self, node: _Node) -> None:
        for slot in node.inputs:
            leads = [0] if slot.frame else slot.leads
            self._add(node, slot, slot.labels, leads)

    def fire(self, event: _Event, buffers: Mapping[str, _Buffer]) -> None:
        for plan in event.plans:
            if not plan.reads:
                self._add(event.node, plan.slot, plan.slot.labels, plan.leads)
                continue
            for lead in plan.leads:
                labels = [r.label for r in plan.fetched if r.lead == int(lead)]
                if labels:
                    self._add(event.node, plan.slot, labels, [lead])
            for read in plan.reads:
                if read.backfill:
                    base = _split_label(read.label)[0]
                    self._add(event.node, plan.slot, [base], read.backfill)


class _Execution:
    """Executor running the models of one work item."""

    def __init__(self, runner: CoupledRunner, item: WorkItem) -> None:
        self.runner = runner
        self.item = item
        self.inputs: dict[str, tuple[Any, ...]] = {}
        self.y: dict[str, Any] = {}
        self.state: dict[str, Any] = {}

    def _fetch(
        self, node: _Node, slot: _Slot, labels: Sequence[str], leads: Any
    ) -> Any:
        source = _source(self.runner, node, slot)
        time = np.array([self.item.time])
        lead_time = np.asarray(leads, dtype=np.int64).astype("timedelta64[ns]")
        device = self.runner.device_of(node.name)
        if slot.frame:
            if source is None:
                return None
            frame = fetch_dataframe(
                source,  # type: ignore[arg-type]
                time,
                np.array(list(labels)),
                lead_time=lead_time,
                device=device,
            )
            return None if len(frame) == 0 else frame
        signature = slot.signature
        if list(labels) != list(slot.labels):
            signature = coord_array_like(signature, {"variable": list(labels)})
        request = DataRequest(
            source,
            time,
            np.array(list(labels)),
            lead_time,
            _source_members(self.runner.source_member, source, self.item),
        )
        x = _fetch(request, signature, device, self.item, self.runner.source_member)
        if slot.static:
            x = x.isel(lead_time=0, drop=True)
        return x

    def initial(self, node: _Node) -> tuple[Any, ...]:
        values = tuple(
            self._fetch(node, slot, slot.labels, [0] if slot.frame else slot.leads)
            for slot in node.inputs
        )
        perturbation = self.runner.perturbation.get(node.name)
        if perturbation is not None:
            x = _perturb(
                perturbation,
                values[0],
                _component_seed(self.runner.seed, node.name),
                self.item,
                self.runner.device_of(node.name),
            )
            values = (x, *values[1:])
        self.inputs[node.name] = values
        return values

    def fire(
        self, event: _Event, buffers: Mapping[str, _Buffer]
    ) -> tuple[xr.DataArray, ...]:
        node = event.node
        args = tuple(self._value(node, plan, buffers) for plan in event.plans)
        if node.stateful:
            if event.index == 0:
                x = self._overwrite(event, self.inputs.pop(node.name), buffers)
                y, state = node.model.initialize(*x, *args)
            else:
                y = self._overwrite(event, self.y[node.name], buffers)
                y, state = node.model.step(*y, *args, state=self.state[node.name])
            ys = y if isinstance(y, tuple) else (y,)
            self.y[node.name], self.state[node.name] = ys, state
            return ys
        outputs = node.model(*args)
        lead = np.array([event.index], dtype=np.int64).astype("timedelta64[ns]")
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return tuple(_with_lead_time(x, lead) for x in outputs)

    def _held(
        self,
        node: _Node,
        plan: _SlotPlan,
        connector: Connector,
        buffers: Mapping[str, _Buffer],
        lead: int,
    ) -> xr.DataArray | None:
        """Labels ``connector`` supplies at ``lead``, ready for the slot.

        Frames are selected or reduced, staged onto the receiving component's
        device and regridded; ``None`` when none are held. A window with
        backfilled early frames is regridded before it is reduced, so the
        producer's frames and the source's share the slot grid.
        """
        buffer = buffers[connector.src]
        device = self.runner.device_of(node.name)
        regridder = self.runner.regridders.get(connector)
        spacing = np.timedelta64(self.runner.spacing[connector.src], "ns")
        label = np.timedelta64(lead, "ns")

        def onto_slot(x: xr.DataArray) -> xr.DataArray:
            x = _stage(x, device)
            return x if regridder is None else regridder(x)

        def reduce(stack: xr.DataArray, modifier: str) -> xr.DataArray:
            axis = stack.get_axis_num("lead_time")
            return apply_time_statistic(
                stack, modifier, label, spacing, "lead_time"
            ).expand_dims(lead_time=[label], axis=axis)

        held: list[xr.DataArray] = []
        backfilled: list[tuple[_Read, str]] = []
        for read in plan.reads:
            if read.connector is not connector or read.lead != lead or not read.frames:
                continue
            if read.modifier is None:
                x = buffer.get(read.variable, read.frames[0])
            elif not read.backfill:
                stack = xr.concat(
                    [buffer.get(read.variable, f) for f in read.frames],
                    dim="lead_time",
                    combine_attrs="override",
                )
                x = reduce(stack, read.modifier)
            else:
                backfilled.append((read, read.modifier))
                continue
            held.append(x.assign_coords(variable=[read.label], lead_time=[label]))
        pieces = []
        if held:
            pieces.append(
                onto_slot(xr.concat(held, "variable", combine_attrs="override"))
            )
        for read, modifier in backfilled:
            produced = xr.concat(
                [buffer.get(read.variable, f) for f in read.frames],
                dim="lead_time",
                combine_attrs="override",
            )
            produced = onto_slot(produced.assign_coords(variable=[read.label]))
            base = _split_label(read.label)[0]
            early = self._fetch(node, plan.slot, [base], read.backfill)
            early = early.assign_coords(variable=[read.label])
            if set(early.dims) == set(produced.dims):
                early = early.transpose(*produced.dims)
            stack = xr.concat(
                [early, produced],
                dim="lead_time",
                join="override",
                combine_attrs="override",
            )
            x = reduce(stack, modifier)
            pieces.append(x.assign_coords(lead_time=[label]))
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0]
        return xr.concat(
            pieces, dim="variable", join="override", combine_attrs="override"
        )

    def _value(
        self, node: _Node, plan: _SlotPlan, buffers: Mapping[str, _Buffer]
    ) -> Any:
        if not plan.reads:
            return self._fetch(node, plan.slot, plan.slot.labels, plan.leads)
        parts = []
        for lead in plan.leads:
            lead = int(lead)
            reads = [r for r in plan.reads if r.lead == lead]
            pieces: list[Any] = []
            for connector in plan.connectors:
                held = self._held(node, plan, connector, buffers, lead)
                if held is not None:
                    if pieces and set(held.dims) == set(pieces[0].dims):
                        held = held.transpose(*pieces[0].dims)
                    pieces.append(held)
            fetched = [r.label for r in reads if not r.frames]
            if fetched:
                x = self._fetch(node, plan.slot, fetched, [lead])
                if pieces and set(x.dims) == set(pieces[0].dims):
                    x = x.transpose(*pieces[0].dims)
                pieces.append(x)
            merged = (
                pieces[0]
                if len(pieces) == 1
                else xr.concat(
                    pieces, dim="variable", join="override", combine_attrs="override"
                )
            )
            parts.append(merged.sel(variable=list(plan.slot.labels)))
        x = (
            parts[0]
            if len(parts) == 1
            else xr.concat(parts, dim="lead_time", combine_attrs="override")
        )
        x = map_field(x, plan.slot.signature)
        # Reductions carry the slot's temporal-statistics metadata, as fetch_data's do.
        x.attrs = {k: v for k, v in x.attrs.items() if k != E2S_STATISTICS}
        statistics = plan.slot.field.attrs.get(E2S_STATISTICS)
        if statistics:
            x.attrs[E2S_STATISTICS] = statistics
        return x

    def _overwrite(
        self, event: _Event, arrays: tuple[Any, ...], buffers: Mapping[str, _Buffer]
    ) -> tuple[Any, ...]:
        """Apply ``event``'s state connectors to the arrays its call receives.

        Each state connector replaces the variables it names in one state slot,
        at each of the slot's lead times the producer supplies. Values it cannot
        supply yet (at or before the item time) keep the model's own. Writes go
        to a copy on the array's device, so the model's previous output, which
        may already be published, is never modified.
        """
        result = list(arrays)
        for plan in event.states:
            x = result[plan.slot.index]
            data = None
            for lead in plan.leads:
                write = self._state_piece(event, plan, x, buffers, int(lead))
                if write is None:
                    continue
                position, values = write
                if data is None:
                    data = x.data.copy()
                data[position] = values
            if data is not None:
                result[plan.slot.index] = x.copy(data=data)
        return tuple(result)

    def _state_piece(
        self,
        event: _Event,
        plan: _SlotPlan,
        x: xr.DataArray,
        buffers: Mapping[str, _Buffer],
        lead: int,
    ) -> tuple[tuple[Any, ...], Any] | None:
        """Where and what one state connector writes into ``x`` at ``lead``.

        Returns the index into ``x.data`` and the values to write there, already
        on ``x``'s device and in its dimension order, or ``None`` when the
        producer supplies nothing at ``lead``. Raises if the delivered fields
        cannot fit the slot.
        """
        (connector,) = plan.connectors or (None,)
        piece = None
        if connector is not None:
            piece = self._held(event.node, plan, connector, buffers, lead)
        if piece is None:
            return None
        where = f"{event.node.name!r} {plan.slot}"
        labels = [str(v) for v in piece.coords["variable"].values]
        signature = coord_array_like(plan.slot.signature, {"variable": labels})
        try:
            piece = map_field(piece, signature)
        except ValueError as error:
            raise CouplingError(
                f"Connector {connector}: delivered state does not match "
                f"{where}: {error}"
            ) from error
        j = list(_ns(x.coords["lead_time"].values)).index(lead)
        names = [str(v) for v in x.coords["variable"].values]
        index = [names.index(label) for label in labels]
        target = x.isel(lead_time=[j], variable=index)
        extra = [d for d in piece.dims if d not in target.dims]
        if extra:
            raise CouplingError(
                f"Connector {connector}: delivered state has dimensions "
                f"{extra} that {where} lacks"
            )
        missing = {d: target.sizes[d] for d in target.dims if d not in piece.dims}
        piece = piece.expand_dims(missing).transpose(*target.dims)
        if piece.shape != target.shape:
            raise CouplingError(
                f"Connector {connector}: delivered state of shape "
                f"{piece.shape} does not fit {where} of shape {target.shape}"
            )
        position: list[Any] = [slice(None)] * x.ndim
        position[x.get_axis_num("lead_time")] = slice(j, j + 1)
        position[x.get_axis_num("variable")] = index
        return tuple(position), _stage_like(piece, x).data


# -- helpers -------------------------------------------------------------------


def _has_hook(model: Any, name: str) -> bool:
    """Whether ``model`` sets an iterator hook other than the pass-through."""
    hook = getattr(model, name, None)
    default = getattr(PrognosticMixin, name)
    return hook is not None and getattr(hook, "__func__", hook) is not default


def _component_seed(base: int, name: str) -> int:
    """Base seed of component ``name``, so stochastic components draw independent noise.

    Offset by a hash of the name, which is stable across processes, unlike ``hash``.
    """
    return base + zlib.crc32(name.encode())


def _slot_sources(
    node: _Node, given: SlotSource | tuple[SlotSource | None, ...] | None
) -> tuple[Any, ...]:
    """One given source or ``None`` per input-then-forcing slot."""
    count = len(node.inputs) + len(node.forcing)
    if given is None:
        given = ()
    elif not isinstance(given, tuple):
        given = (given,)
    if len(given) > count:
        raise CouplingError(
            f"Component {node.name!r} has {count} slots but {len(given)} sources"
        )
    return tuple(given) + (None,) * (count - len(given))


def _source(runner: CoupledRunner, node: _Node, slot: _Slot) -> Any:
    """The slot's given source, else the model's recommendation.

    Recommendations are built only when a fetch needs one, as models may load
    grids or regridders to build them.
    """
    position = len(node.inputs) + slot.index if slot.forcing else slot.index
    source = runner.sources[node.name][position]
    if source is None:
        if node.name not in runner._recommended:
            recommended = recommended_sources(node.model)
            if not isinstance(recommended, tuple):
                recommended = (recommended,)
            runner._recommended[node.name] = tuple(recommended)
        recommended = runner._recommended[node.name]
        source = recommended[position] if position < len(recommended) else None
    if source is None and not slot.frame:
        hint = " or connect the slot" if slot.forcing else ""
        raise CouplingError(
            f"Component {node.name!r} {slot} needs a source: pass one in sources" + hint
        )
    return source


def _check_geometry(
    connector: Connector, produced: CoordinateSystem, slot: CoordinateSystem
) -> None:
    """Check grids where both sides declare one (``C5``)."""
    try:
        source, target = infer_grid(produced), infer_grid(slot)
    except Exception:  # noqa: BLE001 - not every signature carries a grid
        return
    regridder = connector.regridder
    if regridder is None:
        if not same_geometry(source, target):
            raise CouplingError(
                f"Connector {connector}: {connector.src!r} and {connector.dst!r} "
                "slot grids differ; pass a regridder (C5)"
            )
        return
    if not same_geometry(regridder.source_grid, source):
        raise CouplingError(
            f"Connector {connector}: regridder source grid does not match "
            f"{connector.src!r}'s output grid (C5)"
        )
    if not same_geometry(regridder.target_grid, target):
        raise CouplingError(
            f"Connector {connector}: regridder target grid does not match the "
            f"{connector.dst!r} slot grid (C5)"
        )


def _stage(x: xr.DataArray, device: torch.device) -> xr.DataArray:
    """Return ``x`` backed on ``device``: NumPy on the CPU, CuPy on CUDA."""
    if device.type == "cuda":
        return x.e2s.as_cupy(device.index)
    return x.e2s.as_numpy()


def _stage_like(x: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Return ``x`` backed like ``like``."""
    if type(like.data).__module__.startswith("cupy"):
        return x.e2s.as_cupy(like.data.device.id)
    return x.e2s.as_numpy()


def _describe_plan(plan: _SlotPlan, state: bool = False) -> str:
    if not plan.reads:
        return f"{plan.slot} <- source"
    leads = ", ".join(_fmt(v) for v in plan.leads)
    feeds = []
    for connector in plan.connectors:
        reads = [read for read in plan.reads if read.connector is connector]
        frames = sorted({f for read in reads for f in read.backfill + read.frames})
        used = ""
        if frames and frames != sorted(int(v) for v in plan.leads):
            span = (
                _fmt(frames[0])
                if len(frames) == 1
                else f"{_fmt(frames[0])}..{_fmt(frames[-1])}"
            )
            used = f" from {span}"
        feeds.append(f"{connector.src} at {leads}{used}")
    sourced = plan.fetched or any(read.backfill for read in plan.reads)
    if state:
        return f"state {plan.slot} <- {', '.join(feeds)}" + (
            " + kept" if sourced else ""
        )
    return f"{plan.slot} <- {', '.join(feeds)}" + (" + source" if sourced else "")


__all__ = ["CoupledRunner"]
