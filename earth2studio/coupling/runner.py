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
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.coupling._buffer import _Buffer
from earth2studio.coupling._nodes import (
    _Node,
    _node,
    _ns,
    _Slot,
    _Stateful,
)
from earth2studio.coupling._schedule import (
    _Event,
    _fmt,
    _Planner,
    _Scheduler,
    _SlotPlan,
    _window,
)
from earth2studio.coupling.connector import Connector, _split_label
from earth2studio.coupling.errors import CouplingError, _suggest
from earth2studio.data import fetch_dataframe
from earth2studio.grids import infer_grid, same_geometry
from earth2studio.models.utils import recommended_sources
from earth2studio.run._fields import _map_field, _with_lead_time
from earth2studio.run.runner import (
    DataRequest,
    Source,
    _fetch,
    _module_device,
    _seed_model,
    _signature_for_stream,
)
from earth2studio.run.work import WorkItem
from earth2studio.utils.coords import E2S_STATISTICS, coord_array_like
from earth2studio.utils.time_statistics import apply_time_statistic
from earth2studio.utils.type import CoordinateSystem

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
    component publishes each output slot's initial condition, when the input
    slot in the same position holds the same variables, and every output chunk;
    a diagnostic publishes every call.

    Parameters
    ----------
    components : Mapping[str, Any]
        Models keyed by component name; names key streams and connectors.
        Declaration order breaks ties between components ready at once.
    connectors : Sequence[Connector], optional
        Edges between components, by default none
    sources : Mapping[str, Source | tuple[Source | None, ...]], optional
        Sources per component in input-then-forcing slot order, like
        ``default_sources()``; a single source covers the first input slot. Slots
        left out use the model's recommendation, by default None. Connected
        slots use their source only for frames valid at or before the item time
        that the producer does not hold.
    seed : int, optional
        Base seed for per-member seeding of stochastic components, by default 0

    Raises
    ------
    CouplingError
        If a declaration is invalid or the components cannot be scheduled.
    """

    supports_member_batching = False

    def __init__(
        self,
        components: Mapping[str, Any],
        connectors: Sequence[Connector] = (),
        sources: Mapping[str, Source | tuple[Source | None, ...]] | None = None,
        seed: int = 0,
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
        self.connectors: dict[tuple[str, int], Connector] = {}
        for connector in connectors:
            self._add(connector)
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
        self.seed = seed
        self.spacing = self._spacing()
        self._check_windows()
        self._scheduler = _Scheduler(self.nodes, self.connectors, self.spacing)
        self._retention = self._plan_retention()
        self.regridders = {
            key: c.regridder for key, c in self.connectors.items() if c.regridder
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

    def _add(self, connector: Connector) -> None:
        for name in (connector.src, connector.dst):
            if name not in self.nodes:
                raise CouplingError(
                    f"Connector {connector} names unknown component {name!r}"
                    + _suggest(name, self.nodes)
                )
        key = (connector.dst, connector.slot)
        if key in self.connectors:
            raise CouplingError(
                f"Connectors {self.connectors[key]} and {connector} feed the same "
                "slot (C3)"
            )
        src, dst = self.nodes[connector.src], self.nodes[connector.dst]
        slots = {slot.index: slot for slot in dst.imports}
        if connector.slot not in slots:
            kind = "forcing" if dst.stateful else "input"
            raise CouplingError(
                f"Connector {connector}: {connector.dst!r} has no time-varying "
                f"gridded {kind} slot {connector.slot} (C3)"
            )
        slot = slots[connector.slot]
        bases = set()
        used: set[int] = set()
        for label in slot.labels:
            base, modifier = _split_label(label)
            bases.add(base)
            variable = connector.source_variable(base)
            producers = [
                i for i in range(len(src.outputs)) if variable in src.exports(i)
            ]
            if not producers:
                raise CouplingError(
                    f"Connector {connector}: {connector.src!r} does not publish "
                    f"{variable!r} for {label!r} (C4)"
                    + _suggest(variable, src.variables)
                )
            if len(producers) > 1:
                raise CouplingError(
                    f"Connector {connector}: {connector.src!r} publishes "
                    f"{variable!r} in output slots {producers}; rename one (C4)"
                )
            used.update(producers)
            if modifier is not None and connector.time == "hold":
                raise CouplingError(
                    f"Connector {connector}: time='hold' applies to plain labels, "
                    f"not the reduction {label!r} (C7)"
                )
        for base in connector.variables or {}:
            if base not in bases:
                raise CouplingError(
                    f"Connector {connector} renames {base!r}, which slot "
                    f"{connector.slot} does not declare (C4)" + _suggest(base, bases)
                )
        for i in sorted(used):
            _check_geometry(connector, src.outputs[i], slot.signature)
        self.connectors[key] = connector

    def _spacing(self) -> dict[str, int]:
        """Lead-time spacing of each component's published frames."""
        spacing = {
            name: node.spacing
            for name, node in self.nodes.items()
            if isinstance(node, _Stateful)
        }
        for name, node in self.nodes.items():
            if node.stateful:
                continue
            exact = [
                c
                for (dst, _), c in self.connectors.items()
                if dst == name and c.time == "exact"
            ]
            if not exact:
                raise CouplingError(
                    f"Diagnostic {name!r} needs at least one connector with "
                    "time='exact' to tell it when to run (C2)"
                )
        for _ in self.nodes:  # resolve diagnostic chains
            for name, node in self.nodes.items():
                if name in spacing:
                    continue
                for (dst, _), c in self.connectors.items():
                    if dst == name and c.time == "exact" and c.src in spacing:
                        spacing[name] = spacing[c.src]
                        break
        return spacing

    def _check_windows(self) -> None:
        for connector in self.connectors.values():
            slot = next(
                s
                for s in self.nodes[connector.dst].imports
                if s.index == connector.slot
            )
            for label in slot.labels:
                _, modifier = _split_label(label)
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
        stateful = [n for n in self.nodes.values() if isinstance(n, _Stateful)]
        period = math.lcm(*(n.cadence for n in stateful))
        horizon = max(n.due(0) for n in stateful) + 2 * period
        ages: dict[str, int] = {}
        requests = _Requests(self, _PROBE_TIME)
        for _ in self._scheduler.run(requests, horizon, slack=period, ages=ages):
            pass
        longest = max(n.cadence for n in stateful)
        return {name: ages.get(name, 0) + longest for name in self.nodes}

    # -- Runner protocol -------------------------------------------------------

    def to(self, device: torch.device | str) -> CoupledRunner:
        """Move every model and regridder to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        for node in self.nodes.values():
            to = getattr(node.model, "to", None)
            if callable(to):
                moved = to(self.device)
                node.model = moved if moved is not None else node.model
        self.regridders = {k: r.to(self.device) for k, r in self.regridders.items()}
        return self

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each component's stream signature through ``final_lead_time``."""
        leads: dict[tuple[str, int], list[int]] = {}
        for event in self._scheduler.run(_Planner(), _final(final_lead_time)):
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
        requests = _Requests(self, item.time)
        for _ in self._scheduler.run(requests, _final(item.final_lead_time)):
            pass
        return tuple(requests.requests)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield each component's outputs as it publishes them."""
        if len(item.member_ids) > 1:
            raise ValueError("CoupledRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        for node in self.nodes.values():
            _seed_model(self.seed, item.time, member_id, node.model)
        execution = _Execution(self, item)
        for event in self._scheduler.run(
            execution, _final(item.final_lead_time), retention=self._retention
        ):
            if event.slots and event.frames is not None:
                yield dict(zip(event.streams, event.frames))

    def describe(self, final_lead_time: np.timedelta64) -> str:
        """Return the components, connectors and schedule through ``final_lead_time``."""
        width = max(len(name) for name in self.nodes)
        lines = ["Components:"]
        for name, node in self.nodes.items():
            if isinstance(node, _Stateful):
                when = f"every {_fmt(node.cadence)[1:]}"
            else:
                when = "when its inputs arrive"
            if len(node.streams) > 1:
                when += f"; streams {', '.join(node.streams)}"
            lines.append(f"  {name:<{width}}  {node.kind:<10}  {when}")
        lines.append("Connectors:")
        for connector in self.connectors.values():
            dst = self.nodes[connector.dst]
            slot = next(s for s in dst.imports if s.index == connector.slot)
            regrid = ", regridded" if connector.regridder is not None else ""
            lines.append(
                f"  {connector.src} -> {connector.dst} {slot} ({connector.time}"
                f"{regrid}): {', '.join(slot.labels)}"
            )
        if not self.connectors:
            lines.append("  none")
        lines.append(f"Schedule through {_fmt(_final(final_lead_time))}:")
        for event in self._scheduler.run(_Planner(), _final(final_lead_time)):
            published = ", ".join(_fmt(v) for v in event.leads)
            reads = "; ".join(_describe_plan(plan) for plan in event.plans)
            line = (
                f"  {_fmt(event.tick):>7}  {event.node.name:<{width}}  "
                f"{event.action:<17}  -> {published}"
            )
            lines.append(line + (f"  [{reads}]" if reads else ""))
        return "\n".join(lines)


# -- execution -----------------------------------------------------------------


class _Requests:
    """Planning executor recording every ``fetch_data`` call a run would make.

    DataFrame slots are fetched with ``fetch_dataframe``, which ``DataRequest``
    does not describe; they are checked for a source but not recorded.
    """

    def __init__(self, runner: CoupledRunner, time: np.datetime64) -> None:
        self.runner = runner
        self.time = np.array([time])
        self.requests: list[DataRequest] = []

    def _add(self, node: _Node, slot: _Slot, labels: Sequence[str], leads: Any) -> None:
        source = _source(self.runner, node, slot)
        if slot.frame:
            return
        self.requests.append(
            DataRequest(
                source,
                self.time,
                np.array(list(labels)),
                np.asarray(leads, dtype=np.int64).astype("timedelta64[ns]"),
            )
        )

    def initial(self, node: _Stateful) -> None:
        for slot in node.inputs:
            leads = [0] if slot.frame else slot.leads
            self._add(node, slot, slot.labels, leads)

    def fire(self, event: _Event, buffers: Mapping[str, _Buffer]) -> None:
        for plan in event.plans:
            if plan.connector is None:
                self._add(event.node, plan.slot, plan.slot.labels, plan.leads)
                continue
            for lead in plan.leads:
                labels = [r.label for r in plan.fetched if r.lead == int(lead)]
                if labels:
                    self._add(event.node, plan.slot, labels, [lead])


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
        if slot.frame:
            frame = fetch_dataframe(
                source,  # type: ignore[arg-type]
                time,
                np.array(list(labels)),
                lead_time=lead_time,
                device=self.runner.device,
            )
            return None if len(frame) == 0 else frame
        signature = slot.signature
        if list(labels) != list(slot.labels):
            signature = coord_array_like(signature, {"variable": list(labels)})
        request = DataRequest(source, time, np.array(list(labels)), lead_time)
        x = _fetch(request, signature, self.runner.device)
        if slot.static:
            x = x.isel(lead_time=0, drop=True)
        return x

    def initial(self, node: _Stateful) -> tuple[Any, ...]:
        values = tuple(
            self._fetch(node, slot, slot.labels, [0] if slot.frame else slot.leads)
            for slot in node.inputs
        )
        self.inputs[node.name] = values
        return values

    def fire(
        self, event: _Event, buffers: Mapping[str, _Buffer]
    ) -> tuple[xr.DataArray, ...]:
        node = event.node
        args = tuple(self._value(node, plan, buffers) for plan in event.plans)
        if isinstance(node, _Stateful):
            if event.index == 0:
                y, state = node.model.initialize(*self.inputs.pop(node.name), *args)
            else:
                y, state = node.model.step(
                    *self.y[node.name], *args, state=self.state[node.name]
                )
            ys = y if isinstance(y, tuple) else (y,)
            self.y[node.name], self.state[node.name] = ys, state
            return ys
        outputs = node.model(*args)
        lead = np.array([event.index], dtype=np.int64).astype("timedelta64[ns]")
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return tuple(_with_lead_time(x, lead) for x in outputs)

    def _value(
        self, node: _Node, plan: _SlotPlan, buffers: Mapping[str, _Buffer]
    ) -> Any:
        if plan.connector is None:
            return self._fetch(node, plan.slot, plan.slot.labels, plan.leads)
        connector = plan.connector
        buffer = buffers[connector.src]
        spacing = np.timedelta64(self.runner.spacing[connector.src], "ns")
        regridder = self.runner.regridders.get((connector.dst, connector.slot))
        parts = []
        for lead in plan.leads:
            lead = int(lead)
            reads = [r for r in plan.reads if r.lead == lead]
            label = np.timedelta64(lead, "ns")
            pieces = []
            held = []
            for read in reads:
                if not read.frames:
                    continue
                if read.modifier is None:
                    x = buffer.get(read.variable, read.frames[0])
                else:
                    stack = xr.concat(
                        [buffer.get(read.variable, f) for f in read.frames],
                        dim="lead_time",
                        combine_attrs="override",
                    )
                    axis = stack.get_axis_num("lead_time")
                    x = apply_time_statistic(
                        stack, read.modifier, label, spacing, "lead_time"
                    ).expand_dims(lead_time=[label], axis=axis)
                held.append(x.assign_coords(variable=[read.label], lead_time=[label]))
            if held:
                x = xr.concat(held, dim="variable", combine_attrs="override")
                pieces.append(regridder(x) if regridder is not None else x)
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
        x = _map_field(x, plan.slot.signature)
        # Reductions carry the slot's temporal-statistics metadata, as fetch_data's do.
        x.attrs = {k: v for k, v in x.attrs.items() if k != E2S_STATISTICS}
        statistics = plan.slot.signature.attrs.get(E2S_STATISTICS)
        if statistics:
            x.attrs[E2S_STATISTICS] = statistics
        return x


# -- helpers -------------------------------------------------------------------


def _final(final_lead_time: np.timedelta64) -> int:
    final = int(_ns(final_lead_time)[0])
    if final < 0:
        raise ValueError(f"Final lead time {final_lead_time} must be non-negative")
    return final


def _slot_sources(
    node: _Node, given: Source | tuple[Source | None, ...] | None
) -> tuple[Any, ...]:
    """One source per input-then-forcing slot, falling back to recommendations."""
    count = len(node.inputs) + len(node.forcing)
    recommended = recommended_sources(node.model)
    if not isinstance(recommended, tuple):
        recommended = (recommended,)
    if given is None:
        given = ()
    elif not isinstance(given, tuple):
        given = (given,)
    if len(given) > count:
        raise CouplingError(
            f"Component {node.name!r} has {count} slots but {len(given)} sources"
        )
    given = tuple(given) + (None,) * (count - len(given))
    recommended = tuple(recommended)[:count]
    recommended += (None,) * (count - len(recommended))
    return tuple(g if g is not None else r for g, r in zip(given, recommended))


def _source(runner: CoupledRunner, node: _Node, slot: _Slot) -> Any:
    position = len(node.inputs) + slot.index if slot.forcing else slot.index
    source = runner.sources[node.name][position]
    if source is None:
        hint = " or connect the slot" if slot.forcing and not slot.frame else ""
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


def _describe_plan(plan: _SlotPlan) -> str:
    if plan.connector is None:
        return f"{plan.slot} <- source"
    leads = ", ".join(_fmt(v) for v in plan.leads)
    frames = sorted({f for read in plan.reads for f in read.frames})
    used = ""
    if frames and frames != sorted(int(v) for v in plan.leads):
        span = (
            _fmt(frames[0])
            if len(frames) == 1
            else f"{_fmt(frames[0])}..{_fmt(frames[-1])}"
        )
        used = f" from {span}"
    extra = " + source" if plan.fetched else ""
    return f"{plan.slot} <- {plan.connector.src} at {leads}{used}{extra}"


__all__ = ["CoupledRunner"]
