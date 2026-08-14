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

"""Driver: owns the clock and executes the run sequence (NUOPC_Driver analog).

Declare the coupling graph — components plus connections — and the schedule
follows: with no sequence given the Driver derives the canonical run
sequence from the graph (:func:`~.sequence.derive_sequence`). A hand-written
sequence (RunSequence object or DSL text) is the override for schedules the
graph cannot express. The lifecycle mirrors NUOPC: construct, then
``initialize(ics)`` (advertise -> connector matching -> realize -> component
initialize), then ``run()`` / ``steps()`` for inference or ``rollout(n)``
for a gradient-carrying advance. Coupling order is entirely the run
sequence's action order; the driver adds no hidden exchanges.
"""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from earth2studio.io import IOBackend

from earth2studio.utils.type import CoordSystem

from .clock import Clock, as_timedelta
from .component import Component
from .connector import Connector
from .errors import CouplingError, UnmatchedImportError
from .field import Field, State
from .mediator import Mediator
from .sequence import (
    ConnectAction,
    MediateAction,
    RunAction,
    RunSequence,
    derive_sequence,
    parse_run_sequence,
)

MEMORY_WARN_BYTES = 4e9


class Driver:
    """Executes a coupled system declared as components + connections.

    The declarative form needs no run sequence — the schedule is derived
    from the coupling graph::

        Driver(
            {"atmos": atmos, "ocean": ocean},
            clock=Clock("2024-01-01", "2024-01-05", "6h"),
            connectors=[("ocean", "atmos"), ("atmos", "ocean")],
        )

    Bare ``(src, dst)`` name tuples become default Connectors; pass Connector
    instances for regridding/time-policy/window options. Passing a sequence
    (RunSequence object or DSL text) overrides the derived schedule — the
    escape hatch for sequential coupling or hand-tuned action order.

    Parameters
    ----------
    components : dict[str, Component]
        All participants, mediators included; keys must match the names used
        in the run sequence.
    sequence : RunSequence | str, optional
        Run sequence object or DSL text. None (default) derives the
        canonical sequence from components + connectors via
        :func:`~.sequence.derive_sequence`.
    clock : Clock
        Coupling clock; dt must divide every component timestep.
    connectors : list[Connector | tuple[str, str]], optional
        The coupling graph's edges: Connector instances or bare (src, dst)
        name tuples (auto-built into default Connectors). With an explicit
        sequence, any ConnectAction without a matching (src, dst) pair still
        gets a default ``Connector(src, dst)`` built at initialize.
    collect : bool
        Keep per-ring export fields in memory for ``to_xarray()``
        (default True; disable for long runs writing to real IO). Recorded
        fields are detached clones — records are off the exchange path, so
        collection never pins autograd graphs during gradient rollouts.
    allow_unfed_imports : bool
        By default ``initialize`` raises UnmatchedImportError when a
        component advertises an import that no connector in the run sequence
        delivers (the component would silently run on stale initial-condition
        forcing). Pass True to downgrade this to a warning (default False).
    io : dict[str, IOBackend], optional
        Per-component IO backends (e.g. ``{"atmos": ZarrBackend()}``). Each
        backend receives one array per export field of its component, with a
        leading ``time`` dimension covering the component's ring times
        including t0; the driver streams a write after the initial-condition
        seed and after every component run. Independent of ``collect``.

    Attributes
    ----------
    sequence_derived : bool
        True when the run sequence was derived from the coupling graph
        rather than passed in.
    """

    def __init__(
        self,
        components: dict[str, Component],
        sequence: RunSequence | str | None = None,
        clock: Clock | None = None,
        connectors: "list[Connector | tuple[str, str]] | None" = None,
        collect: bool = True,
        io: "dict[str, IOBackend] | None" = None,
        allow_unfed_imports: bool = False,
    ):
        if clock is None:
            raise CouplingError(
                "Driver needs a clock — Driver(components, sequence, clock) "
                "or Driver(components, clock=Clock(start, stop, dt), "
                "connectors=[...])"
            )
        self.components = dict(components)
        self.clock = clock
        self.collect = collect
        self.allow_unfed_imports = allow_unfed_imports
        self._connectors: dict[tuple[str, str], Connector] = {}
        for item in connectors or []:
            if not isinstance(item, Connector):
                src, dst = item
                unknown = [n for n in (src, dst) if n not in self.components]
                if unknown:
                    raise CouplingError(
                        f"Connection ({src!r}, {dst!r}): {unknown} are not "
                        f"component names; known components: "
                        f"{sorted(self.components)}"
                    )
                item = Connector(self.components[src], self.components[dst])
            self._connectors[(item.src.name, item.dst.name)] = item
        self.sequence_derived = sequence is None
        if sequence is None:
            self.sequence = derive_sequence(
                self.components, self._connectors.values()
            )
        else:
            self.sequence = (
                parse_run_sequence(sequence) if isinstance(sequence, str) else sequence
            )
        # per-component record of (time, {std_name: Field}) for to_xarray
        self._records: dict[str, list[tuple[np.datetime64, dict[str, Field]]]] = {
            name: [] for name in self.components
        }
        self._io: dict[str, "IOBackend"] = dict(io or {})
        unknown = [n for n in self._io if n not in self.components]
        if unknown:
            raise CouplingError(
                f"io= keys {unknown} are not component names; known "
                f"components: {sorted(self.components)}"
            )
        self._io_ready: set[str] = set()
        self._initialized = False

    # -- setup -----------------------------------------------------------------
    def initialize(
        self, ics: dict[str, tuple[torch.Tensor, CoordSystem]] | None = None
    ) -> None:
        ics = ics or {}
        self.sequence.validate(self.components, self.clock.dt)
        # build connectors for every ConnectAction not covered by a prebuilt one
        for action in self.sequence.connections():
            key = (action.src, action.dst)
            if key not in self._connectors:
                self._connectors[key] = Connector(
                    self.components[action.src], self.components[action.dst]
                )
        for conn in self._connectors.values():
            conn.match()
        self._check_unfed_imports()
        # realize + initialize components
        for name, comp in self.components.items():
            comp.realize(self.clock)
            if name in ics:
                comp.initialize(*ics[name])
            elif not getattr(comp, "requires_ic", not isinstance(comp, Mediator)):
                comp.initialize()
            else:
                raise CouplingError(
                    f"Component {name!r} needs an initial condition — pass "
                    f"ics={{{name!r}: (x, coords)}}"
                )
            self._record(name, self.clock.start, comp.export_state)
            self._io_write(name, self.clock.start)
        self._warn_unconsumed_exports()
        self._warn_memory()
        self._initialized = True

    def _check_unfed_imports(self) -> None:
        """Every advertised import must be delivered by some connector —
        otherwise the component silently runs the whole simulation on its
        stale initial-condition forcing."""
        fed: dict[str, set[str]] = {name: set() for name in self.components}
        for (_, dst), conn in self._connectors.items():
            fed[dst] |= set(conn.match())
        available = {n: list(c.export_names) for n, c in self.components.items()}
        for name, comp in self.components.items():
            for field in comp.import_names:
                if field in fed[name]:
                    continue
                if self.allow_unfed_imports:
                    logger.warning(
                        "Component {!r} imports {!r} but no connector in the "
                        "run sequence delivers it — it will run on stale "
                        "initial-condition forcing",
                        name,
                        field,
                    )
                else:
                    raise UnmatchedImportError(name, field, available)

    def _warn_unconsumed_exports(self) -> None:
        consumed: set[tuple[str, str]] = set()
        for (src, _), conn in self._connectors.items():
            consumed |= {(src, f) for f in conn.match()}
        for name, comp in self.components.items():
            idle = [f for f in comp.export_names if (name, f) not in consumed]
            if idle:
                logger.warning(
                    "Component {!r} exports {} but no connector consumes them",
                    name,
                    idle,
                )

    def _warn_memory(self) -> None:
        if not self.collect:
            return
        total = 0
        for comp in self.components.values():
            per_ring = sum(
                f.data.numel() * f.data.element_size()
                for f in comp.export_state.values()
            )
            rings = self.clock.n_steps * (
                self.clock.dt.astype(np.int64) / comp.timestep.astype(np.int64)
            )
            total += per_ring * max(rings, 0)
        if total > MEMORY_WARN_BYTES:
            logger.warning(
                "In-memory collection will hold ~{:.1f} GB of export fields; "
                "pass collect=False and a real IO backend (e.g. ZarrBackend) "
                "for runs of this size",
                total / 1e9,
            )

    # -- record keeping ----------------------------------------------------------
    def _record(self, name: str, time: np.datetime64, state: State) -> None:
        """Append a snapshot of a component's export fields to _records.

        Records are off the exchange path (they only feed ``to_xarray``), so
        the field data is stored as detached clones — otherwise collection
        would pin every step's autograd graph during gradient rollouts. The
        Fields in component states and exchanges stay attached.
        """
        if not self.collect:
            return
        snapshot = {
            std: replace(f, data=f.data.detach().clone()) for std, f in state.items()
        }
        self._records[name].append((time, snapshot))

    def _strip_singletons(
        self, name: str, field: Field
    ) -> tuple[torch.Tensor, OrderedDict]:
        """Drop size-1 'time'/'batch' dims (published by components whose
        model coords carry them) so the ring-time axis can be prepended
        without colliding with the field's own stale coordinate."""
        data = field.data
        coords: OrderedDict = OrderedDict(field.coords)
        for key in ("time", "batch"):
            if key not in coords:
                continue
            axis = list(coords).index(key)
            if data.shape[axis] != 1:
                raise CouplingError(
                    f"Component {name!r} export {field.standard_name!r} "
                    f"carries a {key!r} dimension of size {data.shape[axis]}; "
                    "the driver records one snapshot per ring and can only "
                    f"absorb a size-1 {key!r} dim — publish a single "
                    f"{key}-slice per run"
                )
            data = data.squeeze(axis)
            del coords[key]
        return data, coords

    # -- IO streaming ------------------------------------------------------------
    def _io_ring_times(self, comp: Component) -> np.ndarray:
        """A component's ring times including t0 (its 'time' IO coordinate)."""
        span_ns = (
            (self.clock.stop - self.clock.start)
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
        n_rings = int(span_ns // comp.timestep.astype(np.int64))
        return (self.clock.start + np.arange(n_rings + 1) * comp.timestep).astype(
            "datetime64[ns]"
        )

    def _io_setup(self, name: str) -> bool:
        """Allocate backend arrays for a component's export fields; returns
        False when the component has not published any fields yet (mediators
        at t0), in which case setup is retried on the next write."""
        comp = self.components[name]
        if not comp.export_state:
            return False
        backend = self._io[name]
        times = self._io_ring_times(comp)
        # group export fields sharing a coord structure into one add_array call
        groups: dict[tuple, tuple[OrderedDict, list[str]]] = {}
        for std_name, field in comp.export_state.items():
            _, field_coords = self._strip_singletons(name, field)
            key = tuple((k, np.asarray(v).tobytes()) for k, v in field_coords.items())
            if key not in groups:
                total_coords: OrderedDict = OrderedDict(time=times)
                total_coords.update((k, np.asarray(v)) for k, v in field_coords.items())
                groups[key] = (total_coords, [])
            groups[key][1].append(std_name)
        for total_coords, std_names in groups.values():
            # NaN-initialize: zarr's default fill reads back as 0.0, which
            # would let never-written rows (e.g. cadences coarser than the
            # ring times, or a crashed run) masquerade as physical values
            shape = tuple(len(v) for v in total_coords.values())
            nan_init = [
                torch.full(
                    shape,
                    float("nan"),
                    dtype=comp.export_state[n].data.dtype,
                )
                for n in std_names
            ]
            backend.add_array(total_coords, std_names, data=nan_init)
        self._io_ready.add(name)
        return True

    def _io_write(self, name: str, time: np.datetime64) -> None:
        """Stream a component's current export fields to its IO backend."""
        if name not in self._io:
            return
        if name not in self._io_ready and not self._io_setup(name):
            return
        backend = self._io[name]
        comp = self.components[name]
        t = np.asarray([time], dtype="datetime64[ns]")
        for std_name, field in comp.export_state.items():
            data, field_coords = self._strip_singletons(name, field)
            coords: OrderedDict = OrderedDict(time=t)
            coords.update(field_coords)
            # IO is off the exchange path: detaching here keeps backends
            # (which convert to numpy) working during gradient rollouts
            backend.write(data.detach().unsqueeze(0), coords, std_name)

    # -- execution ----------------------------------------------------------------
    def _slot_aligned(self, time: np.datetime64, interval: np.timedelta64) -> bool:
        elapsed = (time - self.clock.start).astype("timedelta64[ns]")
        elapsed_ns = elapsed.astype(np.int64)
        interval_ns = as_timedelta(interval).astype(np.int64)
        return elapsed_ns > 0 and elapsed_ns % interval_ns == 0

    def _execute_time(self, time: np.datetime64) -> None:
        for slot in self.sequence.slots:
            if not self._slot_aligned(time, slot.interval):
                continue
            for action in slot.actions:
                if isinstance(action, RunAction):
                    comp = self.components[action.component]
                    comp.run(time)
                    self._record(action.component, time, comp.export_state)
                    self._io_write(action.component, time)
                elif isinstance(action, ConnectAction):
                    self._connectors[(action.src, action.dst)].execute(time)
                elif isinstance(action, MediateAction):
                    med = self.components[action.mediator]
                    med.run(time)
                    self._record(action.mediator, time, med.export_state)
                    self._io_write(action.mediator, time)

    def _check_not_exhausted(self) -> None:
        if self.clock.done():
            raise CouplingError(
                f"Driver clock exhausted: already at stop time "
                f"{self.clock.stop} — the run has completed. Call "
                "driver.reset() and driver.initialize(ics) to run again"
            )

    def _steps_impl(self) -> Iterator[tuple[np.datetime64, dict[str, State]]]:
        if not self._initialized:
            raise CouplingError("Driver.initialize(ics) must be called before running")
        for time in self.clock:
            self._execute_time(time)
            yield time, {n: c.export_state for n, c in self.components.items()}

    def steps(self) -> Iterator[tuple[np.datetime64, dict[str, State]]]:
        """Yield (time, {component: export State}) after every driver step —
        the notebook-inspection path (computation runs under inference mode)."""
        self._check_not_exhausted()
        return self._steps_gen()

    def _steps_gen(self) -> Iterator[tuple[np.datetime64, dict[str, State]]]:
        it = self._steps_impl()
        while True:
            with torch.inference_mode():
                try:
                    time, states = next(it)
                except StopIteration:
                    return
            yield time, states

    def run(self) -> "dict[str, object]":
        """Run to the clock's stop time; returns ``to_xarray()`` when
        collection is on, else an empty dict."""
        self._check_not_exhausted()
        with torch.inference_mode():
            for _ in self._steps_impl():
                pass
        return self.to_xarray() if self.collect else {}

    def rollout(self, n_steps: int) -> dict[str, State]:
        """Advance n driver steps keeping the autograd graph (when grad is
        enabled) — the coupled fine-tuning entry point. Returns each
        component's export State after the last step."""
        self._check_not_exhausted()
        it = self._steps_impl()
        states: dict[str, State] = {}
        for i in range(n_steps):
            try:
                _, states = next(it)
            except StopIteration:
                raise CouplingError(
                    f"rollout({n_steps}) ran past the clock stop time "
                    f"{self.clock.stop}: only {i} steps remained. Use "
                    f"n_steps <= clock.n_steps ({self.clock.n_steps}) or "
                    "call driver.reset() and re-initialize to run again"
                ) from None
        return states

    def reset(self) -> None:
        """Rewind for a fresh run: reset the clock, clear collected records,
        connector transfer history, and IO-ready state. The driver must be
        re-initialized (``initialize(ics)``) before running again."""
        self.clock.reset()
        self._records = {name: [] for name in self.components}
        for conn in self._connectors.values():
            conn.reset()
        self._io_ready.clear()
        self._initialized = False

    # -- inspection ------------------------------------------------------------------
    def describe(self) -> str:
        """Terraform-plan-style preview of the coupled system (works before
        initialize); see :func:`earth2studio.nvcoupler.api.describe`."""
        from .api import describe

        return describe(self)

    def _repr_html_(self) -> str:
        from .api import describe_html

        return describe_html(self)

    def probe(self, connector: str) -> dict[str, Field]:
        """Last fields exchanged on a connector, addressed as "src->dst"."""
        for conn in self._connectors.values():
            if conn.name == connector.replace(" ", ""):
                return dict(conn.last_transfer)
        known = [c.name for c in self._connectors.values()]
        raise KeyError(f"No connector {connector!r}; have {known}")

    def to_xarray(self) -> "dict[str, object]":
        """Collected export fields as one xarray.Dataset per component."""
        import xarray as xr

        out: dict[str, object] = {}
        for name, records in self._records.items():
            if not records:
                continue
            # size-1 time/batch dims on the fields themselves are stripped so
            # the leading axis is always the RING times, never a field's own
            # (stale) time coordinate
            by_var: dict[str, tuple[list, list, OrderedDict]] = {}
            for time, fields in records:
                for std, f in fields.items():
                    data, field_coords = self._strip_singletons(name, f)
                    times, datas, _ = by_var.setdefault(std, ([], [], field_coords))
                    times.append(time)
                    datas.append(data.detach().cpu().numpy())
            data_vars = {}
            coords: dict[str, object] = {}
            for std, (times, datas, field_coords) in by_var.items():
                dims = ("time", *field_coords.keys())
                data_vars[std] = (dims, np.stack(datas))
                coords["time"] = np.asarray(times, dtype="datetime64[ns]")
                coords.update(field_coords)
            out[name] = xr.Dataset(data_vars, coords=coords)
        return out

    def __repr__(self) -> str:
        return (
            f"Driver(components={sorted(self.components)}, "
            f"clock={self.clock!r}, connectors={[c.name for c in self._connectors.values()]})"
        )
