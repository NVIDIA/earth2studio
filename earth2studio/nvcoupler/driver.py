# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Driver for DataArray-native coupled inference."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import xarray as xr
from loguru import logger

from .clock import Clock
from .component import Component
from .connector import Connector
from .errors import CouplingError, UnmatchedImportError
from .field import Field, State
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
    """Execute a component graph on a shared clock."""

    def __init__(
        self,
        components: dict[str, Component],
        sequence: RunSequence | str | None = None,
        clock: Clock | None = None,
        connectors: list[Connector | tuple[str, str]] | None = None,
        collect: bool = True,
        allow_unfed_imports: bool = False,
    ):
        if clock is None:
            raise CouplingError("Driver needs a clock")
        self.components = dict(components)
        self.clock = clock
        self.collect = collect
        self.allow_unfed_imports = allow_unfed_imports
        self._connectors: dict[tuple[str, str], Connector] = {}
        for item in connectors or []:
            if not isinstance(item, Connector):
                source, destination = item
                unknown = [
                    name
                    for name in (source, destination)
                    if name not in self.components
                ]
                if unknown:
                    raise CouplingError(
                        f"Connection {source!r}->{destination!r} references "
                        f"unknown components {unknown}"
                    )
                item = Connector(self.components[source], self.components[destination])
            self._connectors[(item.src.name, item.dst.name)] = item
        self.sequence_derived = sequence is None
        self.sequence = (
            derive_sequence(self.components, self._connectors.values())
            if sequence is None
            else parse_run_sequence(sequence)
            if isinstance(sequence, str)
            else sequence
        )
        self._records: dict[str, list[tuple[np.datetime64, dict[str, Field]]]] = {
            name: [] for name in self.components
        }
        self._initialized = False

    def initialize(self, ics: dict[str, xr.DataArray] | None = None) -> None:
        ics = ics or {}
        self.sequence.validate(self.components, self.clock.dt)
        for action in self.sequence.connections():
            key = (action.src, action.dst)
            if key not in self._connectors:
                self._connectors[key] = Connector(
                    self.components[action.src], self.components[action.dst]
                )
        for connector in self._connectors.values():
            connector.match()
        self._check_unfed_imports()
        for name, component in self.components.items():
            component.realize(self.clock)
            if name in ics:
                component.initialize(ics[name])
            elif not component.requires_ic:
                component.initialize()
            else:
                raise CouplingError(
                    f"Component {name!r} needs a DataArray initial condition"
                )
            self._record(name, self.clock.start, component.export_state)
        self._warn_unconsumed_exports()
        self._warn_memory()
        self._initialized = True

    def _check_unfed_imports(self) -> None:
        fed: dict[str, set[str]] = {name: set() for name in self.components}
        for (_, destination), connector in self._connectors.items():
            fed[destination] |= set(connector.match())
        available = {
            name: list(component.export_names)
            for name, component in self.components.items()
        }
        for name, component in self.components.items():
            for field in component.import_names:
                if field in fed[name]:
                    continue
                if self.allow_unfed_imports:
                    logger.warning(
                        "Component {!r} import {!r} is not fed by a connector",
                        name,
                        field,
                    )
                else:
                    raise UnmatchedImportError(name, field, available)

    def _warn_unconsumed_exports(self) -> None:
        consumed: set[tuple[str, str]] = set()
        for (source, _), connector in self._connectors.items():
            consumed |= {(source, field) for field in connector.match()}
        for name, component in self.components.items():
            idle = [
                field
                for field in component.export_names
                if (name, field) not in consumed
            ]
            if idle:
                logger.warning(
                    "Component {!r} exports {} but no connector consumes them",
                    name,
                    idle,
                )

    def _warn_memory(self) -> None:
        if not self.collect:
            return
        total = 0.0
        for component in self.components.values():
            per_ring = sum(
                field.array.nbytes for field in component.export_state.values()
            )
            rings = self.clock.n_steps * (
                self.clock.dt.astype(np.int64) / component.timestep.astype(np.int64)
            )
            total += per_ring * max(rings, 0)
        if total > MEMORY_WARN_BYTES:
            logger.warning(
                "In-memory collection will hold approximately {:.1f} GB",
                total / 1e9,
            )

    def _record(self, name: str, time: np.datetime64, state: State) -> None:
        if not self.collect:
            return
        self._records[name].append(
            (time, {standard: field.clone() for standard, field in state.items()})
        )

    def _slot_aligned(self, time: np.datetime64, interval: np.timedelta64) -> bool:
        elapsed = (time - self.clock.start).astype("timedelta64[ns]")
        elapsed_ns = elapsed.astype(np.int64)
        interval_ns: np.int64 = interval.astype("timedelta64[ns]").astype(np.int64)
        return elapsed_ns > 0 and elapsed_ns % interval_ns == 0

    def _execute_time(self, time: np.datetime64) -> None:
        for slot in self.sequence.slots:
            if not self._slot_aligned(time, slot.interval):
                continue
            for action in slot.actions:
                if isinstance(action, RunAction):
                    component = self.components[action.component]
                    component.run(time)
                    self._record(action.component, time, component.export_state)
                elif isinstance(action, ConnectAction):
                    self._connectors[(action.src, action.dst)].execute(time)
                elif isinstance(action, MediateAction):
                    mediator = self.components[action.mediator]
                    mediator.run(time)
                    self._record(action.mediator, time, mediator.export_state)

    def _check_not_exhausted(self) -> None:
        if self.clock.done():
            raise CouplingError(
                f"Driver clock exhausted at {self.clock.stop}; reset first"
            )

    def _steps_impl(
        self,
    ) -> Iterator[tuple[np.datetime64, dict[str, State]]]:
        if not self._initialized:
            raise CouplingError("Driver.initialize(ics) must be called first")
        for time in self.clock:
            self._execute_time(time)
            yield (
                time,
                {
                    name: component.export_state
                    for name, component in self.components.items()
                },
            )

    def steps(self) -> Iterator[tuple[np.datetime64, dict[str, State]]]:
        self._check_not_exhausted()
        return self._steps_impl()

    def run(self) -> dict[str, xr.Dataset]:
        self._check_not_exhausted()
        for _ in self._steps_impl():
            pass
        return self.to_xarray() if self.collect else {}

    def reset(self) -> None:
        self.clock.reset()
        self._records = {name: [] for name in self.components}
        for connector in self._connectors.values():
            connector.reset()
        self._initialized = False

    def describe(self) -> str:
        from .api import describe

        return describe(self)

    def _repr_html_(self) -> str:
        from .api import describe_html

        return describe_html(self)

    def probe(self, connector: str) -> dict[str, Field]:
        for candidate in self._connectors.values():
            if candidate.name == connector.replace(" ", ""):
                return dict(candidate.last_transfer)
        raise KeyError(
            f"No connector {connector!r}; "
            f"have {[item.name for item in self._connectors.values()]}"
        )

    def to_xarray(self) -> dict[str, xr.Dataset]:
        output: dict[str, xr.Dataset] = {}
        for name, records in self._records.items():
            variables: dict[str, list[xr.DataArray]] = {}
            times: dict[str, list[np.datetime64]] = {}
            for time, fields in records:
                for standard, field in fields.items():
                    variables.setdefault(standard, []).append(field.array)
                    times.setdefault(standard, []).append(time)
            if not variables:
                continue
            arrays = []
            for standard, values in variables.items():
                arrays.append(
                    xr.concat(
                        values,
                        xr.IndexVariable(
                            "time",
                            np.asarray(times[standard], dtype="datetime64[ns]"),
                        ),
                        join="exact",
                    ).rename(standard)
                )
            output[name] = xr.merge(arrays, join="outer")
        return output

    @property
    def connectors(self) -> tuple[Connector, ...]:
        return tuple(self._connectors.values())

    def __repr__(self) -> str:
        return (
            f"Driver(components={sorted(self.components)}, clock={self.clock!r}, "
            f"connectors={[connector.name for connector in self._connectors.values()]})"
        )
