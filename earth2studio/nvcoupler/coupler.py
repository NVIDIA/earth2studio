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

"""Coupler: run several models on one clock, feeding each from the others.

The coupler is a :class:`~earth2studio.run.Runner`: ``Pipeline`` drives it over
work items exactly like the built-in runners, with one output stream per model.
Per item it fetches every model's initial condition from its bound sources,
then advances the models in run-sequence order, serving each model's declared
forcing (or a diagnostic's inputs) from the frames its sources have produced
so far. The coupler holds all runtime state; models and connectors hold none.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import DataSource, ForecastSource
from earth2studio.grids import (
    GridDefinition,
    Regridder,
    list_regridders,
    resolve_regridder,
    same_geometry,
)
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources
from earth2studio.nvcoupler._models import (
    SlotKind,
    _grid,
    _labels,
    _lead_times,
    _ModelSpec,
    inspect_model,
)
from earth2studio.nvcoupler._time import (
    DeltaLike,
    as_timedelta,
    fmt_timedelta,
    gcd_timedelta,
    is_multiple,
    nanoseconds,
)
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.errors import CouplingError, _suggest
from earth2studio.nvcoupler.sequence import RunSequence, derive_sequence
from earth2studio.run._fields import _lead_times as _rollout_leads
from earth2studio.run._fields import _map_field, _with_lead_time
from earth2studio.run.runner import (
    DataRequest,
    _fetch,
    _module_device,
    _nsteps,
    _seed_model,
    _signature_for_stream,
)
from earth2studio.run.work import WorkItem
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.time_statistics import time_statistic_metadata
from earth2studio.utils.type import CoordinateSystem

Source: TypeAlias = DataSource | ForecastSource
SourceSpec: TypeAlias = Source | tuple[Source | None, ...] | None
Slot: TypeAlias = tuple[SlotKind, int]

_ZERO = np.timedelta64(0, "ns")


class _Unavailable(CouplingError):
    """A source model has not produced the frames a destination needs."""


@dataclass
class _Provider:
    """One destination label served by one connector."""

    connector: Connector
    src: str
    label: str  # the destination's label, possibly qualified
    base: str  # the source's label
    modifier: str | None
    kind: SlotKind
    slot: int
    src_slot: int
    regridder: Regridder | None = None

    @property
    def window(self) -> np.timedelta64:
        if self.modifier is None:
            return _ZERO
        return cast(np.timedelta64, time_statistic_metadata(self.modifier)["window"])

    @property
    def end_offset(self) -> np.timedelta64:
        if self.modifier is None:
            return _ZERO
        return cast(
            np.timedelta64, time_statistic_metadata(self.modifier)["end_offset"]
        )


def _split(label: str) -> tuple[str, str | None]:
    base, _, modifier = label.partition(":")
    return base, (modifier or None)


def _lead(frame: xr.DataArray) -> np.timedelta64:
    return np.timedelta64(frame.coords["lead_time"].values[0], "ns")


class Coupler:
    """Advance coupled models on one clock and exchange their fields.

    ``models`` are ordinary :class:`~earth2studio.models.px.PrognosticModel` and
    :class:`~earth2studio.models.dx.DiagnosticModel` instances. The coupler reads
    each one's imports from ``forcing_coords()`` (prognostic) or
    ``input_coords()`` (diagnostic), its exports from ``output_coords()``, its
    cadence from the lead-time advance of one step, and its grid from the
    signature. Each :class:`Connector` declares which destination labels a
    source serves; what a connector cannot serve comes from the destination's
    bound data source, as in :class:`~earth2studio.run.PrognosticRunner`.

    The coupler is a :class:`~earth2studio.run.Runner`, so
    ``Pipeline(coupler, output=...)`` runs it over work items. Each step of
    :meth:`run_item` yields the outputs of the models that advanced at that
    step, keyed by model name (a second output slot of ``name`` is stream
    ``name.1``). The initial condition of every prognostic model is published
    as the first step.

    Parameters
    ----------
    models : Mapping[str, PrognosticModel | DiagnosticModel]
        Models keyed by the name used in connectors, sources and streams.
    connectors : Sequence[Connector], optional
        Field flows between models, by default none.
    sources : Mapping[str, DataSource | ForecastSource | tuple | None], optional
        Per model, one source or one per input-then-forcing slot, for whatever
        connectors do not serve; by default each model's ``default_sources()``.
    cadence : Mapping[str, str | np.timedelta64], optional
        Cadence of diagnostics not fed by any coupled model, by default a
        diagnostic runs at the (greatest common) cadence of the models feeding
        it. Prognostic cadences come from their declarations.
    sequence : RunSequence, optional
        Order in which models advance, by default derived from cadences and the
        same-step dependencies in the models' declarations.
    dt : str | np.timedelta64, optional
        Clock step, by default the greatest common divisor of the cadences.
    seed : int, optional
        Base seed for per-member seeding of stochastic models, by default 0
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        models: Mapping[str, Any],
        connectors: Sequence[Connector] = (),
        sources: Mapping[str, SourceSpec] | None = None,
        cadence: Mapping[str, DeltaLike] | None = None,
        sequence: RunSequence | None = None,
        dt: DeltaLike | None = None,
        seed: int = 0,
    ) -> None:
        if not models:
            raise CouplingError("Coupler needs at least one model")
        cadence = dict(cadence or {})
        unknown = sorted(set(cadence) - set(models))
        if unknown:
            raise CouplingError(f"cadence names unknown models {unknown}")
        self.specs: dict[str, _ModelSpec] = {
            name: inspect_model(name, model, cadence.get(name))
            for name, model in models.items()
        }
        self.models = dict(models)
        self.connectors = tuple(connectors)
        self.seed = seed
        self.device = _module_device(next(iter(models.values())))

        # Providers: per destination, per (slot kind, slot index), per label.
        self._providers: dict[str, dict[Slot, dict[str, _Provider]]] = {
            name: {} for name in self.specs
        }
        self._before: list[tuple[str, str]] = []
        for connector in self.connectors:
            self._resolve(connector)
        self._resolve_diagnostic_cadences()
        self._regridders: dict[tuple[int, str, int, SlotKind, int], Regridder] = {}
        for providers in self._providers.values():
            for slot_providers in providers.values():
                for provider in slot_providers.values():
                    provider.regridder = self._regridder(provider)

        self.cadences: dict[str, np.timedelta64] = {
            name: spec.required_cadence() for name, spec in self.specs.items()
        }
        self.dt = (
            gcd_timedelta(self.cadences.values()) if dt is None else as_timedelta(dt)
        )
        for name, model_cadence in self.cadences.items():
            if not is_multiple(model_cadence, self.dt):
                raise CouplingError(
                    f"Model {name!r} cadence {fmt_timedelta(model_cadence)} is not a "
                    f"multiple of the clock step {fmt_timedelta(self.dt)}"
                )
        self.sequence_derived = sequence is None
        self.sequence = (
            derive_sequence(self.cadences, self._before)
            if sequence is None
            else sequence
        )
        self.sequence.validate(self.cadences, self.dt, self._before)

        self._sources = self._bind_sources(sources or {})
        self._retention = self._retentions()
        self._warn_unconsumed()

    # -- construction -------------------------------------------------------

    def _resolve(self, connector: Connector) -> None:
        for endpoint in (connector.src, connector.dst):
            if endpoint not in self.specs:
                raise CouplingError(
                    f"Connector {connector.name} references unknown model "
                    f"{endpoint!r}." + _suggest(endpoint, self.specs)
                )
        src, dst = self.specs[connector.src], self.specs[connector.dst]
        explicit = connector.variables is not None
        if explicit:
            labels = list(cast(tuple[str, ...], connector.variables))
        else:
            labels = [label for label in dst.imports if _split(label)[0] in src.exports]
            if not labels:
                labels = [
                    label for label in dst.state if _split(label)[0] in src.exports
                ]
            if not labels:
                raise CouplingError(
                    f"Connector {connector.name}: {dst.name!r} declares no "
                    f"{dst.import_kind} or state variable that {src.name!r} exports. "
                    f"{dst.name!r} imports {sorted(dst.imports)} and holds state "
                    f"{sorted(dst.state)}; {src.name!r} exports {sorted(src.exports)}"
                )
        for label in labels:
            base, modifier = _split(label)
            if base not in src.exports:
                raise CouplingError(
                    f"Connector {connector.name}: {src.name!r} does not export "
                    f"{base!r}."
                    + _suggest(base, src.exports)
                    + f" It exports {sorted(src.exports)}"
                )
            if modifier is not None:
                try:
                    time_statistic_metadata(modifier)
                except ValueError as error:
                    raise CouplingError(
                        f"Connector {connector.name}: {label!r} is not a valid "
                        f"qualified label: {error}"
                    ) from error
                window = cast(
                    np.timedelta64, time_statistic_metadata(modifier)["window"]
                )
                if src.cadence is not None and not is_multiple(window, src.cadence):
                    raise CouplingError(
                        f"Connector {connector.name}: window of {label!r} is not a "
                        f"multiple of {src.name!r}'s cadence "
                        f"{fmt_timedelta(src.cadence)}"
                    )
            if label in dst.imports:
                kind, slot = dst.import_kind, dst.imports[label]
            elif label in dst.state:
                kind, slot = "state", dst.state[label]
            else:
                raise CouplingError(
                    f"Connector {connector.name}: {dst.name!r} declares no "
                    f"{dst.import_kind} or state variable {label!r}."
                    + _suggest(label, [*dst.imports, *dst.state])
                )
            existing = self._providers[dst.name].setdefault((kind, slot), {})
            if label in existing:
                raise CouplingError(
                    f"{dst.name!r} receives {label!r} from both "
                    f"{existing[label].connector.name} and {connector.name}"
                )
            provider = _Provider(
                connector, src.name, label, base, modifier, kind, slot,
                src.exports[base],
            )  # fmt: skip
            existing[label] = provider
            if self._same_step(dst, provider):
                self._before.append((src.name, dst.name))

    @staticmethod
    def _same_step(dst: _ModelSpec, provider: _Provider) -> bool:
        """Whether ``dst`` needs the frame its source produces in the same step."""
        if provider.kind == "input":
            # A diagnostic takes the newest frame; a window ending at the
            # current lead is complete without it.
            return provider.modifier is None or bool(provider.end_offset > _ZERO)
        if provider.kind == "state":
            return False
        leads = _lead_times(dst.forcing[provider.slot])
        offset = leads[-1] if leads is not None else _ZERO
        return bool(offset + provider.end_offset > _ZERO)

    def _resolve_diagnostic_cadences(self) -> None:
        for name, spec in self.specs.items():
            if spec.kind != "diagnostic" or spec.cadence is not None:
                continue
            feeding = {
                provider.src
                for slot_providers in self._providers[name].values()
                for provider in slot_providers.values()
            }
            cadences: list[np.timedelta64] = [
                cadence
                for src in feeding
                if (cadence := self.specs[src].cadence) is not None
            ]
            if cadences:
                spec.cadence = gcd_timedelta(cadences)

    def _regridder(self, provider: _Provider) -> Regridder | None:
        src, dst = self.specs[provider.src], self.specs[provider.connector.dst]
        source_grid = _grid(src.outputs[provider.src_slot])
        target_grid = _grid(dst.slot(provider.kind, provider.slot))
        key = (
            id(provider.connector),
            provider.src,
            provider.src_slot,
            provider.kind,
            provider.slot,
        )
        if key in self._regridders:
            return self._regridders[key]
        regridder = provider.connector.regridder
        if regridder is not None:
            for grid, bound, what in (
                (source_grid, regridder.source_grid, "source"),
                (target_grid, regridder.target_grid, "target"),
            ):
                if grid is not None and not same_geometry(grid, bound):
                    raise CouplingError(
                        f"Connector {provider.connector.name}: the regridder's "
                        f"{what}_grid does not match the {what} model's grid"
                    )
        elif (
            source_grid is not None
            and target_grid is not None
            and not same_geometry(source_grid, target_grid)
        ):
            regridder = _default_regridder(
                provider.connector.name, source_grid, target_grid
            )
        if regridder is not None:
            self._regridders[key] = regridder
        return regridder

    def _bind_sources(
        self, sources: Mapping[str, SourceSpec]
    ) -> dict[str, tuple[Source | None, ...]]:
        unknown = sorted(set(sources) - set(self.specs))
        if unknown:
            raise CouplingError(f"sources names unknown models {unknown}")
        bound: dict[str, tuple[Source | None, ...]] = {}
        for name, spec in self.specs.items():
            nslots = len(spec.inputs) + len(spec.forcing)
            given: Any = sources.get(name)
            if given is None:
                given = recommended_sources(spec.model)
            if not isinstance(given, tuple):
                given = (given,) + (None,) * (nslots - 1)
            if len(given) != nslots:
                raise CouplingError(
                    f"Model {name!r} has {nslots} input and forcing slots but "
                    f"{len(given)} sources were given"
                )
            bound[name] = tuple(given)
            # Import slots with unserved labels and no source fail now, not mid-run.
            for index, signature in enumerate(spec.import_slots):
                offset = len(spec.inputs) if spec.kind == "prognostic" else 0
                unfed = self._unfed(spec, spec.import_kind, index)
                if unfed and given[offset + index] is None:
                    raise CouplingError(
                        f"Model {name!r} {spec.import_kind} slot {index} needs "
                        f"{unfed} from a data source, but no connector serves them "
                        "and no source is bound; add a Connector or pass sources="
                    )
        return bound

    def _unfed(self, spec: _ModelSpec, kind: SlotKind, index: int) -> list[str]:
        providers = self._providers[spec.name].get((kind, index), {})
        return [
            label for label in _labels(spec.slot(kind, index)) if label not in providers
        ]

    def _source(self, spec: _ModelSpec, kind: SlotKind, index: int) -> Source | None:
        offset = len(spec.inputs) if kind == "forcing" else 0
        return self._sources[spec.name][offset + index]

    def _retentions(self) -> dict[str, np.timedelta64]:
        """How far back each model's frames are kept, per its consumers."""
        retention = {name: 2 * cadence for name, cadence in self.cadences.items()}
        for dst, providers in self._providers.items():
            for (kind, slot), slot_providers in providers.items():
                for provider in slot_providers.values():
                    dst_spec = self.specs[dst]
                    offset: np.timedelta64 = _ZERO
                    if kind == "forcing":
                        leads = _lead_times(dst_spec.forcing[slot])
                        if leads is not None:
                            offset = np.timedelta64(abs(int(leads[-1])), "ns")
                    needed = (
                        provider.window
                        + self.cadences[dst]
                        + self.cadences[provider.src]
                        + offset
                    )
                    if needed > retention[provider.src]:
                        retention[provider.src] = needed
        return retention

    def _warn_unconsumed(self) -> None:
        consumed = {
            (provider.src, provider.base)
            for providers in self._providers.values()
            for slot_providers in providers.values()
            for provider in slot_providers.values()
        }
        for name, spec in self.specs.items():
            if spec.kind == "diagnostic":
                idle = [
                    label for label in spec.exports if (name, label) not in consumed
                ]
                if idle and len(self.specs) > 1:
                    logger.debug(f"Diagnostic {name!r} outputs {idle} feed no model")

    # -- Runner protocol ----------------------------------------------------

    def to(self, device: torch.device | str) -> Coupler:
        """Move every model and regridder to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        for name, model in self.models.items():
            moved = model.to(self.device)
            if moved is not None:
                self.models[name] = moved
                self.specs[name].model = moved
        for regridder in self._regridders.values():
            regridder.to(self.device)
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of clock steps needed to reach ``final_lead_time``."""
        return _nsteps(np.timedelta64(final_lead_time, "ns"), self.dt)

    def _runs(self, name: str, final_lead_time: np.timedelta64) -> int:
        self.nsteps(final_lead_time)
        return int(np.timedelta64(final_lead_time, "ns") // self.cadences[name])

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each stream's planning signature for one item.

        One stream per model output slot. Fixed dimensions are ``lead_time``,
        covering the initial condition and every step through
        ``final_lead_time``, then the slot's field dimensions; ``time`` and
        ``ensemble`` stay in the dynamic prefix for supervision to concretize.
        """
        coords: dict[str, CoordinateSystem] = {}
        for name, spec in self.specs.items():
            runs = self._runs(name, final_lead_time)
            for index, signature in enumerate(spec.outputs):
                if spec.kind == "prognostic" and _lead_times(signature) is not None:
                    leads = _rollout_leads(signature, runs)
                else:
                    leads = np.arange(runs + 1) * self.cadences[name]
                coords[spec.stream(index)] = _signature_for_stream(signature, leads)
        return coords

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Describe every fetch ``item`` makes, or ``None`` when inputs need ``ics``.

        Returns ``None`` when a prognostic input slot has no bound source, since
        :meth:`run_item` then needs its initial condition passed as ``ics``.
        """
        requests: list[DataRequest] = []
        for name, spec in self.specs.items():
            runs = self._runs(name, item.final_lead_time)
            if spec.kind == "prognostic":
                for index, signature in enumerate(spec.inputs):
                    source = self._source(spec, "input", index)
                    if source is None:
                        return None
                    requests.append(
                        self._request(source, item, _labels(signature), signature)
                    )
                if runs == 0:
                    continue
                output_leads = self._output_leads(spec)
                for index, signature in enumerate(spec.forcing):
                    source = self._source(spec, "forcing", index)
                    unfed = self._unfed(spec, "forcing", index)
                    leads = _lead_times(signature)
                    # Coupled labels come from the source too where the window
                    # reaches before the initial condition.
                    initial = list(unfed)
                    if leads is not None and leads[0] < _ZERO:
                        initial = _labels(signature)
                    if source is None or not initial:
                        continue
                    if leads is None:
                        requests.append(self._request(source, item, initial, signature))
                        continue
                    requests.append(
                        self._request(source, item, initial, signature, leads)
                    )
                    if not unfed:
                        continue
                    for run in range(1, runs):
                        required = (
                            leads[-1] + output_leads + (run - 1) * self.cadences[name]
                        )
                        requests.append(
                            self._request(source, item, unfed, signature, required)
                        )
            else:
                for index, signature in enumerate(spec.inputs):
                    source = self._source(spec, "input", index)
                    unfed = self._unfed(spec, "input", index)
                    if source is None or not unfed:
                        continue
                    for run in range(runs + 1):
                        lead = np.array([run * self.cadences[name]])
                        requests.append(
                            self._request(source, item, unfed, signature, lead)
                        )
        return tuple(requests)

    def _output_leads(self, spec: _ModelSpec) -> np.ndarray:
        leads = [
            values for sig in spec.outputs if (values := _lead_times(sig)) is not None
        ]
        return np.unique(np.concatenate(leads)) if leads else np.array([_ZERO])

    @staticmethod
    def _request(
        source: Source,
        item: WorkItem,
        labels: Sequence[str],
        signature: CoordinateSystem,
        leads: np.ndarray | None = None,
    ) -> DataRequest:
        if leads is None:
            declared = _lead_times(signature)
            leads = declared if declared is not None else np.array([_ZERO])
        return DataRequest(
            source,
            np.array([item.time]),
            np.array(list(labels)),
            np.asarray(leads).astype("timedelta64[ns]"),
        )

    def run_item(
        self,
        item: WorkItem,
        ics: Mapping[str, xr.DataArray | tuple[xr.DataArray, ...]] | None = None,
    ) -> Iterator[Mapping[str, xr.DataArray]]:
        """Run one work item, yielding each step's outputs keyed by stream.

        Parameters
        ----------
        item : WorkItem
            Reference time and final lead time. The final lead time must be a
            multiple of the clock step.
        ics : Mapping[str, xr.DataArray | tuple[xr.DataArray, ...]], optional
            Initial fields for prognostic models already held in memory, one
            array per input slot, in place of fetching them from the bound
            sources. Forcing windows are still served by connectors and sources.

        Yields
        ------
        Mapping[str, xr.DataArray]
            Per step, the output of every model that advanced, with dimensions
            ``time`` (the item's time) then ``lead_time`` and the field
            dimensions. The first step holds the prognostic initial conditions
            and the diagnostics computable from them.
        """
        if len(item.member_ids) > 1:
            raise ValueError("Coupler does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        for model in self.models.values():
            _seed_model(self.seed, item.time, member_id, model)
        return _Run(self, item, dict(ics or {})).steps()

    # -- inspection ---------------------------------------------------------

    def describe(self) -> str:
        """Return a plain-text plan: models, exchanges and run sequence."""
        from earth2studio.nvcoupler._describe import describe

        return describe(self)

    def _repr_html_(self) -> str:
        from earth2studio.nvcoupler._describe import describe_html

        return describe_html(self)

    def __repr__(self) -> str:
        return (
            f"Coupler(models={list(self.specs)}, dt={fmt_timedelta(self.dt)}, "
            f"connectors={[c.name for c in self.connectors]})"
        )


def _default_regridder(
    name: str, source_grid: GridDefinition, target_grid: GridDefinition
) -> Regridder:
    methods = list_regridders(source_grid, target_grid)
    for method in ("bilinear", "nearest"):
        if method in methods:
            return resolve_regridder(
                source_grid, target_grid, method, fill_value=np.nan
            )
    raise CouplingError(
        f"Connector {name}: no built-in regridder maps {source_grid.topology} to "
        f"{target_grid.topology} grids; pass regridder="
    )


class _Run:
    """Per-item runtime state: frame buffers, model states and the clock."""

    def __init__(
        self,
        coupler: Coupler,
        item: WorkItem,
        ics: dict[str, xr.DataArray | tuple[xr.DataArray, ...]],
    ) -> None:
        self.coupler = coupler
        self.item = item
        self.t0 = np.datetime64(item.time, "ns")
        self.nsteps = coupler.nsteps(item.final_lead_time)
        unknown = sorted(set(ics) - set(coupler.specs))
        if unknown:
            raise CouplingError(f"ics names unknown models {unknown}")
        self.ics = ics
        # Frames per (model, output slot), one lead time each, sorted by lead.
        self.rings: dict[tuple[str, int], list[xr.DataArray]] = {
            (name, index): []
            for name, spec in coupler.specs.items()
            for index in range(len(spec.outputs))
        }
        self.x: dict[str, tuple[xr.DataArray, ...]] = {}
        self.y: dict[str, tuple[xr.DataArray, ...]] = {}
        self.state: dict[str, Any] = {}
        self.runs: dict[str, int] = {name: 0 for name in coupler.specs}

    # -- clock --------------------------------------------------------------

    def steps(self) -> Iterator[Mapping[str, xr.DataArray]]:
        coupler = self.coupler
        outputs: dict[str, xr.DataArray] = {}
        for name, spec in coupler.specs.items():
            if spec.kind == "prognostic":
                outputs.update(self._initialize(spec))
        for slot in coupler.sequence.slots:
            for name in slot.models:
                spec = coupler.specs[name]
                if spec.kind == "diagnostic":
                    outputs.update(self._diagnostic_if_ready(spec, _ZERO))
        if outputs:
            yield outputs
        for step in range(1, self.nsteps + 1):
            lead = step * coupler.dt
            outputs = {}
            for slot in coupler.sequence.slots:
                if nanoseconds(lead) % nanoseconds(slot.interval):
                    continue
                for name in slot.models:
                    spec = coupler.specs[name]
                    if spec.kind == "prognostic":
                        outputs.update(self._run_prognostic(spec, lead))
                    else:
                        outputs.update(self._diagnostic_if_ready(spec, lead))
            if outputs:
                yield outputs

    def _diagnostic_if_ready(
        self, spec: _ModelSpec, lead: np.timedelta64
    ) -> dict[str, xr.DataArray]:
        """Run a diagnostic, skipping the step while its inputs cannot exist yet.

        A diagnostic over a time window, such as ``z1000:max:24h``, has nothing
        to compute until its sources have produced that window; the initial
        condition may also lack output-only variables. Later gaps are errors.
        """
        try:
            return self._run_diagnostic(spec, lead)
        except _Unavailable:
            if lead > self._warmup(spec):
                raise
            return {}

    def _warmup(self, spec: _ModelSpec) -> np.timedelta64:
        """Lead time before which a diagnostic's windowed inputs may be incomplete."""
        windows = [
            provider.window + provider.end_offset
            for slot_providers in self.coupler._providers[spec.name].values()
            for provider in slot_providers.values()
        ]
        return max(windows, default=_ZERO)

    # -- model advances -----------------------------------------------------

    def _initialize(self, spec: _ModelSpec) -> dict[str, xr.DataArray]:
        """Fetch a prognostic model's inputs and publish its initial condition."""
        if spec.name in self.ics:
            given = self.ics[spec.name]
            arrays = given if isinstance(given, tuple) else (given,)
            if len(arrays) != len(spec.inputs):
                raise CouplingError(
                    f"ics[{spec.name!r}] holds {len(arrays)} arrays but the model "
                    f"declares {len(spec.inputs)} input slots"
                )
            x = tuple(self._conform_ic(spec, array) for array in arrays)
        else:
            fields: list[xr.DataArray] = []
            for index, signature in enumerate(spec.inputs):
                source = self.coupler._source(spec, "input", index)
                if source is None:
                    raise CouplingError(
                        f"Model {spec.name!r} input slot {index} has no bound "
                        "source; pass sources= or its initial condition in ics="
                    )
                request = self.coupler._request(
                    source, self.item, _labels(signature), signature
                )
                fields.append(_fetch(request, signature, self.coupler.device))
            x = tuple(fields)
        self.x[spec.name] = x
        published = initial_condition(x)
        frames = published if isinstance(published, tuple) else (published,)
        return self._publish(spec, frames)

    def _conform_ic(self, spec: _ModelSpec, array: xr.DataArray) -> xr.DataArray:
        if not isinstance(array, xr.DataArray):
            raise CouplingError(f"ics[{spec.name!r}] must hold xarray DataArrays")
        if "lead_time" not in array.dims:
            raise CouplingError(
                f"ics[{spec.name!r}] needs a 'lead_time' dimension matching the "
                "model's input_coords()"
            )
        if "time" not in array.dims:
            array = array.expand_dims(time=[self.t0], axis=0)
        return array

    def _run_prognostic(
        self, spec: _ModelSpec, lead: np.timedelta64
    ) -> dict[str, xr.DataArray]:
        model = spec.model
        if self.runs[spec.name] == 0:
            x = self._overwrite_state(spec, self.x.pop(spec.name), initial=True)
            forcing = [
                self._assemble(spec, "forcing", index, self._forcing_leads(spec, index))
                for index in range(len(spec.forcing))
            ]
            y, state = model.initialize(*x, *forcing)
        else:
            y_in = self._overwrite_state(spec, self.y[spec.name], initial=False)
            forcing = [
                self._assemble(spec, "forcing", index, self._forcing_leads(spec, index))
                for index, signature in enumerate(spec.forcing)
                if "lead_time" in signature.dims  # static slots are init-only
            ]
            y, state = model.step(*y_in, *forcing, state=self.state[spec.name])
        outputs = y if isinstance(y, tuple) else (y,)
        if len(outputs) != len(spec.outputs):
            raise CouplingError(
                f"Model {spec.name!r} returned {len(outputs)} outputs but declares "
                f"{len(spec.outputs)} output slots"
            )
        self.y[spec.name], self.state[spec.name] = outputs, state
        self.runs[spec.name] += 1
        return self._publish(spec, outputs)

    def _run_diagnostic(
        self, spec: _ModelSpec, lead: np.timedelta64
    ) -> dict[str, xr.DataArray]:
        inputs = [
            self._assemble(spec, "input", index, np.array([lead]))
            for index in range(len(spec.inputs))
        ]
        y = spec.model(*inputs)
        outputs = y if isinstance(y, tuple) else (y,)
        if len(outputs) != len(spec.outputs):
            raise CouplingError(
                f"Diagnostic {spec.name!r} returned {len(outputs)} outputs but "
                f"declares {len(spec.outputs)} output slots"
            )
        outputs = tuple(_with_lead_time(out, lead) for out in outputs)
        self.runs[spec.name] += 1
        return self._publish(spec, outputs)

    def _forcing_leads(self, spec: _ModelSpec, index: int) -> np.ndarray:
        """Lead times a forcing slot needs: its window at first, then newest frames."""
        declared = _lead_times(spec.forcing[index])
        if declared is None:
            return np.array([_ZERO])
        if self.runs[spec.name] == 0:
            return declared
        return declared[-1] + self._y_leads(spec)

    def _y_leads(self, spec: _ModelSpec) -> np.ndarray:
        leads = [
            out.coords["lead_time"].values.astype("timedelta64[ns]")
            for out in self.y[spec.name]
            if "lead_time" in out.dims
        ]
        return np.unique(np.concatenate(leads))

    # -- publication and buffers --------------------------------------------

    def _publish(
        self, spec: _ModelSpec, outputs: tuple[xr.DataArray, ...]
    ) -> dict[str, xr.DataArray]:
        published: dict[str, xr.DataArray] = {}
        for index, out in enumerate(outputs):
            if "time" not in out.dims:
                out = out.expand_dims(time=[self.t0], axis=0)
            if "lead_time" not in out.dims:
                raise CouplingError(
                    f"Model {spec.name!r} output lacks a 'lead_time' dimension"
                )
            ring = self.rings[(spec.name, index)]
            for position in range(out.sizes["lead_time"]):
                frame = out.isel(lead_time=[position])
                ring[:] = [f for f in ring if _lead(f) != _lead(frame)]
                ring.append(frame)
            ring.sort(key=_lead)
            horizon = _lead(ring[-1]) - self.coupler._retention[spec.name]
            ring[:] = [f for f in ring if _lead(f) >= horizon]
            published[spec.stream(index)] = out
        return published

    def _pull(self, provider: _Provider, required: np.ndarray) -> xr.DataArray:
        """One label from a source model's buffer at the required lead times."""
        cadence = self.coupler.cadences[provider.src]
        frames = [
            frame
            for frame in self.rings[(provider.src, provider.src_slot)]
            if provider.base in _labels(frame)
        ]
        if not frames:
            raise _Unavailable(
                f"{provider.src!r} has produced no frame of {provider.base!r} yet"
            )
        pieces = []
        if provider.modifier is None:
            for lead in required:
                lead = np.timedelta64(lead, "ns")
                candidates = [f for f in frames if _lead(f) <= lead]
                if not candidates:
                    raise _Unavailable(
                        f"{provider.src!r} has no frame of {provider.base!r} at or "
                        f"before lead {fmt_timedelta(lead)}; its earliest is "
                        f"{fmt_timedelta(_lead(frames[0]))}"
                    )
                frame = candidates[-1]
                if lead - _lead(frame) >= cadence:
                    raise _Unavailable(
                        f"{provider.connector.dst!r} needs {provider.base!r} from "
                        f"{provider.src!r} at lead {fmt_timedelta(lead)} but the "
                        f"newest frame is at {fmt_timedelta(_lead(frame))}; advance "
                        f"{provider.src!r} first in the run sequence"
                    )
                pieces.append(
                    frame.sel(variable=[provider.base]).assign_coords(
                        lead_time=np.array([lead])
                    )
                )
        else:
            from earth2studio.utils.time_statistics import apply_time_statistic

            block = xr.concat(
                [f.sel(variable=[provider.base]) for f in frames], "lead_time"
            )
            axis = block.dims.index("lead_time")
            for lead in required:
                lead = np.timedelta64(lead, "ns")
                try:
                    reduced = apply_time_statistic(
                        block, provider.modifier, lead, cadence, dimension="lead_time"
                    )
                except ValueError as error:
                    raise _Unavailable(
                        f"{provider.connector.dst!r} needs {provider.label!r} at lead "
                        f"{fmt_timedelta(lead)} but {provider.src!r} has not produced "
                        f"the window: {error}"
                    ) from error
                pieces.append(
                    reduced.expand_dims(lead_time=np.array([lead]), axis=axis)
                )
        field = xr.concat(pieces, "lead_time")
        if provider.regridder is not None:
            field = provider.regridder(field)
        if provider.connector.transform is not None:
            field = self._transform(provider, field)
        return field.assign_coords(variable=np.array([provider.label]))

    @staticmethod
    def _transform(provider: _Provider, field: xr.DataArray) -> xr.DataArray:
        result = provider.connector.transform(field)  # type: ignore[misc]
        if not isinstance(result, xr.DataArray) or result.dims != field.dims:
            raise CouplingError(
                f"Connector {provider.connector.name}: transform must return a "
                f"DataArray with dimensions {field.dims}"
            )
        # The vertical coordinate may change (interpolation onto the
        # destination's levels); every other coordinate must survive.
        for dim in field.dims:
            if dim == "level" or dim not in field.coords:
                continue
            if not np.array_equal(result.coords[dim].values, field.coords[dim].values):
                raise CouplingError(
                    f"Connector {provider.connector.name}: transform changed the "
                    f"{dim!r} coordinate"
                )
        return result

    def _assemble(
        self, spec: _ModelSpec, kind: SlotKind, index: int, required: np.ndarray
    ) -> xr.DataArray:
        """Build one slot's array at ``required`` leads from connectors and source."""
        signature = spec.slot(kind, index)
        labels = _labels(signature)
        providers = self.coupler._providers[spec.name].get((kind, index), {})
        source = self.coupler._source(spec, kind, index)
        pieces: dict[str, xr.DataArray] = {}
        from_source = [label for label in labels if label not in providers]
        for label, provider in providers.items():
            try:
                pieces[label] = self._pull(provider, required)
            except _Unavailable as error:
                if source is None:
                    raise _Unavailable(
                        f"{error}. Bind a source for {spec.name!r} {kind} slot "
                        f"{index} to supply {label!r} where no coupled frame exists"
                    ) from error
                from_source.append(label)
        if from_source:
            if source is None:
                raise CouplingError(
                    f"Model {spec.name!r} {kind} slot {index} needs {from_source} "
                    "from a data source but none is bound"
                )
            subset = coord_array_like(signature, {"variable": np.array(from_source)})
            request = self.coupler._request(
                source, self.item, from_source, signature, required
            )
            fetched = _fetch(request, subset, self.coupler.device)
            for label in from_source:
                pieces[label] = fetched.sel(variable=[label])
        try:
            field = xr.concat(
                [pieces[label] for label in labels], "variable", join="exact"
            )
        except (ValueError, KeyError) as error:
            raise CouplingError(
                f"Model {spec.name!r} {kind} slot {index}: delivered fields do not "
                f"share one grid and lead time: {error}"
            ) from error
        try:
            field = _map_field(field, signature)
        except ValueError as error:
            raise CouplingError(
                f"Model {spec.name!r} {kind} slot {index}: delivered field does not "
                f"match the declared signature: {error}"
            ) from error
        if "lead_time" not in signature.dims:
            field = field.isel(lead_time=0, drop=True)
        leading = [dim for dim in field.dims if dim not in signature.dims]
        order = [dim for dim in signature.dims if dim in field.dims]
        return field.transpose(*leading, *order)

    def _overwrite_state(
        self, spec: _ModelSpec, arrays: tuple[xr.DataArray, ...], initial: bool
    ) -> tuple[xr.DataArray, ...]:
        """Replace coupled state variables in the fields fed to the next advance."""
        providers = {
            slot: slot_providers
            for (kind, slot), slot_providers in self.coupler._providers[
                spec.name
            ].items()
            if kind == "state"
        }
        if not providers:
            return arrays
        result = list(arrays)
        for slot, slot_providers in providers.items():
            if initial:
                # Inputs are keyed by input slot; the state's output slot maps to
                # the input slot carrying the same label.
                targets = {
                    label: next(
                        i for i, sig in enumerate(spec.inputs) if label in _labels(sig)
                    )
                    for label in slot_providers
                }
            else:
                targets = {label: slot for label in slot_providers}
            for label, provider in slot_providers.items():
                target = result[targets[label]]
                leads = target.coords["lead_time"].values.astype("timedelta64[ns]")
                if initial:
                    leads = leads[-1:]
                piece = self._pull(provider, leads).sel(variable=label)
                for dim in piece.dims:
                    mismatch = dim != "lead_time" and not np.array_equal(
                        piece.coords[dim].values, target.coords[dim].values
                    )
                    if dim not in target.dims or mismatch:
                        raise CouplingError(
                            f"Connector {provider.connector.name}: {label!r} does "
                            f"not share {spec.name!r}'s {dim!r} coordinate; the "
                            "delivered field must match the state grid exactly"
                        )
                target = target.copy(deep=True)
                target.loc[{"variable": label, "lead_time": leads}] = piece
                result[targets[label]] = target
        return tuple(result)
