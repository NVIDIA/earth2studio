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

"""Coupled runner: several prognostic models feeding each other's forcing slots.

There are no component wrappers. A prognostic model *is* the component: its
``input_coords``, ``output_coords`` and ``forcing_coords`` say which variables it
produces and consumes and how large its step is, and its ``create_iterator``
accepts new forcing between steps. :class:`CoupledRunner` probes those, wires
producers to forcing slots, and drives the models forward on one shared clock as
a :class:`~earth2studio.run.Runner`, so it plugs into
:class:`~earth2studio.run.Pipeline` unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources
from earth2studio.run._fields import _lead_times, _map_field
from earth2studio.run.runner import (
    DataRequest,
    Source,
    _fetch,
    _module_device,
    _nsteps,
    _seed_model,
    _signature_for_stream,
    _slots,
)
from earth2studio.run.work import WorkItem
from earth2studio.utils.type import CoordinateSystem

_NON_SPATIAL = ("batch", "time", "ensemble", "lead_time", "variable")


@dataclass(frozen=True)
class Link:
    """One producer feeding one forcing slot of a consumer.

    Links are optional: :class:`CoupledRunner` wires a forcing slot to the only
    other model that produces all of its variables. Declare a link to override
    that, to pick a reduction, or to bridge differing grids or units.

    Attributes
    ----------
    src : str
        Name of the producing model
    dst : str
        Name of the consuming model
    slot : int, optional
        Index of ``dst``'s forcing slot, by default inferred as the only slot
        whose variables ``src`` produces
    reduce : {"last", "mean"}
        What ``dst`` receives for the step it is about to take, from ``t`` to
        ``t + step``. ``"last"``: ``src``'s latest output valid at or before
        ``t``, held when ``src`` is slower. ``"mean"``: the mean of ``src``'s
        outputs in ``(t, t + step]``; ``src`` is then advanced first, and must
        be at least as fast as ``dst``. By default ``"last"``
    transform : Callable[[xr.DataArray], xr.DataArray], optional
        Applied to the delivered field before it enters ``dst``, e.g. a
        :class:`~earth2studio.grids.Regridder` or a unit conversion, by default
        none
    """

    src: str
    dst: str
    slot: int | None = None
    reduce: Literal["last", "mean"] = "last"
    transform: Callable[[xr.DataArray], xr.DataArray] | None = None


@dataclass
class _Probe:
    """What one model declares, read once at construction."""

    model: PrognosticModel
    input_signature: CoordinateSystem
    forcing_signatures: tuple[CoordinateSystem, ...]
    output_signature: CoordinateSystem
    step: np.timedelta64
    sources: tuple[Source | None, ...]  # Input slot, then forcing slots.


@dataclass
class _Rollout:
    """Mutable per-item state of one model."""

    iterator: Any = None
    lead: np.timedelta64 = np.timedelta64(0, "h")
    y_leads: np.ndarray | None = None
    published: list[xr.DataArray] | None = None
    history: list[tuple[np.timedelta64, xr.DataArray]] | None = None


class CoupledRunner:
    """Run several prognostic models that exchange fields, as one runner.

    Each model is stepped at its own cadence on a shared clock. Whenever a
    model's forcing slot is wired to another model, the producer's outputs are
    sent to the consumer through the iterator's ``send`` in place of fetching
    that slot from a data source. Every model publishes its own stream, named
    after it, at its own cadence; one runner step covers one coupling window
    (by default the slowest model's step), so a faster model publishes several
    lead times per step.

    Unwired forcing slots and all initial conditions come from data sources,
    exactly as :class:`~earth2studio.run.PrognosticRunner` resolves them. A
    wired slot's initial window is taken from the producer's own initial
    condition when that covers it, otherwise from a source.

    Parameters
    ----------
    models : Mapping[str, PrognosticModel]
        Models by name. Each is single-input-slot and single-output-slot. The
        name is the model's stream name
    sources : Mapping[str, Source | tuple[Source | None, ...]], optional
        Per model, a source for the initial condition or a tuple of
        ``(initial, *forcing)`` sources with ``None`` for slots to leave to the
        model's recommended sources, by default the recommendations
    links : Sequence[Link], optional
        Explicit links, by default none; remaining forcing slots are wired
        automatically by variable name where exactly one model can supply them
    window : np.timedelta64, optional
        Coupling window, by default the largest model step. It must be a
        multiple of every model step
    seed : int, optional
        Base seed for per-member seeding of stochastic models, by default 0

    Raises
    ------
    ValueError
        If a model, link or source cannot be resolved, or cadences are not
        integer multiples of the window; nothing runs until this passes
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        models: Mapping[str, PrognosticModel],
        sources: Mapping[str, Source | tuple[Source | None, ...]] | None = None,
        links: Sequence[Link] = (),
        window: np.timedelta64 | None = None,
        seed: int = 0,
    ) -> None:
        if len(models) < 1:
            raise ValueError("Pass at least one model")
        self.seed = seed
        self.names = tuple(models)
        self.probes = {
            name: self._probe(name, model, (sources or {}).get(name))
            for name, model in models.items()
        }
        self.device = _module_device(next(iter(models.values())))
        steps = [p.step for p in self.probes.values()]
        self.window: np.timedelta64 = (
            max(steps) if window is None else np.timedelta64(window)
        )
        for name, probe in self.probes.items():
            if self.window % probe.step:
                raise ValueError(
                    f"Coupling window {self.window} is not a multiple of "
                    f"{name!r}'s step {probe.step}"
                )
        self.links = self._wire(links)
        self.order = self._schedule()
        logger.debug("Coupled schedule:\n{}", self.describe())

    # ------------------------------------------------------------------ probe

    @staticmethod
    def _probe(name: str, model: PrognosticModel, given: Any) -> _Probe:
        input_signature = model.input_coords()
        if isinstance(input_signature, tuple):
            raise ValueError(f"{name!r}: only single-input-slot models are supported")
        output_signature = model.output_coords(input_signature)
        if isinstance(output_signature, tuple):
            raise ValueError(f"{name!r}: only single-output-slot models are supported")
        step = (
            output_signature["lead_time"].values[-1]
            - input_signature["lead_time"].values[-1]
        )
        if step <= np.timedelta64(0, "h"):
            raise ValueError(f"{name!r}: model step must be positive, got {step}")
        forcing_signatures = _slots(model.forcing_coords())
        slots: list[Source | None] = [None] * (1 + len(forcing_signatures))
        if given is not None:
            given = given if isinstance(given, tuple) else (given,)
            if len(given) > len(slots):
                raise ValueError(f"{name!r}: more sources than input/forcing slots")
            slots[: len(given)] = given
        if any(s is None for s in slots):
            recommended = recommended_sources(model)
            if not isinstance(recommended, tuple):
                recommended = (recommended,)
            for i, source in enumerate(recommended[: len(slots)]):
                slots[i] = slots[i] if slots[i] is not None else source
        if slots[0] is None:
            raise ValueError(f"{name!r}: no source for the initial condition")
        return _Probe(
            model,
            input_signature,
            forcing_signatures,
            output_signature,
            step,
            tuple(slots),
        )

    # ------------------------------------------------------------------- wire

    def _wire(self, links: Sequence[Link]) -> dict[tuple[str, int], Link]:
        """Resolve every forcing slot to a producer link or a source."""
        wired: dict[tuple[str, int], Link] = {}
        for link in links:
            for end in (link.src, link.dst):
                if end not in self.probes:
                    raise ValueError(f"Link refers to unknown model {end!r}")
            if link.src == link.dst:
                raise ValueError(f"Link {link.src!r} -> itself; use a lagged model")
            slot = link.slot
            if slot is None:
                fits = self._fitting_slots(link.dst, link.src)
                if len(fits) != 1:
                    raise ValueError(
                        f"Cannot infer the slot for {link.src!r} -> {link.dst!r} "
                        f"(candidates {fits}); set Link.slot"
                    )
                slot = fits[0]
            if not 0 <= slot < len(self.probes[link.dst].forcing_signatures):
                raise ValueError(f"{link.dst!r} has no forcing slot {slot}")
            if (link.dst, slot) in wired:
                raise ValueError(f"{link.dst!r} slot {slot} has two producers")
            wired[(link.dst, slot)] = replace(link, slot=slot)

        for name, probe in self.probes.items():
            for slot, signature in enumerate(probe.forcing_signatures):
                if (name, slot) in wired or "lead_time" not in signature.dims:
                    continue  # Explicit already, or static: sources only.
                suppliers = [
                    other
                    for other in self.names
                    if other != name and slot in self._fitting_slots(name, other)
                ]
                if len(suppliers) > 1:
                    raise ValueError(
                        f"{name!r} slot {slot} can be fed by {suppliers}; "
                        "declare a Link to choose"
                    )
                if suppliers:
                    wired[(name, slot)] = Link(suppliers[0], name, slot)

        for (dst, slot), link in wired.items():
            self._check_link(link, self.probes[dst].forcing_signatures[slot])
        for name, probe in self.probes.items():
            for slot in range(len(probe.forcing_signatures)):
                if (name, slot) not in wired and probe.sources[1 + slot] is None:
                    raise ValueError(
                        f"{name!r} forcing slot {slot} has neither a producing "
                        "model nor a source"
                    )
        return wired

    def _fitting_slots(self, dst: str, src: str) -> list[int]:
        """Time-varying forcing slots of ``dst`` whose variables ``src`` produces."""
        produced = set(map(str, self.probes[src].output_signature["variable"].values))
        return [
            slot
            for slot, signature in enumerate(self.probes[dst].forcing_signatures)
            if "lead_time" in signature.dims
            and set(map(str, signature["variable"].values)) <= produced
        ]

    def _check_link(self, link: Link, signature: CoordinateSystem) -> None:
        slot = cast(int, link.slot)
        if "lead_time" not in signature.dims:
            raise ValueError(f"{link.dst!r} slot {slot} is static; use a source")
        src, dst = self.probes[link.src], self.probes[link.dst]
        produced = set(map(str, src.output_signature["variable"].values))
        missing = set(map(str, signature["variable"].values)) - produced
        if missing:
            raise ValueError(f"{link.src!r} does not produce {sorted(missing)}")
        if link.reduce == "mean" and (dst.step % src.step or src.step > dst.step):
            raise ValueError(
                f"mean link {link.src!r} -> {link.dst!r}: the producer step "
                f"{src.step} must divide the consumer step {dst.step}"
            )
        if link.transform is None:
            for dim in signature.dims:
                if dim in _NON_SPATIAL:
                    continue
                if dim not in src.output_signature.dims or not np.array_equal(
                    signature[dim].values, src.output_signature[dim].values
                ):
                    raise ValueError(
                        f"{link.src!r} -> {link.dst!r}: grids differ along {dim!r}; "
                        "set Link.transform to a regridder"
                    )

    def _schedule(self) -> tuple[str, ...]:
        """Producers before consumers along ``mean`` links; else declaration order."""
        needs = {name: set[str]() for name in self.names}
        for link in self.links.values():
            if link.reduce == "mean":
                needs[link.dst].add(link.src)
        order: list[str] = []
        while len(order) < len(self.names):
            free = [n for n in self.names if n not in order and needs[n] <= set(order)]
            if not free:
                raise ValueError(
                    "Cyclic mean links; break the cycle with a 'last' link"
                )
            order.append(free[0])
        return tuple(order)

    # --------------------------------------------------------------- describe

    def describe(self) -> str:
        """Return the resolved schedule and data flow as text, without running."""
        lines = [f"coupling window: {self.window}"]
        for name in self.order:
            probe = self.probes[name]
            lines.append(
                f"model {name}: step {probe.step}, "
                f"{int(self.window // probe.step)} per window"
            )
        for (dst, slot), link in self.links.items():
            variables = list(
                self.probes[dst].forcing_signatures[slot]["variable"].values
            )
            lines.append(
                f"link {link.src} -> {dst}[{slot}] {variables}: {link.reduce}"
                + (" +transform" if link.transform else "")
            )
        lines.extend(
            f"source -> {name}[{slot}]: {probe.sources[1 + slot]}"
            for name, probe in self.probes.items()
            for slot in range(len(probe.forcing_signatures))
            if (name, slot) not in self.links
        )
        return "\n".join(lines)

    # ----------------------------------------------------------------- Runner

    def to(self, device: torch.device) -> CoupledRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        for probe in self.probes.values():
            probe.model = probe.model.to(self.device)
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of coupling windows needed to reach ``final_lead_time``."""
        return _nsteps(final_lead_time, self.window)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return one stream per model, planned over the whole rollout."""
        self.nsteps(final_lead_time)
        return {
            name: _signature_for_stream(
                probe.output_signature,
                _lead_times(
                    probe.output_signature, _nsteps(final_lead_time, probe.step)
                ),
            )
            for name, probe in self.probes.items()
        }

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Fetches depend on the producers' leads; they are not declared up front."""
        return None

    # -------------------------------------------------------------- execution

    def _request(
        self, item: WorkItem, source: Source, signature: CoordinateSystem, leads: Any
    ) -> xr.DataArray:
        request = DataRequest(
            source,
            np.array([item.time]),
            signature["variable"].values,
            np.asarray(leads),
        )
        return _fetch(request, signature, self.device)

    def _deliver(
        self,
        link: Link,
        history: Mapping[str, list[tuple[np.timedelta64, xr.DataArray]]],
        signature: CoordinateSystem,
        at: np.ndarray,
        step: np.timedelta64,
        labels: np.ndarray,
    ) -> xr.DataArray:
        """Producer data for a step that starts at ``at[-1]``, labelled ``labels``."""
        frames = history[link.src]
        if link.reduce == "last":
            parts = []
            for lead in at:
                valid = [f for t, f in frames if t <= lead]
                if not valid:
                    raise RuntimeError(
                        f"{link.src!r} has no output at or before {lead}"
                    )
                parts.append(valid[-1])
            x = xr.concat(parts, dim="lead_time") if len(parts) > 1 else parts[0]
            x = x.assign_coords(lead_time=labels)
        else:
            window = [f for t, f in frames if at[-1] < t <= at[-1] + step]
            if not window:
                raise RuntimeError(
                    f"{link.src!r} has no outputs in ({at[-1]}, +{step}]"
                )
            mean = xr.concat(window, dim="lead_time").mean(
                "lead_time", skipna=False, keepdims=True
            )
            x = xr.concat(
                [mean.assign_coords(lead_time=[lab]) for lab in labels], dim="lead_time"
            )
        if link.transform is not None:
            x = link.transform(x)
        return _map_field(x, signature)

    def _initial_forcing(
        self,
        name: str,
        item: WorkItem,
        initial: Mapping[str, xr.DataArray],
        history: Mapping[str, list[tuple[np.timedelta64, xr.DataArray]]],
    ) -> tuple[xr.DataArray, ...]:
        """Full initialization windows for every forcing slot of ``name``."""
        probe = self.probes[name]
        forcing = []
        for slot, signature in enumerate(probe.forcing_signatures):
            static = "lead_time" not in signature.dims
            leads = np.array([np.timedelta64(0, "h")]) if static else None
            if leads is None:
                leads = signature["lead_time"].values
            source = probe.sources[1 + slot]
            link = self.links.get((name, slot))
            x: xr.DataArray | None = None
            if link is not None and link.reduce == "last":
                try:  # The producer's own initial window, when it covers this one.
                    x = initial[link.src].sel(lead_time=leads)
                    if link.transform is not None:
                        x = link.transform(x)
                    x = _map_field(x, signature)
                except (KeyError, ValueError):
                    if source is None:
                        raise ValueError(
                            f"{link.src!r}'s initial condition does not cover "
                            f"{name!r} slot {slot}; give that slot a source"
                        ) from None
                    x = None
            elif link is not None:  # mean: the first window itself.
                x = self._deliver(
                    link, history, signature, np.array([np.timedelta64(0, "h")]),
                    probe.step, leads,
                )  # fmt: skip
            if x is None:
                x = self._request(item, cast(Source, source), signature, leads)
            forcing.append(x.isel(lead_time=0, drop=True) if static else x)
        return tuple(forcing)

    def _step_forcing(
        self,
        name: str,
        item: WorkItem,
        history: Mapping[str, list[tuple[np.timedelta64, xr.DataArray]]],
        y_leads: np.ndarray,
    ) -> tuple[xr.DataArray, ...]:
        """Newest forcing frames for the step after an output at ``y_leads``."""
        probe = self.probes[name]
        forcing = []
        for slot, signature in enumerate(probe.forcing_signatures):
            if "lead_time" not in signature.dims:
                continue  # Static: initialization only.
            labels = signature["lead_time"].values[-1] + y_leads
            link = self.links.get((name, slot))
            if link is not None:
                forcing.append(
                    self._deliver(link, history, signature, y_leads, probe.step, labels)
                )
            else:
                source = probe.sources[1 + slot]
                forcing.append(
                    self._request(item, cast(Source, source), signature, labels)
                )
        return tuple(forcing)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish initial conditions, then one coupling window per step."""
        if len(item.member_ids) > 1:
            raise ValueError("CoupledRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        nwindows = self.nsteps(item.final_lead_time)
        producers = {link.src for link in self.links.values()}
        mean_sources = {
            name: [k.src for k in self.links.values() if k.dst == name and k.reduce == "mean"]
            for name in self.names
        }  # fmt: skip
        for probe in self.probes.values():
            _seed_model(self.seed, item.time, member_id, probe.model)
        initial = {
            name: self._request(
                item, cast(Source, p.sources[0]), p.input_signature,
                p.input_signature["lead_time"].values,
            )
            for name, p in self.probes.items()
        }  # fmt: skip
        rollouts = {name: _Rollout() for name in self.names}
        history: dict[str, list[tuple[np.timedelta64, xr.DataArray]]] = {}
        ic = {name: initial_condition(x) for name, x in initial.items()}
        for name in producers:
            history[name] = [(np.timedelta64(0, "h"), ic[name].isel(lead_time=[-1]))]
        yield dict(ic)
        try:
            for window in range(1, nwindows + 1):
                end = window * self.window
                for state in rollouts.values():
                    state.published = []
                while True:
                    pending = [n for n in self.order if rollouts[n].lead < end]
                    if not pending:
                        break
                    ready = [
                        n for n in pending
                        if all(
                            rollouts[p].lead >= rollouts[n].lead + self.probes[n].step
                            for p in mean_sources[n]
                        )
                    ]  # fmt: skip
                    if not ready:
                        raise RuntimeError("Coupled schedule deadlocked")
                    name = min(ready, key=lambda n: rollouts[n].lead)
                    y = self._advance(name, item, initial, history, rollouts[name])
                    state = rollouts[name]
                    leads = y.coords["lead_time"].values
                    state.y_leads, state.lead = leads, leads[-1]
                    cast(list[xr.DataArray], state.published).append(y)
                    if name in producers:
                        for i, lead in enumerate(leads):
                            history[name].append((lead, y.isel(lead_time=[i])))
                outputs = {}
                for name, state in rollouts.items():
                    parts = cast(list[xr.DataArray], state.published)
                    outputs[name] = (
                        xr.concat(parts, dim="lead_time")
                        if len(parts) > 1
                        else parts[0]
                    )
                yield outputs
                for name in history:  # Only the newest frame outlives a window.
                    history[name] = history[name][-1:]
        finally:
            for state in rollouts.values():
                if state.iterator is not None:
                    state.iterator.close()

    def _advance(
        self,
        name: str,
        item: WorkItem,
        initial: Mapping[str, xr.DataArray],
        history: Mapping[str, list[tuple[np.timedelta64, xr.DataArray]]],
        state: _Rollout,
    ) -> xr.DataArray:
        """Take one step of ``name``, creating its iterator on the first."""
        model = self.probes[name].model
        if state.iterator is None:
            forcing = self._initial_forcing(name, item, initial, history)
            state.iterator = model.create_iterator(initial[name], *forcing)
            return cast(xr.DataArray, next(state.iterator))
        forcing = self._step_forcing(
            name, item, history, cast(np.ndarray, state.y_leads)
        )
        return cast(xr.DataArray, state.iterator.send(forcing or None))
