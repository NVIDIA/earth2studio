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
produces and consumes, on which grid, and how large its step is, and its
``create_iterator`` accepts new forcing between steps. :class:`CoupledRunner`
probes those, wires producers to forcing slots, and drives the models forward on
one shared clock as a :class:`~earth2studio.run.Runner`, so it plugs into
:class:`~earth2studio.run.Pipeline` unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.grids import Regridder, infer_grid, same_geometry
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

Reduction = Callable[[xr.DataArray], xr.DataArray]
Frames = list[tuple[np.timedelta64, xr.DataArray]]  # (lead time, one-lead frame)


def window_mean(x: xr.DataArray) -> xr.DataArray:
    """Reduce a window of producer outputs to its mean, e.g. 6-hourly to 48-hourly.

    Parameters
    ----------
    x : xr.DataArray
        Producer outputs in the window, stacked along ``lead_time``

    Returns
    -------
    xr.DataArray
        Mean over ``lead_time``, which is kept with length one
    """
    return x.mean("lead_time", skipna=False, keepdims=True)


@dataclass(frozen=True)
class Link:
    """One producer feeding one forcing slot of a consumer.

    Links are optional: :class:`CoupledRunner` wires a forcing slot to the only
    other model that produces all of its variables, holding the producer's
    latest output. Declare a link to override that, to reduce a window, or to
    bridge differing grids.

    Attributes
    ----------
    src : str
        Name of the producing model
    dst : str
        Name of the consuming model
    slot : int, optional
        Index of ``dst``'s forcing slot, by default inferred as the only slot
        whose variables ``src`` produces
    reduce : Callable[[xr.DataArray], xr.DataArray], optional
        For the step ``dst`` is about to take, from ``t`` to ``t + step``.
        ``None``: ``dst`` receives ``src``'s latest output valid at or before
        ``t``, held when ``src`` is slower. A callable, such as
        :func:`window_mean`: ``src`` is advanced first and the callable reduces
        its outputs in ``(t, t + step]`` stacked along ``lead_time``; ``src``
        must then be at least as fast as ``dst``. By default ``None``
    regridder : Regridder, optional
        Bound to ``src``'s output grid and ``dst``'s slot grid. Required when
        the two differ, by default none
    """

    src: str
    dst: str
    slot: int | None = None
    reduce: Reduction | None = None
    regridder: Regridder | None = None


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
        Models by name. Each has one input slot and one output slot with a
        single lead time. The name is the model's stream name
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
        If a model, link, grid or source cannot be resolved, or cadences are not
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
        self.producers = {link.src for link in self.links.values()}
        self.lookahead = {
            name: tuple(
                k for k in self.links.values() if k.dst == name and k.reduce is not None
            )
            for name in self.names
        }
        self._check_acyclic()
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
        if output_signature["lead_time"].size != 1:
            raise ValueError(
                f"{name!r}: only models with one output lead time per step are "
                "supported"
            )
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

        for (dst, slot), wired_link in wired.items():
            self._check_link(wired_link, self.probes[dst].forcing_signatures[slot])
        for name, probe in self.probes.items():
            for slot, signature in enumerate(probe.forcing_signatures):
                feed = wired.get((name, slot))
                if probe.sources[1 + slot] is not None:
                    continue
                if feed is None:
                    raise ValueError(
                        f"{name!r} forcing slot {slot} has neither a producing "
                        "model nor a source"
                    )
                if not self._covers(feed, signature):
                    raise ValueError(
                        f"{feed.src!r}'s initial condition does not cover "
                        f"{name!r} forcing slot {slot}; give that slot a source"
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
        if link.reduce is not None and (dst.step % src.step or src.step > dst.step):
            raise ValueError(
                f"reducing link {link.src!r} -> {link.dst!r}: the producer step "
                f"{src.step} must divide the consumer step {dst.step}"
            )
        produced_grid, wanted_grid = infer_grid(src.output_signature), infer_grid(
            signature
        )
        if link.regridder is None:
            if not same_geometry(produced_grid, wanted_grid):
                raise ValueError(
                    f"{link.src!r} -> {link.dst!r}: the grids differ; "
                    "set Link.regridder"
                )
        elif not (
            same_geometry(link.regridder.source_grid, produced_grid)
            and same_geometry(link.regridder.target_grid, wanted_grid)
        ):
            raise ValueError(
                f"{link.src!r} -> {link.dst!r}: Link.regridder is bound to other grids"
            )

    def _covers(self, link: Link, signature: CoordinateSystem) -> bool:
        """Whether the producer's initial condition can supply a slot's first window."""
        leads = signature["lead_time"].values
        if link.reduce is not None:  # Only the step right after the start exists.
            return leads.size == 1
        first = self.probes[link.src].input_signature["lead_time"].values[0]
        return bool(leads[0] - leads[-1] >= first)

    def _check_acyclic(self) -> None:
        """Reducing links need their producer first, so they must not loop."""
        done: set[str] = set()
        while len(done) < len(self.names):
            free = [
                n
                for n in self.names
                if n not in done and all(k.src in done for k in self.lookahead[n])
            ]
            if not free:
                raise ValueError(
                    "Cyclic reducing links; break the cycle with a held link"
                )
            done.add(free[0])

    # --------------------------------------------------------------- describe

    def describe(self) -> str:
        """Return the resolved schedule and data flow as text, without running."""
        lines = [f"coupling window: {self.window}"]
        for name in self.names:
            probe = self.probes[name]
            lines.append(
                f"model {name}: step {probe.step}, "
                f"{int(self.window // probe.step)} per window"
            )
        for (dst, slot), link in self.links.items():
            variables = list(
                self.probes[dst].forcing_signatures[slot]["variable"].values
            )
            mode = (
                "hold"
                if link.reduce is None
                else getattr(link.reduce, "__name__", repr(link.reduce))
            )
            lines.append(
                f"link {link.src} -> {dst}[{slot}] {variables}: {mode}"
                + (" +regridder" if link.regridder else "")
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

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe every source fetch of ``item``; exchanged fields are not fetched."""
        requests = [
            self._request(
                item,
                probe.sources[0],
                probe.input_signature,
                probe.input_signature["lead_time"].values,
            )
            for probe in self.probes.values()
        ]
        for (name, slot), signature in self._slots():
            probe, link = self.probes[name], self.links.get((name, slot))
            source = probe.sources[1 + slot]
            if source is None:
                continue
            leads = self._first_leads(signature)
            if link is None or not self._covers(link, signature):
                requests.append(self._request(item, source, signature, leads))
            if link is None and "lead_time" in signature.dims:
                requests.extend(
                    self._request(item, source, signature, leads[-1:] + k * probe.step)
                    for k in range(1, _nsteps(item.final_lead_time, probe.step))
                )
        return tuple(requests)

    # -------------------------------------------------------------- execution

    def _slots(self) -> Iterator[tuple[tuple[str, int], CoordinateSystem]]:
        for name, probe in self.probes.items():
            for slot, signature in enumerate(probe.forcing_signatures):
                yield (name, slot), signature

    @staticmethod
    def _first_leads(signature: CoordinateSystem) -> np.ndarray:
        if "lead_time" not in signature.dims:  # Static slots are read once.
            return np.array([np.timedelta64(0, "h")])
        return cast(np.ndarray, signature["lead_time"].values)

    @staticmethod
    def _request(
        item: WorkItem,
        source: Source | None,
        signature: CoordinateSystem,
        leads: np.ndarray,
    ) -> DataRequest:
        return DataRequest(
            cast(Source, source),
            np.array([item.time]),
            signature["variable"].values,
            np.asarray(leads),
        )

    def _deliver(
        self,
        link: Link,
        frames: Frames,
        signature: CoordinateSystem,
        at: np.ndarray,
        step: np.timedelta64,
        labels: np.ndarray,
    ) -> xr.DataArray:
        """Producer data for steps of ``dst`` that start at ``at``, labelled ``labels``."""
        parts = []
        for start in at:
            if link.reduce is None:
                window = [f for lead, f in frames if lead <= start][-1:]
            else:
                window = [f for lead, f in frames if start < lead <= start + step]
            if not window:
                raise LookupError(f"{link.src!r} has no output for the step at {start}")
            x = (
                window[0]
                if link.reduce is None
                else link.reduce(xr.concat(window, dim="lead_time"))
            )
            if "lead_time" not in x.dims:
                x = x.expand_dims("lead_time")
            if x.sizes["lead_time"] != 1:
                raise ValueError("A Link.reduce must return a single lead time")
            parts.append(x)
        x = xr.concat(parts, dim="lead_time").assign_coords(lead_time=labels)
        if link.regridder is not None:
            x = link.regridder(x)
        return _map_field(x, signature)

    def _forcing(
        self,
        name: str,
        slot: int,
        item: WorkItem,
        history: Mapping[str, Frames],
        labels: np.ndarray,
        use_link: bool = True,
    ) -> xr.DataArray:
        """Forcing of one slot labelled ``labels``, from its producer or its source."""
        probe = self.probes[name]
        signature = probe.forcing_signatures[slot]
        link = self.links.get((name, slot))
        if link is not None and use_link:
            at = labels - signature["lead_time"].values[-1]
            return self._deliver(
                link, history[link.src], signature, at, probe.step, labels
            )
        request = self._request(item, probe.sources[1 + slot], signature, labels)
        return _fetch(request, signature, self.device)

    def _initial_forcing(
        self, name: str, item: WorkItem, history: Mapping[str, Frames]
    ) -> tuple[xr.DataArray, ...]:
        """Full initialization windows for every forcing slot of ``name``."""
        forcing = []
        for slot, signature in enumerate(self.probes[name].forcing_signatures):
            link = self.links.get((name, slot))
            use_link = link is not None and self._covers(link, signature)
            x = self._forcing(
                name, slot, item, history, self._first_leads(signature), use_link
            )
            static = "lead_time" not in signature.dims
            forcing.append(x.isel(lead_time=0, drop=True) if static else x)
        return tuple(forcing)

    def _step_forcing(
        self,
        name: str,
        item: WorkItem,
        history: Mapping[str, Frames],
        lead: np.timedelta64,
    ) -> tuple[xr.DataArray, ...]:
        """Newest forcing frames for the step after an output at ``lead``."""
        return tuple(
            self._forcing(
                name, slot, item, history, signature["lead_time"].values[-1:] + lead
            )
            for slot, signature in enumerate(self.probes[name].forcing_signatures)
            if "lead_time" in signature.dims  # Static slots: initialization only.
        )

    def _next(
        self, rollouts: Mapping[str, _Rollout], end: np.timedelta64
    ) -> str | None:
        """The model to step next in the window ending at ``end``, if any is left.

        The furthest-behind model goes first, so held links see the newest output
        valid at its lead; a reducing link waits until its producer has covered
        the consumer's whole next step.
        """
        pending = [n for n in self.names if rollouts[n].lead < end]
        if not pending:
            return None
        ready = [
            n
            for n in pending
            if all(
                rollouts[k.src].lead >= rollouts[n].lead + self.probes[n].step
                for k in self.lookahead[n]
            )
        ]
        if not ready:
            raise RuntimeError("Coupled schedule deadlocked")
        return min(ready, key=lambda n: rollouts[n].lead)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish initial conditions, then one coupling window per step."""
        if len(item.member_ids) > 1:
            raise ValueError("CoupledRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        for probe in self.probes.values():
            _seed_model(self.seed, item.time, member_id, probe.model)
        initial = {
            name: _fetch(
                self._request(
                    item, p.sources[0], p.input_signature,
                    p.input_signature["lead_time"].values,
                ),
                p.input_signature,
                self.device,
            )
            for name, p in self.probes.items()
        }  # fmt: skip
        history: dict[str, Frames] = {
            name: [
                (lead, initial[name].isel(lead_time=[i]))
                for i, lead in enumerate(initial[name]["lead_time"].values)
            ]
            for name in self.producers
        }
        yield {name: initial_condition(x) for name, x in initial.items()}
        rollouts = {name: _Rollout() for name in self.names}
        try:
            for window in range(1, self.nsteps(item.final_lead_time) + 1):
                published: dict[str, list[xr.DataArray]] = {n: [] for n in self.names}
                while (name := self._next(rollouts, window * self.window)) is not None:
                    state = rollouts[name]
                    if state.iterator is None:
                        forcing = self._initial_forcing(name, item, history)
                        model = self.probes[name].model
                        state.iterator = model.create_iterator(initial[name], *forcing)
                        y = next(state.iterator)
                    else:
                        forcing = self._step_forcing(name, item, history, state.lead)
                        y = state.iterator.send(forcing or None)
                    state.lead = y["lead_time"].values[-1]
                    published[name].append(y)
                    if name in self.producers:
                        history[name].append((state.lead, y))
                yield {
                    name: (
                        xr.concat(frames, dim="lead_time")
                        if len(frames) > 1
                        else frames[0]
                    )
                    for name, frames in published.items()
                }
                for frames in history.values():  # Only the newest frame outlives it.
                    del frames[:-1]
        finally:
            for state in rollouts.values():
                if state.iterator is not None:
                    state.iterator.close()
