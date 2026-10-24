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

A prognostic model *is* the component: its ``input_coords``, ``output_coords`` and
``forcing_coords`` say what it produces and consumes, on which grid and at which
step, and its ``create_iterator`` accepts new forcing between steps.
:class:`CoupledRunner` probes those and drives the models forward on one shared
clock as a :class:`~earth2studio.run.Runner`, so it plugs into
:class:`~earth2studio.run.Pipeline` unchanged.

There is no graph or connector object. What feeds a forcing slot is an injected
function, a *provider*, ``provider(history, start, step) -> xr.DataArray``. Reduction,
regridding and unit conversion are ordinary functions composed around it.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources
from earth2studio.run._fields import _lead_times, _map_field
from earth2studio.run.runner import (
    DataRequest,
    Source,
    _diagnostic_coords,
    _fetch,
    _module_device,
    _nsteps,
    _seed_model,
    _signature_for_stream,
    _slots,
)
from earth2studio.run.work import WorkItem
from earth2studio.utils.type import CoordinateSystem

Frames = list[tuple[np.timedelta64, xr.DataArray]]  # (lead time, one-lead frame)
History = Mapping[str, Frames]  # Every model's outputs since the last window.
Provider = Callable[[History, np.timedelta64, np.timedelta64], xr.DataArray]
Slot = tuple[str, int]  # (consuming model, forcing slot index)


class NotReady(Exception):
    """Raised by a provider whose producer has not advanced far enough yet."""


def latest(src: str) -> Provider:
    """Provider holding ``src``'s newest output at or before the step's start.

    Parameters
    ----------
    src : str
        Name of the producing model

    Returns
    -------
    Provider
        ``provider(history, start, step)`` returning that one-lead output
    """

    def provide(history: History, start: np.timedelta64, step: np.timedelta64) -> Any:
        frames = [frame for lead, frame in history[src] if lead <= start]
        if not frames:
            raise LookupError(f"{src!r} has no output at or before {start}")
        return frames[-1]

    provide.__name__ = f"latest({src})"
    return provide


def reduced(src: str, fn: Callable[[xr.DataArray], xr.DataArray]) -> Provider:
    """Provider reducing ``src``'s outputs over the step ``(start, start + step]``.

    The consumer waits until ``src`` has covered the whole step, so a fast
    producer feeds a slow consumer, e.g. 6-hourly atmosphere to a 48-hourly ocean.

    Parameters
    ----------
    src : str
        Name of the producing model
    fn : Callable[[xr.DataArray], xr.DataArray]
        Reduces the outputs, stacked along ``lead_time``, to one, e.g.
        ``lambda x: x.mean("lead_time", skipna=False)``

    Returns
    -------
    Provider
        ``provider(history, start, step)`` returning the reduced output
    """

    def provide(history: History, start: np.timedelta64, step: np.timedelta64) -> Any:
        if history[src][-1][0] < start + step:
            raise NotReady(src)
        window = [f for lead, f in history[src] if start < lead <= start + step]
        if not window:
            raise ValueError(f"{src!r} has no output in ({start}, {start + step}]")
        return fn(xr.concat(window, dim="lead_time"))

    provide.__name__ = f"reduced({src}, {getattr(fn, '__name__', 'fn')})"
    return provide


def then(provider: Provider, *fns: Callable[[xr.DataArray], xr.DataArray]) -> Provider:
    """Provider applying ``fns`` in turn to ``provider``'s output.

    Use it for regridding (a bound :class:`~earth2studio.grids.Regridder`),
    unit conversion or masking, e.g. ``then(latest("ocean"), lambda x: x - 273.15)``.
    """

    def provide(history: History, start: np.timedelta64, step: np.timedelta64) -> Any:
        x = provider(history, start, step)
        for fn in fns:
            x = fn(x)
        return x

    provide.__name__ = ".".join(
        [getattr(provider, "__name__", "provider")]
        + [getattr(fn, "__name__", "fn") for fn in fns]
    )
    return provide


def match_by_variable(
    models: Mapping[str, PrognosticModel], skip: set[Slot] | None = None
) -> dict[Slot, Provider]:
    """Feed each time-varying forcing slot from the one model producing its variables.

    Parameters
    ----------
    models : Mapping[str, PrognosticModel]
        Models by name
    skip : set[tuple[str, int]], optional
        Slots to leave out, which may then be ambiguous, by default none

    Returns
    -------
    dict[tuple[str, int], Provider]
        :func:`latest` provider per matched slot; edit it freely before passing
        it as ``providers``

    Raises
    ------
    ValueError
        If more than one other model produces a slot's variables
    """
    produced = {
        name: set(
            map(
                str,
                cast(CoordinateSystem, model.output_coords(model.input_coords()))[
                    "variable"
                ].values,
            )
        )
        for name, model in models.items()
    }
    matched: dict[Slot, Provider] = {}
    for name, model in models.items():
        for slot, signature in enumerate(_slots(model.forcing_coords())):
            if "lead_time" not in signature.dims or (name, slot) in (skip or ()):
                continue
            wanted = set(map(str, signature["variable"].values))
            suppliers = [o for o in models if o != name and wanted <= produced[o]]
            if len(suppliers) > 1:
                raise ValueError(
                    f"{name!r} slot {slot} can be fed by {suppliers}; "
                    "pass a provider to choose"
                )
            if suppliers:
                matched[(name, slot)] = latest(suppliers[0])
    return matched


def _check_geometry(x: xr.DataArray, signature: CoordinateSystem, what: str) -> None:
    """Raise unless ``x`` has the spatial coordinates the ``signature`` declares."""
    skip = {"batch", "time", "ensemble", "lead_time", "variable"}
    for dim in signature.dims:
        if dim in skip:
            continue
        want = signature[dim].values
        if dim not in x.coords or x[dim].shape != want.shape:
            raise ValueError(
                f"{what}: delivered {dim!r} differs from the slot grid; "
                "regrid it with then(provider, regridder)"
            )
        have = x[dim].values
        same = (
            np.allclose(have, want, atol=1e-6)
            if np.issubdtype(want.dtype, np.number)
            else np.array_equal(have, want)
        )
        if not same:
            raise ValueError(
                f"{what}: delivered {dim!r} differs from the slot grid; "
                "regrid it with then(provider, regridder)"
            )


@dataclass
class _Probe:
    """What one model declares, read once at construction."""

    model: PrognosticModel
    input_signature: CoordinateSystem
    forcing_signatures: tuple[CoordinateSystem, ...]
    output_signature: CoordinateSystem
    step: np.timedelta64
    sources: tuple[Source | None, ...]  # Input slot, then forcing slots.
    given: tuple[bool, ...]  # Whether the caller (not a recommendation) set each.


@dataclass
class _Rollout:
    """Mutable per-item state of one model."""

    iterator: Any = None
    lead: np.timedelta64 = np.timedelta64(0, "h")


class CoupledRunner:
    """Run several prognostic models that exchange fields, as one runner.

    Each model is stepped at its own cadence on a shared clock and publishes its
    own stream, named after it. One runner step covers one coupling window (by
    default the slowest model's step), so a faster model publishes several lead
    times per step. The furthest-behind model steps next, and a model whose
    provider is :class:`NotReady` waits.

    A forcing slot is fed by, in order: its entry in ``providers``; its given
    source; the single model producing its variables (:func:`match_by_variable`);
    its recommended source. Initial conditions and unprovided slots come from
    sources exactly as :class:`~earth2studio.run.PrognosticRunner` resolves them.
    When a provider cannot cover a slot's initial window (``LookupError``) and the
    slot has a source, the source supplies that window.

    Parameters
    ----------
    models : Mapping[str, PrognosticModel]
        Models by name. Each has one input slot and one output slot with a
        single lead time. The name is the model's stream name
    sources : Mapping[str, Source | tuple[Source | None, ...]], optional
        Per model, a source for the initial condition or a tuple of
        ``(initial, *forcing)`` sources with ``None`` for slots to leave to the
        model's recommended sources, by default the recommendations
    providers : Mapping[tuple[str, int], Provider], optional
        What feeds forcing slot ``(model name, slot)``; see :func:`latest`,
        :func:`reduced` and :func:`then`, by default none
    window : np.timedelta64, optional
        Coupling window, by default the largest model step. It must be a
        multiple of every model step
    seed : int, optional
        Base seed for per-member seeding of stochastic models. Each model and
        diagnostic derives its own seed from it and its name, by default 0
    diagnostics : Mapping[str, tuple[str, DiagnosticModel]], optional
        Diagnostics applied to a model's output as it is published, as
        ``{stream: (model name, diagnostic)}``. The stream name must not be a
        model name, by default none
    check_finite : bool, optional
        Raise if data delivered between models holds NaN or infinity. This
        synchronizes the device on each delivery, by default False

    Raises
    ------
    ValueError
        If a model, provider or source cannot be resolved, or cadences are not
        integer multiples of the window; nothing runs until this passes. A
        delivered field on the wrong grid raises when it is first delivered
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        models: Mapping[str, PrognosticModel],
        sources: Mapping[str, Source | tuple[Source | None, ...]] | None = None,
        providers: Mapping[Slot, Provider] | None = None,
        window: np.timedelta64 | None = None,
        seed: int = 0,
        diagnostics: Mapping[str, tuple[str, DiagnosticModel]] | None = None,
        check_finite: bool = False,
    ) -> None:
        if len(models) < 1:
            raise ValueError("Pass at least one model")
        self.seed = seed
        self.check_finite = check_finite
        self.diagnostics = dict(diagnostics or {})
        for stream, (owner, _) in self.diagnostics.items():
            if stream in models:
                raise ValueError(f"Diagnostic stream {stream!r} clashes with a model")
            if owner not in models:
                raise ValueError(f"Diagnostic {stream!r}: unknown model {owner!r}")
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
        self.providers = self._resolve(models, dict(providers or {}))
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
        explicit = tuple(s is not None for s in slots)
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
            explicit,
        )

    def _resolve(
        self, models: Mapping[str, PrognosticModel], providers: dict[Slot, Provider]
    ) -> dict[Slot, Provider]:
        """Check the given providers and add matched ones; every slot needs a feed."""
        for name, slot in providers:
            if name not in self.probes:
                raise ValueError(f"Provider for unknown model {name!r}")
            signatures = self.probes[name].forcing_signatures
            if not 0 <= slot < len(signatures):
                raise ValueError(f"{name!r} has no forcing slot {slot}")
            if "lead_time" not in signatures[slot].dims:
                raise ValueError(f"{name!r} slot {slot} is static; use a source")
        taken = set(providers) | {
            (name, slot)
            for name, probe in self.probes.items()
            for slot in range(len(probe.forcing_signatures))
            if probe.given[1 + slot]
        }
        providers.update(match_by_variable(models, skip=taken))
        for name, probe in self.probes.items():
            for slot in range(len(probe.forcing_signatures)):
                if (name, slot) not in providers and probe.sources[1 + slot] is None:
                    raise ValueError(
                        f"{name!r} forcing slot {slot} has neither a provider "
                        "nor a source"
                    )
        return providers

    def describe(self) -> str:
        """Return the window, each model's cadence and what feeds every forcing slot."""
        lines = [f"coupling window: {self.window}"]
        for name, probe in self.probes.items():
            lines.append(
                f"model {name}: step {probe.step}, {self.window // probe.step} per window"
            )
        for (name, slot), probe in (
            ((n, s), p)
            for n, p in self.probes.items()
            for s in range(len(p.forcing_signatures))
        ):
            feed = self.providers.get((name, slot))
            feed_name = getattr(feed, "__name__", "provider") if feed else "source"
            lines.append(f"slot {name}[{slot}] <- {feed_name}")
        for stream, (owner, diagnostic) in self.diagnostics.items():
            lines.append(f"diagnostic {stream} = {type(diagnostic).__name__}({owner})")
        return "\n".join(lines)

    # ----------------------------------------------------------------- Runner

    def to(self, device: torch.device) -> CoupledRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        for probe in self.probes.values():
            probe.model = probe.model.to(self.device)
        self.diagnostics = {
            stream: (owner, diagnostic.to(self.device))
            for stream, (owner, diagnostic) in self.diagnostics.items()
        }
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of coupling windows needed to reach ``final_lead_time``."""
        return _nsteps(final_lead_time, self.window)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return one stream per model and diagnostic, planned over the rollout."""
        self.nsteps(final_lead_time)
        coords = {
            name: _signature_for_stream(
                probe.output_signature,
                _lead_times(
                    probe.output_signature, _nsteps(final_lead_time, probe.step)
                ),
            )
            for name, probe in self.probes.items()
        }
        for stream, (owner, diagnostic) in self.diagnostics.items():
            coords[stream] = _diagnostic_coords(
                diagnostic,
                self.probes[owner].output_signature,
                coords[owner]["lead_time"],
            )
        return coords

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe the fetches that are certain: initial conditions and unprovided slots.

        A provided slot with a fallback source is not listed, since whether the
        fallback is used is only known while running.
        """
        requests = [
            self._request(
                item,
                probe.sources[0],
                probe.input_signature,
                probe.input_signature["lead_time"].values,
            )
            for probe in self.probes.values()
        ]
        for name, probe in self.probes.items():
            for slot, signature in enumerate(probe.forcing_signatures):
                source = probe.sources[1 + slot]
                if (name, slot) in self.providers or source is None:
                    continue
                leads = self._first_leads(signature)
                requests.append(self._request(item, source, signature, leads))
                if "lead_time" in signature.dims:
                    requests.extend(
                        self._request(
                            item, source, signature, leads[-1:] + k * probe.step
                        )
                        for k in range(1, _nsteps(item.final_lead_time, probe.step))
                    )
        return tuple(requests)

    # -------------------------------------------------------------- execution

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

    def _provide(
        self,
        name: str,
        slot: int,
        history: History,
        labels: np.ndarray,
    ) -> xr.DataArray:
        """Provider output for steps starting at ``labels - slot's last lead``."""
        provider = self.providers[(name, slot)]
        signature = self.probes[name].forcing_signatures[slot]
        step = self.probes[name].step
        parts = []
        for start in labels - signature["lead_time"].values[-1]:
            x = provider(history, start, step)
            if "lead_time" not in x.dims:
                x = x.expand_dims("lead_time")
            if x.sizes["lead_time"] != 1:
                raise ValueError(f"{name!r} slot {slot}: a provider returns one lead")
            parts.append(x)
        x = xr.concat(parts, dim="lead_time").assign_coords(lead_time=labels)
        _check_geometry(x, signature, f"{name!r} slot {slot}")
        if self.check_finite:
            data = x.data.get() if hasattr(x.data, "get") else x.data
            if not np.isfinite(data).all():
                raise ValueError(f"{name!r} slot {slot}: delivered data is not finite")
        return _map_field(x, signature)

    def _forcing(
        self,
        name: str,
        slot: int,
        item: WorkItem,
        history: History,
        labels: np.ndarray,
        first: bool,
    ) -> xr.DataArray:
        """Forcing of one slot labelled ``labels``, from its provider or its source."""
        probe = self.probes[name]
        signature = probe.forcing_signatures[slot]
        source = probe.sources[1 + slot]
        if (name, slot) in self.providers:
            try:
                return self._provide(name, slot, history, labels)
            except LookupError:
                if not first or source is None:
                    raise
        request = self._request(item, source, signature, labels)
        return _fetch(request, signature, self.device)

    def _inputs(
        self, name: str, state: _Rollout, item: WorkItem, history: History
    ) -> tuple[xr.DataArray, ...]:
        """Forcing for ``name``'s next call: its whole window first, then newest frames."""
        forcing = []
        for slot, signature in enumerate(self.probes[name].forcing_signatures):
            static = "lead_time" not in signature.dims
            first = state.iterator is None
            if static and not first:
                continue  # Static slots: initialization only.
            labels = (
                self._first_leads(signature)
                if first
                else signature["lead_time"].values[-1:] + state.lead
            )
            x = self._forcing(name, slot, item, history, labels, first)
            forcing.append(x.isel(lead_time=0, drop=True) if static else x)
        return tuple(forcing)

    def _turn(
        self,
        rollouts: Mapping[str, _Rollout],
        end: np.timedelta64,
        item: WorkItem,
        history: History,
    ) -> tuple[str, tuple[xr.DataArray, ...]] | None:
        """The furthest-behind model that can step in the window ending at ``end``."""
        pending = sorted(
            (n for n in self.names if rollouts[n].lead < end),
            key=lambda n: rollouts[n].lead,  # Stable: ties keep model order.
        )
        for name in pending:
            try:
                return name, self._inputs(name, rollouts[name], item, history)
            except NotReady:
                continue
        if pending:
            raise RuntimeError(f"Coupled schedule deadlocked; waiting: {pending}")
        return None

    def _salt(self, name: str) -> int:
        """Base seed of ``name``, so stochastic models draw independent noise."""
        return self.seed + zlib.crc32(name.encode()) % 2**31

    def _publish(
        self, streams: Mapping[str, xr.DataArray]
    ) -> Mapping[str, xr.DataArray]:
        """Add the diagnostic streams computed from the models' outputs."""
        out = dict(streams)
        for stream, (owner, diagnostic) in self.diagnostics.items():
            out[stream] = diagnostic(
                _map_field(streams[owner], diagnostic.input_coords())
            )
        return out

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish initial conditions, then one coupling window per step."""
        if len(item.member_ids) > 1:
            raise ValueError("CoupledRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        for name, probe in self.probes.items():
            _seed_model(self._salt(name), item.time, member_id, probe.model)
        for stream, (_, diagnostic) in self.diagnostics.items():
            _seed_model(self._salt(stream), item.time, member_id, diagnostic)
        initial = {
            name: _fetch(
                self._request(
                    item,
                    p.sources[0],
                    p.input_signature,
                    p.input_signature["lead_time"].values,
                ),
                p.input_signature,
                self.device,
            )
            for name, p in self.probes.items()
        }
        history: dict[str, Frames] = {
            name: [
                (lead, x.isel(lead_time=[i]))
                for i, lead in enumerate(x["lead_time"].values)
            ]
            for name, x in initial.items()
        }
        yield self._publish({name: initial_condition(x) for name, x in initial.items()})
        rollouts = {name: _Rollout() for name in self.names}
        try:
            for window in range(1, self.nsteps(item.final_lead_time) + 1):
                published: dict[str, list[xr.DataArray]] = {n: [] for n in self.names}
                end = window * self.window
                while (turn := self._turn(rollouts, end, item, history)) is not None:
                    name, forcing = turn
                    state = rollouts[name]
                    if state.iterator is None:
                        model = self.probes[name].model
                        state.iterator = model.create_iterator(initial[name], *forcing)
                        y = next(state.iterator)
                    else:
                        y = state.iterator.send(forcing or None)
                    state.lead = y["lead_time"].values[-1]
                    published[name].append(y)
                    history[name].append((state.lead, y))
                yield self._publish(
                    {
                        name: (
                            xr.concat(frames, dim="lead_time")
                            if len(frames) > 1
                            else frames[0]
                        )
                        for name, frames in published.items()
                    }
                )
                for frames in history.values():  # Only the newest frame outlives it.
                    del frames[:-1]
        finally:
            for state in rollouts.values():
                if state.iterator is not None:
                    state.iterator.close()
