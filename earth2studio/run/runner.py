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

"""Runners: what one work item does.

A :class:`Runner` holds models and bound data sources, and executes one
:class:`WorkItem` at a time. Supervision -- work distribution, member grouping,
resume, output filtering, and IO -- sits above it and never sees models,
components, or coupling. See ``dev/spec/EXECUTION_CONTRACT_SPEC.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.data.base import DataFrameSource, ForecastFrameSource
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources, step_interval
from earth2studio.run._fields import (
    _lead_times,
    _map_field,
    _publishes_initial_condition,
)
from earth2studio.run.work import WorkItem, member_seed
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, coord_array
from earth2studio.utils.type import CoordinateSystem

Source: TypeAlias = DataSource | ForecastSource
# A source for any model slot: gridded fields, or DataFrames for observation slots.
SlotSource: TypeAlias = Source | DataFrameSource | ForecastFrameSource


@dataclass(frozen=True)
class DataRequest:
    """Arguments for one :func:`~earth2studio.data.fetch_data` call; never fetches."""

    source: Source
    time: np.ndarray
    variable: np.ndarray
    lead_time: np.ndarray = field(
        default_factory=lambda: np.array([np.timedelta64(0, "h")])
    )


class Runner(Protocol):
    """Execute one work item at a time, yielding outputs keyed by stream.

    A runner is built once and reused across items. It binds sources, never
    data: field values, including initial conditions and forcing, are fetched
    inside :meth:`run_item`.

    A stream is a named sequence of outputs sharing one structure: the same
    variables, grid, and lead-time pattern. Supervision writes each stream to its
    own store, keyed by its name. When implementing a runner:

    - Give outputs with different grids or cadences separate streams, as with
      DLESyM's atmosphere and ocean.
    - Name streams after the model or component producing them, and keep names
      stable across items and runs; they key output stores and resume.
    - Declare every stream in :meth:`output_coords`. Each output must match its
      stream's declaration.
    - A stream may skip steps, as a slower component does; a step then omits it.
    """

    supports_member_batching: bool
    """Whether one item may carry several ``member_ids``."""

    def to(self, device: torch.device) -> Runner:
        """Move models to ``device`` and fetch onto it."""
        ...

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each stream's planning signature for one item.

        Each signature is an allocation-free DataArray whose fixed dimensions are
        ``lead_time`` followed by the field dimensions. ``time`` and ``ensemble``
        appear only in the dynamic prefix; supervision replaces them with the
        concrete values from the items.
        """
        ...

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Describe every fetch ``item`` will make, or ``None`` if not knowable."""
        ...

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield, per step, one output for each stream that published at it."""
        ...


def _slots(signature: CoordinateSystem | tuple | None) -> tuple:
    if signature is None:
        return ()
    return signature if isinstance(signature, tuple) else (signature,)


def _module_device(model: object) -> torch.device:
    if isinstance(model, torch.nn.Module):
        for tensor in chain(model.parameters(), model.buffers()):
            return tensor.device
    return torch.device("cpu")


def _nsteps(final_lead_time: np.timedelta64, step: np.timedelta64) -> int:
    if final_lead_time < np.timedelta64(0, "s"):
        raise ValueError(f"Final lead time {final_lead_time} must be non-negative")
    if final_lead_time % step:
        raise ValueError(
            f"Final lead time {final_lead_time} is not a multiple of {step}"
        )
    return int(final_lead_time // step)


def _fetch(
    request: DataRequest, signature: CoordinateSystem, device: torch.device
) -> xr.DataArray:
    # Off-grid sources arrive as RegriddedSource, which fetch_data applies.
    x = fetch_data(
        source=request.source,
        time=request.time,
        variable=request.variable,
        lead_time=request.lead_time,
        device=device,
    )
    return _map_field(x, signature)


def _signature_for_stream(
    signature: CoordinateSystem, leads: np.ndarray
) -> CoordinateSystem:
    """Plan a stream's schema over ``leads`` from one step's model signature.

    ``signature`` keeps whatever dynamic prefix the model declared, such as
    ``(batch, time)``; the stream schema keeps that prefix and fixes
    ``lead_time`` -- the rollout's whole extent -- ahead of the field
    dimensions. A model that drops ``lead_time``, as some diagnostics do, still
    gets the axis, so supervision can label each write with it. Grid metadata,
    auxiliary coordinates, dtype and attributes carry over, and nothing is
    allocated.
    """
    # A diagnostic may declare lead_time dynamic; the stream fixes it instead.
    dynamic = tuple(
        dim for dim in signature.attrs.get(E2S_DYNAMIC_DIMS, ()) if dim != "lead_time"
    )
    for reserved in ("time", "ensemble"):
        if reserved in signature.dims and reserved not in dynamic:
            raise ValueError(
                f"Runner output_coords: {reserved!r} must be a dynamic leading "
                "dimension, not a fixed dimension"
            )
    field_dims = tuple(
        dim for dim in signature.dims if dim not in dynamic and dim != "lead_time"
    )

    coordinates: dict[Hashable, Any] = {
        name: coord.variable
        for name, coord in signature.coords.items()
        if set(coord.dims).issubset(set(dynamic).union(field_dims))
    }
    coordinates["lead_time"] = leads
    sizes: dict[Hashable, int] = {
        dim: signature.sizes[dim] for dim in (*dynamic, *field_dims)
    }

    return coord_array(
        (*dynamic, "lead_time", *field_dims),
        coordinates,
        sizes=sizes,
        dynamic=dynamic,
        dtype=signature.dtype,
        name=signature.name,
        attrs=signature.attrs,
    )


def _diagnostic_coords(
    diagnostic: DiagnosticModel, signature: CoordinateSystem, leads: np.ndarray
) -> CoordinateSystem:
    """Stream signature for ``diagnostic`` applied to fields like ``signature``."""
    # Selects the diagnostic's variables and domain; never regrids.
    output = diagnostic.output_coords(_map_field(signature, diagnostic.input_coords()))
    return _signature_for_stream(output, leads)


def _seed_model(
    base: int, time: np.datetime64, member_id: int, model: object
) -> int | None:
    """Seed one model's own random stream for a member, never global RNG (``X5``).

    Returns the seed applied, or ``None`` when ``model`` is not stochastic.
    """
    set_rng = getattr(model, "set_rng", None)
    if not getattr(model, "stochastic", False) or not callable(set_rng):
        return None
    seed = member_seed(base, time, member_id)
    set_rng(seed, reset=True)
    return seed


class PrognosticRunner:
    """Roll one prognostic model forward, optionally applying diagnostics.

    Each step yields the prognostic output as stream ``"forecast"`` and each
    diagnostic's output under its own stream name. Steps run from the initial
    condition through the item's final lead time -- the steps ``run.deterministic``
    writes. Each step advances by the output's last lead time less the later of
    the input's last lead time and zero, so a model initialized from a background
    before the analysis time steps at its forecast interval. The initial condition
    is published only when the input ends at zero and the output declares every
    input variable on the input's grid; otherwise the streams start at the first
    forecast. Forcing declared by the model is fetched and sent at every step.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model to roll out. Its single input slot is supported.
    source : DataSource | ForecastSource, optional
        Source of initial conditions, by default the model's recommendation
    forcing : tuple[DataSource | ForecastSource, ...], optional
        One source per forcing slot, by default the model's recommendations
    diagnostics : Mapping[str, DiagnosticModel], optional
        Diagnostics applied to every step, by default none. Each key names the
        stream that diagnostic's outputs are yielded under; supervision writes
        each stream to its own store.
    seed : int, optional
        Base seed for per-member seeding of stochastic models, by default 0
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        prognostic: PrognosticModel,
        source: Source | None = None,
        forcing: tuple[Source, ...] | None = None,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
        seed: int = 0,
    ) -> None:
        self.prognostic = prognostic
        input_signature = prognostic.input_coords()
        if isinstance(input_signature, tuple):
            raise ValueError("PrognosticRunner supports single-input-slot models")
        self.input_signature: CoordinateSystem = input_signature
        self.forcing_signatures = _slots(prognostic.forcing_coords())
        recommended = recommended_sources(prognostic)
        nslots = 1 + len(self.forcing_signatures)
        if not isinstance(recommended, tuple):
            recommended = (recommended,) + (None,) * (nslots - 1)
        initial_source = source if source is not None else recommended[0]
        forcing_sources = forcing if forcing is not None else tuple(recommended[1:])
        if initial_source is None or any(s is None for s in forcing_sources):
            raise ValueError("Every input and forcing slot needs a source")
        self.source: Source = initial_source
        self.forcing: tuple[Source, ...] = tuple(
            s for s in forcing_sources if s is not None
        )
        if len(self.forcing) != len(self.forcing_signatures):
            raise ValueError("Pass one forcing source per forcing slot")
        self.diagnostics = dict(diagnostics or {})
        if "forecast" in self.diagnostics:
            raise ValueError("Stream name 'forecast' is reserved for the prognostic")
        self.seed = seed
        self.device = _module_device(prognostic)
        oc = cast(CoordinateSystem, prognostic.output_coords(self.input_signature))
        self.output_leads = oc["lead_time"].values
        self.step = step_interval(
            self.input_signature["lead_time"].values, self.output_leads
        )
        self.publishes_initial_condition = _publishes_initial_condition(
            self.input_signature, oc
        )

    def to(self, device: torch.device) -> PrognosticRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.prognostic = self.prognostic.to(self.device)
        self.diagnostics = {k: v.to(self.device) for k, v in self.diagnostics.items()}
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of model steps needed to reach ``final_lead_time``."""
        return _nsteps(final_lead_time, self.step)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each stream's planning signature for one item.

        Fixed dimensions are ``lead_time`` followed by the field dimensions;
        ``time`` and ``ensemble`` are part of the dynamic prefix, which
        supervision concretizes.
        """
        oc = self.prognostic.output_coords(self.input_signature)
        leads = _lead_times(oc, self.nsteps(final_lead_time), self.step)
        if not self.publishes_initial_condition:
            leads = leads[1:]
        forecast = _signature_for_stream(oc, leads)
        coords: dict[str, CoordinateSystem] = {"forecast": forecast}
        for name, diagnostic in self.diagnostics.items():
            coords[name] = _diagnostic_coords(diagnostic, oc, forecast["lead_time"])
        return coords

    def _forcing_requests(
        self, item: WorkItem, output_leads: np.ndarray | None = None
    ) -> tuple[tuple[DataRequest, CoordinateSystem], ...]:
        """Initial forcing windows, or each time-varying slot's newest frames.

        Static slots, those without ``lead_time``, are fetched only initially.
        """
        requests = []
        for source, signature in zip(self.forcing, self.forcing_signatures):
            static = "lead_time" not in signature.dims
            if static and output_leads is not None:
                continue
            leads = np.array([np.timedelta64(0, "h")])
            if not static:
                leads = signature["lead_time"].values
                if output_leads is not None:
                    leads = leads[-1] + output_leads
            request = DataRequest(
                source, np.array([item.time]), signature["variable"].values, leads
            )
            requests.append((request, signature))
        return tuple(requests)

    def _initial_request(self, item: WorkItem) -> DataRequest:
        return DataRequest(
            self.source,
            np.array([item.time]),
            self.input_signature["variable"].values,
            self.input_signature["lead_time"].values,
        )

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe the initial-condition and forcing fetches for ``item``."""
        steps = range(max(self.nsteps(item.final_lead_time) - 1, 0))
        return (
            self._initial_request(item),
            *(request for request, _ in self._forcing_requests(item)),
            *(
                request
                for k in steps
                for request, _ in self._forcing_requests(
                    item, self.output_leads + k * self.step
                )
            ),
        )

    def _fetch_forcing(
        self, requests: tuple[tuple[DataRequest, CoordinateSystem], ...]
    ) -> tuple[xr.DataArray, ...]:
        fields = []
        for request, signature in requests:
            x = _fetch(request, signature, self.device)
            if "lead_time" not in signature.dims:  # Static slot.
                x = x.isel(lead_time=0, drop=True)
            fields.append(x)
        return tuple(fields)

    def _publish(self, x: xr.DataArray) -> Mapping[str, xr.DataArray]:
        outputs = {"forecast": x}
        for name, diagnostic in self.diagnostics.items():
            outputs[name] = diagnostic(_map_field(x, diagnostic.input_coords()))
        return outputs

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish the initial condition, if any, then every forecast through the final lead time."""
        if len(item.member_ids) > 1:
            raise ValueError("PrognosticRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        _seed_model(self.seed, item.time, member_id, self.prognostic)
        for diagnostic in self.diagnostics.values():
            _seed_model(self.seed, item.time, member_id, diagnostic)
        nsteps = self.nsteps(item.final_lead_time)
        x = _fetch(self._initial_request(item), self.input_signature, self.device)
        if self.publishes_initial_condition:
            yield self._publish(initial_condition(x))
        if nsteps == 0:
            return
        forcing = self._fetch_forcing(self._forcing_requests(item))
        iterator = self.prognostic.create_iterator(x, *forcing)
        try:
            y = cast(xr.DataArray, next(iterator))
            yield self._publish(y)
            for _ in range(nsteps - 1):
                leads = y.coords["lead_time"].values
                step_forcing = self._fetch_forcing(self._forcing_requests(item, leads))
                y = cast(xr.DataArray, iterator.send(step_forcing or None))
                yield self._publish(y)
        finally:
            iterator.close()


class DiagnosticRunner:
    """Apply diagnostics directly to source data, without a prognostic model.

    Step ``k`` fetches each diagnostic's inputs at the item's time plus
    ``k * step`` and yields its output under the diagnostic's stream name, from
    lead time zero through the item's final lead time. A
    :class:`~earth2studio.data.ForecastSource` supplies forecast lead times; a
    :class:`~earth2studio.data.DataSource` supplies later valid times.

    Parameters
    ----------
    diagnostics : Mapping[str, DiagnosticModel]
        Diagnostics keyed by stream name. Each single-input-slot diagnostic
        fetches its own inputs.
    source : DataSource | ForecastSource, optional
        Source for every diagnostic, by default each diagnostic's recommendation
    step : np.timedelta64, optional
        Spacing between steps, by default none: each item is one step and its
        final lead time must be zero
    seed : int, optional
        Base seed for per-member seeding of stochastic diagnostics, by default 0
    """

    supports_member_batching = False  # Member batching is not implemented.

    def __init__(
        self,
        diagnostics: Mapping[str, DiagnosticModel],
        source: Source | None = None,
        step: np.timedelta64 | None = None,
        seed: int = 0,
    ) -> None:
        if not diagnostics:
            raise ValueError("DiagnosticRunner needs at least one diagnostic")
        self.diagnostics = dict(diagnostics)
        self.signatures: dict[str, CoordinateSystem] = {}
        self.sources: dict[str, Source] = {}
        for name, diagnostic in self.diagnostics.items():
            signature = diagnostic.input_coords()
            if isinstance(signature, tuple):
                raise ValueError("DiagnosticRunner supports single-input-slot models")
            if "variable" not in signature.dims or not signature.sizes["variable"]:
                raise ValueError(f"Diagnostic {name!r} declares no input variables")
            recommended = recommended_sources(diagnostic)
            if isinstance(recommended, tuple):
                recommended = recommended[0]
            chosen = source if source is not None else recommended
            if chosen is None:
                raise ValueError(f"Diagnostic {name!r} needs a source")
            self.signatures[name] = signature
            self.sources[name] = chosen
        self.seed = seed
        self.step = step
        self.device = _module_device(next(iter(self.diagnostics.values())))

    def to(self, device: torch.device) -> DiagnosticRunner:
        """Move every diagnostic to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.diagnostics = {k: v.to(self.device) for k, v in self.diagnostics.items()}
        return self

    def lead_times(self, final_lead_time: np.timedelta64) -> np.ndarray:
        """Return the lead time of every step through ``final_lead_time``."""
        if self.step is None:
            if final_lead_time != np.timedelta64(0, "s"):
                raise ValueError("A nonzero final lead time needs a step")
            return np.array([np.timedelta64(0, "h")])
        return np.arange(_nsteps(final_lead_time, self.step) + 1) * self.step

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return each stream's planning signature for one item.

        Fixed dimensions are ``lead_time`` followed by the field dimensions;
        ``time`` and ``ensemble`` are part of the dynamic prefix.
        """
        leads = self.lead_times(final_lead_time)
        return {
            name: _diagnostic_coords(diagnostic, self.signatures[name], leads)
            for name, diagnostic in self.diagnostics.items()
        }

    def _requests(self, item: WorkItem, lead: np.timedelta64) -> dict[str, DataRequest]:
        return {
            name: DataRequest(
                self.sources[name],
                np.array([item.time]),
                signature["variable"].values,
                np.array([lead]),
            )
            for name, signature in self.signatures.items()
        }

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe every diagnostic's fetch at every step of ``item``."""
        return tuple(
            request
            for lead in self.lead_times(item.final_lead_time)
            for request in self._requests(item, lead).values()
        )

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Fetch inputs and apply every diagnostic at each step."""
        if len(item.member_ids) > 1:
            raise ValueError("DiagnosticRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        for diagnostic in self.diagnostics.values():
            _seed_model(self.seed, item.time, member_id, diagnostic)
        for lead in self.lead_times(item.final_lead_time):
            outputs = {}
            for name, request in self._requests(item, lead).items():
                diagnostic = self.diagnostics[name]
                x = _fetch(request, self.signatures[name], self.device)
                outputs[name] = diagnostic(x)
            yield outputs
