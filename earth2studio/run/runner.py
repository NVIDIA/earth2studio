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

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Protocol, TypeAlias, cast

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources
from earth2studio.run._fields import _map_field, _output_dimensions
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.type import CoordinateSystem

Source: TypeAlias = DataSource | ForecastSource


@dataclass(frozen=True)
class WorkItem:
    """One forecast reference time, horizon, and optional ensemble member group."""

    time: np.datetime64
    horizon: np.timedelta64
    member_ids: tuple[int, ...] = ()


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

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return each stream's coordinates for one item.

        Includes ``lead_time``; excludes ``time`` and ``ensemble``, which
        supervision adds.
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


def _yields_initial_condition(model: object) -> bool:
    """Whether an unmigrated wrapper's iterator still yields the initial condition."""
    return not callable(getattr(type(model), "initialize", None))


def _module_device(model: object) -> torch.device:
    if isinstance(model, torch.nn.Module):
        for tensor in chain(model.parameters(), model.buffers()):
            return tensor.device
    return torch.device("cpu")


def _nsteps(horizon: np.timedelta64, step: np.timedelta64) -> int:
    if horizon < np.timedelta64(0, "s"):
        raise ValueError(f"Horizon {horizon} must be non-negative")
    if horizon % step:
        raise ValueError(f"Horizon {horizon} is not a multiple of {step}")
    return int(horizon // step)


def _fetch(
    request: DataRequest,
    signature: CoordinateSystem,
    device: torch.device,
    interp_method: str | None,
) -> xr.DataArray:
    x = fetch_data(
        source=request.source,
        time=request.time,
        variable=request.variable,
        lead_time=request.lead_time,
        device=device,
        target_grid=signature if interp_method else None,
        regridder=interp_method or "nearest",
    )
    return _map_field(x, signature)


def _diagnostic_coords(
    diagnostic: DiagnosticModel, signature: CoordinateSystem, leads: np.ndarray
) -> CoordSystem:
    """Stream coordinates for ``diagnostic`` applied to fields like ``signature``."""
    # Selects the diagnostic's variables and domain; never regrids.
    output = cast(
        CoordinateSystem,
        diagnostic.output_coords(_map_field(signature, diagnostic.input_coords())),
    )
    return OrderedDict(
        [("lead_time", leads)]
        + [
            (dim, output.coords[dim].values)
            for dim in output.dims
            if dim not in ("time", "lead_time") and output.sizes[dim]
        ]
    )


class PrognosticRunner:
    """Roll one prognostic model forward, optionally applying diagnostics.

    Each step yields the prognostic output as stream ``"forecast"`` and each
    diagnostic's output under its own stream name. Steps run from the initial
    condition through the item's horizon -- the steps ``run.deterministic``
    writes. Forcing declared by the model is fetched and sent at every step.

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
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        prognostic: PrognosticModel,
        source: Source | None = None,
        forcing: tuple[Source, ...] | None = None,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
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
        self.device = _module_device(prognostic)
        oc = cast(CoordinateSystem, prognostic.output_coords(self.input_signature))
        self.output_leads = oc["lead_time"].values
        self.step = self.output_leads[-1] - self.input_signature["lead_time"].values[-1]

    def to(self, device: torch.device) -> PrognosticRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.prognostic = self.prognostic.to(self.device)
        self.diagnostics = {k: v.to(self.device) for k, v in self.diagnostics.items()}
        return self

    def nsteps(self, horizon: np.timedelta64) -> int:
        """Return the number of model steps needed to reach ``horizon``."""
        return _nsteps(horizon, self.step)

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return each stream's coordinates for one item.

        Includes ``lead_time``; excludes ``time`` and ``ensemble``, which
        supervision adds.
        """
        forecast = _output_dimensions(
            self.prognostic, np.empty(0, "datetime64[ns]"), self.nsteps(horizon)
        )
        del forecast["time"]
        coords: dict[str, CoordSystem] = {"forecast": forecast}
        oc = self.prognostic.output_coords(self.input_signature)
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
        steps = range(max(self.nsteps(item.horizon) - 1, 0))
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
        interp_method = getattr(self.prognostic, "interp_method", None)
        fields = []
        for request, signature in requests:
            x = _fetch(request, signature, self.device, interp_method)
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
        """Publish the initial condition, then every forecast through the horizon."""
        if len(item.member_ids) > 1:
            raise ValueError("PrognosticRunner does not batch ensemble members")
        nsteps = self.nsteps(item.horizon)
        x = _fetch(
            self._initial_request(item),
            self.input_signature,
            self.device,
            getattr(self.prognostic, "interp_method", None),
        )
        yield self._publish(initial_condition(x))
        if nsteps == 0:
            return
        forcing = self._fetch_forcing(self._forcing_requests(item))
        iterator = self.prognostic.create_iterator(x, *forcing)
        try:
            if _yields_initial_condition(self.prognostic):
                next(iterator)  # Already published above.
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
    lead time zero through the item's horizon. A
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
        horizon must be zero
    """

    supports_member_batching = False  # Member seeding is not implemented.

    def __init__(
        self,
        diagnostics: Mapping[str, DiagnosticModel],
        source: Source | None = None,
        step: np.timedelta64 | None = None,
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
        self.step = step
        self.device = _module_device(next(iter(self.diagnostics.values())))

    def to(self, device: torch.device) -> DiagnosticRunner:
        """Move every diagnostic to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.diagnostics = {k: v.to(self.device) for k, v in self.diagnostics.items()}
        return self

    def lead_times(self, horizon: np.timedelta64) -> np.ndarray:
        """Return the lead time of every step through ``horizon``."""
        if self.step is None:
            if horizon != np.timedelta64(0, "s"):
                raise ValueError("A nonzero horizon needs a step")
            return np.array([np.timedelta64(0, "h")])
        return np.arange(_nsteps(horizon, self.step) + 1) * self.step

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return each stream's coordinates for one item.

        Includes ``lead_time``; excludes ``time`` and ``ensemble``, which
        supervision adds.
        """
        leads = self.lead_times(horizon)
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
            for lead in self.lead_times(item.horizon)
            for request in self._requests(item, lead).values()
        )

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Fetch inputs and apply every diagnostic at each step."""
        if len(item.member_ids) > 1:
            raise ValueError("DiagnosticRunner does not batch ensemble members")
        for lead in self.lead_times(item.horizon):
            outputs = {}
            for name, request in self._requests(item, lead).items():
                diagnostic = self.diagnostics[name]
                x = _fetch(
                    request,
                    self.signatures[name],
                    self.device,
                    getattr(diagnostic, "interp_method", None),
                )
                outputs[name] = diagnostic(x)
            yield outputs
