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
from typing import Protocol, TypeAlias

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.base import recommended_sources
from earth2studio.models.px.utils import initial_condition
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


def _module_device(model: object) -> torch.device:
    if isinstance(model, torch.nn.Module):
        for tensor in chain(model.parameters(), model.buffers()):
            return tensor.device
    return torch.device("cpu")


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
        *,
        forcing: tuple[Source, ...] | None = None,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
    ) -> None:
        self.prognostic = prognostic
        self.input_signature = prognostic.input_coords()
        if isinstance(self.input_signature, tuple):
            raise ValueError("PrognosticRunner supports single-input-slot models")
        self.forcing_signatures = _slots(prognostic.forcing_coords())
        recommended = recommended_sources(prognostic)
        self.source = source if source is not None else recommended[0]
        self.forcing = forcing if forcing is not None else tuple(recommended[1:])
        if self.source is None or any(s is None for s in self.forcing):
            raise ValueError("Every input and forcing slot needs a source")
        if len(self.forcing) != len(self.forcing_signatures):
            raise ValueError("Pass one forcing source per forcing slot")
        self.diagnostics = dict(diagnostics or {})
        if "forecast" in self.diagnostics:
            raise ValueError("Stream name 'forecast' is reserved for the prognostic")
        self.device = _module_device(prognostic)
        oc = prognostic.output_coords(self.input_signature)
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
        if horizon < np.timedelta64(0, "s"):
            raise ValueError(f"Horizon {horizon} must be non-negative")
        if horizon % self.step:
            raise ValueError(f"Horizon {horizon} is not a multiple of {self.step}")
        return int(horizon // self.step)

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
            # Selects the diagnostic's variables and domain; never regrids.
            signature = diagnostic.output_coords(
                _map_field(oc, diagnostic.input_coords())
            )
            coords[name] = OrderedDict(
                [("lead_time", forecast["lead_time"])]
                + [
                    (dim, signature.coords[dim].values)
                    for dim in signature.dims
                    if dim not in ("time", "lead_time") and signature.sizes[dim]
                ]
            )
        return coords

    def _forcing_requests(
        self, item: WorkItem, output_leads: np.ndarray | None
    ) -> tuple[DataRequest, ...]:
        """Initial forcing windows, or the newest frames for ``output_leads``."""
        requests = []
        for source, signature in zip(self.forcing, self.forcing_signatures):
            leads = signature["lead_time"].values
            if output_leads is not None:
                leads = leads[-1] + output_leads
            requests.append(
                DataRequest(
                    source, np.array([item.time]), signature["variable"].values, leads
                )
            )
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
            *self._forcing_requests(item, None),
            *chain.from_iterable(
                self._forcing_requests(item, self.output_leads + k * self.step)
                for k in steps
            ),
        )

    def _fetch(self, request: DataRequest, signature: CoordinateSystem) -> xr.DataArray:
        interp_method = getattr(self.prognostic, "interp_method", None)
        x = fetch_data(
            source=request.source,
            time=request.time,
            variable=request.variable,
            lead_time=request.lead_time,
            device=self.device,
            target_grid=signature if interp_method else None,
            regridder=interp_method or "nearest",
        )
        return _map_field(x, signature)

    def _fetch_forcing(
        self, requests: tuple[DataRequest, ...]
    ) -> xr.DataArray | tuple[xr.DataArray, ...] | None:
        fields = tuple(
            self._fetch(r, s) for r, s in zip(requests, self.forcing_signatures)
        )
        if not fields:
            return None
        declared = self.prognostic.forcing_coords()
        return fields if isinstance(declared, tuple) else fields[0]

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
        x = self._fetch(self._initial_request(item), self.input_signature)
        yield self._publish(initial_condition(x))
        if nsteps == 0:
            return
        forcing = self._fetch_forcing(self._forcing_requests(item, None))
        iterator = self.prognostic.rollout_iterator(x, forcing)
        try:
            y = next(iterator)
            yield self._publish(y)
            for _ in range(nsteps - 1):
                leads = y.coords["lead_time"].values
                y = iterator.send(
                    self._fetch_forcing(self._forcing_requests(item, leads))
                )
                yield self._publish(y)
        finally:
            iterator.close()
