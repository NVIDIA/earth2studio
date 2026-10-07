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
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.run._fields import _map_field, _output_dimensions
from earth2studio.utils.coords import CoordSystem


@dataclass(frozen=True)
class WorkItem:
    """One forecast reference time, horizon, and optional ensemble member group."""

    time: np.datetime64
    horizon: np.timedelta64
    member_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DataRequest:
    """One :func:`~earth2studio.data.fetch_data` call a work item will make."""

    source: DataSource | ForecastSource
    time: np.ndarray
    variable: np.ndarray
    lead_time: np.ndarray


class Runner(Protocol):
    """Execute one work item at a time, yielding outputs keyed by stream.

    A runner is built once and reused across items. It binds sources, never
    data: field values, including initial conditions, are fetched inside
    :meth:`run_item`.
    """

    supports_member_batching: bool
    """Whether one item may carry several ``member_ids``."""

    def to(self, device: torch.device) -> Runner:
        """Move models to ``device`` and fetch onto it."""
        ...

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return each stream's coordinates for one item, excluding ``time``."""
        ...

    def requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Return every fetch ``item`` makes, or ``None`` if not knowable upfront."""
        ...

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield, per step, one output for each stream that published at it."""
        ...


class ModelRunner:
    """Roll one prognostic model forward, optionally applying diagnostics.

    Each step yields the prognostic output as stream ``"forecast"`` and each
    diagnostic's output under its own name, from the initial condition through
    the item's horizon -- the steps ``run.deterministic`` writes.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model to roll out.
    source : DataSource | ForecastSource
        Source of initial conditions: live, cached, or predownloaded.
    diagnostics : Mapping[str, DiagnosticModel], optional
        Diagnostics applied to every step, keyed by stream name, by default none
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        prognostic: PrognosticModel,
        source: DataSource | ForecastSource,
        *,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
    ) -> None:
        self.prognostic = prognostic
        self.source = source
        self.diagnostics = dict(diagnostics or {})
        if "forecast" in self.diagnostics:
            raise ValueError("Stream name 'forecast' is reserved for the prognostic")
        self.device = torch.device("cpu")
        ic = prognostic.input_coords()
        oc = prognostic.output_coords(ic)
        self.step = oc["lead_time"].values[-1] - ic["lead_time"].values[-1]

    def to(self, device: torch.device) -> ModelRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.prognostic = self.prognostic.to(self.device)
        self.diagnostics = {k: v.to(self.device) for k, v in self.diagnostics.items()}
        return self

    def nsteps(self, horizon: np.timedelta64) -> int:
        """Return the number of model steps needed to reach ``horizon``."""
        if horizon % self.step:
            raise ValueError(f"Horizon {horizon} is not a multiple of {self.step}")
        return int(horizon // self.step)

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return each stream's coordinates for one item, excluding ``time``."""
        forecast = _output_dimensions(
            self.prognostic, np.empty(0, "datetime64[ns]"), self.nsteps(horizon)
        )
        del forecast["time"]
        coords: dict[str, CoordSystem] = {"forecast": forecast}
        oc = self.prognostic.output_coords(self.prognostic.input_coords())
        for name, diagnostic in self.diagnostics.items():
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

    def requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Return the initial-condition fetch for ``item``."""
        ic = self.prognostic.input_coords()
        return (
            DataRequest(
                self.source,
                np.array([item.time]),
                ic.coords["variable"].values,
                ic.coords["lead_time"].values,
            ),
        )

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Fetch the initial condition and yield every step through the horizon."""
        if len(item.member_ids) > 1:
            raise ValueError("ModelRunner does not batch ensemble members")
        nsteps = self.nsteps(item.horizon)
        ic = self.prognostic.input_coords()
        interp_method = getattr(self.prognostic, "interp_method", None)
        (request,) = self.requests(item)
        x = fetch_data(
            source=request.source,
            time=request.time,
            variable=request.variable,
            lead_time=request.lead_time,
            device=self.device,
            target_grid=ic if interp_method else None,
            regridder=interp_method or "nearest",
        )
        iterator = self.prognostic.create_iterator(_map_field(x, ic))
        for step, x in enumerate(iterator):
            outputs = {"forecast": x}
            for name, diagnostic in self.diagnostics.items():
                outputs[name] = diagnostic(_map_field(x, diagnostic.input_coords()))
            yield outputs
            if step == nsteps:
                break
