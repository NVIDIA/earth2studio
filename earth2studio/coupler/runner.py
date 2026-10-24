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

"""Coupled runner: several prognostic models exchanging forcing, as one ``Runner``.

Each model keeps its own cadence and publishes its own output stream. Within one
coupling window the models run in the order given, so a later model sees the new
outputs of the earlier ones and an earlier model sees the later ones as they were
at the end of the previous window. What a model receives as forcing is a plain
function of the outputs so far, supplied by the caller.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.grids import (
    Regridder,
    infer_grid,
    resolve_regridder,
    same_geometry,
    select_subset,
)
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

Outputs = Mapping[str, xr.DataArray]  # Each model's outputs, along ``lead_time``.
Coupling = Callable[[Outputs, np.ndarray], xr.DataArray]  # (outputs, leads) -> forcing


_NON_SPATIAL = {"batch", "time", "ensemble", "lead_time", "variable"}


class CoupledRunner:
    """Run several prognostic models that feed each other, as one runner.

    Models step at their own cadence and publish one stream each, named after the
    model. A runner step is one coupling window (by default the slowest model's
    step), so a faster model publishes several lead times per step. Within a
    window the models run in the given order, and a coupling can read whatever
    has been produced by then.

    A ``couplings`` entry is a function ``coupling(outputs, leads) -> DataArray``
    that supplies the model's first forcing slot from the other models' outputs.
    ``outputs[name]`` holds model ``name``'s outputs from the start of the
    previous window to now, along ``lead_time``, and ``leads`` are the lead times
    the forcing must be valid at. Return one entry per lead; they are relabeled
    to ``leads``. Plain xarray covers the usual cases, for example::

        couplings = {
            # hold the newest GOES frame
            "mrms": lambda out, leads: out["goes"].sel(lead_time=leads, method="pad"),
            # mean of the atmosphere over the ocean's whole step (48 h, 6 h steps)
            "ocean": lambda out, leads: out["atmos"]
            .sel(lead_time=slice(leads[0] + H6, leads[0] + H48))
            .mean("lead_time", keepdims=True),
        }

    Regridding, unit conversion and masking are applied inside the function. Other
    forcing slots and the initial conditions come from sources exactly as in
    :class:`~earth2studio.run.PrognosticRunner`.

    Ensembles: this runner takes one member per work item (``member_batch=1``)
    and seeds every stochastic model from ``(seed, model name, time, member)``.
    Perturbed initial conditions come from
    :class:`~earth2studio.coupler.EnsembleCoupledRunner`.

    Parameters
    ----------
    models : Mapping[str, PrognosticModel]
        Models by name, in the order they run within a window. Each has one input
        slot and one output slot with a single lead time
    sources : Mapping[str, Source | tuple[Source | None, ...]], optional
        Per model, a source for the initial condition or a tuple of
        ``(initial, *forcing)`` sources with ``None`` for slots left to the model's
        recommended sources, by default the recommendations
    couplings : Mapping[str, Coupling], optional
        Model name to the function supplying its first forcing slot, which must
        vary in time, by default none
    window : np.timedelta64, optional
        Coupling window, a multiple of every model step, by default the largest
        model step
    seed : int, optional
        Base seed for stochastic models; each model derives its own from it and
        its name, by default 0

    Raises
    ------
    ValueError
        If a model, coupling or source cannot be resolved, or the window is not
        a multiple of every model step. A coupling returning the wrong number of
        leads or the wrong grid raises when it is called
    """

    supports_member_batching = False  # Member perturbation is not implemented.

    def __init__(
        self,
        models: Mapping[str, PrognosticModel],
        sources: Mapping[str, Source | tuple[Source | None, ...]] | None = None,
        couplings: Mapping[str, Coupling] | None = None,
        window: np.timedelta64 | None = None,
        seed: int = 0,
    ) -> None:
        if len(models) < 1:
            raise ValueError("Pass at least one model")
        self.models = dict(models)
        self.couplings = dict(couplings or {})
        self.seed = seed
        self._regridders: dict[str, Regridder] = {}
        self.device = _module_device(next(iter(self.models.values())))
        self.inputs: dict[str, CoordinateSystem] = {}
        self.outputs: dict[str, CoordinateSystem] = {}
        self.forcings: dict[str, tuple[CoordinateSystem, ...]] = {}
        self.steps: dict[str, np.timedelta64] = {}
        self.sources: dict[str, tuple[Source | None, ...]] = {}
        for name, model in self.models.items():
            self._probe(name, model, (sources or {}).get(name))
        self.window: np.timedelta64 = (
            max(self.steps.values()) if window is None else np.timedelta64(window)
        )
        for name, step in self.steps.items():
            if self.window % step:
                raise ValueError(
                    f"Coupling window {self.window} is not a multiple of "
                    f"{name!r}'s step {step}"
                )
        for name in self.couplings:
            if name not in self.models:
                raise ValueError(f"Coupling for unknown model {name!r}")
            slots = self.forcings[name]
            if not slots or "lead_time" not in slots[0].dims:
                raise ValueError(f"{name!r}: first forcing slot is absent or static")

    def _probe(self, name: str, model: PrognosticModel, given: Any) -> None:
        """Read what ``model`` declares and settle its sources."""
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
        forcing = _slots(model.forcing_coords())
        sources: list[Source | None] = [None] * (1 + len(forcing))
        if given is not None:
            given = given if isinstance(given, tuple) else (given,)
            if len(given) > len(sources):
                raise ValueError(f"{name!r}: more sources than input/forcing slots")
            sources[: len(given)] = given
        if any(s is None for s in sources):
            recommended = recommended_sources(model)
            if not isinstance(recommended, tuple):
                recommended = (recommended,)
            for i, source in enumerate(recommended[: len(sources)]):
                sources[i] = sources[i] if sources[i] is not None else source
        if sources[0] is None:
            raise ValueError(f"{name!r}: no source for the initial condition")
        coupled = 1 if name in self.couplings else 0  # Slot 0 needs no source.
        if any(s is None for s in sources[1 + coupled :]):
            raise ValueError(f"{name!r}: a forcing slot has no coupling or source")
        self.inputs[name] = input_signature
        self.outputs[name] = output_signature
        self.forcings[name] = forcing
        self.steps[name] = step
        self.sources[name] = tuple(sources)

    # ----------------------------------------------------------------- Runner

    def to(self, device: torch.device) -> CoupledRunner:
        """Move every model to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.models = {
            name: model.to(self.device) for name, model in self.models.items()
        }
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of coupling windows needed to reach ``final_lead_time``."""
        return _nsteps(final_lead_time, self.window)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return one stream per model, planned over the rollout."""
        self.nsteps(final_lead_time)
        return {
            name: _signature_for_stream(
                self.outputs[name],
                _lead_times(
                    self.outputs[name], _nsteps(final_lead_time, self.steps[name])
                ),
            )
            for name in self.models
        }

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe the source fetches: initial conditions and uncoupled forcing."""
        requests = [
            self._request(
                item,
                self.sources[name][0],
                self.inputs[name],
                self.inputs[name]["lead_time"].values,
            )
            for name in self.models
        ]
        for name in self.models:
            for slot, signature in enumerate(self.forcings[name]):
                if slot == 0 and name in self.couplings:
                    continue
                source = self.sources[name][1 + slot]
                leads = self._first_leads(signature)
                requests.append(self._request(item, source, signature, leads))
                if "lead_time" in signature.dims:
                    requests.extend(
                        self._request(
                            item, source, signature, leads[-1:] + k * self.steps[name]
                        )
                        for k in range(
                            1, _nsteps(item.final_lead_time, self.steps[name])
                        )
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

    def _couple(self, name: str, outputs: Outputs, leads: np.ndarray) -> xr.DataArray:
        """Call ``name``'s coupling and check what it returns against the slot."""
        signature = self.forcings[name][0]
        x = self.couplings[name](outputs, leads)
        if "lead_time" not in x.dims:
            x = x.expand_dims("lead_time")
        if x.sizes["lead_time"] != len(leads):
            raise ValueError(
                f"{name!r}: coupling returned {x.sizes['lead_time']} lead times, "
                f"expected {len(leads)}"
            )
        x = x.assign_coords(lead_time=leads)
        x = self._on_slot_grid(name, x, signature)
        return _map_field(x, signature)

    def _on_slot_grid(
        self, name: str, x: xr.DataArray, signature: CoordinateSystem
    ) -> xr.DataArray:
        """Bring a coupled field onto the forcing slot's grid.

        Uses the same grid utilities as :func:`~earth2studio.data.fetch_data`: a
        field already on the grid passes, an exact crop is selected, and anything
        else is regridded by nearest neighbor within the source resolution (never
        extrapolating). The regridder is built once per model and reused. To
        choose another method or a fill, regrid inside the coupling instead.
        """
        spatial = [d for d in signature.dims if d not in _NON_SPATIAL]
        try:
            target = infer_grid(signature)
            source = infer_grid(x)
        except ValueError as error:
            if self._matches(x, signature, spatial):
                return x
            raise ValueError(
                f"{name!r}: cannot determine the grid of the coupling output or "
                f"the forcing slot ({error}); regrid inside the coupling"
            ) from error
        if same_geometry(source, target):
            return x
        subset = select_subset(x, source, target)
        if subset is not None:
            return subset
        if name not in self._regridders:
            try:
                self._regridders[name] = resolve_regridder(
                    source, target, "nearest", max_dist_km="auto", fill_value=None
                )
            except ValueError as error:
                raise ValueError(
                    f"{name!r}: cannot regrid the coupling output automatically: "
                    f"{error}; regrid inside the coupling to choose a method"
                ) from error
            logger.info(
                "Regridding {!r} coupling output from a {} grid to a {} grid by "
                "nearest neighbor",
                name,
                source.topology,
                target.topology,
            )
        return self._regridders[name](x)

    @staticmethod
    def _matches(
        x: xr.DataArray, signature: CoordinateSystem, dims: list[Hashable]
    ) -> bool:
        """Whether ``x`` carries the signature's coordinate values on ``dims``."""
        for dim in dims:
            want = signature[dim].values
            if dim not in x.coords or x[dim].shape != want.shape:
                return False
            have = x[dim].values
            if not (
                np.allclose(have, want, atol=1e-6)
                if np.issubdtype(want.dtype, np.number)
                else np.array_equal(have, want)
            ):
                return False
        return True

    def _forcing(
        self,
        name: str,
        item: WorkItem,
        outputs: Outputs,
        lead: np.timedelta64,
        first: bool,
    ) -> tuple[xr.DataArray, ...]:
        """Forcing for ``name``'s next call: the whole window first, then the newest."""
        forcing = []
        for slot, signature in enumerate(self.forcings[name]):
            static = "lead_time" not in signature.dims
            if static and not first:
                continue  # Static slots: initialization only.
            leads = (
                self._first_leads(signature)
                if first
                else signature["lead_time"].values[-1:] + lead
            )
            if slot == 0 and name in self.couplings:
                x = self._couple(name, outputs, leads)
            else:
                request = self._request(
                    item, self.sources[name][1 + slot], signature, leads
                )
                x = _fetch(request, signature, self.device)
            forcing.append(x.isel(lead_time=0, drop=True) if static else x)
        return tuple(forcing)

    def _start(
        self, item: WorkItem, member_id: int, initial: dict[str, xr.DataArray]
    ) -> dict[str, xr.DataArray]:
        """Initial conditions each model starts from; the hook for subclasses."""
        return initial

    def _salt(self, name: str) -> int:
        """Base seed of ``name``, so each model draws independent noise."""
        return self.seed + zlib.crc32(name.encode()) % 2**31

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish initial conditions, then one coupling window per step."""
        if len(item.member_ids) > 1:
            raise ValueError(
                "CoupledRunner runs one ensemble member per item; "
                "build items with member_batch=1"
            )
        member_id = item.member_ids[0] if item.member_ids else 0
        for name, model in self.models.items():
            _seed_model(self._salt(name), item.time, member_id, model)
        initial = {
            name: _fetch(
                self._request(
                    item,
                    self.sources[name][0],
                    self.inputs[name],
                    self.inputs[name]["lead_time"].values,
                ),
                self.inputs[name],
                self.device,
            )
            for name in self.models
        }
        initial = self._start(item, member_id, initial)
        outputs = dict(
            initial
        )  # Coupling input; trimmed to the newest frame per window.
        yield {name: initial_condition(x) for name, x in initial.items()}
        iterators: dict[str, Any] = {}
        leads = {name: np.timedelta64(0, "h") for name in self.models}
        try:
            for window in range(1, self.nsteps(item.final_lead_time) + 1):
                published: dict[str, list[xr.DataArray]] = {n: [] for n in self.models}
                for name, model in self.models.items():
                    while leads[name] < window * self.window:
                        first = name not in iterators
                        forcing = self._forcing(name, item, outputs, leads[name], first)
                        if first:
                            iterators[name] = model.create_iterator(
                                initial[name], *forcing
                            )
                            y = next(iterators[name])
                        else:
                            y = iterators[name].send(forcing or None)
                        leads[name] = y["lead_time"].values[-1]
                        published[name].append(y)
                        outputs[name] = xr.concat([outputs[name], y], dim="lead_time")
                yield {
                    name: xr.concat(frames, dim="lead_time")
                    for name, frames in published.items()
                }
                outputs = {  # Only the newest frame outlives the window.
                    name: x.isel(lead_time=[-1]) for name, x in outputs.items()
                }
        finally:
            for iterator in iterators.values():
                iterator.close()
