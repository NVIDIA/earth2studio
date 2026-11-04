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

"""Coupled GOES to MRMS nowcasting on the StormScope wrappers.

:class:`StormScopeRunner` couples a :class:`~earth2studio.models.px.StormScopeGOES`
model with a :class:`~earth2studio.models.px.StormScopeMRMS` model the way the
MRMS wrapper documents: both start from the same GOES history, and at every
step the radar model advances first, conditioned on the GOES forecast frames
at the lead times of its previous forecast, before the GOES model advances.
The runner yields the two forecasts as the streams ``goes`` and ``mrms``,
both on the shared HRRR projected grid.

Regridding onto the model grid, the radar and lightning composition and the
GOES conditioning source are the models' own ``default_sources()``; the
runner only binds sources and drives one work item.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.grids import Region
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import recommended_sources
from earth2studio.run import WorkItem
from earth2studio.run._fields import _lead_times, _map_field
from earth2studio.run.runner import (
    DataRequest,
    Source,
    _fetch,
    _module_device,
    _nsteps,
    _seed_model,
    _signature_for_stream,
)
from earth2studio.utils.type import CoordinateSystem

from ..data import resolve_source
from ..models import load_from_node


def _default_source(model: Any, slot: int, label: str) -> Source:
    """Return the model's recommended source for input slot ``slot``."""
    recommended = recommended_sources(model)
    if isinstance(recommended, tuple):
        recommended = recommended[slot] if slot < len(recommended) else None
    elif slot:
        recommended = None
    if recommended is None:
        raise ValueError(
            f"StormScope {label} needs a source; the model recommends none"
        )
    return cast(Source, recommended)


class StormScopeRunner:
    """Roll a GOES model and an MRMS model forward together.

    Each step yields the GOES forecast as stream ``"goes"`` and the radar
    forecast as stream ``"mrms"``, after the initial conditions at lead time
    zero.  One item carries one ensemble member; the runner seeds each
    stochastic model per member before the rollout.

    Parameters
    ----------
    goes : StormScopeGOES
        GOES model; an unforced variant, or a conditioned variant that reads
        its forcing from ``goes_forcing_source``.
    mrms : StormScopeMRMS
        Radar model conditioned on GOES; its grid must equal the GOES grid.
    goes_source : DataSource, optional
        GOES observation history source on the model grid, by default the
        GOES model's recommendation
    mrms_source : DataSource, optional
        Radar (and lightning) history source on the model grid, by default the
        MRMS model's recommendation
    conditioning_source : DataSource, optional
        Source of the GOES history that initializes the radar model, by
        default the GOES observations fetched for the GOES model when they
        cover the conditioning variables and lead times, otherwise the MRMS
        model's recommendation
    goes_forcing_source : DataSource | ForecastSource, optional
        Forcing for a conditioned GOES variant, by default its recommendation
    seed : int, optional
        Base seed for per-member seeding, by default 0
    """

    supports_member_batching = False  # One member per item.

    def __init__(
        self,
        goes: Any,
        mrms: Any,
        goes_source: Source | None = None,
        mrms_source: Source | None = None,
        conditioning_source: Source | None = None,
        goes_forcing_source: Source | None = None,
        seed: int = 0,
    ) -> None:
        self.goes = goes
        self.mrms = mrms
        self.seed = seed
        self.device = _module_device(goes)

        self.goes_signature = cast(CoordinateSystem, goes.input_coords())
        self.mrms_signature = cast(CoordinateSystem, mrms.input_coords())
        conditioning = mrms.forcing_coords()
        if conditioning is None or isinstance(conditioning, tuple):
            raise ValueError("The MRMS model must declare one GOES conditioning slot")
        self.conditioning_signature: CoordinateSystem = conditioning
        forcing = goes.forcing_coords()
        if isinstance(forcing, tuple):
            raise ValueError("StormScopeRunner supports one GOES forcing slot at most")
        self.goes_forcing_signature: CoordinateSystem | None = forcing

        goes_oc = cast(CoordinateSystem, goes.output_coords(self.goes_signature))
        mrms_oc = cast(CoordinateSystem, mrms.output_coords(self.mrms_signature))
        for dim in ("y", "x"):
            if not np.array_equal(goes_oc[dim].values, mrms_oc[dim].values):
                raise ValueError(
                    "StormScope GOES and MRMS models must share one grid; their "
                    f"{dim!r} coordinates differ (apply the same region to both)"
                )
        self.output_leads = goes_oc["lead_time"].values
        self.step = self.output_leads[-1] - self.goes_signature["lead_time"].values[-1]
        mrms_step = (
            mrms_oc["lead_time"].values[-1]
            - self.mrms_signature["lead_time"].values[-1]
        )
        if self.step != mrms_step:
            raise ValueError(
                "StormScope GOES and MRMS models must share a forecast stride; "
                f"got {self.step} and {mrms_step}"
            )

        self.goes_source: Source = (
            goes_source if goes_source is not None else _default_source(goes, 0, "GOES")
        )
        self.mrms_source: Source = (
            mrms_source if mrms_source is not None else _default_source(mrms, 0, "MRMS")
        )
        self.conditioning_source: Source | None = conditioning_source
        if conditioning_source is None and not self._conditioning_from_goes():
            self.conditioning_source = _default_source(mrms, 1, "MRMS conditioning")
        self.goes_forcing_source: Source | None = None
        if self.goes_forcing_signature is not None:
            self.goes_forcing_source = (
                goes_forcing_source
                if goes_forcing_source is not None
                else _default_source(goes, 1, "GOES forcing")
            )

    def _conditioning_from_goes(self) -> bool:
        """Whether the GOES history fetched for the GOES model covers the radar conditioning."""
        goes_vars = set(self.goes_signature["variable"].values.tolist())
        cond = self.conditioning_signature
        return (
            set(cond["variable"].values.tolist()) <= goes_vars
            and np.isin(
                cond["lead_time"].values, self.goes_signature["lead_time"].values
            ).all()
        )

    # -- planning ------------------------------------------------------------

    def to(self, device: torch.device) -> StormScopeRunner:
        """Move both models to ``device`` and fetch onto it."""
        self.device = torch.device(device)
        self.goes = self.goes.to(self.device)
        self.mrms = self.mrms.to(self.device)
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Return the number of coupled steps needed to reach ``final_lead_time``."""
        return _nsteps(final_lead_time, self.step)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return the ``goes`` and ``mrms`` stream signatures for one item."""
        nsteps = self.nsteps(final_lead_time)
        goes_oc = cast(CoordinateSystem, self.goes.output_coords(self.goes_signature))
        mrms_oc = cast(CoordinateSystem, self.mrms.output_coords(self.mrms_signature))
        leads = _lead_times(goes_oc, nsteps)
        return {
            "goes": _signature_for_stream(goes_oc, leads),
            "mrms": _signature_for_stream(mrms_oc, leads),
        }

    def _request(
        self, source: Source, item: WorkItem, signature: CoordinateSystem
    ) -> DataRequest:
        return DataRequest(
            source,
            np.array([item.time]),
            signature["variable"].values,
            signature["lead_time"].values,
        )

    def _forcing_request(
        self, item: WorkItem, output_leads: np.ndarray | None = None
    ) -> DataRequest | None:
        """GOES forcing: the initial window, or the new frames at ``output_leads``."""
        signature = self.goes_forcing_signature
        if signature is None or self.goes_forcing_source is None:
            return None
        leads = signature["lead_time"].values
        if output_leads is not None:
            if "lead_time" not in signature.dims:
                return None
            leads = leads[-1] + output_leads
        return DataRequest(
            self.goes_forcing_source,
            np.array([item.time]),
            signature["variable"].values,
            leads,
        )

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """Describe the GOES, radar, conditioning and forcing fetches of ``item``."""
        requests = [
            self._request(self.goes_source, item, self.goes_signature),
            self._request(self.mrms_source, item, self.mrms_signature),
        ]
        if self.conditioning_source is not None:
            requests.append(
                self._request(
                    self.conditioning_source, item, self.conditioning_signature
                )
            )
        initial = self._forcing_request(item)
        if initial is not None:
            requests.append(initial)
            for k in range(max(self.nsteps(item.final_lead_time) - 1, 0)):
                step = self._forcing_request(item, self.output_leads + k * self.step)
                if step is not None:
                    requests.append(step)
        return tuple(requests)

    # -- execution -----------------------------------------------------------

    def _fetch_forcing(self, request: DataRequest | None) -> tuple[xr.DataArray, ...]:
        if request is None or self.goes_forcing_signature is None:
            return ()
        x = _fetch(request, self.goes_forcing_signature, self.device)
        if "lead_time" not in self.goes_forcing_signature.dims:
            x = x.isel(lead_time=0, drop=True)
        return (x,)

    def _initial_conditioning(
        self, item: WorkItem, x_goes: xr.DataArray
    ) -> xr.DataArray:
        """GOES history initializing the radar model."""
        signature = self.conditioning_signature
        if self.conditioning_source is not None:
            return _fetch(
                self._request(self.conditioning_source, item, signature),
                signature,
                self.device,
            )
        leads = signature["lead_time"].values.astype("timedelta64[ns]")
        return _map_field(x_goes.sel(lead_time=leads), signature)

    def _conditioning_frames(self, y_goes: xr.DataArray) -> xr.DataArray:
        """GOES forecast frames conditioning the next radar step."""
        return _map_field(y_goes, self.conditioning_signature)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish both initial conditions, then every coupled step through the final lead time."""
        if len(item.member_ids) > 1:
            raise ValueError("StormScopeRunner does not batch ensemble members")
        member_id = item.member_ids[0] if item.member_ids else 0
        _seed_model(self.seed, item.time, member_id, self.goes)
        _seed_model(self.seed, item.time, member_id, self.mrms)
        nsteps = self.nsteps(item.final_lead_time)

        x_goes = _fetch(
            self._request(self.goes_source, item, self.goes_signature),
            self.goes_signature,
            self.device,
        )
        x_mrms = _fetch(
            self._request(self.mrms_source, item, self.mrms_signature),
            self.mrms_signature,
            self.device,
        )
        yield {"goes": initial_condition(x_goes), "mrms": initial_condition(x_mrms)}
        if nsteps == 0:
            return
        conditioning = self._initial_conditioning(item, x_goes)
        forcing = self._fetch_forcing(self._forcing_request(item))

        mrms_iterator = self.mrms.create_iterator(x_mrms, conditioning)
        goes_iterator = self.goes.create_iterator(x_goes, *forcing)
        try:
            y_mrms = cast(xr.DataArray, next(mrms_iterator))
            y_goes = cast(xr.DataArray, next(goes_iterator))
            yield {"goes": y_goes, "mrms": y_mrms}
            for _ in range(nsteps - 1):
                # Radar first: it consumes the GOES forecast at its own lead times.
                y_mrms = cast(
                    xr.DataArray, mrms_iterator.send(self._conditioning_frames(y_goes))
                )
                step_forcing = self._fetch_forcing(
                    self._forcing_request(item, y_goes.coords["lead_time"].values)
                )
                y_goes = cast(xr.DataArray, goes_iterator.send(step_forcing or None))
                yield {"goes": y_goes, "mrms": y_mrms}
        finally:
            mrms_iterator.close()
            goes_iterator.close()


def _region(node: Any) -> Region | None:
    """Build a :class:`~earth2studio.grids.Region` from ``cfg.model.region``."""
    if not node:
        return None
    return Region(
        kind=str(node["kind"]), params=tuple(float(p) for p in node["params"])
    )


_SOURCE_KEYS = ("ic_source", "conditioning_source", "forcing_source")


def _model_node(node: DictConfig) -> dict[str, Any]:
    """The model-loading part of a StormScope node, without its source slots."""
    return {
        k: v
        for k, v in OmegaConf.to_container(node, resolve=True).items()
        if k not in _SOURCE_KEYS
    }


def stormscope(cfg: DictConfig, prefer_predownloaded: bool = True) -> StormScopeRunner:
    """Build the coupled GOES to MRMS runner from ``cfg.model.{goes, mrms}``.

    Each model node loads through :func:`src.models.load_from_node` (its
    ``load_args`` such as ``model_name``, ``amp`` and ``compile`` pass to
    ``load_model``).  Both models take an optional ``cfg.model.region`` box
    (``{kind: box, params: [west, south, east, north]}``).  Sources resolve
    per role: ``goes`` and ``mrms`` from each node's ``ic_source``,
    ``mrms_conditioning`` from ``cfg.model.mrms.conditioning_source`` and
    ``goes_forcing`` from ``cfg.model.goes.forcing_source``; an unset node
    falls back to the predownloaded store, then the model default.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True

    Returns
    -------
    StormScopeRunner
        Runner yielding the ``goes`` and ``mrms`` streams.
    """
    node = cfg.model
    if "goes" not in node or "mrms" not in node:
        raise ValueError(
            "The stormscope runner needs cfg.model.goes and cfg.model.mrms nodes; "
            "see cfg/model/stormscope_goes_mrms.yaml"
        )
    goes = load_from_node(_model_node(node.goes), "StormScope GOES model")
    mrms = load_from_node(_model_node(node.mrms), "StormScope MRMS model")
    region = _region(node.get("region"))
    if region is not None:
        logger.info(f"StormScope: restricting both models to region {region}")
        goes = goes.with_region(region)
        mrms = mrms.with_region(region)

    def resolve(role: str, live: Any) -> Any:
        return resolve_source(
            cfg, role, live=live, prefer_predownloaded=prefer_predownloaded
        )

    runner = StormScopeRunner(
        goes,
        mrms,
        goes_source=resolve("goes", node.goes.get("ic_source")),
        mrms_source=resolve("mrms", node.mrms.get("ic_source")),
        conditioning_source=resolve(
            "mrms_conditioning", node.mrms.get("conditioning_source")
        ),
        goes_forcing_source=(
            resolve("goes_forcing", node.goes.get("forcing_source"))
            if goes.forcing_coords() is not None
            else None
        ),
        seed=int(cfg.get("random_seed", 42)),
    )
    roles: dict[str, Any] = {"goes": runner.goes_source, "mrms": runner.mrms_source}
    if runner.conditioning_source is not None:
        roles["mrms_conditioning"] = runner.conditioning_source
    if runner.goes_forcing_source is not None:
        roles["goes_forcing"] = runner.goes_forcing_source
    runner.source_roles = roles  # type: ignore[attr-defined]
    return runner
