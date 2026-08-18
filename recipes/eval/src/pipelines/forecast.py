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

"""Standard prognostic forecast pipeline."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from ..distributed import get_rank
from ..models import load_diagnostics, load_prognostic
from ..output import build_forecast_coords
from ..work import WorkItem
from .base import Pipeline, PredownloadStore


def _align_to_grid(
    x: torch.Tensor,
    coords: CoordSystem,
    target: CoordSystem,
    method: str = "linear",
) -> tuple[torch.Tensor, CoordSystem]:
    """Regrid a fetched tensor to the target's lat/lon if they don't match.

    No-op when the source's spatial coords already equal the model's
    native grid — the common case when the data source is configured
    to match the model.  Otherwise runs an xarray interpolation (linear
    by default) so that models whose native resolution differs from the
    underlying source (e.g. 1° GraphCast/GenCast on top of a 0.25° ARCO
    store) can still be driven by the standard pipeline.
    """
    src_lat = coords.get("lat")
    src_lon = coords.get("lon")
    tgt_lat = target.get("lat")
    tgt_lon = target.get("lon")
    if src_lat is None or src_lon is None or tgt_lat is None or tgt_lon is None:
        return x, coords
    if (
        src_lat.shape == tgt_lat.shape
        and src_lon.shape == tgt_lon.shape
        and np.allclose(src_lat, tgt_lat)
        and np.allclose(src_lon, tgt_lon)
    ):
        return x, coords

    dims = list(coords.keys())
    da = xr.DataArray(
        x.detach().cpu().numpy(),
        dims=dims,
        coords={d: np.asarray(coords[d]) for d in dims},
    )
    da = da.interp(lat=tgt_lat, lon=tgt_lon, method=method)
    new_coords = OrderedDict(coords)
    new_coords["lat"] = np.asarray(tgt_lat)
    new_coords["lon"] = np.asarray(tgt_lon)
    return torch.from_numpy(np.asarray(da.values)).to(x.device), new_coords


def _is_stochastic(model: PrognosticModel) -> bool:
    """Best-effort check for a model that draws randomness at inference.

    Two signals, because models express it differently: an explicit
    ``set_rng`` hook (FCN3), or a truthy ``stochastic`` flag for models
    that keep dropout active instead (U-CAST).  Used only to decide
    whether a batched rollout deserves a reproducibility warning, so a
    false negative costs a missing log line, not correctness.
    """
    return hasattr(model, "set_rng") or bool(getattr(model, "stochastic", False))


class ForecastPipeline(Pipeline):
    """Standard prognostic forecast pipeline with optional diagnostics.

    Runs a prognostic model forward in time from each initial condition,
    optionally applying diagnostic models at every step.  Yields one
    ``(tensor, coords)`` pair per lead-time step (including step 0).
    """

    supports_online_scoring = True
    prognostic: PrognosticModel
    diagnostics: list[DiagnosticModel]
    perturbation: Perturbation | None
    nsteps: int
    _prognostic_ic: CoordSystem
    _dx_input_coords: dict[int, CoordSystem]

    @staticmethod
    def _model_node(cfg: DictConfig) -> DictConfig:
        """Config node holding the prognostic model spec.

        Defaults to ``cfg.model``.  Subclasses whose model config nests
        the prognostic next to other components (e.g.
        ``AssimilationForecastPipeline`` with ``cfg.model.forecast``
        beside ``cfg.model.da``) override this.
        """
        return cfg.model

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        self.nsteps = cfg.nsteps

        # All ranks must participate in model loading for barrier correctness.
        self.prognostic = load_prognostic(cfg, self._model_node(cfg)).to(device)
        self.diagnostics = [dx.to(device) for dx in load_diagnostics(cfg)]

        self.perturbation = None
        if cfg.get("ensemble_size", 1) > 1 and "perturbation" in cfg:
            self.perturbation = hydra.utils.instantiate(cfg.perturbation)

        self._prognostic_ic = self.prognostic.input_coords()
        self._spatial_ref = self.prognostic.output_coords(self._prognostic_ic)
        self._dx_input_coords = {id(dx): dx.input_coords() for dx in self.diagnostics}

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        return build_forecast_coords(
            self.prognostic,
            times,
            self.nsteps,
            ensemble_size,
            spatial_ref=self.effective_spatial_ref(),
        )

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare IC + optional verification stores for a standard forecast.

        Builds the IC fetch-time grid (all unique ``t + lead_time`` across
        ICs) and the verification valid-time grid (every output tick across
        the full rollout), then delegates to
        :func:`src.predownload_utils.declare_single_source_stores` for the
        shared BYO / merged / separate-source resolution.
        """
        from ..predownload_utils import (
            compute_verification_times,
            declare_single_source_stores,
            infer_step_hours,
            single_source_stores_disabled,
        )
        from ..work import build_work_items

        if single_source_stores_disabled(cfg):
            return []

        # Inspect the prognostic (CPU — no weights copied to device) to infer
        # IC lead_times, variables, and step stride.
        model = load_prognostic(cfg, self._model_node(cfg))
        ic_coords = model.input_coords()
        spatial_ref = model.output_coords(ic_coords)

        all_items = build_work_items(cfg)
        unique_ic_times: list[np.datetime64] = sorted({i.time for i in all_items})

        ic_variables = list(ic_coords["variable"])
        ic_lead_times = ic_coords["lead_time"]
        ic_fetch_times: list[np.datetime64] = sorted(
            {t + lt for t in unique_ic_times for lt in ic_lead_times}
        )

        step_hours = infer_step_hours(model)
        verif_times = compute_verification_times(
            unique_ic_times, cfg.nsteps, step_hours
        )

        return declare_single_source_stores(
            cfg,
            ic_variables=ic_variables,
            ic_times=ic_fetch_times,
            verif_variables=list(cfg.output.variables),
            verif_times=verif_times,
            spatial_ref=spatial_ref,
        )

    def _fetch_initial_state(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Assemble the prognostic's initial state for one work item.

        Default: fetch from the resolved ``DataSource``, align to the
        model grid, and sub-select onto the model's input coords.
        Subclasses that build the initial state differently (e.g. from a
        data-assimilation analysis) override this — the perturbation /
        RNG / rollout machinery in :meth:`run_item` is shared.
        """
        x, coords = fetch_data(
            source=data_source,
            time=[item.time],
            variable=self._prognostic_ic["variable"],
            lead_time=self._prognostic_ic["lead_time"],
            device=device,
        )
        x, coords = _align_to_grid(x, coords, self._prognostic_ic)
        x, coords = map_coords(x, coords, self._prognostic_ic)
        return x, coords

    def stochastic_components(self) -> list[Any]:
        return [self.prognostic]

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        x, coords = self._fetch_initial_state(item, data_source, device)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        yield from self._rollout(x, coords, item, f"IC {item.time}")

    def run_item_batched(
        self,
        items: list[WorkItem],
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Roll several ensemble members of one IC forward together.

        Members ride on a leading ``ensemble`` axis, which prognostic
        models absorb into their ``batch`` dimension — the same layout
        ``earth2studio.run.ensemble`` uses.

        Perturbations are drawn **per member** rather than once for the
        stacked state.  Seeding the global RNG and letting a single
        ``perturbation`` call fill a ``K``-member tensor would consume
        draws in a different order than the unbatched path, so members
        would silently stop matching a ``members_per_rank=1`` run.  Drawing
        each member under its own seed keeps them identical.

        Model-internal stochasticity is a different matter: ``set_rng``
        seeds the model once for the whole batch, so a stochastic model's
        member *m* will not reproduce its unbatched trajectory.  The
        ensemble remains a valid sample (each batch element still gets
        independent noise) — it is just a different draw, which is why this
        is logged rather than silently accepted.
        """
        if not items:
            return
        times = {item.time for item in items}
        if len(times) != 1:
            raise ValueError(
                f"run_item_batched expects one initial condition per batch, "
                f"got {sorted(str(t) for t in times)}."
            )

        x0, coords0 = self._fetch_initial_state(items[0], data_source, device)
        member_ids = np.array([item.ensemble_id for item in items])

        x = x0.unsqueeze(0).repeat(len(items), *([1] * x0.ndim))
        coords = CoordSystem({"ensemble": member_ids} | dict(coords0))

        if self.perturbation is not None:
            for m, item in enumerate(items):
                member_coords = CoordSystem(
                    {"ensemble": member_ids[m : m + 1]} | dict(coords0)
                )
                torch.manual_seed(item.seed)
                x_m, _ = self.perturbation(x[m : m + 1], member_coords)
                x[m] = x_m[0]

        if len(items) > 1 and _is_stochastic(self.prognostic):
            logger.warning(
                f"{type(self.prognostic).__name__} is stochastic but is being "
                f"driven with {len(items)} members per rollout; the batch is "
                "seeded once, so individual members will not reproduce a "
                "members_per_rank=1 run (the ensemble is still a valid draw)."
            )

        yield from self._rollout(
            x, coords, items[0], f"IC {items[0].time} x{len(items)}"
        )

    def _rollout(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        item: WorkItem,
        label: str,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Drive the prognostic iterator, applying diagnostics at each step.

        Shared by :meth:`run_item` and :meth:`run_item_batched`; the only
        difference between them is how the initial state was assembled.
        ``item`` stands in for the whole batch in :meth:`run_item_batched`
        (its ``seed`` seeds the shared rollout), matching the previous
        per-batch seeding behavior.
        """
        self.seed_member(item)

        model_iter = self.prognostic.create_iterator(x, coords)

        # Rank only gates tqdm output below.
        rank = get_rank()

        for step, (x_step, coords_step) in enumerate(
            tqdm(
                model_iter,
                total=self.nsteps + 1,
                desc=label,
                position=1,
                leave=False,
                disable=rank != 0,
            )
        ):
            for dx in self.diagnostics:
                dx_ic = self._dx_input_coords[id(dx)]
                y, y_coords = map_coords(x_step, coords_step, dx_ic)
                y, y_coords = dx(y, y_coords)
                x_step, coords_step = cat_coords(
                    (x_step, y), (coords_step, y_coords), "variable"
                )

            yield x_step, coords_step

            if step >= self.nsteps:
                break

