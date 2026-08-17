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

"""Standard forecast and diagnostic pipelines."""

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
from ..output import build_diagnostic_coords, build_forecast_coords
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


def _spatial_ref_from_output_coords(coords: CoordSystem) -> CoordSystem:
    """Strip a generative model's own ``sample`` axis from a coords reference.

    ``src/output.py`` and ``src/online.py`` classify any dim outside
    ``{batch, time, lead_time, variable, ensemble}`` as spatial.  A
    generative diagnostic's ``sample`` axis (e.g. CorrDiff) is not
    spatial — each call's output gets it renamed to ``ensemble`` via
    :func:`_rename_sample_axis` instead, and the store's ``ensemble`` axis
    is already sized from ``ensemble_size``.  Used wherever a diagnostic's
    raw ``output_coords()`` stands in for the pipeline's static spatial
    reference (the output-coords schema, the write-time variable/spatial
    filter, predownload grids), so ``sample`` never leaks in as a bogus
    spatial dimension.
    """
    return OrderedDict((d, v) for d, v in coords.items() if d != "sample")


def _rename_sample_axis(
    x: torch.Tensor, coords: CoordSystem, member_ids: np.ndarray
) -> tuple[torch.Tensor, CoordSystem]:
    """Rename a generative model's own ``sample`` axis to ``ensemble``.

    CorrDiff-style diagnostics emit their own ``sample`` dimension rather
    than expressing draws through the pipeline's ensemble machinery.  This
    routes it through the existing ensemble path instead, carrying this
    rank's *global* member ids — not ``0..S-1`` — so that
    ``OnlineScorer._check_member_block`` validates them against the rank's
    actual member block rather than silently accepting a renumbered one.
    ``number_of_samples`` is set from :attr:`Pipeline._members_per_rank` by
    :meth:`Pipeline.seed_member`, so ``len(coords["sample"])`` is expected
    to equal ``len(member_ids)``.

    No-op when *coords* carries no ``sample`` dim.
    """
    if "sample" not in coords:
        return x, coords
    n = len(coords["sample"])
    if n != len(member_ids):
        raise ValueError(
            f"Diagnostic produced {n} sample(s) but this call covers "
            f"{len(member_ids)} ensemble member(s) ({list(member_ids)}) — "
            "number_of_samples must equal the member block size."
        )
    coords = OrderedDict(
        (("ensemble", np.asarray(member_ids)) if k == "sample" else (k, v))
        for k, v in coords.items()
    )
    return x, coords


def _broadcast_ensemble(
    x: torch.Tensor, coords: CoordSystem, member_ids: np.ndarray
) -> tuple[torch.Tensor, CoordSystem]:
    """Insert a broadcast ``ensemble`` axis right before ``variable``.

    Used to align a deterministic diagnostic's (or the raw fetched input's)
    output with another diagnostic's ``sample``-derived ``ensemble`` axis
    before :func:`~earth2studio.utils.coords.cat_coords`, which requires
    every operand to carry the exact same dim names in the exact same
    order.  ``variable`` is where a batch-decorated model's own leading
    dim (here, the renamed ``sample`` axis) always lands relative to any
    pass-through dims it received (``time``, ``lead_time``) — right after
    them, right before its own declared output dims.  No-op when
    ``ensemble`` is already present.
    """
    if "ensemble" in coords:
        return x, coords
    keys = list(coords.keys())
    axis = keys.index("variable") if "variable" in keys else len(keys)
    x = x.unsqueeze(axis)
    shape = [-1] * x.ndim
    shape[axis] = len(member_ids)
    x = x.expand(*shape).contiguous()
    new_coords: CoordSystem = OrderedDict()
    for i, k in enumerate(keys):
        if i == axis:
            new_coords["ensemble"] = np.asarray(member_ids)
        new_coords[k] = coords[k]
    if axis == len(keys):
        new_coords["ensemble"] = np.asarray(member_ids)
    return x, new_coords


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


class DiagnosticPipeline(Pipeline):
    """Diagnostic-only pipeline (no prognostic rollout).

    Fetches input data at analysis time (lead_time=0) for each work item,
    runs all diagnostic models, and yields a single ``(tensor, coords)``
    pair containing the accumulated diagnostic output.
    """

    diagnostics: list[DiagnosticModel]
    _dx_input_coords: dict[int, CoordSystem]
    _all_input_vars: list[str]
    _zero_lead: np.ndarray

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        self.diagnostics = [dx.to(device) for dx in load_diagnostics(cfg)]
        if not self.diagnostics:
            raise ValueError(
                "Diagnostic pipeline requires at least one entry in 'diagnostics'."
            )

        self._dx_input_coords = {id(dx): dx.input_coords() for dx in self.diagnostics}

        # Build the union of all input variables needed from the data source.
        all_input_vars: list[str] = []
        seen: set[str] = set()
        for dx in self.diagnostics:
            for v in self._dx_input_coords[id(dx)]["variable"]:
                if v not in seen:
                    all_input_vars.append(str(v))
                    seen.add(str(v))
        self._all_input_vars = all_input_vars

        dx0 = self.diagnostics[0]
        self._spatial_ref = _spatial_ref_from_output_coords(
            dx0.output_coords(self._dx_input_coords[id(dx0)])
        )
        self._zero_lead = np.array([np.timedelta64(0, "ns")])

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        return build_diagnostic_coords(
            self.diagnostics,
            times,
            ensemble_size,
            spatial_ref=self.effective_spatial_ref(),
        )

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare IC + optional verification stores for a diagnostic run.

        Verification always lives in a separate store because diagnostic
        inputs and verification variables rarely overlap.
        """
        from ..predownload_utils import (
            declare_single_source_stores,
            single_source_stores_disabled,
            union_variables,
        )
        from ..work import build_work_items

        if single_source_stores_disabled(cfg):
            return []

        diagnostics = load_diagnostics(cfg)
        if not diagnostics:
            raise ValueError(
                "Diagnostic pipeline requires at least one entry in 'diagnostics'."
            )

        input_variables = union_variables(
            *([str(v) for v in dx.input_coords()["variable"]] for dx in diagnostics)
        )

        all_items = build_work_items(cfg)
        unique_times: list[np.datetime64] = sorted({i.time for i in all_items})

        dx0 = diagnostics[0]
        spatial_ref = _spatial_ref_from_output_coords(
            dx0.output_coords(dx0.input_coords())
        )

        return declare_single_source_stores(
            cfg,
            ic_variables=input_variables,
            ic_times=unique_times,
            verif_variables=list(cfg.output.variables),
            verif_times=unique_times,
            spatial_ref=spatial_ref,
            always_separate_verification=True,
        )

    def stochastic_components(self) -> list[Any]:
        return list(self.diagnostics)

    def _run_diagnostics(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        member_ids: np.ndarray,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run every diagnostic and accumulate outputs onto the raw input.

        A generative diagnostic's ``sample`` axis is renamed to
        ``ensemble`` (carrying *member_ids*) before merging.  Once any
        operand carries that axis, every other operand — including the
        raw fetched input — is broadcast to match before concatenation,
        since :func:`~earth2studio.utils.coords.cat_coords` requires
        identical dim names/order across all its operands.
        """
        x_combined, coords_combined = x, coords
        for dx in self.diagnostics:
            dx_ic = self._dx_input_coords[id(dx)]
            x_in, coords_in = map_coords(x, coords, dx_ic)
            y, y_coords = dx(x_in, coords_in)
            y, y_coords = _rename_sample_axis(y, y_coords, member_ids)

            if "ensemble" in y_coords or "ensemble" in coords_combined:
                if "ensemble" not in coords_combined:
                    x_combined, coords_combined = _broadcast_ensemble(
                        x_combined, coords_combined, member_ids
                    )
                if "ensemble" not in y_coords:
                    y, y_coords = _broadcast_ensemble(y, y_coords, member_ids)

            x_combined, coords_combined = cat_coords(
                (x_combined, y), (coords_combined, y_coords), "variable"
            )
        return x_combined, coords_combined

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        self.seed_member(item)

        x, coords = fetch_data(
            source=data_source,
            time=[item.time],
            variable=self._all_input_vars,
            lead_time=self._zero_lead,
            device=device,
        )

        yield self._run_diagnostics(x, coords, np.array([item.ensemble_id]))

    def run_item_batched(
        self,
        items: list[WorkItem],
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run several ensemble members of one IC's diagnostics together.

        The input fetch is member-independent (no perturbation stage in
        this pipeline), so it runs once; what varies per member is a
        generative diagnostic's own draw, requested in one call via
        ``number_of_samples = len(items)`` (set by :meth:`seed_member` from
        :attr:`Pipeline._members_per_rank`).  A deterministic diagnostic
        run this way simply produces the same output broadcast across all
        members — correct, if not the point of batching.
        """
        if not items:
            return
        times = {item.time for item in items}
        if len(times) != 1:
            raise ValueError(
                f"run_item_batched expects one initial condition per batch, "
                f"got {sorted(str(t) for t in times)}."
            )

        self.seed_member(items[0])

        x, coords = fetch_data(
            source=data_source,
            time=[items[0].time],
            variable=self._all_input_vars,
            lead_time=self._zero_lead,
            device=device,
        )

        member_ids = np.array([item.ensemble_id for item in items])
        yield self._run_diagnostics(x, coords, member_ids)
