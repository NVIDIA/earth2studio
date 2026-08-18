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

"""Diagnostic-only pipeline (no prognostic rollout)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from ..models import load_diagnostics
from ..output import build_diagnostic_coords
from ..work import WorkItem
from .base import Pipeline, PredownloadStore


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
