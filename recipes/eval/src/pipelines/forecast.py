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

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

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


class ForecastPipeline(Pipeline):
    """Standard prognostic forecast pipeline with optional diagnostics.

    Runs a prognostic model forward in time from each initial condition,
    optionally applying diagnostic models at every step.  Yields one
    ``(tensor, coords)`` pair per lead-time step (including step 0).
    """

    prognostic: PrognosticModel
    diagnostics: list[DiagnosticModel]
    perturbation: Perturbation | None
    nsteps: int
    _prognostic_ic: CoordSystem
    _dx_input_coords: dict[int, CoordSystem]

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        self.nsteps = cfg.nsteps

        # All ranks must participate in model loading for barrier correctness.
        self.prognostic = load_prognostic(cfg).to(device)
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
        model = load_prognostic(cfg)
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

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        x, coords = fetch_data(
            source=data_source,
            time=[item.time],
            variable=self._prognostic_ic["variable"],
            lead_time=self._prognostic_ic["lead_time"],
            device=device,
        )
        x, coords = _align_to_grid(x, coords, self._prognostic_ic)
        x, coords = map_coords(x, coords, self._prognostic_ic)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        if hasattr(self.prognostic, "set_rng"):
            self.prognostic.set_rng(seed=item.seed, reset=True)
        else:
            # Fallback for stochastic models that don't expose set_rng:
            # seed torch's global RNG so per-ensemble-member draws are
            # reproducible.  No-op for deterministic models.
            torch.manual_seed(item.seed)

        model_iter = self.prognostic.create_iterator(x, coords)

        rank = DistributedManager().rank

        for step, (x_step, coords_step) in enumerate(
            tqdm(
                model_iter,
                total=self.nsteps + 1,
                desc=f"IC {item.time}",
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
        self._spatial_ref = dx0.output_coords(self._dx_input_coords[id(dx0)])
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
        spatial_ref = dx0.output_coords(dx0.input_coords())

        return declare_single_source_stores(
            cfg,
            ic_variables=input_variables,
            ic_times=unique_times,
            verif_variables=list(cfg.output.variables),
            verif_times=unique_times,
            spatial_ref=spatial_ref,
            always_separate_verification=True,
        )

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        x, coords = fetch_data(
            source=data_source,
            time=[item.time],
            variable=self._all_input_vars,
            lead_time=self._zero_lead,
            device=device,
        )

        # Run each diagnostic, accumulating outputs into the state.
        x_combined, coords_combined = x, coords
        for dx in self.diagnostics:
            dx_ic = self._dx_input_coords[id(dx)]
            x_in, coords_in = map_coords(x, coords, dx_ic)
            y, y_coords = dx(x_in, coords_in)
            x_combined, coords_combined = cat_coords(
                (x_combined, y), (coords_combined, y_coords), "variable"
            )

        yield x_combined, coords_combined
