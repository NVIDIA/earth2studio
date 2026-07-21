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

"""Data-assimilation (DA) pipelines.

Two evaluation modes for ``AssimilationModel``-class models (e.g. HealDA):

* :class:`AssimilationPipeline` — the DA analysis **is** the product.  Each
  work item produces one analysis at its time, written as a single
  ``lead_time=0`` slice (the same layout as ``DiagnosticPipeline``) and
  scored directly against reanalysis.
* :class:`AssimilationForecastPipeline` — the DA analysis **initializes a
  forecast** (e.g. healda+FCN3).  Subclasses ``ForecastPipeline`` and only
  replaces initial-condition acquisition; rollout, diagnostics,
  ensemble/perturbation, output, and scoring are all inherited.

Both pipelines resolve their own inputs (observation DataFrame sources
declared under ``cfg.model.da.obs_sources``), so ``needs_data_source`` is
``False`` and ``main.py`` skips top-level source resolution.  The DA model
is expected to emit its analysis on the grid evaluation happens on — for
HealDA, load with ``lat_lon: true`` and an ``output_resolution`` matching
the verification source (``[721, 1440]`` for ARCO/ERA5 0.25°).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.da.base import AssimilationModel
from earth2studio.models.prognostic.base import PrognosticModel
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from ..assimilation import (
    AssimilationRunner,
    ObsSourceSet,
    analysis_spatial_ref,
    analysis_to_tensor,
    analysis_variables,
    build_runner,
    insert_zero_lead_time,
    load_assimilation,
)
from ..output import build_analysis_coords
from ..work import WorkItem
from .base import Pipeline, PredownloadFrameStore, PredownloadStore
from .forecast import ForecastPipeline, _align_to_grid


def _setup_da_components(
    da_cfg: DictConfig,
    device: torch.device,
    output_path: str | None = None,
) -> tuple[AssimilationModel, ObsSourceSet, AssimilationRunner]:
    """Load the DA model, its observation sources, and the runner.

    When *output_path* is given, observation sources are resolved with
    predownloaded frame-store substitution (``obs_<name>.parquet`` dirs
    written by ``predownload.py`` take precedence over live sources).
    """
    if "obs_sources" not in da_cfg:
        raise ValueError(
            "DA model config requires an 'obs_sources' block — one entry "
            "per observation input of the model, in input_coords() order."
        )
    model = load_assimilation(da_cfg).to(device)
    obs_set = ObsSourceSet.from_config(
        da_cfg.obs_sources, model.input_coords(), output_path=output_path
    )
    runner = build_runner(da_cfg, model, obs_set)
    return model, obs_set, runner


def _obs_predownload_enabled(cfg: DictConfig) -> bool:
    """Whether observation predownload is enabled (default: true)."""
    pd_cfg = cfg.get("predownload", {})
    obs_cfg = pd_cfg.get("observations", {}) if pd_cfg else {}
    return bool(obs_cfg.get("enabled", True)) if obs_cfg is not None else True


def _declare_obs_frame_stores(
    cfg: DictConfig,
    times: list[np.datetime64],
) -> list[PredownloadFrameStore]:
    """Build the obs frame-store declarations shared by both DA pipelines.

    Loads the DA model (for its input schemas) and the *live* observation
    sources (no frame-store substitution — predownload must fetch from
    the real sources), then declares one store per enabled source at the
    given analysis *times*.
    """
    if not _obs_predownload_enabled(cfg):
        logger.info("Observation predownload disabled — obs fetched live.")
        return []
    model = load_assimilation(cfg.model.da)
    obs_set = ObsSourceSet.from_config(cfg.model.da.obs_sources, model.input_coords())
    return obs_set.predownload_frame_stores(times)


class AssimilationPipeline(Pipeline):
    """Analysis-mode DA evaluation — score the analysis against reanalysis.

    For each work item, fetches observations around the item's time, runs
    the DA model once, and yields a single analysis field on the model's
    output grid.  The output store carries a singleton ``lead_time=[0]``
    axis so the standard scoring and report machinery apply unchanged
    (verification is aligned at ``valid_time = time + 0``).
    """

    needs_data_source = False

    model: AssimilationModel
    obs_set: ObsSourceSet
    runner: AssimilationRunner

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        self.model, self.obs_set, self.runner = _setup_da_components(
            cfg.model.da, device, output_path=cfg.output.path
        )
        self._spatial_ref = analysis_spatial_ref(self.model)

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        return build_analysis_coords(
            times,
            ensemble_size,
            spatial_ref=self.effective_spatial_ref(),
        )

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare the verification store (analyses need no gridded IC data).

        Observations are fetched live at inference time (the DataFrame
        sources have their own caches); only gridded verification data is
        predownloaded, at the analysis times themselves.  The store grid
        comes from the DA model's output coords, so the verification
        source must return data on that grid (e.g. ARCO with HealDA's
        ``output_resolution: [721, 1440]``) — a mismatch fails loudly at
        download time.
        """
        from ..predownload_utils import declare_verification_only_store
        from ..work import build_work_items

        # Gate before the (expensive) model load.
        pd_cfg = cfg.get("predownload", {})
        verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
        if not verif_cfg.get("enabled", False) or (
            cfg.get("verification_source") is not None
        ):
            return []

        model = load_assimilation(cfg.model.da)
        times: list[np.datetime64] = sorted({i.time for i in build_work_items(cfg)})
        return declare_verification_only_store(
            cfg,
            verif_variables=list(cfg.output.variables),
            verif_times=times,
            spatial_ref=analysis_spatial_ref(model),
        )

    def predownload_frame_stores(self, cfg: DictConfig) -> list[PredownloadFrameStore]:
        """Declare one obs frame store per enabled source, at the work-item times."""
        from ..work import build_work_items

        times: list[np.datetime64] = sorted({i.time for i in build_work_items(cfg)})
        return _declare_obs_frame_stores(cfg, times)

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        if hasattr(self.model, "set_rng"):
            self.model.set_rng(seed=item.seed, reset=True)
        else:
            # No-op for deterministic DA models; makes stochastic ones
            # reproducible per ensemble member.
            torch.manual_seed(item.seed)

        analysis = self.runner.analysis(item.time)
        x, coords = analysis_to_tensor(analysis, device)
        x, coords = insert_zero_lead_time(x, coords)
        yield x, coords


class AssimilationForecastPipeline(ForecastPipeline):
    """DA-initialized forecast — e.g. healda+FCN3.

    Runs the standard prognostic rollout, but assembles each initial
    condition from DA analyses instead of a gridded data source: one
    analysis per entry in the prognostic's input ``lead_time`` axis
    (single-time models need one; history models get one per offset).

    Prognostic input variables the DA model does not produce are filled
    from the standard IC path (``cfg.ic_source`` BYO → predownloaded
    ``data.zarr`` → ``cfg.data_source``), unless
    ``cfg.model.fill_missing_variables`` is set to ``false`` — in which
    case any gap is an error.

    Model config layout (see ``cfg/model/healda_fcn3.yaml``)::

        model:
            da:       {architecture: ..., load_args: ..., obs_sources: ...}
            forecast: {architecture: ..., load_args: ...}
    """

    needs_data_source = False

    da_model: AssimilationModel
    obs_set: ObsSourceSet
    runner: AssimilationRunner
    _missing_vars: list[str]
    _fill_source: DataSource | None

    @staticmethod
    def _model_node(cfg: DictConfig) -> DictConfig:
        return cfg.model.forecast

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        super().setup(cfg, device)
        self.da_model, self.obs_set, self.runner = _setup_da_components(
            cfg.model.da, device, output_path=cfg.output.path
        )

        da_vars = set(analysis_variables(self.da_model))
        prognostic_vars = [str(v) for v in self._prognostic_ic["variable"]]
        self._missing_vars = [v for v in prognostic_vars if v not in da_vars]

        self._fill_source = None
        if self._missing_vars:
            if not cfg.model.get("fill_missing_variables", True):
                raise ValueError(
                    "The DA analysis does not provide these prognostic "
                    f"input variables: {self._missing_vars}.  Either allow "
                    "filling them from the IC data source "
                    "(model.fill_missing_variables=true) or use a "
                    "prognostic whose inputs the DA model covers."
                )
            logger.warning(
                f"DA analysis is missing {len(self._missing_vars)} of "
                f"{len(prognostic_vars)} prognostic input variables "
                f"({self._missing_vars}) — filling from the IC data source."
            )
            from ..data import resolve_ic_source

            self._fill_source = resolve_ic_source(
                cfg, byo=cfg.get("ic_source"), live_source=cfg.data_source
            )

    def _fetch_initial_state(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> tuple[torch.Tensor, CoordSystem]:
        ic_leads = np.asarray(self._prognostic_ic["lead_time"])

        # One analysis per input lead offset (offsets are <= 0 for
        # history models, so these are analyses at or before item.time).
        slices: list[torch.Tensor] = []
        analysis_coords: CoordSystem | None = None
        for lt in ic_leads:
            analysis = self.runner.analysis(item.time + lt)
            x_a, analysis_coords = analysis_to_tensor(analysis, device)
            slices.append(x_a.squeeze(0))  # drop singleton time dim
        if analysis_coords is None:
            raise ValueError(
                "Failed to extract coordinates from analysis — 'analysis_coords' is None. "
                "This likely means the analysis runner did not return a valid analysis "
                "at the requested time(s), or the analysis object is missing required metadata. "
                "Check that your DA runner produces valid analysis results and that the "
                "'analysis_to_tensor' function is compatible with your analysis structure."
            )

        x = torch.stack(slices, dim=0).unsqueeze(0)
        coords: CoordSystem = OrderedDict()
        coords["time"] = np.array([item.time], dtype="datetime64[ns]")
        coords["lead_time"] = ic_leads
        for dim, values in analysis_coords.items():
            if dim != "time":
                coords[dim] = values

        x, coords = _align_to_grid(x, coords, self._prognostic_ic)

        if self._fill_source is not None:
            x_fill, coords_fill = fetch_data(
                source=self._fill_source,
                time=[item.time],
                variable=np.array(self._missing_vars),
                lead_time=ic_leads,
                device=device,
            )
            x_fill, coords_fill = _align_to_grid(
                x_fill, coords_fill, self._prognostic_ic
            )
            coords_fill = OrderedDict(coords_fill)
            coords_fill["time"] = coords["time"]
            x, coords = cat_coords((x, x_fill), (coords, coords_fill), "variable")

        x, coords = map_coords(x, coords, self._prognostic_ic)
        return x, coords

    _predownload_prognostic = None
    """Prognostic model memoized across the predownload hooks —
    :meth:`predownload_stores` and :meth:`predownload_frame_stores` both
    need its ``input_coords()``, and loading a large model twice in one
    ``predownload.py`` run is wasteful."""

    def _load_prognostic_for_predownload(self, cfg: DictConfig) -> PrognosticModel:
        if self._predownload_prognostic is None:
            from ..models import load_prognostic

            self._predownload_prognostic = load_prognostic(cfg, self._model_node(cfg))
        return self._predownload_prognostic

    def predownload_frame_stores(self, cfg: DictConfig) -> list[PredownloadFrameStore]:
        """Declare obs frame stores at every analysis time IC assembly touches.

        One analysis is requested per prognostic input lead offset, so
        obs are cached at ``IC time + offset`` for every IC time (offsets
        are ``[0]`` for single-time models; history models add earlier
        entries).
        """
        from ..work import build_work_items

        if not _obs_predownload_enabled(cfg):
            logger.info("Observation predownload disabled — obs fetched live.")
            return []

        model = self._load_prognostic_for_predownload(cfg)
        ic_lead_times = model.input_coords()["lead_time"]
        unique_ic_times = sorted({i.time for i in build_work_items(cfg)})
        times: list[np.datetime64] = sorted(
            {t + lt for t in unique_ic_times for lt in ic_lead_times}
        )
        return _declare_obs_frame_stores(cfg, times)

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare verification plus (only if needed) a fill-variable IC store.

        The IC store is narrowed to the prognostic input variables the DA
        analysis does not provide; when the DA model covers all inputs, no
        IC store is declared at all and only verification is fetched.
        """
        from ..predownload_utils import (
            compute_verification_times,
            declare_single_source_stores,
            declare_verification_only_store,
            infer_step_hours,
            single_source_stores_disabled,
        )
        from ..work import build_work_items

        if single_source_stores_disabled(cfg):
            return []

        model = self._load_prognostic_for_predownload(cfg)
        ic_coords = model.input_coords()
        spatial_ref = model.output_coords(ic_coords)

        all_items = build_work_items(cfg)
        unique_ic_times: list[np.datetime64] = sorted({i.time for i in all_items})
        step_hours = infer_step_hours(model)
        verif_times = compute_verification_times(
            unique_ic_times, cfg.nsteps, step_hours
        )

        da_model = load_assimilation(cfg.model.da)
        da_vars = set(analysis_variables(da_model))
        missing = [str(v) for v in ic_coords["variable"] if str(v) not in da_vars]

        if not missing:
            return declare_verification_only_store(
                cfg,
                verif_variables=list(cfg.output.variables),
                verif_times=verif_times,
                spatial_ref=spatial_ref,
            )

        ic_lead_times = ic_coords["lead_time"]
        # Fill data is fetched at every analysis time the IC assembly
        # touches (IC time + each input lead offset).
        ic_fetch_times: list[np.datetime64] = sorted(
            {t + lt for t in unique_ic_times for lt in ic_lead_times}
        )
        return declare_single_source_stores(
            cfg,
            ic_variables=missing,
            ic_times=ic_fetch_times,
            verif_variables=list(cfg.output.variables),
            verif_times=verif_times,
            spatial_ref=spatial_ref,
        )
