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

"""Pipeline abstraction for eval recipe inference loops.

The :class:`Pipeline` ABC defines a standard interface that decouples the
*per-work-item inference logic* from the shared scaffolding (work iteration,
output coordinate filtering, ensemble dimension injection, and zarr writes).

Built-in implementations:

* :class:`ForecastPipeline` — prognostic rollout with optional diagnostics.
* :class:`DiagnosticPipeline` — diagnostic-only (no prognostic model).

To add a custom pipeline, subclass :class:`Pipeline`, implement the three
required methods (:meth:`setup`, :meth:`build_total_coords`,
:meth:`run_item`), and register it in the Hydra config.

Pipelines may also declare:

* :meth:`Pipeline.predownload_stores` — what data needs to be pre-fetched
  for this pipeline (consumed by ``predownload.py``).
* A custom ensemble-injection path by overriding :meth:`_inject_ensemble`
  (e.g. models that already carry ensemble along a batch dimension).
"""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from .data import (
    CadenceRoundedSource,
    CompositeSource,
    PredownloadedSource,
    ValidTimeForecastAdapter,
)
from .models import load_diagnostics, load_prognostic
from .output import (
    OutputManager,
    build_diagnostic_coords,
    build_forecast_coords,
    build_output_coords,
)
from .regrid import NearestNeighborRegridder, RegriddedSource, Regridder
from .work import WorkItem, write_marker

_STRUCTURAL_DIMS = frozenset({"batch", "time", "lead_time", "variable", "ensemble"})


# ======================================================================
# Predownload declarations
# ======================================================================


@dataclass(frozen=True)
class PredownloadStore:
    """Declarative spec for a zarr store that ``predownload.py`` should populate.

    Each pipeline returns a list of these from :meth:`Pipeline.predownload_stores`;
    ``predownload.py`` then iterates, downloading each store's data with resume
    support and per-timestamp progress markers.

    Parameters
    ----------
    name : str
        Logical store name.  Controls the on-disk location
        (``<output.path>/<name>.zarr``) and the progress-marker namespace.
        Typical values: ``"data"``, ``"verification"``, or pipeline-specific
        names like ``"data_goes"``, ``"data_mrms"``, ``"cond_gfs"``.
    source : DataSource
        Data source to fetch from.  May be a :class:`~src.regrid.RegriddedSource`
        that wraps a raw Earth2Studio source with a regridder — in that case the
        stored data is already on the target grid.
    times : list[np.datetime64]
        Timestamps to fetch (analysis times; no lead-time expansion).
    variables : list[str]
        Variable names to fetch.
    spatial_ref : CoordSystem
        Spatial coordinate system of the **stored** data (post-regrid if the
        source is a :class:`~src.regrid.RegriddedSource`).  Used to build the
        zarr schema.
    role : str
        Informational tag: ``"ic"``, ``"verification"``, ``"conditioning"``,
        or ``"data"``.  Shown in logs and can be used to sort / filter
        downstream.  Defaults to ``"data"``.
    extra_coords : dict[str, np.ndarray]
        Optional non-dimensional coordinate arrays to attach to the store
        (e.g. 2D ``lat``/``lon`` alongside ``y``/``x`` for Lambert-conformal
        grids).  Not currently used by the default store writer; reserved
        for future zarr attribute attachment.
    """

    name: str
    source: DataSource
    times: list[np.datetime64]
    variables: list[str]
    spatial_ref: CoordSystem
    role: str = "data"
    extra_coords: dict[str, np.ndarray] = field(default_factory=dict)


# ======================================================================
# Pipeline ABC
# ======================================================================


class Pipeline(ABC):
    """Abstract base class for eval recipe inference pipelines.

    A pipeline encapsulates three concerns that vary between inference modes:

    1. **Setup** — load models, move them to device, and cache any metadata
       needed during inference (e.g. input coordinate systems).
    2. **Output coordinates** — define the full shape of the zarr output store
       (ensemble, time, lead_time, spatial dims).
    3. **Per-item inference** — given a single :class:`WorkItem`, fetch data,
       run models, and *yield* ``(tensor, coords)`` pairs that should be
       written to the output store.

    Everything else — iterating over work items, building the output variable
    filter, optionally regridding model outputs, injecting the ensemble
    dimension, and calling :meth:`OutputManager.write` — is handled by the
    shared :meth:`run` method and does **not** need to be reimplemented.

    Optional hooks
    --------------
    * :meth:`predownload_stores` — declarative predownload spec consumed by
      ``predownload.py``.  Default returns an empty list.
    * :meth:`_inject_ensemble` — how the ensemble dim is expressed in the
      output.  Default prepends a new ``ensemble`` axis; override for
      pipelines that already carry ensemble along another dim (e.g.
      ``batch``).

    Subclassing guide
    -----------------
    Implement these three abstract methods:

    * :meth:`setup` — called once before inference begins.
    * :meth:`build_total_coords` — called once to create the zarr store schema.
    * :meth:`run_item` — called once per work item; yield output tensors.

    The base class provides :attr:`_spatial_ref` which subclasses must set
    during :meth:`setup`.  It is used by :meth:`run` to build the output
    coordinate filter (variable + lat/lon sub-selection).
    """

    _spatial_ref: CoordSystem
    """Reference coordinate system whose spatial dimensions define the output
    grid.  Must be set by :meth:`setup`.  When an output regridder is
    configured, :meth:`effective_spatial_ref` returns the *regridded* target
    coords; ``_spatial_ref`` always refers to the model's native grid."""

    _output_regridder: Regridder | None = None
    """Optional output-side regridder.  When set (by subclasses during
    :meth:`setup`), model outputs are regridded between :meth:`run_item` and
    :meth:`OutputManager.write`, and the zarr store uses the regridder's
    target coords.  Concrete regridders are added in later phases."""

    needs_data_source: bool = True
    """Whether this pipeline consumes the single ``DataSource`` that
    ``main.py`` resolves from ``cfg.ic_source`` / predownloaded cache /
    ``cfg.data_source``.

    Single-source pipelines (``forecast``, ``diagnostic``, ``dlesym``)
    leave this ``True`` — ``main.py`` hands them the resolved source,
    which they use inside :meth:`run_item` for ``fetch_data`` calls.

    Multi-source pipelines (``stormscope``, which needs separate IC
    sources per model plus a conditioning source) should set this to
    ``False`` — ``main.py`` then skips source instantiation and passes
    ``None`` as the ``data_source`` argument.  BYO is handled inside
    the pipeline's own config block (e.g. overriding
    ``cfg.model.goes.ic_source._target_``)."""

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load models and prepare pipeline state.

        Called exactly once before any inference begins.  Implementations
        should:

        * Load and ``.to(device)`` all models.
        * Cache any coordinate metadata needed by :meth:`run_item`
          (e.g. ``input_coords()`` lookups).
        * Set ``self._spatial_ref`` to the model's native spatial coord
          system.
        * Optionally build an output regridder and assign it to
          ``self._output_regridder`` (concrete regridder classes land in
          later phases).

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.
        device : torch.device
            Target device for inference.
        """
        ...

    @abstractmethod
    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        """Build the full output coordinate system for the zarr store.

        The returned ``CoordSystem`` defines every dimension and its values
        for the output arrays.  It is passed directly to
        :meth:`OutputManager.validate_output_store` to create or validate
        the zarr store.

        Typical dimensions: ``[ensemble], time, lead_time, <spatial...>``.
        The ``ensemble`` dimension should be included only when
        ``ensemble_size > 1``.  Spatial dims should come from
        :meth:`effective_spatial_ref` so that output regridders, when
        configured, are honored.

        Parameters
        ----------
        times : np.ndarray
            All initial-condition times (sorted, unique).
        ensemble_size : int
            Total number of ensemble members.  When 1 the ensemble
            dimension should be omitted.

        Returns
        -------
        CoordSystem
            Full coordinate system for the output store.
        """
        ...

    @abstractmethod
    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run inference for a single work item and yield output chunks.

        This is the core method that varies between pipelines.  It should:

        1. Fetch input data for the work item's time.
        2. Run the pipeline's model(s).
        3. ``yield`` one or more ``(tensor, coords)`` pairs on the **model's
           native grid** (:attr:`_spatial_ref`).

        Each yielded pair is automatically:

        * filtered to the requested output variables and native spatial grid,
        * regridded to the output grid (if an output regridder is configured),
        * decorated with an ensemble dimension if needed, and
        * written to the output store.

        Implementations do **not** need to handle any of that.

        Parameters
        ----------
        item : WorkItem
            The work item to process (contains time, ensemble_id, seed).
        data_source : DataSource
            Source for fetching input data.
        device : torch.device
            Target device for tensors.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            ``(data, coords)`` pairs to be written to the output store.
            These should include all variables the pipeline produces
            (the caller handles sub-selection to output variables).
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks with defaults
    # ------------------------------------------------------------------

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare zarr stores that ``predownload.py`` should populate.

        Default implementation returns an empty list, meaning the pipeline
        requires no pre-fetched data.  Concrete pipelines should override
        to declare their IC, verification, and conditioning needs.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.  Pipelines read ``data_source``,
            ``ic_source`` / ``verification_source`` (for BYO skips),
            ``start_times``, ``nsteps``, and ``output.variables`` from
            this.

        Returns
        -------
        list[PredownloadStore]
            One entry per zarr store to populate.
        """
        return []

    def verification_source(self, cfg: DictConfig) -> DataSource | None:
        """Return a verification data source, or ``None`` to use the default.

        ``score.py`` calls this first; when it returns ``None`` the caller
        falls back to its built-in lookup (user-provided
        ``cfg.verification_source``, then ``<output.path>/verification.zarr``,
        then ``<output.path>/data.zarr``).

        Multi-source pipelines whose verification is spread across
        multiple zarrs (e.g. StormScope: ``data_goes.zarr`` +
        ``data_mrms.zarr``) should override this to return a
        :class:`~src.data.CompositeSource` that dispatches variable
        requests to the right underlying store.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.

        Returns
        -------
        DataSource | None
            A :class:`~earth2studio.data.DataSource` to use for
            verification, or ``None`` to fall back to score.py's default.
        """
        return None

    def _inject_ensemble(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        item: WorkItem,
        has_ensemble: bool,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Inject the ensemble dimension before writing.

        Default: ``unsqueeze(0)`` on the tensor and prepend an ``ensemble``
        key carrying ``[item.ensemble_id]`` to the coord system.  Override
        for pipelines that already carry ensemble members along another
        dimension (e.g. ``batch`` for StormScope).

        Parameters
        ----------
        x : torch.Tensor
            Output tensor from :meth:`run_item`, post regrid and filter.
        coords : CoordSystem
            Matching coord system.
        item : WorkItem
            The work item being written (supplies ``ensemble_id``).
        has_ensemble : bool
            True when the output zarr has an ``ensemble`` dimension.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Possibly-modified ``(x, coords)`` pair.
        """
        if not has_ensemble:
            return x, coords
        x = x.unsqueeze(0)
        coords = CoordSystem(
            {"ensemble": np.array([item.ensemble_id])} | dict(coords)
        )
        return x, coords

    def effective_spatial_ref(self) -> CoordSystem:
        """Return the spatial coord system used for the output zarr.

        If an output regridder is configured, returns its target coords
        (the coarsened grid); otherwise returns :attr:`_spatial_ref`
        (the model's native grid).
        """
        if self._output_regridder is None:
            return self._spatial_ref

        merged: CoordSystem = OrderedDict()
        # Preserve any non-spatial keys (e.g. a lingering 'variable' from
        # model output_coords) — in practice spatial_ref only has spatial
        # dims, so this loop is usually a no-op.
        for d, v in self._spatial_ref.items():
            if d in _STRUCTURAL_DIMS:
                merged[d] = v
        for d, v in self._output_regridder.target_coords().items():
            merged[d] = v
        return merged

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    @torch.no_grad()  # not using inference_mode() due to problem with DLWP
    def run(
        self,
        work_items: list[WorkItem],
        data_source: DataSource | None,
        output_mgr: OutputManager,
        output_variables: list[str],
        device: torch.device,
        cfg: DictConfig | None = None,
    ) -> None:
        """Iterate work items, filter+regrid outputs, and write to the store.

        Shared outer loop.  For each work item:

        1. Calls :meth:`run_item`.
        2. Filters each yielded chunk to the configured output variables
           and the model's native spatial grid.
        3. If an output regridder is configured, applies it.
        4. Injects the ensemble dim via :meth:`_inject_ensemble`.
        5. Writes the chunk via :class:`OutputManager`.

        When ``cfg.resume`` is true, a completion marker is written after
        each work item so that resumed runs can skip it.

        Parameters
        ----------
        work_items : list[WorkItem]
            Work items assigned to this rank.
        data_source : DataSource | None
            Source for fetching input data.  ``None`` is allowed for
            pipelines that set :attr:`needs_data_source` to ``False``
            (they manage their own sources internally).
        output_mgr : OutputManager
            Context-managed output handler (store already validated).
        output_variables : list[str]
            Variable names to sub-select before writing.
        device : torch.device
            Target device for inference.
        cfg : DictConfig | None
            Full Hydra config.  Required when ``resume=true`` so that
            completion markers can be written.
        """
        if not work_items:
            logger.warning("No work items for this rank — skipping inference.")
            return

        resume = cfg.get("resume", False) if cfg is not None else False

        # Filter at the model's native grid — cheaper than regridding
        # all model-output channels only to throw some away.
        native_filter = build_output_coords(self._spatial_ref, output_variables)
        has_ensemble = "ensemble" in output_mgr.io.coords

        # Move output regridder to device once.
        if self._output_regridder is not None:
            self._output_regridder = self._output_regridder.to(device)

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        rank = DistributedManager().rank

        for item in tqdm(work_items, desc="Work items", position=0, disable=rank != 0):
            for x_step, coords_step in self.run_item(item, data_source, device):
                x_out, coords_out = map_coords(x_step, coords_step, native_filter)

                if self._output_regridder is not None:
                    x_out, coords_out = self._output_regridder.apply_with_coords(
                        x_out, coords_out
                    )

                x_out, coords_out = self._inject_ensemble(
                    x_out, coords_out, item, has_ensemble
                )
                output_mgr.write(x_out, coords_out)

            if resume:
                output_mgr.flush()
                write_marker(item, cfg)

        logger.success("Inference complete.")


# ======================================================================
# Built-in pipeline implementations
# ======================================================================


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

        Mirrors the original behavior of ``_predownload_forecast`` in
        ``predownload.py``: merges IC and verification into a single
        ``data.zarr`` when they share a source; splits into two stores
        otherwise.  Respects ``ic_source`` / ``verification_source`` BYO
        overrides by omitting the corresponding store.
        """
        # Local imports avoid a cycle through predownload.py at import time.
        from .predownload_utils import (
            compute_verification_times,
            infer_step_hours,
            union_variables,
        )
        from .work import build_work_items

        ic_byo = cfg.get("ic_source") is not None
        verif_byo = cfg.get("verification_source") is not None
        pd_cfg = cfg.get("predownload", {})
        verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
        verif_enabled = verif_cfg.get("enabled", False)
        verif_has_separate_source = verif_cfg.get("source") is not None

        do_verif = verif_enabled and not verif_byo
        do_data_store = not ic_byo
        do_merged = do_data_store and do_verif and not verif_has_separate_source
        do_verif_store = do_verif and (ic_byo or verif_has_separate_source)

        if not do_data_store and not do_verif_store:
            return []

        # Inspect the prognostic (CPU — no weights copied to device) to infer
        # IC lead_times, variables, and step stride.
        model = load_prognostic(cfg)
        ic_coords = model.input_coords()
        spatial_ref = model.output_coords(ic_coords)

        all_items = build_work_items(cfg)
        unique_ic_times: list[np.datetime64] = sorted({i.time for i in all_items})

        stores: list[PredownloadStore] = []

        if do_data_store:
            ic_variables = list(ic_coords["variable"])
            ic_lead_times = ic_coords["lead_time"]
            ic_fetch_times: list[np.datetime64] = sorted(
                {t + lt for t in unique_ic_times for lt in ic_lead_times}
            )

            if do_merged:
                step_hours = infer_step_hours(model)
                verif_times = compute_verification_times(
                    unique_ic_times, cfg.nsteps, step_hours
                )
                verif_variables = list(cfg.output.variables)
                all_times = sorted(set(ic_fetch_times) | set(verif_times))
                all_variables = union_variables(ic_variables, verif_variables)
            else:
                all_times = ic_fetch_times
                all_variables = ic_variables

            stores.append(
                PredownloadStore(
                    name="data",
                    source=hydra.utils.instantiate(cfg.data_source),
                    times=all_times,
                    variables=all_variables,
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

        if do_verif_store:
            step_hours = infer_step_hours(model)
            verif_times = compute_verification_times(
                unique_ic_times, cfg.nsteps, step_hours
            )
            verif_variables = list(cfg.output.variables)
            verif_source_cfg = verif_cfg.get("source") or cfg.data_source
            stores.append(
                PredownloadStore(
                    name="verification",
                    source=hydra.utils.instantiate(verif_source_cfg),
                    times=verif_times,
                    variables=verif_variables,
                    spatial_ref=spatial_ref,
                    role="verification",
                )
            )

        return stores

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
        x, coords = map_coords(x, coords, self._prognostic_ic)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        if hasattr(self.prognostic, "set_rng"):
            self.prognostic.set_rng(seed=item.seed, reset=True)

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
        from .predownload_utils import union_variables
        from .work import build_work_items

        ic_byo = cfg.get("ic_source") is not None
        verif_byo = cfg.get("verification_source") is not None
        pd_cfg = cfg.get("predownload", {})
        verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
        verif_enabled = verif_cfg.get("enabled", False)

        do_data_store = not ic_byo
        do_verif_store = verif_enabled and not verif_byo

        if not do_data_store and not do_verif_store:
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

        stores: list[PredownloadStore] = []
        data_source = None

        if do_data_store:
            data_source = hydra.utils.instantiate(cfg.data_source)
            stores.append(
                PredownloadStore(
                    name="data",
                    source=data_source,
                    times=unique_times,
                    variables=input_variables,
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

        if do_verif_store:
            verif_source_cfg = verif_cfg.get("source")
            if verif_source_cfg is not None:
                verif_source = hydra.utils.instantiate(verif_source_cfg)
            elif data_source is not None:
                verif_source = data_source
            else:
                verif_source = hydra.utils.instantiate(cfg.data_source)
            stores.append(
                PredownloadStore(
                    name="verification",
                    source=verif_source,
                    times=unique_times,
                    variables=list(cfg.output.variables),
                    spatial_ref=spatial_ref,
                    role="verification",
                )
            )

        return stores

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


class DLESyMPipeline(ForecastPipeline):
    """Coupled Earth-system forecast pipeline for DLESyM / DLESyMLatLon.

    DLESyM differs from standard prognostic models in three ways that make it
    incompatible with :class:`ForecastPipeline`:

    1. **Multi-lead-time per iterator step.**  Each call to the model
       advances the state by 96 hours and yields 16 output lead times
       (6h, 12h, …, 96h) in a single chunk — rather than one lead time per
       iterator step.  :attr:`nsteps` therefore counts *model steps*, not
       output lead times; the zarr schema covers ``nsteps × 16 + 1`` lead
       times at 6-hour stride.
    2. **Coupled atmosphere / ocean with different time scales.**  Ocean
       variables are only valid at the 48h and 96h ticks within each 96-hour
       step; atmosphere variables are valid at all 16 ticks.  Invalid ocean
       values are masked to NaN before writing so the zarr store reflects
       physical validity.
    3. **Initial condition yield.**  The model iterator yields the IC window
       (lead_times ``[-48h .. 0h]``) as its first output; those negative
       lead times are outside the output-zarr schema and are explicitly
       skipped.

    The model's own :meth:`retrieve_valid_ocean_outputs` determines which
    lead times are ocean-valid — we don't hardcode the [48h, 96h] list, so
    variants of DLESyM with different ocean cadences would continue to
    work.

    Diagnostics on top of DLESyM are not supported — :meth:`setup` raises
    if ``cfg.diagnostics`` is configured.  The build-in deterministic
    model has no ``set_rng`` method, so per-member seeding is a no-op
    (but the ensemble dim is still honored for repeated runs with
    different perturbations).
    """

    _ocean_variables: list[str]

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        if cfg.get("diagnostics"):
            raise ValueError(
                "DLESyMPipeline does not support diagnostic models; "
                "remove the 'diagnostics' config key."
            )
        super().setup(cfg, device)
        if not hasattr(self.prognostic, "ocean_variables"):
            raise AttributeError(
                "DLESyMPipeline expects the loaded prognostic to expose "
                "`ocean_variables` (DLESyM / DLESyMLatLon from earth2studio). "
                f"Got: {type(self.prognostic).__name__}"
            )
        self._ocean_variables = list(self.prognostic.ocean_variables)

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare IC + optional verification stores for a DLESyM run.

        Verification times are computed by flattening the model's
        per-step output lead times across ``nsteps`` iterator steps —
        this produces the full 6-hourly grid of valid times rather than
        the single-tick-per-step output that
        :meth:`ForecastPipeline.predownload_stores` would give.
        """
        from .predownload_utils import union_variables
        from .work import build_work_items

        ic_byo = cfg.get("ic_source") is not None
        verif_byo = cfg.get("verification_source") is not None
        pd_cfg = cfg.get("predownload", {})
        verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
        verif_enabled = verif_cfg.get("enabled", False)
        verif_has_separate_source = verif_cfg.get("source") is not None

        do_verif = verif_enabled and not verif_byo
        do_data_store = not ic_byo
        do_merged = do_data_store and do_verif and not verif_has_separate_source
        do_verif_store = do_verif and (ic_byo or verif_has_separate_source)

        if not do_data_store and not do_verif_store:
            return []

        # CPU-inspect the model to infer IC requirements + output lead times.
        model = load_prognostic(cfg)
        ic_coords = model.input_coords()
        out_coords = model.output_coords(ic_coords)
        spatial_ref = out_coords  # lat/lon (LatLon) or face/height/width (raw)

        all_items = build_work_items(cfg)
        unique_ic_times: list[np.datetime64] = sorted({i.time for i in all_items})

        # All unique forecast valid times across the full nsteps rollout.
        # Flattens per-step output lead times so every 6h tick is covered.
        verif_times = _unique_forecast_valid_times(
            unique_ic_times, out_coords["lead_time"], cfg.nsteps
        )

        stores: list[PredownloadStore] = []

        if do_data_store:
            ic_variables = list(ic_coords["variable"])
            ic_lead_times = ic_coords["lead_time"]
            ic_fetch_times: list[np.datetime64] = sorted(
                {t + lt for t in unique_ic_times for lt in ic_lead_times}
            )

            if do_merged:
                verif_variables = list(cfg.output.variables)
                all_times = sorted(set(ic_fetch_times) | set(verif_times))
                all_variables = union_variables(ic_variables, verif_variables)
            else:
                all_times = ic_fetch_times
                all_variables = ic_variables

            stores.append(
                PredownloadStore(
                    name="data",
                    source=hydra.utils.instantiate(cfg.data_source),
                    times=all_times,
                    variables=all_variables,
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

        if do_verif_store:
            verif_variables = list(cfg.output.variables)
            verif_source_cfg = verif_cfg.get("source") or cfg.data_source
            stores.append(
                PredownloadStore(
                    name="verification",
                    source=hydra.utils.instantiate(verif_source_cfg),
                    times=verif_times,
                    variables=verif_variables,
                    spatial_ref=spatial_ref,
                    role="verification",
                )
            )

        return stores

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
        x, coords = map_coords(x, coords, self._prognostic_ic)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        if hasattr(self.prognostic, "set_rng"):
            self.prognostic.set_rng(seed=item.seed, reset=True)

        model_iter = self.prognostic.create_iterator(x, coords)

        # Skip the IC yield — its lead_times are in the input window
        # ([-48h..0h] for DLESyM), outside the output zarr schema.
        next(model_iter)

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        rank = DistributedManager().rank

        for step, (x_step, coords_step) in enumerate(
            tqdm(
                model_iter,
                total=self.nsteps,
                desc=f"IC {item.time}",
                position=1,
                leave=False,
                disable=rank != 0,
            )
        ):
            x_step = self._mask_invalid_ocean(x_step, coords_step)
            yield x_step, coords_step

            if step + 1 >= self.nsteps:
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mask_invalid_ocean(
        self,
        x_step: torch.Tensor,
        coords_step: CoordSystem,
    ) -> torch.Tensor:
        """Replace ocean-variable values at non-valid lead times with NaN.

        DLESyM's ocean component predicts only at a subset of the atmos
        output lead times (by default 48h and 96h).  The atmos output
        tensor still has entries for every lead time — those entries are
        meaningless for ocean variables and should not be written to the
        output zarr as if they were forecasts.

        The set of valid ocean lead times is queried from the model
        (via ``retrieve_valid_ocean_outputs``) rather than hardcoded, so
        DLESyM variants with different ocean cadences continue to work.
        """
        if not self._ocean_variables:
            return x_step

        _, valid_coords = self.prognostic.retrieve_valid_ocean_outputs(
            x_step, coords_step
        )
        valid_lt = set(valid_coords["lead_time"].tolist())
        all_lt = list(coords_step["lead_time"].tolist())
        all_vars = list(coords_step["variable"])
        ocean_set = set(self._ocean_variables)

        lt_invalid = torch.tensor(
            [lt not in valid_lt for lt in all_lt],
            device=x_step.device,
        )
        var_ocean = torch.tensor(
            [v in ocean_set for v in all_vars],
            device=x_step.device,
        )
        if not lt_invalid.any() or not var_ocean.any():
            return x_step

        lt_axis = list(coords_step.keys()).index("lead_time")
        var_axis = list(coords_step.keys()).index("variable")

        lt_shape = [1] * x_step.ndim
        lt_shape[lt_axis] = -1
        var_shape = [1] * x_step.ndim
        var_shape[var_axis] = -1

        mask = lt_invalid.view(lt_shape) & var_ocean.view(var_shape)
        nan = torch.full_like(x_step, float("nan"))
        return torch.where(mask, nan, x_step)


def _unique_forecast_valid_times(
    ic_times: list[np.datetime64],
    step_lead_times: np.ndarray,
    nsteps: int,
) -> list[np.datetime64]:
    """Flatten per-step output lead times into the full set of valid times.

    For a model whose single forward pass outputs multiple lead times
    (e.g. DLESyM: 16 per step), :func:`compute_verification_times` from
    ``predownload_utils`` *undercounts* — it assumes one tick per step.
    This helper takes the model's output lead times and stride (derived
    from the last entry) and walks ``nsteps`` iterator steps to produce
    the full time set, including the IC itself.

    Parameters
    ----------
    ic_times : list[np.datetime64]
        Unique initial-condition times.
    step_lead_times : np.ndarray
        Output lead times from a single model step, e.g. ``[6h, 12h, …, 96h]``.
    nsteps : int
        Number of iterator steps.

    Returns
    -------
    list[np.datetime64]
        Sorted unique set of ``{ic + lead}`` across all ICs and all
        flattened lead times.
    """
    step_stride = step_lead_times[-1]
    zero = np.array([np.timedelta64(0, "ns")])
    all_offsets = np.concatenate(
        [
            zero,
            np.asarray([step_lead_times + step_stride * i for i in range(nsteps)])
            .flatten()
            .astype("timedelta64[ns]"),
        ]
    )
    return sorted({t + off for t in ic_times for off in all_offsets})


# ======================================================================
# StormScope coupled nowcasting pipeline
# ======================================================================


class StormScopePipeline(Pipeline):
    """Coupled GOES/MRMS nowcasting pipeline using StormScope models.

    Runs two prognostic models together:

    * :class:`earth2studio.models.px.StormScopeGOES` — forecasts GOES
      satellite channels.  Conditioned on synoptic-scale GFS data (60-min
      variants) or nothing (10-min variants).
    * :class:`earth2studio.models.px.StormScopeMRMS` — forecasts MRMS
      composite reflectivity (``refc``), conditioned on GOES (either
      observations at IC time or the GOES model's own predictions
      during rollout).

    Both models share the HRRR Lambert-conformal grid (``y``, ``x``)
    and the pipeline yields a combined ``(variables × y × x)`` tensor
    per forecast step — GOES channels and MRMS refc stacked along the
    ``variable`` axis so the output zarr is a single unified store.

    Config shape
    ------------
    Expected structure under ``cfg.model``::

        goes:
            architecture: earth2studio.models.px.StormScopeGOES
            load_args:
                model_name: 6km_60min_natten_cos_zenith_input_eoe_v2
                conditioning_data_source: {_target_: earth2studio.data.GFS_FX}
            ic_source: {_target_: earth2studio.data.GOES, satellite: goes16, scan_mode: C}
            ic_grid: {_target_: src.stormscope.goes_grid, satellite: goes16, scan_mode: C}
            conditioning_grid: {_target_: src.stormscope.gfs_grid}
        mrms:
            architecture: earth2studio.models.px.StormScopeMRMS
            load_args:
                model_name: 6km_60min_natten_cos_zenith_input_mrms_eoe
                conditioning_data_source: {_target_: earth2studio.data.GOES, ...}
            ic_source: {_target_: earth2studio.data.MRMS}
            ic_grid: {_target_: src.stormscope.mrms_grid}
            conditioning_grid: {_target_: src.stormscope.goes_grid, ...}
        max_dist_km: 30.0     # optional; passed to build_*_interpolator

    The pipeline caches the two IC sources during :meth:`setup` and
    uses them in :meth:`run_item`, ignoring the ``data_source`` argument
    that ``main.py`` wires in (which is a single-source abstraction).

    Ensembles
    ---------
    StormScope exposes ensemble diversity through the diffusion sampler's
    stochasticity.  We honor :attr:`cfg.ensemble_size` the same way
    other pipelines do — one :class:`WorkItem` per member, seeded
    deterministically.  The model's ``batch`` dimension is used
    internally (required by ``call_with_conditioning``) but is squeezed
    out of the yielded tensor so :meth:`Pipeline._inject_ensemble` can
    prepend a proper ``ensemble`` axis at write time.

    Predownload
    -----------
    :meth:`predownload_stores` declares one zarr per IC source
    (``data_goes.zarr``, ``data_mrms.zarr``), each wrapped in a
    :class:`~src.regrid.RegriddedSource` that resamples the raw source
    onto the model's HRRR sub-region at write time.  The resulting
    store has ``(time, y, x)`` dims whose ``y``/``x`` values match the
    model's native grid exactly — so at inference time
    :meth:`StormScopeBase.prep_input` detects the match and skips its
    live interpolation.

    :meth:`setup` auto-detects the predownloaded stores under
    ``<output.path>/data_{goes,mrms}.zarr`` and uses
    :class:`~src.data.PredownloadedSource` in their place.  Users can
    skip predownload entirely with per-model BYO overrides (see
    :class:`StormScopePipeline` docstring) or by pointing to raw live
    sources and running with the model's live interpolators.
    """

    needs_data_source = False
    """StormScope resolves its own IC sources per-model from
    ``cfg.model.{goes,mrms}.ic_source`` — ``main.py`` should not
    instantiate a primary data source on our behalf."""

    model_goes: Any
    model_mrms: Any
    nsteps: int
    _goes_ic_source: Any
    _mrms_ic_source: Any
    _goes_ic_coords: CoordSystem
    _mrms_ic_coords: CoordSystem

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        if "model" not in cfg or "goes" not in cfg.model or "mrms" not in cfg.model:
            raise ValueError(
                "StormScopePipeline requires cfg.model.{goes, mrms} sub-blocks. "
                "See cfg/model/stormscope_goes_mrms.yaml for a template."
            )

        self.nsteps = cfg.nsteps
        max_dist_km = cfg.model.get("max_dist_km", None)

        # All ranks participate in model loading for barrier correctness.
        self.model_goes = _load_stormscope_model(cfg.model.goes).to(device)
        self.model_goes.eval()
        self.model_mrms = _load_stormscope_model(cfg.model.mrms).to(device)
        self.model_mrms.eval()

        # Build input + conditioning interpolators for each model.  The grid
        # resolvers return (lats, lons) via Hydra-instantiable helpers —
        # see src/stormscope.py.
        goes_in_lat, goes_in_lon = hydra.utils.instantiate(cfg.model.goes.ic_grid)
        self.model_goes.build_input_interpolator(
            goes_in_lat, goes_in_lon, max_dist_km=max_dist_km
        )
        mrms_in_lat, mrms_in_lon = hydra.utils.instantiate(cfg.model.mrms.ic_grid)
        self.model_mrms.build_input_interpolator(
            mrms_in_lat, mrms_in_lon, max_dist_km=max_dist_km
        )

        if cfg.model.goes.get("conditioning_grid") is not None:
            cgoes_lat, cgoes_lon = hydra.utils.instantiate(cfg.model.goes.conditioning_grid)
            self.model_goes.build_conditioning_interpolator(
                cgoes_lat, cgoes_lon, max_dist_km=max_dist_km
            )
        if cfg.model.mrms.get("conditioning_grid") is not None:
            cmrms_lat, cmrms_lon = hydra.utils.instantiate(cfg.model.mrms.conditioning_grid)
            self.model_mrms.build_conditioning_interpolator(
                cmrms_lat, cmrms_lon, max_dist_km=max_dist_km
            )

        # IC sources: prefer predownloaded, regridded zarrs when present
        # (<output.path>/data_{goes,mrms}.zarr).  Their y/x coords already
        # match self.model_{goes,mrms}.y/x so the models' prep_input skips
        # re-interpolation.  Fall back to the live ic_source otherwise.
        self._goes_ic_source = self._resolve_ic_source(
            cfg, "data_goes", cfg.model.goes.ic_source
        )
        self._mrms_ic_source = self._resolve_ic_source(
            cfg, "data_mrms", cfg.model.mrms.ic_source
        )

        # Conditioning source: swap in the predownloaded, regridded zarr
        # (<output.path>/cond_goes.zarr) when present so the GOES model's
        # internal fetch_conditioning reads locally instead of hitting a
        # live forecast source per step.  The conditioning zarr's y/x
        # match self.model_goes.y/x so prep_input's conditioning regrid
        # is also skipped.  MRMS is intentionally excluded — its
        # conditioning is supplied externally via call_with_conditioning.
        self._maybe_swap_conditioning(cfg, "goes", self.model_goes)

        # Cache input coords for efficiency in run_item.
        self._goes_ic_coords = self.model_goes.input_coords()
        self._mrms_ic_coords = self.model_mrms.input_coords()

        # Spatial reference = HRRR grid, with the unified output variable list
        # so that build_output_coords filters correctly against yielded tensors.
        output_vars = _concat_var_lists(
            list(self._goes_ic_coords["variable"]),
            list(self._mrms_ic_coords["variable"]),
        )
        self._spatial_ref = OrderedDict(
            [
                ("variable", np.array(output_vars)),
                ("y", _as_np(self.model_goes.y)),
                ("x", _as_np(self.model_goes.x)),
            ]
        )

    # ------------------------------------------------------------------
    # Output schema
    # ------------------------------------------------------------------

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        # Derive a step-size from each model's input→output stride.  Require
        # both models to agree so the combined zarr has a single lead_time
        # axis with consistent stride.
        goes_step = _infer_step_delta(self.model_goes)
        mrms_step = _infer_step_delta(self.model_mrms)
        if goes_step != mrms_step:
            raise ValueError(
                f"StormScope GOES and MRMS models must share a forecast stride; "
                f"got {goes_step} and {mrms_step}."
            )

        total: CoordSystem = OrderedDict()
        if ensemble_size > 1:
            total["ensemble"] = np.arange(ensemble_size)
        total["time"] = times
        total["lead_time"] = np.array(
            [goes_step * (i + 1) for i in range(self.nsteps)]
        ).astype("timedelta64[ns]")

        spatial_ref = self.effective_spatial_ref()
        for d in ("y", "x"):
            total[d] = spatial_ref[d]
        return total

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # ``data_source`` is ignored — StormScope uses the two IC sources
        # cached during setup.

        y, y_coords = self._fetch_ic(
            self._goes_ic_source, self._goes_ic_coords, item, device
        )
        y_m, y_m_coords = self._fetch_ic(
            self._mrms_ic_source, self._mrms_ic_coords, item, device
        )

        # Seed torch's global RNG so the diffusion sampler's per-member
        # noise is deterministic across runs.  StormScope models don't
        # expose a set_rng hook; torch.manual_seed is the supported path.
        torch.manual_seed(item.seed)

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        rank = DistributedManager().rank

        for step_idx in tqdm(
            range(self.nsteps),
            desc=f"IC {item.time}",
            position=1,
            leave=False,
            disable=rank != 0,
        ):
            pred_goes, pred_goes_coords = self.model_goes(y, y_coords)
            pred_mrms, pred_mrms_coords = self.model_mrms.call_with_conditioning(
                y_m,
                y_m_coords,
                conditioning=y,
                conditioning_coords=y_coords,
            )

            # Concat predictions along the variable axis.  Both tensors
            # are on the HRRR grid after prep_input, so spatial dims
            # line up; cat_coords handles coord-system merging.
            combined, combined_coords = cat_coords(
                (pred_goes, pred_mrms),
                (pred_goes_coords, pred_mrms_coords),
                "variable",
            )

            # Squeeze the model-internal batch dim (batch=1) so the yield
            # matches Pipeline.run's (time, lead_time, variable, y, x)
            # contract.  Pipeline._inject_ensemble re-introduces the
            # ensemble axis at write time.
            combined, combined_coords = _squeeze_batch(combined, combined_coords)

            yield combined, combined_coords

            # Prepare next-step inputs.  next_input handles sliding
            # window (10min variants) or passes pred through (60min).
            y, y_coords = self.model_goes.next_input(
                pred_goes, pred_goes_coords, y, y_coords
            )
            y_m, y_m_coords = self.model_mrms.next_input(
                pred_mrms, pred_mrms_coords, y_m, y_m_coords
            )

    # ------------------------------------------------------------------
    # Predownload
    # ------------------------------------------------------------------

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare one regridded IC + verification store per StormScope model.

        Each store holds data pre-resampled onto the model's HRRR
        sub-region via a :class:`~src.regrid.NearestNeighborRegridder`,
        so the inference-time ``prep_input`` grid-match check passes
        and the live interpolator never runs.

        The stored times are the union of:

        * ``{ic + input_lead}`` for every IC-input lead time (typically
          just ``ic + 0h`` for single-step variants; the full sliding
          window for 10-min variants).
        * ``{ic + k·stride}`` for every forecast step ``k = 1..nsteps``.
          For nowcasting these are real observations at future times —
          the same data :mod:`src.scoring` will load as verification.

        Storing both classes in the same zarr keeps the predownload /
        scoring paths aligned (both read from ``data_{goes,mrms}.zarr``)
        without needing a separate ``verification.zarr``.

        BYO overrides are honored by skipping the corresponding store:
        pass ``cfg.model.{goes,mrms}.ic_byo=true`` (or set a non-default
        ``_target_`` on the ``ic_source`` block that already points
        at a local store) to take full control of that source.

        Additionally, for the GOES model the conditioning data source
        (typically ``GFS_FX``) is predownloaded into ``cond_goes.zarr``
        — the model's ``fetch_conditioning`` path is invoked once per
        forecast step, so a local cache meaningfully reduces S3 traffic
        for multi-IC campaigns.  The MRMS model's conditioning is
        provided externally in this pipeline (GOES state → MRMS via
        ``call_with_conditioning``), so its internal conditioning
        source is bypassed and not predownloaded.
        """
        # CPU-load the models so we can read their cropped HRRR y/x/lat/lon
        # buffers without materializing weights to a device.  The load is
        # the same `_load_stormscope_model` call that ``setup()`` uses, so
        # no new weights are downloaded at predownload time.
        from .work import build_work_items

        stores: list[PredownloadStore] = []
        all_items = build_work_items(cfg)
        unique_ic_times = sorted({i.time for i in all_items})

        for side, model_cfg_key in (("goes", "goes"), ("mrms", "mrms")):
            model_cfg = cfg.model[model_cfg_key]
            if model_cfg.get("ic_byo", False):
                logger.info(
                    f"StormScope {side}: ic_byo=true — skipping predownload"
                )
                continue

            model = _load_stormscope_model(model_cfg)
            ic_coords = model.input_coords()
            stride = _infer_step_delta(model)

            max_dist_km = cfg.model.get("max_dist_km", None)
            src_lat, src_lon = hydra.utils.instantiate(model_cfg.ic_grid)
            regridder = NearestNeighborRegridder(
                source_lats=src_lat,
                source_lons=src_lon,
                target_lats=model.latitudes,
                target_lons=model.longitudes,
                target_y=_as_np(model.y),
                target_x=_as_np(model.x),
                max_dist_km=(
                    float(max_dist_km) if max_dist_km is not None else 12.0
                ),
            )

            raw_source = hydra.utils.instantiate(model_cfg.ic_source)
            wrapped = RegriddedSource(raw_source, regridder)

            # Offsets covering both the IC input window and every forecast
            # valid time.  Cast to ns for datetime64 arithmetic.
            input_offsets = [
                np.timedelta64(lt, "ns") for lt in ic_coords["lead_time"]
            ]
            forecast_offsets = [stride * (k + 1) for k in range(cfg.nsteps)]
            fetch_times = sorted(
                {t + off for t in unique_ic_times for off in input_offsets + forecast_offsets}
            )

            spatial_ref: CoordSystem = OrderedDict(
                [
                    ("y", _as_np(model.y)),
                    ("x", _as_np(model.x)),
                ]
            )

            stores.append(
                PredownloadStore(
                    name=f"data_{side}",
                    source=wrapped,
                    times=fetch_times,
                    variables=[str(v) for v in ic_coords["variable"]],
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

            # Conditioning predownload — only for the GOES model.  In our
            # coupling loop MRMS is always called via call_with_conditioning,
            # so its internal conditioning_data_source is bypassed.  Users
            # can opt out with cfg.model.<side>.cond_byo=true.
            if side != "goes" or model_cfg.get("cond_byo", False):
                continue

            cond_store = self._build_conditioning_store(
                model=model,
                model_cfg=model_cfg,
                side=side,
                unique_ic_times=unique_ic_times,
                input_offsets=input_offsets,
                forecast_offsets=forecast_offsets,
                spatial_ref=spatial_ref,
                max_dist_km=max_dist_km,
            )
            if cond_store is not None:
                stores.append(cond_store)

        return stores

    def _build_conditioning_store(
        self,
        *,
        model: Any,
        model_cfg: DictConfig,
        side: str,
        unique_ic_times: list[np.datetime64],
        input_offsets: list[np.timedelta64],
        forecast_offsets: list[np.timedelta64],
        spatial_ref: CoordSystem,
        max_dist_km: float | None,
    ) -> PredownloadStore | None:
        """Build a predownload store for a StormScope model's conditioning source.

        Returns ``None`` when the model doesn't advertise conditioning
        variables or when the campaign config lacks a ``conditioning_grid``
        resolver (can't regrid without knowing the source grid).

        Supports both :class:`~earth2studio.data.base.ForecastSource`
        conditioning (e.g. ``GFS_FX``) — via
        :class:`~src.data.ValidTimeForecastAdapter` — and plain
        :class:`~earth2studio.data.base.DataSource` conditioning
        (e.g. ``ARCO`` ERA5) — forwarded directly.  Source type is
        detected by inspecting ``__call__``'s signature for a
        ``lead_time`` parameter, matching
        :func:`earth2studio.data.fetch_data`.

        The optional ``cfg.model.<side>.conditioning_cadence`` rounds
        each ``(ic_time + offset)`` to the coarser native resolution of
        the source (e.g. ``"1h"`` for hourly GFS / ERA5) and dedupes,
        cutting redundant 10-minute fetches from campaigns that use a
        sub-hour-stride model.
        """
        cond_vars: list[str] = [
            str(v) for v in (model.conditioning_variables or [])
        ]
        if not cond_vars:
            return None
        if model_cfg.get("conditioning_grid") is None:
            logger.info(
                f"StormScope {side}: conditioning_grid not configured — "
                "skipping conditioning predownload."
            )
            return None
        cond_source_cfg = model_cfg.get("load_args", {}).get(
            "conditioning_data_source"
        )
        if cond_source_cfg is None:
            logger.info(
                f"StormScope {side}: no conditioning_data_source in load_args — "
                "skipping conditioning predownload."
            )
            return None

        cond_src_lat, cond_src_lon = hydra.utils.instantiate(
            model_cfg.conditioning_grid
        )
        cond_regridder = NearestNeighborRegridder(
            source_lats=cond_src_lat,
            source_lons=cond_src_lon,
            target_lats=model.latitudes,
            target_lons=model.longitudes,
            target_y=_as_np(model.y),
            target_x=_as_np(model.x),
            max_dist_km=(
                float(max_dist_km) if max_dist_km is not None else 26.0
            ),
        )

        # Cadence rounding: optional per-model knob that collapses
        # sub-hour-stride requests down to the source's native
        # resolution.  Example: a 10-min StormScope variant pulling
        # hourly GFS conditioning — 6 requests per hour collapse to 1.
        cadence = model_cfg.get("conditioning_cadence")
        cadence_ns: np.timedelta64 | None = None
        if cadence is not None:
            cadence_ns = (
                pd.Timedelta(cadence).to_timedelta64().astype("timedelta64[ns]")
            )
            logger.info(
                f"StormScope {side}: conditioning_cadence={cadence} — "
                f"rounding fetch times to the nearest {cadence} boundary."
            )

        def _round_to_cadence(t: np.datetime64) -> np.datetime64:
            if cadence_ns is None:
                return t
            c = int(cadence_ns.astype("int64"))
            t_int = int(np.datetime64(t, "ns").astype("int64"))
            return np.datetime64(((t_int + c // 2) // c) * c, "ns")

        # Lookup covers every valid time the GOES model will request at
        # inference — rounded to cadence when configured.  First-IC-wins
        # when two ICs share a (rounded) valid time.
        lookup: dict[np.datetime64, tuple[np.datetime64, np.timedelta64]] = {}
        for ic_time in unique_ic_times:
            ic_ns = np.datetime64(ic_time, "ns")
            for off in list(input_offsets) + list(forecast_offsets):
                vt_raw = ic_ns + np.timedelta64(off, "ns")
                vt = _round_to_cadence(vt_raw)
                if vt not in lookup:
                    lookup[vt] = (ic_ns, vt - ic_ns)

        # Detect source type — ForecastSource needs the (init, lead)
        # adapter; DataSource goes straight in.
        raw_source = hydra.utils.instantiate(cond_source_cfg)
        sig = inspect.signature(raw_source.__call__)
        if "lead_time" in sig.parameters:
            source_for_predownload: Any = ValidTimeForecastAdapter(
                raw_source, lookup
            )
            logger.info(
                f"StormScope {side}: wrapping ForecastSource conditioning "
                "with ValidTimeForecastAdapter for predownload."
            )
        else:
            source_for_predownload = raw_source
            logger.info(
                f"StormScope {side}: conditioning source is a DataSource — "
                "no valid-time adapter needed."
            )

        wrapped = RegriddedSource(source_for_predownload, cond_regridder)

        return PredownloadStore(
            name=f"cond_{side}",
            source=wrapped,
            times=sorted(lookup),
            variables=cond_vars,
            spatial_ref=spatial_ref,
            role="conditioning",
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def verification_source(self, cfg: DictConfig) -> DataSource | None:
        """Return a :class:`~src.data.CompositeSource` over the per-model IC zarrs.

        For nowcasting the verification data is just the raw observations
        at the forecast valid times — i.e. the same GOES/MRMS data that
        predownload already wrote (post-regrid) to ``data_goes.zarr`` /
        ``data_mrms.zarr``.  Both stores are indexed by ``time``, ``y``,
        ``x``, so the composite can transparently dispatch variable
        requests across them for :mod:`src.scoring`.

        Falls back to ``None`` (score.py's default lookup) when neither
        predownloaded store exists — the user is then expected to
        provide ``cfg.verification_source``.
        """
        stores: dict[str, str] = {}
        for side in ("goes", "mrms"):
            path = os.path.join(cfg.output.path, f"data_{side}.zarr")
            if os.path.exists(path):
                stores[f"data_{side}"] = path

        if not stores:
            logger.info(
                "StormScope: no data_{goes,mrms}.zarr found — deferring "
                "to score.py's default verification lookup."
            )
            return None

        logger.info(
            f"StormScope: using CompositeSource for verification "
            f"({', '.join(stores.values())})"
        )
        return CompositeSource.from_predownloaded_stores(stores)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_ic_source(
        self,
        cfg: DictConfig,
        store_name: str,
        live_source_cfg: DictConfig,
    ) -> DataSource:
        """Return a predownloaded source if present, else the live one.

        Analogous to ``main.py::_resolve_ic_source`` but specialised
        per-source for the multi-source StormScope case.  Checks for
        ``<cfg.output.path>/<store_name>.zarr`` and wraps it in a
        :class:`~src.data.PredownloadedSource` when found.
        """
        cache_path = os.path.join(cfg.output.path, f"{store_name}.zarr")
        if os.path.exists(cache_path):
            logger.info(
                f"StormScope: using predownloaded IC store {cache_path}"
            )
            return PredownloadedSource(cache_path)
        logger.info(
            f"StormScope: no cache at {cache_path} — instantiating live "
            f"ic_source for this model"
        )
        return hydra.utils.instantiate(live_source_cfg)

    def _maybe_swap_conditioning(
        self,
        cfg: DictConfig,
        side: str,
        model: Any,
    ) -> None:
        """Swap *model*'s conditioning source for a predownloaded zarr.

        Looks for ``<cfg.output.path>/cond_<side>.zarr`` and assigns a
        :class:`~src.data.PredownloadedSource` reading from it to the
        model's ``conditioning_data_source`` attribute.  When the cache
        is absent this is a no-op — the model retains whatever
        forecast source the campaign config installed at load time
        (typically ``GFS_FX``).

        The predownloaded zarr is valid-time-keyed with ``(y, x)``
        coords matching ``model.y`` / ``model.x``; at inference time
        :meth:`StormScopeBase.fetch_conditioning` converts the model's
        ``(time, lead_time)`` request into a valid-time lookup via
        :func:`earth2studio.data.fetch_data` and
        :meth:`StormScopeBase.prep_input` skips the conditioning
        interpolator on the grid match.
        """
        cache_path = os.path.join(cfg.output.path, f"cond_{side}.zarr")
        if not os.path.exists(cache_path):
            logger.info(
                f"StormScope {side}: no conditioning cache at {cache_path} — "
                f"model will fetch conditioning from its configured source."
            )
            return

        source: Any = PredownloadedSource(cache_path)

        # If the predownload was cadence-deduplicated, the cached zarr only
        # holds values on the coarser cadence.  Wrap with CadenceRoundedSource
        # so the model's finer-grained queries route to the nearest stored
        # value while still receiving DataArrays labeled at the requested
        # valid times (preserving fetch_data's lead_time reconstruction).
        cadence = cfg.model[side].get("conditioning_cadence")
        if cadence is not None:
            source = CadenceRoundedSource(source, cadence)
            logger.info(
                f"StormScope {side}: using predownloaded conditioning store "
                f"{cache_path} with cadence={cadence} query rounding."
            )
        else:
            logger.info(
                f"StormScope {side}: using predownloaded conditioning store "
                f"{cache_path}"
            )
        model.conditioning_data_source = source

    def _fetch_ic(
        self,
        source: DataSource,
        ic_coords: CoordSystem,
        item: WorkItem,
        device: torch.device,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch IC data for one of the two StormScope models.

        The returned ``(x, coords)`` has a ``batch`` dim prepended —
        required by :meth:`StormScopeMRMS.call_with_conditioning`.
        """
        x, coords = fetch_data(
            source=source,
            time=[item.time],
            variable=ic_coords["variable"],
            lead_time=ic_coords["lead_time"],
            device=device,
        )
        x = x.unsqueeze(0)
        coords = OrderedDict([("batch", np.arange(1))] + list(coords.items()))
        return x, coords


# ------------------------------------------------------------------
# StormScope helpers (module-level for reuse in tests and config)
# ------------------------------------------------------------------


def _load_stormscope_model(model_cfg: DictConfig) -> Any:
    """Load a StormScope model (GOES or MRMS) from its config sub-block.

    Uses the same pattern as ``load_prognostic`` in ``src.models``:
    resolve the class via ``architecture``, optionally load a package
    from ``package_path``, then call ``cls.load_model(package, **load_args)``
    — the StormScope API requires ``conditioning_data_source`` and
    ``model_name`` under ``load_args``.
    """
    cls = hydra.utils.get_class(model_cfg.architecture)

    if model_cfg.get("package_path"):
        from earth2studio.models.auto import Package

        pkg = Package(model_cfg.package_path)
    else:
        from .distributed import run_on_rank0_first

        pkg = run_on_rank0_first(cls.load_default_package)

    load_args = dict(model_cfg.get("load_args", {}))
    # Resolve nested _target_ entries (e.g. conditioning_data_source).
    resolved: dict[str, Any] = {}
    for k, v in load_args.items():
        if isinstance(v, (DictConfig, dict)) and "_target_" in v:
            resolved[k] = hydra.utils.instantiate(v)
        else:
            resolved[k] = v

    model = cls.load_model(package=pkg, **resolved)
    logger.success(
        f"Loaded StormScope model: {cls.__name__} "
        f"(model_name={resolved.get('model_name', 'default')})"
    )
    return model


def _infer_step_delta(model: Any) -> np.timedelta64:
    """Return the forecast stride (single model step) as a timedelta64[ns].

    Computed as ``output_coords.lead_time[-1] - input_coords.lead_time[-1]``,
    which gives the per-step advance for both sliding-window (10-min)
    and single-step (60-min) StormScope variants.
    """
    ic = model.input_coords()
    out = model.output_coords(ic)
    delta = out["lead_time"][-1] - ic["lead_time"][-1]
    return np.timedelta64(delta, "ns")


def _concat_var_lists(a: list[str], b: list[str]) -> list[str]:
    """Append *b*'s entries to *a*, skipping duplicates.  Preserves order."""
    seen = set(a)
    out = list(a)
    for v in b:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _squeeze_batch(
    x: torch.Tensor, coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Remove the ``batch`` dim from a (tensor, coords) pair if present."""
    if "batch" not in coords:
        return x, coords
    batch_axis = list(coords.keys()).index("batch")
    x = x.squeeze(batch_axis)
    coords = OrderedDict((k, v) for k, v in coords.items() if k != "batch")
    return x, coords


def _as_np(arr: Any) -> np.ndarray:
    """Coerce a torch tensor / array-like into a numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ======================================================================
# Pipeline registry
# ======================================================================

PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    "forecast": ForecastPipeline,
    "diagnostic": DiagnosticPipeline,
    "dlesym": DLESyMPipeline,
    "stormscope": StormScopePipeline,
}


def build_pipeline(cfg: DictConfig) -> Pipeline:
    """Instantiate a pipeline from the Hydra config.

    Looks up ``cfg.pipeline`` (default ``"forecast"``) in the built-in
    registry.  If the value is a fully-qualified class name (contains a dot),
    it is resolved via ``hydra.utils.get_class`` instead, allowing custom
    pipelines without modifying this module.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    Pipeline
        An uninitialized pipeline instance (call :meth:`Pipeline.setup`
        before use).
    """
    name = cfg.get("pipeline", "forecast")

    if "." in name:
        cls = hydra.utils.get_class(name)
        if not (isinstance(cls, type) and issubclass(cls, Pipeline)):
            raise TypeError(f"Custom pipeline '{name}' must be a subclass of Pipeline.")
        return cls()

    if name not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {available}. "
            "Use a fully-qualified class name for custom pipelines."
        )

    return PIPELINE_REGISTRY[name]()
