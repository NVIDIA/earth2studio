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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from .models import load_diagnostics, load_prognostic
from .output import (
    OutputManager,
    build_diagnostic_coords,
    build_forecast_coords,
    build_output_coords,
)
from .work import WorkItem


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
    filter, injecting the ensemble dimension at write time, and calling
    :meth:`OutputManager.write` — is handled by the shared :meth:`run` method
    and does **not** need to be reimplemented.

    Subclassing guide
    -----------------
    Implement these three abstract methods:

    * :meth:`setup` — called once before inference begins.
    * :meth:`build_total_coords` — called once to create the zarr store schema.
    * :meth:`run_item` — called once per work item; yield output tensors.

    The base class provides :attr:`_spatial_ref` which subclasses must set
    during :meth:`setup`.  It is used by :meth:`run` to build the output
    coordinate filter (variable + lat/lon sub-selection).

    Example
    -------
    A minimal custom pipeline::

        class MyPipeline(Pipeline):
            def setup(self, cfg, device):
                self.model = load_my_model(cfg).to(device)
                self._spatial_ref = self.model.output_coords(
                    self.model.input_coords()
                )

            def build_total_coords(self, times, ensemble_size):
                total = OrderedDict()
                if ensemble_size > 1:
                    total["ensemble"] = np.arange(ensemble_size)
                total["time"] = times
                total["lead_time"] = np.array([np.timedelta64(0, "ns")])
                for dim in ("lat", "lon"):
                    if dim in self._spatial_ref:
                        total[dim] = self._spatial_ref[dim]
                return total

            def run_item(self, item, data_source, device):
                x, coords = fetch_data(...)
                x, coords = self.model(x, coords)
                yield x, coords
    """

    _spatial_ref: CoordSystem
    """Reference coordinate system whose spatial dimensions (lat/lon) define
    the output grid.  Must be set by :meth:`setup`."""

    @abstractmethod
    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load models and prepare pipeline state.

        Called exactly once before any inference begins.  Implementations
        should:

        * Load and ``.to(device)`` all models.
        * Cache any coordinate metadata needed by :meth:`run_item`
          (e.g. ``input_coords()`` lookups).
        * Set ``self._spatial_ref`` to the coordinate system whose
          ``lat`` / ``lon`` entries define the output grid.

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

        Typical dimensions: ``[ensemble], time, lead_time, lat, lon``.
        The ``ensemble`` dimension should be included only when
        ``ensemble_size > 1``.

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
        3. ``yield`` one or more ``(tensor, coords)`` pairs.

        Each yielded pair is automatically filtered to the requested output
        variables and spatial grid, has the ensemble dimension injected if
        needed, and is written to the output store.  Implementations do
        **not** need to handle any of that.

        For forecast pipelines this yields once per lead-time step.  For
        diagnostic pipelines this typically yields once.  Custom pipelines
        may yield any number of times.

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
    # Shared machinery — not overridden by subclasses
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run(
        self,
        work_items: list[WorkItem],
        data_source: DataSource,
        output_mgr: OutputManager,
        output_variables: list[str],
        device: torch.device,
    ) -> None:
        """Iterate work items, filter outputs, and write to the output store.

        This is the shared outer loop.  It calls :meth:`run_item` for each
        work item, applies the output coordinate filter (variable + spatial
        sub-selection), injects the ensemble dimension when present, and
        writes each chunk via the :class:`OutputManager`.

        Parameters
        ----------
        work_items : list[WorkItem]
            Work items assigned to this rank.
        data_source : DataSource
            Source for fetching input data.
        output_mgr : OutputManager
            Context-managed output handler (store already validated).
        output_variables : list[str]
            Variable names to sub-select before writing.
        device : torch.device
            Target device for inference.
        """
        if not work_items:
            logger.warning("No work items for this rank — skipping inference.")
            return

        output_coords = build_output_coords(self._spatial_ref, output_variables)
        has_ensemble = "ensemble" in output_mgr.io.coords

        for item in tqdm(work_items, desc="Work items", position=0):
            for x_step, coords_step in self.run_item(item, data_source, device):
                x_out, coords_out = map_coords(x_step, coords_step, output_coords)

                if has_ensemble:
                    x_out = x_out.unsqueeze(0)
                    coords_out = CoordSystem(
                        {"ensemble": np.array([item.ensemble_id])}
                        | dict(coords_out)
                    )

                output_mgr.write(x_out, coords_out)

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
        self._dx_input_coords = {
            id(dx): dx.input_coords() for dx in self.diagnostics
        }

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        return build_forecast_coords(
            self.prognostic, times, self.nsteps, ensemble_size
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
        x, coords = map_coords(x, coords, self._prognostic_ic)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        model_iter = self.prognostic.create_iterator(x, coords)

        for step, (x_step, coords_step) in enumerate(
            tqdm(
                model_iter,
                total=self.nsteps + 1,
                desc=f"IC {item.time}",
                position=1,
                leave=False,
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

        self._dx_input_coords = {
            id(dx): dx.input_coords() for dx in self.diagnostics
        }

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
        return build_diagnostic_coords(self.diagnostics, times, ensemble_size)

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


# ======================================================================
# Pipeline registry
# ======================================================================

PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    "forecast": ForecastPipeline,
    "diagnostic": DiagnosticPipeline,
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
            raise TypeError(
                f"Custom pipeline '{name}' must be a subclass of Pipeline."
            )
        return cls()

    if name not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {available}. "
            "Use a fully-qualified class name for custom pipelines."
        )

    return PIPELINE_REGISTRY[name]()
