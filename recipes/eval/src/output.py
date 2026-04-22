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

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
import zarr
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.io import ZarrBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem, handshake_coords, split_coords

from .distributed import run_on_rank0_first

_NON_SPATIAL_DIMS = frozenset({"batch", "time", "lead_time", "variable", "ensemble"})


def _spatial_dims(coords: CoordSystem) -> list[str]:
    """Return dimension names from *coords* that are spatial (not structural).

    Structural dimensions (batch, time, lead_time, variable, ensemble) are
    excluded; everything else is treated as a spatial dimension whose values
    should be carried through to output stores.
    """
    return [d for d in coords if d not in _NON_SPATIAL_DIMS]


def build_output_coords(
    spatial_ref: CoordSystem,
    output_variables: list[str],
) -> CoordSystem:
    """Build the coordinate filter used to sub-select model output before writing.

    Parameters
    ----------
    spatial_ref : CoordSystem
        Reference coordinate system whose spatial entries define the output
        grid.  Typically the output coords of the prognostic or diagnostic
        model.
    output_variables : list[str]
        Variable names to extract from the model state at each step.

    Returns
    -------
    CoordSystem
        Filter with ``variable`` and any spatial dimension keys.
    """
    oc: CoordSystem = OrderedDict()
    oc["variable"] = np.array(output_variables)
    for dim in _spatial_dims(spatial_ref):
        oc[dim] = spatial_ref[dim]
    return oc


def sentinel_path(cfg: DictConfig) -> Path:
    """Return the path of the pre-download sentinel file for this eval run.

    The sentinel is written by ``predownload.py`` on successful completion and
    checked by ``main.py`` when ``require_predownload`` is true.  It lives
    alongside the forecast zarr so it is trivially co-located with the run
    outputs and visible to every node on a shared filesystem.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an ``output.path`` key.

    Returns
    -------
    Path
        ``<output.path>/.predownload.done``
    """
    return Path(cfg.output.path) / ".predownload.done"


def build_forecast_coords(
    prognostic: PrognosticModel,
    times: np.ndarray,
    nsteps: int,
    ensemble_size: int = 1,
    spatial_ref: CoordSystem | None = None,
) -> CoordSystem:
    """Build the full coordinate system for a standard prognostic forecast.

    Derives lead-time coordinates from the model's single-step output (the
    same convention used in ``earth2studio.run.deterministic``), then
    combines them with time, ensemble, and spatial dimensions.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model whose output coordinates determine lead-time stride and
        spatial dimensions.
    times : np.ndarray
        All initial-condition times that will appear in the output.
    nsteps : int
        Number of forecast steps.
    ensemble_size : int
        Total number of ensemble members.  When 1 the ensemble dimension
        is omitted.
    spatial_ref : CoordSystem | None
        If provided, the spatial dims of the output store are taken from
        this coord system instead of the model's output coords.  Used
        when an output regridder is active so that the zarr schema
        reflects the *regridded* grid rather than the model's native one.

    Returns
    -------
    CoordSystem
        Full coordinate system suitable for passing to
        :meth:`OutputManager.validate_output_store`.
    """
    input_c = prognostic.input_coords()
    output_c = prognostic.output_coords(input_c)

    total: CoordSystem = OrderedDict()
    if ensemble_size > 1:
        total["ensemble"] = np.arange(ensemble_size)
    total["time"] = times

    zero = np.array([np.timedelta64(0, "ns")])
    step_lead = output_c["lead_time"]
    step_stride = step_lead[-1]
    total["lead_time"] = np.concatenate(
        [
            zero,
            np.asarray([step_lead + step_stride * i for i in range(nsteps)])
            .flatten()
            .astype("timedelta64[ns]"),
        ]
    )

    ref = spatial_ref if spatial_ref is not None else output_c
    for dim in _spatial_dims(ref):
        total[dim] = ref[dim]

    return total


def build_diagnostic_coords(
    diagnostics: list[DiagnosticModel],
    times: np.ndarray,
    ensemble_size: int = 1,
    spatial_ref: CoordSystem | None = None,
) -> CoordSystem:
    """Build the full coordinate system for a diagnostic-only pipeline.

    Uses the first diagnostic model to determine the output spatial grid.
    The time dimension is a single ``lead_time=0`` slice per initial
    condition (no forecast rollout).

    Parameters
    ----------
    diagnostics : list[DiagnosticModel]
        Diagnostic models that will be run.  The first model's output
        coordinates define the spatial grid.
    times : np.ndarray
        All initial-condition times that will appear in the output.
    ensemble_size : int
        Total number of ensemble members.  When 1 the ensemble dimension
        is omitted.
    spatial_ref : CoordSystem | None
        If provided, the spatial dims of the output store are taken from
        this coord system instead of the first diagnostic's output coords.

    Returns
    -------
    CoordSystem
        Full coordinate system suitable for passing to
        :meth:`OutputManager.validate_output_store`.
    """
    if not diagnostics:
        raise ValueError("At least one diagnostic model is required.")

    dx = diagnostics[0]
    ref = spatial_ref if spatial_ref is not None else dx.output_coords(dx.input_coords())

    total: CoordSystem = OrderedDict()
    if ensemble_size > 1:
        total["ensemble"] = np.arange(ensemble_size)
    total["time"] = times
    total["lead_time"] = np.array([np.timedelta64(0, "ns")])

    for dim in _spatial_dims(ref):
        total[dim] = ref[dim]

    return total


def build_predownload_coords(
    spatial_ref: CoordSystem,
    times: np.ndarray,
) -> CoordSystem:
    """Build coords for a predownload zarr store: ``(time, <spatial...>)``.

    The returned coordinate system has a ``time`` dimension followed by
    whatever spatial dimensions are present in *spatial_ref*.  Variables
    are not included — they are passed separately as array names to
    :meth:`OutputManager.validate_output_store`.

    Parameters
    ----------
    spatial_ref : CoordSystem
        Reference coordinate system whose spatial entries define the grid.
    times : np.ndarray
        All valid times to be stored (IC-adjusted times, verification
        times, or the union of both).

    Returns
    -------
    CoordSystem
        ``(time, <spatial...>)`` coordinate system.
    """
    coords: CoordSystem = OrderedDict()
    coords["time"] = times
    for dim in _spatial_dims(spatial_ref):
        coords[dim] = spatial_ref[dim]
    return coords


def build_score_coords(
    metric: Any,
    times: np.ndarray,
    input_coords_template: CoordSystem,
) -> CoordSystem:
    """Build the output coordinate system for a single metric's score arrays.

    Calls ``metric.output_coords()`` on a template coordinate system (which
    should mirror the shape of the tensors that will be passed to the metric)
    to determine which dimensions survive reduction.  A ``time`` axis is
    prepended since scores are computed per initial-condition time.

    Parameters
    ----------
    metric : Metric
        An ``earth2studio.statistics`` metric instance (or any object with an
        ``output_coords`` method).
    times : np.ndarray
        All initial-condition times that will appear in the score store.
    input_coords_template : CoordSystem
        Representative coordinate system matching the tensors that will be
        passed to the metric (e.g. ``{lead_time, variable, lat, lon}``).

    Returns
    -------
    CoordSystem
        Coordinate system for the score arrays: ``(time, <surviving dims>)``.
    """
    score_output = metric.output_coords(input_coords_template)
    total: CoordSystem = OrderedDict()
    total["time"] = times
    for dim, vals in score_output.items():
        if dim != "time":
            total[dim] = vals
    return total


class OutputManager:
    """Distributed-safe lifecycle manager for a zarr store.

    Handles the full output lifecycle: store creation (rank 0), validation
    (other ranks), writes (optionally threaded), and metadata consolidation.
    Use as a context manager to guarantee safe creation and cleanup.

    The constructor performs only basic config validation.  Call
    :meth:`validate_output_store` with a pre-built coordinate system and
    variable list to create or validate the zarr store before writing.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config — expects ``output`` section with ``path``,
        ``chunks``, and optionally ``thread_writers``.
    store_name : str
        Name of the zarr store directory relative to ``output.path``.
        Defaults to ``"forecast.zarr"``.
    overwrite : bool | None
        If provided, overrides ``output.overwrite`` from config.
    resume : bool | None
        If provided, overrides the top-level ``resume`` from config.
    """

    def __init__(
        self,
        cfg: DictConfig,
        store_name: str = "forecast.zarr",
        overwrite: bool | None = None,
        resume: bool | None = None,
    ) -> None:
        output_cfg = cfg.output
        self._dist = DistributedManager()

        self._path = os.path.join(output_cfg.path, store_name)
        self._overwrite = (
            overwrite if overwrite is not None else output_cfg.get("overwrite", False)
        )
        self._resume = resume if resume is not None else cfg.get("resume", False)
        self._thread_io = output_cfg.get("thread_writers", 0)
        self._chunks: dict[str, int] = dict(
            output_cfg.get("chunks", {"time": 1, "lead_time": 1})
        )

        self._total_coords: CoordSystem | None = None
        self._variables: list[str] | None = None
        self._io: ZarrBackend | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[Any]] = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> OutputManager:
        if self._thread_io > 0:
            self._executor = ThreadPoolExecutor(max_workers=self._thread_io)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        write_error: BaseException | None = None
        if self._executor is not None:
            for f in self._futures:
                try:
                    f.result()
                except Exception as e:
                    if write_error is None:
                        write_error = e
                        logger.error(f"Threaded write failed: {e}")
            self._executor.shutdown(wait=True)
            self._futures.clear()

        has_error = exc_type is not None or write_error is not None
        if self._dist.distributed:
            err_flag = torch.tensor(
                [1.0 if has_error else 0.0],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            torch.distributed.all_reduce(err_flag, op=torch.distributed.ReduceOp.MAX)
            any_error = err_flag.item() > 0
        else:
            any_error = has_error

        if any_error:
            logger.warning(
                f"Skipping barrier/consolidation due to error (local={has_error})"
            )
            if write_error is not None and exc_type is None:
                raise write_error
            return

        if self._dist.distributed:
            torch.distributed.barrier()

        if self._dist.rank == 0 and self._io is not None:
            logger.info("Consolidating zarr metadata")
            zarr.consolidate_metadata(self._io.store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_output_store(
        self,
        total_coords: CoordSystem,
        variables: list[str],
    ) -> None:
        """Create or validate the zarr store against the given schema.

        On first call this opens the zarr store using rank-ordered execution
        (rank 0 creates, others validate).  All ranks in a distributed job
        **must** call this method so that the internal barriers are satisfied.

        If the store already exists on disk (and ``overwrite`` is false),
        the existing coordinates are validated against *total_coords*.

        Parameters
        ----------
        total_coords : CoordSystem
            Full coordinate system for the output arrays (e.g. from
            :func:`build_forecast_coords`).
        variables : list[str]
            Variable names to create in the store.
        """
        self._total_coords = total_coords
        self._variables = variables
        self._io = run_on_rank0_first(self._open_store)

    @property
    def io(self) -> ZarrBackend:
        """The underlying IO backend (available after :meth:`validate_output_store`)."""
        if self._io is None:
            raise RuntimeError(
                "Output store not initialized. "
                "Call validate_output_store() before accessing io."
            )
        return self._io

    def write(self, x: torch.Tensor, coords: CoordSystem) -> None:
        """Write a chunk of forecast data, optionally using a thread pool.

        Parameters
        ----------
        x : torch.Tensor
            Data tensor to write.
        coords : CoordSystem
            Coordinates corresponding to *x*.
        """
        arrays, coord_sets, var_names = split_coords(x, coords)
        if self._executor is not None:
            future = self._executor.submit(self.io.write, arrays, coord_sets, var_names)
            self._futures.append(future)
        else:
            self.io.write(arrays, coord_sets, var_names)

    def flush(self) -> None:
        """Wait for all pending threaded writes to complete.

        Called between work items during resume runs to guarantee that all
        zarr writes have landed before a completion marker is written.
        Raises immediately if any pending write failed.
        """
        if self._executor is not None:
            for f in self._futures:
                f.result()
            self._futures.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_store(self) -> ZarrBackend:
        """Create or open the zarr store (called inside rank-ordered context)."""
        if self._total_coords is None or self._variables is None:
            raise ValueError(
                "Total coordinates and variables must be set before opening the store"
            )
        is_rank0 = self._dist.rank == 0 if self._dist.distributed else True

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

        store_exists = os.path.exists(self._path)

        if is_rank0 and store_exists:
            if self._resume:
                logger.info(f"Resuming into existing store: {self._path}")
            elif self._overwrite:
                logger.warning(f"Overwriting existing store: {self._path}")
                shutil.rmtree(self._path)
                store_exists = False
            else:
                raise FileExistsError(
                    f"{self._path} already exists. "
                    "Set output.overwrite=true to replace, "
                    "or resume=true to append."
                )

        chunks = dict(self._chunks)
        if "ensemble" not in chunks and "ensemble" in self._total_coords:
            chunks["ensemble"] = 1

        io = ZarrBackend(
            file_name=self._path,
            chunks=chunks,
            backend_kwargs={"overwrite": False},
        )

        if is_rank0 and not store_exists:
            write_coords = self._total_coords.copy()
            variables = np.array(self._variables)
            io.add_array(write_coords, variables)
            logger.info(f"Created forecast store: {self._path}")
        else:
            for v in self._variables:
                if v not in io:
                    raise ValueError(
                        f"Variable '{v}' missing from store — "
                        "rank 0 initialization failed?"
                    )
            for dim in self._total_coords:
                handshake_coords(io.coords, self._total_coords, required_dim=dim)
            if is_rank0:
                logger.info(f"Validated existing store for resume: {self._path}")
            else:
                logger.debug(f"Rank {self._dist.rank} validated store: {self._path}")

        return io
