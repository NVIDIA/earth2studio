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
from types import TracebackType
from typing import Any

import numpy as np
import torch
import zarr
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.io import ZarrBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem, handshake_coords, split_coords

from .distributed import run_on_rank0_first


class OutputManager:
    """Distributed-safe lifecycle manager for zarr forecast output.

    Handles the full output lifecycle: store creation (rank 0), validation
    (other ranks), writes (optionally threaded), and metadata consolidation.
    Use as a context manager to guarantee cleanup.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config — expects ``output`` section with ``path``, ``variables``,
        ``overwrite``, ``chunks``, and optionally ``thread_writers``.
    prognostic : PrognosticModel
        Used to derive the coordinate system for the output arrays.
    times : np.ndarray
        All initial-condition times that will appear in the output.
    nsteps : int
        Number of forecast steps.
    ensemble_size : int
        Total number of ensemble members across all ranks.
    """

    def __init__(
        self,
        cfg: DictConfig,
        prognostic: PrognosticModel,
        times: np.ndarray,
        nsteps: int,
        ensemble_size: int = 1,
    ) -> None:
        self._cfg = cfg
        self._dist = DistributedManager()

        output_cfg = cfg.output
        self._path = os.path.join(output_cfg.path, "forecast.zarr")
        self._overwrite = output_cfg.get("overwrite", False)
        self._variables = list(output_cfg.variables)
        self._thread_io = output_cfg.get("thread_writers", 0)

        self._total_coords = self._build_total_coords(
            prognostic, times, nsteps, ensemble_size
        )
        self._output_coords = self._build_output_coords(output_cfg)

        self._io: ZarrBackend | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[Any]] = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> OutputManager:
        self._io = run_on_rank0_first(self._open_store)
        if self._thread_io > 0:
            self._executor = ThreadPoolExecutor(max_workers=self._thread_io)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._executor is not None:
            for f in self._futures:
                f.result()
            self._executor.shutdown(wait=True)
            self._futures.clear()

        if self._dist.rank == 0 and self._io is not None:
            logger.info("Consolidating zarr metadata")
            zarr.consolidate_metadata(self._io.store)

        if self._dist.distributed:
            torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def io(self) -> ZarrBackend:
        """The underlying IO backend (available after ``__enter__``)."""
        if self._io is None:
            raise RuntimeError("OutputManager must be used as a context manager.")
        return self._io

    @property
    def output_coords(self) -> CoordSystem:
        """Coordinate system used for sub-selecting model output before writing."""
        return self._output_coords

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
            future = self._executor.submit(
                self.io.write, arrays, coord_sets, var_names
            )
            self._futures.append(future)
        else:
            self.io.write(arrays, coord_sets, var_names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_total_coords(
        self,
        prognostic: PrognosticModel,
        times: np.ndarray,
        nsteps: int,
        ensemble_size: int,
    ) -> CoordSystem:
        """Derive the full coordinate system for the output zarr.

        Follows the same lead-time derivation used in
        ``earth2studio.run.deterministic``: the model's single-step output
        lead-times are scaled by step index to produce the full forecast
        timeline.
        """
        input_c = prognostic.input_coords()
        output_c = prognostic.output_coords(input_c)

        total: CoordSystem = OrderedDict()
        if ensemble_size > 1:
            total["ensemble"] = np.arange(ensemble_size)
        total["time"] = times

        step_lead = output_c["lead_time"]
        total["lead_time"] = (
            np.asarray([step_lead * i for i in range(nsteps + 1)])
            .flatten()
            .astype("timedelta64[ns]")
        )

        for dim in ("lat", "lon"):
            if dim in output_c:
                total[dim] = output_c[dim]

        return total

    def _build_output_coords(self, output_cfg: DictConfig) -> CoordSystem:
        """Build the coordinate filter applied before each write."""
        oc: CoordSystem = OrderedDict()
        oc["variable"] = np.array(self._variables)

        for dim in ("lat", "lon"):
            if dim in self._total_coords:
                oc[dim] = self._total_coords[dim]

        return oc

    def _open_store(self) -> ZarrBackend:
        """Create or open the zarr store (called inside rank-ordered context)."""
        is_rank0 = self._dist.rank == 0 if self._dist.distributed else True
        output_cfg = self._cfg.output

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

        if is_rank0 and os.path.exists(self._path):
            if self._overwrite:
                logger.warning(f"Overwriting existing store: {self._path}")
                shutil.rmtree(self._path)
            else:
                raise FileExistsError(
                    f"{self._path} already exists. Set output.overwrite=true to replace."
                )

        chunks = dict(output_cfg.get("chunks", {"time": 1, "lead_time": 1}))
        if "ensemble" not in chunks and "ensemble" in self._total_coords:
            chunks["ensemble"] = 1

        io = ZarrBackend(
            file_name=self._path,
            chunks=chunks,
            backend_kwargs={"overwrite": False},
        )

        if is_rank0:
            write_coords = self._total_coords.copy()
            variables = np.array(self._variables)
            io.add_array(write_coords, variables)
            logger.info(f"Created forecast store: {self._path}")
        else:
            for v in self._variables:
                if v not in io:
                    raise ValueError(
                        f"Variable '{v}' missing from store — rank 0 initialization failed?"
                    )
            for dim in self._total_coords:
                handshake_coords(io.coords, self._total_coords, required_dim=dim)
            logger.debug(f"Rank {self._dist.rank} validated store: {self._path}")

        return io
