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

"""Window-mean forecast pipeline for subseasonal-to-seasonal archives."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from earth2studio.utils.coords import CoordSystem

from ..predownload_utils import infer_step
from ..regrid import LinearRegridder
from ..work import WorkItem
from .base import PredownloadStore
from .forecast import ForecastPipeline


class WindowMeanForecastPipeline(ForecastPipeline):
    """Forecast pipeline that stores time means over fixed lead windows.

    Subseasonal-to-seasonal archives and verification use weekly or
    monthly means rather than individual steps.  This pipeline drives
    the standard prognostic rollout, averages the steps that fall into
    consecutive windows of ``window`` length counted from the initial
    condition, and yields one field per window, interpolated onto a coarser
    regular grid with :class:`~src.regrid.LinearRegridder` when
    ``target_resolution`` asks for one.  Window ``k`` covers the leads in
    ``(k * window, (k + 1) * window]`` and takes its end as its label, so the
    output ``lead_time`` axis reads ``window, 2 * window, ...`` and the
    lead-0 analysis never enters a mean.  The forecast store is otherwise an
    ordinary one with a short lead axis.

    Windows span a fixed length from the initial condition rather than
    calendar months, so every initial condition shares one lead axis.  A
    model that consumes time means itself (FuXi-S2S) takes them from a
    :class:`~src.data.WindowMeanSource` data source, see
    ``cfg/data_source/arco_era5_daily.yaml``.  The pipeline leaves scoring to
    the user: it declares no verification store, and a bring-your-own
    ``verification_source`` would have to serve window means at the window
    ends on the output grid.

    Parameters
    ----------
    window : str
        Window length, parsed by ``pandas.Timedelta`` (``"30D"``, ``"7D"``,
        ``"720h"``).  Must be a whole number of model steps, and ``nsteps``
        must cover a whole number of windows.
    target_resolution : float | None
        Spacing in degrees of the regular latitude/longitude grid to
        interpolate the output onto.  ``None`` keeps the model grid, as
        does a model whose grid is already that coarse.
    """

    def __init__(self, window: str, target_resolution: float | None = None) -> None:
        super().__init__()
        self._window = pd.Timedelta(window).to_timedelta64().astype("timedelta64[ns]")
        if self._window <= np.timedelta64(0, "ns"):
            raise ValueError(f"window must be positive, got {window!r}.")
        self._target_resolution = (
            None if target_resolution is None else float(target_resolution)
        )
        if self._target_resolution is not None and self._target_resolution <= 0:
            raise ValueError(
                f"target_resolution must be positive, got {target_resolution!r}."
            )
        self._leads: np.ndarray = np.array([], dtype="timedelta64[ns]")
        self._steps_per_window = 0

    # ------------------------------------------------------------------
    # Lead schema
    # ------------------------------------------------------------------

    def window_leads(self, step: np.timedelta64, nsteps: int) -> np.ndarray:
        """Lead axis of the window means: the end of every window.

        Parameters
        ----------
        step : np.timedelta64
            Model time step.
        nsteps : int
            Model steps per forecast.

        Raises
        ------
        ValueError
            If the window is not a whole number of model steps, or ``nsteps``
            does not cover a whole number of windows.  A partial window would
            be a mean over fewer steps than its label claims.
        """
        step_ns = int(np.timedelta64(step, "ns").astype("int64"))
        window_ns = int(self._window.astype("int64"))
        per_window, remainder = divmod(window_ns, step_ns) if step_ns > 0 else (0, 1)
        if per_window < 1 or remainder:
            raise ValueError(
                f"window {pd.Timedelta(self._window)} is not a whole number of "
                f"model steps of {pd.Timedelta(step_ns, 'ns')}."
            )
        nwindows, remainder = divmod(int(nsteps), per_window)
        if nwindows < 1 or remainder:
            raise ValueError(
                f"nsteps={nsteps} does not cover a whole number of "
                f"{per_window}-step windows; use a multiple of {per_window}."
            )
        return (np.arange(1, nwindows + 1) * self._window).astype("timedelta64[ns]")

    def _regridder(self, spatial_ref: CoordSystem) -> LinearRegridder | None:
        """Regridder from the model grid onto the target grid, or ``None``."""
        if self._target_resolution is None:
            return None
        if "lat" not in spatial_ref or "lon" not in spatial_ref:
            raise ValueError(
                "target_resolution needs a model on a regular lat/lon grid; "
                f"this model has dims {list(spatial_ref)}."
            )
        return LinearRegridder.to_resolution(
            spatial_ref["lat"], spatial_ref["lon"], self._target_resolution
        )

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load the model, then derive the window lead axis and the output regridder."""
        super().setup(cfg, device)
        step = infer_step(self.prognostic)
        self._leads = self.window_leads(step, self.nsteps)
        self._steps_per_window = int(self._window // np.timedelta64(step, "ns"))
        self._output_regridder = self._regridder(self._spatial_ref)

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        """Standard forecast coords with the lead axis replaced by the window ends."""
        total = super().build_total_coords(times, ensemble_size)
        total["lead_time"] = self._leads
        return total

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare the initial-condition store only; verification is not supported."""
        if OmegaConf.select(cfg, "predownload.verification.enabled", default=False):
            raise ValueError(
                "WindowMeanForecastPipeline stores window means for external "
                "scoring and declares no verification store; set "
                "predownload.verification.enabled=false."
            )
        return super().predownload_stores(cfg)

    def _rollout(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        item: WorkItem,
        label: str,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Average the parent rollout's steps into windows, one yield per window."""
        total: torch.Tensor | None = None
        count = 0
        done = 0  # windows yielded so far
        for x_step, coords_step in super()._rollout(x, coords, item, label):
            lt_axis = list(coords_step).index("lead_time")
            leads = np.asarray(coords_step["lead_time"]).astype("timedelta64[ns]")
            for j, lead in enumerate(leads):
                if lead <= np.timedelta64(0, "ns"):
                    continue  # the initial state the iterator echoes first
                # Sum in float32 so a half-precision model does not lose the mean.
                sample = x_step.select(lt_axis, j).to(torch.float32)
                total = sample.clone() if total is None else total.add_(sample)
                count += 1
                if count < self._steps_per_window:
                    continue
                expected = self._leads[done : done + 1]
                if expected.size == 0 or lead != expected[0]:
                    raise RuntimeError(
                        f"{label}: a window closed at lead {pd.Timedelta(lead)}, "
                        "which is not the next window end on the lead axis."
                    )
                out_coords: CoordSystem = OrderedDict(coords_step)
                out_coords["lead_time"] = expected
                yield (total / count).unsqueeze(lt_axis), out_coords
                total, count, done = None, 0, done + 1
        if count:
            raise RuntimeError(
                f"{label}: the rollout ended {self._steps_per_window - count} "
                "steps short of a window end."
            )
