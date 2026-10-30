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
    """Forecast pipeline that stores time means over fixed or calendar windows.

    Subseasonal-to-seasonal archives and verification use weekly or
    monthly means rather than individual steps.  This pipeline drives
    the standard prognostic rollout, averages the steps that fall into each
    window, and yields one field per window, interpolated onto a coarser
    regular grid with :class:`~src.regrid.LinearRegridder` when
    ``target_resolution`` asks for one.  The forecast store is otherwise an
    ordinary one with a short lead axis.  ``window`` selects the windows:

    * A fixed length such as ``"30D"``.  Window ``k`` covers the leads in
      ``(k * window, (k + 1) * window]`` and takes its end as its label, so
      the ``lead_time`` axis reads ``window, 2 * window, ...`` and the
      lead-0 analysis never enters a mean.  Every initial condition shares
      this lead axis.
    * ``"month"``, for UTC calendar months, the unit that seasonal hindcast
      archives such as C3S use for verification.  A step belongs to the
      month containing its valid time, and ``lead_time`` holds integer lead
      months: lead month ``k`` is the ``k``-th month after the month that
      contains the initial condition.  Lead month 0, that month itself,
      needs an initial condition at 00 UTC on the 1st, whose initial state
      then counts as the month's first sample; any other start stores NaN
      there.  Integer lead months leave no lead times to verify against, so
      online scoring and ``score.py`` reject this mode.

    A model that consumes time means itself (FuXi-S2S) takes them from a
    :class:`~src.data.WindowMeanSource` data source, see
    ``cfg/data_source/arco_era5_daily.yaml``.  The pipeline leaves scoring to
    the user: it declares no verification store, and with a fixed window a
    bring-your-own ``verification_source`` would have to serve window means
    at the window ends on the output grid.

    Parameters
    ----------
    window : str
        ``"month"``, or a fixed window length parsed by ``pandas.Timedelta``
        (``"30D"``, ``"7D"``, ``"720h"``).  A fixed window must be a whole
        number of model steps, and ``nsteps`` must cover a whole number of
        windows.
    target_resolution : float | None
        Spacing in degrees of the regular latitude/longitude grid to
        interpolate the output onto.  ``None`` keeps the model grid, as
        does a model whose grid is already that coarse.
    months : int | None
        Lead months stored with ``window="month"``: lead months 0 to
        ``months - 1``.  ``nsteps`` must reach the end of the last one for
        every initial condition, and each rollout stops once that month is
        complete.  Leave unset for a fixed window.
    """

    def __init__(
        self,
        window: str,
        target_resolution: float | None = None,
        months: int | None = None,
    ) -> None:
        super().__init__()
        self._calendar = str(window).strip().lower() == "month"
        if self._calendar:
            if months is None or int(months) < 1:
                raise ValueError(
                    f"window='month' needs a positive months, got {months!r}."
                )
            self._months = int(months)
            self._window = np.timedelta64(0, "ns")
            # Integer lead months give the online scorer no valid times.
            self.supports_online_scoring = False
        else:
            if months is not None:
                raise ValueError(
                    f"months={months!r} only applies to window='month', "
                    f"not window={window!r}."
                )
            self._months = 0
            self._window = (
                pd.Timedelta(window).to_timedelta64().astype("timedelta64[ns]")
            )
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
        self._step = np.timedelta64(0, "ns")

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
        if self._calendar:
            raise ValueError("window_leads needs a fixed window, not window='month'.")
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

    def month_windows(
        self, ic: np.datetime64
    ) -> list[tuple[int, np.timedelta64, np.timedelta64]]:
        """Lead bounds of the calendar months a forecast from *ic* fills.

        Returns ``(k, start, end)`` for every lead month ``k`` the forecast
        fills from start to end: the steps with lead times in ``[start, end)`` have
        valid times in the ``k``-th UTC calendar month after the one
        containing *ic*.  Lead month 0 appears when *ic* is 00 UTC on the
        1st, so that the initial state opens the month.
        """
        t0 = np.datetime64(ic, "ns")
        month = t0.astype("datetime64[M]")
        first = 0 if month.astype("datetime64[ns]") == t0 else 1
        return [
            (
                k,
                (month + k).astype("datetime64[ns]") - t0,
                (month + k + 1).astype("datetime64[ns]") - t0,
            )
            for k in range(first, self._months)
        ]

    def _check_month_horizon(self, times: np.ndarray, leads: np.ndarray) -> None:
        """Raise unless ``nsteps`` reaches the last lead month of every start.

        Parameters
        ----------
        times : np.ndarray
            Initial conditions of the run.
        leads : np.ndarray
            Lead axis of the full rollout, as the parent pipeline builds it.
        """
        horizon = np.asarray(leads).astype("timedelta64[ns]").max()
        worst: tuple[np.datetime64, np.timedelta64] | None = None
        for t in np.atleast_1d(times):
            windows = self.month_windows(t)
            if not windows:
                raise ValueError(
                    f"The initial condition {pd.Timestamp(t)} is not 00 UTC on "
                    f"the 1st, so lead month 0 holds NaN and months="
                    f"{self._months} stores nothing; use months >= 2."
                )
            needed = (_ceil_steps(windows[-1][2], self._step) - 1) * self._step
            if needed > horizon and (worst is None or needed > worst[1]):
                worst = (t, needed)
        if worst is None:
            return
        t, needed = worst
        # Lead covered per iterator step, which spans more than one model
        # step for a model that yields more than one lead at a time.
        stride = horizon / self.nsteps if self.nsteps > 0 else self._step
        if stride <= np.timedelta64(0, "ns"):
            stride = self._step
        raise ValueError(
            f"nsteps={self.nsteps} reaches lead {pd.Timedelta(horizon)}, but the "
            f"initial condition {pd.Timestamp(t)} needs lead "
            f"{pd.Timedelta(needed)} to finish lead month {self._months - 1}; "
            f"use nsteps >= {int(np.ceil(needed / stride))}."
        )

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
        self._step = np.timedelta64(step, "ns")
        if self._calendar:
            self._leads = np.arange(self._months, dtype=np.int64)
        else:
            self._leads = self.window_leads(step, self.nsteps)
            self._steps_per_window = int(self._window // self._step)
        self._output_regridder = self._regridder(self._spatial_ref)

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        """Standard forecast coords with the lead axis replaced by the window labels."""
        total = super().build_total_coords(times, ensemble_size)
        if self._calendar:
            self._check_month_horizon(times, total["lead_time"])
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
        if self._calendar:
            yield from self._rollout_months(x, coords, item, label)
            return
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
        if done != len(self._leads):
            raise RuntimeError(
                f"{label}: the rollout ended after {done} of "
                f"{len(self._leads)} windows."
            )

    def _rollout_months(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        item: WorkItem,
        label: str,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Average the parent rollout's steps into calendar months.

        Yields one field per lead month and stops the rollout once the last
        month is complete.  A start inside a month first yields NaN for lead
        month 0, so the store never shows its zero fill value as data.
        """
        windows = self.month_windows(item.time)
        if not windows:
            raise ValueError(
                f"{label}: a start inside a month has no full month among lead "
                f"months 0 to {self._months - 1}; use months >= 2."
            )
        fill_month0 = windows[0][0] == 1
        total: torch.Tensor | None = None
        count = 0
        done = 0  # full lead months yielded so far
        for x_step, coords_step in super()._rollout(x, coords, item, label):
            lt_axis = list(coords_step).index("lead_time")
            if fill_month0:
                empty = torch.full_like(
                    x_step.select(lt_axis, 0), float("nan"), dtype=torch.float32
                )
                yield empty.unsqueeze(lt_axis), _with_lead_month(coords_step, 0)
                fill_month0 = False
            leads = np.asarray(coords_step["lead_time"]).astype("timedelta64[ns]")
            for j, lead in enumerate(leads):
                k, start, end = windows[done]
                if lead < start:
                    continue  # input history, or the rest of the start month
                if lead >= end:
                    raise RuntimeError(
                        f"{label}: lead {pd.Timedelta(lead)} skipped past the end "
                        f"of lead month {k}."
                    )
                # Sum in float32 so a half-precision model does not lose the mean.
                sample = x_step.select(lt_axis, j).to(torch.float32)
                total = sample.clone() if total is None else total.add_(sample)
                count += 1
                if lead + self._step < end:
                    continue
                expected = _ceil_steps(end, self._step) - _ceil_steps(start, self._step)
                if count != expected:
                    raise RuntimeError(
                        f"{label}: lead month {k} averaged {count} steps, "
                        f"expected {expected}."
                    )
                mean = (total / count).unsqueeze(lt_axis)
                yield mean, _with_lead_month(coords_step, k)
                total, count, done = None, 0, done + 1
                if done == len(windows):
                    return  # later steps fall outside every stored month
        raise RuntimeError(
            f"{label}: the rollout ended after {done} of {len(windows)} calendar "
            "months."
        )


def _ceil_steps(lead: np.timedelta64, step: np.timedelta64) -> int:
    """Model steps from lead 0 that come before *lead*: ``ceil(lead / step)``."""
    return -(-int(lead.astype("int64")) // int(step.astype("int64")))


def _with_lead_month(coords: CoordSystem, k: int) -> CoordSystem:
    """Copy of *coords* whose lead axis holds the single lead month *k*."""
    out: CoordSystem = OrderedDict(coords)
    out["lead_time"] = np.array([k], dtype=np.int64)
    return out
