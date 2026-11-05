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

"""Window-mean rollouts for subseasonal-to-seasonal archives.

Subseasonal-to-seasonal archives and their verification use weekly or monthly
means rather than individual steps.  :class:`WindowMeanRunner` wraps the
recipe's forecast runner, averages the steps that fall into each window, and
yields one field per window, so the forecast store has a short ``lead_time``
axis holding the window ends.  ``output.regrid`` selects the coarser grid
these archives use, as for any other runner.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig

from earth2studio.run import PrognosticRunner, WorkItem
from earth2studio.run.runner import DataRequest
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

from . import forecast

_ZERO = np.timedelta64(0, "ns")


def _to_ns(value: Any) -> np.timedelta64:
    """Return ``value`` as a nanosecond :class:`numpy.timedelta64`."""
    return np.timedelta64(value, "ns")


class WindowMeanRunner:
    """Average a prognostic rollout over fixed windows, one output per window.

    Window ``k`` covers the leads in ``((k - 1) * window, k * window]`` and
    takes its end as its label, so the ``lead_time`` axis reads ``window,
    2 * window, ...``.  The lead-0 initial condition never enters a mean.
    The runner averages every stream of the inner runner the same way: the
    ``forecast`` and each diagnostic.  It accumulates the means in float32 so a
    half-precision model does not lose them, and stores them in the stream's
    dtype.

    The window must be a whole number of model output steps and the item's
    final lead time a whole number of windows; a partial window would be a
    mean over fewer steps than its label claims.

    Calendar-month windows (``window="month"``) are not supported: the lead axis of a
    stream is a ``timedelta64`` axis shared by every initial condition, and a
    calendar month has no fixed length.  Approximate them with ``"30D"``.

    Parameters
    ----------
    inner : PrognosticRunner
        Runner whose rollout to average, typically the recipe's
        :class:`~src.runners.ensemble.EnsembleRunner`.
    window : str | np.timedelta64 | pd.Timedelta
        Window length parsed by :class:`pandas.Timedelta` (``"30D"``,
        ``"7D"``, ``"720h"``).
    """

    def __init__(self, inner: PrognosticRunner, window: Any = "30D") -> None:
        if isinstance(window, str) and window.strip().lower() == "month":
            raise NotImplementedError(
                "window='month' (calendar-month means labelled by lead month) is "
                "not supported: stream lead times are timedelta64 labels shared "
                "by every initial condition, and a calendar month has no fixed "
                "length. Use a fixed window such as '30D'."
            )
        self.inner = inner
        self.window = _to_ns(pd.Timedelta(window).to_timedelta64())
        if self.window <= _ZERO:
            raise ValueError(f"window must be positive, got {window!r}.")
        self.step = inner.step
        if _to_ns(self.step) <= _ZERO:
            raise ValueError(
                f"The inner runner's step must be positive, got {self.step}."
            )
        self.source_roles: dict[str, Any] = dict(getattr(inner, "source_roles", {}))

    # ------------------------------------------------------------------
    # Delegation to the inner runner
    # ------------------------------------------------------------------

    @property
    def supports_member_batching(self) -> bool:
        """Whether one item may carry more than one member: the inner runner's."""
        return bool(getattr(self.inner, "supports_member_batching", False))

    @property
    def prognostic(self) -> Any:
        """The inner runner's prognostic model."""
        return self.inner.prognostic

    @property
    def diagnostics(self) -> dict[str, Any]:
        """The inner runner's diagnostics by stream name."""
        return self.inner.diagnostics

    @property
    def source(self) -> Any:
        """The inner runner's initial-condition source."""
        return self.inner.source

    @property
    def forcing(self) -> tuple[Any, ...]:
        """The inner runner's forcing sources."""
        return self.inner.forcing

    @property
    def seed(self) -> int:
        """The inner runner's base seed."""
        return self.inner.seed

    @property
    def device(self) -> torch.device:
        """Device the inner runner runs on."""
        return self.inner.device

    def to(self, device: torch.device) -> WindowMeanRunner:
        """Move the inner runner to ``device``."""
        self.inner.to(device)
        return self

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Describe every fetch ``item`` makes: the inner rollout's."""
        return self.inner.data_requests(item)

    # ------------------------------------------------------------------
    # Lead schema
    # ------------------------------------------------------------------

    def window_leads(self, final_lead_time: np.timedelta64) -> np.ndarray:
        """Lead axis of the window means: the end of every window.

        Parameters
        ----------
        final_lead_time : np.timedelta64
            Final lead time of the item.

        Returns
        -------
        np.ndarray
            ``window, 2 * window, ..., final_lead_time`` as ``timedelta64[ns]``.

        Raises
        ------
        ValueError
            If the window is not a whole number of model output steps, or the
            final lead time does not cover at least one whole window.
        """
        final = _to_ns(final_lead_time)
        rollout = self.inner.output_coords(final_lead_time)["forecast"]
        leads = np.asarray(rollout["lead_time"].values).astype("timedelta64[ns]")
        positive = leads[leads > _ZERO]
        if positive.size == 0:
            raise ValueError(
                f"final lead time {pd.Timedelta(final)} covers no window of "
                f"{pd.Timedelta(self.window)}; use nsteps covering at least one window."
            )
        cadence = int(
            np.min(np.diff(np.concatenate([[_ZERO], positive]))).astype("int64")
        )
        per_window, remainder = divmod(int(self.window.astype("int64")), cadence)
        if per_window < 1 or remainder:
            raise ValueError(
                f"window {pd.Timedelta(self.window)} is not a whole number of "
                f"model steps of {pd.Timedelta(cadence, 'ns')}."
            )
        nwindows, remainder = divmod(
            int(final.astype("int64")), int(self.window.astype("int64"))
        )
        if nwindows < 1 or remainder:
            raise ValueError(
                f"final lead time {pd.Timedelta(final)} ({len(positive)} model "
                f"steps) does not cover a whole number of "
                f"{pd.Timedelta(self.window)} windows ({per_window} steps each); "
                f"use a multiple of {per_window} steps."
            )
        ends = (np.arange(1, nwindows + 1) * self.window).astype("timedelta64[ns]")
        missing = ends[~np.isin(ends, positive)]
        if missing.size:
            raise ValueError(
                f"the model never outputs lead {pd.Timedelta(missing[0])}, so the "
                f"window {pd.Timedelta(self.window)} does not end on a model step."
            )
        return ends

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Every inner stream's signature with ``lead_time`` set to the window ends."""
        leads = self.window_leads(final_lead_time)
        return {
            name: coord_array_like(signature, {"lead_time": leads})
            for name, signature in self.inner.output_coords(final_lead_time).items()
        }

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Average the inner rollout's steps into windows, one yield per window.

        Each yield maps every stream to its mean over the window just closed,
        labelled by the window end on ``lead_time``.
        """
        ends = self.window_leads(item.final_lead_time)
        totals: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        templates: dict[str, xr.DataArray] = {}
        done = 0
        for outputs in self.inner.run_item(item):
            closed: dict[str, xr.DataArray] = {}
            forecast_leads = _leads(outputs.get("forecast"))
            for stream, x in outputs.items():
                leads = _leads(x) if "lead_time" in x.dims else forecast_leads
                if leads is None:
                    raise ValueError(
                        f"Stream {stream!r} carries no lead_time and no forecast "
                        "stream accompanies it."
                    )
                for j, lead in enumerate(leads):
                    if lead <= _ZERO:
                        continue  # the initial condition the runner publishes first
                    if lead > ends[done]:
                        raise RuntimeError(
                            f"Stream {stream!r} reached lead {pd.Timedelta(lead)} "
                            f"before window {done + 1} closed at "
                            f"{pd.Timedelta(ends[done])}."
                        )
                    sample = x.isel(lead_time=[j]) if "lead_time" in x.dims else x
                    tensor, _ = sample.e2s.to_torch()
                    tensor = tensor.to(torch.float32)
                    if stream in totals:
                        totals[stream] = totals[stream].add_(tensor)
                        counts[stream] += 1
                    else:
                        totals[stream] = tensor.clone()
                        counts[stream] = 1
                        templates[stream] = sample
                    if lead == ends[done]:
                        closed[stream] = self._close(
                            totals.pop(stream),
                            counts.pop(stream),
                            templates.pop(stream),
                            ends[done],
                        )
            if closed:
                if totals:
                    raise RuntimeError(
                        f"Streams {sorted(totals)} did not close window {done + 1} "
                        f"together with {sorted(closed)}."
                    )
                yield closed
                done += 1
                if done == len(ends):
                    break
        if done != len(ends):
            raise RuntimeError(
                f"The rollout ended after {done} of {len(ends)} windows."
            )

    @staticmethod
    def _close(
        total: torch.Tensor,
        count: int,
        template: xr.DataArray,
        end: np.timedelta64,
    ) -> xr.DataArray:
        """Finish one stream's window: the mean, labelled by the window end."""
        reference, _ = template.e2s.to_torch()
        mean = (total / count).to(reference.dtype)
        if "lead_time" in template.dims:
            template = template.assign_coords(lead_time=np.array([end]))
        else:
            axis = 1 if "time" in template.dims else 0
            template = template.expand_dims(lead_time=np.array([end]), axis=axis)
            mean = mean.unsqueeze(axis)
        return from_torch(mean, template)


def _leads(x: xr.DataArray | None) -> np.ndarray | None:
    """The ``lead_time`` labels of ``x`` in nanoseconds, or ``None``."""
    if x is None or "lead_time" not in x.dims:
        return None
    return np.asarray(x.coords["lead_time"].values).astype("timedelta64[ns]")


def window_mean(
    cfg: DictConfig, prefer_predownloaded: bool = True, window: Any = "30D"
) -> WindowMeanRunner:
    """Build the window-mean runner over the recipe's forecast runner.

    Configured as::

        runner:
            name: window_mean
            window: 30D       # a whole number of model steps

    ``nsteps`` keeps counting model steps and must cover a whole number of
    windows.  A coarser output grid comes from ``output.regrid``.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config, as for :func:`src.runners.forecast`.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True
    window : str, optional
        Window length parsed by :class:`pandas.Timedelta`, by default ``"30D"``

    Returns
    -------
    WindowMeanRunner
        Runner yielding one mean per window for every stream.
    """
    return WindowMeanRunner(forecast(cfg, prefer_predownloaded), window=window)
