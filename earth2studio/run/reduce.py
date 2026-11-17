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

"""Lead-time reductions: store fewer, reduced outputs of a stream.

A reduction turns the outputs a runner yields for one stream into fewer outputs
on another lead axis, such as weekly means of a daily rollout. Pass one per
stream to :class:`~earth2studio.run.Pipeline`; the runner itself is unchanged::

    Pipeline(runner, output={"forecast": store},
             reduction={"forecast": LeadWindow("7d")})

:class:`LeadWindow` is the built-in reduction. Other reductions implement two
protocols, because the reduction and what it has seen so far live for
different lengths of time:

- :class:`LeadReduction` is the setting passed to ``Pipeline``. It is built
  once, never changes, and may serve several streams. It plans the reduced
  store and starts each item.
- :class:`LeadAccumulator` is what :meth:`LeadReduction.begin` returns: the
  partial results of one stream over one work item. It is fed the stream's
  outputs one at a time and returns reduced outputs as they complete.

``Pipeline`` drives them as follows, for each reduced stream::

    stored = reduction.output_coords(signature)        # once, plans the store
    for item in items:
        accumulator = reduction.begin(signature, item.final_lead_time)
        for x in outputs_of_the_stream(item):           # filtered to output_variables
            for reduced in accumulator.update(x):
                regrid_score_and_write(reduced)
        accumulator.finish()                            # raises if incomplete

The split mirrors :mod:`hashlib`: an algorithm, and a hash object fed with
``update``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import E2S_STATISTICS, coord_array_like
from earth2studio.utils.cupy import _torch_dtype, from_torch
from earth2studio.utils.time_statistics import (
    _TIME_STATISTICS,
    Duration,
    _duration,
    compose_time_statistic,
    format_duration,
)
from earth2studio.utils.type import CoordinateSystem

_ZERO = np.timedelta64(0, "ns")


class LeadReduction(Protocol):
    """How to reduce a stream over lead time: the setting passed to ``Pipeline``.

    A reduction holds no per-item state, so one instance may serve several
    streams and every item. ``Pipeline`` applies it after selecting the
    stream's ``output_variables`` and before its regridder, scorer and store,
    so it sees the selected variables on the runner's grid.
    """

    def output_coords(self, signature: CoordinateSystem) -> CoordinateSystem:
        """Plan the stored signature of a reduced stream.

        Called once before any item runs, with the stream's signature for the
        longest item; the result plans the store, the progress manifest and
        what the scorer sees.

        Parameters
        ----------
        signature : CoordinateSystem
            The stream's planning signature, filtered to its
            ``output_variables``.

        Returns
        -------
        CoordinateSystem
            The signature of the reduced outputs.

        Raises
        ------
        ValueError
            If the stream cannot be reduced.
        """
        ...

    def begin(
        self, signature: CoordinateSystem, final_lead_time: np.timedelta64
    ) -> LeadAccumulator:
        """Start reducing the stream for one work item.

        Called at the start of every item, and once per distinct final lead
        time before any item runs, so an item that cannot be reduced fails
        early.

        Parameters
        ----------
        signature : CoordinateSystem
            The signature passed to :meth:`output_coords`.
        final_lead_time : np.timedelta64
            The item's final lead time.

        Returns
        -------
        LeadAccumulator
            Fresh state for this item.

        Raises
        ------
        ValueError
            If an item with this final lead time cannot be reduced.
        """
        ...


class LeadAccumulator(Protocol):
    """One stream's partial results over one work item, from :meth:`LeadReduction.begin`."""

    def update(self, x: xr.DataArray) -> list[xr.DataArray]:
        """Take the stream's next output; return the reduced outputs it completes.

        Parameters
        ----------
        x : xr.DataArray
            One output of the stream, holding one or more lead times, in the
            order the runner yields them.

        Returns
        -------
        list[xr.DataArray]
            Reduced outputs completed by ``x``, matching the signature from
            :meth:`LeadReduction.output_coords`; often empty.
        """
        ...

    def finish(self) -> None:
        """End the item.

        Raises
        ------
        RuntimeError
            If the item ended before every reduced output it owes completed.
        """
        ...


def _ns(value: Any) -> np.timedelta64:
    return np.timedelta64(value, "ns")


def _leads(x: xr.DataArray) -> np.ndarray:
    return np.asarray(x.coords["lead_time"].values).astype("timedelta64[ns]")


class LeadWindow(LeadReduction):
    """Reduce a stream over fixed windows of lead time, one output per window.

    The window ending at lead ``e`` covers ``(e - window, e]``, right-closed as
    in time statistics labels, and its output is labelled ``e`` on
    ``lead_time``. A lead-0 initial condition therefore never enters a window
    starting at 0. Each output enters every window containing its lead, so
    windows given by ``ends`` may overlap; leads outside every window are
    dropped.

    ``mean``, ``sum``, ``min`` and ``max`` accumulate in place, holding one
    field per open window. Other registered statistics keep a window's fields
    until it closes. A missing sample, such as a variable an output does not
    hold, makes that variable's window NaN for the built-in statistics.
    Accumulation runs in at least float32, so a half-precision model does not
    lose precision; results are cast back to the stream's dtype.

    With ``relabel`` set, each variable is renamed with the qualified label of
    the reduction, from :func:`~earth2studio.utils.time_statistics.compose_time_statistic`:
    ``t2m`` becomes ``t2m:mean:7d`` for seven-day means, so stored fields say
    what they hold and verification can be fetched under the same labels.

    Parameters
    ----------
    window : str | np.timedelta64
        Window length, such as ``"7d"``, ``"30days"`` or ``"168h"``; a whole
        number of the stream's output cadence. Calendar months are not
        supported; use a number of days.
    method : str, optional
        Registered time statistic, by default ``"mean"``
    ends : Sequence[str | np.timedelta64] | None, optional
        Lead times at which windows end, each at least ``window``. By default
        ``window, 2 * window, ...`` up to the final lead time, which must then
        be a whole number of windows, by default None
    relabel : bool, optional
        Rename variables with the reduction's qualified label, by default True

    Examples
    --------
    >>> weekly = LeadWindow("7d")
    >>> Pipeline(runner, output={"forecast": store}, reduction={"forecast": weekly})
    >>> # Days 19-25 and 26-32, as subseasonal benchmarks score them
    >>> LeadWindow("7d", ends=["25d", "32d"])
    """

    def __init__(
        self,
        window: Duration,
        method: str = "mean",
        ends: Sequence[Duration] | None = None,
        relabel: bool = True,
    ) -> None:
        self.window = _ns(_duration(window, name="window"))
        if self.window <= _ZERO:
            raise ValueError(f"window must be positive, got {window!r}")
        self.method = method.strip().lower()
        if self.method not in _TIME_STATISTICS:
            raise ValueError(f"Unknown time statistic '{method}'")
        self.ends: np.ndarray | None = None
        if ends is not None:
            values = np.array(
                [_ns(_duration(end, name="ends")) for end in ends],
                dtype="timedelta64[ns]",
            )
            if values.size == 0:
                raise ValueError("ends names no windows")
            if np.unique(values).size != values.size:
                raise ValueError("ends repeats a window")
            if np.any(values < self.window):
                raise ValueError(
                    f"Every window end must be at least the window "
                    f"{format_duration(self.window)}, so the window starts at or "
                    "after lead 0"
                )
            self.ends = np.sort(values)
        self.relabel = relabel

    def __repr__(self) -> str:
        ends = (
            None
            if self.ends is None
            else [format_duration(_ns(end), days=True) for end in self.ends]
        )
        return (
            f"LeadWindow({format_duration(self.window, days=True)!r}, "
            f"method={self.method!r}, ends={ends!r}, relabel={self.relabel!r})"
        )

    # -- planning ------------------------------------------------------------

    def _cadence(self, signature: CoordinateSystem) -> np.timedelta64:
        if "lead_time" not in signature.dims:
            raise ValueError("A lead-time reduction needs a lead_time dimension")
        leads = _leads(signature)
        positive = leads[leads > _ZERO]
        if positive.size == 0:
            raise ValueError("The stream publishes no positive lead time to reduce")
        return _ns(np.min(np.diff(np.concatenate([[_ZERO], np.sort(positive)]))))

    def _window_ends(
        self, signature: CoordinateSystem, final_lead_time: np.timedelta64
    ) -> np.ndarray:
        """Window ends an item reaching ``final_lead_time`` covers, checked."""
        cadence = self._cadence(signature)
        if self.window % cadence:
            raise ValueError(
                f"Window {format_duration(self.window)} is not a whole number of "
                f"the stream's {format_duration(cadence)} steps"
            )
        final = _ns(final_lead_time)
        if self.ends is None:
            count, remainder = divmod(final, self.window)
            if count < 1 or remainder:
                raise ValueError(
                    f"Final lead time {format_duration(final)} is not a whole, "
                    f"nonzero number of {format_duration(self.window)} windows; "
                    "pass ends= to choose the windows"
                )
            ends = np.arange(1, int(count) + 1) * self.window
        else:
            ends = self.ends[self.ends <= final]
        published = _leads(signature)
        for end in ends:
            needed = np.arange(end - self.window + cadence, end + cadence, cadence)
            missing = needed[~np.isin(needed, published)]
            if missing.size:
                raise ValueError(
                    f"The stream never publishes lead {format_duration(missing[0])}, "
                    f"which the window ending at {format_duration(end)} needs"
                )
        return ends.astype("timedelta64[ns]")

    def _labels(self, signature: CoordinateSystem) -> np.ndarray | None:
        if not self.relabel or "variable" not in signature.dims:
            return None
        cadence = self._cadence(signature)
        return np.array(
            [
                compose_time_statistic(str(label), self.method, self.window, cadence)
                for label in np.asarray(signature["variable"].values)
            ]
        )

    def output_coords(self, signature: CoordinateSystem) -> CoordinateSystem:
        """Plan the reduced signature: window ends on ``lead_time``, relabelled.

        Parameters
        ----------
        signature : CoordinateSystem
            The stream's planning signature for the longest item.

        Returns
        -------
        CoordinateSystem
            Signature with ``lead_time`` holding every window end and, with
            ``relabel``, the reduction's qualified variable labels.

        Raises
        ------
        ValueError
            If the window is not a whole number of the stream's steps, the
            stream does not publish every lead a window needs, an explicit end
            lies beyond the longest item, or a label does not compose.
        """
        leads = _leads(signature) if "lead_time" in signature.dims else None
        final = _ns(leads.max()) if leads is not None and leads.size else _ZERO
        ends = self._window_ends(signature, final)
        if self.ends is not None and ends.size < self.ends.size:
            raise ValueError(
                f"Window end {format_duration(_ns(self.ends[ends.size]))} lies beyond "
                f"the longest item's final lead time {format_duration(final)}"
            )
        replacements: dict[str, Any] = {"lead_time": ends}
        labels = self._labels(signature)
        if labels is not None:
            replacements["variable"] = labels
        return coord_array_like(signature, replacements)

    def begin(
        self, signature: CoordinateSystem, final_lead_time: np.timedelta64
    ) -> LeadAccumulator:
        """Start reducing one item: the windows ending by its final lead time.

        Parameters
        ----------
        signature : CoordinateSystem
            The stream's planning signature, as passed to :meth:`output_coords`.
        final_lead_time : np.timedelta64
            The item's final lead time.

        Returns
        -------
        LeadAccumulator
            Running statistics of the item's windows.

        Raises
        ------
        ValueError
            If the item's final lead time is not a whole number of windows
            (without ``ends``), or the stream misses a lead a window needs.
        """
        ends = self._window_ends(signature, final_lead_time)
        variables = (
            np.asarray(signature["variable"].values)
            if "variable" in signature.dims
            else None
        )
        return _WindowAccumulator(
            self,
            ends,
            self._cadence(signature),
            variables,
            self._labels(signature),
            signature.dtype,
        )


class _RunningStatistic:
    """One window's running statistic over fields of identical shape."""

    _UPDATES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "mean": torch.Tensor.add_,
        "sum": torch.Tensor.add_,
        "min": lambda total, x: torch.minimum(total, x, out=total),
        "max": lambda total, x: torch.maximum(total, x, out=total),
    }

    def __init__(self, method: str) -> None:
        self.method = method
        self.count = 0
        self.total: torch.Tensor | None = None
        self.samples: list[torch.Tensor] = []

    def add(self, x: torch.Tensor) -> None:
        self.count += 1
        update = self._UPDATES.get(self.method)
        if update is None:
            self.samples.append(x)
        elif self.total is None:
            self.total = x.clone()
        else:
            update(self.total, x)

    def result(self) -> torch.Tensor:
        if self.method not in self._UPDATES:
            stacked = torch.stack(self.samples)
            coords = OrderedDict(
                (f"axis{n}", np.arange(size)) for n, size in enumerate(stacked.shape)
            )
            array = from_torch(stacked, coords)
            reduced = _TIME_STATISTICS[self.method](array, "axis0")
            tensor, _ = reduced.e2s.to_torch()
            return tensor
        assert self.total is not None  # noqa: S101 - result follows add
        if self.method == "mean":
            return self.total / self.count
        return self.total


class _WindowAccumulator(LeadAccumulator):
    """The windows of one :class:`LeadWindow` item, closed as their last lead arrives."""

    def __init__(
        self,
        spec: LeadWindow,
        ends: np.ndarray,
        cadence: np.timedelta64,
        variables: np.ndarray | None,
        labels: np.ndarray | None,
        dtype: np.dtype,
    ) -> None:
        self.spec = spec
        self.ends = ends
        self.samples = int(spec.window // cadence)
        self.variables = variables
        self.labels = labels
        self.dtype = dtype
        self.open: dict[np.timedelta64, _RunningStatistic] = {}
        self.closed: set[np.timedelta64] = set()

    def update(self, x: xr.DataArray) -> list[xr.DataArray]:
        """Add one output to its windows; return the windows it completes.

        Parameters
        ----------
        x : xr.DataArray
            One output of the stream, holding one or more lead times.

        Returns
        -------
        list[xr.DataArray]
            The reduced output of each window closed by ``x``, in lead order.
        """
        if "lead_time" not in x.dims:
            raise ValueError("A lead-time reduction needs outputs with lead_time")
        start = self.ends - self.spec.window
        done: list[xr.DataArray] = []
        for j, lead in enumerate(_leads(x)):
            windows = self.ends[(start < lead) & (lead <= self.ends)]
            if windows.size == 0:
                continue
            sample = x.isel(lead_time=[j])
            tensor = self._tensor(sample)
            for end in windows:
                if end in self.closed:
                    raise RuntimeError(
                        f"Lead {format_duration(lead)} arrived after the window "
                        f"ending at {format_duration(end)} closed"
                    )
                if end not in self.open:
                    self.open[end] = _RunningStatistic(self.spec.method)
                self.open[end].add(tensor)
                if lead == end:
                    done.append(self._close(end, sample))
        return done

    def finish(self) -> None:
        """Check every window of the item closed.

        Raises
        ------
        RuntimeError
            If the runner never published the lead closing some window.
        """
        unclosed = [end for end in self.ends if end not in self.closed]
        if unclosed:
            raise RuntimeError(
                f"The item ended before the window ending at "
                f"{format_duration(unclosed[0])} closed"
            )

    def _tensor(self, sample: xr.DataArray) -> torch.Tensor:
        """The sample in float32 or wider, NaN for planned variables it lacks."""
        tensor, _ = sample.e2s.to_torch()
        dtype = torch.promote_types(tensor.dtype, torch.float32)
        tensor = tensor.to(dtype)
        if self.variables is None:
            return tensor
        held = np.asarray(sample["variable"].values)
        if np.array_equal(held, self.variables):
            return tensor
        axis = sample.get_axis_num("variable")
        index = {str(name): n for n, name in enumerate(held)}
        shape = list(tensor.shape)
        shape[axis] = len(self.variables)
        full = torch.full(shape, torch.nan, dtype=dtype, device=tensor.device)
        for n, name in enumerate(self.variables):
            if str(name) in index:
                full.select(axis, n).copy_(tensor.select(axis, index[str(name)]))
        return full

    def _close(self, end: np.timedelta64, sample: xr.DataArray) -> xr.DataArray:
        statistic = self.open.pop(end)
        if statistic.count != self.samples:
            raise RuntimeError(
                f"The window ending at {format_duration(end)} received "
                f"{statistic.count} outputs, not {self.samples}"
            )
        self.closed.add(end)
        replacements: dict[str, Any] = {"lead_time": np.array([end])}
        if self.variables is not None:
            replacements["variable"] = (
                self.variables if self.labels is None else self.labels
            )
        signature = coord_array_like(sample, replacements)
        output = from_torch(statistic.result().to(_torch_dtype(self.dtype)), signature)
        output.attrs.pop(E2S_STATISTICS, None)
        statistics = signature.attrs.get(E2S_STATISTICS)
        if statistics:
            output.attrs[E2S_STATISTICS] = statistics
        return output
