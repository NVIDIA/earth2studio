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

"""Lead-time reductions applied by :class:`~earth2studio.run.Pipeline`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import E2S_STATISTICS, coord_array_like
from earth2studio.utils.cupy import _torch_dtype, from_torch
from earth2studio.utils.time_statistics import (
    Duration,
    _duration,
    compose_time_statistic,
    format_duration,
)
from earth2studio.utils.type import CoordinateSystem

_ZERO = np.timedelta64(0, "ns")

# In-place running update of each method; a mean divides by the count at the end.
_UPDATES: dict[str, Callable[[torch.Tensor, torch.Tensor], Any]] = {
    "mean": torch.Tensor.add_,
    "sum": torch.Tensor.add_,
    "min": lambda total, x: torch.minimum(total, x, out=total),
    "max": lambda total, x: torch.maximum(total, x, out=total),
}


class LeadReduction(Protocol):
    """Reduce one stream over lead time, item by item.

    A reduction turns the outputs a runner yields for one stream into fewer
    outputs on another lead axis, such as weekly means of a daily rollout,
    without changing the runner. ``Pipeline`` applies it after selecting the
    stream's ``output_variables`` and before its regridder, scorer and store,
    so it sees the selected variables on the runner's grid.

    The lifecycle matches :class:`~earth2studio.run.Scorer`. ``Pipeline``
    deep-copies each reduction per stream, so a reduction may keep the current
    item's partial results on itself, and drives it as follows::

        stored = reduction.output_coords(signature)        # once, plans the store
        for item in items:
            reduction.begin_item(signature, item.final_lead_time)
            for x in outputs_of_the_stream(item):
                for reduced in reduction.update(x):
                    regrid_score_and_write(reduced)
            reduction.finish_item()                         # raises if incomplete

    :class:`LeadWindow` is the built-in implementation.
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

    def begin_item(
        self, signature: CoordinateSystem, final_lead_time: np.timedelta64
    ) -> None:
        """Discard any partial results and start one work item.

        Called at the start of every item, and once per distinct final lead
        time before any item runs, so an item that cannot be reduced fails
        early.

        Parameters
        ----------
        signature : CoordinateSystem
            The signature passed to :meth:`output_coords`.
        final_lead_time : np.timedelta64
            The item's final lead time.

        Raises
        ------
        ValueError
            If an item with this final lead time cannot be reduced.
        """
        ...

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
            :meth:`output_coords`; often empty.
        """
        ...

    def finish_item(self) -> None:
        """End the item.

        Raises
        ------
        RuntimeError
            If the item ended before every reduced output it owes completed.
        """
        ...


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

    Each variable is renamed with the qualified label of the reduction, from
    :func:`~earth2studio.utils.time_statistics.compose_time_statistic`:
    ``t2m`` becomes ``t2m:mean:168h`` for seven-day means, so stored fields say
    what they hold and verification can be fetched under the same labels. A
    stream whose labels do not compose with ``method``, such as a mean of
    six-hourly sums, cannot be reduced.

    Statistics accumulate in place, one field per open window, in at least
    float32, so a half-precision model does not lose precision; results are
    cast back to the stream's dtype. A missing sample, such as a variable an
    output does not hold, makes that variable's window NaN.

    Parameters
    ----------
    window : str | np.timedelta64
        Window length, such as ``"7d"``, ``"30days"`` or ``"168h"``; a whole
        number of the stream's output cadence. Calendar months are not
        supported; use a number of days.
    method : {"mean", "sum", "min", "max"}, optional
        Statistic over each window, by default ``"mean"``
    ends : Sequence[str | np.timedelta64] | None, optional
        Lead times at which windows end, each at least ``window``. By default
        ``window, 2 * window, ...`` up to the final lead time, which must then
        be a whole number of windows, by default None

    Examples
    --------
    Store weekly means of the ``forecast`` stream:

    >>> Pipeline(
    ...     runner,
    ...     output={"forecast": store},
    ...     reduction={"forecast": LeadWindow("7d")},
    ... )

    Days 19-25 and 26-32, as subseasonal benchmarks score them:

    >>> LeadWindow("7d", ends=["25d", "32d"])
    """

    def __init__(
        self,
        window: Duration,
        method: Literal["mean", "sum", "min", "max"] = "mean",
        ends: Sequence[Duration] | None = None,
    ) -> None:
        self.window = _duration(window, name="window").astype("timedelta64[ns]")
        if self.window <= _ZERO:
            raise ValueError(f"window must be positive, got {window!r}")
        if method not in _UPDATES:
            raise ValueError(f"method must be one of {list(_UPDATES)}, got {method!r}")
        self.method = method
        self.ends: np.ndarray | None = None
        if ends is not None:
            values = np.array(
                [_duration(end, name="ends") for end in ends], dtype="timedelta64[ns]"
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
        # The current item, set by begin_item: its window ends, the samples
        # each window takes, the planned variables and their labels, the
        # stream's dtype, and each open window's running total and count.
        self._item_ends: np.ndarray | None = None
        self._samples = 0
        self._variables: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._dtype: Any = None
        self._open: dict[np.timedelta64, tuple[torch.Tensor, int]] = {}
        self._closed: set[np.timedelta64] = set()

    def __repr__(self) -> str:
        ends = None if self.ends is None else [format_duration(e) for e in self.ends]
        return (
            f"LeadWindow({format_duration(self.window)!r}, "
            f"method={self.method!r}, ends={ends!r})"
        )

    # -- planning ------------------------------------------------------------

    def _cadence(self, signature: CoordinateSystem) -> np.timedelta64:
        if "lead_time" not in signature.dims:
            raise ValueError("A lead-time reduction needs a lead_time dimension")
        leads = _leads(signature)
        positive = np.sort(leads[leads > _ZERO])
        if positive.size == 0:
            raise ValueError("The stream publishes no positive lead time to reduce")
        if positive.size == 1:
            return positive[0]
        # Gaps between published leads, so an offset stream such as 3, 9, 15 h
        # has a 6 h cadence rather than the 3 h from zero to its first lead.
        return np.min(np.diff(positive))

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
        final = np.timedelta64(final_lead_time, "ns")
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

    def _relabel(self, signature: CoordinateSystem) -> np.ndarray | None:
        if "variable" not in signature.dims:
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
            Signature with ``lead_time`` holding every window end and the
            reduction's qualified variable labels.

        Raises
        ------
        ValueError
            If the window is not a whole number of the stream's steps, the
            stream does not publish every lead a window needs, an explicit end
            lies beyond the longest item, or a label does not compose.
        """
        leads = _leads(signature) if "lead_time" in signature.dims else None
        final = leads.max() if leads is not None and leads.size else _ZERO
        ends = self._window_ends(signature, final)
        if self.ends is not None and ends.size < self.ends.size:
            raise ValueError(
                f"Window end {format_duration(self.ends[ends.size])} lies beyond "
                f"the longest item's final lead time {format_duration(final)}"
            )
        replacements: dict[str, Any] = {"lead_time": ends}
        labels = self._relabel(signature)
        if labels is not None:
            replacements["variable"] = labels
        return coord_array_like(signature, replacements)

    # -- one item ------------------------------------------------------------

    def begin_item(
        self, signature: CoordinateSystem, final_lead_time: np.timedelta64
    ) -> None:
        """Start one item: the windows ending by its final lead time.

        Parameters
        ----------
        signature : CoordinateSystem
            The stream's planning signature, as passed to :meth:`output_coords`.
        final_lead_time : np.timedelta64
            The item's final lead time.

        Raises
        ------
        ValueError
            If the item's final lead time is not a whole number of windows
            (without ``ends``), or the stream misses a lead a window needs.
        """
        self._item_ends = None
        self._open = {}
        self._closed = set()
        ends = self._window_ends(signature, final_lead_time)
        self._samples = int(self.window // self._cadence(signature))
        self._variables = (
            np.asarray(signature["variable"].values)
            if "variable" in signature.dims
            else None
        )
        self._labels = self._relabel(signature)
        self._dtype = signature.dtype
        self._item_ends = ends

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
        ends = self._current()
        if "lead_time" not in x.dims:
            raise ValueError("A lead-time reduction needs outputs with lead_time")
        done: list[xr.DataArray] = []
        for j, lead in enumerate(_leads(x)):
            windows = ends[(ends - self.window < lead) & (lead <= ends)]
            if windows.size == 0:
                continue
            sample = x.isel(lead_time=[j])
            tensor = self._tensor(sample)
            for end in windows:
                if end in self._closed:
                    raise RuntimeError(
                        f"Lead {format_duration(lead)} arrived after the window "
                        f"ending at {format_duration(end)} closed"
                    )
                if end in self._open:
                    total, count = self._open[end]
                    _UPDATES[self.method](total, tensor)
                    self._open[end] = (total, count + 1)
                else:
                    self._open[end] = (tensor.clone(), 1)
                if lead == end:
                    done.append(self._close(end, sample))
        return done

    def finish_item(self) -> None:
        """Check every window of the item closed, then end it.

        Raises
        ------
        RuntimeError
            If the runner never published the lead closing some window.
        """
        ends = self._current()
        self._item_ends = None
        unclosed = [end for end in ends if end not in self._closed]
        if unclosed:
            raise RuntimeError(
                f"The item ended before the window ending at "
                f"{format_duration(unclosed[0])} closed"
            )

    def _current(self) -> np.ndarray:
        if self._item_ends is None:
            raise RuntimeError("LeadWindow.begin_item was not called for this item")
        return self._item_ends

    def _tensor(self, sample: xr.DataArray) -> torch.Tensor:
        """The sample in float32 or wider, NaN for planned variables it lacks."""
        tensor, _ = sample.e2s.to_torch()
        dtype = torch.promote_types(tensor.dtype, torch.float32)
        tensor = tensor.to(dtype)
        if self._variables is None:
            return tensor
        held = np.asarray(sample["variable"].values)
        if np.array_equal(held, self._variables):
            return tensor
        axis = sample.get_axis_num("variable")
        index = {str(name): n for n, name in enumerate(held)}
        shape = list(tensor.shape)
        shape[axis] = len(self._variables)
        full = torch.full(shape, torch.nan, dtype=dtype, device=tensor.device)
        for n, name in enumerate(self._variables):
            if str(name) in index:
                full.select(axis, n).copy_(tensor.select(axis, index[str(name)]))
        return full

    def _close(self, end: np.timedelta64, sample: xr.DataArray) -> xr.DataArray:
        """Reduce the window ending at ``end``, shaped like ``sample``."""
        total, count = self._open.pop(end)
        if count != self._samples:
            raise RuntimeError(
                f"The window ending at {format_duration(end)} received "
                f"{count} outputs, not {self._samples}"
            )
        self._closed.add(end)
        result = total / count if self.method == "mean" else total
        replacements: dict[str, Any] = {"lead_time": np.array([end])}
        if self._labels is not None:
            replacements["variable"] = self._labels
        signature = coord_array_like(sample, replacements)
        output = from_torch(result.to(_torch_dtype(self._dtype)), signature)
        output.attrs.pop(E2S_STATISTICS, None)
        statistics = signature.attrs.get(E2S_STATISTICS)
        if statistics:
            output.attrs[E2S_STATISTICS] = statistics
        return output
