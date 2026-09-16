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

"""Utilities for planning and applying temporal statistics."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

TimeReduction: TypeAlias = Callable[[xr.DataArray, Hashable], xr.DataArray]
StatisticDeclaration: TypeAlias = Mapping[str, str] | str | None
Duration: TypeAlias = str | np.timedelta64
TemporalTarget: TypeAlias = np.datetime64 | np.timedelta64

_DURATION_PATTERN = re.compile(r"^([+-]?)(\d+)([a-z]+)$")
_DURATION_UNITS = {
    "ns": (1, "ns"),
    "nanosecond": (1, "ns"),
    "us": (1, "us"),
    "microsecond": (1, "us"),
    "ms": (1, "ms"),
    "millisecond": (1, "ms"),
    "s": (1, "s"),
    "sec": (1, "s"),
    "second": (1, "s"),
    "m": (1, "m"),
    "min": (1, "m"),
    "minute": (1, "m"),
    "h": (1, "h"),
    "hr": (1, "h"),
    "hour": (1, "h"),
    "d": (1, "D"),
    "day": (1, "D"),
    "w": (7, "D"),
    "wk": (7, "D"),
    "week": (7, "D"),
    "mo": (30, "D"),
    "mon": (30, "D"),
    "month": (30, "D"),
}
_TIME_STATISTICS: dict[str, TimeReduction] = {}


def _duration(value: Duration, *, name: str) -> np.timedelta64:
    if isinstance(value, np.timedelta64):
        duration = value
    elif isinstance(value, str):
        match = _DURATION_PATTERN.fullmatch(value.strip().lower())
        if match is None:
            raise ValueError(f"{name} must be an integer duration such as '6h'")
        sign, magnitude, unit = match.groups()
        unit = unit if unit in _DURATION_UNITS else unit.removesuffix("s")
        if unit not in _DURATION_UNITS:
            raise ValueError(f"Unsupported {name} unit '{unit}'")
        scale, numpy_unit = _DURATION_UNITS[unit]
        duration = np.timedelta64(
            (-1 if sign == "-" else 1) * int(magnitude) * scale,
            numpy_unit,
        )
    else:
        raise TypeError(f"{name} must be a string or numpy.timedelta64")
    if np.isnat(duration):
        raise ValueError(f"{name} must not be NaT")
    return duration


def _nanoseconds(value: np.timedelta64) -> int:
    try:
        return int(value.astype("timedelta64[ns]").astype(np.int64))
    except TypeError as error:
        raise ValueError("Calendar-relative durations are not supported") from error


def _compact_duration(value: np.timedelta64, *, signed: bool = False) -> str:
    nanoseconds = _nanoseconds(value)
    sign = "-" if nanoseconds < 0 else "+" if signed and nanoseconds > 0 else ""
    magnitude = abs(nanoseconds)
    for suffix, scale in (
        ("h", 3_600_000_000_000),
        ("m", 60_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ):
        if magnitude and magnitude % scale == 0:
            return f"{sign}{magnitude // scale}{suffix}"
    return "0h"


@dataclass(frozen=True)
class _Window:
    """Parsed window using NumPy timedeltas as the duration representation.

    Public declarations remain compact strings, while metadata exposes these same
    timedelta values without converting to another duration format.
    """

    method: str
    start: np.timedelta64
    end: np.timedelta64

    @property
    def modifier(self) -> str:
        window = self.end - self.start
        if self.start == -window and _nanoseconds(self.end) == 0:
            return f"{self.method}:{_compact_duration(window)}"
        return (
            f"{self.method}:{_compact_duration(self.start, signed=True)}:"
            f"{_compact_duration(self.end, signed=True)}"
        )

    def offsets(self, delta_t: np.timedelta64) -> NDArray[Any]:
        if not isinstance(delta_t, np.timedelta64):
            raise TypeError("delta_t must be numpy.timedelta64")
        delta = _duration(delta_t, name="delta_t")
        step = _nanoseconds(delta)
        window = _nanoseconds(self.end - self.start)
        if step <= 0:
            raise ValueError("delta_t must be positive")
        if window % step:
            raise ValueError("Statistic window must be divisible by delta_t")
        return np.arange(self.start, self.end, delta)


@lru_cache
def _parse(modifier: str) -> _Window:
    parts = modifier.strip().lower().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "Statistic modifier must be 'method:window' or 'method:start:end'"
        )
    method = parts[0]
    if method not in _TIME_STATISTICS:
        raise ValueError(f"Unknown time statistic '{method}'")
    if len(parts) == 2:
        window = _duration(parts[1], name="window")
        if _nanoseconds(window) <= 0:
            raise ValueError("Statistic window must be positive")
        return _Window(method, -window, np.timedelta64(0, "h"))
    start = _duration(parts[1], name="start_offset")
    end = _duration(parts[2], name="end_offset")
    if _nanoseconds(end - start) <= 0:
        raise ValueError("Statistic end_offset must be after start_offset")
    return _Window(method, start, end)


def register_time_statistic(name: str, reduction: TimeReduction) -> None:
    """Register a block-wise temporal reduction."""
    method = name.strip().lower()
    if not method or not re.fullmatch(r"[a-z][a-z0-9_-]*", method):
        raise ValueError("Time statistic name must be a lowercase identifier")
    if not callable(reduction):
        raise TypeError("Time statistic reduction must be callable")
    existing = _TIME_STATISTICS.get(method)
    if existing is not None and existing is not reduction:
        raise ValueError(f"Time statistic '{method}' is already registered")
    _TIME_STATISTICS[method] = reduction
    _parse.cache_clear()


def list_time_statistics() -> tuple[str, ...]:
    """Return registered temporal reduction names."""
    return tuple(sorted(_TIME_STATISTICS))


def _group_time_statistics(
    variables: Sequence[str], statistics: StatisticDeclaration
) -> dict[str, tuple[str, ...]]:
    """Group variables by normalized temporal-statistic modifier."""
    labels = tuple(str(variable) for variable in variables)
    parsed = tuple(split_time_statistic(label) for label in labels)
    if statistics is None:
        declarations = parsed
    elif isinstance(statistics, str):
        if any(modifier for _, modifier in parsed):
            raise ValueError("Qualified variables cannot use a separate statistic")
        declarations = tuple((label, statistics) for label in labels)
    else:
        if any(modifier for _, modifier in parsed):
            raise ValueError("Qualified variables cannot use a statistics mapping")
        unknown = set(statistics) - set(labels)
        if unknown:
            raise ValueError(
                f"Statistics reference unknown variables: {sorted(unknown)}"
            )
        declarations = tuple((label, statistics.get(label)) for label in labels)
    groups: dict[str, list[str]] = {}
    for variable, modifier in declarations:
        if modifier is not None:
            groups.setdefault(_parse(modifier).modifier, []).append(variable)
    return {modifier: tuple(group) for modifier, group in groups.items()}


def split_time_statistic(variable: str) -> tuple[str, str | None]:
    """Split a variable label into its source name and statistic modifier.

    Parameters
    ----------
    variable : str
        Plain or statistic-qualified variable label.

    Returns
    -------
    tuple[str, str | None]
        Source variable name and optional normalized modifier.
    """
    name, separator, modifier = variable.partition(":")
    if not name:
        raise ValueError("Variable name must not be empty")
    return (name, _parse(modifier).modifier) if separator else (name, None)


def source_times(
    modifier: str, valid_time: Any, delta_t: np.timedelta64
) -> NDArray[Any]:
    """Return analysis timestamps required by one statistic."""
    target = np.asarray(valid_time)
    if not np.issubdtype(target.dtype, np.datetime64):
        raise TypeError("valid_time must contain numpy datetime values")
    return target[..., None] + _parse(modifier).offsets(delta_t)


def source_lead_times(
    modifier: str, lead_time: Any, delta_t: np.timedelta64
) -> NDArray[Any]:
    """Return forecast lead times required by one statistic."""
    target = np.asarray(lead_time)
    if not np.issubdtype(target.dtype, np.timedelta64):
        raise TypeError("lead_time must contain numpy timedelta values")
    return target[..., None] + _parse(modifier).offsets(delta_t)


def _select_time_window(
    array: xr.DataArray,
    window: _Window,
    target: TemporalTarget,
    delta_t: np.timedelta64,
    dimension: Hashable,
) -> xr.DataArray:
    index = array.get_index(dimension)
    if not index.is_unique:
        raise ValueError(f"Reduction coordinate '{dimension}' contains duplicates")

    required = target + window.offsets(delta_t)
    positions = index.get_indexer(required)
    if np.any(positions < 0):
        raise ValueError(
            f"Reduction coordinate '{dimension}' is missing required values: "
            f"{required[positions < 0]}"
        )
    return array.isel({dimension: positions})


def apply_time_statistic(
    array: xr.DataArray,
    modifier: str,
    target: TemporalTarget,
    delta_t: np.timedelta64,
    dimension: Hashable | None = None,
) -> xr.DataArray:
    """Select one temporal window and reduce an entire variable block.

    Parameters
    ----------
    array : xr.DataArray
        Data containing the source coordinates required by the statistic.
    modifier : str
        Statistic method and window, such as ``"mean:24h"``.
    target : np.datetime64 | np.timedelta64
        Scalar valid time or forecast lead time anchoring the window.
    delta_t : np.timedelta64
        Source cadence used to enumerate the required coordinates.
    dimension : Hashable, optional
        Temporal dimension to reduce. If omitted, ``lead_time`` then ``time`` is
        selected.

    Returns
    -------
    xr.DataArray
        Data reduced over the requested temporal window.
    """
    if dimension is None:
        try:
            dimension = next(
                name for name in ("lead_time", "time") if name in array.dims
            )
        except StopIteration as error:
            raise ValueError("Array has no 'time' or 'lead_time' dimension") from error
    if dimension not in array.dims:
        raise ValueError(f"Reduction dimension '{dimension}' is not present")

    if array.get_index(dimension).dtype.kind not in "Mm":
        raise TypeError(
            f"Reduction coordinate '{dimension}' must contain datetime or timedelta values"
        )
    window = _parse(modifier)
    selected = _select_time_window(array, window, target, delta_t, dimension)
    return _TIME_STATISTICS[window.method](selected, dimension)


def time_statistic_metadata(
    modifier: str,
) -> dict[str, str | np.timedelta64]:
    """Return the normalized metadata for one statistic."""
    window = _parse(modifier)
    return {
        "modifier": window.modifier,
        "method": window.method,
        "window": window.end - window.start,
        "start_offset": window.start,
        "end_offset": window.end,
        "closed": "left",
    }


def _mean(array: xr.DataArray, dimension: Hashable) -> xr.DataArray:
    return array.mean(dim=dimension, keep_attrs=True)


def _sum(array: xr.DataArray, dimension: Hashable) -> xr.DataArray:
    return array.sum(dim=dimension, keep_attrs=True)


def _minimum(array: xr.DataArray, dimension: Hashable) -> xr.DataArray:
    return array.min(dim=dimension, keep_attrs=True)


def _maximum(array: xr.DataArray, dimension: Hashable) -> xr.DataArray:
    return array.max(dim=dimension, keep_attrs=True)


register_time_statistic("mean", _mean)
register_time_statistic("sum", _sum)
register_time_statistic("min", _minimum)
register_time_statistic("max", _maximum)
