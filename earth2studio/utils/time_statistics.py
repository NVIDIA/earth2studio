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

_DURATION_PATTERN = re.compile(r"^([+-]?)(\d+)(ns|us|ms|min|m|s|h|d)$")
_DURATION_UNITS = {"min": "m", "m": "m", "d": "D"}
_TIME_STATISTICS: dict[str, TimeReduction] = {}


def _duration(value: Duration, *, name: str) -> np.timedelta64:
    if isinstance(value, np.timedelta64):
        duration = value
    elif isinstance(value, str):
        match = _DURATION_PATTERN.fullmatch(value.strip().lower())
        if match is None:
            raise ValueError(f"{name} must be an integer duration such as '6h'")
        sign, magnitude, unit = match.groups()
        duration = np.timedelta64(
            (-1 if sign == "-" else 1) * int(magnitude),
            _DURATION_UNITS.get(unit, unit),
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


def _iso_duration(value: np.timedelta64) -> str:
    nanoseconds = _nanoseconds(value)
    sign = "-" if nanoseconds < 0 else ""
    magnitude = abs(nanoseconds)
    for suffix, scale in (
        ("H", 3_600_000_000_000),
        ("M", 60_000_000_000),
        ("S", 1_000_000_000),
    ):
        if magnitude and magnitude % scale == 0:
            return f"{sign}PT{magnitude // scale}{suffix}"
    return f"{sign}PT{magnitude / 1_000_000_000:g}S" if magnitude else "PT0S"


@dataclass(frozen=True)
class _Window:
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
    if statistics is None:
        return {}
    if isinstance(statistics, str):
        return {_parse(statistics).modifier: labels}
    unknown = set(statistics) - set(labels)
    if unknown:
        raise ValueError(f"Statistics reference unknown variables: {sorted(unknown)}")
    groups: dict[str, list[str]] = {}
    for variable in labels:
        if variable in statistics:
            modifier = _parse(statistics[variable]).modifier
            groups.setdefault(modifier, []).append(variable)
    return {modifier: tuple(group) for modifier, group in groups.items()}


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


def apply_time_statistic(
    array: xr.DataArray, modifier: str, dimension: Hashable | None = None
) -> xr.DataArray:
    """Apply one statistic to an entire variable block."""
    if dimension is None:
        dimension = next(
            (name for name in ("lead_time", "time") if name in array.dims), None
        )
        if dimension is None:
            raise ValueError("Array has no 'time' or 'lead_time' dimension")
    if dimension not in array.dims:
        raise ValueError(f"Reduction dimension '{dimension}' is not present")
    window = _parse(modifier)
    return _TIME_STATISTICS[window.method](array, dimension)


def time_statistic_metadata(modifier: str) -> dict[str, str]:
    """Return a serializable description of one statistic."""
    window = _parse(modifier)
    return {
        "modifier": window.modifier,
        "method": window.method,
        "window": _iso_duration(window.end - window.start),
        "start_offset": _iso_duration(window.start),
        "end_offset": _iso_duration(window.end),
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
