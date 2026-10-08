# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Vertical-coordinate descriptors and DataArray pressure interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .errors import VerticalMismatchError

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


@dataclass(frozen=True)
class PressureLevels:
    """Constant pressure levels in hPa, ordered top to bottom."""

    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if list(self.levels) != sorted(self.levels):
            raise ValueError("Pressure levels must be increasing (top to bottom)")

    def pressure_pa(self) -> np.ndarray:
        return np.asarray(self.levels, dtype=np.float64) * 100.0


@dataclass(frozen=True)
class HybridLevels:
    """Hybrid levels ``p_k = a_k + b_k * p_s``."""

    a: tuple[float, ...]
    b: tuple[float, ...]
    ps_field: str = "surface_pressure"

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b):
            raise ValueError("Hybrid coefficients a and b must have equal length")
        for surface_pressure in (50000.0, 110000.0):
            pressure = np.asarray(self.a) + np.asarray(self.b) * surface_pressure
            if np.any(np.diff(pressure) <= 0):
                raise ValueError(
                    "Hybrid coefficients must produce strictly increasing "
                    "pressure from top to bottom"
                )

    def __len__(self) -> int:
        return len(self.a)


VerticalCoordinate = PressureLevels | HybridLevels


def _namespace(data: Any) -> Any:
    if cp is not None and isinstance(data, cp.ndarray):
        return cp
    return np


def interp_to_pressure(
    array: xr.DataArray,
    src: VerticalCoordinate,
    dst: PressureLevels,
    surface_pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Interpolate a DataArray linearly in log pressure."""
    if "level" not in array.dims:
        raise VerticalMismatchError(
            f"interp_to_pressure: array has no 'level' dimension ({array.dims})"
        )
    if isinstance(src, PressureLevels):
        levels = np.asarray(array.coords["level"], dtype=np.float64)
        declared = np.asarray(src.levels, dtype=np.float64)
        if levels.shape != declared.shape or not np.allclose(levels, declared):
            raise VerticalMismatchError(
                "DataArray level coordinate does not match declared source levels"
            )
        if src.levels == dst.levels:
            return array

    axis = array.get_axis_num("level")
    data = array.data
    xp = _namespace(data)
    values = xp.moveaxis(data, axis, -1)
    if isinstance(src, PressureLevels):
        pressure = xp.asarray(src.pressure_pa(), dtype=values.dtype)
        pressure = xp.broadcast_to(pressure, values.shape)
    else:
        if surface_pressure is None:
            raise VerticalMismatchError(
                f"Hybrid interpolation requires {src.ps_field!r}"
            )
        target = array.isel(level=0, drop=True)
        try:
            ps = surface_pressure.broadcast_like(target).transpose(*target.dims).data
        except ValueError as error:
            raise VerticalMismatchError(
                "Surface pressure is not aligned with the field"
            ) from error
        pressure = (
            xp.asarray(src.a, dtype=values.dtype)
            + xp.asarray(src.b, dtype=values.dtype) * ps[..., None]
        )
        if bool(xp.any(pressure <= 0)) or bool(
            xp.any(pressure[..., 1:] <= pressure[..., :-1])
        ):
            raise VerticalMismatchError(
                "Hybrid levels are not positive and strictly increasing"
            )

    log_source = xp.log(pressure)
    log_target = xp.log(xp.asarray(dst.pressure_pa(), dtype=values.dtype))
    target = xp.broadcast_to(log_target, (*values.shape[:-1], len(dst.levels)))
    high = xp.sum(log_source[..., None, :] < target[..., :, None], axis=-1)
    high = xp.clip(high, 1, values.shape[-1] - 1)
    low = high - 1
    low_value = xp.take_along_axis(values, low, axis=-1)
    high_value = xp.take_along_axis(values, high, axis=-1)
    low_pressure = xp.take_along_axis(log_source, low, axis=-1)
    high_pressure = xp.take_along_axis(log_source, high, axis=-1)
    weight = xp.clip((target - low_pressure) / (high_pressure - low_pressure), 0.0, 1.0)
    output = low_value * (1 - weight) + high_value * weight
    output = xp.moveaxis(output, -1, axis)
    coords = {
        name: coordinate
        for name, coordinate in array.coords.items()
        if "level" not in coordinate.dims
    }
    coords["level"] = np.asarray(dst.levels, dtype=np.float64)
    return xr.DataArray(
        output,
        dims=array.dims,
        coords=coords,
        attrs=dict(array.attrs),
        name=array.name,
    )
