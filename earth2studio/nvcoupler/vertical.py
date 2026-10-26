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

"""Vertical-coordinate descriptors and pressure interpolation for exchanges.

Most Earth2Studio models encode levels in variable names (``z500``, ``t850``)
and never need this. Models that exchange a field with a ``level`` dimension,
such as chemistry emulators on hybrid sigma-pressure levels, interpolate it
onto the destination's pressure levels with a connector ``transform`` built by
:func:`to_pressure_levels`. Interpolation is linear in log pressure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.nvcoupler.connector import Transform
from earth2studio.nvcoupler.errors import CouplingError

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU installations
    cp = None


@dataclass(frozen=True)
class PressureLevels:
    """Constant pressure levels in hPa, strictly increasing from top to bottom."""

    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        levels = tuple(float(level) for level in self.levels)
        object.__setattr__(self, "levels", levels)
        if not levels or any(level <= 0 for level in levels):
            raise ValueError("Pressure levels must be positive")
        if np.any(np.diff(levels) <= 0):
            raise ValueError(
                "Pressure levels must be strictly increasing (top to bottom)"
            )

    def pressure_pa(self) -> np.ndarray:
        """Return the levels in Pa."""
        return np.asarray(self.levels, dtype=np.float64) * 100.0

    def __len__(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class HybridLevels:
    """Hybrid sigma-pressure levels ``p_k = a_k + b_k * p_s`` (Pa), top to bottom."""

    a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        a = tuple(float(value) for value in self.a)
        b = tuple(float(value) for value in self.b)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        if not a or len(a) != len(b):
            raise ValueError("Hybrid coefficients a and b must be equally long")
        for surface_pressure in (50000.0, 110000.0):
            pressure = np.asarray(a) + np.asarray(b) * surface_pressure
            if np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0):
                raise ValueError(
                    "Hybrid coefficients must give positive, strictly increasing "
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
    source: VerticalCoordinate,
    target: PressureLevels,
    surface_pressure: xr.DataArray | float | None = None,
) -> xr.DataArray:
    """Interpolate a field with a ``level`` dimension onto pressure levels.

    Linear in log pressure; targets beyond the source column take the end
    values. The ``level`` coordinate of the result holds ``target.levels``.

    Parameters
    ----------
    array : xr.DataArray
        Field with a ``level`` dimension of length ``len(source)``.
    source : PressureLevels | HybridLevels
        Vertical coordinate of ``array``. Pressure levels must match its
        ``level`` coordinate.
    target : PressureLevels
        Levels to interpolate onto.
    surface_pressure : xr.DataArray | float, optional
        Surface pressure in Pa, required for hybrid sources: a field
        broadcastable to ``array`` without ``level``, or one value.

    Returns
    -------
    xr.DataArray
        ``array`` on ``target`` levels, same dimensions and backing.

    Raises
    ------
    CouplingError
        If ``array`` has no ``level`` dimension, its length or coordinate does
        not match ``source``, or a hybrid source lacks surface pressure.
    """
    if "level" not in array.dims:
        raise CouplingError(f"Field has no 'level' dimension ({array.dims})")
    if array.sizes["level"] != len(source):
        raise CouplingError(
            f"Field has {array.sizes['level']} levels but the source coordinate "
            f"declares {len(source)}"
        )
    axis = array.get_axis_num("level")
    data = array.data
    xp = _namespace(data)
    values = xp.moveaxis(data, axis, -1)
    if isinstance(source, PressureLevels):
        levels = np.asarray(array.coords["level"], dtype=np.float64)
        if not np.allclose(levels, source.levels):
            raise CouplingError(
                "Field 'level' coordinate does not match the declared source levels"
            )
        if source.levels == target.levels:
            return array
        pressure = xp.broadcast_to(
            xp.asarray(source.pressure_pa(), dtype=values.dtype), values.shape
        )
    else:
        if surface_pressure is None:
            raise CouplingError(
                "Hybrid-level interpolation needs surface_pressure (Pa)"
            )
        a = xp.asarray(source.a, dtype=values.dtype)
        b = xp.asarray(source.b, dtype=values.dtype)
        if isinstance(surface_pressure, xr.DataArray):
            column = array.isel(level=0, drop=True)
            try:
                ps = surface_pressure.broadcast_like(column).transpose(*column.dims)
            except ValueError as error:
                raise CouplingError(
                    "surface_pressure is not broadcastable to the field"
                ) from error
            pressure = a + b * xp.asarray(ps.data, dtype=values.dtype)[..., None]
        else:
            pressure = xp.broadcast_to(a + b * float(surface_pressure), values.shape)
        if bool(xp.any(pressure <= 0)) or bool(
            xp.any(pressure[..., 1:] <= pressure[..., :-1])
        ):
            raise CouplingError(
                "Hybrid levels are not positive and strictly increasing for the "
                "given surface pressure"
            )

    log_source = xp.log(pressure)
    log_target = xp.log(xp.asarray(target.pressure_pa(), dtype=values.dtype))
    wanted = xp.broadcast_to(log_target, (*values.shape[:-1], len(target)))
    high = xp.sum(log_source[..., None, :] < wanted[..., :, None], axis=-1)
    high = xp.clip(high, 1, values.shape[-1] - 1)
    low = high - 1
    low_value = xp.take_along_axis(values, low, axis=-1)
    high_value = xp.take_along_axis(values, high, axis=-1)
    low_pressure = xp.take_along_axis(log_source, low, axis=-1)
    high_pressure = xp.take_along_axis(log_source, high, axis=-1)
    weight = xp.clip((wanted - low_pressure) / (high_pressure - low_pressure), 0, 1)
    output = xp.moveaxis(low_value * (1 - weight) + high_value * weight, -1, axis)
    coords = {
        name: coordinate
        for name, coordinate in array.coords.items()
        if "level" not in coordinate.dims
    }
    coords["level"] = np.asarray(target.levels, dtype=np.float64)
    return xr.DataArray(
        output, dims=array.dims, coords=coords, attrs=dict(array.attrs), name=array.name
    )


def to_pressure_levels(
    source: VerticalCoordinate,
    target: PressureLevels | Sequence[float],
    surface_pressure: xr.DataArray | float | None = None,
) -> Transform:
    """Return a connector ``transform`` interpolating onto pressure levels.

    Parameters
    ----------
    source : PressureLevels | HybridLevels
        Vertical coordinate of the source model's field.
    target : PressureLevels | Sequence[float]
        Destination levels in hPa.
    surface_pressure : xr.DataArray | float, optional
        Surface pressure in Pa for hybrid sources. A static field or constant;
        a time-varying surface pressure is a model-specific transform.

    Returns
    -------
    Callable[[xr.DataArray], xr.DataArray]
        ``transform`` for :class:`~earth2studio.nvcoupler.Connector`.
    """
    levels = (
        target if isinstance(target, PressureLevels) else PressureLevels(tuple(target))
    )

    def transform(array: xr.DataArray) -> xr.DataArray:
        return interp_to_pressure(array, source, levels, surface_pressure)

    return transform
