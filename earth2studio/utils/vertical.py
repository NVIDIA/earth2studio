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

"""Vertical coordinates and interpolation onto pressure levels.

Earth2Studio fields carry vertical levels in their variable names, such as
``t850`` on pressure levels or ``t12k`` on model level 12, so the vertical is
handled separately from the horizontal grid. Interpolation is linear in the
logarithm of pressure, one column at a time; hybrid levels need the surface
pressure at the same valid time. See
:class:`~earth2studio.models.dx.DerivedPressureLevels` for a diagnostic over
lexicon-named fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PressureLevels:
    """Isobaric levels.

    Parameters
    ----------
    levels : Sequence[float]
        Pressure of each level in hPa, in any order.
    """

    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        levels = tuple(float(level) for level in self.levels)
        if not levels or min(levels) <= 0:
            raise ValueError("Pressure levels must be positive")
        if len(set(levels)) != len(levels):
            raise ValueError(f"Pressure levels {levels} repeat a level")
        object.__setattr__(self, "levels", levels)

    def __len__(self) -> int:
        return len(self.levels)

    def pressure(self) -> torch.Tensor:
        """Return the pressure of each level in Pa, shape ``(level,)``."""
        return torch.tensor(self.levels, dtype=torch.float64) * 100.0


@dataclass(frozen=True)
class HybridLevels:
    """Hybrid sigma-pressure levels, ``p_k = a_k + b_k * p_s``.

    Parameters
    ----------
    a : Sequence[float]
        Pressure coefficient of each level in Pa, from the top of the model down.
    b : Sequence[float]
        Dimensionless surface-pressure coefficient of each level.
    """

    a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        a = tuple(float(value) for value in self.a)
        b = tuple(float(value) for value in self.b)
        if not a or len(a) != len(b):
            raise ValueError("Hybrid coefficients a and b must have equal length")
        for surface in (50000.0, 110000.0):
            pressure = np.asarray(a) + np.asarray(b) * surface
            if np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0):
                raise ValueError(
                    "Hybrid coefficients must give positive pressure increasing "
                    "from the top of the model down"
                )
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @classmethod
    def from_interfaces(cls, a: Sequence[float], b: Sequence[float]) -> HybridLevels:
        """Return the full levels midway between interface coefficients.

        Parameters
        ----------
        a : Sequence[float]
            Pressure coefficients of the ``n + 1`` layer interfaces in Pa.
        b : Sequence[float]
            Surface-pressure coefficients of the interfaces.

        Returns
        -------
        HybridLevels
            The ``n`` full levels.
        """
        a_, b_ = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return cls(tuple((a_[1:] + a_[:-1]) / 2), tuple((b_[1:] + b_[:-1]) / 2))

    def __len__(self) -> int:
        return len(self.a)

    def pressure(self, surface_pressure: torch.Tensor) -> torch.Tensor:
        """Return the pressure of each level in Pa.

        Parameters
        ----------
        surface_pressure : torch.Tensor
            Surface pressure in Pa, any shape.

        Returns
        -------
        torch.Tensor
            Pressure with a trailing ``level`` dimension,
            ``(*surface_pressure.shape, level)``.
        """
        options = {"dtype": surface_pressure.dtype, "device": surface_pressure.device}
        a = torch.tensor(self.a, **options)  # type: ignore[arg-type]
        b = torch.tensor(self.b, **options)  # type: ignore[arg-type]
        return a + b * surface_pressure.unsqueeze(-1)


def interpolate_pressure(
    x: torch.Tensor,
    pressure: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1,
    fill_value: float | None = None,
) -> torch.Tensor:
    """Interpolate ``x`` linearly in log pressure onto ``target`` levels.

    Parameters
    ----------
    x : torch.Tensor
        Field with a level dimension ``dim``.
    pressure : torch.Tensor
        Pressure of each of ``x``'s levels, in Pa: either one value per level,
        shape ``(level,)`` in any order, or one column per point, shaped like
        ``x`` (or broadcastable to it) and increasing along ``dim``.
    target : torch.Tensor
        Pressure of each output level in Pa, shape ``(target,)``.
    dim : int, optional
        Level dimension of ``x`` and ``pressure``, by default -1
    fill_value : float | None, optional
        Value for targets outside a column's pressure range, by default None,
        which takes the nearest level's value instead.

    Returns
    -------
    torch.Tensor
        ``x`` with dimension ``dim`` replaced by the target levels.
    """
    if x.shape[dim] < 2:
        raise ValueError("Interpolation needs at least two source levels")
    values = x.movedim(dim, -1)
    log_target = torch.log(target.to(values))
    if pressure.ndim == 1:
        if len(pressure) != values.shape[-1]:
            raise ValueError(
                f"{len(pressure)} pressures given for {values.shape[-1]} levels"
            )
        log_pressure, order = torch.sort(torch.log(pressure.to(values)))
        values = values[..., order]
        high = torch.searchsorted(log_pressure, log_target)
        high = high.clamp(1, len(log_pressure) - 1)
        low_pressure, high_pressure = log_pressure[high - 1], log_pressure[high]
        low_value, high_value = values[..., high - 1], values[..., high]
        bottom, top = log_pressure[-1], log_pressure[0]
    else:
        log_pressure = torch.log(pressure.to(values).movedim(dim, -1))
        log_pressure = log_pressure.expand(values.shape).contiguous()
        wanted = log_target.expand(*values.shape[:-1], len(target)).contiguous()
        high = torch.searchsorted(log_pressure, wanted)
        high = high.clamp(1, values.shape[-1] - 1)
        low_pressure = torch.gather(log_pressure, -1, high - 1)
        high_pressure = torch.gather(log_pressure, -1, high)
        low_value = torch.gather(values, -1, high - 1)
        high_value = torch.gather(values, -1, high)
        bottom, top = log_pressure[..., -1:], log_pressure[..., :1]
    weight = (log_target - low_pressure) / (high_pressure - low_pressure)
    if fill_value is None:
        weight = weight.clamp(0, 1)
    output = low_value + weight * (high_value - low_value)
    if fill_value is not None:
        outside = (log_target < top) | (log_target > bottom)
        output = output.masked_fill(outside, fill_value)
    return output.movedim(-1, dim)


__all__ = ["HybridLevels", "PressureLevels", "interpolate_pressure"]
