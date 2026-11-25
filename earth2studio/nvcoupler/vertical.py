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

"""Vertical coordinate descriptors and differentiable level interpolation.

Most earth2studio models encode pressure levels in variable names (z500,
t850) and never see this module. It exists for components with an explicit
"level" dimension — chiefly chemistry emulators on hybrid sigma-pressure
model levels (p_k = a_k + b_k * p_s) coupled to met components on pressure
levels. Interpolation is linear in log-pressure via torch.searchsorted and
gathers, so gradients flow through it (values, not indices).
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem

from .errors import VerticalMismatchError


@dataclass(frozen=True)
class PressureLevels:
    """Constant pressure levels in hPa, ordered top to bottom (increasing)."""

    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if list(self.levels) != sorted(self.levels):
            raise ValueError("Pressure levels must be increasing (top to bottom)")

    def pressure_pa(self) -> np.ndarray:
        return np.asarray(self.levels, dtype=np.float64) * 100.0


@dataclass(frozen=True)
class HybridLevels:
    """Hybrid sigma-pressure levels: p_k = a_k + b_k * p_s.

    `a` in Pa, `b` dimensionless, ordered top to bottom. `ps_field` names the
    surface-pressure field (Pa) required to realize the levels; a Connector
    performing a hybrid->pressure transform takes it from the source
    component's exports automatically.
    """

    a: tuple[float, ...]
    b: tuple[float, ...]
    ps_field: str = "surface_pressure"

    def __post_init__(self) -> None:
        if len(self.a) != len(self.b):
            raise ValueError("Hybrid coefficients a and b must have equal length")
        # Levels must be strictly increasing in pressure (top to bottom) for
        # every plausible surface pressure, else interpolation would pair
        # level slices with wrong pressures. p_k(ps) = a_k + b_k * ps is
        # linear in ps, so strict monotonicity at both ends of the plausible
        # Earth surface-pressure range [50000, 110000] Pa (high terrain to
        # strong anticyclone) is sufficient for every ps inside that range;
        # ps values outside it are re-checked at interpolation time.
        for ps in (50000.0, 110000.0):
            p = (
                np.asarray(self.a, dtype=np.float64)
                + np.asarray(self.b, dtype=np.float64) * ps
            )
            if np.any(np.diff(p) <= 0):
                raise ValueError(
                    f"Hybrid coefficients a={list(self.a)}, b={list(self.b)} "
                    f"produce non-increasing pressures {p.tolist()} Pa at "
                    f"surface pressure {ps:.0f} Pa. Order a and b top to "
                    "bottom so p_k = a_k + b_k * ps strictly increases for "
                    "all surface pressures in [50000, 110000] Pa."
                )

    def __len__(self) -> int:
        return len(self.a)


VerticalCoordinate = PressureLevels | HybridLevels


def _log_source_pressure(
    vertical: VerticalCoordinate,
    ps: torch.Tensor | None,
    like: torch.Tensor,
) -> torch.Tensor:
    """Log source pressure with shape (..., L) broadcastable to `like`
    (which has the level axis moved to last)."""
    if isinstance(vertical, PressureLevels):
        p = torch.as_tensor(
            vertical.pressure_pa(), dtype=like.dtype, device=like.device
        )
        return torch.log(p).expand(like.shape)
    if ps is None:
        raise VerticalMismatchError(
            f"Hybrid->pressure interpolation requires the surface pressure "
            f"field {vertical.ps_field!r}, which was not available"
        )
    a = torch.as_tensor(vertical.a, dtype=like.dtype, device=like.device)
    b = torch.as_tensor(vertical.b, dtype=like.dtype, device=like.device)
    p = a + b * ps.to(dtype=like.dtype, device=like.device).unsqueeze(-1)
    if torch.any(p <= 0):
        raise VerticalMismatchError("Non-positive pressure from hybrid coefficients")
    if torch.any(p[..., 1:] <= p[..., :-1]):
        raise VerticalMismatchError(
            f"Hybrid levels a + b * ps are not strictly increasing along the "
            f"level axis for the given surface pressure field "
            f"{vertical.ps_field!r} — interpolation would pair level slices "
            "with wrong pressures. Check that the hybrid coefficients a and b "
            "are ordered top to bottom and that the surface pressure values "
            "are physical (Pa)."
        )
    return torch.log(p).expand(like.shape)


def interp_to_pressure(
    x: torch.Tensor,
    coords: CoordSystem,
    src: VerticalCoordinate,
    dst: PressureLevels,
    ps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, CoordSystem]:
    """Interpolate a field with a "level" dim onto constant pressure levels.

    Linear in log-pressure; clamped to the source column ends (no
    extrapolation beyond top/bottom values). `ps` (Pa) must broadcast to the
    field with the level dim removed and is required for hybrid sources. For
    :class:`PressureLevels` sources the data's ``level`` coordinate (hPa)
    must match `src.levels` exactly, so the level axis is guaranteed to be
    paired with the declared pressures.
    """
    if "level" not in coords:
        raise VerticalMismatchError(
            f"interp_to_pressure: coords have no 'level' dim ({list(coords)})"
        )
    if isinstance(src, PressureLevels):
        lev = np.asarray(coords["level"], dtype=np.float64)
        src_lev = np.asarray(src.levels, dtype=np.float64)
        if lev.shape != src_lev.shape or not np.allclose(lev, src_lev):
            raise VerticalMismatchError(
                f"interp_to_pressure: data 'level' coordinate {lev.tolist()} "
                f"does not match the declared PressureLevels source "
                f"{list(src.levels)} (hPa). Reorder the data so levels "
                "increase top to bottom, or fix the source component's "
                "export_vertical declaration to match its 'level' coordinate."
            )
    if isinstance(src, PressureLevels) and tuple(src.levels) == tuple(dst.levels):
        return x, coords
    lev_axis = list(coords).index("level")
    n_src = len(coords["level"])
    xp = x.movedim(lev_axis, -1)  # (..., L)
    if n_src != xp.shape[-1]:
        raise VerticalMismatchError(
            f"'level' coord length {n_src} != tensor level size {xp.shape[-1]}"
        )
    logp_src = _log_source_pressure(src, ps, xp)
    logp_dst = torch.log(
        torch.as_tensor(dst.pressure_pa(), dtype=xp.dtype, device=xp.device)
    ).expand(*xp.shape[:-1], len(dst.levels))

    idx_hi = torch.searchsorted(logp_src.contiguous(), logp_dst.contiguous())
    idx_hi = idx_hi.clamp(1, xp.shape[-1] - 1)
    idx_lo = idx_hi - 1
    x_lo = torch.gather(xp, -1, idx_lo)
    x_hi = torch.gather(xp, -1, idx_hi)
    p_lo = torch.gather(logp_src, -1, idx_lo)
    p_hi = torch.gather(logp_src, -1, idx_hi)
    w = ((logp_dst - p_lo) / (p_hi - p_lo)).clamp(0.0, 1.0)
    out = x_lo * (1.0 - w) + x_hi * w

    out = out.movedim(-1, lev_axis)
    new_coords = OrderedDict(coords)
    new_coords["level"] = np.asarray(dst.levels, dtype=np.float64)
    return out, new_coords
