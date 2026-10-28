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

"""Connector: a frozen declaration feeding one component's fields into another."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import xarray as xr

from earth2studio.coupling.errors import CouplingError
from earth2studio.grids.base import Regridder


@dataclass(frozen=True)
class Connector:
    """Feed fields published by ``src`` into slot ``slot`` of ``dst``.

    A connector declares an edge; it holds no runtime state. The runner buffers
    what ``src`` publishes by valid time and, just before ``dst`` runs, selects
    or reduces the frames the slot needs, regrids and transforms them, and maps
    them onto the slot signature.

    Each slot label names a variable ``src`` exports. A qualified label such as
    ``"tp:sum:24h"`` is reduced from ``src``'s ``tp`` frames over the half-open
    window ``[T - 24h, T)`` ending at the required valid time ``T``.

    With ``state=True`` the connector instead overwrites variables of a stateful
    ``dst``'s own state: before each call, the variables ``variables`` names are
    replaced in the state slot ``slot``, an input slot whose output slot in the
    same position holds the same variables. ``initialize`` receives them in its
    inputs and each ``step`` in the output it advances, at those lead times;
    values ``src`` cannot supply at or before the item time are kept.

    Parameters
    ----------
    src : str
        Name of the producing component.
    dst : str
        Name of the consuming component.
    slot : int, optional
        Forcing slot of a stateful ``dst``, input slot of a diagnostic ``dst``,
        or state slot with ``state=True``, by default 0.
    variables : Mapping[str, str] | Sequence[str] | None, optional
        Renames from a ``dst`` base label to the ``src`` variable providing it,
        such as ``{"tp": "tp06"}``; a sequence keeps each name. Required with
        ``state=True``, where it names the variables overwritten. By default
        None.
    regridder : Regridder | None, optional
        Maps ``src``'s grid onto the slot's grid, by default None, which
        requires both grids to have the same geometry.
    transform : Callable[[xr.DataArray], xr.DataArray] | None, optional
        Applied to the delivered frames after regridding, before they are mapped
        onto the slot, for conversions no other component expresses, such as
        units or vertical levels. It receives the slot's labels at the required
        lead times and must keep the ``variable`` and ``lead_time`` coordinates.
        Frames fetched from the slot's source are not transformed. By default
        None.
    time : {"exact", "hold"}, optional
        ``"exact"`` needs frames at each required valid time and waits for them,
        so ``dst`` runs after ``src`` when both are due. ``"hold"`` takes the
        latest frame at or before each required valid time that ``src`` published
        on an earlier step, so the edge lags and breaks feedback loops. By
        default ``"exact"``.
    state : bool, optional
        Overwrite variables of ``dst``'s state instead of feeding a slot, by
        default False.
    """

    src: str
    dst: str
    slot: int = 0
    variables: Mapping[str, str] | Sequence[str] | None = field(
        default=None, hash=False
    )
    regridder: Regridder | None = field(default=None, hash=False)
    time: Literal["exact", "hold"] = "exact"
    transform: Callable[[xr.DataArray], xr.DataArray] | None = field(
        default=None, hash=False
    )
    state: bool = False

    def __post_init__(self) -> None:
        if not self.src or not self.dst:
            raise CouplingError("Connector src and dst must be component names")
        if self.src == self.dst:
            raise CouplingError(
                f"Connector {self.src!r} -> {self.dst!r} connects a component to itself"
            )
        if self.slot < 0:
            raise CouplingError(f"Connector slot must be non-negative, got {self.slot}")
        if self.time not in ("exact", "hold"):
            raise CouplingError(
                f"Connector time must be 'exact' or 'hold', got {self.time!r}"
            )
        if self.regridder is not None and not isinstance(self.regridder, Regridder):
            raise CouplingError("Connector regridder must implement Regridder")
        if self.transform is not None and not callable(self.transform):
            raise CouplingError("Connector transform must be callable")
        if isinstance(self.variables, str):
            raise CouplingError(
                f"Connector variables must be a mapping or a sequence of names, "
                f"got the string {self.variables!r}"
            )
        if self.variables is not None:
            renames = (
                dict(self.variables)
                if isinstance(self.variables, Mapping)
                else {str(v): str(v) for v in self.variables}
            )
            object.__setattr__(self, "variables", renames)
        if self.state and not self.variables:
            raise CouplingError(
                f"Connector {self.src!r} -> {self.dst!r} with state=True must name "
                "the state variables it overwrites in variables"
            )

    def source_variable(self, base: str) -> str:
        """Return the ``src`` variable providing ``dst`` base label ``base``."""
        renames = self.variables if isinstance(self.variables, Mapping) else {}
        return renames.get(base, base)

    def __str__(self) -> str:
        kind = ", state" if self.state else ""
        return f"{self.src} -> {self.dst}[{self.slot}] ({self.time}{kind})"


def _split_label(label: str) -> tuple[str, str | None]:
    """Split ``"tp:sum:24h"`` into ``("tp", "sum:24h")``; plain labels have no modifier."""
    base, separator, modifier = label.partition(":")
    return base, (modifier if separator else None)
