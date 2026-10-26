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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from earth2studio.coupling.errors import CouplingError
from earth2studio.grids.base import Regridder


@dataclass(frozen=True)
class Connector:
    """Feed fields published by ``src`` into slot ``slot`` of ``dst``.

    A connector declares an edge; it holds no runtime state. The runner buffers
    what ``src`` publishes by valid time and, just before ``dst`` runs, selects
    or reduces the frames the slot needs, regrids them, and maps them onto the
    slot signature.

    Each slot label names a variable ``src`` exports. A qualified label such as
    ``"tp:sum:24h"`` is reduced from ``src``'s ``tp`` frames over the half-open
    window ``[T - 24h, T)`` ending at the required valid time ``T``.

    Parameters
    ----------
    src : str
        Name of the producing component.
    dst : str
        Name of the consuming component.
    slot : int, optional
        Forcing slot of a stateful ``dst``, or input slot of a diagnostic
        ``dst``, by default 0.
    variables : Mapping[str, str] | None, optional
        Renames from a ``dst`` base label to the ``src`` variable providing it,
        such as ``{"tp": "tp06"}``, by default None.
    regridder : Regridder | None, optional
        Maps ``src``'s grid onto the slot's grid, by default None, which
        requires both grids to have the same geometry.
    time : {"exact", "hold"}, optional
        ``"exact"`` needs frames at each required valid time and waits for them,
        so ``dst`` runs after ``src`` when both are due. ``"hold"`` takes the
        latest frame at or before each required valid time that ``src`` published
        on an earlier step, so the edge lags and breaks feedback loops. By
        default ``"exact"``.
    """

    src: str
    dst: str
    slot: int = 0
    variables: Mapping[str, str] | None = field(default=None, hash=False)
    regridder: Regridder | None = field(default=None, hash=False)
    time: Literal["exact", "hold"] = "exact"

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
        if self.variables is not None:
            object.__setattr__(self, "variables", dict(self.variables))

    def source_variable(self, base: str) -> str:
        """Return the ``src`` variable providing ``dst`` base label ``base``."""
        return (self.variables or {}).get(base, base)

    def __str__(self) -> str:
        return f"{self.src} -> {self.dst}[{self.slot}] ({self.time})"


def _split_label(label: str) -> tuple[str, str | None]:
    """Split ``"tp:sum:24h"`` into ``("tp", "sum:24h")``; plain labels have no modifier."""
    base, separator, modifier = label.partition(":")
    return base, (modifier if separator else None)
