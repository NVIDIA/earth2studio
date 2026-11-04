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

"""Frames one component has published during a work item, keyed by lead time."""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class _Entry:
    variables: frozenset[str]
    ordinal: int  # scheduler step that published the frame; 0 is initialization
    frame: xr.DataArray | None  # None during planning


class _Buffer:
    """One producer's frames keyed by lead time in nanoseconds from the item time.

    Planning runs store variable names only; execution also stores each frame, a
    DataArray with a single ``lead_time`` entry on the producer's grid.
    """

    def __init__(self) -> None:
        self._entries: dict[int, list[_Entry]] = {}

    def add(
        self,
        lead: int,
        variables: frozenset[str],
        ordinal: int,
        frame: xr.DataArray | None = None,
    ) -> None:
        """Record a frame valid at ``lead``."""
        self._entries.setdefault(lead, []).append(_Entry(variables, ordinal, frame))

    def _find(self, variable: str, lead: int) -> _Entry | None:
        for entry in self._entries.get(lead, ()):
            if variable in entry.variables:
                return entry
        return None

    def has(self, variable: str, lead: int) -> bool:
        """Whether a frame holding ``variable`` is valid at ``lead``."""
        return self._find(variable, lead) is not None

    def latest(self, variable: str, lead: int, before: int) -> int | None:
        """Latest lead at or before ``lead`` holding ``variable``, published before ``before``."""
        found = [
            key
            for key, entries in self._entries.items()
            if key <= lead
            and any(variable in e.variables and e.ordinal < before for e in entries)
        ]
        return max(found) if found else None

    def get(self, variable: str, lead: int) -> xr.DataArray:
        """Return ``variable`` at ``lead`` with its ``variable`` dimension kept."""
        entry = self._find(variable, lead)
        if entry is None or entry.frame is None:
            raise KeyError(f"No frame for {variable!r} at lead {lead}")
        return entry.frame.sel(variable=[variable])

    def prune(self, before: int) -> None:
        """Drop frames valid before ``before``, keeping each variable's latest frame."""
        keep: set[int] = set()
        latest: dict[str, int] = {}
        for key, entries in self._entries.items():
            for entry in entries:
                for variable in entry.variables:
                    latest[variable] = max(latest.get(variable, key), key)
        keep.update(latest.values())
        for key in [k for k in self._entries if k < before and k not in keep]:
            del self._entries[key]
