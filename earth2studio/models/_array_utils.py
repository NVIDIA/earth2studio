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

"""Internal domain resolution for DataArray model signatures."""

from typing import Any

from earth2studio.grids import (
    CurvilinearGrid,
    GridDefinition,
    LatLonGrid,
    list_grids,
    resolve_grid,
)
from earth2studio.utils.type import CoordSystem


def _registered_grid(grid: str | GridDefinition) -> str | GridDefinition:
    """Reuse a registered name only when geometry and metadata match exactly."""
    if isinstance(grid, str):
        return grid
    for name in list_grids():
        registered = resolve_grid(name)
        if (
            grid.dims == registered.dims
            and grid.topology == registered.topology
            and grid.fingerprint() == registered.fingerprint()
            and grid.crs == registered.crs
            and grid.attrs == registered.attrs
            and grid.coords().to_dataset().identical(registered.coords().to_dataset())
        ):
            return name
    return grid


def _resolve_domain(
    domain: CoordSystem | GridDefinition | str,
) -> tuple[tuple[str, ...], dict[str, Any], str | GridDefinition | None]:
    """Resolve legacy domain maps or grids for direct coordinate-array construction."""
    if not isinstance(domain, (str, GridDefinition)):
        if tuple(domain) != ("lat", "lon"):
            return tuple(domain), dict(domain), None
        domain = (
            LatLonGrid(domain["lat"], domain["lon"])
            if domain["lat"].ndim == 1
            else CurvilinearGrid(domain["lat"], domain["lon"])
        )
    grid = _registered_grid(domain)
    definition = resolve_grid(grid) if isinstance(grid, str) else grid
    return definition.dims, {}, grid
