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

"""Grid protocol and Earth2Studio metadata keys."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS

E2S_CRS = "earth2studio_crs"
E2S_GRID_ID = "earth2studio_grid_id"

GridTopology = Literal["rectilinear", "projected", "curvilinear", "healpix", "points"]


# sphinx - grid protocol start
@runtime_checkable
class GridDefinition(Protocol):
    """Describe spatial geometry through a structural interface."""

    @property
    def dims(self) -> tuple[str, ...]:
        """Return ordered spatial dimensions."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the spatial shape."""
        ...

    @property
    def topology(self) -> GridTopology:
        """Return the grid topology."""
        ...

    @property
    def crs(self) -> CRS | None:
        """Return the native coordinate reference system, if defined."""
        ...

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return complete coordinates or only dimension indexes."""
        ...

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate a selection into positional dimension indexers."""
        ...

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> xr.Coordinates | None:
        """Return geographic cell boundaries when available."""
        ...

    @property
    def attrs(self) -> dict[str, Any]:
        """Return serializable Xarray attributes."""
        ...

    def fingerprint(self) -> str:
        """Return a stable geometry fingerprint."""
        ...


# sphinx - grid protocol end
