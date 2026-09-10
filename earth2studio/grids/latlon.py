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

"""Rectilinear latitude-longitude grids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS

from earth2studio.grids._utils import (
    array,
    coordinate_hash,
    geographic_subset_indexers,
    metadata,
)
from earth2studio.grids.base import GridTopology


@dataclass(frozen=True)
class LatLonGrid:
    """Define a rectilinear latitude-longitude grid."""

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    coordinate_reference_system: Any = "EPSG:4326"

    def __post_init__(self) -> None:
        object.__setattr__(self, "latitude", array(self.latitude, 1, "latitude"))
        object.__setattr__(self, "longitude", array(self.longitude, 1, "longitude"))
        object.__setattr__(
            self,
            "coordinate_reference_system",
            CRS.from_user_input(self.coordinate_reference_system),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        return ("lat", "lon")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.latitude.size, self.longitude.size)

    @property
    def topology(self) -> GridTopology:
        return "rectilinear"

    @property
    def crs(self) -> CRS:
        return self.coordinate_reference_system

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return latitude and longitude coordinates."""
        indexes = indexes or {"lat": self.latitude, "lon": self.longitude}
        return xr.Coordinates({"lat": indexes["lat"], "lon": indexes["lon"]})

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate geographic bounds into indexers."""
        return geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell bounds."""
        return None

    @property
    def attrs(self) -> dict[str, Any]:
        """Return serializable grid attributes."""
        return metadata(self)

    def fingerprint(self) -> str:
        """Return stable geometry identity."""
        return coordinate_hash(self.latitude, self.longitude) + self.crs.to_wkt()
