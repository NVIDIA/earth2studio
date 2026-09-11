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

"""PointSet: an irregular collection of (lat, lon) sample locations.

The "point" analog of :mod:`.vertical` — a component whose grid is a set of
scattered coordinates (stations, sites, asset locations, arbitrary query
points) rather than a lat/lon mesh. A component targeting points advertises
a ``"point"`` dim in its :meth:`Component.grid_coords` (an index or name
array, following the FieldDictionary convention that a coords value is the
dim's own coordinate labels) and carries the actual locations on
``Component.points``; the Connector's ``sample=`` path reads the latter to
build an auto sampler (see ``connector.py``).
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from earth2studio.utils.type import CoordSystem

from .errors import CouplingError


@dataclass(frozen=True)
class PointSet:
    """A fixed set of N sample locations.

    Parameters
    ----------
    lat, lon : np.ndarray [N]
        Latitude/longitude of each point, degrees.
    names : tuple[str, ...], optional
        Point identifiers (station IDs, site names, ...). Defaults to an
        integer index ``0..N-1`` when omitted, which is what the "point" dim
        coordinate carries in that case.
    """

    lat: np.ndarray
    lon: np.ndarray
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.lat, dtype=np.float64)
        lon = np.asarray(self.lon, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1:
            raise CouplingError(
                f"PointSet: lat/lon must be 1-D, got shapes {self.lat.shape} "
                f"and {self.lon.shape}"
            )
        if lat.shape != lon.shape:
            raise CouplingError(
                f"PointSet: lat and lon must have the same length, got "
                f"{lat.shape[0]} and {lon.shape[0]}"
            )
        if lat.shape[0] == 0:
            raise CouplingError("PointSet: at least one point is required")
        if self.names is not None and len(self.names) != lat.shape[0]:
            raise CouplingError(
                f"PointSet: names has {len(self.names)} entries but lat/lon "
                f"have {lat.shape[0]}"
            )
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)

    def __len__(self) -> int:
        return self.lat.shape[0]

    def labels(self) -> np.ndarray:
        """The 'point' dim's own coordinate array: names if given, else
        integer index — matches every other dim in a CoordSystem carrying
        its own labels."""
        if self.names is not None:
            return np.array(self.names)
        return np.arange(len(self))

    def grid_coords(self) -> CoordSystem:
        """A one-entry CoordSystem, the point-grid analog of a lat/lon
        mesh's coords, for :meth:`Component.grid_coords`."""
        return OrderedDict({"point": self.labels()})

    def signature(self) -> tuple:
        """Hashable signature for sampler caching (mirrors
        Field.grid_signature)."""
        return (self.lat.shape, self.lat.tobytes(), self.lon.tobytes())
