# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured projected grids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from earth2studio.grids._utils import (
    array,
    coordinate_hash,
    geographic_subset_indexers,
    metadata,
)
from earth2studio.grids.base import GridTopology


@dataclass(frozen=True)
class ProjectedGrid:
    """Define a grid from one-dimensional projected coordinates and a CRS."""

    y: NDArray[Any]
    x: NDArray[Any]
    coordinate_reference_system: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", array(self.y, 1, "y"))
        object.__setattr__(self, "x", array(self.x, 1, "x"))
        object.__setattr__(
            self,
            "coordinate_reference_system",
            CRS.from_user_input(self.coordinate_reference_system),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        return ("y", "x")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.y.size, self.x.size)

    @property
    def topology(self) -> GridTopology:
        return "projected"

    @property
    def crs(self) -> CRS:
        return self.coordinate_reference_system

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return projected and geographic coordinates."""
        indexes = indexes or {"y": self.y, "x": self.x}
        coordinates: dict[str, Any] = {"y": indexes["y"], "x": indexes["x"]}
        if only_index:
            return xr.Coordinates(coordinates)
        xx, yy = np.meshgrid(indexes["x"], indexes["y"])
        longitude, latitude = Transformer.from_crs(
            self.crs, CRS.from_epsg(4326), always_xy=True
        ).transform(xx, yy)
        coordinates.update(
            lat=(("y", "x"), latitude),
            lon=(("y", "x"), np.mod(longitude, 360)),
        )
        return xr.Coordinates(coordinates)

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
        return coordinate_hash(self.y, self.x) + self.crs.to_wkt()
