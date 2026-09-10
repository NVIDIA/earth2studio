# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arbitrary point grids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from earth2studio.grids._utils import (
    array,
    coordinate_hash,
    geographic_subset_indexers,
    metadata,
)
from earth2studio.grids.base import GridTopology


@dataclass(frozen=True)
class PointGrid:
    """Define arbitrary geographic points along an x index."""

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    x: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        latitude = array(self.latitude, 1, "latitude")
        longitude = array(self.longitude, 1, "longitude")
        if latitude.shape != longitude.shape:
            raise ValueError("latitude and longitude shapes must match")
        x = array(np.arange(latitude.size) if self.x is None else self.x, 1, "x")
        if x.size != latitude.size:
            raise ValueError("x size must match geographic coordinates")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "x", x)

    @property
    def dims(self) -> tuple[str, ...]:
        return ("x",)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.latitude.size,)

    @property
    def topology(self) -> GridTopology:
        return "points"

    @property
    def crs(self) -> None:
        return None

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        indexes = indexes or {"x": self.x}
        coordinates = xr.Coordinates({"x": indexes["x"]})
        if only_index:
            return coordinates
        latitude = xr.DataArray(self.latitude, dims="x", coords={"x": self.x}).sel(
            x=indexes["x"]
        )
        longitude = xr.DataArray(self.longitude, dims="x", coords={"x": self.x}).sel(
            x=indexes["x"]
        )
        return coordinates.assign(lat=latitude, lon=longitude)

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        return geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        return None

    def to_metadata(self) -> dict[str, Any]:
        return metadata(self)

    def fingerprint(self) -> str:
        return coordinate_hash(self.latitude, self.longitude)
