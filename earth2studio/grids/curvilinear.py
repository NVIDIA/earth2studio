# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curvilinear grids."""

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
class CurvilinearGrid:
    """Define a structured grid from two-dimensional geographic coordinates."""

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    y: NDArray[Any] | None = None
    x: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        latitude = array(self.latitude, 2, "latitude")
        longitude = array(self.longitude, 2, "longitude")
        if latitude.shape != longitude.shape:
            raise ValueError("latitude and longitude shapes must match")
        y = array(np.arange(latitude.shape[0]) if self.y is None else self.y, 1, "y")
        x = array(np.arange(latitude.shape[1]) if self.x is None else self.x, 1, "x")
        if (y.size, x.size) != latitude.shape:
            raise ValueError("y and x sizes must match geographic coordinates")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "x", x)

    @property
    def dims(self) -> tuple[str, ...]:
        return ("y", "x")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.latitude.shape

    @property
    def topology(self) -> GridTopology:
        return "curvilinear"

    @property
    def crs(self) -> None:
        return None

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return index and geographic coordinates."""
        indexes = indexes or {"y": self.y, "x": self.x}
        coordinates = xr.Coordinates({"y": indexes["y"], "x": indexes["x"]})
        if only_index:
            return coordinates
        latitude = xr.DataArray(
            self.latitude, dims=self.dims, coords={"y": self.y, "x": self.x}
        ).sel(y=indexes["y"], x=indexes["x"])
        longitude = xr.DataArray(
            self.longitude, dims=self.dims, coords={"y": self.y, "x": self.x}
        ).sel(y=indexes["y"], x=indexes["x"])
        return coordinates.assign(lat=latitude, lon=longitude)

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
        return coordinate_hash(self.latitude, self.longitude)
