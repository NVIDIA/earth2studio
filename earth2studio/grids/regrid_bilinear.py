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

"""Bilinear regridding between grid definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from earth2studio.grids._regrid import (
    _axis_neighbors,
    _geographic,
    _index_axes,
    _IndexRegridder,
    _latlon_axes,
    _latlon_neighbors,
    _monotonic,
    _Tables,
)
from earth2studio.grids.base import GridDefinition
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from scipy.interpolate import LinearNDInterpolator
except ImportError:
    OptionalDependencyFailure("utils")
    LinearNDInterpolator = None


@check_optional_dependencies()
def _triangulated_positions(
    source: GridDefinition, latitude: NDArray[Any], longitude: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Fractional source indices of geographic points by Delaunay triangulation.

    Interpolation is planar in degrees, so targets must not cross the longitude seam
    of the source. Points outside the source's convex hull are NaN.
    """
    source_lat, source_lon = _geographic(source)
    rows, columns = np.unravel_index(np.arange(source_lat.size), source.shape)
    valid = np.isfinite(source_lat) & np.isfinite(source_lon)
    points = np.stack((source_lat[valid], source_lon[valid]), axis=-1)
    longitude = source_lon[valid].min() + np.mod(
        longitude - source_lon[valid].min(), 360.0
    )
    queries = np.stack((latitude, longitude), axis=-1)
    return (
        LinearNDInterpolator(points, rows[valid].astype(np.float64))(queries),
        LinearNDInterpolator(points, columns[valid].astype(np.float64))(queries),
    )


def _cell_neighbors(
    position: NDArray[Any], size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Bracket fractional indices, such as triangulated positions, on an axis."""
    valid = np.isfinite(position)
    position = np.where(valid, position, 0.0)
    lower = np.clip(np.floor(position), 0, size - 2).astype(np.int64)
    return lower, lower + 1, position - lower, valid


class BilinearRegridder(_IndexRegridder):
    """Interpolate between the four source points surrounding each target point.

    Source points are located by topology:

    - Rectilinear sources bracket each target along their latitude and longitude
      axes. Longitude wraps when the grid covers the globe, and target longitudes
      are matched to the source convention.
    - Projected sources transform targets into the source CRS and bracket them
      along the projected axes, which is exact for any target.
    - Curvilinear sources triangulate their geographic coordinates. This is planar
      in degrees, needs SciPy (the ``utils`` extra), and does not wrap longitude.

    Targets outside the source grid are outside coverage. Interpolation is in index
    space, so a source NaN propagates to targets that weight it.

    Parameters
    ----------
    source_grid : str | GridDefinition
        Rectilinear, projected or curvilinear grid of the fields to regrid, or a
        registered grid name.
    target_grid : str | GridDefinition
        Grid to produce, or a registered grid name. A name is recorded in the
        output's ``earth2studio_grid_id``.
    fill_value : float | None, optional
        Value for targets outside coverage, by default NaN. None raises
        ``ValueError`` at construction if any target is outside coverage.
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        fill_value: float | None = np.nan,
    ) -> None:
        super().__init__(source_grid, target_grid, fill_value)

    @classmethod
    def supports(cls, source_grid: GridDefinition, target_grid: GridDefinition) -> bool:
        """Whether the source grid has the structure bilinear interpolation needs.

        Parameters
        ----------
        source_grid : GridDefinition
            Grid of the fields to regrid.
        target_grid : GridDefinition
            Grid to produce.

        Returns
        -------
        bool
            True for rectilinear sources with monotonic axes of at least two
            points, projected sources with a CRS, and curvilinear sources.
        """
        if source_grid.topology == "rectilinear":
            return all(
                axis.size > 1 and _monotonic(axis) for axis in _latlon_axes(source_grid)
            )
        if source_grid.topology == "projected":
            return source_grid.crs is not None
        return source_grid.topology == "curvilinear"

    def _build(self) -> _Tables:
        source, target = self._source_grid, self._target_grid
        if source.topology == "rectilinear":
            latitude, longitude = _geographic(target)
            rows, columns = _latlon_neighbors(source, latitude, longitude)
        elif source.topology == "projected":
            if source.crs is None:
                raise ValueError("Projected source grids need a CRS")
            y_axis, x_axis = _index_axes(source)
            if target.topology == "projected" and target.crs == source.crs:
                target_y, target_x = _index_axes(target)
                y, x = (
                    values.ravel()
                    for values in np.meshgrid(target_y, target_x, indexing="ij")
                )
            else:
                latitude, longitude = _geographic(target)
                x, y = Transformer.from_crs(
                    CRS.from_epsg(4326), source.crs, always_xy=True
                ).transform(longitude, latitude)
            rows, columns = _axis_neighbors(y_axis, y), _axis_neighbors(x_axis, x)
        elif source.topology == "curvilinear":
            latitude, longitude = _geographic(target)
            row, column = _triangulated_positions(source, latitude, longitude)
            rows = _cell_neighbors(row, source.shape[0])
            columns = _cell_neighbors(column, source.shape[1])
        else:
            raise ValueError(
                f"Bilinear regridding needs a structured source grid, not "
                f"{source.topology}; use NearestRegridder"
            )
        width = source.shape[1]
        row0, row1, row_weight, row_valid = rows
        column0, column1, column_weight, column_valid = columns
        valid = row_valid & column_valid
        index = np.stack(
            (
                row0 * width + column0,
                row0 * width + column1,
                row1 * width + column0,
                row1 * width + column1,
            )
        )
        weight = np.stack(
            (
                (1 - row_weight) * (1 - column_weight),
                (1 - row_weight) * column_weight,
                row_weight * (1 - column_weight),
                row_weight * column_weight,
            )
        )
        return np.where(valid, index, 0), np.where(valid, weight, 0.0), valid
