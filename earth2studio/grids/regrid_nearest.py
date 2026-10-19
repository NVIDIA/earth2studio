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

"""Nearest-neighbor regridding between grid definitions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from earth2studio.grids._regrid import (
    _EARTH_RADIUS_KM,
    _axis_neighbors,
    _geographic,
    _great_circle_km,
    _IndexRegridder,
    _latlon_axes,
    _longitude_period,
    _monotonic,
    _Tables,
    _unit_vectors,
    _wrap_longitude,
)
from earth2studio.grids.base import GridDefinition
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

_TOPOLOGIES = frozenset(
    {"rectilinear", "projected", "curvilinear", "points", "healpix"}
)

# Source points sampled to estimate a grid's resolution for ``max_dist_km="auto"``.
_SPACING_SAMPLES = 10_000

try:
    from scipy.spatial import KDTree
except ImportError:
    OptionalDependencyFailure("utils")
    KDTree = None


def _rectilinear_nearest(
    grid: GridDefinition, latitude: NDArray[Any], longitude: NDArray[Any]
) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Nearest point on the sphere of a rectilinear grid, without a search tree.

    Along every latitude row the nearest point lies in the column nearest in
    longitude, so the nearest grid point lies on that column's meridian. When the
    meridian's closest approach to the target lies between the poles, distance
    along it has a single minimum there, so the nearest row brackets that
    latitude. Otherwise the closest approach is on the far side of the globe,
    distance peaks between the poles, and the nearest row is the first or last.
    Both cases are covered by choosing among those four rows.
    """
    lat_axis, lon_axis = _latlon_axes(grid)
    columns = np.zeros(latitude.shape, dtype=np.int64)
    if lon_axis.size > 1:
        period = _longitude_period(lon_axis)
        wrapped = longitude
        if period is None:
            ascending = lon_axis if lon_axis[-1] > lon_axis[0] else lon_axis[::-1]
            wrapped = _wrap_longitude(longitude, ascending)
        lower, upper, weight, _ = _axis_neighbors(lon_axis, wrapped, period=period)
        columns = np.where(weight >= 0.5, upper, lower)
    meridian = lon_axis[columns]

    rows = np.zeros(latitude.shape, dtype=np.int64)
    if lat_axis.size > 1:
        lat, dlon = np.radians(latitude), np.radians(longitude - meridian)
        closest = np.degrees(np.arctan2(np.sin(lat), np.cos(lat) * np.cos(dlon)))
        lower, upper, _, _ = _axis_neighbors(lat_axis, np.clip(closest, -90, 90))
        ends = np.zeros_like(lower), np.full_like(lower, lat_axis.size - 1)
        # Upper first, so exact ties resolve as rounding the bracket would.
        candidates = np.stack((upper, lower, *ends))
        km = _great_circle_km(latitude, longitude, lat_axis[candidates], meridian)
        rows = np.take_along_axis(candidates, km.argmin(axis=0)[None], axis=0)[0]
    distance = _great_circle_km(latitude, longitude, lat_axis[rows], meridian)
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    return rows * lon_axis.size + columns, distance, valid


@check_optional_dependencies()
def _tree_neighbors(
    source: tuple[NDArray[Any], NDArray[Any]],
    target: tuple[NDArray[Any], NDArray[Any]],
    spacing: bool = False,
) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_], float | None]:
    """Nearest source point and its distance for each target point, on the sphere.

    With ``spacing``, also estimate the source resolution: twice the largest
    distance between a sample of source points and their nearest neighbors, which
    bounds the distance from any point inside the grid to its nearest point.
    """
    source_valid = np.isfinite(source[0]) & np.isfinite(source[1])
    if not source_valid.any():
        raise ValueError("Source grid has no finite geographic coordinates")
    target_valid = np.isfinite(target[0]) & np.isfinite(target[1])
    tree = KDTree(_unit_vectors(source[0][source_valid], source[1][source_valid]))
    chord = np.full(target[0].size, np.inf)
    nearest = np.zeros(target[0].size, dtype=np.int64)
    chord[target_valid], found = tree.query(
        _unit_vectors(target[0][target_valid], target[1][target_valid]), k=1
    )
    nearest[target_valid] = np.flatnonzero(source_valid)[found]
    distance = _chord_km(chord)
    resolution = None
    if spacing:
        points = tree.data[:: max(1, tree.n // _SPACING_SAMPLES)]
        neighbor, _ = tree.query(points, k=2)
        resolution = 2 * float(_chord_km(neighbor[:, 1]).max()) if tree.n > 1 else 0.0
    return nearest, distance, target_valid, resolution


def _chord_km(chord: NDArray[Any]) -> NDArray[Any]:
    return 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * _EARTH_RADIUS_KM


def _rectilinear_resolution_km(grid: GridDefinition) -> float:
    """Largest cell diagonal of a rectilinear grid, measured at the equator."""
    steps = [
        float(np.abs(np.diff(axis)).max()) if axis.size > 1 else 0.0
        for axis in _latlon_axes(grid)
    ]
    return float(np.radians(np.hypot(*steps)) * _EARTH_RADIUS_KM)


class NearestRegridder(_IndexRegridder):
    """Map each target point to its nearest source point.

    Distances are great-circle, so the regridder is indifferent to the longitude
    convention of either grid. Source points without finite coordinates, such as
    off-disk satellite pixels, are never selected. On a latitude-longitude source
    the nearest point is found exactly from the grid axes, without a search tree.

    Parameters
    ----------
    source_grid : str | GridDefinition
        Grid of the fields to regrid, or a registered grid name.
    target_grid : str | GridDefinition
        Grid to produce, or a registered grid name. A name is recorded in the
        output's ``earth2studio_grid_id``.
    max_dist_km : float | "auto" | None, optional
        Targets farther than this from their nearest source point are outside
        coverage. ``"auto"`` uses the source grid's resolution, its largest cell
        diagonal, so targets beyond the source are never filled by extrapolation.
        By default None, which accepts every target.
    fill_value : float, optional
        Value for targets outside coverage, by default NaN

    Note
    ----
    Sources other than latitude-longitude grids need SciPy (the ``utils`` extra)
    to build the search tree.
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        max_dist_km: float | Literal["auto"] | None = None,
        fill_value: float = np.nan,
    ) -> None:
        self.max_dist_km = max_dist_km
        super().__init__(source_grid, target_grid, fill_value)

    @classmethod
    def supports(cls, source_grid: GridDefinition, target_grid: GridDefinition) -> bool:
        """Whether both grids provide geographic coordinates.

        Parameters
        ----------
        source_grid : GridDefinition
            Grid of the fields to regrid.
        target_grid : GridDefinition
            Grid to produce.

        Returns
        -------
        bool
            True for every built-in grid topology.
        """
        return {source_grid.topology, target_grid.topology} <= _TOPOLOGIES

    def _build(self) -> _Tables:
        source, target = self._source_grid, self._target_grid
        latitude, longitude = _geographic(target)
        limit = self.max_dist_km
        if source.topology == "rectilinear" and all(
            map(_monotonic, _latlon_axes(source))
        ):
            index, distance, valid = _rectilinear_nearest(source, latitude, longitude)
            if limit == "auto":
                limit = _rectilinear_resolution_km(source)
        else:
            index, distance, valid, spacing = _tree_neighbors(
                _geographic(source), (latitude, longitude), spacing=limit == "auto"
            )
            if limit == "auto":
                limit = spacing
        if limit is not None:
            valid = valid & (distance <= float(limit))
        return np.where(valid, index, 0)[None], None, valid
