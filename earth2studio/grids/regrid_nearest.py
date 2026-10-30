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
    _geographic,
    _IndexRegridder,
    _Tables,
    _unit_vectors,
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


class NearestRegridder(_IndexRegridder):
    """Map each target point to its nearest source point.

    Distances are great-circle, so the regridder is indifferent to the longitude
    convention of either grid. Source points without finite coordinates, such as
    off-disk satellite pixels, are never selected.

    Parameters
    ----------
    source_grid : str | GridDefinition
        Grid of the fields to regrid, or a registered grid name.
    target_grid : str | GridDefinition
        Grid to produce, or a registered grid name. A name is recorded in the
        output's ``earth2studio_grid_id``.
    max_dist_km : float | "auto" | None, optional
        Targets farther than this from their nearest source point are outside
        coverage. ``"auto"`` uses the source grid's resolution, estimated from the
        spacing of its points, so targets beyond the source are never filled by
        extrapolation.
        By default None, which accepts every target.
    fill_value : float | None, optional
        Value for targets outside coverage, by default NaN. None raises
        ``ValueError`` at construction if any target is outside coverage.

    Note
    ----
    Needs SciPy (the ``utils`` extra) to build the search tree.
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        max_dist_km: float | Literal["auto"] | None = None,
        fill_value: float | None = np.nan,
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
        limit = self.max_dist_km
        index, distance, valid, spacing = _tree_neighbors(
            _geographic(self._source_grid),
            _geographic(self._target_grid),
            spacing=limit == "auto",
        )
        if limit == "auto":
            limit = spacing
        if limit is not None:
            valid = valid & (distance <= float(limit))
        return np.where(valid, index, 0)[None], None, valid
