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
    max_dist_km: float | Literal["auto"] | None = None,
) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Nearest source point within ``max_dist_km`` of each target, on the sphere.

    ``"auto"`` first estimates the source resolution: twice the largest distance
    between a sample of source points and their nearest neighbors, which bounds the
    distance from any point inside the grid to its nearest point. The search is
    bounded by the limit, so targets far beyond the source (for example a global
    target over a regional source) are rejected quickly instead of searching the
    whole tree.
    """
    source_valid = np.isfinite(source[0]) & np.isfinite(source[1])
    if not source_valid.any():
        raise ValueError("Source grid has no finite geographic coordinates")
    target_valid = np.isfinite(target[0]) & np.isfinite(target[1])
    tree = KDTree(_unit_vectors(source[0][source_valid], source[1][source_valid]))
    limit = max_dist_km
    if limit == "auto":
        points = tree.data[:: max(1, tree.n // _SPACING_SAMPLES)]
        neighbor, _ = tree.query(points, k=2)
        limit = 2 * float(_chord_km(neighbor[:, 1]).max()) if tree.n > 1 else 0.0
    bound = np.inf
    if limit is not None:
        # Chord length of the great-circle limit, slightly widened so targets
        # exactly at the limit are still found and then compared in kilometres.
        angle = min(float(limit) / _EARTH_RADIUS_KM, np.pi)
        bound = 2 * np.sin(angle / 2) * (1 + 1e-9) + 1e-12
    chord = np.full(target[0].size, np.inf)
    nearest = np.zeros(target[0].size, dtype=np.int64)
    chord[target_valid], found = tree.query(
        _unit_vectors(target[0][target_valid], target[1][target_valid]),
        k=1,
        distance_upper_bound=bound,
    )
    hit = np.zeros(target[0].size, dtype=bool)
    hit[target_valid] = found < tree.n
    nearest[hit] = np.flatnonzero(source_valid)[found[found < tree.n]]
    distance = _chord_km(chord)
    valid = target_valid & hit
    if limit is not None:
        valid &= distance <= float(limit)
    return nearest, distance, valid


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
    source_mask : array-like of bool, optional
        Validity of each source point, shaped like ``source_grid`` (an
        ``xr.DataArray`` is transposed to the grid's dimensions). Masked points
        are never selected, so every target takes the value of its nearest
        *valid* source point: a land-masked SST field fills coastal and inland
        targets from the nearest ocean point. By default every point is valid.

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
        source_mask: Any | None = None,
    ) -> None:
        self.max_dist_km = max_dist_km
        self._source_mask = source_mask
        super().__init__(source_grid, target_grid, fill_value)

    @property
    def source_mask(self) -> NDArray[np.bool_] | None:
        """Validity of each source point, shaped like ``source_grid``, or None."""
        return self._mask

    def _valid_source(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """Flattened source coordinates with masked points set to NaN."""
        latitude, longitude = _geographic(self._source_grid)
        self._mask: NDArray[np.bool_] | None = None
        if self._source_mask is None:
            return latitude, longitude
        mask = self._source_mask
        if hasattr(mask, "dims"):
            mask = mask.transpose(*self._source_grid.dims).values
        mask = np.array(mask, dtype=bool, copy=True)
        if mask.shape != self._source_grid.shape:
            raise ValueError(
                f"source_mask shape {mask.shape} does not match the source grid "
                f"shape {self._source_grid.shape}"
            )
        if not mask.any():
            raise ValueError("source_mask leaves no valid source points")
        mask.setflags(write=False)
        self._mask = mask
        keep = mask.ravel()
        return np.where(keep, latitude, np.nan), np.where(keep, longitude, np.nan)

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
        index, _, valid = _tree_neighbors(
            self._valid_source(),
            _geographic(self._target_grid),
            self.max_dist_km,
        )
        return np.where(valid, index, 0)[None], None, valid
