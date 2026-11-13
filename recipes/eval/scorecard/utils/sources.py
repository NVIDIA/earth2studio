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

"""Source adapters the scorecard campaigns reference from yaml.

StormCast runs on a window of the HRRR grid: the model cuts it from the
registered ``hrrr`` grid at construction (``hrrr_lat_lim`` / ``hrrr_lon_lim``),
so the window has no registered name a campaign could pass to a regridder.
The helpers here rebuild that window from the same limits and wrap a source so the
forecast runner receives fields on the model grid::

    forcing_sources:
      - _target_: scorecard.utils.sources.stormcast_conditioning
        source: {_target_: earth2studio.data.ARCO_ERA5, cache: false}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from earth2studio.data import RegriddedSource
from earth2studio.grids import ProjectedGrid, resolve_grid, resolve_regridder

#: StormCast v1 window of the HRRR grid, as ``earth2studio.models.px.StormCast``
#: cuts it by default (512 x 640 points over the central United States).
STORMCAST_LAT_LIM: tuple[int, int] = (273, 785)
STORMCAST_LON_LIM: tuple[int, int] = (579, 1219)


def stormcast_grid(
    hrrr_lat_lim: Sequence[int] = STORMCAST_LAT_LIM,
    hrrr_lon_lim: Sequence[int] = STORMCAST_LON_LIM,
) -> ProjectedGrid:
    """Build the StormCast model grid: a window of the registered ``hrrr`` grid.

    Parameters
    ----------
    hrrr_lat_lim : Sequence[int], optional
        Index limits along the HRRR ``y`` axis, by default the v1 window.
    hrrr_lon_lim : Sequence[int], optional
        Index limits along the HRRR ``x`` axis, by default the v1 window.

    Returns
    -------
    ProjectedGrid
        The model grid, identical to ``StormCast.grid`` for the same limits.
    """
    parent = cast(ProjectedGrid, resolve_grid("hrrr"))
    return ProjectedGrid(
        parent.y[slice(*hrrr_lat_lim)], parent.x[slice(*hrrr_lon_lim)], parent.crs
    )


def stormcast_conditioning(
    source: Any,
    source_grid: str = "latlon-0.25deg",
    method: str = "bilinear",
    hrrr_lat_lim: Sequence[int] = STORMCAST_LAT_LIM,
    hrrr_lon_lim: Sequence[int] = STORMCAST_LON_LIM,
    **options: Any,
) -> RegriddedSource:
    """Regrid a global source onto the StormCast grid for its conditioning slot.

    Parameters
    ----------
    source : DataSource | ForecastSource
        Global source on ``source_grid`` (for example ``ARCO_ERA5``).
    source_grid : str, optional
        Registered name of the source grid, by default ``latlon-0.25deg``.
    method : str, optional
        Registered regridding method, by default ``bilinear`` (what the
        model's own ``default_sources()`` uses).
    hrrr_lat_lim : Sequence[int], optional
        Model window along the HRRR ``y`` axis, by default the v1 window.
    hrrr_lon_lim : Sequence[int], optional
        Model window along the HRRR ``x`` axis, by default the v1 window.
    **options : Any
        Keyword arguments of the regridder.

    Returns
    -------
    RegriddedSource
        ``source`` regridded onto the model grid.
    """
    grid = stormcast_grid(hrrr_lat_lim, hrrr_lon_lim)
    return RegriddedSource(
        source, resolve_regridder(source_grid, grid, method, **options)
    )


def stormcast_window(
    source: Any,
    hrrr_lat_lim: Sequence[int] = STORMCAST_LAT_LIM,
    hrrr_lon_lim: Sequence[int] = STORMCAST_LON_LIM,
) -> RegriddedSource:
    """Crop a full-grid HRRR source to the StormCast window.

    The model window is an exact sub-grid of ``hrrr``, so a nearest-neighbour
    map between the two is a pure gather and values pass through unchanged.
    Use it for truth (``verification_source``) when the scorer compares the
    model output against a source returning the whole HRRR grid.

    Parameters
    ----------
    source : DataSource
        Source on the full ``hrrr`` grid (for example ``HRRR``).
    hrrr_lat_lim : Sequence[int], optional
        Model window along the HRRR ``y`` axis, by default the v1 window.
    hrrr_lon_lim : Sequence[int], optional
        Model window along the HRRR ``x`` axis, by default the v1 window.

    Returns
    -------
    RegriddedSource
        ``source`` cropped to the model grid.
    """
    grid = stormcast_grid(hrrr_lat_lim, hrrr_lon_lim)
    return RegriddedSource(source, resolve_regridder("hrrr", grid, "nearest"))
