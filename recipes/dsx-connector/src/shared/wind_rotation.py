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

"""Rotate wind components from Lambert Conformal Conic grid axes to true east and north.

This calculation applies only to Lambert Conformal Conic (LCC) grids whose x and y axes follow the
projection. The caller supplies the projection's central meridian and cone constant. Winds already
expressed relative to true east and north need no rotation, while other map projections require
their own rotation method. The StormCast collector supplies the required HRRR parameters.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

_FloatOrArray: TypeAlias = np.float64 | NDArray[np.float64]


def rotate_lcc_grid_to_earth(
    u_grid: ArrayLike,
    v_grid: ArrayLike,
    lon_deg: ArrayLike,
    lon0_deg: float,
    cone_n: float,
) -> tuple[_FloatOrArray, _FloatOrArray]:
    """Rotate LCC grid-relative winds to earth-relative (true east/north).

    On a Lambert Conformal Conic grid, the model's u and v components follow the projection's x and
    y axes instead of true east and north. The angle between those coordinate systems depends on
    the grid-cell longitude, central meridian, and cone constant. This function calculates that
    angle and rotates the two wind components.

    Parameters
    ----------
    u_grid : ArrayLike
        Wind component along the projection's x axis.
    v_grid : ArrayLike
        Wind component along the projection's y axis.
    lon_deg : ArrayLike
        Grid-cell longitude in degrees. Either common longitude convention is accepted.
    lon0_deg : float
        Central meridian of the LCC projection, degrees (e.g. HRRR: 262.5).
    cone_n : float
        Cone constant of the LCC projection. For a tangent LCC this is ``sin(lat_0)``; for a
        secant LCC it is derived from the two standard parallels. The caller precomputes it (e.g.
        HRRR tangent at 38.5 N: ``sin(radians(38.5))``).

    Returns
    -------
    tuple[np.float64 | np.ndarray, np.float64 | np.ndarray]
        Earth-relative (u_east, v_north) components.
    """
    longitude_difference = ((np.asarray(lon_deg) - lon0_deg + 180.0) % 360.0) - 180.0
    rotation_angle = cone_n * np.deg2rad(longitude_difference)
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    u = np.asarray(u_grid)
    v = np.asarray(v_grid)
    u_east = u * cosine + v * sine
    v_north = -u * sine + v * cosine
    return u_east, v_north
