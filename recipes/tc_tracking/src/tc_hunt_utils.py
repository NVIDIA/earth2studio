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

import numpy as np
from omegaconf import DictConfig

from earth2studio.utils.time import to_time_array

EARTH_RADIUS_M = 6371000


def set_initial_times(cfg: DictConfig) -> np.ndarray:
    """Build array of initial conditions.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.

    Returns
    -------
    np.ndarray
        Array of initial conditions (dtype ``datetime64``).
    """
    # list of ICs
    if "ics" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError(
                "either provide a list of start times or define a block, not both"
            )
        ics = to_time_array(sorted(cfg.ics))

    # block of ICs
    else:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        ics = np.arange(
            ics[0],
            ics[1] + np.timedelta64(cfg.ic_block_step, "h"),
            np.timedelta64(cfg.ic_block_step, "h"),
        )

    return ics


def great_circle_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> float | np.ndarray:
    """Compute the great-circle distance between two points on a sphere.

    Uses the Haversine formula on the sphere, the radius of which defaults
    to Earth's mean radius of 6371 km.

    Parameters
    ----------
    lat1 : float or np.ndarray
        Latitude(s) of the first point in degrees.
    lon1 : float or np.ndarray
        Longitude(s) of the first point in degrees.
    lat2 : float or np.ndarray
        Latitude(s) of the second point in degrees.
    lon2 : float or np.ndarray
        Longitude(s) of the second point in degrees.
    radius : float, optional
        Sphere radius in metres, by default ``EARTH_RADIUS_M`` (6 371 km).

    Returns
    -------
    float or np.ndarray
        Distance(s) in metres.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    cc = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1 - aa))

    return radius * cc
