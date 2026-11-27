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

import os
import re

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from earth2studio.utils.time import to_time_array

EARTH_RADIUS_M = 6371000

_TRACK_FILE = re.compile(
    r"tracks_(?P<ic>[\dT:-]+)_mem_(?P<member>\d+)_seed_(?P<seed>\d+)_bs_\d+\.csv"
)


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


def track_file_name(ic: np.datetime64, member: int, seed: int, batch_size: int) -> str:
    """Name of a member's track file, the same for every tracker.

    Parameters
    ----------
    ic : np.datetime64
        Initial time of the forecast
    member : int
        Member id
    seed : int
        Seed the member was run with
    batch_size : int
        Batch size of the run

    Returns
    -------
    str
        ``tracks_<ic>_mem_<member>_seed_<seed>_bs_<batch_size>.csv``, see
        :func:`parse_track_file_name`
    """
    ic_str = np.datetime_as_string(ic, unit="s")
    return f"tracks_{ic_str}_mem_{member:04d}_seed_{seed}_bs_{batch_size}.csv"


def parse_track_file_name(path: str) -> tuple[pd.Timestamp, int, int]:
    """Initial time, member and seed of a track file, see :func:`track_file_name`.

    Parameters
    ----------
    path : str
        Track file, with or without directory

    Returns
    -------
    tuple[pd.Timestamp, int, int]
        Initial time, member id and seed

    Raises
    ------
    ValueError
        If the file name does not follow :func:`track_file_name`
    """
    match = _TRACK_FILE.fullmatch(os.path.basename(path))
    if match is None:
        raise ValueError(f"not a track file name: {path}")
    return (
        pd.Timestamp(match["ic"]),
        int(match["member"]),
        int(match["seed"]),
    )


def match_track(
    tracks: pd.DataFrame, reference: pd.DataFrame, max_dist: float = 300_000.0
) -> pd.DataFrame:
    """Track following a reference track, e.g. a storm's best track.

    The match is the track closest to the reference at the first time any
    track lies within ``max_dist`` of it. The same rule identifies a storm in
    reference tracks, in forecasts and in reproduced members.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks with ``track_id``, ``time``, ``lat`` and ``lon``
    reference : pd.DataFrame
        Reference track with ``time``, ``lat`` and ``lon``
    max_dist : float, optional
        Largest distance to the reference counting as a match in metres, by
        default 300 km

    Returns
    -------
    pd.DataFrame
        The rows of the matching track, with its distance to the reference in
        ``dist`` (metres, NaN at times the reference lacks); empty, with the
        columns of ``tracks``, if no track matches
    """
    pairs = tracks[["track_id", "time", "lat", "lon"]].merge(
        reference[["time", "lat", "lon"]], on="time", suffixes=("", "_ref")
    )
    pairs["dist"] = great_circle_distance(
        pairs["lat_ref"].to_numpy(),
        pairs["lon_ref"].to_numpy(),
        pairs["lat"].to_numpy(),
        pairs["lon"].to_numpy(),
    )
    close = pairs[pairs["dist"] < max_dist]
    if close.empty:
        return tracks.iloc[0:0].assign(dist=[])

    track_id = close.sort_values(["time", "dist"]).iloc[0]["track_id"]
    track = tracks[tracks["track_id"] == track_id]
    dist = pairs.loc[pairs["track_id"] == track_id, ["time", "dist"]]
    return track.merge(dist, on="time", how="left").reset_index(drop=True)
