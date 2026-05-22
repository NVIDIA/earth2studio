# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Make ``src/`` importable when the plotting modules are run from
# ``recipes/tc_tracking/plotting/`` (the conventional working directory for
# the notebooks).  Mirrors how ``tc_hunt.py`` exposes ``src`` to the rest of
# the recipe.
_RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _RECIPE_ROOT not in sys.path:
    sys.path.insert(0, _RECIPE_ROOT)

from src.tc_hunt_utils import EARTH_RADIUS_M, great_circle_distance  # noqa: E402

_DEFAULT_TIME_STEP = np.timedelta64(6, "h")


def merge_tracks_by_time(track: pd.DataFrame, tru_track: pd.DataFrame) -> pd.DataFrame:
    """Left-join a predicted track onto a reference track by time.

    Parameters
    ----------
    track : pd.DataFrame
        Predicted track.
    tru_track : pd.DataFrame
        Reference (true) track.

    Returns
    -------
    pd.DataFrame
        Merged frame with ``_tru`` suffixes on reference columns,
        clipped to the time range of the reference track.
    """
    merged_track = pd.merge(
        track, tru_track, on="time", how="left", suffixes=("", "_tru")
    )

    merged_track = merged_track[merged_track["time"] <= tru_track["time"].max()]

    return merged_track


def add_track_distance(track: pd.DataFrame, tru_track: pd.DataFrame) -> pd.DataFrame:
    """Augment *track* with a ``dist`` column measuring great-circle distance to *tru_track*.

    Parameters
    ----------
    track : pd.DataFrame
        Predicted track containing ``lat`` and ``lon`` columns.
    tru_track : pd.DataFrame
        Reference track containing ``lat`` and ``lon`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *track* with an additional ``dist`` column (metres).
    """
    merged_track = merge_tracks_by_time(track, tru_track)[
        ["time", "lat", "lon", "lat_tru", "lon_tru"]
    ]

    dist = great_circle_distance(
        merged_track["lat"],
        merged_track["lon"],
        merged_track["lat_tru"],
        merged_track["lon_tru"],
    )

    merged_track["dist"] = dist

    track = pd.merge(
        track, merged_track[["time", "dist"]], on="time", how="left", suffixes=("", "")
    )

    return track


def match_tracks(
    pred_tracks: list[dict[str, Any]],
    true_track: pd.DataFrame,
    max_dist: float = 300000,
) -> list[dict[str, Any]]:
    """Match predicted tracks to a reference track by proximity at first overlap.

    A predicted track is considered a match if its first position is within
    ``max_dist`` metres of the reference track at the same time step.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        List of prediction dicts, each containing ``"ic"``, ``"member"``,
        and ``"tracks"`` (a DataFrame).
    true_track : pd.DataFrame
        Reference track with ``lat_ib`` and ``lon_ib`` columns for
        IBTrACS positions.
    max_dist : float, optional
        Maximum great-circle distance (in metres) between the first
        predicted position and the reference position to count as a
        match, by default 300000 (300 km).

    Returns
    -------
    list[dict[str, Any]]
        Subset of matched tracks, each augmented with ``"first_match"``,
        ``"initial_dist"``, and a ``dist`` column on the track DataFrame.
    """
    matched_tracks: list[dict[str, Any]] = []
    min_seen, max_seen = float("inf"), float("-inf")

    for _pred_track_dict in pred_tracks:
        _pred_tracks = _pred_track_dict["tracks"]

        if len(_pred_tracks) == 0:
            continue

        n_tracks = _pred_tracks["track_id"].iloc[-1] + 1

        for ii in range(n_tracks):
            track = _pred_tracks.loc[_pred_tracks["track_id"] == ii].copy()

            lat_pred = track["lat"].iloc[0]
            lon_pred = track["lon"].iloc[0]

            time_mask = true_track["time"] == track["time"].iloc[0]
            if not time_mask.any():
                continue

            ref_row = true_track.loc[time_mask]
            lat_true = ref_row["lat_ib"].item()
            lon_true = ref_row["lon_ib"].item()
            dist = great_circle_distance(lat_pred, lon_pred, lat_true, lon_true)

            if dist <= max_dist:
                min_seen, max_seen = min(min_seen, dist), max(max_seen, dist)

                track = add_track_distance(track, true_track)

                matched_tracks.append(
                    {
                        "ic": _pred_track_dict["ic"],
                        "member": _pred_track_dict["member"],
                        "first_match": track["time"].iloc[0],
                        "initial_dist": dist,
                        "tracks": track,
                    }
                )
                break

    if matched_tracks:
        logger.info(
            f"matched {len(matched_tracks)} out of {len(pred_tracks)} tracks, "
            f"with distances ranging from {min_seen/1000:.1f} to "
            f"{max_seen/1000:.1f} km"
        )
    else:
        logger.info(f"matched 0 out of {len(pred_tracks)} tracks")

    return matched_tracks


def extract_tracks_from_file(csv_file: str) -> pd.DataFrame:
    """Read a TempestExtremes track CSV and convert date columns to a single ``time`` column.

    Parameters
    ----------
    csv_file : str
        Path to a CSV file produced by TempestExtremes ``StitchNodes``.

    Returns
    -------
    pd.DataFrame
        Track data with a ``time`` column (datetime64) prepended.
    """
    tracks = pd.read_csv(csv_file, sep=",")
    tracks.columns = tracks.columns.str.strip()

    times = pd.to_datetime(tracks[["year", "month", "day", "hour"]].astype(int))

    tracks.drop(columns=["year", "month", "day", "hour"], inplace=True)
    if "i" in tracks.columns:
        tracks.drop(columns=["i", "j"], inplace=True)

    tracks.insert(0, "time", times)

    return tracks


def extract_tracks(in_dir: str) -> list[dict[str, Any]]:
    """Load all track CSV files from a directory.

    Parameters
    ----------
    in_dir : str
        Directory containing track CSV files whose names encode the
        initial condition timestamp, member ID, and random seed.

    Returns
    -------
    list[dict[str, Any]]
        One dict per file with keys ``"ic"`` (Timestamp), ``"member"``
        (int), and ``"tracks"`` (DataFrame).
    """
    tracks: list[dict[str, Any]] = []
    files = glob.glob(f"{in_dir}/*.csv")
    files.sort()

    for csv_file in files:
        _tracks = extract_tracks_from_file(csv_file)

        mem = int(csv_file.split("_mem_")[-1].split("_seed_")[0])
        ic = pd.to_datetime(csv_file.split("_mem_")[0][-19:])

        tracks.append({"ic": ic, "member": mem, "tracks": _tracks})

    return tracks


def merge_track_dict_by_time(
    track_dict: dict[str, Any], tru_track: pd.DataFrame
) -> pd.DataFrame:
    """Left-join the track stored under ``track_dict["tracks"]`` onto a reference track.

    Thin wrapper around :func:`merge_tracks_by_time` that accepts a
    prediction dictionary rather than the inner DataFrame.

    Parameters
    ----------
    track_dict : dict[str, Any]
        Prediction dict containing a ``"tracks"`` DataFrame.
    tru_track : pd.DataFrame
        Reference track.

    Returns
    -------
    pd.DataFrame
        Merged frame, clipped to the time range of *tru_track*.
    """
    return merge_tracks_by_time(track_dict["tracks"], tru_track)


def compute_mae(tru_vars: np.ndarray, pred_vars: np.ndarray) -> np.ndarray:
    """Compute mean absolute error along the first axis, ignoring NaNs."""
    return np.nanmean(np.abs(tru_vars - pred_vars), axis=0)


def compute_mse(tru_vars: np.ndarray, pred_vars: np.ndarray) -> np.ndarray:
    """Compute mean squared error along the first axis, ignoring NaNs."""
    return np.nanmean((tru_vars - pred_vars) ** 2, axis=0)


def compute_variance(arr: np.ndarray) -> np.ndarray:
    """Compute variance along the first axis, ignoring NaNs."""
    return np.nanvar(arr, axis=0)


def remove_trailing_nans(merged_track: pd.DataFrame, var: str) -> pd.DataFrame:
    """Trim rows after the last time step where both predicted and true values are present.

    Parameters
    ----------
    merged_track : pd.DataFrame
        Merged track with ``var`` and ``var_tru`` columns.
    var : str
        Variable name (the true column is ``var + "_tru"``).

    Returns
    -------
    pd.DataFrame
        Truncated frame.
    """
    either_nans = np.logical_or(
        merged_track[var + "_tru"].isna(), merged_track[var].isna()
    )
    cut_off = np.where(~either_nans)[0][-1]

    return merged_track.iloc[: cut_off + 1]


def rebase_by_lead_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    variables: list[str],
) -> tuple[dict[str, dict[str, list]], int]:
    """Align predicted and true values by lead time for error computation.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    variables : list[str]
        Variable names to extract.

    Returns
    -------
    tuple[dict[str, dict[str, list]], int]
        Per-variable dict of ``{"pred": [...], "tru": [...]}`` lists
        and the maximum lead-time length across all tracks.
    """
    err_dict: dict[str, dict[str, list]] = {}
    for var in variables:
        err_dict[var] = {"pred": [], "tru": []}

    max_len = 0
    for track in pred_tracks:
        merged_track = merge_track_dict_by_time(track, tru_track)
        if merged_track is None:
            continue

        merged_track = remove_trailing_nans(merged_track, "msl")

        max_len = max(max_len, len(merged_track))

        for var in err_dict.keys():
            err_dict[var]["pred"].append(merged_track[var])
            err_dict[var]["tru"].append(merged_track[var + "_tru"])

    return err_dict, max_len


def compute_error_metrics(
    err_dict: dict[str, dict[str, list]], max_len: int
) -> dict[str, dict[str, np.ndarray]]:
    """Compute MAE, MSE, variance, extremes and member counts from aligned predictions.

    Parameters
    ----------
    err_dict : dict[str, dict[str, list]]
        Per-variable dict of ``{"pred": [...], "tru": [...]}`` lists
        as returned by :func:`rebase_by_lead_time`.
    max_len : int
        Maximum lead-time length used for NaN-padding.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per-variable dict of computed metrics.
    """
    for var in err_dict.keys():
        pred_vars = err_dict[var]["pred"]
        tru_vars = err_dict[var]["tru"]

        counts = np.zeros(max_len, dtype=int)
        for ii in range(len(pred_vars)):
            counts[: len(pred_vars[ii])] += 1

            pred_vars[ii] = np.pad(
                pred_vars[ii],
                (0, max_len - len(pred_vars[ii])),
                mode="constant",
                constant_values=np.nan,
            )
            tru_vars[ii] = np.pad(
                tru_vars[ii],
                (0, max_len - len(tru_vars[ii])),
                mode="constant",
                constant_values=np.nan,
            )

        pred_vars, tru_vars = np.array(pred_vars), np.array(tru_vars)

        err_dict[var] = {
            "mae": compute_mae(tru_vars, pred_vars),
            "mse": compute_mse(tru_vars, pred_vars),
            "variance": compute_variance(pred_vars),
            "max": np.nanmax(pred_vars, axis=-1),
            "min": np.nanmin(pred_vars, axis=-1),
            "n_members": counts,
        }

    return err_dict


def compute_averages_of_errors_over_lead_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    variables: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Compute error metrics averaged over ensemble members as a function of lead time.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    variables : list[str]
        Variable names to evaluate.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], int]
        Per-variable error metrics and the maximum lead-time length.
    """
    err_dict, max_len = rebase_by_lead_time(pred_tracks, tru_track, variables)

    err_dict = compute_error_metrics(err_dict, max_len)

    return err_dict, max_len


def lat_lon_to_xyz(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert latitude/longitude to 3-D Cartesian coordinates.

    Parameters
    ----------
    lat : float or np.ndarray
        Latitude(s) in degrees (range [-90, 90]).
    lon : float or np.ndarray
        Longitude(s) in degrees (range [0, 360)).
    radius : float, optional
        Sphere radius in metres, by default ``EARTH_RADIUS_M`` (6 371 km).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, z)`` Cartesian coordinates.
    """
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    xx = radius * np.cos(lat_rad) * np.cos(lon_rad)
    yy = radius * np.cos(lat_rad) * np.sin(lon_rad)
    zz = radius * np.sin(lat_rad)

    return xx, yy, zz


def xyz_to_lat_lon(
    xx: float | np.ndarray,
    yy: float | np.ndarray,
    zz: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3-D Cartesian coordinates back to latitude/longitude.

    Parameters
    ----------
    xx : float or np.ndarray
        X coordinate(s).
    yy : float or np.ndarray
        Y coordinate(s).
    zz : float or np.ndarray
        Z coordinate(s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lat, lon)`` in degrees. Longitude is in [0, 360).
    """
    radius = np.sqrt(xx**2 + yy**2 + zz**2)

    lat_rad = np.arcsin(zz / (radius + 1e-9))

    lon_rad = np.arctan2(yy, xx)

    lat = np.degrees(lat_rad)
    lon = (np.degrees(lon_rad) + 360) % 360

    return lat, lon


def cartesian_to_spherical_track(
    stats: dict[str, Any],
    tru_track: pd.DataFrame,
    frame_of_reference: pd.DataFrame,
) -> dict[str, Any]:
    """Convert Cartesian ensemble-mean position back to spherical and compute distance.

    Replaces the ``x, y, z`` entries in ``stats["mean"]`` and
    ``stats["variance"]`` with ``lat``, ``lon``, and ``dist``.

    Parameters
    ----------
    stats : dict[str, Any]
        Ensemble statistics dict (modified in place).
    tru_track : pd.DataFrame
        Reference track.
    frame_of_reference : pd.DataFrame
        Single-column ``time`` frame covering all lead times.

    Returns
    -------
    dict[str, Any]
        Updated *stats*.
    """
    mean_lat, mean_lon = xyz_to_lat_lon(
        stats["mean"]["x"], stats["mean"]["y"], stats["mean"]["z"]
    )

    for var in ["x", "y", "z"]:
        for metric in ["mean", "variance"]:
            del stats[metric][var]

    stats["mean"]["lat"] = mean_lat
    stats["mean"]["lon"] = mean_lon

    tru_cont = pd.merge(frame_of_reference, tru_track, on="time", how="left")

    dist = great_circle_distance(
        tru_cont["lat"], tru_cont["lon"], stats["mean"]["lat"], stats["mean"]["lon"]
    )

    stats["mean"]["dist"] = np.asarray(dist, dtype=float)

    for var in ["msl", "wind_speed"]:
        stats["mean"][var + "_err_of_mean"] = stats["mean"][var] - tru_cont[var]

    return stats


def get_ensemble_averages(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    variables: list[str] | None = None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> dict[str, Any]:
    """Compute ensemble mean and variance on the sphere.

    Averaging is performed in Cartesian space to avoid artefacts near
    the antimeridian, then converted back to lat/lon.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    variables : list[str] | None, optional
        Variables to average (must include ``x``, ``y``, ``z`` for the
        Cartesian round-trip), by default
        ``["msl", "wind_speed", "x", "y", "z"]``
    time_step : np.timedelta64, optional
        Spacing of the output time axis, by default 6 h.

    Returns
    -------
    dict[str, Any]
        Dict with keys ``"time"``, ``"n_members"``, ``"mean"``, and
        ``"variance"``.
    """
    if variables is None:
        variables = ["msl", "wind_speed", "x", "y", "z"]

    stats: dict[str, Any] = {
        "time": None,
        "n_members": None,
        "mean": {var: [] for var in variables},
        "variance": {var: [] for var in variables},
    }

    last_time = pred_tracks[0]["ic"]
    for track in pred_tracks:
        last_time = max(last_time, track["tracks"]["time"].values[-1])

    all_times = np.arange(pred_tracks[0]["ic"], last_time, time_step)
    stats["time"] = all_times

    frame_of_reference = pd.DataFrame(
        data=all_times, index=np.arange(len(all_times)), columns=["time"]
    )

    for track in pred_tracks:
        xx, yy, zz = lat_lon_to_xyz(track["tracks"]["lat"], track["tracks"]["lon"])
        track["tracks"]["x"] = xx
        track["tracks"]["y"] = yy
        track["tracks"]["z"] = zz

        contextualised = pd.merge(
            frame_of_reference, track["tracks"], on="time", how="left"
        )

        for var in variables:
            stats["mean"][var].append(contextualised[var])

    for var in variables:
        stacked = np.stack(stats["mean"][var])
        counts = np.count_nonzero(~np.isnan(stacked), axis=0)

        if stats["n_members"] is None:
            stats["n_members"] = counts
        elif not np.all(stats["n_members"] == counts):
            raise ValueError(
                "n_members is not the same for all variables but should be"
            )

        stats["variance"][var] = np.nanvar(stacked, axis=0)
        stats["mean"][var] = np.nanmean(stacked, axis=0)

    stats = cartesian_to_spherical_track(stats, tru_track, frame_of_reference)

    return stats
