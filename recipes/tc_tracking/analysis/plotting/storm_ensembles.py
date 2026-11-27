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

"""Ensembles of one storm's tracks, from several trackers and models.

Used by ``wn2_tracker_comparison.py``, with the plots of
:mod:`plotting.comparison_plots`. All tracks are in the units of the recipe's
track files: pressure in Pa, wind in m/s, longitude from 0 to 360.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from src.tc_hunt_utils import match_track

from .data_handling import extract_tracks

# rapid intensification: maximum wind rising by at least 15.4 m/s in 24 h
RI_THRESHOLD_MS = 15.4
RI_HOURS = 24

# distance to the best track within which a forecast track counts as the storm
MATCH_DIST_M = 300_000.0


@dataclass(frozen=True)
class Storm:
    """A storm, as named in IBTrACS and in the recipe's output directories.

    Parameters
    ----------
    name : str
        Name in IBTrACS and in the run directories, e.g. ``"melissa"``
    season : int
        Season, the year the storm formed in on the northern hemisphere
    basin : str
        Basin, as in the names of the reference track files
    """

    name: str
    season: int
    basin: str

    @property
    def label(self) -> str:
        """Name for titles, e.g. ``"Melissa"``."""
        return self.name.title()

    @property
    def reference_file(self) -> str:
        """Name of the file ``extract_baseline`` writes for the storm."""
        return f"reference_track_{self.name}_{self.season}_{self.basin}.csv"


# the storms of the notebook's storm_name; add one with its IBTrACS name,
# season and basin
STORMS = {
    "melissa": Storm("melissa", 2025, "north_atlantic"),
    "ragasa": Storm("ragasa", 2025, "west_pacific"),
}


@dataclass(frozen=True)
class Run:
    """A tracker applied to the forecast of one model, and how it is shown.

    Parameters
    ----------
    label : str
        Name in legends and tables
    color : str
        Colour of its lines and points
    track_dir : str
        Track directory below the run's output directory, ``{storm}`` standing
        for the storm's name
    """

    label: str
    color: str
    track_dir: str


# colour-blind safe, in fixed order; the best track is drawn in black. The
# runs are those of README section 6.1: cfg/<storm>_wn2.yaml, and the same
# config with model fcn3 or aifs
RUNS = {
    "wn2": Run("WN2 built-in tracker", "#2a78d6", "{storm}_wn2/cyclone_tracks_wn2"),
    "wn2_te": Run("TE on WN2", "#eb6834", "{storm}_wn2/cyclone_tracks_te"),
    "fcn3_te": Run("TE on FCN3", "#1baf7a", "{storm}_fcn3/cyclone_tracks_te"),
    "aifs_te": Run("TE on AIFS-ENS", "#eda100", "{storm}_aifs/cyclone_tracks_te"),
}


@dataclass
class Ensemble:
    """The tracks one run's members have of a storm.

    Parameters
    ----------
    run : Run
        Tracker and model
    tracks : pd.DataFrame
        One track per member that follows the storm, one row per member and
        time, with ``member``, ``time``, ``track_error_km`` and the columns of
        the track files
    n_members : int
        Members of the run, including those in which the storm was not found
    """

    run: Run
    tracks: pd.DataFrame
    n_members: int


def load_ensemble(track_dir: str, run: Run, best_track: pd.DataFrame) -> Ensemble:
    """Load the track of a storm from every member's track file.

    A tracker finds every cyclone on the globe; the storm's track is the one
    following the best track, see :func:`src.tc_hunt_utils.match_track`.

    Parameters
    ----------
    track_dir : str
        Directory of track files, ``cyclone_tracks_te`` or ``cyclone_tracks_wn2``
    run : Run
        Tracker and model that wrote them
    best_track : pd.DataFrame
        Best track of the storm, with ``time``, ``lat`` and ``lon``

    Returns
    -------
    Ensemble
        The storm's tracks of all members

    Raises
    ------
    FileNotFoundError
        If the directory holds no track files
    ValueError
        If the track files are of more than one initial time
    """
    members = extract_tracks(track_dir)
    if not members:
        raise FileNotFoundError(f"no track files in {track_dir}")
    # members are told apart by number, which repeats for every initial time
    initial_times = sorted({member["ic"] for member in members})
    if len(initial_times) > 1:
        raise ValueError(
            f"{track_dir} holds forecasts from {len(initial_times)} initial times, "
            f"{initial_times[0]} to {initial_times[-1]}; an ensemble has one"
        )

    tracks = []
    for member in members:
        track = match_track(member["tracks"], best_track, MATCH_DIST_M)
        track["track_error_km"] = track.pop("dist") / 1000.0
        track["member"] = member["member"]
        tracks.append(track)
    tracks = pd.concat(tracks, ignore_index=True)
    if "r17_ne_km" in tracks:
        quadrants = tracks[[f"r17_{q}_km" for q in ("ne", "se", "sw", "nw")]]
        tracks["r17_km"] = quadrants.where(quadrants > 0).mean(axis=1)
    return Ensemble(run, tracks, n_members=len(members))


def load_runs(
    runs_dir: str, storm: Storm, best_track: pd.DataFrame, prefix: str = "outputs_"
) -> dict[str, Ensemble]:
    """Load the storm's ensemble of every run of :data:`RUNS` that exists.

    Parameters
    ----------
    runs_dir : str
        Directory holding the runs' output directories,
        ``<prefix><storm>_<model>``
    storm : Storm
        The storm
    best_track : pd.DataFrame
        Best track of the storm
    prefix : str, optional
        Prefix of the run directories, by default ``"outputs_"`` as written by
        ``main.py``; the sample tracks have none

    Returns
    -------
    dict[str, Ensemble]
        Ensemble of each run whose track directory exists, keyed as :data:`RUNS`

    Raises
    ------
    FileNotFoundError
        If none of the runs exists
    """
    ensembles = {}
    track_dirs = []
    for key, run in RUNS.items():
        track_dir = os.path.join(
            runs_dir, prefix + run.track_dir.format(storm=storm.name)
        )
        track_dirs.append(track_dir)
        if os.path.isdir(track_dir):
            ensembles[key] = load_ensemble(track_dir, run, best_track)
    if not ensembles:
        raise FileNotFoundError(
            f"no runs of {storm.name} found, looked for {', '.join(track_dirs)}"
        )
    return ensembles


def load_analysis_track(reference_csv: str) -> pd.DataFrame:
    """Read the storm's track in the analysis from a reference track file.

    Parameters
    ----------
    reference_csv : str
        Reference track written by mode ``extract_baseline``, e.g. with
        ``cfg/extract_ifs.yaml``

    Returns
    -------
    pd.DataFrame
        ``time``, ``lat``, ``lon``, ``msl`` and ``wind_speed`` of
        TempestExtremes' track in the analysis, at the times it found the storm
    """
    reference = pd.read_csv(reference_csv)
    time = pd.to_datetime(reference[["year", "month", "day", "hour"]])
    track = reference[["lat", "lon", "msl", "wind_speed"]].assign(time=time)
    return track.dropna(subset=["msl"]).reset_index(drop=True)


def ensemble_mean(
    tracks: pd.DataFrame, variable: str, min_members: int = 3
) -> pd.Series:
    """Mean of a variable over the members, at every valid time.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks of an :class:`Ensemble`
    variable : str
        Column to average
    min_members : int, optional
        Times with fewer members tracking the storm are left out, by default 3

    Returns
    -------
    pd.Series
        Mean, indexed by time
    """
    by_time = tracks.dropna(subset=[variable]).groupby("time")[variable]
    mean = by_time.mean()
    return mean[by_time.count() >= min_members]


def add_intensity_change(
    track: pd.DataFrame, variable: str = "wind_speed", hours: int = RI_HOURS
) -> pd.DataFrame:
    """Add the change of a variable over the preceding hours, per member.

    Parameters
    ----------
    track : pd.DataFrame
        Best track, or tracks of an :class:`Ensemble`, with a ``member`` column
    variable : str, optional
        Column to difference, by default ``"wind_speed"``
    hours : int, optional
        Length of the period, by default 24

    Returns
    -------
    pd.DataFrame
        Copy with column ``<variable>_change``, NaN where the track does not
        reach back ``hours``
    """
    key = ["member", "time"] if "member" in track else ["time"]
    earlier = track[[*key, variable]].copy()
    earlier["time"] += pd.Timedelta(hours=hours)
    merged = track.merge(earlier, on=key, how="left", suffixes=("", "_earlier"))
    merged[f"{variable}_change"] = merged[variable] - merged.pop(f"{variable}_earlier")
    return merged


def rapid_intensification_share(ensemble: Ensemble) -> pd.Series:
    """Share of members intensifying rapidly, at every valid time.

    Parameters
    ----------
    ensemble : Ensemble
        The storm's tracks of a run

    Returns
    -------
    pd.Series
        Indexed by time: share of all members whose maximum wind rose by at
        least 15.4 m/s over the preceding 24 h. Members not tracking the storm
        count as not intensifying
    """
    tracks = add_intensity_change(ensemble.tracks)
    rapid = (tracks["wind_speed_change"] >= RI_THRESHOLD_MS).groupby(tracks["time"])
    share = rapid.sum() / ensemble.n_members
    every_step = pd.date_range(share.index.min(), share.index.max(), freq="6h")
    return share.reindex(every_step, fill_value=0.0)


def peak_intensity_table(
    ensembles: Mapping[str, Ensemble], best_track: pd.DataFrame
) -> pd.DataFrame:
    """Peak intensity of the storm: best track against each run's members.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    best_track : pd.DataFrame
        Best track of the storm

    Returns
    -------
    pd.DataFrame
        One row for the best track and one per run, with the peak wind and the
        minimum pressure within the forecast period (mean over the members,
        and 10 to 90 % range), and the share of members tracking the storm and
        intensifying rapidly
    """
    start, end = forecast_period(ensembles)
    observed = best_track[best_track["time"].between(start, end)]
    changes = add_intensity_change(best_track)
    observed_change = changes.loc[
        changes["time"].between(start, end), "wind_speed_change"
    ]
    rows = {
        "IBTrACS best track": {
            "peak wind [m/s]": f"{observed['wind_speed'].max():.0f}",
            "minimum pressure [hPa]": f"{observed['msl'].min() / 100:.0f}",
            "storm tracked": "",
            "rapid intensification": (
                "yes" if (observed_change >= RI_THRESHOLD_MS).any() else "no"
            ),
        }
    }
    for ensemble in ensembles.values():
        members = ensemble.tracks.groupby("member")
        changes = add_intensity_change(ensemble.tracks)
        rapid = changes.loc[changes["wind_speed_change"] >= RI_THRESHOLD_MS, "member"]
        rows[ensemble.run.label] = {
            "peak wind [m/s]": _mean_and_range(members["wind_speed"].max()),
            "minimum pressure [hPa]": _mean_and_range(members["msl"].min() / 100),
            "storm tracked": f"{members.ngroups} of {ensemble.n_members} members",
            "rapid intensification": (
                f"{rapid.nunique()} of {ensemble.n_members} members"
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def forecast_period(
    ensembles: Mapping[str, Ensemble],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last valid time of any run's tracks.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        Start and end of the period the forecasts cover
    """
    times = pd.concat([ensemble.tracks["time"] for ensemble in ensembles.values()])
    return times.min(), times.max()


def _mean_and_range(values: pd.Series) -> str:
    q10, q90 = values.quantile([0.1, 0.9])
    return f"{values.mean():.0f} ({q10:.0f} to {q90:.0f})"
