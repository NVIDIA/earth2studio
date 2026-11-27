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

"""Compare the tracks of several trackers of one storm with its best track.

Used by ``wn2_tracker_comparison.py``. All tracks are in the units of the
recipe's track files: pressure in Pa, wind in m/s, longitude from 0 to 360.

An *ensemble* here is a long table with one row per member and time, holding
the one track of each member that follows the storm, see
:func:`load_storm_ensemble`.
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

_RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _RECIPE_ROOT not in sys.path:
    sys.path.insert(0, _RECIPE_ROOT)

from src.tc_hunt_utils import great_circle_distance  # noqa: E402

from .data_handling import extract_tracks  # noqa: E402

# rapid intensification: maximum wind rising by at least 15.4 m/s in 24 h
RI_THRESHOLD_MS = 15.4
RI_HOURS = 24

BEST_TRACK_COLOR = "#1a1a19"
# maps: pale ocean and land, the best track slightly softer than in the plots
OCEAN_COLOR = "#e3eef8"
LAND_COLOR = "#e6eedd"
COAST_COLOR = "#8a9a86"
MAP_BEST_TRACK_COLOR = "#3d3d3a"
REFERENCE_COLOR = "#8a8a84"


@dataclass(frozen=True)
class Run:
    """A tracker applied to the forecast of one model, and how it is shown.

    ``track_dir`` is the run's directory, ``<case>_<model>`` with ``{case}``
    standing for the storm, and its track directory. ``main.py`` writes it
    as ``outputs_<case>_<model>`` with ``cfg/<case>_wn2.yaml`` and its model
    overrides, the sample tracks without the prefix.
    """

    label: str
    color: str
    track_dir: str


# colour-blind safe, in fixed order; the best track is drawn in black
RUNS = {
    "wn2": Run("WN2 built-in tracker", "#2a78d6", "{case}_wn2/cyclone_tracks_wn2"),
    "wn2_te": Run("TE on WN2", "#eb6834", "{case}_wn2/cyclone_tracks_te"),
    "fcn3_te": Run("TE on FCN3", "#1baf7a", "{case}_fcn3/cyclone_tracks_te"),
    "aifs_te": Run("TE on AIFS-ENS", "#eda100", "{case}_aifs/cyclone_tracks_te"),
}

# variable: (axis label, factor from the unit of the track files)
VARIABLES = {
    "wind_speed": ("maximum 10 m wind [m/s]", 1.0),
    "msl": ("minimum sea-level pressure [hPa]", 0.01),
    "track_error_km": ("distance to best track [km]", 1.0),
    "rmw_km": ("radius of maximum wind [km]", 1.0),
    "r17_km": ("radius of 17.5 m/s wind [km]", 1.0),
}


# --------------------------------------------------------------------------- data


def select_storm_track(
    tracks: pd.DataFrame, best_track: pd.DataFrame, max_dist_km: float = 300.0
) -> pd.DataFrame:
    """Pick the track of a member that follows the storm.

    A tracker finds every cyclone on the globe. The storm's track is the one
    closest to the best track at the first time any track lies within
    ``max_dist_km`` of it, the criterion of the reference track extraction.

    Parameters
    ----------
    tracks : pd.DataFrame
        All tracks of one member, with ``track_id``, ``time``, ``lat``, ``lon``
    best_track : pd.DataFrame
        Best track of the storm, with ``time``, ``lat``, ``lon``
    max_dist_km : float, optional
        Largest distance to the best track counting as a match, by default 300

    Returns
    -------
    pd.DataFrame
        The matching track with its distance to the best track in
        ``track_error_km`` (NaN outside the best track's times), empty if no
        track matches
    """
    pairs = tracks.merge(
        best_track[["time", "lat", "lon"]], on="time", suffixes=("", "_best")
    )
    pairs["track_error_km"] = (
        great_circle_distance(
            pairs["lat_best"].to_numpy(),
            pairs["lon_best"].to_numpy(),
            pairs["lat"].to_numpy(),
            pairs["lon"].to_numpy(),
        )
        / 1000.0
    )
    close = pairs[pairs["track_error_km"] < max_dist_km]
    if close.empty:
        return tracks.iloc[0:0].assign(track_error_km=[])

    track_id = close.sort_values(["time", "track_error_km"]).iloc[0]["track_id"]
    track = tracks[tracks["track_id"] == track_id]
    errors = pairs.loc[pairs["track_id"] == track_id, ["time", "track_error_km"]]
    return track.merge(errors, on="time", how="left").reset_index(drop=True)


def load_storm_ensemble(
    track_dir: str, best_track: pd.DataFrame, max_dist_km: float = 300.0
) -> pd.DataFrame:
    """Load the track of a storm from every member's track file.

    Parameters
    ----------
    track_dir : str
        Directory of track files, ``cyclone_tracks_te`` or ``cyclone_tracks_wn2``
    best_track : pd.DataFrame
        Best track of the storm, see :func:`select_storm_track`
    max_dist_km : float, optional
        Largest distance to the best track counting as a match, by default 300

    Returns
    -------
    pd.DataFrame
        The storm's track of every member that has one, with the columns
        ``member``, ``ic``, ``lead_time_h`` and ``track_error_km`` added.
        ``attrs["n_members"]`` is the number of track files, including members
        in which the storm was not found

    Raises
    ------
    FileNotFoundError
        If the directory holds no track files
    """
    members = extract_tracks(track_dir)
    if not members:
        raise FileNotFoundError(f"no track files in {track_dir}")

    storm_tracks = []
    for member in members:
        track = select_storm_track(member["tracks"], best_track, max_dist_km)
        track.insert(0, "member", member["member"])
        track.insert(1, "ic", member["ic"])
        storm_tracks.append(track)

    ensemble = pd.concat(storm_tracks, ignore_index=True)
    lead_time = ensemble["time"] - ensemble["ic"]
    ensemble.insert(3, "lead_time_h", lead_time.dt.total_seconds() / 3600.0)
    if "r17_ne_km" in ensemble:
        quadrants = ensemble[[f"r17_{q}_km" for q in ("ne", "se", "sw", "nw")]]
        ensemble["r17_km"] = quadrants.where(quadrants > 0).mean(axis=1)
    ensemble.attrs["n_members"] = len(members)
    return ensemble


def load_runs(
    outputs_root: str, case: str, best_track: pd.DataFrame, prefix: str = "outputs_"
) -> dict[str, pd.DataFrame]:
    """Load the storm's ensemble of every run of :data:`RUNS` that exists.

    Parameters
    ----------
    outputs_root : str
        Directory holding the runs' directories, ``<prefix><case>_<model>``
    case : str
        Storm, as in the output directory names, e.g. ``"melissa"``
    best_track : pd.DataFrame
        Best track of the storm, see :func:`select_storm_track`
    prefix : str, optional
        Prefix of the run directories, by default ``"outputs_"`` as written by
        ``main.py``; the sample tracks have none

    Returns
    -------
    dict[str, pd.DataFrame]
        Ensemble of each run whose track directory exists, keyed as :data:`RUNS`
    """
    ensembles = {}
    for key, run in RUNS.items():
        track_dir = os.path.join(outputs_root, prefix + run.track_dir.format(case=case))
        if os.path.isdir(track_dir):
            ensembles[key] = load_storm_ensemble(track_dir, best_track)
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


def ensemble_spread(
    ensemble: pd.DataFrame, variable: str, min_members: int = 3
) -> pd.DataFrame:
    """Mean and spread of a variable over the members, at every valid time.

    Parameters
    ----------
    ensemble : pd.DataFrame
        Ensemble of storm tracks, see :func:`load_storm_ensemble`
    variable : str
        Column to summarise
    min_members : int, optional
        Times with fewer members tracking the storm are dropped, by default 3

    Returns
    -------
    pd.DataFrame
        Indexed by time, with ``mean``, ``q10``, ``q90`` and ``members``
    """
    by_time = ensemble.dropna(subset=[variable]).groupby("time")[variable]
    spread = pd.DataFrame(
        {
            "mean": by_time.mean(),
            "q10": by_time.quantile(0.1),
            "q90": by_time.quantile(0.9),
            "members": by_time.count(),
        }
    )
    return spread[spread["members"] >= min_members]


def add_intensity_change(
    track: pd.DataFrame, variable: str = "wind_speed", hours: int = RI_HOURS
) -> pd.DataFrame:
    """Add the change of a variable over the preceding hours, per member.

    Parameters
    ----------
    track : pd.DataFrame
        Best track, or ensemble of tracks with a ``member`` column
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
    earlier = track[[*_member_key(track), "time", variable]].copy()
    earlier["time"] += pd.Timedelta(hours=hours)
    merged = track.merge(
        earlier,
        on=[*_member_key(track), "time"],
        how="left",
        suffixes=("", "_earlier"),
    )
    merged[f"{variable}_change"] = merged[variable] - merged[f"{variable}_earlier"]
    return merged.drop(columns=f"{variable}_earlier")


def _member_key(track: pd.DataFrame) -> list[str]:
    return ["member"] if "member" in track else []


def rapid_intensification_probability(ensemble: pd.DataFrame) -> pd.Series:
    """Share of members intensifying rapidly, at every valid time.

    Parameters
    ----------
    ensemble : pd.DataFrame
        Ensemble of storm tracks, see :func:`load_storm_ensemble`

    Returns
    -------
    pd.Series
        Indexed by time: share of all members of the ensemble whose maximum
        wind rose by at least 15.4 m/s over the preceding 24 h. Members that do
        not track the storm count as not intensifying
    """
    change = add_intensity_change(ensemble)["wind_speed_change"]
    rapid = (change >= RI_THRESHOLD_MS).groupby(ensemble["time"]).sum()
    return rapid / ensemble.attrs["n_members"]


def peak_intensity_table(
    ensembles: Mapping[str, pd.DataFrame], best_track: pd.DataFrame
) -> pd.DataFrame:
    """Peak intensity of the storm: best track against each tracker's members.

    Parameters
    ----------
    ensembles : Mapping[str, pd.DataFrame]
        Ensemble of storm tracks of each run, keyed as :data:`RUNS`
    best_track : pd.DataFrame
        Best track of the storm

    Returns
    -------
    pd.DataFrame
        One row for the best track and one per tracker, with the peak wind and
        the minimum pressure within the forecast period (mean over the
        members, and 10 to 90 % range), and the share of members tracking the
        storm and intensifying rapidly
    """
    start = min(ensemble["time"].min() for ensemble in ensembles.values())
    end = max(ensemble["time"].max() for ensemble in ensembles.values())
    observed = best_track[best_track["time"].between(start, end)]
    observed_change = add_intensity_change(best_track)["wind_speed_change"]
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
    for key, ensemble in ensembles.items():
        members = ensemble.groupby("member")
        change = add_intensity_change(ensemble)
        rapid = change[change["wind_speed_change"] >= RI_THRESHOLD_MS]
        n_members = ensemble.attrs["n_members"]
        rows[RUNS[key].label] = {
            "peak wind [m/s]": _mean_and_range(members["wind_speed"].max()),
            "minimum pressure [hPa]": _mean_and_range(members["msl"].min() / 100),
            "storm tracked": f"{members.ngroups} of {n_members} members",
            "rapid intensification": (
                f"{rapid['member'].nunique()} of {n_members} members"
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _mean_and_range(values: pd.Series) -> str:
    q10, q90 = values.quantile([0.1, 0.9])
    return f"{values.mean():.0f} ({q10:.0f} to {q90:.0f})"


# -------------------------------------------------------------------------- plots


def plot_over_time(
    ax: Axes,
    ensembles: Mapping[str, pd.DataFrame],
    variable: str,
    best_track: pd.DataFrame | None = None,
    analysis_track: pd.DataFrame | None = None,
) -> None:
    """Plot a variable over time: every member thin and pale, the ensemble mean bold.

    Parameters
    ----------
    ax : Axes
        Axes to draw on
    ensembles : Mapping[str, pd.DataFrame]
        Ensemble of storm tracks of each run, keyed as :data:`RUNS`
    variable : str
        Column to plot, a key of :data:`VARIABLES`
    best_track : pd.DataFrame | None, optional
        Best track, drawn in black if it has the variable, by default None
    analysis_track : pd.DataFrame | None, optional
        Track in the analysis, see :func:`load_analysis_track`, by default None
    """
    label, factor = VARIABLES[variable]
    start, end = None, None
    for key, ensemble in ensembles.items():
        if variable not in ensemble:
            continue
        run = RUNS[key]
        for _, member in ensemble.groupby("member"):
            ax.plot(
                member["time"],
                member[variable] * factor,
                color=run.color,
                lw=0.6,
                alpha=0.25,
            )
        spread = ensemble_spread(ensemble, variable) * factor
        ax.plot(
            spread.index,
            spread["mean"],
            color=run.color,
            lw=2.5,
            label=run.label,
        )
        start = spread.index.min() if start is None else min(start, spread.index.min())
        end = spread.index.max() if end is None else max(end, spread.index.max())

    for track, style in (
        (analysis_track, dict(color=REFERENCE_COLOR, ls="--", label="TE on analysis")),
        (best_track, dict(color=BEST_TRACK_COLOR, marker="o", ms=4, label="IBTrACS")),
    ):
        if track is None or variable not in track:
            continue
        shown = track[track["time"].between(start, end)]
        ax.plot(shown["time"], shown[variable] * factor, lw=2, **style)

    ax.set_ylabel(label)
    _style_time_axis(ax)


def plot_wind_pressure(
    ax: Axes, ensembles: Mapping[str, pd.DataFrame], best_track: pd.DataFrame
) -> None:
    """Scatter maximum wind against minimum pressure, for every track point.

    Parameters
    ----------
    ax : Axes
        Axes to draw on
    ensembles : Mapping[str, pd.DataFrame]
        Ensemble of storm tracks of each run, keyed as :data:`RUNS`
    best_track : pd.DataFrame
        Best track of the storm
    """
    for key, ensemble in ensembles.items():
        run = RUNS[key]
        ax.scatter(
            ensemble["msl"] / 100,
            ensemble["wind_speed"],
            s=10,
            color=run.color,
            alpha=0.35,
            linewidths=0,
            label=run.label,
        )
    ax.plot(
        best_track["msl"] / 100,
        best_track["wind_speed"],
        color=BEST_TRACK_COLOR,
        marker="o",
        ms=4,
        lw=1.5,
        label="IBTrACS",
    )
    ax.set_xlabel(VARIABLES["msl"][0])
    ax.set_ylabel(VARIABLES["wind_speed"][0])
    _style_axes(ax)


def plot_rapid_intensification(
    ax: Axes, ensembles: Mapping[str, pd.DataFrame], best_track: pd.DataFrame
) -> None:
    """Plot the share of members intensifying rapidly over time.

    The times the storm itself intensified rapidly are shaded.

    Parameters
    ----------
    ax : Axes
        Axes to draw on
    ensembles : Mapping[str, pd.DataFrame]
        Ensemble of storm tracks of each run, keyed as :data:`RUNS`
    best_track : pd.DataFrame
        Best track of the storm
    """
    for key, ensemble in ensembles.items():
        run = RUNS[key]
        probability = rapid_intensification_probability(ensemble) * 100
        ax.plot(
            probability.index,
            probability,
            color=run.color,
            lw=2,
            marker="o",
            ms=4,
            label=run.label,
        )

    observed = add_intensity_change(best_track)
    rapid = observed[observed["wind_speed_change"] >= RI_THRESHOLD_MS]["time"]
    half_step = pd.Timedelta(hours=3)
    for n, time in enumerate(rapid):
        ax.axvspan(
            time - half_step,
            time + half_step,
            color=BEST_TRACK_COLOR,
            alpha=0.1,
            lw=0,
            label="observed (IBTrACS)" if n == 0 else None,
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("members intensifying by 15.4 m/s in 24 h [%]")
    _style_time_axis(ax)


def plot_tracks(
    ensembles: Mapping[str, pd.DataFrame],
    best_track: pd.DataFrame,
    margin_deg: float = 4.0,
    panel_size: float = 4.5,
) -> Figure:
    """Draw the tracks of all members on maps, one panel per run.

    Every panel shows the best track and covers the same area, so the runs can
    be compared side by side without their tracks hiding each other.

    Parameters
    ----------
    ensembles : Mapping[str, pd.DataFrame]
        Ensemble of storm tracks of each run, keyed as :data:`RUNS`
    best_track : pd.DataFrame
        Best track of the storm
    margin_deg : float, optional
        Margin around the tracks in degrees, by default 4
    panel_size : float, optional
        Width of a panel in inches, by default 4.5

    Returns
    -------
    Figure
        The figure, with a row of panels, or two rows for more than two runs
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    start = min(ensemble["time"].min() for ensemble in ensembles.values())
    end = max(ensemble["time"].max() for ensemble in ensembles.values())
    observed = best_track[best_track["time"].between(start, end)]
    points = pd.concat([observed, *ensembles.values()])
    extent = [
        points["lon"].min() - margin_deg,
        points["lon"].max() + margin_deg,
        points["lat"].min() - margin_deg,
        points["lat"].max() + margin_deg,
    ]

    n_cols = min(len(ensembles), 2)
    n_rows = -(-len(ensembles) // n_cols)
    projection = ccrs.PlateCarree(central_longitude=float(points["lon"].mean()))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * 0.8 * n_rows),
        subplot_kw={"projection": projection},
        squeeze=False,
        layout="constrained",
    )
    geodetic = ccrs.Geodetic()
    for ax, (key, ensemble) in zip(axes.flat, ensembles.items()):
        run = RUNS[key]
        # pastel version of the run's colour, so that the best track stands out
        color = _pastel(run.color)
        for _, track in ensemble.groupby("member"):
            ax.plot(
                track["lon"],
                track["lat"],
                color=color,
                lw=1.1,
                alpha=0.8,
                transform=geodetic,
            )
        ax.plot(
            observed["lon"],
            observed["lat"],
            color=MAP_BEST_TRACK_COLOR,
            lw=2,
            marker="o",
            ms=3,
            transform=geodetic,
            label="IBTrACS",
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR)
        ax.coastlines(lw=0.8, color=COAST_COLOR)
        gridlines = ax.gridlines(draw_labels=True, lw=0.3, color="#c3c2b7")
        gridlines.top_labels = gridlines.right_labels = False
        n_members = ensemble.attrs["n_members"]
        ax.set_title(f"{run.label}, {n_members} members", fontsize="medium")
    for ax in axes.flat[len(ensembles) :]:
        ax.set_visible(False)
    return fig


def _pastel(color: str, whiteness: float = 0.35) -> tuple[float, float, float]:
    """``color`` mixed with white: 0 keeps it, 1 is white."""
    return tuple(c + (1.0 - c) * whiteness for c in to_rgb(color))


def add_header(fig: Figure, title: str) -> None:
    """Give a figure a bold title and, beneath it, one legend for all its axes.

    The figure needs ``layout="constrained"``, which makes room for both above
    the axes. Entries with the same label on several axes are shown once.

    Parameters
    ----------
    fig : Figure
        Figure whose axes have labelled artists
    title : str
        Title of the figure
    """
    entries: dict[str, Artist] = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    # the title heads the legend, so that the layout keeps both clear of the axes
    legend = fig.legend(
        entries.values(),
        entries.keys(),
        loc="outside upper center",
        ncol=len(entries),
        frameon=False,
        title=title,
        title_fontproperties={"weight": "bold", "size": "large"},
    )
    # scatter points are drawn transparent; show them opaque in the legend
    for handle in legend.legend_handles:
        if isinstance(handle, PathCollection):
            handle.set_alpha(1.0)


def _style_axes(ax: Axes) -> None:
    ax.grid(lw=0.4, color="#dddcd5")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _style_time_axis(ax: Axes) -> None:
    import matplotlib.dates as mdates

    _style_axes(ax)
    start, end = ax.get_xlim()  # in days
    # every day, or every second day for spans longer than a week
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if end - start <= 8 else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("valid time [UTC]")
