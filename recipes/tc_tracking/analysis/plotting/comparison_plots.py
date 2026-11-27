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

"""Figures comparing the ensembles of several trackers with a storm's best track.

Used by ``wn2_tracker_comparison.py``. Every function returns a finished
figure: a bold title, beneath it one legend for all panels, then the plots.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from .storm_ensembles import (
    RI_THRESHOLD_MS,
    Ensemble,
    Storm,
    add_intensity_change,
    ensemble_mean,
    forecast_period,
    rapid_intensification_share,
)

BEST_TRACK_COLOR = "#1a1a19"
ANALYSIS_COLOR = "#8a8a84"
GRID_COLOR = "#dddcd5"
# maps: pale ocean and land, the best track slightly softer than in the plots
OCEAN_COLOR = "#e3eef8"
LAND_COLOR = "#e6eedd"
COAST_COLOR = "#8a9a86"
MAP_GRID_COLOR = "#c3c2b7"
MAP_BEST_TRACK_COLOR = "#3d3d3a"


@dataclass(frozen=True)
class Variable:
    """How a column of the tracks is plotted over time."""

    title: str  # in figure titles
    axis_label: str
    factor: float = 1.0  # from the unit of the track files to the plotted one


VARIABLES = {
    "track_error_km": Variable("track error", "distance to best track [km]"),
    "wind_speed": Variable("maximum wind", "maximum 10 m wind [m/s]"),
    "msl": Variable(
        "minimum sea-level pressure", "minimum sea-level pressure [hPa]", 0.01
    ),
    "rmw_km": Variable("radius of maximum wind", "radius of maximum wind [km]"),
    "r17_km": Variable("radius of 17.5 m/s wind", "radius of 17.5 m/s wind [km]"),
}


def plot_tracks(
    ensembles: Mapping[str, Ensemble],
    best_track: pd.DataFrame,
    storm: Storm,
    margin_deg: float = 4.0,
    panel_size: float = 4.5,
) -> Figure:
    """Spaghetti plots: the tracks of all members on maps, one panel per run.

    Every panel shows the best track and covers the same area, so the runs can
    be compared side by side without their tracks hiding each other.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    best_track : pd.DataFrame
        Best track of the storm
    storm : Storm
        The storm, for the title
    margin_deg : float, optional
        Margin around the tracks in degrees, by default 4
    panel_size : float, optional
        Width of a panel in inches, by default 4.5

    Returns
    -------
    Figure
        One row of panels, or two for more than two runs
    """
    observed = best_track[best_track["time"].between(*forecast_period(ensembles))]
    points = pd.concat([observed, *(e.tracks for e in ensembles.values())])
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
    for ax, ensemble in zip(axes.flat, ensembles.values()):
        # pastel version of the run's colour, so that the best track stands out
        color = _pastel(ensemble.run.color)
        for _, track in ensemble.tracks.groupby("member"):
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
        gridlines = ax.gridlines(draw_labels=True, lw=0.3, color=MAP_GRID_COLOR)
        gridlines.top_labels = gridlines.right_labels = False
        ax.set_title(
            f"{ensemble.run.label}, {ensemble.n_members} members", fontsize="medium"
        )
    for ax in axes.flat[len(ensembles) :]:
        ax.set_visible(False)
    _add_header(fig, f"Spaghetti plots {storm.label}")
    return fig


def plot_over_time(
    ensembles: Mapping[str, Ensemble],
    variable: str,
    storm: Storm,
    best_track: pd.DataFrame | None = None,
    analysis_track: pd.DataFrame | None = None,
) -> Figure:
    """A variable over time: every member thin and pale, the ensemble mean bold.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    variable : str
        Column to plot, a key of :data:`VARIABLES`
    storm : Storm
        The storm, for the title
    best_track : pd.DataFrame | None, optional
        Best track, drawn in black if it has the variable, by default None
    analysis_track : pd.DataFrame | None, optional
        Track in the analysis, see :func:`~plotting.storm_ensembles.load_analysis_track`,
        by default None

    Returns
    -------
    Figure
        The plot
    """
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    _draw_over_time(ax, ensembles, variable, best_track, analysis_track)
    _add_header(fig, f"Ensemble mean {VARIABLES[variable].title} {storm.label}")
    return fig


def plot_storm_size(
    ensembles: Mapping[str, Ensemble], best_track: pd.DataFrame, storm: Storm
) -> Figure:
    """Radius of maximum wind and radius of 17.5 m/s wind over time.

    Only runs reporting the storm's size, the WN2 tracker, are drawn; the time
    axis spans the forecasts of all runs, as in the other plots.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    best_track : pd.DataFrame
        Best track of the storm
    storm : Storm
        The storm, for the title

    Returns
    -------
    Figure
        One panel per radius
    """
    variables = ("rmw_km", "r17_km")
    fig, axes = plt.subplots(1, len(variables), figsize=(12, 4.5), layout="constrained")
    for ax, variable in zip(axes, variables):
        _draw_over_time(ax, ensembles, variable, best_track)
        ax.set_title(VARIABLES[variable].title.capitalize())
    _add_header(fig, f"Ensemble mean storm size {storm.label}")
    return fig


def plot_wind_pressure(
    ensembles: Mapping[str, Ensemble], best_track: pd.DataFrame, storm: Storm
) -> Figure:
    """Maximum wind against minimum pressure, for every track point.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    best_track : pd.DataFrame
        Best track of the storm
    storm : Storm
        The storm, for the title

    Returns
    -------
    Figure
        The scatter plot
    """
    wind, pressure = VARIABLES["wind_speed"], VARIABLES["msl"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for ensemble in ensembles.values():
        ax.scatter(
            ensemble.tracks["msl"] * pressure.factor,
            ensemble.tracks["wind_speed"],
            s=10,
            color=ensemble.run.color,
            alpha=0.35,
            linewidths=0,
            label=ensemble.run.label,
        )
    ax.plot(
        best_track["msl"] * pressure.factor,
        best_track["wind_speed"],
        color=BEST_TRACK_COLOR,
        marker="o",
        ms=4,
        lw=1.5,
        label="IBTrACS",
    )
    ax.set_xlabel(pressure.axis_label)
    ax.set_ylabel(wind.axis_label)
    _style_axes(ax)
    _add_header(fig, f"Wind-pressure relationship {storm.label}")
    return fig


def plot_rapid_intensification(
    ensembles: Mapping[str, Ensemble], best_track: pd.DataFrame, storm: Storm
) -> Figure:
    """Share of members intensifying rapidly over time.

    The times the storm itself intensified rapidly are shaded.

    Parameters
    ----------
    ensembles : Mapping[str, Ensemble]
        The storm's tracks of each run
    best_track : pd.DataFrame
        Best track of the storm
    storm : Storm
        The storm, for the title

    Returns
    -------
    Figure
        The plot
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for ensemble in ensembles.values():
        share = rapid_intensification_share(ensemble) * 100
        ax.plot(
            share.index,
            share,
            color=ensemble.run.color,
            lw=2,
            marker="o",
            ms=4,
            label=ensemble.run.label,
        )

    observed = add_intensity_change(best_track)
    rapid = observed.loc[observed["wind_speed_change"] >= RI_THRESHOLD_MS, "time"]
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
    ax.set_ylabel(f"members intensifying by {RI_THRESHOLD_MS} m/s in 24 h [%]")
    _style_time_axis(ax, forecast_period(ensembles))
    _add_header(fig, f"Rapid intensification {storm.label}")
    return fig


def _draw_over_time(
    ax: Axes,
    ensembles: Mapping[str, Ensemble],
    variable: str,
    best_track: pd.DataFrame | None = None,
    analysis_track: pd.DataFrame | None = None,
) -> None:
    """Draw members and ensemble means of a variable, and the reference tracks."""
    shown = VARIABLES[variable]
    for ensemble in ensembles.values():
        if variable not in ensemble.tracks:
            continue
        color = ensemble.run.color
        for _, member in ensemble.tracks.groupby("member"):
            ax.plot(
                member["time"],
                member[variable] * shown.factor,
                color=color,
                lw=0.6,
                alpha=0.25,
            )
        mean = ensemble_mean(ensemble.tracks, variable) * shown.factor
        ax.plot(mean.index, mean, color=color, lw=2.5, label=ensemble.run.label)

    period = forecast_period(ensembles)
    for track, style in (
        (analysis_track, dict(color=ANALYSIS_COLOR, ls="--", label="TE on analysis")),
        (best_track, dict(color=BEST_TRACK_COLOR, marker="o", ms=4, label="IBTrACS")),
    ):
        if track is None or variable not in track:
            continue
        track = track[track["time"].between(*period)]
        ax.plot(track["time"], track[variable] * shown.factor, lw=2, **style)

    ax.set_ylabel(shown.axis_label)
    _style_time_axis(ax, period)


def _add_header(fig: Figure, title: str) -> None:
    """Give a figure a bold title and, beneath it, one legend for all its axes.

    The figure needs ``layout="constrained"``, which makes room for both above
    the axes. Entries with the same label on several axes are shown once.
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


def _pastel(color: str, whiteness: float = 0.35) -> tuple[float, float, float]:
    """``color`` mixed with white: 0 keeps it, 1 is white."""
    red, green, blue = (c + (1.0 - c) * whiteness for c in to_rgb(color))
    return red, green, blue


def _style_axes(ax: Axes) -> None:
    ax.grid(lw=0.4, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _style_time_axis(ax: Axes, period: tuple[pd.Timestamp, pd.Timestamp]) -> None:
    """Style a time axis spanning ``period``, the same in every plot of a storm."""
    _style_axes(ax)
    start, end = period
    margin = pd.Timedelta(hours=6)
    ax.set_xlim(start - margin, end + margin)
    # every day, or every second day for spans longer than a week
    days = (end - start) / pd.Timedelta(days=1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if days <= 8 else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("valid time [UTC]")
