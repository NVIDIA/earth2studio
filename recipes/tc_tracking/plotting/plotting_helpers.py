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

import os
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_handling import merge_tracks_by_time
from matplotlib.collections import LineCollection

_DEFAULT_TIME_STEP = np.timedelta64(6, "h")
_DEFAULT_P_REF_PA = 101325


def _make_suptitle(
    case: str,
    ic: np.datetime64 | None = None,
    n_tracks: int | None = None,
    n_members: int | None = None,
) -> str:
    """Build a standardised plot suptitle from storm metadata."""
    title = case.split("_")[0].upper()
    if ic is not None:
        title += f"\n initialised on {ic}"
    if n_tracks is not None and n_members is not None:
        title += f"\n {n_tracks} tracks in {n_members} ensemble members"
    return title


def _var_display_info(var: str) -> tuple[str, str, float]:
    """Return ``(display_label, unit_string, scale_divisor)`` for a tracked variable."""
    info = {
        "msl": ("msl", "hPa", 100),
        "dist": ("distance", "km", 1000),
        "wind_speed": ("maximum instantaneous wind speed", "m/s", 1),
    }
    return info.get(var, (var, "", 1))


def add_some_gap(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[float, float, float, float]:
    """Expand a lat/lon bounding box by 10 % on each side and correct extreme aspect ratios.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in degrees.
    lon_min, lon_max : float
        Longitude bounds in degrees.

    Returns
    -------
    tuple[float, float, float, float]
        ``(lat_min, lat_max, lon_min, lon_max)`` with padding applied.
    """
    gap_fac = 0.1
    lat_gap = (lat_max - lat_min) * gap_fac
    lon_gap = (lon_max - lon_min) * gap_fac

    lat_min, lat_max = lat_min - lat_gap, lat_max + lat_gap
    lon_min, lon_max = lon_min - lon_gap, lon_max + lon_gap

    if lat_gap / lon_gap > 2:
        d_lon = 0.5 * (lat_max - lat_min)
        med_lon = 0.5 * (lon_min + lon_max)
        lon_min, lon_max = med_lon - d_lon / 2, med_lon + d_lon / 2

    elif lon_gap / lat_gap > 2:
        d_lat = 0.5 * (lon_max - lon_min)
        med_lat = 0.5 * (lat_min + lat_max)
        lat_min, lat_max = med_lat - d_lat / 2, med_lat + d_lat / 2

    return lat_min, lat_max, lon_min, lon_max


def get_central_coords(track: pd.DataFrame) -> tuple[float, float]:
    """Return the median latitude and longitude of a track.

    Parameters
    ----------
    track : pd.DataFrame
        Track with ``lat`` and ``lon`` columns.

    Returns
    -------
    tuple[float, float]
        ``(lat_median, lon_median)``
    """
    lat_cen = track["lat"].median()
    lon_cen = track["lon"].median()

    return lat_cen, lon_cen


def plot_spaghetti(
    true_track: pd.DataFrame,
    pred_tracks: list[dict[str, Any]],
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    out_dir: str | None = None,
    alpha: float = 0.2,
    line_width: float = 2,
    ic: np.datetime64 | list[np.datetime64] | None = None,
) -> None:
    """Plot ensemble track trajectories (spaghetti plot) on a map.

    Parameters
    ----------
    true_track : pd.DataFrame
        Reference track (plotted in red).
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    ensemble_mean : dict[str, Any]
        Ensemble-mean track with ``"lat"`` and ``"lon"`` arrays.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members (including unmatched).
    out_dir : str | None, optional
        If provided, the figure is saved here.
    alpha : float, optional
        Transparency for ensemble member lines, by default 0.2
    line_width : float, optional
        Line width for all tracks, by default 2
    ic : np.datetime64 | list[np.datetime64] | None, optional
        If provided, only plot members whose ``"ic"`` is in *ic*.
    """
    plt.close("all")

    lat_cen, lon_cen = get_central_coords(true_track)

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    projection = ccrs.LambertAzimuthalEqualArea(
        central_longitude=lon_cen, central_latitude=lat_cen
    )
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.RIVERS, lw=0.5)
    if case != "debbie_2017_southern_pacific":  # cartopy issues with small islands
        ax.add_feature(cfeature.OCEAN, facecolor="#b0c4de")
        ax.add_feature(cfeature.LAND, facecolor="#c4b9a3")

    # Seed lat/lon bounds from the true track so the loop only needs to widen
    # them; works even when ``ic`` filters out every predicted member.
    lat_min, lat_max = true_track["lat"].min(), true_track["lat"].max()
    lon_min, lon_max = true_track["lon"].min(), true_track["lon"].max()

    segments = []
    for _track in pred_tracks:
        track = _track["tracks"]
        if ic is not None and _track["ic"] not in ic:
            continue

        lat_min, lat_max = min(lat_min, track["lat"].min()), max(
            lat_max, track["lat"].max()
        )
        lon_min, lon_max = min(lon_min, track["lon"].min()), max(
            lon_max, track["lon"].max()
        )

        segments.append(np.column_stack([track["lon"].values, track["lat"].values]))

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=line_width,
                alpha=alpha,
                transform=ccrs.PlateCarree(),
            )
        )

    ax.plot(
        true_track["lon"],
        true_track["lat"],
        transform=ccrs.PlateCarree(),
        color="red",
        linewidth=line_width,
        alpha=1.0,
    )

    ax.plot(
        ensemble_mean["lon"],
        ensemble_mean["lat"],
        transform=ccrs.PlateCarree(),
        color="lime",
        linewidth=line_width,
        alpha=1.0,
    )

    lat_min, lat_max, lon_min, lon_max = add_some_gap(
        lat_min, lat_max, lon_min, lon_max
    )

    plt.tight_layout()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)

    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{case}_tracks.png"))


def normalised_intensities(
    track: pd.DataFrame,
    tru_track: pd.DataFrame,
    var: str,
    p_ref: float = _DEFAULT_P_REF_PA,
) -> pd.DataFrame:
    """Normalise a track variable relative to the reference track.

    For pressure (``msl``), the normalisation is
    ``(pred - ref) / (p_ref - ref)``.  For other variables it is
    ``(pred - ref) / ref``.

    Parameters
    ----------
    track : pd.DataFrame
        Predicted or ensemble-mean track.
    tru_track : pd.DataFrame
        Reference track.
    var : str
        Variable name to normalise.
    p_ref : float, optional
        Reference pressure (Pa) used for the ``msl`` normalisation, by
        default 101 325 Pa.

    Returns
    -------
    pd.DataFrame
        Merged frame with *var* replaced by its normalised values.
    """
    merged_track = merge_tracks_by_time(track, tru_track)

    if var == "msl":
        merged_track[var] = (merged_track[var] - merged_track[var + "_tru"]) / (
            p_ref - merged_track[var + "_tru"]
        )
    else:
        merged_track[var] = (
            merged_track[var] - merged_track[var + "_tru"]
        ) / merged_track[var + "_tru"]

    return merged_track


def plot_relative_over_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    ics: np.datetime64 | list[np.datetime64] | None = None,
    out_dir: str | None = None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> None:
    """Plot normalised intensity deviations from the reference track over time.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict with ``"time"`` and ``"mean"`` keys.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members.
    ics : np.datetime64 | list[np.datetime64] | None, optional
        If provided, only plot members whose ``"ic"`` is in *ics*.
    out_dir : str | None, optional
        If provided, the figure is saved here.
    time_step : np.timedelta64, optional
        Model time step, by default 6 h.
    """
    fig, _ax = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    variables = ["msl", "wind_speed"]
    labels = [
        "(msl - msl_ref)/(101325Pa - msl_ref)",
        "max_wind/max_wind_ref - 1",
    ]

    rel_steps = int(
        ((tru_track["time"].max() - pred_tracks[0]["ic"]) / time_step + 1) * 0.75
    )

    for ii in range(_ax.shape[0]):
        ax = _ax[ii]
        # ``+inf``/``-inf`` are the natural identity values for a running
        # min/max and survive an empty inner loop without flipping the axis.
        vmin, vmax = float("inf"), float("-inf")
        for _track in pred_tracks:
            track = _track["tracks"]
            if ics is not None and _track["ic"] not in ics:
                continue

            track = normalised_intensities(track, tru_track, variables[ii])

            vmin = min(vmin, track[variables[ii]][:rel_steps].min())
            vmax = max(vmax, track[variables[ii]][:rel_steps].max())

            ax.plot(track["time"], track[variables[ii]], color="black", alpha=0.1)

        ax.set_ylabel(labels[ii])
        ax.grid(True)
        if np.isfinite(vmin) and np.isfinite(vmax):
            ax.set_ylim(vmin, vmax)

        ax.plot(
            tru_track["time"],
            np.zeros(len(tru_track)),
            color="orangered",
            linewidth=2.5,
            label="era5 comparison",
        )

        mean = pd.DataFrame(
            {
                "time": ensemble_mean["time"],
                variables[ii]: ensemble_mean["mean"][variables[ii]],
            }
        )
        _track = normalised_intensities(mean, tru_track, variables[ii])
        ax.plot(
            _track["time"],
            _track[variables[ii]],
            color="lime",
            linewidth=2.5,
            label="ensemble mean",
            linestyle="--",
        )

        ax.legend()

    _ax[-1].set_xlabel("time [UTC]")

    plt.xlim(
        pred_tracks[0]["ic"] - time_step,
        tru_track["time"].max() + time_step,
    )

    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_rel_intensities.png"))


def plot_over_time(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    n_members: int,
    variables: list[str] | None = None,
    labels: list[str] | None = None,
    ics: np.datetime64 | list[np.datetime64] | None = None,
    out_dir: str | None = None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> None:
    """Plot absolute intensity and distance time series for all ensemble members.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict.
    case : str
        Storm identifier for the plot title.
    n_members : int
        Total number of ensemble members.
    variables : list[str] | None, optional
        Variables to plot (one subplot each), by default
        ``["msl", "wind_speed", "dist"]``
    labels : list[str] | None, optional
        Y-axis labels corresponding to *variables*.  Auto-derived from
        :func:`_var_display_info` when *None*.
    ics : np.datetime64 | list[np.datetime64] | None, optional
        If provided, only plot members whose ``"ic"`` is in *ics*.
    out_dir : str | None, optional
        If provided, the figure is saved here.
    time_step : np.timedelta64, optional
        Model time step, by default 6 h.
    """
    if variables is None:
        variables = ["msl", "wind_speed", "dist"]
    if labels is None:
        labels = [
            f"{_var_display_info(v)[0]} [{_var_display_info(v)[1]}]" for v in variables
        ]

    fig, _ax = plt.subplots(len(variables), 1, figsize=(11, 15), sharex=True)
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"], len(pred_tracks), n_members),
        fontsize=16,
    )

    # Seed from the first predicted track and widen as the loop progresses.
    first_times = pred_tracks[0]["tracks"]["time"]
    t_min, t_max = first_times.min(), first_times.max()

    for ii in range(_ax.shape[0]):
        _, _, scale = _var_display_info(variables[ii])

        for _track in pred_tracks:
            track = _track["tracks"]
            if ics is not None and _track["ic"] not in ics:
                continue

            _ax[ii].plot(
                track["time"], track[variables[ii]] / scale, color="black", alpha=0.1
            )

            t_min, t_max = min(t_min, track["time"].min()), max(
                t_max, track["time"].max()
            )

        _ax[ii].set_xlim(t_min - time_step, t_max + time_step)
        _ax[ii].set_ylabel(labels[ii])
        _ax[ii].grid(True)

        _ax[ii].plot(
            tru_track["time"],
            tru_track[variables[ii]] / scale,
            color="orangered",
            linewidth=2.5,
            label="era5 comparison",
        )

        _ax[ii].plot(
            ensemble_mean["time"],
            ensemble_mean["mean"][variables[ii]] / scale,
            color="lime",
            linewidth=2.5,
            label="ensemble mean",
            linestyle="--",
        )
        _ax[ii].legend()

    _ax[-1].set_xlabel("time [UTC]")

    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_abs_intensities.png"))


def plot_ib_era5(
    tru_track: pd.DataFrame,
    case: str,
    variables: list[str] | None = None,
    out_dir: str | None = None,
    p_ref: float = _DEFAULT_P_REF_PA,
) -> None:
    """Plot ERA5-vs-IBTrACS intensity ratios on twin y-axes.

    Parameters
    ----------
    tru_track : pd.DataFrame
        Reference track containing both ERA5 and IBTrACS columns.
    case : str
        Storm identifier for the plot title.
    variables : list[str] | None, optional
        Variables to compare (``"msl"`` and/or ``"wind_speed"``), by
        default ``["msl", "wind_speed"]``
    out_dir : str | None, optional
        If provided, the figure is saved here.
    p_ref : float, optional
        Reference pressure (Pa) used for the ``msl`` ratio, by default
        101 325 Pa.
    """
    if variables is None:
        variables = ["msl", "wind_speed"]

    plt.close("all")

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(_make_suptitle(case), fontsize=16)

    ax2 = ax1.twinx()

    if "msl" in variables:
        ax1.plot(
            tru_track["time"],
            (p_ref - tru_track["msl"]) / (p_ref - tru_track["msl_ib"]),
            "black",
        )
        ax1.set_ylabel("(1013.25hPa-msl_era5)/(1013.25hPa-msl_ib)", color="black")

    if "wind_speed" in variables:
        ax2.plot(
            tru_track["time"],
            tru_track["wind_speed"] / tru_track["wind_speed_ib"],
            "orangered",
        )
        ax2.set_ylabel("wind_speed_era5/wind_speed_ib", color="orangered")

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_ib_era5_wind_speed.png"))


def root_metrics(
    err_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Replace MSE/variance with RMSE/standard-deviation and drop member counts.

    Parameters
    ----------
    err_dict : dict[str, dict[str, np.ndarray]]
        Per-variable error metrics (modified in place).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Updated *err_dict*.
    """
    for var in err_dict.keys():
        mse = err_dict[var].pop("mse")
        err_dict[var]["rmse"] = np.sqrt(mse)
        variance = err_dict[var].pop("variance")
        err_dict[var]["standard_deviation"] = np.sqrt(variance)
        err_dict[var].pop("n_members")

    return err_dict


def plot_errors_over_lead_time(
    err_dict: dict[str, dict[str, np.ndarray]],
    case: str,
    ic: np.datetime64,
    n_members: int,
    n_tracks: int,
    norm_dict: dict[str, float] | None = None,
    unit_dict: dict[str, str] | None = None,
    out_dir: str | None = None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> None:
    """Plot error metrics (RMSE, MAE, standard deviation) as a function of lead time.

    Parameters
    ----------
    err_dict : dict[str, dict[str, np.ndarray]]
        Per-variable error metrics.
    case : str
        Storm identifier for the plot title.
    ic : np.datetime64
        Initial condition timestamp.
    n_members : int
        Total number of ensemble members.
    n_tracks : int
        Number of matched tracks.
    norm_dict : dict[str, float] | None, optional
        Normalisation divisors for display units.  Auto-derived from
        :func:`_var_display_info` when *None*.
    unit_dict : dict[str, str] | None, optional
        Display unit strings.  Auto-derived when *None*.
    out_dir : str | None, optional
        If provided, the figure is saved here.
    time_step : np.timedelta64, optional
        Model time step used for the lead-time axis, by default 6 h.
    """
    if "mse" in err_dict[list(err_dict.keys())[0]].keys():
        err_dict = root_metrics(err_dict)

    if norm_dict is None:
        norm_dict = {v: _var_display_info(v)[2] for v in err_dict}
    if unit_dict is None:
        unit_dict = {v: _var_display_info(v)[1] for v in err_dict}

    variables = list(err_dict.keys())
    metrics = list(err_dict[variables[0]].keys())

    for extreme in ["min", "max"]:
        if extreme in metrics:
            metrics.remove(extreme)

    lead_time = np.arange(err_dict[variables[0]][metrics[0]].shape[0]) * time_step

    fig, ax = plt.subplots(
        len(variables),
        len(metrics),
        figsize=((len(metrics) + 1) * 2, (len(variables) + 1) * 2),
        sharex=True,
    )

    for ivar, var in enumerate(err_dict.keys()):
        for imet, metric in enumerate(metrics):

            ax[ivar, imet].plot(lead_time, err_dict[var][metric] / norm_dict[var])

            if ivar == 0:
                ax[ivar, imet].set_title(metric, fontsize=12, weight="bold")

            if imet == 0:
                ax[ivar, imet].set_ylabel(
                    f"{var} [{unit_dict[var]}]", fontsize=12, weight="bold"
                )

            if ivar == len(variables) - 1:
                ax[ivar, imet].set_xlabel("lead time [h]", fontsize=12)

    fig.suptitle(_make_suptitle(case, ic, n_tracks, n_members), fontsize=16)

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"{case}_error_metrics_over_lead_time.png"))


def extract_reference_extremes(
    tru_track: pd.DataFrame,
    pred_tracks: list[dict[str, Any]],
    ens_mean: dict[str, Any],
    variables: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract per-member extreme values and the corresponding reference extremes.

    Parameters
    ----------
    tru_track : pd.DataFrame
        Reference track.
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    ens_mean : dict[str, Any]
        Ensemble statistics dict.
    variables : list[str]
        Variables to extract extremes for.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-variable dict with ``"pred"`` (array), ``"tru"`` (scalar),
        and ``"ens_mean"`` (scalar).
    """
    extreme_dict: dict[str, dict[str, Any]] = {}
    for var in variables:
        if var in ["wind_speed"]:
            reduce_fn = np.nanmax
        elif var in ["msl"]:
            reduce_fn = np.nanmin
        else:
            continue

        extreme_dict[var] = {
            "pred": np.zeros(len(pred_tracks)),
            "tru": reduce_fn(tru_track[var]),
            "ens_mean": reduce_fn(ens_mean["mean"][var]),
        }
        for ii, track in enumerate(pred_tracks):
            extreme_dict[var]["pred"][ii] = reduce_fn(track["tracks"][var])

    return extreme_dict


def add_stats_box(
    ax: plt.Axes,
    pred_var: np.ndarray,
    tru_var: float,
    var: str,
    reduction: str,
    unit: str,
) -> None:
    """Add a text box with summary statistics below a histogram axis.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to annotate.
    pred_var : np.ndarray
        Per-member extreme values.
    tru_var : float
        Reference extreme value.
    var : str
        Variable name.
    reduction : str
        ``"max"`` or ``"min"``.
    unit : str
        Display unit string.
    """
    # For wind speed, "more intense" means larger values; for msl it means
    # smaller values.  Pick the comparator that actually counts members that
    # are *more intense* than the reference.
    if var == "wind_speed":
        n_beyond_ref = int((pred_var > tru_var).sum())
        comp = "exceeding"
    else:
        n_beyond_ref = int((pred_var < tru_var).sum())
        comp = "below"
    n_total = len(pred_var)

    stats = [
        ("era5 reference:", f"{tru_var:.1f} {unit}"),
        (
            f"members {comp} ref:",
            f"{n_beyond_ref} of {n_total} ({(n_beyond_ref/n_total)*100:.1f}%)",
        ),
        (f"max {reduction} {var}:", f"{pred_var.max():.1f} {unit}"),
        (f"min {reduction} {var}:", f"{pred_var.min():.1f} {unit}"),
        (f"avg {reduction} {var}:", f"{pred_var.mean():.1f} {unit}"),
        (f"std {reduction} {var}:", f"{pred_var.std():.1f} {unit}"),
    ]

    max_label_width = max(len(label) for label, _ in stats)
    text = "\n".join([f"{label:<{max_label_width}}  {value}" for label, value in stats])

    ax.text(
        0.01,
        -0.25,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        fontfamily="monospace",
    )


def plot_extreme_extremes_histograms(
    pred_tracks: list[dict[str, Any]],
    tru_track: pd.DataFrame,
    ensemble_mean: dict[str, Any],
    case: str,
    variables: list[str] | None = None,
    out_dir: str | None = None,
    nbins: int = 12,
) -> None:
    """Plot histograms of per-member extreme values with reference lines.

    Parameters
    ----------
    pred_tracks : list[dict[str, Any]]
        Matched prediction dicts.
    tru_track : pd.DataFrame
        Reference track.
    ensemble_mean : dict[str, Any]
        Ensemble statistics dict.
    case : str
        Storm identifier for the plot title.
    variables : list[str] | None, optional
        Variables to plot (one subplot each), by default
        ``["wind_speed", "msl"]``
    out_dir : str | None, optional
        If provided, the figure is saved here.
    nbins : int, optional
        Number of histogram bins, by default 12
    """
    if variables is None:
        variables = ["wind_speed", "msl"]

    extreme_dict = extract_reference_extremes(
        tru_track, pred_tracks, ensemble_mean, variables
    )

    fig, ax = plt.subplots(
        1, len(variables), figsize=(3 * (len(variables) + 1), 6), sharey=True
    )
    fig.suptitle(
        _make_suptitle(case, pred_tracks[0]["ic"]),
        fontsize=16,
    )
    ax[0].set_ylabel("count")

    for ii, var in enumerate(variables):

        reduction = "max" if var in ["wind_speed"] else "min"
        _, unit, scale = _var_display_info(var)

        pred_var = extreme_dict[var]["pred"] / scale
        tru_var = extreme_dict[var]["tru"] / scale
        mean_var = extreme_dict[var]["ens_mean"] / scale

        ax[ii].hist(pred_var, bins=nbins)
        ax[ii].axvline(
            tru_var, color="orangered", linestyle="--", label="era5 reference"
        )
        ax[ii].axvline(mean_var, color="lime", linestyle="--", label="ensemble mean")

        ax[ii].set_title(f"{reduction} {var} (x, t)")
        ax[ii].set_xlabel(f"{var} [{unit}]")
        ax[ii].legend()

        add_stats_box(ax[ii], pred_var, tru_var, var, reduction, unit)

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f"{case}_histograms.png"))
