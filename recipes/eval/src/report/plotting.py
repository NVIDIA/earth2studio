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

"""Matplotlib-based plotting primitives for report figures."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from .aggregation import (  # noqa: E402
    aggregate_over_ensemble,
    aggregate_over_time,
    compute_spread_skill,
    display_name,
)

try:
    import cartopy.crs as ccrs

    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Lead-time axis formatting
# ---------------------------------------------------------------------------


_LEAD_TIME_AXIS_UNITS: dict[str, tuple[np.timedelta64, str]] = {
    "days": (np.timedelta64(1, "D"), "days"),
    "hours": (np.timedelta64(1, "h"), "hours"),
    "minutes": (np.timedelta64(1, "m"), "minutes"),
    # Aliases
    "day": (np.timedelta64(1, "D"), "days"),
    "hour": (np.timedelta64(1, "h"), "hours"),
    "minute": (np.timedelta64(1, "m"), "minutes"),
    "min": (np.timedelta64(1, "m"), "minutes"),
    "h": (np.timedelta64(1, "h"), "hours"),
    "d": (np.timedelta64(1, "D"), "days"),
}


def _lead_time_axis(
    lead_times: np.ndarray, unit: str = "days"
) -> tuple[np.ndarray, str]:
    """Convert timedelta64 lead times to fractional *unit* for plotting.

    Returns ``(values, axis_label)``.  ``unit`` defaults to ``"days"`` so
    existing callers retain their behavior when a report config omits
    ``lead_time_axis_unit``; StormScope-scale campaigns override to
    ``"hours"`` or ``"minutes"``.  Accepts the canonical names plus
    short aliases (``h``, ``d``, ``min``) for convenience.
    """
    key = unit.lower()
    if key not in _LEAD_TIME_AXIS_UNITS:
        logger.warning(
            f"Unknown lead_time_axis_unit '{unit}'; "
            f"falling back to 'days'.  Known: {sorted(set(_LEAD_TIME_AXIS_UNITS.values()))}"
        )
        key = "days"
    divisor, label = _LEAD_TIME_AXIS_UNITS[key]
    return lead_times / divisor, f"Lead time ({label})"


# Line styles cycled for time-group overlays so each group is distinguishable.
_GROUP_LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1))]


# ---------------------------------------------------------------------------
# Metric-vs-leadtime plots
# ---------------------------------------------------------------------------


def plot_metric_vs_leadtime(
    ds: xr.Dataset,
    metric_name: str,
    variables: list[str],
    variable_group_name: str | None = None,
    time_groups: dict[str, np.ndarray] | None = None,
    time_unit: str = "days",
) -> Figure:
    """Plot a metric as a function of lead time, one line per variable.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_name : str
        Metric name (e.g. ``"mse"``).
    variables : list[str]
        Variable names to plot.
    variable_group_name : str | None
        Label for the variable group (used in title).
    time_groups : dict[str, np.ndarray] | None
        If provided, plots one set of lines per time group using
        distinct line styles in addition to the "all" line (solid).
    time_unit : str
        Lead-time axis unit (``days`` / ``hours`` / ``minutes``).

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    xlabel = "Lead time"
    for var in variables:
        array_name = f"{metric_name}__{var}"
        if array_name not in ds:
            continue
        da = aggregate_over_ensemble(ds[array_name])

        # "All times" line
        agg = aggregate_over_time(da, metric_name)
        if "lead_time" not in agg.dims:
            continue
        axis_vals, xlabel = _lead_time_axis(agg.lead_time.values, time_unit)
        line = ax.plot(axis_vals, agg.values, label=var, linewidth=1.5)
        color = line[0].get_color()

        # Per-group lines with distinct styles
        if time_groups:
            for i, (group_name, group_times) in enumerate(time_groups.items()):
                da_sub = da.sel(time=group_times)
                agg_sub = aggregate_over_time(da_sub, metric_name)
                ls = _GROUP_LINESTYLES[i % len(_GROUP_LINESTYLES)]
                ax.plot(
                    axis_vals,
                    agg_sub.values,
                    color=color,
                    linestyle=ls,
                    alpha=0.6,
                    linewidth=1.0,
                    label=f"{var} ({group_name})",
                )

    ylabel = display_name(metric_name)
    title = ylabel
    if variable_group_name:
        title += f" — {variable_group_name}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _grid_shape(n: int, max_cols: int = 4) -> tuple[int, int]:
    """Pick ``(nrows, ncols)`` for *n* subplots.

    Aims for a roughly landscape-oriented grid (wider than tall)
    with at most *max_cols* columns.
    """
    if n <= 0:
        return 1, 1
    if n <= max_cols:
        return 1, n
    ncols = min(math.ceil(math.sqrt(n * 1.3)), max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# Subplot dimensions (inches) when used inside a grid.
_SUBPLOT_WIDTH = 5
_SUBPLOT_HEIGHT = 4


def _plot_leadtime_on_ax(
    ax: plt.Axes,
    ds: xr.Dataset,
    metric_name: str,
    variables: list[str],
    time_groups: dict[str, np.ndarray] | None = None,
    time_unit: str = "days",
) -> str:
    """Draw lead-time curves for one variable group on a single axes.

    Returns the x-axis label derived from *time_unit* so the caller can
    apply it consistently across a grid of axes.
    """
    xlabel = "Lead time"
    for var in variables:
        array_name = f"{metric_name}__{var}"
        if array_name not in ds:
            continue
        da = aggregate_over_ensemble(ds[array_name])

        agg = aggregate_over_time(da, metric_name)
        if "lead_time" not in agg.dims:
            continue
        axis_vals, xlabel = _lead_time_axis(agg.lead_time.values, time_unit)
        line = ax.plot(axis_vals, agg.values, label=var, linewidth=1.5)
        color = line[0].get_color()

        if time_groups:
            for i, (group_name, group_times) in enumerate(time_groups.items()):
                da_sub = da.sel(time=group_times)
                agg_sub = aggregate_over_time(da_sub, metric_name)
                ls = _GROUP_LINESTYLES[i % len(_GROUP_LINESTYLES)]
                ax.plot(
                    axis_vals,
                    agg_sub.values,
                    color=color,
                    linestyle=ls,
                    alpha=0.6,
                    linewidth=1.0,
                    label=f"{var} ({group_name})",
                )
    return xlabel


def plot_metric_grid(
    ds: xr.Dataset,
    metric_name: str,
    variable_groups: OrderedDict[str, list[str]],
    time_groups: dict[str, np.ndarray] | None = None,
    time_unit: str = "days",
) -> Figure:
    """Plot lead-time curves for all variable groups in a subplot grid.

    Each subplot shows one variable group.  The metric's display name
    is used as the figure suptitle.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_name : str
        Metric name (e.g. ``"mse"``).
    variable_groups : OrderedDict[str, list[str]]
        Mapping from group name to variable list.  Only groups with
        at least one available variable should be passed.
    time_groups : dict[str, np.ndarray] | None
        If provided, each subplot gets per-group overlay lines.

    Returns
    -------
    Figure
    """
    n = len(variable_groups)
    nrows, ncols = _grid_shape(n)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_SUBPLOT_WIDTH * ncols, _SUBPLOT_HEIGHT * nrows),
        squeeze=False,
    )

    group_items = list(variable_groups.items())

    xlabel = "Lead time"
    for idx, (group_name, group_vars) in enumerate(group_items):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        xlabel = _plot_leadtime_on_ax(
            ax, ds, metric_name, group_vars, time_groups, time_unit=time_unit
        )
        ax.set_title(group_name, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        # Only label x-axis on the bottom row.
        if row == nrows - 1:
            ax.set_xlabel(xlabel)
        # Only label y-axis on the left column.
        if col == 0:
            ax.set_ylabel(display_name(metric_name))

    # Hide empty cells.
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(display_name(metric_name), fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_ic_heatmap(
    ds: xr.Dataset,
    metric_name: str,
    variable: str,
    time_unit: str = "days",
) -> Figure:
    """Plot a time x lead_time heatmap for one metric/variable.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_name : str
        Metric name.
    variable : str
        Variable name.
    time_unit : str
        Lead-time axis unit (``days`` / ``hours`` / ``minutes``).

    Returns
    -------
    Figure
    """
    array_name = f"{metric_name}__{variable}"
    da = aggregate_over_ensemble(ds[array_name])

    if "lead_time" not in da.dims or "time" not in da.dims:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "Insufficient dimensions for heatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    axis_vals, xlabel = _lead_time_axis(da.lead_time.values, time_unit)
    time_labels = [str(t)[:10] for t in pd.DatetimeIndex(da.time.values)]

    # Single-IC runs: a heatmap can't show anything meaningful across time,
    # and pcolormesh's auto-scaled ylim on a 1-element Y axis often clips
    # the single cell.  Fall back to a bar chart of metric vs lead time.
    if len(time_labels) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        values = np.asarray(da.values).reshape(-1)
        width = (
            float(np.median(np.diff(axis_vals))) * 0.7
            if len(axis_vals) > 1
            else 0.5
        )
        ax.bar(axis_vals, values, width=width, color="C0")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_name.upper())
        ax.set_title(
            f"{metric_name.upper()} — {variable}  (IC: {time_labels[0]})"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(12, max(4, len(time_labels) * 0.4)))
    im = ax.pcolormesh(
        axis_vals,
        range(len(time_labels)),
        da.values,
        shading="nearest",
        cmap="plasma",
    )
    ax.set_yticks(range(len(time_labels)))
    ax.set_yticklabels(time_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(f"{metric_name.upper()} — {variable}")
    fig.colorbar(im, ax=ax, label=metric_name.upper())
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction vs truth maps (visualization)
# ---------------------------------------------------------------------------


def _load_visualization_slice(
    pred_ds: xr.Dataset,
    verif_ds: xr.Dataset,
    variable: str,
    time: str | None,
    lead_time: str,
) -> tuple[np.ndarray, np.ndarray, "OrderedDict[str, np.ndarray]", str, str]:
    """Load a single 2D prediction and truth slice for visualization.

    Parameters
    ----------
    pred_ds : xr.Dataset
        Prediction store.
    verif_ds : xr.Dataset
        Verification store (keyed by valid time).
    variable : str
        Variable to visualize.
    time : str | None
        IC time string.  ``None`` uses the first available time.
    lead_time : str
        Lead time string (e.g. ``"5 days"``).

    Returns
    -------
    tuple
        ``(pred_2d, truth_2d, spatial_coords, time_label, lt_label)``

        ``spatial_coords`` is an :class:`OrderedDict` preserving the
        dataset's dimension order — e.g. ``{"lat": ..., "lon": ...}`` for
        regular lat/lon grids, or ``{"y": ..., "x": ...}`` for projection-
        native grids like HRRR.  :func:`plot_prediction_vs_truth`
        inspects the keys to pick the right cartopy ``transform``.

    Raises
    ------
    KeyError
        If the variable, time, or lead time is not found.
    """
    if variable not in pred_ds:
        raise KeyError(f"Variable '{variable}' not in prediction store.")

    da = pred_ds[variable]
    times = da.time.values
    ic_time = np.datetime64(time) if time else times[0]
    lt = np.timedelta64(pd.Timedelta(lead_time))

    sel = {"time": ic_time, "lead_time": lt}
    # Handle ensemble — take member 0 if present
    if "ensemble" in da.dims:
        sel["ensemble"] = da.ensemble.values[0]
    pred_2d = da.sel(**sel).values

    # Verification is indexed by valid time
    valid_time = ic_time + lt
    if variable not in verif_ds:
        raise KeyError(f"Variable '{variable}' not in verification store.")
    truth_2d = verif_ds[variable].sel(time=valid_time).values

    # Spatial coords in dataset-dim order (y/x for HRRR, lat/lon for global).
    spatial_dims = [
        d for d in da.dims if d not in {"time", "lead_time", "ensemble", "variable"}
    ]
    spatial_coords: OrderedDict[str, np.ndarray] = OrderedDict()
    for dim in spatial_dims:
        spatial_coords[dim] = da.coords[dim].values

    time_label = str(ic_time)[:10]
    lt_label = str(pd.Timedelta(lt))

    return pred_2d, truth_2d, spatial_coords, time_label, lt_label


_PROJECTIONS = {
    "robinson": "Robinson",
    "mollweide": "Mollweide",
    "platecarree": "PlateCarree",
    "orthographic": "Orthographic",
}


def _hrrr_lambert_projection() -> Any:
    """Return a cartopy CRS matching HRRR's Lambert-conformal projection.

    Parameters match :meth:`earth2studio.data.HRRR.grid` — single standard
    parallel at 38.5°N, central longitude 262.5°E, and a spherical globe
    of radius 6 371 229 m.  Prediction zarrs on the HRRR grid carry
    ``y`` / ``x`` coordinates in this projection's meter space, so using
    this CRS as both the map projection *and* the data transform
    renders them natively with no reprojection.
    """
    globe = ccrs.Globe(
        ellipse=None, semimajor_axis=6371229, semiminor_axis=6371229
    )
    return ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=globe,
    )


_CUSTOM_PROJECTIONS = {
    "hrrr_lambert": _hrrr_lambert_projection,
}


def _resolve_projection(name: str | None) -> Any:
    """Resolve a projection name to a cartopy CRS instance, or None.

    Parameters
    ----------
    name : str | None
        Projection name (e.g. ``"robinson"``, ``"hrrr_lambert"``).
        ``None`` means flat plot.

    Returns
    -------
    cartopy.crs.Projection | None
    """
    if name is None:
        return None
    if not _HAS_CARTOPY:
        logger.warning(
            f"Projection '{name}' requested but cartopy is not installed. "
            "Falling back to flat plot."
        )
        return None
    name_lower = name.lower()
    if name_lower in _CUSTOM_PROJECTIONS:
        return _CUSTOM_PROJECTIONS[name_lower]()
    cls_name = _PROJECTIONS.get(name_lower)
    if cls_name is None:
        logger.warning(
            f"Unknown projection '{name}'. "
            f"Available: {list(_PROJECTIONS.keys()) + list(_CUSTOM_PROJECTIONS.keys())}. "
            "Falling back to flat plot."
        )
        return None
    return getattr(ccrs, cls_name)()


def plot_prediction_vs_truth(
    pred_2d: np.ndarray,
    truth_2d: np.ndarray,
    spatial_coords: "OrderedDict[str, np.ndarray]",
    variable: str,
    time_label: str,
    lt_label: str,
    cmap: str = "turbo",
    projection: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    diff_abs: float | None = None,
    show_diff: bool = True,
) -> Figure:
    """Plot prediction, truth, and difference side-by-side.

    Parameters
    ----------
    pred_2d, truth_2d : np.ndarray
        2D field arrays shaped ``(H, W)`` with axes in the same order as
        ``spatial_coords``.
    spatial_coords : OrderedDict[str, np.ndarray]
        Dataset spatial coords in dim order.  The keys determine how the
        data is projected:

        * ``{"lat": ..., "lon": ...}`` — coordinates in degrees; plotted
          with ``transform=PlateCarree()``, global extent, coastlines.
        * ``{"y": ..., "x": ...}`` — coordinates are in the map
          projection's native meter space (e.g. HRRR Lambert
          conformal); plotted with ``transform=projection`` and an
          auto-extent that frames the data.
    variable : str
        Variable name (for titles).
    time_label, lt_label : str
        IC time and lead time labels.
    cmap : str
        Colormap for prediction/truth panels.
    projection : cartopy.crs.Projection | None
        Map projection.  ``None`` produces a flat plot without cartopy.
    vmin, vmax : float | None
        Prediction/truth color range.  When ``None`` (default) they are
        derived from ``nanmin`` / ``nanmax`` of the truth field.  Set
        explicitly to suppress the effect of fill-value outliers (e.g.
        MRMS ``refc``'s large-negative no-data pixels, which otherwise
        crush the usable color range).
    diff_abs : float | None
        Symmetric limit for the difference panel (``[-diff_abs,
        diff_abs]``).  When ``None``, derived from the difference's
        ``nanmin`` / ``nanmax``.  Ignored when *show_diff* is ``False``.
    show_diff : bool
        When ``False``, the third (difference) panel is omitted.  Useful
        when the prediction and truth use different fill-value
        conventions (e.g. MRMS ``refc``'s no-data pixels vs. the
        diffusion model's zero-fill) so the difference would be
        dominated by that offset rather than real skill signal.

    Returns
    -------
    Figure
    """
    if vmin is None:
        vmin = float(np.nanmin(truth_2d))
    if vmax is None:
        vmax = float(np.nanmax(truth_2d))
    diff: np.ndarray | None = None
    if show_diff:
        diff = pred_2d - truth_2d
        if diff_abs is None:
            diff_abs = float(max(abs(np.nanmin(diff)), abs(np.nanmax(diff))))

    use_cartopy = projection is not None and _HAS_CARTOPY
    subplot_kw = {"projection": projection} if use_cartopy else {}
    n_panels = 3 if show_diff else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 5), subplot_kw=subplot_kw
    )

    # Determine the coord arrays + data transform + extent strategy.
    # - lat/lon (degrees): PlateCarree transform, global extent.
    # - y/x (projection meters): same projection as the map, regional extent.
    # - anything else: fall back to imshow.
    dim_keys = tuple(spatial_coords.keys())
    coord_values = tuple(spatial_coords.values())
    is_latlon = set(dim_keys) >= {"lat", "lon"}
    is_yx = set(dim_keys) >= {"y", "x"}

    if use_cartopy and (is_latlon or is_yx) and len(coord_values) >= 2:
        if is_latlon:
            # coord_values preserves dataset dim order (usually lat, lon).
            lat_idx = dim_keys.index("lat")
            lon_idx = dim_keys.index("lon")
            row_coord = coord_values[lat_idx]
            col_coord = coord_values[lon_idx]
            transform = ccrs.PlateCarree()
            extent_mode: str | None = "global"
        else:
            y_idx = dim_keys.index("y")
            x_idx = dim_keys.index("x")
            row_coord = coord_values[y_idx]
            col_coord = coord_values[x_idx]
            transform = projection  # data already in map's CRS
            extent_mode = "data"
    else:
        row_coord = coord_values[0] if coord_values else np.array([])
        col_coord = coord_values[1] if len(coord_values) > 1 else np.array([])
        transform = None
        extent_mode = None

    panels: list[tuple[Any, np.ndarray, str, dict[str, Any]]] = [
        (axes[0], pred_2d, "Prediction", {"vmin": vmin, "vmax": vmax, "cmap": cmap}),
        (axes[1], truth_2d, "Truth", {"vmin": vmin, "vmax": vmax, "cmap": cmap}),
    ]
    if show_diff and diff is not None:
        panels.append(
            (
                axes[2],
                diff,
                "Difference",
                {"vmin": -diff_abs, "vmax": diff_abs, "cmap": "RdBu_r"},
            )
        )

    for ax, data, title, kwargs in panels:
        if use_cartopy and transform is not None and row_coord.size and col_coord.size:
            im = ax.pcolormesh(
                col_coord,
                row_coord,
                data,
                shading="auto",
                transform=transform,
                **kwargs,
            )
            ax.coastlines(linewidth=0.5)
            if extent_mode == "global":
                ax.set_global()
            elif extent_mode == "data":
                ax.set_extent(
                    [
                        float(col_coord.min()),
                        float(col_coord.max()),
                        float(row_coord.min()),
                        float(row_coord.max()),
                    ],
                    crs=projection,
                )
        elif row_coord.size and col_coord.size:
            im = ax.pcolormesh(col_coord, row_coord, data, shading="auto", **kwargs)
        else:
            im = ax.imshow(data, origin="lower", aspect="auto", **kwargs)
        fig.colorbar(im, ax=ax, shrink=0.8, extend="both")
        ax.set_title(title)

    fig.suptitle(
        f"{variable}  |  IC: {time_label}  |  Lead: {lt_label}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _color_range_for(
    report_cfg: DictConfig, variable: str
) -> dict[str, float]:
    """Pull ``{vmin, vmax, diff_abs}`` for *variable* from the report config.

    Reads ``report_cfg.color_ranges.<variable>`` — a mapping with any of
    the three keys.  Missing keys default to ``None`` (auto-scale from
    the data).  Missing variable block returns an empty dict.  Used to
    override nanmin/nanmax-derived scales when a variable has fill-value
    outliers (e.g. MRMS ``refc``'s large-negative no-data pixels).
    """
    cr = report_cfg.get("color_ranges")
    if cr is None or variable not in cr:
        return {}
    entry = cr[variable]
    out: dict[str, float] = {}
    for key in ("vmin", "vmax", "diff_abs"):
        if key in entry and entry[key] is not None:
            out[key] = float(entry[key])
    return out


# ---------------------------------------------------------------------------
# Spread-skill plots
# ---------------------------------------------------------------------------


def plot_spread_skill(
    ds: xr.Dataset,
    mse_metric: str,
    variance_metric: str,
    variables: list[str],
    variable_group_name: str | None = None,
    time_groups: dict[str, np.ndarray] | None = None,
    height_ratios: tuple[int, int] = (3, 1),
    time_unit: str = "days",
) -> Figure:
    """Plot a 2-panel spread-skill figure.

    Top panel: RMSE (ensemble mean) and Spread vs lead time.
    Bottom panel: Spread / Skill ratio with R=1 reference line and
    dynamic y-axis symmetric around 1.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    mse_metric, variance_metric : str
        Config keys for the MSE and variance metrics.
    variables : list[str]
        Variables to plot.
    variable_group_name : str | None
        Label for the variable group (used in title).
    time_groups : dict[str, np.ndarray] | None
        If provided, overlay per-group lines with distinct styles.
    height_ratios : tuple[int, int]
        Relative heights of the top and bottom panels.

    Returns
    -------
    Figure
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0},
        figsize=(10, 8),
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Collect all ratio values for dynamic y-limits.
    all_ratio_vals: list[float] = []

    # --- "all times" lines ---
    results = compute_spread_skill(ds, mse_metric, variance_metric, variables)

    xlabel = "Lead time"
    for i, var in enumerate(variables):
        if var not in results["rmse"]:
            continue
        color = colors[i % len(colors)]
        rmse_da = results["rmse"][var]
        spread_da = results["spread"][var]
        ratio_da = results["ratio"][var]

        if "lead_time" not in rmse_da.dims:
            continue
        axis_vals, xlabel = _lead_time_axis(rmse_da.lead_time.values, time_unit)

        ax_top.plot(
            axis_vals,
            rmse_da.values,
            color=color,
            ls="-",
            linewidth=1.5,
            label=f"RMSE {var}",
        )
        ax_top.plot(
            axis_vals,
            spread_da.values,
            color=color,
            ls="--",
            linewidth=1.5,
            label=f"Spread {var}",
        )
        ratio_vals = ratio_da.values
        ax_bot.plot(
            axis_vals,
            ratio_vals,
            color=color,
            linewidth=1.5,
            label=var,
        )
        all_ratio_vals.extend(ratio_vals[np.isfinite(ratio_vals)])

        # --- per-group overlays ---
        if time_groups:
            for gi, (gname, gtimes) in enumerate(time_groups.items()):
                gres = compute_spread_skill(
                    ds,
                    mse_metric,
                    variance_metric,
                    [var],
                    time_sel=gtimes,
                )
                if var not in gres["ratio"]:
                    continue
                gls = _GROUP_LINESTYLES[gi % len(_GROUP_LINESTYLES)]
                g_ratio = gres["ratio"][var].values
                ax_bot.plot(
                    axis_vals,
                    g_ratio,
                    color=color,
                    linestyle=gls,
                    alpha=0.6,
                    linewidth=1.0,
                    label=f"{var} ({gname})",
                )
                all_ratio_vals.extend(g_ratio[np.isfinite(g_ratio)])

    # --- formatting ---
    ax_bot.axhline(1.0, color="gray", ls=":", alpha=0.7, label="R = 1")

    # Dynamic y-limits symmetric around 1.0.
    if all_ratio_vals:
        max_dev = max(abs(v - 1.0) for v in all_ratio_vals)
        margin = max(max_dev * 1.15, 0.1)  # at least ±0.1
        ratio_ylim = (max(0, 1.0 - margin), min(3.0, 1.0 + margin))
    else:
        ratio_ylim = (0.0, 2.0)
    ax_bot.set_ylim(*ratio_ylim)

    title = "Spread-Skill"
    if variable_group_name:
        title += f" — {variable_group_name}"
    ax_top.set_title(title)
    ax_top.set_ylabel("RMSE / Spread")
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.3)
    # Hide x tick labels on top panel (shared axis, no gap).
    ax_top.tick_params(labelbottom=False)

    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Spread / Skill")
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3)

    return fig
