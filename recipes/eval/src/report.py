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

"""Report generation module: aggregate scores and produce markdown reports.

Reads a ``scores.zarr`` store produced by ``score.py``, computes standard
aggregations (time-mean RMSE, CRPS, etc.), generates
matplotlib figures, and writes a self-contained markdown report with
collapsible sections.

Designed for single-process use (no GPU required).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")  # non-interactive backend

try:
    import cartopy.crs as ccrs

    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default lead times for summary tables (as timedelta strings)
_DEFAULT_SUMMARY_LEAD_TIMES = [
    "1 days",
    "3 days",
    "5 days",
    "7 days",
    "10 days",
    "14 days",
]

# Metric names whose proper aggregation is sqrt(mean(x²)) not mean(x).
# This is the legacy path for any RMSE values stored directly.
_QUADRATIC_METRICS = frozenset({"rmse"})

# Metrics that store pre-sqrt values (MSE/variance).  Correct time
# aggregation is sqrt(mean(x)) — take the mean first, then sqrt.
_SQRT_AFTER_MEAN = frozenset({"mse", "ensemble_mean_mse"})

# Human-readable display names for metric keys.
_DISPLAY_NAMES: dict[str, str] = {
    "mse": "RMSE",
    "ensemble_mean_mse": "RMSE (ens. mean)",
    "ensemble_variance": "Ens. Variance",
    "rmse": "RMSE",
    "crps": "CRPS",
    "mae": "MAE",
}


def display_name(metric_name: str) -> str:
    """Return a human-readable label for *metric_name*."""
    return _DISPLAY_NAMES.get(metric_name, metric_name.upper())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SectionOutput:
    """Result from a section renderer."""

    markdown: str
    figures: dict[str, Figure] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scores(cfg: DictConfig) -> xr.Dataset:
    """Open the scores zarr store as an xarray Dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path`` and ``scoring.output.store_name``.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    FileNotFoundError
        If the score store does not exist.
    """
    store_name = cfg.scoring.output.get("store_name", "scores.zarr")
    store_path = os.path.join(cfg.output.path, store_name)
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Score store not found at '{store_path}'.\n"
            "Run scoring (score.py) before generating a report."
        )
    logger.info(f"Opened score store: {store_path}")
    return xr.open_zarr(store_path)


def parse_score_arrays(
    ds: xr.Dataset,
) -> dict[str, list[str]]:
    """Parse score array names into {metric: [variable, ...]} groups.

    Array names follow the ``{metric}__{variable}`` convention.  Arrays
    without ``__`` are treated as single-variable metrics (the metric name
    is the array name, variable is ``None``).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.

    Returns
    -------
    dict[str, list[str]]
        Mapping from metric name to list of variable names.
    """
    groups: dict[str, list[str]] = {}
    for name in ds.data_vars:
        if "__" in name:
            metric, variable = name.split("__", 1)
        else:
            metric = name
            variable = name
        groups.setdefault(metric, []).append(variable)
    return groups


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_over_time(
    da: xr.DataArray,
    metric_name: str,
) -> xr.DataArray:
    """Aggregate a score array over the time dimension.

    Three modes depending on what the metric stores:

    * **Quadratic** (``rmse``): stores RMSE per time → ``sqrt(mean(x²))``
      to recover global RMSE.
    * **Sqrt-after-mean** (``mse``, ``ensemble_mean_mse``): stores MSE
      per time → ``sqrt(mean(x))`` to produce RMSE.
    * **Default**: simple ``mean(x)`` (e.g. CRPS, MAE, variance).

    Parameters
    ----------
    da : xr.DataArray
        Score array with a ``time`` dimension.
    metric_name : str
        Metric name (used to determine aggregation method).

    Returns
    -------
    xr.DataArray
        Time-aggregated score (time dimension removed).
    """
    if metric_name in _SQRT_AFTER_MEAN:
        return np.sqrt(da.mean(dim="time"))
    if metric_name in _QUADRATIC_METRICS:
        return np.sqrt((da**2).mean(dim="time"))
    return da.mean(dim="time")


def aggregate_over_ensemble(
    da: xr.DataArray,
) -> xr.DataArray:
    """Mean over the ensemble dimension if present.

    Parameters
    ----------
    da : xr.DataArray
        Score array, optionally with an ``ensemble`` dimension.

    Returns
    -------
    xr.DataArray
        Ensemble-averaged score.
    """
    if "ensemble" in da.dims:
        return da.mean(dim="ensemble")
    return da


def resolve_time_groups(
    time_groups_cfg: Any,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Match IC times to named groups based on date ranges.

    Parameters
    ----------
    time_groups_cfg : dict
        Mapping of group names to lists of ``{start, end}`` date ranges.
    times : np.ndarray
        Array of datetime64 values from the score store.

    Returns
    -------
    dict[str, np.ndarray]
        Group name to matching time values.
    """
    groups: dict[str, np.ndarray] = {}
    for name, ranges in time_groups_cfg.items():
        mask = np.zeros(len(times), dtype=bool)
        for r in ranges:
            start = np.datetime64(r["start"])
            end = np.datetime64(r["end"])
            mask |= (times >= start) & (times <= end)
        matched = times[mask]
        if len(matched) > 0:
            groups[name] = matched
        else:
            logger.warning(f"Time group '{name}' matched no IC times — skipping.")
    return groups


def snapshot_at_lead_times(
    ds: xr.Dataset,
    metric_groups: dict[str, list[str]],
    lead_times: list[str],
) -> pd.DataFrame:
    """Extract scores at specific lead times into a summary DataFrame.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from :func:`parse_score_arrays`.
    lead_times : list[str]
        Lead time strings to snapshot (e.g. ``["1 days", "5 days"]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``metric, variable, lead_time, value``.
    """
    rows: list[dict[str, Any]] = []
    target_lts = [np.timedelta64(pd.Timedelta(lt)) for lt in lead_times]

    for metric_name, variables in metric_groups.items():
        for var in variables:
            array_name = f"{metric_name}__{var}"
            if array_name not in ds:
                continue
            da = aggregate_over_ensemble(ds[array_name])
            da = aggregate_over_time(da, metric_name)

            for lt, lt_str in zip(target_lts, lead_times):
                if "lead_time" in da.dims and lt in da.lead_time.values:
                    val = float(da.sel(lead_time=lt).values)
                    rows.append(
                        {
                            "metric": metric_name,
                            "variable": var,
                            "lead_time": lt_str,
                            "value": val,
                        }
                    )

    return pd.DataFrame(rows)


def build_summary_csv(
    ds: xr.Dataset,
    metric_groups: dict[str, list[str]],
    run_id: str,
) -> pd.DataFrame:
    """Build the full aggregation CSV with all lead times.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from :func:`parse_score_arrays`.
    run_id : str
        Run/model identifier for the ``model`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``model, metric, variable, lead_time, value``.
    """
    rows: list[dict[str, Any]] = []

    for metric_name, variables in metric_groups.items():
        for var in variables:
            array_name = f"{metric_name}__{var}"
            if array_name not in ds:
                continue
            da = aggregate_over_ensemble(ds[array_name])
            da = aggregate_over_time(da, metric_name)

            if "lead_time" not in da.dims:
                rows.append(
                    {
                        "model": run_id,
                        "metric": metric_name,
                        "variable": var,
                        "lead_time": "aggregate",
                        "value": float(da.values),
                    }
                )
                continue

            for lt in da.lead_time.values:
                val = float(da.sel(lead_time=lt).values)
                rows.append(
                    {
                        "model": run_id,
                        "metric": metric_name,
                        "variable": var,
                        "lead_time": str(pd.Timedelta(lt)),
                        "value": val,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _lead_time_to_days(lead_times: np.ndarray) -> np.ndarray:
    """Convert timedelta64 lead times to fractional days for plotting."""
    return lead_times / np.timedelta64(1, "D")


# Line styles cycled for time-group overlays so each group is distinguishable.
_GROUP_LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1))]


def plot_metric_vs_leadtime(
    ds: xr.Dataset,
    metric_name: str,
    variables: list[str],
    variable_group_name: str | None = None,
    time_groups: dict[str, np.ndarray] | None = None,
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

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for var in variables:
        array_name = f"{metric_name}__{var}"
        if array_name not in ds:
            continue
        da = aggregate_over_ensemble(ds[array_name])

        # "All times" line
        agg = aggregate_over_time(da, metric_name)
        if "lead_time" not in agg.dims:
            continue
        days = _lead_time_to_days(agg.lead_time.values)
        line = ax.plot(days, agg.values, label=var, linewidth=1.5)
        color = line[0].get_color()

        # Per-group lines with distinct styles
        if time_groups:
            for i, (group_name, group_times) in enumerate(time_groups.items()):
                da_sub = da.sel(time=group_times)
                agg_sub = aggregate_over_time(da_sub, metric_name)
                ls = _GROUP_LINESTYLES[i % len(_GROUP_LINESTYLES)]
                ax.plot(
                    days,
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
        title += f" \u2014 {variable_group_name}"
    ax.set_title(title)
    ax.set_xlabel("Lead time (days)")
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
) -> None:
    """Draw lead-time curves for one variable group on a single axes."""
    for var in variables:
        array_name = f"{metric_name}__{var}"
        if array_name not in ds:
            continue
        da = aggregate_over_ensemble(ds[array_name])

        agg = aggregate_over_time(da, metric_name)
        if "lead_time" not in agg.dims:
            continue
        days = _lead_time_to_days(agg.lead_time.values)
        line = ax.plot(days, agg.values, label=var, linewidth=1.5)
        color = line[0].get_color()

        if time_groups:
            for i, (group_name, group_times) in enumerate(time_groups.items()):
                da_sub = da.sel(time=group_times)
                agg_sub = aggregate_over_time(da_sub, metric_name)
                ls = _GROUP_LINESTYLES[i % len(_GROUP_LINESTYLES)]
                ax.plot(
                    days,
                    agg_sub.values,
                    color=color,
                    linestyle=ls,
                    alpha=0.6,
                    linewidth=1.0,
                    label=f"{var} ({group_name})",
                )


def plot_metric_grid(
    ds: xr.Dataset,
    metric_name: str,
    variable_groups: OrderedDict[str, list[str]],
    time_groups: dict[str, np.ndarray] | None = None,
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

    for idx, (group_name, group_vars) in enumerate(group_items):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _plot_leadtime_on_ax(ax, ds, metric_name, group_vars, time_groups)
        ax.set_title(group_name, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        # Only label x-axis on the bottom row.
        if row == nrows - 1:
            ax.set_xlabel("Lead time (days)")
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

    days = _lead_time_to_days(da.lead_time.values)
    time_labels = [str(t)[:10] for t in pd.DatetimeIndex(da.time.values)]

    fig, ax = plt.subplots(figsize=(12, max(4, len(time_labels) * 0.4)))
    im = ax.pcolormesh(
        days,
        range(len(time_labels)),
        da.values,
        shading="auto",
        cmap="plasma",
    )
    ax.set_yticks(range(len(time_labels)))
    ax.set_yticklabels(time_labels, fontsize=8)
    ax.set_xlabel("Lead time (days)")
    ax.set_title(f"{metric_name.upper()} \u2014 {variable}")
    fig.colorbar(im, ax=ax, label=metric_name.upper())
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _wrap_details(title: str, body: str, collapsed: bool = True) -> str:
    """Wrap markdown content in a <details> block if collapsed."""
    if not collapsed:
        return f"## {title}\n\n{body}\n"
    return f"<details>\n<summary><b>{title}</b></summary>\n\n" f"{body}\n\n</details>\n"


def render_summary_table(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render a summary table at key lead times.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    section_cfg : DictConfig
        Section-level config (``lead_times``, ``collapsed``).
    report_cfg : DictConfig
        Full report config.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` grouping.
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    SectionOutput
    """
    lead_times = list(section_cfg.get("lead_times", _DEFAULT_SUMMARY_LEAD_TIMES))
    collapsed = section_cfg.get("collapsed", False)
    summary_vars = list(section_cfg.get("variables", []))

    # If a variable subset is specified, filter metric_groups down.
    if summary_vars:
        metric_groups = {
            m: [v for v in vs if v in summary_vars] for m, vs in metric_groups.items()
        }
        metric_groups = {m: vs for m, vs in metric_groups.items() if vs}

    df = snapshot_at_lead_times(ds, metric_groups, lead_times)
    if df.empty:
        return SectionOutput(
            markdown=_wrap_details("Summary", "_No data available._", collapsed)
        )

    # Pivot: rows = (variable), columns = (metric, lead_time)
    pivot = df.pivot_table(
        index="variable",
        columns=["metric", "lead_time"],
        values="value",
    )

    # Build markdown table
    metrics_in_data = list(dict.fromkeys(df["metric"]))
    lts_in_data = [lt for lt in lead_times if lt in df["lead_time"].values]

    # Header row
    header_parts = ["Variable"] + [
        f"{metric.upper()} ({lt})" for metric in metrics_in_data for lt in lts_in_data
    ]
    header = "| " + " | ".join(header_parts) + " |"
    separator = "| " + " | ".join(["---"] * len(header_parts)) + " |"

    rows = [header, separator]
    for var in sorted(pivot.index):
        parts = [var]
        for metric in metrics_in_data:
            for lt in lts_in_data:
                try:
                    val = pivot.loc[var, (metric, lt)]
                    parts.append(f"{val:.4g}")
                except KeyError:
                    parts.append("—")
        rows.append("| " + " | ".join(parts) + " |")

    body = "\n".join(rows)
    md = _wrap_details("Summary", body, collapsed)
    return SectionOutput(markdown=md)


def render_lead_time_curves(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render metric-vs-lead-time plots for configured variable groups.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    section_cfg : DictConfig
        Section config (``metrics``, ``time_groups``, ``title_suffix``,
        ``collapsed``).
    report_cfg : DictConfig
        Full report config (reads ``variable_groups``, ``time_groups``).
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` grouping.
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    SectionOutput
    """
    collapsed = section_cfg.get("collapsed", True)
    use_time_groups = section_cfg.get("time_groups", False)
    title_suffix = section_cfg.get("title_suffix", "")
    section_metrics = list(section_cfg.get("metrics", list(metric_groups.keys())))

    # Resolve variable groups
    variable_groups = _resolve_variable_groups(report_cfg, metric_groups)

    # Resolve time groups
    time_groups = None
    if use_time_groups and "time_groups" in report_cfg:
        time_groups = resolve_time_groups(
            OmegaConf.to_container(report_cfg.time_groups, resolve=True),
            ds.time.values,
        )

    figures: dict[str, Figure] = {}
    md_parts: list[str] = []

    for metric_name in section_metrics:
        if metric_name not in metric_groups:
            continue

        # Build the subset of variable groups that have data for this
        # metric, preserving the configured order.
        available_groups: OrderedDict[str, list[str]] = OrderedDict()
        for group_name, group_vars in variable_groups.items():
            available = [v for v in group_vars if f"{metric_name}__{v}" in ds]
            if available:
                available_groups[group_name] = available

        if not available_groups:
            continue

        fig = plot_metric_grid(
            ds,
            metric_name,
            available_groups,
            time_groups=time_groups,
        )
        fig_name = f"{metric_name}_vs_leadtime"
        if title_suffix:
            fig_name += f"_{title_suffix.lower().replace(' ', '_')}"
        figures[fig_name] = fig
        md_parts.append(f"![{display_name(metric_name)}](figures/{fig_name}.png)\n")

    title = "Metric vs Lead Time"
    if title_suffix:
        title += f" \u2014 {title_suffix}"

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details(title, body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


def render_ic_heatmap(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render per-IC heatmaps for selected metrics and variables.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    section_cfg : DictConfig
        Section config (``metrics``, ``variables``, ``collapsed``).
    report_cfg : DictConfig
        Full report config.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` grouping.
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    SectionOutput
    """
    collapsed = section_cfg.get("collapsed", True)
    section_metrics = list(section_cfg.get("metrics", list(metric_groups.keys())))
    section_vars = list(
        section_cfg.get(
            "variables",
            [v for vs in metric_groups.values() for v in vs],
        )
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_vars: list[str] = []
    for v in section_vars:
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)

    figures: dict[str, Figure] = {}
    md_parts: list[str] = []

    for metric_name in section_metrics:
        for var in unique_vars:
            array_name = f"{metric_name}__{var}"
            if array_name not in ds:
                continue
            fig = plot_ic_heatmap(ds, metric_name, var)
            fig_name = f"ic_heatmap_{metric_name}__{var}"
            figures[fig_name] = fig
            md_parts.append(
                f"### {metric_name.upper()} \u2014 {var}\n\n"
                f"![{metric_name} {var}](figures/{fig_name}.png)\n"
            )

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details("Per-IC Heatmaps", body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


# ---------------------------------------------------------------------------
# Visualization — prediction vs truth maps
# ---------------------------------------------------------------------------


def _open_data_stores(
    cfg: DictConfig,
) -> tuple[xr.Dataset | None, xr.Dataset | None]:
    """Open prediction and verification zarr stores for visualization.

    Returns ``(None, None)`` if either store is missing — visualization
    sections degrade gracefully.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path``.

    Returns
    -------
    tuple[xr.Dataset | None, xr.Dataset | None]
        ``(prediction_ds, verification_ds)``
    """
    pred_path = os.path.join(cfg.output.path, "forecast.zarr")
    pred_ds = None
    if os.path.exists(pred_path):
        pred_ds = xr.open_zarr(pred_path)
    else:
        logger.info(
            f"Prediction store not found at '{pred_path}' — "
            "visualization sections will be skipped."
        )

    verif_ds = None
    for name in ("verification.zarr", "data.zarr"):
        vpath = os.path.join(cfg.output.path, name)
        if os.path.exists(vpath):
            verif_ds = xr.open_zarr(vpath)
            break

    if verif_ds is None:
        logger.info(
            "Verification store not found — " "visualization sections will be skipped."
        )

    return pred_ds, verif_ds


def _load_visualization_slice(
    pred_ds: xr.Dataset,
    verif_ds: xr.Dataset,
    variable: str,
    time: str | None,
    lead_time: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
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
        ``(pred_2d, truth_2d, lat, lon, time_label, lt_label)``

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

    # Spatial coords
    spatial_dims = [
        d for d in da.dims if d not in {"time", "lead_time", "ensemble", "variable"}
    ]
    lat = da.coords[spatial_dims[0]].values if spatial_dims else np.array([])
    lon = da.coords[spatial_dims[1]].values if len(spatial_dims) > 1 else np.array([])

    time_label = str(ic_time)[:10]
    lt_label = str(pd.Timedelta(lt))

    return pred_2d, truth_2d, lat, lon, time_label, lt_label


_PROJECTIONS = {
    "robinson": "Robinson",
    "mollweide": "Mollweide",
    "platecarree": "PlateCarree",
    "orthographic": "Orthographic",
}


def _resolve_projection(name: str | None) -> Any:
    """Resolve a projection name to a cartopy CRS instance, or None.

    Parameters
    ----------
    name : str | None
        Projection name (e.g. ``"robinson"``).  ``None`` means flat plot.

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
    cls_name = _PROJECTIONS.get(name.lower())
    if cls_name is None:
        logger.warning(
            f"Unknown projection '{name}'. "
            f"Available: {list(_PROJECTIONS.keys())}. "
            "Falling back to flat plot."
        )
        return None
    return getattr(ccrs, cls_name)()


def plot_prediction_vs_truth(
    pred_2d: np.ndarray,
    truth_2d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    variable: str,
    time_label: str,
    lt_label: str,
    cmap: str = "turbo",
    projection: Any = None,
) -> Figure:
    """Plot prediction, truth, and difference side-by-side.

    Parameters
    ----------
    pred_2d, truth_2d : np.ndarray
        2D field arrays (lat x lon).
    lat, lon : np.ndarray
        Coordinate arrays.
    variable : str
        Variable name (for titles).
    time_label : str
        IC time label.
    lt_label : str
        Lead time label.
    cmap : str
        Colormap for prediction/truth panels.
    projection : cartopy.crs.Projection | None
        Map projection.  ``None`` produces a flat (PlateCarree) plot
        without cartopy.

    Returns
    -------
    Figure
    """
    diff = pred_2d - truth_2d
    vmin = np.nanmin(truth_2d)
    vmax = np.nanmax(truth_2d)
    diff_abs = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))

    use_cartopy = projection is not None and _HAS_CARTOPY
    subplot_kw = {"projection": projection} if use_cartopy else {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw=subplot_kw)

    transform = ccrs.PlateCarree() if use_cartopy else None

    for ax, data, title, kwargs in [
        (axes[0], pred_2d, "Prediction", {"vmin": vmin, "vmax": vmax, "cmap": cmap}),
        (axes[1], truth_2d, "Truth", {"vmin": vmin, "vmax": vmax, "cmap": cmap}),
        (
            axes[2],
            diff,
            "Difference",
            {"vmin": -diff_abs, "vmax": diff_abs, "cmap": "RdBu_r"},
        ),
    ]:
        if use_cartopy and lat.size and lon.size:
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                shading="auto",
                transform=transform,
                **kwargs,
            )
            ax.coastlines(linewidth=0.5)
            ax.set_global()
        elif lat.size and lon.size:
            im = ax.pcolormesh(lon, lat, data, shading="auto", **kwargs)
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


def render_header_visualization(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render a single hero visualization (prediction vs truth).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset (unused, but matches renderer signature).
    section_cfg : DictConfig
        Section config: ``variable``, ``lead_time``, optional ``time``.
    report_cfg : DictConfig
        Full report config.
    metric_groups : dict[str, list[str]]
        Metric groups (unused).
    cfg : DictConfig
        Full Hydra config (used to locate data stores).

    Returns
    -------
    SectionOutput
    """
    pred_ds, verif_ds = _open_data_stores(cfg)
    if pred_ds is None or verif_ds is None:
        return SectionOutput(markdown="")

    variable = section_cfg.get("variable", "z500")
    lead_time = section_cfg.get("lead_time", "5 days")
    time = section_cfg.get("time", None)
    projection = _resolve_projection(report_cfg.get("projection", None))

    try:
        pred_2d, truth_2d, lat, lon, time_label, lt_label = _load_visualization_slice(
            pred_ds,
            verif_ds,
            variable,
            time,
            lead_time,
        )
    except (KeyError, IndexError) as e:
        logger.warning(f"Sample visualization failed: {e}")
        pred_ds.close()
        verif_ds.close()
        return SectionOutput(markdown="")

    fig = plot_prediction_vs_truth(
        pred_2d,
        truth_2d,
        lat,
        lon,
        variable,
        time_label,
        lt_label,
        projection=projection,
    )
    fig_name = f"header_{variable}_{lead_time.replace(' ', '')}"

    pred_ds.close()
    verif_ds.close()

    body = f"![{variable} prediction vs truth](figures/{fig_name}.png)\n"
    md = f"## Sample Visualization\n\n{body}\n"
    return SectionOutput(markdown=md, figures={fig_name: fig})


def render_visualization(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render prediction vs truth maps for multiple variables/lead times.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset (unused).
    section_cfg : DictConfig
        Section config: ``variables``, ``lead_times``, optional ``time``,
        ``collapsed``.
    report_cfg : DictConfig
        Full report config.
    metric_groups : dict[str, list[str]]
        Metric groups (unused).
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    SectionOutput
    """
    collapsed = section_cfg.get("collapsed", True)
    pred_ds, verif_ds = _open_data_stores(cfg)

    if pred_ds is None or verif_ds is None:
        md = _wrap_details(
            "Visualization",
            "_Prediction or verification data not available._",
            collapsed,
        )
        return SectionOutput(markdown=md)

    variables = list(section_cfg.get("variables", ["z500"]))
    lead_times = list(section_cfg.get("lead_times", ["5 days"]))
    time = section_cfg.get("time", None)
    projection = _resolve_projection(report_cfg.get("projection", None))

    figures: dict[str, Figure] = {}
    md_parts: list[str] = []

    for variable in variables:
        for lead_time in lead_times:
            try:
                pred_2d, truth_2d, lat, lon, time_label, lt_label = (
                    _load_visualization_slice(
                        pred_ds,
                        verif_ds,
                        variable,
                        time,
                        lead_time,
                    )
                )
            except (KeyError, IndexError) as e:
                logger.warning(
                    f"Visualization skipped for {variable} at {lead_time}: {e}"
                )
                continue

            fig = plot_prediction_vs_truth(
                pred_2d,
                truth_2d,
                lat,
                lon,
                variable,
                time_label,
                lt_label,
                projection=projection,
            )
            fig_name = f"vis_{variable}_{lead_time.replace(' ', '')}"
            figures[fig_name] = fig
            md_parts.append(
                f"### {variable} — {lt_label}\n\n"
                f"![{variable} {lt_label}](figures/{fig_name}.png)\n"
            )

    pred_ds.close()
    verif_ds.close()

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details("Visualization", body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


# ---------------------------------------------------------------------------
# Spread-skill analysis
# ---------------------------------------------------------------------------


def compute_spread_skill(
    ds: xr.Dataset,
    mse_metric: str,
    variance_metric: str,
    variables: list[str],
    time_sel: np.ndarray | None = None,
) -> dict[str, xr.Dataset]:
    """Compute WB2-correct RMSE, Spread, and Spread-Skill Ratio.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with MSE and variance arrays.
    mse_metric : str
        Config key used for the ensemble-mean MSE metric
        (e.g. ``"ensemble_mean_mse"``).
    variance_metric : str
        Config key used for the ensemble variance metric
        (e.g. ``"ensemble_variance"``).
    variables : list[str]
        Variable names to compute over.
    time_sel : np.ndarray | None
        Optional time subset for seasonal / grouped analysis.

    Returns
    -------
    dict[str, xr.Dataset]
        Keys: ``"rmse"``, ``"spread"``, ``"ratio"`` — each an
        ``xr.Dataset`` with one ``DataArray`` per variable.
    """
    rmse_arrays: dict[str, xr.DataArray] = {}
    spread_arrays: dict[str, xr.DataArray] = {}
    ratio_arrays: dict[str, xr.DataArray] = {}

    for var in variables:
        mse_name = f"{mse_metric}__{var}"
        var_name = f"{variance_metric}__{var}"
        if mse_name not in ds or var_name not in ds:
            continue

        mse_da = ds[mse_name]
        var_da = ds[var_name]
        if time_sel is not None:
            mse_da = mse_da.sel(time=time_sel)
            var_da = var_da.sel(time=time_sel)

        rmse_vals = np.sqrt(mse_da.mean(dim="time"))
        spread_vals = np.sqrt(var_da.mean(dim="time"))

        rmse_arrays[var] = rmse_vals
        spread_arrays[var] = spread_vals
        ratio_arrays[var] = spread_vals / rmse_vals

    return {
        "rmse": xr.Dataset(rmse_arrays),
        "spread": xr.Dataset(spread_arrays),
        "ratio": xr.Dataset(ratio_arrays),
    }


def plot_spread_skill(
    ds: xr.Dataset,
    mse_metric: str,
    variance_metric: str,
    variables: list[str],
    variable_group_name: str | None = None,
    time_groups: dict[str, np.ndarray] | None = None,
    height_ratios: tuple[int, int] = (3, 1),
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

    for i, var in enumerate(variables):
        if var not in results["rmse"]:
            continue
        color = colors[i % len(colors)]
        rmse_da = results["rmse"][var]
        spread_da = results["spread"][var]
        ratio_da = results["ratio"][var]

        if "lead_time" not in rmse_da.dims:
            continue
        days = _lead_time_to_days(rmse_da.lead_time.values)

        ax_top.plot(
            days,
            rmse_da.values,
            color=color,
            ls="-",
            linewidth=1.5,
            label=f"RMSE {var}",
        )
        ax_top.plot(
            days,
            spread_da.values,
            color=color,
            ls="--",
            linewidth=1.5,
            label=f"Spread {var}",
        )
        ratio_vals = ratio_da.values
        ax_bot.plot(
            days,
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
                    days,
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
        title += f" \u2014 {variable_group_name}"
    ax_top.set_title(title)
    ax_top.set_ylabel("RMSE / Spread")
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.3)
    # Hide x tick labels on top panel (shared axis, no gap).
    ax_top.tick_params(labelbottom=False)

    ax_bot.set_xlabel("Lead time (days)")
    ax_bot.set_ylabel("Spread / Skill")
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3)

    return fig


def render_spread_skill(
    ds: xr.Dataset,
    section_cfg: DictConfig,
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
    cfg: DictConfig,
) -> SectionOutput:
    """Render a spread-skill section with 2-panel plots.

    When ``variables`` is specified, any variable group containing at
    least one requested variable is included — and *all* members of
    that group that have available data are plotted, not just the
    originally-requested subset.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    section_cfg : DictConfig
        Section config: ``mse_metric``, ``variance_metric``,
        ``variables``, ``height_ratios``, ``collapsed``,
        ``time_groups``.
    report_cfg : DictConfig
        Full report config.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` grouping.
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    SectionOutput
    """
    collapsed = section_cfg.get("collapsed", True)
    mse_metric = section_cfg.get("mse_metric", "ensemble_mean_mse")
    variance_metric = section_cfg.get("variance_metric", "ensemble_variance")
    height_ratios = tuple(section_cfg.get("height_ratios", [3, 1]))
    use_time_groups = section_cfg.get("time_groups", False)

    # Determine which variables are available for both metrics.
    mse_vars = set(metric_groups.get(mse_metric, []))
    var_vars = set(metric_groups.get(variance_metric, []))
    available = sorted(mse_vars & var_vars)

    if not available:
        md = _wrap_details(
            "Spread-Skill",
            "_No matching MSE/variance arrays found._",
            collapsed,
        )
        return SectionOutput(markdown=md)

    # Requested variables are used to *select groups*, not to filter
    # members within a group.  If z500 is requested and belongs to
    # geopotential: [z500, z850], the whole group is included.
    requested = set(section_cfg.get("variables", available))

    # Resolve time groups.
    time_groups = None
    if use_time_groups and "time_groups" in report_cfg:
        time_groups = resolve_time_groups(
            OmegaConf.to_container(report_cfg.time_groups, resolve=True),
            ds.time.values,
        )

    # Resolve variable groups from ALL available variables so group
    # membership is complete.
    variable_groups = _resolve_variable_groups(
        report_cfg,
        {
            mse_metric: available,
        },
    )

    figures: dict[str, Figure] = {}
    md_parts: list[str] = []

    for group_name, group_vars in variable_groups.items():
        # Keep only group members that have data.
        plot_vars = [v for v in group_vars if v in available]
        if not plot_vars:
            continue
        # Include group only if at least one member was requested.
        if not any(v in requested for v in plot_vars):
            continue

        fig = plot_spread_skill(
            ds,
            mse_metric,
            variance_metric,
            plot_vars,
            variable_group_name=group_name,
            time_groups=time_groups,
            height_ratios=height_ratios,
        )
        fig_name = f"spread_skill_{group_name}"
        figures[fig_name] = fig
        md_parts.append(f"![Spread-Skill {group_name}](figures/{fig_name}.png)\n")

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details("Spread-Skill", body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


# ---------------------------------------------------------------------------
# Section dispatch
# ---------------------------------------------------------------------------

_SECTION_RENDERERS = {
    "summary_table": render_summary_table,
    "lead_time_curves": render_lead_time_curves,
    "ic_heatmap": render_ic_heatmap,
    "header_visualization": render_header_visualization,
    "visualization": render_visualization,
    "spread_skill": render_spread_skill,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_variable_groups(
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
) -> OrderedDict[str, list[str]]:
    """Resolve variable groups from config, or fall back to one plot per variable.

    When ``variable_groups`` is configured, variables that appear in a group
    share a plot.  Variables present in the score store but not listed in any
    group each get their own single-variable plot.

    When ``variable_groups`` is not configured, every variable gets its own
    plot (avoids mixing incompatible physical scales).

    Parameters
    ----------
    report_cfg : DictConfig
        Report config (may contain ``variable_groups``).
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from score arrays.

    Returns
    -------
    OrderedDict[str, list[str]]
        Named variable groups.
    """
    all_vars = list(dict.fromkeys(v for vs in metric_groups.values() for v in vs))

    if "variable_groups" in report_cfg:
        groups: OrderedDict[str, list[str]] = OrderedDict(
            OmegaConf.to_container(report_cfg.variable_groups, resolve=True)
        )
        # Variables not in any explicit group get their own plot.
        grouped = {v for vs in groups.values() for v in vs}
        for v in all_vars:
            if v not in grouped:
                groups[v] = [v]
        return groups

    # Fallback: one plot per variable
    return OrderedDict((v, [v]) for v in all_vars)


# ---------------------------------------------------------------------------
# Report orchestration
# ---------------------------------------------------------------------------


def generate_report(cfg: DictConfig) -> Path:
    """Generate a complete evaluation report.

    Reads scores, runs configured sections, saves figures and markdown.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config with ``report`` section.

    Returns
    -------
    Path
        Path to the generated ``report.md``.
    """
    report_cfg = cfg.report
    ds = load_scores(cfg)
    metric_groups = parse_score_arrays(ds)
    run_id = cfg.get("run_id", "unknown")

    sections = list(report_cfg.get("sections", []))
    fig_format = report_cfg.get("figure_format", "png")

    # Output directory
    report_dir = Path(cfg.output.path) / "report"
    fig_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Build the report
    title = report_cfg.get("title", f"{run_id} \u2014 Evaluation Report")
    md_parts: list[str] = [f"# {title}\n"]
    all_figures: dict[str, Figure] = {}

    for section_cfg in sections:
        section_type = section_cfg.get("type", "")
        renderer = _SECTION_RENDERERS.get(section_type)
        if renderer is None:
            logger.warning(f"Unknown section type '{section_type}' — skipping.")
            continue

        logger.info(f"Rendering section: {section_type}")
        result = renderer(ds, section_cfg, report_cfg, metric_groups, cfg)
        md_parts.append(result.markdown)
        all_figures.update(result.figures)

    # Save figures
    for name, fig in all_figures.items():
        fig_path = fig_dir / f"{name}.{fig_format}"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved figure: {fig_path}")
    logger.info(f"Saved {len(all_figures)} figures.")

    # Save summary CSV
    csv_df = build_summary_csv(ds, metric_groups, run_id)
    csv_path = table_dir / "scores_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary CSV: {csv_path}")

    # Save per-group CSVs if time groups are configured
    if "time_groups" in report_cfg:
        time_groups = resolve_time_groups(
            OmegaConf.to_container(report_cfg.time_groups, resolve=True),
            ds.time.values,
        )
        for group_name, group_times in time_groups.items():
            ds_sub = ds.sel(time=group_times)
            group_df = build_summary_csv(ds_sub, metric_groups, run_id)
            group_path = table_dir / f"scores_summary_{group_name}.csv"
            group_df.to_csv(group_path, index=False)
            logger.debug(f"Saved group CSV: {group_path}")

    # Write markdown
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(md_parts))
    logger.success(f"Report written to: {report_path}")

    ds.close()
    return report_path
