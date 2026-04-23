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

"""Section renderers that assemble markdown + figures for the report."""

from __future__ import annotations

from collections import OrderedDict

import xarray as xr
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from .aggregation import (
    SectionOutput,
    _DEFAULT_SUMMARY_LEAD_TIMES,
    _open_data_stores,
    _resolve_variable_groups,
    display_name,
    resolve_time_groups,
    snapshot_at_lead_times,
)
from .plotting import (
    _color_range_for,
    _load_visualization_slice,
    _resolve_projection,
    plot_ic_heatmap,
    plot_metric_grid,
    plot_prediction_vs_truth,
    plot_spread_skill,
)


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
            time_unit=str(report_cfg.get("lead_time_axis_unit", "days")),
        )
        fig_name = f"{metric_name}_vs_leadtime"
        if title_suffix:
            fig_name += f"_{title_suffix.lower().replace(' ', '_')}"
        figures[fig_name] = fig
        md_parts.append(f"![{display_name(metric_name)}](figures/{fig_name}.png)\n")

    title = "Metric vs Lead Time"
    if title_suffix:
        title += f" — {title_suffix}"

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
            fig = plot_ic_heatmap(
                ds,
                metric_name,
                var,
                time_unit=str(report_cfg.get("lead_time_axis_unit", "days")),
            )
            fig_name = f"ic_heatmap_{metric_name}__{var}"
            figures[fig_name] = fig
            md_parts.append(
                f"### {metric_name.upper()} — {var}\n\n"
                f"![{metric_name} {var}](figures/{fig_name}.png)\n"
            )

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details("Per-IC Heatmaps", body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


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
        pred_2d, truth_2d, spatial_coords, time_label, lt_label = (
            _load_visualization_slice(
                pred_ds,
                verif_ds,
                variable,
                time,
                lead_time,
            )
        )
    except (KeyError, IndexError) as e:
        logger.warning(f"Sample visualization failed: {e}")
        pred_ds.close()
        verif_ds.close()
        return SectionOutput(markdown="")

    fig = plot_prediction_vs_truth(
        pred_2d,
        truth_2d,
        spatial_coords,
        variable,
        time_label,
        lt_label,
        projection=projection,
        show_diff=bool(report_cfg.get("show_diff", True)),
        **_color_range_for(report_cfg, variable),
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
                pred_2d, truth_2d, spatial_coords, time_label, lt_label = (
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
                spatial_coords,
                variable,
                time_label,
                lt_label,
                projection=projection,
                show_diff=bool(report_cfg.get("show_diff", True)),
                **_color_range_for(report_cfg, variable),
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
            time_unit=str(report_cfg.get("lead_time_axis_unit", "days")),
        )
        fig_name = f"spread_skill_{group_name}"
        figures[fig_name] = fig
        md_parts.append(f"![Spread-Skill {group_name}](figures/{fig_name}.png)\n")

    body = "\n".join(md_parts) if md_parts else "_No data available._"
    md = _wrap_details("Spread-Skill", body, collapsed)
    return SectionOutput(markdown=md, figures=figures)


_SECTION_RENDERERS = {
    "summary_table": render_summary_table,
    "lead_time_curves": render_lead_time_curves,
    "ic_heatmap": render_ic_heatmap,
    "header_visualization": render_header_visualization,
    "visualization": render_visualization,
    "spread_skill": render_spread_skill,
}
