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

"""End-to-end report orchestration: run configured sections, save outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from .aggregation import (
    build_summary_csv,
    load_scores,
    parse_score_arrays,
    resolve_time_groups,
)
from .sections import _SECTION_RENDERERS


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
    title = report_cfg.get("title", f"{run_id} — Evaluation Report")
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
