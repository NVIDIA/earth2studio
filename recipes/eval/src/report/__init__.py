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

"""Report generation package.

Reads a ``scores.zarr`` store produced by :mod:`src.scoring`, computes
standard aggregations (time-mean RMSE, CRPS, etc.), generates matplotlib
figures, and writes a self-contained markdown report with collapsible
sections.  Designed for single-process use (no GPU required).

Submodules:

* :mod:`.aggregation` — score loading, metric/time aggregation, spread-skill
  computation, and dataset-opening helpers.
* :mod:`.plotting` — matplotlib/cartopy plotting primitives.
* :mod:`.sections` — section renderers that produce markdown + figures.
* :mod:`.main` — :func:`generate_report` orchestrator.
"""

from __future__ import annotations

# Select the non-interactive backend before any submodule imports pyplot.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from .aggregation import (  # noqa: E402
    SectionOutput,
    _open_data_stores,
    _resolve_variable_groups,
    aggregate_over_ensemble,
    aggregate_over_time,
    build_summary_csv,
    compute_spread_skill,
    display_name,
    load_scores,
    parse_score_arrays,
    resolve_time_groups,
    snapshot_at_lead_times,
)
from .main import generate_report  # noqa: E402
from .plotting import (  # noqa: E402
    _color_range_for,
    _grid_shape,
    _lead_time_axis,
    _load_visualization_slice,
    _resolve_projection,
    plot_ic_heatmap,
    plot_metric_grid,
    plot_metric_vs_leadtime,
    plot_prediction_vs_truth,
    plot_spread_skill,
)
from .sections import (  # noqa: E402
    _SECTION_RENDERERS,
    _wrap_details,
    render_header_visualization,
    render_ic_heatmap,
    render_lead_time_curves,
    render_spread_skill,
    render_summary_table,
    render_visualization,
)

__all__ = [
    "SectionOutput",
    "aggregate_over_ensemble",
    "aggregate_over_time",
    "build_summary_csv",
    "compute_spread_skill",
    "display_name",
    "generate_report",
    "load_scores",
    "parse_score_arrays",
    "plot_ic_heatmap",
    "plot_metric_grid",
    "plot_metric_vs_leadtime",
    "plot_prediction_vs_truth",
    "plot_spread_skill",
    "render_header_visualization",
    "render_ic_heatmap",
    "render_lead_time_curves",
    "render_spread_skill",
    "render_summary_table",
    "render_visualization",
    "resolve_time_groups",
    "snapshot_at_lead_times",
]
