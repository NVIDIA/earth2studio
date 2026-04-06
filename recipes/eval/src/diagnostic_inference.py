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

"""Backward-compatible shim — delegates to :class:`DiagnosticPipeline`.

The canonical implementation now lives in :mod:`src.pipeline`.  This module
preserves the ``run_diagnostic_inference`` function signature so that existing
call sites (including tests) continue to work unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from earth2studio.data import DataSource
from earth2studio.models.dx import DiagnosticModel

from .output import OutputManager
from .pipeline import DiagnosticPipeline
from .work import WorkItem

__all__ = ["run_diagnostic_inference"]


def run_diagnostic_inference(
    work_items: list[WorkItem],
    diagnostics: list[DiagnosticModel],
    data_source: DataSource,
    output_mgr: OutputManager,
    output_variables: list[str],
    device: torch.device | None = None,
) -> None:
    """Run diagnostic-only inference over a list of work items.

    This is a compatibility wrapper around :class:`DiagnosticPipeline`.
    See :mod:`src.pipeline` for the canonical implementation.
    """
    if not diagnostics:
        raise ValueError("At least one diagnostic model is required.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DiagnosticPipeline()
    pipeline.diagnostics = [dx.to(device) for dx in diagnostics]
    pipeline._dx_input_coords = {
        id(dx): dx.input_coords() for dx in pipeline.diagnostics
    }

    # Build union of input variables.
    all_input_vars: list[str] = []
    seen: set[str] = set()
    for dx in pipeline.diagnostics:
        for v in pipeline._dx_input_coords[id(dx)]["variable"]:
            if v not in seen:
                all_input_vars.append(str(v))
                seen.add(str(v))
    pipeline._all_input_vars = all_input_vars

    dx0 = pipeline.diagnostics[0]
    pipeline._spatial_ref = dx0.output_coords(pipeline._dx_input_coords[id(dx0)])
    pipeline._zero_lead = np.array([np.timedelta64(0, "ns")])

    pipeline.run(work_items, data_source, output_mgr, output_variables, device)
