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

"""Backward-compatible shim — delegates to :class:`ForecastPipeline`.

The canonical implementation now lives in :mod:`src.pipeline`.  This module
preserves the ``run_inference`` / ``build_output_coords`` function signatures
so that existing call sites (including tests) continue to work unchanged.
"""

from __future__ import annotations

import torch

from earth2studio.data import DataSource
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation

from .output import OutputManager
from .pipeline import ForecastPipeline, build_output_coords
from .work import WorkItem

# Re-export so existing ``from src.inference import build_output_coords`` works.
__all__ = ["build_output_coords", "run_inference"]


def run_inference(
    work_items: list[WorkItem],
    prognostic: PrognosticModel,
    data_source: DataSource,
    output_mgr: OutputManager,
    output_variables: list[str],
    nsteps: int,
    perturbation: Perturbation | None = None,
    diagnostics: list[DiagnosticModel] | None = None,
    device: torch.device | None = None,
) -> None:
    """Run distributed inference over a list of work items.

    This is a compatibility wrapper around :class:`ForecastPipeline`.
    See :mod:`src.pipeline` for the canonical implementation.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = ForecastPipeline()
    pipeline.prognostic = prognostic.to(device)
    pipeline.diagnostics = [dx.to(device) for dx in (diagnostics or [])]
    pipeline.perturbation = perturbation
    pipeline.nsteps = nsteps

    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {
        id(dx): dx.input_coords() for dx in pipeline.diagnostics
    }

    pipeline.run(work_items, data_source, output_mgr, output_variables, device)
