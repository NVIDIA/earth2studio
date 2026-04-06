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

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.dx import DiagnosticModel
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from .inference import build_output_coords
from .output import OutputManager
from .work import WorkItem


@torch.inference_mode()
def run_diagnostic_inference(
    work_items: list[WorkItem],
    diagnostics: list[DiagnosticModel],
    data_source: DataSource,
    output_mgr: OutputManager,
    output_variables: list[str],
    device: torch.device | None = None,
) -> None:
    """Run diagnostic-only inference over a list of work items.

    Each work item represents a single input time (and optional ensemble
    member).  For each, the function fetches input data, runs all
    diagnostic models, and writes the combined output through the
    OutputManager.  There is no prognostic rollout — each time is
    processed independently.

    Parameters
    ----------
    work_items : list[WorkItem]
        Work items assigned to this rank (already distributed).
    diagnostics : list[DiagnosticModel]
        Diagnostic models to run.  The union of their input variables
        determines what is fetched from the data source.
    data_source : DataSource
        Source for input data.
    output_mgr : OutputManager
        Context-managed output handler (store must already be validated).
    output_variables : list[str]
        Variable names to sub-select from diagnostic output before writing.
    device : torch.device, optional
        Device for inference.  Defaults to CUDA if available.
    """
    if not work_items:
        logger.warning("No work items for this rank — skipping inference.")
        return

    if not diagnostics:
        raise ValueError("At least one diagnostic model is required.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostics = [dx.to(device) for dx in diagnostics]

    # Pre-compute input/output coord metadata for each diagnostic.
    dx_input_coords: dict[int, CoordSystem] = {
        id(dx): dx.input_coords() for dx in diagnostics
    }

    # Build the union of all input variables needed from the data source.
    all_input_vars: list[str] = []
    seen: set[str] = set()
    for dx in diagnostics:
        for v in dx_input_coords[id(dx)]["variable"]:
            if v not in seen:
                all_input_vars.append(str(v))
                seen.add(str(v))

    # Build the output filter from the first diagnostic's spatial grid.
    spatial_ref = diagnostics[0].output_coords(dx_input_coords[id(diagnostics[0])])
    output_coords = build_output_coords(spatial_ref, output_variables)
    has_ensemble = "ensemble" in output_mgr.io.coords

    zero_lead = np.array([np.timedelta64(0, "ns")])

    for item in tqdm(work_items, desc="Work items", position=0):
        _run_single_diagnostic(
            item=item,
            diagnostics=diagnostics,
            dx_input_coords=dx_input_coords,
            all_input_vars=all_input_vars,
            data_source=data_source,
            output_mgr=output_mgr,
            output_coords=output_coords,
            has_ensemble=has_ensemble,
            zero_lead=zero_lead,
            device=device,
        )

    logger.success("Diagnostic inference complete")


def _run_single_diagnostic(
    item: WorkItem,
    diagnostics: list[DiagnosticModel],
    dx_input_coords: dict[int, CoordSystem],
    all_input_vars: list[str],
    data_source: DataSource,
    output_mgr: OutputManager,
    output_coords: CoordSystem,
    has_ensemble: bool,
    zero_lead: np.ndarray,
    device: torch.device,
) -> None:
    """Run all diagnostic models for a single WorkItem and write output."""
    time = [item.time]

    # Fetch input data at the analysis time (lead_time=0).
    x, coords = fetch_data(
        source=data_source,
        time=time,
        variable=all_input_vars,
        lead_time=zero_lead,
        device=device,
    )

    # Run each diagnostic, accumulating outputs.
    x_combined, coords_combined = x, coords
    for dx in diagnostics:
        dx_ic = dx_input_coords[id(dx)]
        x_in, coords_in = map_coords(x, coords, dx_ic)
        y, y_coords = dx(x_in, coords_in)
        x_combined, coords_combined = cat_coords(
            (x_combined, y), (coords_combined, y_coords), "variable"
        )

    # Sub-select output variables and spatial dims.
    x_out, coords_out = map_coords(x_combined, coords_combined, output_coords)

    if has_ensemble:
        x_out = x_out.unsqueeze(0)
        coords_out = CoordSystem(
            {"ensemble": np.array([item.ensemble_id])} | dict(coords_out)
        )

    output_mgr.write(x_out, coords_out)
