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
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, cat_coords, map_coords

from .output import OutputManager
from .work import WorkItem


@torch.inference_mode()
def run_inference(
    work_items: list[WorkItem],
    prognostic: PrognosticModel,
    data_source: DataSource,
    output_mgr: OutputManager,
    nsteps: int,
    perturbation: Perturbation | None = None,
    diagnostics: list[DiagnosticModel] | None = None,
    device: torch.device | None = None,
) -> None:
    """Run distributed inference over a list of work items.

    Each work item represents a single forecast (one initial time and
    ensemble member).  The function iterates through the assigned work,
    fetches initial conditions, optionally perturbs them, runs the
    prognostic model, applies any diagnostic models, and writes results
    through the OutputManager.

    Parameters
    ----------
    work_items : list[WorkItem]
        Work items assigned to this rank (already distributed).
    prognostic : PrognosticModel
        The prognostic forecast model.
    data_source : DataSource
        Source for initial condition data.
    output_mgr : OutputManager
        Context-managed output handler (must already be entered).
    nsteps : int
        Number of forecast steps per initial condition.
    perturbation : Perturbation, optional
        IC perturbation method.  If None, forecasts are deterministic.
    diagnostics : list[DiagnosticModel], optional
        Diagnostic models to apply at each step.
    device : torch.device, optional
        Device for inference.  Defaults to CUDA if available.
    """
    if not work_items:
        logger.warning("No work items for this rank — skipping inference.")
        return

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostics = diagnostics or []

    prognostic = prognostic.to(device)
    diagnostics = [dx.to(device) for dx in diagnostics]
    dx_input_coords: dict[int, CoordSystem] = {
        id(dx): dx.input_coords() for dx in diagnostics
    }

    prognostic_ic = prognostic.input_coords()
    output_coords = output_mgr.output_coords
    has_ensemble = "ensemble" in output_mgr.io.coords

    for item in tqdm(work_items, desc="Work items", position=0):
        _run_single_forecast(
            item=item,
            prognostic=prognostic,
            prognostic_ic=prognostic_ic,
            data_source=data_source,
            output_mgr=output_mgr,
            output_coords=output_coords,
            nsteps=nsteps,
            perturbation=perturbation,
            diagnostics=diagnostics,
            dx_input_coords=dx_input_coords,
            has_ensemble=has_ensemble,
            device=device,
        )

    logger.success("Inference complete")


def _run_single_forecast(
    item: WorkItem,
    prognostic: PrognosticModel,
    prognostic_ic: CoordSystem,
    data_source: DataSource,
    output_mgr: OutputManager,
    output_coords: CoordSystem,
    nsteps: int,
    perturbation: Perturbation | None,
    diagnostics: list[DiagnosticModel],
    dx_input_coords: dict[int, CoordSystem],
    has_ensemble: bool,
    device: torch.device,
) -> None:
    """Run one forecast for a single WorkItem and write output.

    The model runs without any ensemble dimension — each work item is a
    single deterministic or perturbed forecast.  The ensemble coordinate
    is injected at write time so the data lands in the correct slice of
    the output store.
    """
    time = [item.time]

    x, coords = fetch_data(
        source=data_source,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
    )

    x, coords = map_coords(x, coords, prognostic_ic)

    if perturbation is not None:
        torch.manual_seed(item.seed)
        x, coords = perturbation(x, coords)

    model_iter = prognostic.create_iterator(x, coords)

    for step, (x_step, coords_step) in enumerate(
        tqdm(
            model_iter,
            total=nsteps + 1,
            desc=f"IC {item.time}",
            position=1,
            leave=False,
        )
    ):
        for dx in diagnostics:
            dx_ic = dx_input_coords[id(dx)]
            y, y_coords = map_coords(x_step, coords_step, dx_ic)
            y, y_coords = dx(y, y_coords)
            x_step, coords_step = cat_coords(
                (x_step, y), (coords_step, y_coords), "variable"
            )

        x_out, coords_out = map_coords(x_step, coords_step, output_coords)

        if has_ensemble:
            x_out = x_out.unsqueeze(0)
            coords_out = CoordSystem(
                {"ensemble": np.array([item.ensemble_id])} | dict(coords_out)
            )

        output_mgr.write(x_out, coords_out)

        if step >= nsteps:
            break
