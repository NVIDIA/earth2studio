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

"""Built-in forecast workflows and work-item runners.

A :class:`~earth2studio.run.runner.Runner` executes one
:class:`~earth2studio.run.runner.WorkItem`. The built-ins are
:class:`PrognosticRunner`, which rolls out one prognostic model, and
:class:`DiagnosticRunner`, which applies diagnostics to source data. Coupled
runners live with the coupler, never here.
"""

from collections import OrderedDict
from datetime import datetime
from math import ceil
from typing import cast

import numpy as np
import torch
import xarray as xr
from loguru import logger
from tqdm import tqdm  # type: ignore[import-untyped]

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.io.base import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.run._fields import (  # noqa: F401 - re-exported helpers
    _dimension_coords,
    _io_template,
    _lead_times,
    _map_field,
    _rollout,
    _with_lead_time,
)
from earth2studio.run.runner import (
    DataRequest,
    DiagnosticRunner,
    PrognosticRunner,
    Runner,
    WorkItem,
)
from earth2studio.utils.checkpoint import (
    Checkpoint,
    CheckpointSession,
    NullCheckpoint,
)
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

__all__ = [
    "DataRequest",
    "DiagnosticRunner",
    "PrognosticRunner",
    "Runner",
    "WorkItem",
    "deterministic",
    "diagnostic",
    "ensemble",
]


def deterministic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    verbose: bool = True,
    checkpoint: Checkpoint | CheckpointSession | NullCheckpoint = NullCheckpoint(),
) -> IOBackend:
    """Built in deterministic workflow.
    This workflow creates a determinstic inference pipeline to produce a forecast
    prediction using a prognostic model.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic model
    data : DataSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    verbose : bool, optional
        Print inference progress, by default True
    checkpoint : Checkpoint, optional
        Checkpoint manager or checkpoint session used to record and resume workflow
        progress, by default no checkpoint

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running simple workflow!")
    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    prognostic_ic = cast(xr.DataArray, prognostic.input_coords())
    time = to_time_array(time)

    # Set up IO backend
    template = _io_template(
        prognostic.output_coords(prognostic_ic),
        nsteps,
        leading={"time": np.asarray(time)},
        output_coords=output_coords,
    )
    io.add_array(template)

    with checkpoint as ckpt:
        restart_step = None
        if ckpt.exists and ckpt.write_count > 0:
            if ckpt.catalog.level < 2:
                logger.warning(
                    "deterministic received checkpoint level "
                    f"{ckpt.catalog.level}; component state may not be "
                    "complete enough to resume a rollout. Re-running from "
                    "lead time zero."
                )
            else:
                restart_step = ckpt.write_count - 1
                if restart_step >= nsteps:
                    logger.success("\nInference complete")
                    return io

        # --8<-- [start:fetch-data]
        # Fetch data from data source and load onto device
        if hasattr(prognostic, "interp_method"):
            interp_to = prognostic_ic
            interp_method = prognostic.interp_method
        else:
            interp_to = None
            interp_method = "nearest"

        x = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic.coords["variable"].values,
            lead_time=prognostic_ic.coords["lead_time"].values,
            device=device,
            target_grid=interp_to,
            regridder=interp_method,
        )

        logger.success(f"Fetched data from {data.__class__.__name__}")
        # --8<-- [end:fetch-data]

        # Map lat and lon if needed
        x = _map_field(x, prognostic_ic)
        # Create prognostic iterator
        model = _rollout(prognostic, x, restart_step is None)

        logger.info("Inference starting!")
        initial_progress = 0 if restart_step is None else restart_step + 1
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, x in enumerate(model):
                step = (
                    local_step
                    if restart_step is None
                    else restart_step + local_step + 1
                )

                x = cast(xr.DataArray, x)
                current_lead_time = x.coords["lead_time"].values[-1]
                # Subselect domain/variables as indicated in output_coords
                x = _map_field(x, output_coords)
                io.write(x)
                ckpt.write(lead_time=current_lead_time)
                pbar.update(1)
                if step == nsteps:
                    break

        ckpt.flush()

    logger.success("\nInference complete")
    return io


def diagnostic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    diagnostic: DiagnosticModel,
    data: DataSource | ForecastSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    verbose: bool = True,
    checkpoint: Checkpoint | CheckpointSession | NullCheckpoint = NullCheckpoint(),
) -> IOBackend:
    """Built in diagnostic workflow.
    This workflow creates a determinstic inference pipeline that couples a prognostic
    model with a diagnostic model.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic model
    diagnostic: DiagnosticModel
        Diagnostic model, must be on same coordinate axis as prognostic
    data : DataSource | ForecastSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    verbose : bool, optional
        Print inference progress, by default True
    checkpoint : Checkpoint, optional
        Checkpoint manager or checkpoint session used to record and resume workflow
        progress, by default no checkpoint

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running diagnostic workflow!")
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    diagnostic = diagnostic.to(device)

    prognostic_ic = cast(xr.DataArray, prognostic.input_coords())
    diagnostic_ic = cast(xr.DataArray, diagnostic.input_coords())
    time = to_time_array(time)

    prognostic_oc = prognostic.output_coords(prognostic_ic)
    diagnostic_oc = diagnostic.output_coords(_map_field(prognostic_oc, diagnostic_ic))
    template = _io_template(
        diagnostic_oc,
        nsteps,
        leading={"time": np.asarray(time)},
        output_coords=output_coords,
        lead_time=_lead_times(prognostic_oc, nsteps),
    )
    io.add_array(template)

    with checkpoint as ckpt:
        restart_step = None
        if ckpt.exists and ckpt.write_count > 0:
            if ckpt.catalog.level < 2:
                logger.warning(
                    "diagnostic received checkpoint level "
                    f"{ckpt.catalog.level}; component state may not be "
                    "complete enough to resume a rollout. Re-running from "
                    "lead time zero."
                )
            else:
                restart_step = ckpt.write_count - 1
                if restart_step >= nsteps:
                    logger.success("\nInference complete")
                    return io

        if hasattr(prognostic, "interp_method"):
            interp_to = prognostic_ic
            interp_method = prognostic.interp_method
        else:
            interp_to = None
            interp_method = "nearest"

        x = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic.coords["variable"].values,
            lead_time=prognostic_ic.coords["lead_time"].values,
            device=device,
            target_grid=interp_to,
            regridder=interp_method,
        )
        logger.success(f"Fetched data from {data.__class__.__name__}")

        x = _map_field(x, prognostic_ic)
        model = _rollout(prognostic, x, restart_step is None)

        logger.info("Inference starting!")
        initial_progress = 0 if restart_step is None else restart_step + 1
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, x in enumerate(model):
                step = (
                    local_step
                    if restart_step is None
                    else restart_step + local_step + 1
                )

                x = cast(xr.DataArray, x)
                current_lead_time = x.coords["lead_time"].values[-1]
                x = diagnostic(_map_field(x, diagnostic_ic))
                x = _with_lead_time(x, current_lead_time)
                x = _map_field(x, output_coords)
                io.write(x)
                ckpt.write(lead_time=current_lead_time)
                pbar.update(1)
                if step == nsteps:
                    break

        ckpt.flush()

    logger.success("\nInference complete")
    return io


def ensemble(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    perturbation: Perturbation,
    batch_size: int | None = None,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    verbose: bool = True,
    checkpoint: Checkpoint | CheckpointSession | NullCheckpoint = NullCheckpoint(),
) -> IOBackend:
    """Built in ensemble workflow.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    nensemble : int
        Number of ensemble members to run inference for.
    prognostic : PrognosticModel
        Prognostic models
    data : DataSource
        Data source
    io : IOBackend
        IO object
    perturbation : Perturbation
        Method to perturb the initial condition to create an ensemble.
    batch_size: int, optional
        Number of ensemble members to run in a single batch,
        by default None.
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    verbose : bool, optional
        Print inference progress, by default True
    checkpoint : Checkpoint, optional
        Checkpoint manager or checkpoint session used to record and resume workflow
        progress, by default no checkpoint

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running ensemble inference!")

    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)

    prognostic_ic = cast(xr.DataArray, prognostic.input_coords())
    time = to_time_array(time)
    if hasattr(prognostic, "interp_method"):
        interp_to = prognostic_ic
        interp_method = prognostic.interp_method
    else:
        interp_to = None
        interp_method = "nearest"

    x0 = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic.coords["variable"].values,
        lead_time=prognostic_ic.coords["lead_time"].values,
        device=device,
        target_grid=interp_to,
        regridder=interp_method,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    x0 = _map_field(x0, prognostic_ic)
    template = _io_template(
        prognostic.output_coords(prognostic_ic),
        nsteps,
        leading={"ensemble": np.arange(nensemble), "time": np.asarray(time)},
        output_coords=output_coords,
    )
    io.add_array(template)
    # Member selections pick which members of each batch are written
    stored_members = template.coords["ensemble"].values
    field_coords = OrderedDict(
        (k, v) for k, v in output_coords.items() if k != "ensemble"
    )

    if batch_size is None:
        batch_size = nensemble
    batch_size = min(nensemble, batch_size)
    with checkpoint as ckpt:
        completed_ensembles = []
        if ckpt.exists and not isinstance(ckpt, NullCheckpoint):
            completed_ensembles = [
                int(value) for value in ckpt.metadata.get("completed_ensembles", [])
            ]

        completed = set(completed_ensembles)
        start_batch_id = next(
            (index for index in range(nensemble) if index not in completed),
            nensemble,
        )
        number_of_batches = ceil((nensemble - start_batch_id) / batch_size)
        restart_first_batch = (
            ckpt.exists
            and ckpt.write_count > 0
            and start_batch_id < nensemble
            and ckpt.lead_time != template.coords["lead_time"].values[-1]
        )

        logger.info(f"Starting {nensemble} Member Ensemble Inference with \
            {number_of_batches} number of batches.")
        for batch_index, batch_id in enumerate(
            tqdm(
                range(start_batch_id, nensemble, batch_size),
                total=number_of_batches,
                desc="Total Ensemble Batches",
                position=2,
                disable=(not verbose),
            )
        ):
            mini_batch_size = min(batch_size, nensemble - batch_id)
            ensemble_coords = np.arange(batch_id, batch_id + mini_batch_size)
            ensemble_members = [int(value) for value in ensemble_coords]
            written_members = np.intersect1d(ensemble_coords, stored_members)
            restart_step = None
            if batch_index == 0 and restart_first_batch:
                if ckpt.catalog.level < 2:
                    logger.warning(
                        "ensemble received checkpoint level "
                        f"{ckpt.catalog.level}; component state may not be "
                        "complete enough to resume a rollout. Re-running from "
                        "lead time zero."
                    )
                    ckpt.write_count = 0
                else:
                    restart_step = ckpt.write_count - 1
                    if restart_step >= nsteps:
                        continue
            elif not isinstance(ckpt, NullCheckpoint):
                ckpt.write_count = 0

            x = x0.expand_dims(ensemble=ensemble_coords).copy(deep=True)
            x = perturbation(x)

            model = _rollout(prognostic, x, restart_step is None)
            initial_progress = 0 if restart_step is None else restart_step + 1
            with tqdm(
                total=nsteps + 1,
                initial=initial_progress,
                desc=f"Running batch {batch_id} inference",
                position=1,
                leave=False,
                disable=(not verbose),
            ) as pbar:
                for local_step, x in enumerate(model):
                    step = (
                        local_step
                        if restart_step is None
                        else restart_step + local_step + 1
                    )

                    x = cast(xr.DataArray, x)
                    current_lead_time = x.coords["lead_time"].values[-1]
                    x = _map_field(x, field_coords)
                    if len(written_members) < mini_batch_size:
                        x = x.sel(ensemble=written_members)
                    if len(written_members):
                        io.write(x)
                    if step == nsteps:
                        completed.update(ensemble_members)
                        completed_ensembles = sorted(completed)
                    ckpt.write(
                        lead_time=current_lead_time,
                        completed_ensembles=completed_ensembles,
                    )
                    pbar.update(1)
                    if step == nsteps:
                        break

            ckpt.flush()

    logger.success("\nInference complete")
    return io
