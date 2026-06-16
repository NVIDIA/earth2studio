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

from collections import OrderedDict
from datetime import datetime
from math import ceil

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.checkpoint import (
    NO_CHECKPOINT,
    Checkpoint,
    CheckpointSession,
    NullCheckpointSession,
)
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


# sphinx - deterministic start
def deterministic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    verbose: bool = True,
    checkpoint: Checkpoint | CheckpointSession | NullCheckpointSession = NO_CHECKPOINT,
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
        progress, by default no checkpoint. Use the checkpoint as a context manager
        to make the active session explicit when restart-aware components need to
        bind state before the workflow starts.

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - deterministic end
    logger.info("Running simple workflow!")
    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    # sphinx - fetch data start
    # Fetch data from data source and load onto device
    prognostic_ic = prognostic.input_coords()
    time = to_time_array(time)

    # Set up IO backend
    total_coords = prognostic.output_coords(prognostic.input_coords()).copy()
    for key, value in prognostic.output_coords(
        prognostic.input_coords()
    ).items():  # Scrub batch dims
        if value.shape == (0,):
            del total_coords[key]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)

    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)
    var_names = total_coords.pop("variable")
    missing = list(var_names)
    try:
        missing = [name for name in var_names if name not in io]
    except TypeError:
        pass
    if missing:
        io.add_array(total_coords, missing)

    with checkpoint as ckpt:
        restart_step = None
        if ckpt.exists and ckpt.write_count > 0:
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

        x, coords = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic["variable"],
            lead_time=prognostic_ic["lead_time"],
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )

        logger.success(f"Fetched data from {data.__class__.__name__}")
        # sphinx - fetch data end

        # Map lat and lon if needed
        x, coords = map_coords(x, coords, prognostic.input_coords())
        # Create prognostic iterator
        model = prognostic.create_iterator(x, coords)

        logger.info("Inference starting!")
        initial_progress = 0 if restart_step is None else restart_step + 1
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, (x, coords) in enumerate(model):
                step = (
                    local_step
                    if restart_step is None
                    else restart_step + local_step + 1
                )

                current_lead_time = coords["lead_time"][-1]
                # Subselect domain/variables as indicated in output_coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                ckpt.write(lead_time=current_lead_time)
                pbar.update(1)
                if step == nsteps:
                    break

        ckpt.flush()

    logger.success("\nInference complete")
    return io


# sphinx - diagnostic start
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
    checkpoint: Checkpoint | CheckpointSession | NullCheckpointSession = NO_CHECKPOINT,
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
        progress, by default no checkpoint. When resuming, the workflow fetches the
        normal initial condition and checkpoint-aware models restore from their own
        bound checkpoint state.

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - diagnostic end
    logger.info("Running diagnostic workflow!")
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    diagnostic = diagnostic.to(device)

    prognostic_ic = prognostic.input_coords()
    diagnostic_ic = diagnostic.input_coords()
    time = to_time_array(time)

    total_coords = prognostic.output_coords(prognostic.input_coords())
    for key, value in prognostic.output_coords(
        prognostic.input_coords()
    ).items():  # Scrub batch dims
        if key in diagnostic.output_coords(diagnostic_ic):
            total_coords[key] = diagnostic.output_coords(diagnostic_ic)[key]
        if value.shape == (0,):
            del total_coords[key]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)

    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)
    var_names = total_coords.pop("variable")
    missing = list(var_names)
    try:
        missing = [name for name in var_names if name not in io]
    except TypeError:
        pass
    if missing:
        io.add_array(total_coords, missing)

    with checkpoint as ckpt:
        restart_step = None
        if ckpt.exists and ckpt.write_count > 0:
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

        x, coords = fetch_data(
            source=data,
            time=time,
            variable=prognostic_ic["variable"],
            lead_time=prognostic_ic["lead_time"],
            device=device,
            interp_to=interp_to,
            interp_method=interp_method,
        )
        logger.success(f"Fetched data from {data.__class__.__name__}")

        x, coords = map_coords(x, coords, prognostic_ic)
        model = prognostic.create_iterator(x, coords)

        logger.info("Inference starting!")
        initial_progress = 0 if restart_step is None else restart_step + 1
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, (x, coords) in enumerate(model):
                step = (
                    local_step
                    if restart_step is None
                    else restart_step + local_step + 1
                )

                current_lead_time = coords["lead_time"][-1]
                x, coords = map_coords(x, coords, diagnostic_ic)
                x, coords = diagnostic(x, coords)
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                ckpt.write(lead_time=current_lead_time)
                pbar.update(1)
                if step == nsteps:
                    break

        ckpt.flush()

    logger.success("\nInference complete")
    return io


# sphinx - ensemble start
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
    checkpoint: Checkpoint | CheckpointSession | NullCheckpointSession = NO_CHECKPOINT,
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
    perturbation_method : Perturbation
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
        progress, by default no checkpoint. When a checkpoint manager is provided, rows are tracked
        independently for each ensemble batch.

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - ensemble end
    logger.info("Running ensemble inference!")

    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)

    prognostic_ic = prognostic.input_coords()
    time = to_time_array(time)
    if hasattr(prognostic, "interp_method"):
        interp_to = prognostic_ic
        interp_method = prognostic.interp_method
    else:
        interp_to = None
        interp_method = "nearest"

    x0, coords0 = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
        interp_to=interp_to,
        interp_method=interp_method,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    total_coords = prognostic.output_coords(prognostic.input_coords()).copy()
    if "batch" in total_coords:
        del total_coords["batch"]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [
            prognostic.output_coords(prognostic.input_coords())["lead_time"] * i
            for i in range(nsteps + 1)
        ]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    total_coords = {"ensemble": np.arange(nensemble)} | total_coords

    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)
    variables_to_save = total_coords.pop("variable")
    missing = list(variables_to_save)
    try:
        missing = [name for name in variables_to_save if name not in io]
    except TypeError:
        pass
    if missing:
        io.add_array(total_coords, missing)

    if batch_size is None:
        batch_size = nensemble
    batch_size = min(nensemble, batch_size)
    number_of_batches = ceil(nensemble / batch_size)

    logger.info(
        f"Starting {nensemble} Member Ensemble Inference with \
            {number_of_batches} number of batches."
    )
    for batch_id in tqdm(
        range(0, nensemble, batch_size),
        total=number_of_batches,
        desc="Total Ensemble Batches",
        position=2,
        disable=(not verbose),
    ):
        mini_batch_size = min(batch_size, nensemble - batch_id)
        ensemble_coords = np.arange(batch_id, batch_id + mini_batch_size)
        batch_checkpoint = checkpoint
        if isinstance(batch_checkpoint, Checkpoint):
            active = batch_checkpoint.active
            batch_checkpoint = (
                active
                if active is not None
                else batch_checkpoint.select(ensemble_batch=batch_id)
            )

        with batch_checkpoint as ckpt:
            restart_step = None
            if ckpt.exists and ckpt.write_count > 0:
                restart_step = ckpt.write_count - 1
                if restart_step >= nsteps:
                    continue

            x = x0.to(device)
            coords = OrderedDict({"ensemble": ensemble_coords}) | coords0.copy()
            x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * x.ndim))
            x, coords = map_coords(x, coords, prognostic_ic)
            x, coords = perturbation(x, coords)

            model = prognostic.create_iterator(x, coords)
            initial_progress = 0 if restart_step is None else restart_step + 1
            with tqdm(
                total=nsteps + 1,
                initial=initial_progress,
                desc=f"Running batch {batch_id} inference",
                position=1,
                leave=False,
                disable=(not verbose),
            ) as pbar:
                for local_step, (x, coords) in enumerate(model):
                    step = (
                        local_step
                        if restart_step is None
                        else restart_step + local_step + 1
                    )

                    current_lead_time = coords["lead_time"][-1]
                    x, coords = map_coords(x, coords, output_coords)
                    io.write(*split_coords(x, coords))
                    ckpt.write(lead_time=current_lead_time)
                    pbar.update(1)
                    if step == nsteps:
                        break

            ckpt.flush()

    logger.success("\nInference complete")
    return io
