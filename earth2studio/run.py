# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
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
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    resume_from_step: int | None = None,
) -> IOBackend:
    """Built in deterministic workflow.
    This workflow creates a deterministic inference pipeline to produce a forecast
    prediction using a prognostic model. Supports saving and resuming from checkpoints to
    handle GPU memory constraints in long-running simulations.


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
    checkpoint_path : str, optional
        Path to save/load checkpoints, by default None
    checkpoint_interval : int, optional
        Save checkpoint every N steps, by default None
    resume_from_step : int, optional
        Resume from this step number, by default None

    Returns
    -------
    IOBackend
        Output IO object


    Examples
    --------

    Basic usage without checkpointing:
    >>> io = deterministic(time, nsteps, prognostic_model, data, io_backend)

    Save checkpoints every 5 steps:
    >>> io = deterministic(time, nsteps, prognostic_model, data, io_backend,
                            checkpoint_path="checkpoint.pt", checkpoint_interval=5)

    Resume from step 10:
    >>> io = deterministic(time, nsteps, prognostic_model, data, io_backend,
                            checkpoint_path="checkpoint.pt", resume_from_step=10)

    """
    from earth2studio.utils.checkpoint import (
        load_checkpoint,
        save_checkpoint,
        should_checkpoint,
        validate_checkpoint_compatibility,
    )

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

    # Handle resume from checkpoint
    if resume_from_step is not None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be provided when resume_from_step is specified"
            )
        checkpoint = load_checkpoint(checkpoint_path, device)

        logger.info(f"Resuming from checkpoint at step {resume_from_step}")

        if not validate_checkpoint_compatibility(checkpoint["coords"], prognostic):
            raise ValueError("Checkpoint incompatible with current prognostic model")

        x, coords = checkpoint["state"], checkpoint["coords"]
        start_step = resume_from_step

        logger.success("Resumed from checkpoint, skipping data fetch")
    else:
        # Normal initialization - fetch from data source

        # sphinx - fetch data start
        # Fetch data from data source and load onto device
        prognostic_ic = prognostic.input_coords()
        time = to_time_array(time)

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
        start_step = 0

        logger.success(f"Fetched data from {data.__class__.__name__}")
        # sphinx - fetch data end

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
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords())

    if resume_from_step is not None:
        # CHECKPOINT RESUME PATH - Manual time-stepping
        logger.info("Using manual time-stepping for checkpointed run")
        with tqdm(
            total=(nsteps + 1) - start_step, desc="Running inference", position=1
        ) as pbar:
            for current_step in range(start_step, nsteps + 1):
                x_out, coords_out = map_coords(x, coords, output_coords)
                io.write(*split_coords(x_out, coords_out))
                pbar.update(1)

                if (
                    should_checkpoint(
                        current_step, checkpoint_interval, checkpoint_path
                    )
                    and checkpoint_path is not None
                ):
                    save_checkpoint(
                        current_step, x, coords, checkpoint_path, "deterministic"
                    )
                    logger.info(f"Saved checkpoint at step {current_step}")

                if current_step < nsteps:
                    x, coords = prognostic(x, coords)

        logger.success("Inference complete")
        return io
    else:
        # NORMAL PATH - Use existing iterator
        # Create prognostic iterator
        model = prognostic.create_iterator(x, coords)

        logger.info("Inference starting!")
        with tqdm(total=nsteps + 1, desc="Running inference", position=1) as pbar:
            for step, (x, coords) in enumerate(model):
                # Subselect domain/variables as indicated in output_coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                pbar.update(1)

                # Save checkpoint if needed
                if (
                    should_checkpoint(
                        current_step, checkpoint_interval, checkpoint_path
                    )
                    and checkpoint_path is not None
                ):
                    save_checkpoint(step, x, coords, checkpoint_path, "deterministic")
                    logger.info(f"Saved checkpoint at step {step}")

                if step == nsteps:
                    break

        logger.success("Inference complete")
        return io


# sphinx - diagnostic start
def diagnostic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    diagnostic: DiagnosticModel,
    data: DataSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    resume_from_step: int | None = None,
) -> IOBackend:
    """Built in diagnostic workflow.
    This workflow creates a deterministic inference pipeline that couples a prognostic
    model with a diagnostic model. Supports saving and resuming from checkpoints to handle
    GPU memory constraints in long-running simulations.

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
    data : DataSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    checkpoint_path : str, optional
        Path to save/load checkpoints, by default None
    checkpoint_interval : int, optional
        Save checkpoint every N steps, by default None
    resume_from_step : int, optional
        Resume from this step number, by default None

    Returns
    -------
    IOBackend
        Output IO object


    Examples
    --------

    Basic usage without checkpointing:
    >>> io = diagnostic(time, nsteps, prognostic_model, diagnostic_model, data, io_backend)

    Save checkpoints every 5 steps:
    >>> io = diagnostic(time, nsteps, prognostic_model, diagnostic_model, data, io_backend,
                            checkpoint_path="checkpoint.pt", checkpoint_interval=5)

    Resume from step 10:
    >>> io = diagnostic(time, nsteps, prognostic_model, diagnostic_model, data, io_backend,
                            checkpoint_path="checkpoint.pt", resume_from_step=10)
    """

    from earth2studio.utils.checkpoint import (
        load_checkpoint,
        save_checkpoint,
        should_checkpoint,
        validate_checkpoint_compatibility,
    )

    # sphinx - diagnostic end
    logger.info("Running diagnostic workflow!")
    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    diagnostic = diagnostic.to(device)

    diagnostic_ic = diagnostic.input_coords()

    if resume_from_step is not None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be provided when resume_from_step is specified"
            )
        checkpoint = load_checkpoint(checkpoint_path, device)
        logger.info(f"Resuming from checkpoint at step {resume_from_step}")

        if not validate_checkpoint_compatibility(checkpoint["coords"], prognostic):
            raise ValueError("Checkpoint incompatible with current prognostic model")

        x, coords = checkpoint["state"], checkpoint["coords"]
        start_step = resume_from_step

        logger.success("Resumed from checkpoint, skipping data fetch")

    else:

        # Normal initialization - fetch from data source
        # Fetch data from data source and load onto device
        prognostic_ic = prognostic.input_coords()
        diagnostic_ic = diagnostic.input_coords()
        time = to_time_array(time)
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
        start_step = 0

        logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
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
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    prognostic_ic = prognostic.input_coords()
    x, coords = map_coords(x, coords, prognostic_ic)

    if resume_from_step is not None:
        # CHECKPOINT RESUME PATH - Manual time-stepping
        logger.info("Using manual time-stepping for checkpointed diagnostic run")
        prognostic_ic = prognostic.input_coords()

        with tqdm(
            total=(nsteps + 1) - start_step, desc="Running inference", position=1
        ) as pbar:
            for current_step in range(start_step, nsteps + 1):
                # Run diagnostic on current state
                x_diag, coords_diag = map_coords(x, coords, diagnostic_ic)
                x_diag, coords_diag = diagnostic(x_diag, coords_diag)

                # Output the diagnostic result
                x_out, coords_out = map_coords(x_diag, coords_diag, output_coords)
                io.write(*split_coords(x_out, coords_out))
                pbar.update(1)

                if (
                    should_checkpoint(
                        current_step, checkpoint_interval, checkpoint_path
                    )
                    and checkpoint_path is not None
                ):
                    save_checkpoint(
                        current_step, x, coords, checkpoint_path, "diagnostic"
                    )
                    logger.info(f"Saved checkpoint at step {current_step}")

                if current_step < nsteps:
                    x, coords = prognostic(x, coords)

        logger.success("Inference complete")
        return io

    else:
        # Create prognostic iterator
        model = prognostic.create_iterator(x, coords)

        logger.info("Inference starting!")
        with tqdm(total=nsteps + 1, desc="Running inference", position=1) as pbar:
            for step, (x, coords) in enumerate(model):

                # Run diagnostic
                x, coords = map_coords(x, coords, diagnostic_ic)
                x, coords = diagnostic(x, coords)
                # Subselect domain/variables as indicated in output_coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                pbar.update(1)
                if step == nsteps:
                    break

        logger.success("Inference complete")
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
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    resume_from_step: int | None = None,
) -> IOBackend:
    """Built in ensemble workflow.
    This workflow creates multiple forecast runs with perturbed initial conditions.
    Supports saving and resuming from checkpoints to handle GPU memory constraints
    in large ensemble simulations.

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
    perturbation: Perturbation
        Method to perturb the initial condition to create an ensemble.
    batch_size: int, optional
        Number of ensemble members to run in a single batch, by default None.
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None
    checkpoint_path : str, optional
        Path to save/load checkpoints, by default None
    checkpoint_interval : int, optional
        Save checkpoint every N steps, by default None
    resume_from_step : int, optional
        Resume from this step number, by default None

    Returns
    -------
    IOBackend
        Output IO object


    Examples
    --------
    Basic usage without checkpointing:
    >>> io = ensemble(time, nsteps, nensemble, prognostic_model, data, io_backend, perturbation)

    Save checkpoints every 5 steps:
    >>> io = ensemble(time, nsteps, nensemble, prognostic_model, data, io_backend, perturbation,
    ...             checkpoint_path="checkpoint.pt", checkpoint_interval=5)

    Resume from step 10:
    >>> io = ensemble(time, nsteps, nensemble, prognostic_model, data, io_backend, perturbation,
    ...             checkpoint_path="checkpoint.pt", resume_from_step=10)
    """

    from earth2studio.utils.checkpoint import (
        load_checkpoint,
        save_checkpoint,
        should_checkpoint,
        validate_checkpoint_compatibility,
    )

    # sphinx - ensemble end
    logger.info("Running ensemble inference!")

    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)

    x0: torch.Tensor
    coords0: CoordSystem

    if resume_from_step is not None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be provided when resume_from_step is specified"
            )
        checkpoint = load_checkpoint(checkpoint_path, device)
        logger.info(f"Resuming ensemble from checkpoint at step {resume_from_step}")

        if not validate_checkpoint_compatibility(checkpoint["coords"], prognostic):
            raise ValueError("Checkpoint incompatible with current prognostic model")

        # Expect checkpoint to contain all ensemble members
        x, coords = checkpoint["state"], checkpoint["coords"]
        start_step = resume_from_step

        if coords.get("ensemble") is None or len(coords["ensemble"]) != nensemble:
            raise ValueError(
                f"Checkpoint ensemble size {len(coords.get('ensemble', []))} does not match requested ensemble size {nensemble}"
            )

        logger.success(
            "Resumed ensemble from checkpoint, skipping data fetch perturbation"
        )

    else:

        # Fetch data from data source and load onto device
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
        start_step = 0

        logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend with information from output_coords (if applicable).
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
    io.add_array(total_coords, variables_to_save)

    # Compute batch sizes
    if batch_size is None:
        batch_size = nensemble
    batch_size = min(nensemble, batch_size)
    number_of_batches = ceil(nensemble / batch_size)

    if resume_from_step is not None:
        # CHECKPOINT RESUME PATH - Manual time-stepping
        logger.info("Using manual time-stepping for checkpointed ensemble run")

        with tqdm(
            total=(nsteps + 1) - start_step, desc="Running inference", position=1
        ) as pbar:
            for current_step in range(start_step, nsteps + 1):

                # Output current ensemble state (all members at once)
                x_out, coords_out = map_coords(x, coords, output_coords)
                io.write(*split_coords(x_out, coords_out))
                pbar.update(1)

                if (
                    should_checkpoint(
                        current_step, checkpoint_interval, checkpoint_path
                    )
                    and checkpoint_path is not None
                ):
                    save_checkpoint(
                        current_step, x, coords, checkpoint_path, "ensemble"
                    )
                    logger.info(f"Saved ensemble checkpoint at step {current_step}")

                if current_step < nsteps:
                    x, coords = prognostic(x, coords)

        logger.success("Inference complete")
        return io

    else:
        logger.info(
            f"Starting {nensemble} Member Ensemble Inference with \
                {number_of_batches} number of batches."
        )
        batch_id = 0
        for batch_id in tqdm(
            range(0, nensemble, batch_size),
            total=number_of_batches,
            desc="Total Ensemble Batches",
            position=2,
        ):

            # Get fresh batch data
            x = x0.to(device)

            # Expand x, coords for ensemble
            mini_batch_size = min(batch_size, nensemble - batch_id)
            coords = {
                "ensemble": np.arange(batch_id, batch_id + mini_batch_size)
            } | coords0.copy()

            # Unsqueeze x for batching ensemble
            x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * x.ndim))

            # Map lat and lon if needed
            x, coords = map_coords(x, coords, prognostic_ic)

            # Perturb ensemble
            x, coords = perturbation(x, coords)

            # Create prognostic iterator
            model = prognostic.create_iterator(x, coords)

            with tqdm(
                total=nsteps + 1,
                desc=f"Running batch {batch_id} inference",
                position=1,
                leave=False,
            ) as pbar:
                for step, (x, coords) in enumerate(model):
                    # Subselect domain/variables as indicated in output_coords
                    x, coords = map_coords(x, coords, output_coords)

                    io.write(*split_coords(x, coords))
                    pbar.update(1)

                    if batch_id == 0 and should_checkpoint(
                        step, checkpoint_interval, checkpoint_path
                    ):
                        logger.warning(
                            "Ensemble checkpointing in batched mode requires manual time-stepping - use resume_from_step for full functionality"
                        )

                    if step == nsteps:
                        break

            batch_id += 1

        logger.success("Inference complete")
        return io
