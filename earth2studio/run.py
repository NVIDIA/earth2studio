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
from contextlib import nullcontext
from datetime import datetime
from math import ceil
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.checkpoint import Checkpoint, CheckpointSelection
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
    checkpoint: Checkpoint | CheckpointSelection | None = None,
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
        Checkpoint catalog or selected checkpoint context used to record and resume
        workflow progress, by default None. Pass a selected context when components
        need to bind checkpoint state during construction.

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
    _add_output_array_if_needed(io, total_coords, var_names)

    checkpoint_context = _checkpoint_context(checkpoint, time=time)

    with checkpoint_context as ckpt:
        restart_step = None
        if ckpt is not None and ckpt.exists:
            restart_step = _lead_time_index(total_coords["lead_time"], ckpt.lead_time)
            if restart_step >= nsteps:
                logger.success("\nInference complete")
                return io
            x, coords = _read_restart_from_io(
                io, prognostic_ic, time, ckpt.lead_time, device
            )
        else:
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
        last_checkpoint_entry = None
        last_coords = coords
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, (x, coords) in enumerate(model):
                step = local_step if restart_step is None else restart_step + local_step
                if restart_step is not None and local_step == 0:
                    continue

                last_coords = coords
                # Subselect domain/variables as indicated in output_coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                if ckpt is not None:
                    last_checkpoint_entry = ckpt.write(coords=last_coords)
                pbar.update(1)
                if step == nsteps:
                    break

        if ckpt is not None and last_checkpoint_entry is None:
            ckpt.flush(coords=last_coords)

    logger.success("\nInference complete")
    return io


def _checkpoint_context(
    checkpoint: Checkpoint | CheckpointSelection | None, **labels: Any
):
    if isinstance(checkpoint, CheckpointSelection):
        return checkpoint
    if checkpoint is not None:
        return checkpoint.select(**labels)
    return nullcontext(None)


def _add_output_array_if_needed(
    io: IOBackend, coords: CoordSystem, var_names: np.ndarray
) -> None:
    missing = list(var_names)
    try:
        missing = [name for name in var_names if name not in io]
    except TypeError:
        pass
    if missing:
        io.add_array(coords, missing)


def _lead_time_index(lead_times: np.ndarray, lead_time: Any) -> int:
    value = np.asarray(lead_time).reshape(-1)[0]
    index = np.where(lead_times == value)[0]
    if index.shape[0] == 0:
        raise ValueError(
            f"Checkpoint lead_time {lead_time} is not in workflow lead_time coordinates."
        )
    return int(index[0])


def _read_restart_from_io(
    io: IOBackend,
    prognostic_coords: CoordSystem,
    time: np.ndarray,
    lead_time: Any,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    if not hasattr(io, "read"):
        raise RuntimeError(
            "Checkpoint resume requires an IO backend with read(coords, array_name, device)."
        )

    variable = prognostic_coords["variable"]
    read_coords = _restart_read_coords(prognostic_coords, time, lead_time)
    coords = _insert_variable_coord(read_coords, variable)

    xs = []
    for name in variable:
        try:
            x, _ = io.read(read_coords, str(name), device=device)
        except (AssertionError, KeyError, ValueError) as error:
            raise RuntimeError(
                f"Checkpoint resume requires IO data for variable {str(name)!r} "
                "at the selected checkpoint lead time."
            ) from error
        xs.append(x)

    variable_index = list(coords).index("variable")
    x = torch.stack(xs, dim=variable_index)
    return x, coords


def _restart_read_coords(
    prognostic_coords: CoordSystem, time: np.ndarray, lead_time: Any
) -> CoordSystem:
    restart_lead_time = _restart_lead_time(
        prognostic_coords["lead_time"], lead_time
    )
    read_coords: CoordSystem = OrderedDict()
    time_added = False

    def add_time() -> None:
        nonlocal time_added
        if not time_added:
            read_coords["time"] = time
            time_added = True

    for key, value in prognostic_coords.items():
        if key in ("batch", "variable"):
            continue
        if key == "time":
            add_time()
            continue
        if value.shape == (0,):
            continue
        if key == "lead_time":
            add_time()
            read_coords["lead_time"] = restart_lead_time
            continue
        read_coords[key] = value

    add_time()
    if "lead_time" not in read_coords:
        read_coords["lead_time"] = np.asarray([lead_time])
    return read_coords


def _insert_variable_coord(read_coords: CoordSystem, variable: np.ndarray) -> CoordSystem:
    coords: CoordSystem = OrderedDict()
    inserted = False
    for key, value in read_coords.items():
        coords[key] = value
        if key == "lead_time":
            coords["variable"] = variable
            inserted = True
    if not inserted:
        coords["variable"] = variable
    return coords


def _restart_lead_time(input_lead_time: np.ndarray, lead_time: Any) -> np.ndarray:
    base_lead_time = np.asarray(lead_time).reshape(-1)[0]
    return np.asarray(input_lead_time) + base_lead_time


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
    checkpoint: Checkpoint | CheckpointSelection | None = None,
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
        Checkpoint catalog or selected checkpoint context used to record and resume
        workflow progress, by default None. Resume requires the IO backend to contain
        the prognostic variables needed for the next forecast step.

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
    _add_output_array_if_needed(io, total_coords, var_names)

    checkpoint_context = _checkpoint_context(checkpoint, time=time)
    with checkpoint_context as ckpt:
        restart_step = None
        if ckpt is not None and ckpt.exists:
            restart_step = _lead_time_index(total_coords["lead_time"], ckpt.lead_time)
            if restart_step >= nsteps:
                logger.success("\nInference complete")
                return io
            x, coords = _read_restart_from_io(
                io, prognostic_ic, time, ckpt.lead_time, device
            )
        else:
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
        last_checkpoint_entry = None
        last_coords = coords
        with tqdm(
            total=nsteps + 1,
            initial=initial_progress,
            desc="Running inference",
            position=1,
            disable=(not verbose),
        ) as pbar:
            for local_step, (x, coords) in enumerate(model):
                step = local_step if restart_step is None else restart_step + local_step
                if restart_step is not None and local_step == 0:
                    continue

                x, coords = map_coords(x, coords, diagnostic_ic)
                x, coords = diagnostic(x, coords)
                last_coords = coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*split_coords(x, coords))
                if ckpt is not None:
                    last_checkpoint_entry = ckpt.write(coords=last_coords)
                pbar.update(1)
                if step == nsteps:
                    break

        if ckpt is not None and last_checkpoint_entry is None:
            ckpt.flush(coords=last_coords)

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
    checkpoint: Checkpoint | CheckpointSelection | None = None,
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
        Checkpoint catalog or selected checkpoint context used to record and resume
        workflow progress, by default None. When a catalog is provided, rows are
        tracked independently for each ensemble batch.

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
    _add_output_array_if_needed(io, total_coords, variables_to_save)

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
        checkpoint_context = _checkpoint_context(
            checkpoint, time=time, ensemble_batch=batch_id
        )

        with checkpoint_context as ckpt:
            restart_step = None
            if ckpt is not None and ckpt.exists:
                restart_step = _lead_time_index(
                    total_coords["lead_time"], ckpt.lead_time
                )
                if restart_step >= nsteps:
                    continue
                restart_coords = OrderedDict({"ensemble": ensemble_coords}) | prognostic_ic
                x, coords = _read_restart_from_io(
                    io, restart_coords, time, ckpt.lead_time, device
                )
            else:
                x = x0.to(device)
                coords = OrderedDict({"ensemble": ensemble_coords}) | coords0.copy()
                x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * x.ndim))
                x, coords = map_coords(x, coords, prognostic_ic)
                x, coords = perturbation(x, coords)

            model = prognostic.create_iterator(x, coords)
            initial_progress = 0 if restart_step is None else restart_step + 1
            last_checkpoint_entry = None
            last_coords = coords
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
                        else restart_step + local_step
                    )
                    if restart_step is not None and local_step == 0:
                        continue

                    last_coords = coords
                    x, coords = map_coords(x, coords, output_coords)
                    io.write(*split_coords(x, coords))
                    if ckpt is not None:
                        last_checkpoint_entry = ckpt.write(coords=last_coords)
                    pbar.update(1)
                    if step == nsteps:
                        break

            if ckpt is not None and last_checkpoint_entry is None:
                ckpt.flush(coords=last_coords)

    logger.success("\nInference complete")
    return io
