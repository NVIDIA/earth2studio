# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import PerturbationMethod
from earth2studio.utils.coords import CoordSystem, extract_coords, map_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def deterministic(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    device: Optional[torch.device] = None,
) -> IOBackend:
    """Simple built in deterministic workflow

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic models
    data : DataSource
        Data source
    io : IOBackend
        IO object
    device : Optional[torch.device], optional
        Device to run inference on, by default None

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running simple workflow!")
    # Load model onto the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    # Fetch data from data source and load onto device
    time = to_time_array(time)
    x, coords = fetch_data(
        source=data,
        time=time,
        lead_time=prognostic.input_coords["lead_time"],
        variable=prognostic.input_coords["variable"],
        device=device,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    total_coords = prognostic.output_coords.copy()
    del total_coords["batch"]  # Unsafe if batch not supported
    for key, value in total_coords.items():
        if value.shape == 0:
            del total_coords[key]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [prognostic.output_coords["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    var_names = total_coords.pop("variable")
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords)
    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
        for step, (x, coords) in enumerate(model):
            io.write(*extract_coords(x, coords))
            pbar.update(1)
            if step == nsteps:
                break

    logger.success("Inference complete")
    return io


def ensemble(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    prognostic: PrognosticModel,
    data: DataSource,
    io: IOBackend,
    perturbation_method: PerturbationMethod,
    batch_size: Optional[int] = None,
    output_coords: CoordSystem = OrderedDict({}),
    device: Optional[torch.device] = None,
) -> IOBackend:
    """Ensemble workflow

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
    perturbation_method : PerturbationMethod
        Method to perturb the initial condition to create an ensemble.
    batch_size: Optional[int], optional
        Number of ensemble members to run in a single batch,
        by default None.
    device : Optional[torch.device], optional
        Device to run inference on, by default None

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running ensemble inference!")

    # Load model onto the device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)

    # Fetch data from data source and load onto device
    time = to_time_array(time)
    x0, coords0 = fetch_data(
        source=data,
        time=time,
        lead_time=prognostic.input_coords["lead_time"],
        variable=prognostic.input_coords["variable"],
        device="cpu",
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend with information from output_coords (if applicable).
    total_coords = {"ensemble": np.arange(nensemble)} | coords0.copy()
    total_coords["lead_time"] = np.asarray(
        [prognostic.output_coords["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()
    for key, value in total_coords.items():
        total_coords[key] = output_coords.get(key, value)

    variables_to_save = total_coords.pop("variable")
    io.add_array(total_coords, variables_to_save)

    # Compute batch sizes
    if batch_size is None:
        batch_size = nensemble
    batch_size = min(nensemble, batch_size)
    number_of_batches = ceil(nensemble / batch_size)

    logger.info(
        f"Starting {nensemble} Member Ensemble Inference with \
            {number_of_batches} number of batches."
    )
    batch_id = 0
    for batch_id in tqdm(
        range(0, nensemble, batch_size),
        total=number_of_batches,
        desc="Total Ensemble Batches",
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
        x, coords = map_coords(x, coords, prognostic.input_coords)

        # Perturb ensemble
        dx, coords = perturbation_method(x, coords)
        x += dx

        # Create prognostic iterator
        model = prognostic.create_iterator(x, coords)

        with tqdm(
            total=nsteps + 1, desc=f"Running batch {batch_id} inference", leave=False
        ) as pbar:
            for step, (x, coords) in enumerate(model):
                # Subselect domain/variables as indicated in output_coords
                x, coords = map_coords(x, coords, output_coords)
                io.write(*extract_coords(x, coords))
                pbar.update(1)
                if step == nsteps:
                    break

        batch_id += 1

    logger.success("Ensemble Inference complete")
    return io
