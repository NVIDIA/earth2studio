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

from datetime import datetime
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import extract_coords, map_coords
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
