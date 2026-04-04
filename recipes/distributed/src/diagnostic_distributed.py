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

import numpy as np
import torch
from loguru import logger
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.distributed import (
    DistributedInference,
    local_concurrent_pipeline,
)
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class PrecipDiagnostic:
    """Wrapper for a precipitation diagnostic model distributed using DistributedInference.

    This provides an example of a class that can be used as the `dist_diagnostic` argument
    to `diagnostic_distributed`.
    """

    def __init__(self, output_coords: CoordSystem = OrderedDict({})):
        dist = DistributedManager()
        self.diagnostic = PrecipitationAFNO.load_model(
            PrecipitationAFNO.load_default_package()
        )
        self.diagnostic.to(dist.device)
        self.diagnostic_ic = self.diagnostic.input_coords()
        self.diagnostic_oc = self.diagnostic.output_coords(self.diagnostic_ic)
        self.output_coords = output_coords

    def get_coords(self) -> tuple[CoordSystem, CoordSystem]:
        """Get the input and output coordinates of the diagnostic model."""
        return (self.diagnostic_ic, self.diagnostic_oc)

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of the diagnostic model.

        Maps the input coordinates to the diagnostic model coordinates, runs the diagnostic,
        and maps the diagnostic model result to the output coordinates.
        """
        x, coords = map_coords(x, coords, self.diagnostic_ic)
        x, coords = self.diagnostic(x, coords)
        # Subselect domain/variables as indicated in output_coords
        (x, coords) = map_coords(x, coords, self.output_coords)
        return (x, coords)


# sphinx - diagnostic start
def diagnostic_distributed(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    dist_diagnostic: DistributedInference,
    data: DataSource,
    io: IOBackend,
    output_coords: CoordSystem = OrderedDict({}),
) -> IOBackend:
    """Distributed diagnostic workflow.
    This workflow creates a distributed inference pipeline that couples a prognostic
    model on the local rank with a diagnostic model on remote rank(s).

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        List of string, datetimes or np.datetime64
    nsteps : int
        Number of forecast steps
    prognostic : PrognosticModel
        Prognostic model
    dist_diagnostic: DistributedInference
        Wrapper for a diagnostic model distributed using DistributedInference,
        must be on same coordinate axis as prognostic. Must implement a `forward`
        method that wraps the call to the diagnostic model, and a `get_coords`
        method that returns a 2-tuple of (input_coords, output_coords).
    data : DataSource
        Data source
    io : IOBackend
        IO object
    output_coords: CoordSystem, optional
        IO output coordinate system override, by default OrderedDict({})
    device : torch.device, optional
        Device to run inference on, by default None

    Returns
    -------
    IOBackend
        Output IO object
    """
    # sphinx - diagnostic end
    logger.info("Running diagnostic workflow!")

    dist = DistributedManager()
    device = dist.device

    # Get information about the prognostic model
    logger.info(f"Prognostic rank: {dist.rank}")
    logger.info(f"Prognostic device: {device}")
    prognostic = prognostic.to(device)
    # Fetch data from data source and load onto device
    prognostic_ic = prognostic.input_coords()
    time = to_time_array(time)

    # Fetch initial conditions from data source and load onto device
    x, coords = fetch_data(
        source=data,
        time=time,
        variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"],
        device=device,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Get the input and output coordinates of the remote diagnostic model
    logger.info(f"Diagnostic ranks: {dist_diagnostic.remote_ranks}")
    (diagnostic_ic, diagnostic_oc) = dist_diagnostic.call_func("get_coords")[0]

    # Set up IO backend and create output variables
    _setup_io(io, time, nsteps, prognostic, diagnostic_oc, output_coords=output_coords)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic_ic)
    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    def prognostic_loop() -> None:
        """Pull outputs from prognostic model and pass to diagnostic models asynchronously."""
        for step, (x, coords) in enumerate(model):
            dist_diagnostic(x.clone(), coords)
            if step == nsteps:
                break
        dist_diagnostic.wait()

    def io_loop() -> None:
        """Receive outputs from diagnostic models and write to IO backend."""
        with tqdm(total=nsteps + 1, desc="Waiting for diagnostic model data") as pbar:
            for x, coords in dist_diagnostic.results():
                io.write(*split_coords(x, coords))
                pbar.update(1)

    logger.info("Inference starting!")
    # launch the functions making up the inference pipeline in their own threads
    # and wait for them to finish
    local_concurrent_pipeline([prognostic_loop, io_loop])
    logger.success("Inference complete")

    return io


def _setup_io(
    io: IOBackend,
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    diagnostic_oc: CoordSystem,
    output_coords: CoordSystem = OrderedDict({}),
) -> None:
    """Set up IO backend and create output variables."""

    total_coords = prognostic.output_coords(prognostic.input_coords())
    for key, value in prognostic.output_coords(
        prognostic.input_coords()
    ).items():  # Scrub batch dims
        if key in diagnostic_oc:
            total_coords[key] = diagnostic_oc[key]
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
