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

# %%
"""
Extending Diagostic Models
==========================

Implementing a custom diagnostic model

This example will demonstrate how extend Earth2Studio by implementing a custom
diagnostic model and running it in a general workflow.

In this example you will learn:

- API requirements of diagnostic models
- Implementing a custom diagnostic model
- Running this model in existing workflows
"""

# %%
# Custom Diagnostic
# -----------------
# As dicussed in the :ref:`diagnostic_model_userguide` section of the userguide,
# Earth2Studio defines a diagnostic model through a simple interface
# :py:class:`earth2studio.models.dx.base.Diagnostic Model`. This can be used to help
# guide the required APIs needed to successfully create our own model.
#
# In this example, lets consider a simple diagnostic that converts the surface
# temperature in Kelvin to Celcius to make it more readable for the average person.
#
# Our diagnostic model has a base class of :py:class:`torch.nn.Module` which allows us
# to get the required :py:obj:`to(device)` method for free.

# %%
from typing import Generator, Iterator
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.batch import batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class CustomDiagnostic(torch.nn.Module):
    """Custom dianostic model"""

    def __init__(self):
        super().__init__()

    input_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(["t2m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "batch": np.empty(1),
            "variable": np.array(["t2m_c"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs diagnostic model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        """
        for i, (key, value) in enumerate(self.input_coords.items()):
            if key != "batch":
                handshake_dim(coords, key, i)
                handshake_coords(coords, self.input_coords, key)

        out_coords = coords.copy()
        out_coords["variable"] = self.output_coords["variable"]
        out = x - 273.15  # To celcius

        return out, out_coords


# %%
# Input/Output Coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Defining the input/output coordinate systems is essential for any model in
# Earth2Studio since this how both the package and users can learn what type of data
# the model expects. Have a look at :ref:`coordinates_userguide` for details on
# coordinate system. For this diagnostic model, we simply define the input coordinates
# to be the global surface temperature specificed in
# :py:file:`earth2studio.lexicon.base.py`. The output is a custom variable
# :py:var:`t2m_c` that represents the temperature in Celcius.

# %%
# :py:func:`__call__` API
# ~~~~~~~~~~~~~~~~~~~~~~~
# The call function is the main API of diagnostic models that have a tensor and
# coordinate system as input/output. This function first valids the coordinate system
# are to be expected. Then both the input data tensor and also coordinate system are
# updated and returned.
#
# .. note::
#   You may notice the :py:func:`batch_func` decorator, which is used to make batched
#   operations easier. For more details about this refer to the :ref:`batch_function_userguide`
#   section of the userguide.

# %%
# Set Up
# ------
# With the custom diagnostic model defined, its now easily usable in a workflow. Lets
# create our own simple diagnostic workflow based on the ones that exist already in
# Earth2Studio.

# %%
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px import DiagnosticModel
from earth2studio.utils.coords import extract_coords, map_coords
from earth2studio.utils.time import to_time_array


def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    prognostic: PrognosticModel,
    diagnostic: DiagnosticModel,
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
    logger.info("Running diagnostic workflow!")
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

    for name, value in diagnostic.out_coords.items():
        if name == "batch":
            continue
        total_coords[name] = value

    var_names = total_coords.pop("variable")
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords)
    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
        for step, (x, coords) in enumerate(model):

            # Run diagnostic
            x, coords = map_coords(x, coords, diagnostic.output_coords)
            x, coords = diagnostic(x, coords)

            io.write(*extract_coords(x, coords))
            pbar.update(1)
            if step == nsteps:
                break

    logger.success("Inference complete")
    return io


# %%
# Lets instantiate the components needed.
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - Diagnostic Model: The custom diagnostic model defined above
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Lets save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
import numpy as np
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend

# Load the default model package which downloads the check point from NGC
package = DLWP.load_default_package()
model = DLWP.load_model(package)

# Diagnostic model
diagnostic = CustomDiagnostic()

# Create the data source
data = GFS()

# Create the IO handler, store in memory
io = ZarrBackend()

# %%
# Execute the Workflow
# --------------------
# Because the prognostic meets the needs of the interface, the workflow will execute
# just like any other model.

# %%
import earth2studio.run as run

nsteps = 24
io = run(["2024-01-01"], nsteps, model, diagnostic, data, io)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# To confirm that out prognostic model is working as expect, we should expect the fields
# to be progressively more noisy as time progresses.

# %%
import os

os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt

forecast = "2024-01-01"
variable = "t2m_c"

plt.close("all")

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(0, 2, figsize=(6, 4))

# Plot u10m every 6 hours
ax[0, 0].imshow(io[variable][0, 0], vmin=-20, vmax=20)
ax[0, 1].imshow(io[variable][0, 6], vmin=-20, vmax=20)
ax[1, 0].imshow(io[variable][0, 12], vmin=-20, vmax=20)
ax[1, 1].imshow(io[variable][0, 18], vmin=-20, vmax=20)


# Set title
plt.suptitle(f"{variable} - {forecast}")
times = io["lead_time"].astype("timedelta64[h]").astype(int)
ax[0, 0].set_title(f"Lead time: {times[0]}hrs")
ax[0, 1].set_title(f"Lead time: {times[6]}hrs")
ax[1, 0].set_title(f"Lead time: {times[12]}hrs")
ax[1, 1].set_title(f"Lead time: {times[18]}hrs")

plt.savefig("outputs/custom_prognostic_prediction.jpg", bbox_inches="tight")
