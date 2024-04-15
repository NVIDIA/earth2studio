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
Single Variable Perturbation Method
===================================

Intermediate ensemble inference using a custom perturbation method.

This example will demonstrate how to run a an ensemble inference workflow
with a custom perturbation method that only applies noise to a specific variable.

In this example you will learn:

- How to extend an existing pertubration method
- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Running a simple built in workflow
- Extend a built-in method using custom code.
- Post-processing results
"""

# %%
# Creating an Ensemble Workflow
# -----------------------------------
#
# To start lets begin with creating a ensemble workflow to use. We encourage
# users to explore and experiment with their own custom workflows that borrow ideas from
# built in workflows inside :py:obj:`earth2studio.run` or the examples.
#
# Creating our own generalizable ensemble workflow is easy when we rely on the component
# interfaces defined in Earth2Studio (use dependency injection). Here we create a run
# method that accepts the following:
#
# - time: Input list of datetimes / strings to run inference for
# - nsteps: Number of forecast steps to predict
# - nensemble: Number of ensembles to run for
# - prognostic: Our initialized prognostic model
# - perturbation_method: Our initialized pertubation method
# - data: Initialized data source to fetch initial conditions from
# - io: IOBackend

# %%
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import PerturbationMethod
from earth2studio.utils.coords import extract_coords, map_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def run_ensemble(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    prognostic: PrognosticModel,
    perturbation_method: PerturbationMethod,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:
    """Simple ensemble workflow

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
    perturbation_method : PerturbationMethod
        Method of perturbing the initial condition to form an ensemble.
    data : DataSource
        Data source
    io : IOBackend
        IO object

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

    # Expand x, coords for ensemble
    x = x.unsqueeze(0).repeat(nensemble, *([1] * x.ndim))
    coords = {"ensemble": np.arange(nensemble)} | coords

    # Set up IO backend
    total_coords = coords.copy()
    total_coords["lead_time"] = np.asarray(
        [prognostic.output_coords["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()

    var_names = total_coords.pop("variable")
    io.add_array(total_coords, var_names)

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords)

    # Perturb ensemble
    dx, coords = perturbation_method(x, coords)
    x += dx

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


# %%
# Set Up
# ------
# With the ensemble workflow defined, we now need to create the indivdual components.
#
# We need the following:
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - perturbation_method: Extend the Spherical Gaussian Method :py:class:`earth2studio.perturbation.SphericalGaussian`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Lets save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# %%
from typing import List, Union

import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP
from earth2studio.perturbation import PerturbationMethod, SphericalGaussian
from earth2studio.utils.type import CoordSystem

# Load the default model package which downloads the check point from NGC
package = DLWP.load_default_package()
model = DLWP.load_model(package)

# Create the data source
data = GFS()

# %%
# The perturbation method in 02_ensemble_workflow.py is naive because it applies the
# same noise amplitude to every variable. We can create a custom wrapper that only
# applies the perturbation method to a particular variable instead.

# %%
class ApplyToVariable:
    """Apply a perturbation to only a particular variable."""

    def __init__(self, pm: PerturbationMethod, variable: Union[str, List[str]]):
        self.pm = pm
        if isinstance(variable, str):
            variable = [variable]
        self.variable = variable

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        # Construct perturbation
        dx = self.pm(x, coords)
        # Find variable in data
        ind = np.in1d(coords["variable"], self.variable)
        dx[..., ~ind, :, :] = 0.0
        return dx


# Generate a new noise amplitude that specifically targets 't2m' with a 1 K noise amplitude
avsg = ApplyToVariable(SphericalGaussian(noise_amplitude=1.0), "t2m")

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1}
io = ZarrBackend(file_name="outputs/ensemble_avsg.zarr", chunks=chunks)

# %%
# Execute the Workflow
# --------------------
# With all componments intialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 forecast steps which is 2.5 days. To reduce
# computation time we will only simulate 8 ensemble members.
# %%

nsteps = 10
nensemble = 8
io = run_ensemble(["2024-01-01"], nsteps, nensemble, model, avsg, data, io)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Cartopy is a greate library for plotting
# fields on projects of a sphere.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = "2024-01-01"


def plot_(axi, data, title, cmap):
    """Convenience function for plotting pcolormesh."""
    # Plot the field using pcolormesh
    im = axi.pcolormesh(
        io["lon"][:],
        io["lat"][:],
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    plt.colorbar(im, ax=axi)
    # Set title
    axi.set_title(title)
    # Add coastlines and gridlines
    axi.coastlines()
    axi.gridlines()


for variable, cmap in zip(["t2m", "tcwv"], ["coolwarm", "Blues"]):
    step = 0  # lead time = 24 hrs

    plt.close("all")
    # Create a Robinson projection
    projection = ccrs.Robinson()

    # Create a figure and axes with the specified projection
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(12, 5)
    )

    plot_(
        ax1,
        io[variable][0, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {0}",
        cmap,
    )
    plot_(
        ax2,
        io[variable][1, 0, step],
        f"{forecast} - Lead time: {6*step}hrs - Member: {1}",
        cmap,
    )
    plot_(
        ax3,
        np.std(io[variable][:, 0, step], axis=0),
        f"{forecast} - Lead time: {6*step}hrs - Std",
        cmap,
    )

    plt.savefig(f"outputs/04_{forecast}_{variable}_{step}_ensemble.jpg")
