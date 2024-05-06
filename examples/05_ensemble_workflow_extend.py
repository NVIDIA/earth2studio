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
# Set Up
# ------
# All workflows inside Earth2Studio require constructed components to be
# handed to them. In this example, we will use the built in ensemble workflow
# :py:meth:`earth2studio.run.ensemble`.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :lines: 116-156

# %%
# We need the following:
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - perturbation_method: Extend the Spherical Gaussian Method :py:class:`earth2studio.perturbation.SphericalGaussian`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP
from earth2studio.perturbation import Perturbation, SphericalGaussian
from earth2studio.run import ensemble
from earth2studio.utils.type import CoordSystem

# Load the default model package which downloads the check point from NGC
package = DLWP.load_default_package()
model = DLWP.load_model(package)

# Create the data source
data = GFS()

# %%
# The perturbation method in :ref:`sphx_glr_examples_03_ensemble_workflow.py` is naive because it
# applies the same noise amplitude to every variable. We can create a custom wrapper
# that only applies the perturbation method to a particular variable instead.

# %%
class ApplyToVariable:
    """Apply a perturbation to only a particular variable."""

    def __init__(self, pm: Perturbation, variable: str | list[str]):
        self.pm = pm
        if isinstance(variable, str):
            variable = [variable]
        self.variable = variable

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # Apply perturbation
        xp, _ = self.pm(x, coords)
        # Add perturbed slice back into original tensor
        ind = np.in1d(coords["variable"], self.variable)
        x[..., ind, :, :] = xp[..., ind, :, :]
        return x, coords


# Generate a new noise amplitude that specifically targets 't2m' with a 1 K noise amplitude
avsg = ApplyToVariable(SphericalGaussian(noise_amplitude=1.0), "t2m")

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1}
io = ZarrBackend(file_name="outputs/05_ensemble_avsg.zarr", chunks=chunks)

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
#
# For the forecast we will predict for 10 steps (for FCN, this is 60 hours) with 8 ensemble
# members which will be ran in 2 batches with batch size 4.

# %%
nsteps = 10
nensemble = 8
batch_size = 4
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    avsg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results. Lets plot both the perturbed t2m field
# and also the unperturbed tcwv field. First to confirm the perturbation method works as
# expect, the initial state is plotted.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import matplotlib.pyplot as plt

forecast = "2024-01-01"


def plot_(axi, data, title, cmap):
    """Simple plot util function"""
    im = axi.imshow(data, cmap=cmap)
    plt.colorbar(im, ax=axi, shrink=0.5, pad=0.04)
    axi.set_title(title)


step = 0  # lead time = 24 hrs
plt.close("all")

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
plot_(
    ax[0, 0],
    np.mean(io["t2m"][:, 0, step], axis=0),
    f"{forecast} - t2m - Lead time: {6*step}hrs - Mean",
    "coolwarm",
)
plot_(
    ax[0, 1],
    np.std(io["t2m"][:, 0, step], axis=0),
    f"{forecast} - t2m - Lead time: {6*step}hrs - Std",
    "coolwarm",
)
plot_(
    ax[1, 0],
    np.mean(io["tcwv"][:, 0, step], axis=0),
    f"{forecast} - tcwv - Lead time: {6*step}hrs - Mean",
    "Blues",
)
plot_(
    ax[1, 1],
    np.std(io["tcwv"][:, 0, step], axis=0),
    f"{forecast} - tcwv - Lead time: {6*step}hrs - Std",
    "Blues",
)

plt.savefig(f"outputs/05_{forecast}_{step}_ensemble.jpg")

# %%
# Due to the intrinsic coupling between all fields, we should expect all variables to
# have some uncertainty for later lead times. Here the total column water vapor is
# plotted at a lead time of 24 hours, note the variance in the members despite just
# perturbing the temperature field.

# %%
step = 4  # lead time = 24 hrs
plt.close("all")

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
plot_(
    ax[0],
    np.mean(io["tcwv"][:, 0, step], axis=0),
    f"{forecast} - tcwv - Lead time: {6*step}hrs - Mean",
    "Blues",
)
plot_(
    ax[1],
    np.std(io["tcwv"][:, 0, step], axis=0),
    f"{forecast} - tcwv - Lead time: {6*step}hrs - Std",
    "Blues",
)

plt.savefig(f"outputs/05_{forecast}_{step}_ensemble.jpg")
