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

# %%
"""
Extending Diagnostic Models
===========================

Implementing a custom diagnostic model

This example will demonstrate how to extend Earth2Studio by implementing a custom
diagnostic model and running it in a general workflow.

In this example you will learn:

- API requirements of diagnostic models
- Implementing a custom diagnostic model
- Running this custom model in a workflow with built in prognostic
"""

# /// script
# dependencies = [
#   "earth2studio[dlwp] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Custom Diagnostic
# -----------------
# As discussed in the [Diagnostic Models](../../userguide/components/diagnostic.md#diagnostic_model_userguide) section of the user guide,
# Earth2Studio defines a diagnostic model through a simple interface
# [`earth2studio.models.dx.base.DiagnosticModel`][earth2studio.models.dx.base.DiagnosticModel]. This can be used to help
# guide the required APIs needed to successfully create our own model.
#
# In this example, lets consider a simple diagnostic that converts the surface
# temperature in Kelvin to Celsius to make it more readable for the average person.
#
# Our diagnostic model has a base class of `torch.nn.Module` which allows us
# to get the required `to(device)` method for free.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function


import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


class CustomDiagnostic(torch.nn.Module):
    """Custom dianostic model"""

    def __init__(self):
        super().__init__()

    def input_coords(self) -> xr.DataArray:
        """Input coordinate system of the prognostic model

        Returns
        -------
        xr.DataArray
            Allocation-free coordinate signature
        """
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": np.array(["t2m"])},
            dynamic=("batch",),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : xr.DataArray
            Input coordinate system to transform into output_coords

        Returns
        -------
        xr.DataArray
            Allocation-free coordinate signature
        """
        # Check input coordinates are valid
        handshake_dataarray(input_coords, self.input_coords())
        return coord_array_like(input_coords, {"variable": np.array(["t2m_c"])})

    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Runs diagnostic model

        Parameters
        ----------
        x : xr.DataArray
            Input field carrying its coordinates and grid metadata
        """
        out_coords = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        return from_torch(tensor - 273.15, out_coords)


# %%
# Input/Output Coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Defining the input/output coordinate systems is essential for any model in
# Earth2Studio since this is how both the package and users can learn what type of data
# the model expects. This requires the definition of  `input_coords` and
# `output_coords`. Have a look at [Coordinate Systems](../../userguide/about/overview.md#coordinates_userguide) for details on
# coordinate system.
#
# For this diagnostic model, we simply define the input coordinates
# to be the global surface temperature specified in [`earth2studio/lexicon/base.py`](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/lexicon/base.py).
# The output is a custom variable `t2m_c` that represents the temperature in
# Celsius.

# %%
# `__call__` API
# ~~~~~~~~~~~~~~~~~~~~~~~
# The call function is the main API of diagnostic models that have a tensor and
# coordinate system as input/output. This function first validates that the coordinate
# system is correct. Then both the input data tensor and also coordinate system are
# updated and returned.
#
# !!! note
#     You may notice the `batch_func` decorator, which is used to make batched
#     operations easier. For more details about this refer to the [Batch Dimension](../../userguide/advanced/batch.md#batch_function_userguide)
#     section of the user guide.

# %%
# Set Up
# ------
# With the custom diagnostic model defined, the next step is to set up and run a
# workflow. We will use the built in workflow [`earth2studio.run.diagnostic`][earth2studio.run.diagnostic].

# %%
# Lets instantiate the components needed.
#
# - Prognostic Model: Use the built in DLWP model [`earth2studio.models.px.DLWP`][earth2studio.models.px.DLWP].
# - Diagnostic Model: The custom diagnostic model defined above
# - Datasource: Pull data from the GFS data api [`earth2studio.data.GFS`][earth2studio.data.GFS].
# - IO Backend: Save the outputs into a Zarr store [`earth2studio.io.ZarrBackend`][earth2studio.io.ZarrBackend].

# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP

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
# Running our workflow with a build in prognostic model and a custom diagnostic is the
# same as running a built in diagnostic.

# %%
import earth2studio.run as run

nsteps = 20
io = run.diagnostic(["2024-01-01"], nsteps, model, diagnostic, data, io)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# Let's plot the Celsius temperature field from our custom diagnostic model.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

forecast = "2024-01-01"
variable = "t2m_c"

plt.close("all")

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(
    1,
    5,
    figsize=(12, 4),
    subplot_kw={"projection": ccrs.Orthographic()},
    constrained_layout=True,
)

times = (
    io["lead_time"][:].astype("timedelta64[ns]").astype("timedelta64[h]").astype(int)
)
step = 4  # 24hrs
for i, t in enumerate(range(0, 20, step)):

    ctr = ax[i].contourf(
        io["lon"][:],
        io["lat"][:],
        io[variable][0, t],
        vmin=-10,
        vmax=30,
        transform=ccrs.PlateCarree(),
        levels=20,
        cmap="coolwarm",
    )
    ax[i].set_title(f"{times[t]}hrs")
    ax[i].coastlines()
    ax[i].gridlines()

plt.suptitle(f"{variable} - {forecast}")

cbar = plt.cm.ScalarMappable(cmap="coolwarm")
cbar.set_array(io[variable][0, 0])
cbar.set_clim(-10.0, 30)
cbar = fig.colorbar(cbar, ax=ax[-1], orientation="vertical", label="C", shrink=0.8)


plt.savefig("outputs/02_custom_diagnostic_dlwp_prediction.jpg")
