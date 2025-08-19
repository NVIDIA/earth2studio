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
# As discussed in the :ref:`diagnostic_model_userguide` section of the user guide,
# Earth2Studio defines a diagnostic model through a simple interface
# :py:class:`earth2studio.models.dx.base.Diagnostic Model`. This can be used to help
# guide the required APIs needed to successfully create our own model.
#
# In this example, lets consider a simple diagnostic that converts the surface
# temperature in Kelvin to Celsius to make it more readable for the average person.
#
# Our diagnostic model has a base class of :py:class:`torch.nn.Module` which allows us
# to get the required :py:obj:`to(device)` method for free.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class CustomDiagnostic(torch.nn.Module):
    """Custom dianostic model"""

    def __init__(self):
        super().__init__()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["t2m"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # Check input coordinates are valid
        target_input_coords = self.input_coords()
        for i, (key, value) in enumerate(target_input_coords.items()):
            if key != "batch":
                handshake_dim(input_coords, key, i)
                handshake_coords(input_coords, target_input_coords, key)

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["t2m_c"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        output_coords["batch"] = input_coords["batch"]
        return output_coords

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
        out_coords = self.output_coords(coords)
        out = x - 273.15  # To celcius
        return out, out_coords


# %%
# Input/Output Coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Defining the input/output coordinate systems is essential for any model in
# Earth2Studio since this is how both the package and users can learn what type of data
# the model expects. This requires the definition of  :py:func:`input_coords` and
# :py:func:`output_coords`. Have a look at :ref:`coordinates_userguide` for details on
# coordinate system.
#
# For this diagnostic model, we simply define the input coordinates
# to be the global surface temperature specified in :file:`earth2studio.lexicon.base.py`.
# The output is a custom variable :code:`t2m_c` that represents the temperature in
# Celsius.

# %%
# :py:func:`__call__` API
# ~~~~~~~~~~~~~~~~~~~~~~~
# The call function is the main API of diagnostic models that have a tensor and
# coordinate system as input/output. This function first validates that the coordinate
# system is correct. Then both the input data tensor and also coordinate system are
# updated and returned.
#
# .. note::
#   You may notice the :py:func:`batch_func` decorator, which is used to make batched
#   operations easier. For more details about this refer to the :ref:`batch_function_userguide`
#   section of the user guide.

# %%
# Set Up
# ------
# With the custom diagnostic model defined, the next step is to set up and run a
# workflow. We will use the built in workflow :py:meth:`earth2studio.run.diagnostic`.

# %%
# Lets instantiate the components needed.
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - Diagnostic Model: The custom diagnostic model defined above
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

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
