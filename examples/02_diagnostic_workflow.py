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
Running Diagnostic Inference
============================

Basic prognostic + diagnostic inference workflow.

This example will demonstrate how to run a deterministic inference workflow that couples
a prognostic model with a diagnostic model. This diagnostic model will predict a new
atmospheric quantity from the predicted fields of the prognostic.

In this example you will learn:

- How to instantiate a prognostic model
- How to instantiate a diagnostic model
- Creating a data source and IO object
- Running the built in diagnostic workflow
- Post-processing results
"""
# /// script
# dependencies = [
#   "earth2studio[dlwp] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# For this example, the built in diagnostic workflow :py:meth:`earth2studio.run.diagnostic`
# will be used.

# %%
# .. literalinclude:: ../../earth2studio/run.py
#    :language: python
#    :start-after: # sphinx - diagnostic start
#    :end-before: # sphinx - diagnostic end


# %%
# Thus, we need the following:
#
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.
# - Diagnostic Model: Use the built in precipitation AFNO model :py:class:`earth2studio.models.dx.PrecipitationAFNO`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import FCN

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
prognostic_model = FCN.load_model(package)

package = PrecipitationAFNO.load_default_package()
diagnostic_model = PrecipitationAFNO.load_model(package)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
io = ZarrBackend()

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.

# %%
import earth2studio.run as run

nsteps = 8
io = run.diagnostic(
    ["2021-06-01"], nsteps, prognostic_model, diagnostic_model, data, io
)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# The last step is to plot the resulting predicted total precipitation. The power of
# diagnostic models is that they allow the prediction of any variable from a pre-trained
# prognostic model.
#
# .. note::
#   The built in workflow will only save the direct outputs of the diagnostic. In this
#   example only total precipitation is accessible for plotting. If you wish to save
#   outputs of both the prognostic and diagnostic, we recommend writing a custom
#   workflow.

# %%
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

forecast = datetime(2021, 6, 1)
variable = "tp"
step = 8  # lead time = 48 hrs

plt.close("all")
# Create a Orthographic projection of USA
projection = ccrs.Orthographic(-100, 40)

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 6))

# Plot the field using pcolormesh
levels = np.arange(0.0, 0.01, 0.001)
im = ax.contourf(
    io["lon"][:],
    io["lat"][:],
    io[variable][0, step],
    levels,
    transform=ccrs.PlateCarree(),
    vmax=0.01,
    vmin=0.00,
    cmap="terrain",
)

# Set title
ax.set_title(f"{forecast.strftime('%Y-%m-%d')} - Lead time: {6*step}hrs")

# Add coastlines and gridlines6
ax.set_extent([220, 340, 20, 70])  # [lat min, lat max, lon min, lon max]
ax.coastlines()
ax.gridlines()
plt.colorbar(
    im, ax=ax, ticks=levels, shrink=0.75, pad=0.04, label="Total precipitation (m)"
)

plt.savefig("outputs/02_tp_prediction.jpg")
