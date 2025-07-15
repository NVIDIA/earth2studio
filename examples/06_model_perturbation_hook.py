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
Model Hook Injection: Perturbation
==================================

Adding model noise by using custom hooks.

This example will demonstrate how to run an ensemble inference workflow to generate a
perturbed ensemble forecast. This perturbation is done by injecting code into the model
front and rear hooks.
These hooks are applied to the tensor data before/after the model forward call.

This example also illustrates how you can subselect data for IO. In this example we
will only output two variables:
total column water vapor (tcwv) and 500 hPa geopotential (z500).
To run this, make sure that the model selected predicts these variables are change
appropriately.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Changing the model forward/rear hooks
- Choose a subselection of coordinates to save to an IO object.
- Post-processing results
"""
# /// script
# dependencies = [
#   "earth2studio[dlwp] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Creating an Ensemble Workflow
# -----------------------------
#
# To start let's begin with creating an ensemble workflow to use. We encourage
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
# - data: Initialized data source to fetch initial conditions from
# - io: io store that data is written to.
# - output_coords: CoordSystem of output coordinates that should be saved. Should be
#   a proper subset of model output coordinates.

# %%
# Set Up
# ------
# With the ensemble workflow defined, we now need to create the individual components.
#
# We need the following:
#
# - Prognostic Model: Use the built in DLWP model :py:class:`earth2studio.models.px.DLWP`.
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.
#
# We will first run the ensemble workflow using an unmodified function, that is a model that has the
# default (identity) forward and rear hooks. Then we will define new hooks for the model and rerun the
# inference request.
# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP
from earth2studio.perturbation import Gaussian
from earth2studio.run import ensemble

# Load the default model package which downloads the check point from NGC
package = DLWP.load_default_package()
model = DLWP.load_model(package)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
chunks = {"ensemble": 1, "time": 1, "lead_time": 1}
io_unperturbed = ZarrBackend(file_name="outputs/05_ensemble.zarr", chunks=chunks)


# %%
# Execute the Workflow
# --------------------
# First, we will run the ensemble workflow but with a :py:meth:`earth2studio.perturbation.Gaussian`
# perturbation as the control.
#
# The workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.

# %%
nsteps = 4 * 12
nensemble = 16
batch_size = 4
forecast_date = "2024-01-30"
output_coords = {
    "lat": np.arange(25.0, 60.0, 0.25),
    "lon": np.arange(230.0, 300.0, 0.25),
    "variable": np.array(["tcwv", "z500"]),
}

# First run with no model perturbation
io_unperturbed = ensemble(
    [forecast_date],
    nsteps,
    nensemble,
    model,
    data,
    io_unperturbed,
    Gaussian(noise_amplitude=0.01),
    output_coords=output_coords,
    batch_size=batch_size,
)

# %%
# Now let's introduce slight model perturbation using the prognostic model hooks defined
# in :py:class:`earth2studio.models.px.utils.PrognosticMixin`.
# Note that :py:obj:`center.unsqueeze(-1)` is DLWP specific since it operates on a cubed sphere
# with grid dimensions (nface, lat, lon) instead of just (lat,lon).
# To switch out the model, consider removing the :py:meth:`unsqueeze` .

# %%
model.front_hook = lambda x, coords: (
    x
    - 0.1
    * x.var(dim=0)
    * (x - model.center.unsqueeze(-1))
    / (model.scale.unsqueeze(-1)) ** 2
    + 0.1 * (x - x.mean(dim=0)),
    coords,
)
# Also could use model.rear_hook = ...

io_perturbed = ZarrBackend(
    file_name="outputs/05_ensemble_model_perturbation.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)
io_perturbed = ensemble(
    [forecast_date],
    nsteps,
    nensemble,
    model,
    data,
    io_perturbed,
    Gaussian(noise_amplitude=0.01),
    output_coords=output_coords,
    batch_size=batch_size,
)

# %%
# Post Processing
# ---------------
# The last step is to post process our results.
# Here we plot and compare the ensemble mean and standard deviation from using an
# unperturbed/perturbed model.
#
# Notice that the Zarr IO function has additional APIs to interact with the stored data.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

levels_unperturbed = np.linspace(0, io_unperturbed["tcwv"][:].max())
levels_perturbed = np.linspace(0, io_perturbed["tcwv"][:].max())


std_levels_perturbed = np.linspace(0, io_perturbed["tcwv"][:].std(axis=0).max())

plt.close("all")
fig = plt.figure(figsize=(20, 10), tight_layout=True)
ax0 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())


def update(frame):
    """This function updates the frame with a new lead time for animation."""
    import warnings

    warnings.filterwarnings("ignore")
    ax0.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ## Update unperturbed image
    im0 = ax0.contourf(
        io_unperturbed["lon"][:],
        io_unperturbed["lat"][:],
        io_unperturbed["tcwv"][:, 0, frame].mean(axis=0),
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        levels=levels_unperturbed,
    )
    ax0.coastlines()
    ax0.gridlines()

    im1 = ax1.contourf(
        io_unperturbed["lon"][:],
        io_unperturbed["lat"][:],
        io_unperturbed["tcwv"][:, 0, frame].std(axis=0),
        transform=ccrs.PlateCarree(),
        cmap="RdPu",
        levels=std_levels_perturbed,
        norm=LogNorm(vmin=1e-1, vmax=std_levels_perturbed[-1]),
    )
    ax1.coastlines()
    ax1.gridlines()

    im2 = ax2.contourf(
        io_perturbed["lon"][:],
        io_perturbed["lat"][:],
        io_perturbed["tcwv"][:, 0, frame].mean(axis=0),
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        levels=levels_perturbed,
    )
    ax2.coastlines()
    ax2.gridlines()

    im3 = ax3.contourf(
        io_perturbed["lon"][:],
        io_perturbed["lat"][:],
        io_perturbed["tcwv"][:, 0, frame].std(axis=0),
        transform=ccrs.PlateCarree(),
        cmap="RdPu",
        levels=std_levels_perturbed,
        norm=LogNorm(vmin=1e-1, vmax=std_levels_perturbed[-1]),
    )
    ax3.coastlines()
    ax3.gridlines()

    for i in range(16):
        ax0.contour(
            io_unperturbed["lon"][:],
            io_unperturbed["lat"][:],
            io_unperturbed["z500"][i, 0, frame] / 100.0,
            transform=ccrs.PlateCarree(),
            levels=np.arange(485, 580, 15),
            colors="black",
            linestyle="dashed",
        )

        ax2.contour(
            io_perturbed["lon"][:],
            io_perturbed["lat"][:],
            io_perturbed["z500"][i, 0, frame] / 100.0,
            transform=ccrs.PlateCarree(),
            levels=np.arange(485, 580, 15),
            colors="black",
            linestyle="dashed",
        )
    plt.suptitle(
        f'Forecast Starting on {forecast_date} - Lead Time - {io_perturbed["lead_time"][frame]}'
    )

    ax0.set_title("Unperturbed Ensemble Mean - tcwv + z500 countors")
    ax1.set_title("Unperturbed Ensemble Std - tcwv")
    ax2.set_title("Perturbed Ensemble Mean - tcwv + z500 contours")
    ax3.set_title("Perturbed Ensemble Std - tcwv")

    if frame == 0:
        plt.colorbar(
            im0, ax=ax0, shrink=0.75, pad=0.04, label="kg m^-2", format="%2.1f"
        )
        plt.colorbar(
            im1, ax=ax1, shrink=0.75, pad=0.04, label="kg m^-2", format="%1.2e"
        )
        plt.colorbar(
            im2, ax=ax2, shrink=0.75, pad=0.04, label="kg m^-2", format="%2.1f"
        )
        plt.colorbar(
            im3, ax=ax3, shrink=0.75, pad=0.04, label="kg m^-2", format="%1.2e"
        )


# Uncomment this for animation
# import matplotlib.animation as animation
# update(0)
# ani = animation.FuncAnimation(
# fig=fig, func=update, frames=range(1, nsteps), cache_frame_data=False
# )
# ani.save(f"outputs/05_model_perturbation_{forecast_date}.gif", dpi=300)


for lt in [10, 20, 30, 40]:
    update(lt)
    plt.savefig(
        f"outputs/05_model_perturbation_{forecast_date}_leadtime_{lt}.png",
        dpi=300,
        bbox_inches="tight",
    )
