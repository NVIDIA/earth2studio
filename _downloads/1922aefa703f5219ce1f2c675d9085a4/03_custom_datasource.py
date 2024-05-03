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
Extending Data Sources
======================

Implementing a custom data source

This example will demonstrate how to extend Earth2Studio by implementing a custom data
source to use in a built in workflow.

In this example you will learn:

- API requirements of data soruces
- Implementing a custom data soruce
"""

# %%
# Custom Data Source
# ------------------
# Earth2Studio defines the required APIs for data sources in
# :py:class:`earth2studio.data.base.DataSource` which requires just a call function.
# For this example, we will consider extending an existing remote data source with
# another atmospheric field we can calculate.
#
# The :py:class:`earth2studio.data.ARCO` data source provides the ERA5 dataset in a cloud
# optimized format, however it only provides specific humidity. This is a problem for
# models that may use relative humidity as an input. Based on ECMWF documentation we can
# calculate the relative humidity based on temperature and geo-potential.

# %%
from datetime import datetime

import numpy as np
import xarray as xr

from earth2studio.data import ARCO, GFS
from earth2studio.data.utils import prep_data_inputs
from earth2studio.utils.type import TimeArray, VariableArray


class CustomDataSource:
    """Custom ARCO datasource"""

    relative_humidity_ids = [
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]

    def __init__(self, cache: bool = True, verbose: bool = True):
        self.arco = ARCO(cache, verbose)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in IFS lexicon.

        Returns
        -------
        xr.DataArray
        """
        time, variable = prep_data_inputs(time, variable)

        # Replace relative humidity with respective temperature
        # and specifc humidity fields
        variable_expanded = []
        for v in variable:
            if v in self.relative_humidity_ids:
                level = int(v[1:])
                variable_expanded.extend([f"t{level}", f"q{level}"])
            else:
                variable_expanded.append(v)
        variable_expanded = list(set(variable_expanded))

        # Fetch from ARCO
        da_exp = self.arco(time, variable_expanded)

        # Calculate relative humidity when needed
        arrays = []
        for v in variable:
            if v in self.relative_humidity_ids:
                level = int(v[1:])
                t = da_exp.sel(variable=f"t{level}").values
                q = da_exp.sel(variable=f"q{level}").values
                rh = self.calc_relative_humdity(t, q, 100 * level)
                arrays.append(rh)
            else:
                arrays.append(da_exp.sel(variable=v).values)

        da = xr.DataArray(
            data=np.stack(arrays, axis=1),
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=da_exp.coords["time"].values,
                variable=np.array(variable),
                lat=da_exp.coords["lat"].values,
                lon=da_exp.coords["lon"].values,
            ),
        )
        return da

    def calc_relative_humdity(
        self, temperature: np.array, specific_humidity: np.array, pressure: float
    ) -> np.array:
        """Relative humidity calculation

        Parameters
        ----------
        temperature : np.array
            Temperature field (K)
        specific_humidity : np.array
            Specific humidity field (g.kg-1)
        pressure : float
            Pressure (P)

        Returns
        -------
        np.array
        """
        epsilon = 0.621981
        p = pressure
        q = specific_humidity
        t = temperature

        e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

        es_w = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
        es_i = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))

        alpha = np.clip((t - 250.16) / (273.16 - 250.16), 0, 1.2) ** 2
        es = alpha * es_w + (1 - alpha) * es_i
        rh = 100 * e / es

        return rh


# %%
# :py:func:`__call__` API
# ~~~~~~~~~~~~~~~~~~~~~~~
# The call function is the main API of data source which return the Xarray data array
# with the requested data. For this custom data source we intercept relative humidity
# variables, replace them with temperature and specific humidity requests then calculate
# the relative humidity from these fields. Note that the ARCO data source is handling
# the remote complexity, we are just manipulating Numpy arrays

# %%
# :py:func:`calc_relative_humdity`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Based on the calculations ECMWF uses in their IFS numerical simulator which accounts
# for estimating the water vapor and ice present in the atmosphere.
#
# .. note::
#   See reference, equation 7.98 onwards:
#   https://www.ecmwf.int/en/elibrary/81370-ifs-documentation-cy48r1-part-iv-physical-processes

# %%
# Verification
# ------------
# Before plugging this into our workflow, let's quickly verify our data source is
# consistent with when GFS provides for relative humidity.

# %%
ds = CustomDataSource()
da_custom = ds(time=datetime(2022, 1, 1, hour=0), variable=["r500"])

ds_gfs = GFS()
da_gfs = ds_gfs(time=datetime(2022, 1, 1, hour=0), variable=["r500"])

print(da_custom)

# %%
import os

os.makedirs("outputs", exist_ok=True)
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 3),
    subplot_kw={"projection": ccrs.Mollweide()},
    constrained_layout=True,
)

ax[0].imshow(
    da_custom.sel(variable="r500")[0], transform=ccrs.PlateCarree(), vmin=0, vmax=100
)
ax[1].imshow(
    da_gfs.sel(variable="r500")[0], transform=ccrs.PlateCarree(), vmin=0, vmax=100
)

ax[0].set_title("Custom ARCO")
ax[1].set_title("GFS")
plt.suptitle("r500", fontsize=24)
cbar = plt.cm.ScalarMappable()
cbar.set_array(da_custom.sel(variable="r500")[0])
cbar.set_clim(0, 100)
cbar = fig.colorbar(cbar, ax=ax[-1], orientation="vertical", shrink=0.8)

plt.savefig("outputs/custom_datasource_gfs_versus_custom.jpg")


# %%
# Execute Workflow
# ----------------
# We will use this custom data source to run deterministic inference with a model that
# requires relative humidity. :mod:`earth2studio.models.px.FCN` is one such model. Since
# we are using ARCO, we can run inference for a time quite far back in time.
#
# Let's instantiate the components needed.
#
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.
# - Datasource: Custom data source above
# - IO Backend: Save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import earth2studio.run as run
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN

package = FCN.load_default_package()
model = FCN.load_model(package)

# Create the data source
data = CustomDataSource()

# Create the IO handler, store in memory
io = ZarrBackend()

nsteps = 4
io = run.deterministic(["1993-04-05"], nsteps, model, data, io)

print(io.root.tree())

# %%
# Post Processing
# ---------------
# To confirm that our model is working as expected, we will plot the total column water
# vapor field for a few time-steps.

# %%
forecast = "1993-04-05"
variable = "tcwv"

plt.close("all")

# Create a figure and axes with the specified projection
fig, ax = plt.subplots(2, 2, figsize=(6, 4))

# Plot tcwv every 6 hours
ax[0, 0].imshow(io[variable][0, 0], vmin=0, vmax=80, cmap="magma")
ax[0, 1].imshow(io[variable][0, 1], vmin=0, vmax=80, cmap="magma")
ax[1, 0].imshow(io[variable][0, 2], vmin=0, vmax=80, cmap="magma")
ax[1, 1].imshow(io[variable][0, 3], vmin=0, vmax=80, cmap="magma")

# Set title
plt.suptitle(f"{variable} - {forecast}")
times = io["lead_time"].astype("timedelta64[h]").astype(int)
ax[0, 0].set_title(f"Lead time: {times[0]}hrs")
ax[0, 1].set_title(f"Lead time: {times[1]}hrs")
ax[1, 0].set_title(f"Lead time: {times[2]}hrs")
ax[1, 1].set_title(f"Lead time: {times[3]}hrs")

plt.savefig("outputs/custom_datasource_prediction.jpg", bbox_inches="tight")
