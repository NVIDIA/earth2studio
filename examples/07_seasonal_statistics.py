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
Statistical Inference
=====================

Simple seasonal statistic inference workflow.

This example will demonstrate how to run a simple inference workflow to generate a
forecast and then to save a statistic of that data. There are a handful of built-in
statistics available in `earth2studio.statistics`, but here we will demonstrate how
to define a custom statistic and run inference.

In this example you will learn:

- How to instantiate a built in prognostic model
- Creating a data source and IO object
- Create a custom statistic
- Running a simple built in workflow
- Post-processing results
"""
# /// script
# dependencies = [
#   "earth2studio[pangu,statistics] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "matplotlib",
# ]
# ///

# %%
# Creating a Statistical Workflow
# -------------------------------
#
# Start with creating a simple inference workflow to use. We encourage
# users to explore and experiment with their own custom workflows that borrow ideas from
# built in workflows inside :py:obj:`earth2studio.run` or the examples.
#
# Creating our own generalizable workflow to use with statistics is easy when we rely on
# the component interfaces defined in Earth2Studio (use dependency injection). Here we
# create a run method that accepts the following:
#
# - time: Input list of datetimes / strings to run inference for
# - nsteps: Number of forecast steps to predict
# - prognostic: Our initialized prognostic model
# - statistic: our custom statistic
# - data: Initialized data source to fetch initial conditions from
# - io: IOBackend
#
# We do not run an ensemble inference workflow here, even though it is common for statistical
# inference. See ensemble examples for details on how to extend this example for that purpose.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.statistics import Statistic
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def run_stats(
    time: list[str] | list[datetime] | list[np.datetime64],
    nsteps: int,
    nensemble: int,
    prognostic: PrognosticModel,
    statistic: Statistic,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:
    """Simple statistics workflow

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
    statistic : Statistic
        Custom statistic to compute and write to IO.
    data : DataSource
        Data source
    io : IOBackend
        IO object

    Returns
    -------
    IOBackend
        Output IO object
    """
    logger.info("Running simple statistics workflow!")
    # Load model onto the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")
    prognostic = prognostic.to(device)
    # Fetch data from data source and load onto device
    time = to_time_array(time)
    x, coords = fetch_data(
        source=data,
        time=time,
        lead_time=prognostic.input_coords()["lead_time"],
        variable=prognostic.input_coords()["variable"],
        device=device,
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    # Set up IO backend
    total_coords = coords.copy()
    output_coords = prognostic.output_coords(prognostic.input_coords())
    total_coords["lead_time"] = np.asarray(
        [output_coords["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()
    # Remove reduced dimensions from statistic
    for d in statistic.reduction_dimensions:
        total_coords.pop(d, None)

    io.add_array(total_coords, str(statistic))

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, prognostic.input_coords())

    # Create prognostic iterator
    model = prognostic.create_iterator(x, coords)

    logger.info("Inference starting!")
    with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
        for step, (x, coords) in enumerate(model):
            s, coords = statistic(x, coords)
            io.write(s, coords, str(statistic))
            pbar.update(1)
            if step == nsteps:
                break

    logger.success("Inference complete")
    return io


# %%
# Set Up
# ------
# With the statistical workflow defined, we now need to create the individual components.
#
# We need the following:
#
# - Prognostic Model: Use the built in Pangu 24 hour model :py:class:`earth2studio.models.px.Pangu24`.
# - statistic: We define our own statistic: the Southern Oscillation Index (SOI).
# - Datasource: Pull data from the GFS data api :py:class:`earth2studio.data.GFS`.
# - IO Backend: Save the outputs into a NetCDF4 store :py:class:`earth2studio.io.NetCDF4Backend`.

# %%
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import GFS
from earth2studio.io import NetCDF4Backend
from earth2studio.models.px import Pangu24
from earth2studio.utils.type import CoordSystem

# Load the default model package which downloads the check point from NGC
package = Pangu24.load_default_package()
model = Pangu24.load_model(package)

# Create the data source
data = GFS()

# Create the IO handler, store in memory
io = NetCDF4Backend(
    file_name="outputs/soi.nc",
    backend_kwargs={"mode": "w"},
)


# Create the custom statistic
class SOI:
    """Custom metric calculation the Southern Oscillation Index.

    SOI = ( standardized_tahiti_slp - standardized_darwin_slp ) / soi_normalization

    soi_normalization = std( historical ( standardized_tahiti_slp - standardized_darwin_slp ) )

    standardized_*_slp = (*_slp - climatological_mean_*_slp) / climatological_std_*_slp

    Note
    ----
    __str__
        Name that will be applied to the output of this statistic, primarily for IO purposes.
    reduction_dimensions
        Dimensions that this statistic reduces over. This is used to help automatically determine
        the output coordinates, primarily used for IO purposes.
    """

    def __str__(self) -> str:
        return "soi"

    def __init__(
        self,
    ):
        # Read in Tahiti and Darwin SLP data
        from physicsnemo.utils.filesystem import _download_cached

        file_path = _download_cached(
            "https://data.longpaddock.qld.gov.au/SeasonalClimateOutlook/SouthernOscillationIndex/SOIDataFiles/DailySOI1933-1992Base.txt"
        )
        ds = pd.read_csv(file_path, sep=r"\s+")
        dates = pd.date_range("1999-01-01", freq="d", periods=len(ds))
        ds["date"] = dates
        ds = ds.set_index("date")
        ds = ds.drop(["Year", "Day", "SOI"], axis=1)
        ds = ds.rolling(30, min_periods=1).mean().dropna()

        self.climatological_means = torch.tensor(
            ds.groupby(ds.index.month).mean().to_numpy(), dtype=torch.float32
        )
        self.climatological_std = torch.tensor(
            ds.groupby(ds.index.month).std().to_numpy(), dtype=torch.float32
        )

        standardized = ds.groupby(ds.index.month).transform(
            lambda x: (x - x.mean()) / x.std()
        )
        diff = standardized["Tahiti"] - standardized["Darwin"]

        self.normalization = torch.tensor(
            diff.groupby(ds.index.month).std().to_numpy(), dtype=torch.float32
        )

        self.tahiti_coords = {
            "variable": np.array(["msl"]),
            "lat": np.array([-17.65]),
            "lon": np.array([210.57]),
        }
        self.darwin_coords = {
            "variable": np.array(["msl"]),
            "lat": np.array([-12.46]),
            "lon": np.array([130.84]),
        }

        self.reduction_dimensions = list(self.tahiti_coords)

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Computes the SOI given an input.

        coords must be a superset of both

        tahiti_coords = {
            'variable': np.array(['msl']),
            'lat': np.array([-17.65]),
            'lon': np.array([210.57])
        }

        and

        darwin_coords = {
            'variable': np.array(['msl']),
            'lat': np.array([-12.46]),
            'lon': np.array([130.84])
        }

        So make sure that the model chosen predicts the `msl` variable.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            coordinate system belonging to the input tensor.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns the SOI and appropriate coordinate system.
        """
        tahiti, _ = map_coords(x, coords, self.tahiti_coords)
        darwin, _ = map_coords(x, coords, self.darwin_coords)

        tahiti = tahiti.squeeze(-3, -2, -1) / 100.0
        darwin = darwin.squeeze(-3, -2, -1) / 100.0
        output_coords = OrderedDict(
            {k: v for k, v in coords.items() if k not in self.reduction_dimensions}
        )

        # Get time coordinates
        times = coords["time"].reshape(-1, 1) + coords["lead_time"].reshape(1, -1)
        months = torch.broadcast_to(
            torch.as_tensor(
                [pd.Timestamp(t).month for t in times.flatten()],
                device=tahiti.device,
                dtype=torch.int32,
            ).reshape(times.shape),
            tahiti.shape,
        )

        cm = self.climatological_means.to(tahiti.device)
        cs = self.climatological_std.to(tahiti.device)
        norm = self.normalization.to(tahiti.device)

        tahiti_std_anomaly = (tahiti - cm[months, 0]) / cs[months, 0]
        darwin_std_anomaly = (tahiti - cm[months, 1]) / cs[months, 1]

        return (tahiti_std_anomaly - darwin_std_anomaly) / norm[months], output_coords


soi = SOI()

# %%
# Execute the Workflow
# --------------------
# With all components initialized, running the workflow is a single line of Python code.
# Workflow will return the provided IO object back to the user, which can be used to
# then post process. Some have additional APIs that can be handy for post-processing or
# saving to file. Check the API docs for more information.
# We simulate a trajectory of 60 time steps, or 2 months using Pangu24

# %%
nsteps = 60
nensemble = 1
io = run_stats(["2022-01-01"], nsteps, nensemble, model, soi, data, io)


# %%
# Post Processing
# ---------------
# The last step is to post process our results.
#
# Notice that the NetCDF IO function has additional APIs to interact with the stored data.

# %%
import matplotlib.pyplot as plt

times = io["time"][:].flatten() + io["lead_time"][:].flatten()

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(times, io["soi"][:].flatten())
ax.set_title("Southern Oscillation Index")
ax.grid("on")

plt.savefig("outputs/07_southern_oscillation_index_prediction_2022.png")
io.close()
