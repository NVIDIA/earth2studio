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
Tropical Cyclone Tracking
=========================

Tropical cyclone tracking with tracker diagnostic models.

This example will demonstrate how to use the tropical cyclone (TC) tracker diagnostic
models for creating TC paths.
The diagnostics used here can be combined with other AI weather models and ensemble
methods to create complex inference workflow that enable downstream analysis.


In this example you will learn:

- How to instantiate a TC tracker diagnostic
- How to apply the TC tracker to data
- How to couple the TC tracker to a prognostic model
- Post-processing results
"""

# %%
# Set Up
# ------
# This example will look at tracking cyclones during August 2009, a moment in time when
# multiple tropical cyclones where impacting East Asia.
# Earth2Studio provides multiple variations of TC trackers such as :py:class:`earth2studio.models.dx.TCTrackerVitart`
# and :py:class:`earth2studio.models.dx.TCTrackerWuDuan`.
# The difference being the underlying algorithm used to identify the center.
#
# This example needs the following:
#
# - Diagostic Model: Use the TC tracker :py:class:`earth2studio.models.dx.TCTrackerWuDuan`.
# - Datasource: Pull data from the WB2 ERA5 data api :py:class:`earth2studio.data.WB2ERA5`.
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from datetime import datetime, timedelta

import torch

from earth2studio.data import ARCO
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import SFNO

# Create tropical cyclone tracker
tracker = TCTrackerWuDuan()

# Load the default model package which downloads the check point from NGC
package = SFNO.load_default_package()
prognostic = SFNO.load_model(package)

# Create the data source
data = ARCO()

nsteps = 16  # Number of steps to run the tracker for into future
start_time = datetime(2009, 8, 5)  # Start date for inference

# %%
# Tracking Analysis Data
# ----------------------
# Before coupling the TC tracker with a prognostic model, we will first apply it to
# analysis data.
# We can fetch a small time range from the data source and provide it to our model.
#
# For the forecast we will predict for two days (these will get executed as a batch) for
# 20 forecast steps which is 5 days.

# %%

from earth2studio.data import fetch_data, prep_data_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tracker = tracker.to(device)

# Land fall occured August 25th 2017
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps + 1)]
for step, time in enumerate(times):
    da = data(time, tracker.input_coords()["variable"])
    x, coords = prep_data_array(da, device=device)
    output, output_coords = tracker(x, coords)
    print(f"Step {step}: ARCO tracks output shape {output.shape}")

era5_tracks = output.cpu()
torch.save(era5_tracks, "era5.pt")

# %%
# Notice that the output tensor grows as iterations are performed.
# This is because the tracker builds tracks based on previous forward passes returning
# a tensor with the dimensions [batch, path, step, variable].
# Not all paths are garenteed to be the same length or have the same start / stop time
# so any missing data is populated with a nan value.
#
# Up next lets also repeat the same process using the prognostic AI model.
# One could use one of the build in workflows but here we will manually implement the
# inference loop.

# %%

from tqdm import tqdm

from earth2studio.utils.coords import map_coords

prognostic = prognostic.to(device)
# Reset the internal path buffer of tracker
tracker.reset_path_buffer()

# Load the initial state
x, coords = fetch_data(
    source=data,
    time=[start_time],
    variable=prognostic.input_coords()["variable"],
    lead_time=prognostic.input_coords()["lead_time"],
    device=device,
)

# Create prognostic iterator
model = prognostic.create_iterator(x, coords)
with tqdm(total=nsteps + 1, desc="Running inference") as pbar:
    for step, (x, coords) in enumerate(model):
        # Run tracker
        x, coords = map_coords(x, coords, tracker.input_coords())
        output, output_coords = tracker(x, coords)
        # lets remove the lead time dim
        output = output[:, 0]
        print(f"Step {step}: SFNO tracks output shape {output.shape}")

        pbar.update(1)
        if step == nsteps:
            break

sfno_tracks = output.cpu()
torch.save(sfno_tracks, "sfno.pt")

# %%
# Note that before the inference loop of the AI model, the path buffer of the tracker
# was reset which refreshes the tracker's path state starting from scratch.
# Otherwise, it would attempt to append to the existing tracks from the data source
# loop in the previous section.
#
# Finally we can plot the results to compare the track ground truths from ERA5 with
# those produced by SFNO.
# Recall the outputs of :py:class:`earth2studio.models.dx.TCTrackerWuDuan` has the path
# ID in the second dimension, thus that is what will determine the number of lines.
# The lat/lon coords are the first two variables in the last dimension.
# Lastly we just need to be mindful of the NaN filler values which can get easily
# masked out and any "path" that isnt over 2 points long

# %%

from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Convert tracks from tensors to numpy arrays
era5_paths = era5_tracks.numpy()
sfno_paths = sfno_tracks.numpy()

# Calculate end date
end_time = start_time + timedelta(hours=6 * nsteps)

# Create figure with cartopy projection
plt.figure(figsize=(10, 8))
projection = ccrs.LambertConformal(
    central_longitude=130.0, central_latitude=30.0, standard_parallels=(20.0, 40.0)
)
ax = plt.axes(projection=projection)

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, alpha=0.1)
ax.gridlines(draw_labels=True, alpha=0.6)
ax.set_extent([90, 170, 0, 50], crs=ccrs.PlateCarree())

era5_cmap = plt.cm.autumn
sfno_cmap = plt.cm.winter

for path in range(era5_paths.shape[1]):
    # Get lat/lon coordinates, filtering out nans
    lats = era5_paths[0, path, :, 0]
    lons = era5_paths[0, path, :, 1]
    mask = ~np.isnan(lats) & ~np.isnan(lons)
    if mask.any() and len(lons[mask]) > 2:
        color = era5_cmap(path / era5_paths.shape[1])
        ax.plot(
            lons[mask],
            lats[mask],
            color=color,
            linestyle="-.",
            marker="x",
            label="ERA5" if path == 0 else "",
            transform=ccrs.PlateCarree(),
        )

for path in range(sfno_paths.shape[1]):
    # Get lat/lon coordinates, filtering out nans
    lats = sfno_paths[0, path, :, 0]
    lons = sfno_paths[0, path, :, 1]
    mask = ~np.isnan(lats) & ~np.isnan(lons)
    if mask.any() and len(lons[mask]) > 2:
        color = sfno_cmap(path / sfno_paths.shape[1])
        ax.plot(
            lons[mask],
            lats[mask],
            color=color,
            linestyle="-",
            label="SFNO" if path == 0 else "",
            transform=ccrs.PlateCarree(),
        )

era5_patch = mpatches.Rectangle(
    (0, 0), 1, 1, fc=era5_cmap(0.3), alpha=0.9, label="ERA5"
)
sfno_patch = mpatches.Rectangle(
    (0, 0), 1, 1, fc=sfno_cmap(0.3), alpha=0.9, label="SFNO"
)
ax.legend(handles=[era5_patch, sfno_patch], loc="upper right", title="Cyclone Tracks")

plt.title(
    f'Tropical Cyclone Tracks\n{start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")}'
)
plt.savefig(f"outputs/13_{start_time}_cyclone_tracks.jpg", bbox_inches="tight", dpi=300)

# %%
# In addition to filtering out the NaN values, users may want to apply other post
# processing steps on the paths which may be enforcing path lengths are above a certain
# threshold or other geography based filters.
#
# No cyclone tracker is perfect, we encourage users to experiment and tune the tracker
# as needed.
