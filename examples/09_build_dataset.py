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
Building a Dataset From Data Sources
====================================
"""

import zarr
import datetime
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from apache_beam.options.pipeline_options import PipelineOptions


from earth2studio.data import ARCO, build_dataset
import earth2grid

# Create zarr cache
zarr_cache = zarr.open("my_zarr_cache.zarr", mode="a")

# Create the data source
data = ARCO(zarr_cache=zarr_cache)

# Make healpix transformation
level = 8
nside = 2**level
hpx = earth2grid.healpix.Grid(level=level, pixel_order=earth2grid.healpix.XY())
src = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)
regrid = earth2grid.get_regridder(src, hpx)
def _transform(regrid, nside): # Functional Closure
    import torch
    def transform(time, variable, x):
        x_torch = torch.as_tensor(x)
        x_torch = x_torch.double()
        hlp_x = regrid(x_torch).reshape(12, nside, nside)
        return hlp_x.numpy()
    return transform
transform = _transform(regrid, nside)

# Apache Beam options
options = PipelineOptions([ 
    '--runner=DirectRunner',   
    '--direct_num_workers=8',
    '--direct_running_mode=multi_processing',
])  

# Make Store Dataset
time = pd.date_range("2000-01-01", freq="6h", periods=1000)
zarr_dataset = zarr.open("my_dataset.zarr", mode="r")

# Build the predicted variables
variable = ["u10m", "v10m"]
dataset = zarr_dataset.create_dataset("predicted_variables", shape=(len(time), len(variable), 12, nside, nside), chunks=(1, len(variable), 12, nside, nside), dtype="f4")
build_dataset(data, time, variable, dataset, apache_beam_options=options, transform=transform)

# Build the un-predicted variables
variable = ["land_sea_mask"]
dataset = zarr_dataset.create_dataset("unpredicted_variables", shape=(len(time), len(variable), 12, nside, nside), chunks=(1, len(variable), 12, nside, nside), dtype="f4")
build_dataset(data, time, variable, dataset, apache_beam_options=options, transform=transform)

# Make animation of the dataset
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    ax.scatter(hpx.lon, hpx.lat, c=zarr_dataset["predicted_variables"][i, 0].flatten(), s=0.1)
    ax.set_title(time[i])
ani = animation.FuncAnimation(fig, animate, frames=len(time), interval=100)
plt.show()
