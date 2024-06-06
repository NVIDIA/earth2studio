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

import datetime
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions

from earth2studio.data import ARCO, build_dataset

# Create the data source
data = ARCO()

# Create Zarr Cache
zarr_cache = data.initialize_zarr_cache(
    file_path="my_zarr_cache.zarr",
)

# Fetch Data from Zarr Cache, This will automatically download the data from the source if needed
data.fetch_cached_array(
    time=datetime.datetime(2020, 1, 1),
    variable="u10m",
    zarr_cache=zarr_cache,
) 

# Lets try to fetch a dataset with standard interface
ds = data(datetime.datetime(2022, 1, 1), variable=["u10m", "v10m"], zarr_cache=zarr_cache)

# Run again to show its cached, this will print some cached message
ds = data(datetime.datetime(2022, 1, 1), variable=["u10m", "v10m"], zarr_cache=zarr_cache)

# Generate dataset
options = PipelineOptions([ 
    '--runner=DirectRunner',   
    '--direct_num_workers=0',
    '--direct_running_mode=multi_processing',
])  
time = pd.date_range("2000-01-01", freq="6h", periods=1000)
variable = ["u10m", "v10m"]
dataset_name = "my_dataset.zarr"
build_dataset(data, time, variable, dataset_name, zarr_cache=zarr_cache, apache_beam_options=options)
