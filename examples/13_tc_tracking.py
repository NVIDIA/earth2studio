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

Tropical cyclone tracking with tracker diagnostic models

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
# In this example, we will use hurricane Harvey as the focus study.
# Harvey was a storm that cause signifcant damage in the United States in August 2017.
# Earth2Studio provides two variations of TC trackers :py:class:`earth2studio.models.dx.TCTrackerVitart`
# and :py:class:`earth2studio.models.dx.TCTrackerWuDuan`.
# The difference being the underlying algorithm used to identify the center.
# This example will demonstrate the later.
#
# Thus, we need the following:
#
# - Diagostic Model: Use the TC tracker :py:class:`earth2studio.models.dx.TCTrackerWuDuan`.
# - Datasource: Pull data from the WB2 ERA5 data api :py:class:`earth2studio.data.WB2ERA5`.
# - Prognostic Model: Use the built in FourCastNet Model :py:class:`earth2studio.models.px.FCN`.
# - IO Backend: Let's save the outputs into a Zarr store :py:class:`earth2studio.io.ZarrBackend`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

from earth2studio.data import WB2ERA5
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import FCN

# Create tropical cyclone tracker
tracker = TCTrackerWuDuan()

# Load the default model package which downloads the check point from NGC
package = FCN.load_default_package()
model = FCN.load_model(package)

# Create the data source
data = WB2ERA5()

# Create the IO handler, store in memory
io = ZarrBackend()

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
from datetime import datetime, timedelta

import torch

from earth2studio.data import prep_data_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tracker = tracker.to(device)

# Land fall occured August 25th 2017
times = [datetime(2017, 8, 25) + timedelta(hours=6 * i) for i in range(3)]
print(tracker.path_buffer.shape)
for time in times:
    da = data(time, tracker.input_coords()["variable"])
    input, input_coords = prep_data_array(da, device=device)

    output, output_coords = tracker(input, input_coords)

    print(output.shape)

torch.save(output, "output.pt")

# from earth2studio.models.dx.tc_tracking import get_tracks_from_positions
# df_tracks = get_tracks_from_positions(
#     output, output_coords, min_length=3, search_radius_km=250, max_skips=1
# )

# df_tracks.to_csv("output_2.csv")
# print(df_tracks['point_name'].isin([0]))
