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

#
# Target reference for eval 8:
# Fetch marine/buoy surface observations using UFSObsConv dataframe source.
# UFSObsConv provides all conventional obs including ships and buoys (SFCSHP class).
# Filter the returned DataFrame by observation class "SFCSHP" for marine data.

from datetime import datetime, timedelta

from earth2studio.data import UFSObsConv

# Initialize UFSObsConv data source (UFS conventional observations)
ds = UFSObsConv(time_tolerance=timedelta(hours=3))

# Fetch temperature observations (includes all platforms)
time = [datetime(2020, 1, 1, 0)]
variable = ["t"]

data = ds(time, variable)

# Filter for marine/buoy observations (SFCSHP class includes ships and buoys)
marine_data = data[data["class"] == "SFCSHP"]

# Inspect the result (returns a pandas DataFrame)
print(marine_data)
print(f"Columns: {list(marine_data.columns)}")
print(f"Shape: {marine_data.shape}")
