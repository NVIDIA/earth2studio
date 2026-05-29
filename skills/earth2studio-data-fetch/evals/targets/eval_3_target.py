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
# Target reference for eval 3:
# Fetch high-resolution surface wind data over North America for 2024-03-10 18Z.
# HRRR is the best choice for high-res NA coverage (3km).

from datetime import datetime

from earth2studio.data import HRRR

# Initialize HRRR data source (3km North America, no auth required)
ds = HRRR()

# Fetch 10m wind components for a single time
time = [datetime(2024, 3, 10, 18)]
variable = ["u10m", "v10m"]

data = ds(time, variable)

# Inspect the result
print(data)
print(f"Shape: {data.shape}")
print(f"Coords: {list(data.coords)}")
