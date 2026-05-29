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
# Target reference for eval 2:
# Fetch 500 hPa geopotential height from ERA5 for 2020-06-15 12Z.
# Use ARCO (no auth required) instead of CDS.

from datetime import datetime

from earth2studio.data import ARCO

# Initialize ARCO data source (ERA5 reanalysis, no authentication needed)
ds = ARCO()

# Fetch z500 for a single time
time = [datetime(2020, 6, 15, 12)]
variable = ["z500"]

data = ds(time, variable)

# Inspect the result
print(data)
print(f"Shape: {data.shape}")
print(f"Coords: {list(data.coords)}")
