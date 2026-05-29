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
# Target reference for eval 5:
# Fetch IFS ensemble forecast for t2m initialized 2024-02-01 00Z
# at lead times of 24h and 72h.

from datetime import datetime, timedelta

from earth2studio.data import IFS_ENS_FX

# Initialize IFS ensemble forecast data source
ds = IFS_ENS_FX()

# Fetch t2m at 24h and 72h lead times
time = [datetime(2024, 2, 1, 0)]
lead_time = [timedelta(hours=24), timedelta(hours=72)]
variable = ["t2m"]

data = ds(time, lead_time, variable)

# Inspect the result (should have ensemble and lead_time dimensions)
print(data)
print(f"Shape: {data.shape}")
print(f"Coords: {list(data.coords)}")
