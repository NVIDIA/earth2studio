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
# Target reference for eval 6:
# Fetch ensemble precipitation forecast 5 days out from 2024-04-01 00Z.
# GEFS_FX supports tp and provides ensemble members.

from datetime import datetime, timedelta

from earth2studio.data import GEFS_FX

# Initialize GEFS ensemble forecast data source
ds = GEFS_FX()

# Fetch total precipitation at 120h (5 days) lead time
time = [datetime(2024, 4, 1, 0)]
lead_time = [timedelta(hours=120)]
variable = ["tp"]

data = ds(time, lead_time, variable)

# Inspect the result (should have ensemble dimension)
print(data)
print(f"Shape: {data.shape}")
print(f"Coords: {list(data.coords)}")
