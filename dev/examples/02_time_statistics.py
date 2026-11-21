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

"""
Temporal Statistics
===================

Use compact variable labels to plan source coordinates and reduce Xarray data.
"""

# %%
# Declare Statistical Variables
# -----------------------------
# Variable labels remain strings and carry their temporal statistic.

from collections.abc import Hashable

import numpy as np
import xarray as xr

import earth2studio.utils.time_statistics as e2s

# Statistic-qualified labels distinguish multiple windows of one source quantity.
variables = ["t2m:mean:1day", "t2m:mean:1week", "t2m:mean:30days"]
print(variables)

# %%
# Plan Analysis Times
# -------------------
# A modifier describes both the reduction and its valid-time window.

modifier = "mean:24h"
valid_time = np.datetime64("2026-08-28T06:00")
delta_t = np.timedelta64(6, "h")
times = e2s.source_times(modifier, valid_time, delta_t=delta_t)
print(times)

# %%
# Reduce a Variable Block
# -----------------------
# One call reduces both wind components on the existing array backend.

wind = xr.DataArray(
    np.arange(8).reshape(4, 2),
    dims=("time", "variable"),
    coords={"time": times, "variable": ["u10m", "v10m"]},
)
result = e2s.apply_time_statistic(wind, modifier, target=valid_time, delta_t=delta_t)
result = result.assign_coords(
    variable=[f"{variable}:{modifier}" for variable in result.coords["variable"].values]
)
result.attrs["earth2studio_statistics"] = {
    variable: e2s.time_statistic_metadata(modifier)
    for variable in result.coords["variable"].values
}
print(result)

# %%
# Plan Forecast Lead Times
# ------------------------
# Forecast sources keep initialization time fixed and vary lead time.

leads = e2s.source_lead_times(modifier, np.timedelta64(6, "h"), delta_t=delta_t)
print(leads)

# %%
# Extend the Registry
# -------------------
# Custom functions also receive an entire variable block.


def value_range(array: xr.DataArray, dimension: Hashable) -> xr.DataArray:
    """Return the range along one dimension."""
    return array.max(dim=dimension) - array.min(dim=dimension)


e2s.register_time_statistic("range", value_range)
print(e2s.apply_time_statistic(wind, "range:24h", target=valid_time, delta_t=delta_t))
