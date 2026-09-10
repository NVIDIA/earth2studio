"""
Temporal Statistics
===================

Use one compact string to plan source coordinates and reduce Xarray data.
"""

# %%
# Plan Analysis Times
# -------------------
# A modifier describes both the reduction and its valid-time window.

from collections.abc import Hashable

import numpy as np
import xarray as xr

import earth2studio.utils.time_statistics as e2s

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
result = e2s.apply_time_statistic(wind, modifier)
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
print(e2s.apply_time_statistic(wind, "range:24h"))
