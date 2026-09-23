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

"""Define how long StormCast conditioning forecasts must remain usable.

One conditioning forecast is created from each six-hour GFS cycle and reused
for hourly StormCast forecasts. Extra coverage allows the previous
conditioning forecast to remain usable when the next GFS cycle arrives late.
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Plan for StormCast forecasts that start up to 10 hours after the GFS cycle.
_CONDITIONING_WINDOW_HOURS = 10

# Extra protection for timing near the edge of that window.
_COVERAGE_MARGIN_HOURS = 2


def conditioning_hours(nsteps: int) -> int:
    """Forecast length for one conditioning run (per 6 h GFS cycle).

    StormCast starts forecasts every hour, but creates conditioning data only
    once per GFS cycle. The conditioning data must cover every hourly start and
    continue through the end of each forecast.

    Parameters
    ----------
    nsteps : int
        StormCast rollout length (``nsteps``).

    Returns
    -------
    int
        Conditioning forecast length in hours.
    """
    return _CONDITIONING_WINDOW_HOURS + nsteps + _COVERAGE_MARGIN_HOURS


def conditioning_covers(cond_cycle: datetime, sc_init: datetime, nsteps: int) -> bool:
    """Return whether conditioning data covers the full StormCast forecast.

    The conditioning forecast starts at ``cond_cycle`` and the StormCast
    forecast starts at ``sc_init``. Return ``True`` when conditioning data
    remains available through the final StormCast step.

    If this returns ``False``, the caller should wait for newer conditioning
    data instead of publishing an incomplete forecast.
    """
    conditioning_end = cond_cycle + timedelta(hours=conditioning_hours(nsteps))
    stormcast_end = sc_init + timedelta(hours=nsteps)
    return stormcast_end <= conditioning_end
