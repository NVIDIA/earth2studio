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

"""Building blocks for writing couplings.

A coupling is ``(outputs, leads) -> DataArray``; see
:class:`~earth2studio.coupler.CoupledRunner`. These return couplings for the two
exchanges that recur, and compose with ordinary functions for regridding or unit
conversion. Hookups for specific model pairs belong in this module too, as
functions returning a coupling, so they are discoverable next to these.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from earth2studio.coupler.runner import Coupling, Outputs


def hold(src: str) -> Coupling:
    """Coupling giving ``src``'s newest output at or before each requested lead.

    Parameters
    ----------
    src : str
        Name of the producing model

    Returns
    -------
    Coupling
        Holds a slower producer constant; an equal-cadence producer that has run
        first gives its output at exactly the requested lead
    """

    def coupling(outputs: Outputs, leads: np.ndarray) -> xr.DataArray:
        return outputs[src].sel(lead_time=leads, method="pad")

    return coupling


def window_mean(src: str, step: np.timedelta64) -> Coupling:
    """Coupling giving the mean of ``src`` over ``(lead, lead + step]``, per lead.

    Parameters
    ----------
    src : str
        Name of the producing model, which must run earlier in the window
    step : np.timedelta64
        Averaging window, normally the consuming model's step

    Returns
    -------
    Coupling
        Raises ``ValueError`` if ``src`` has not yet covered a requested window,
        which means ``src`` is listed after the consumer
    """

    def coupling(outputs: Outputs, leads: np.ndarray) -> xr.DataArray:
        x = outputs[src]
        means = []
        for lead in leads:
            if x["lead_time"].values[-1] < lead + step:
                raise ValueError(
                    f"{src!r} has not reached {lead + step}; "
                    "list it before the model that consumes its mean"
                )
            window = x.sel(lead_time=slice(lead + np.timedelta64(1, "ns"), lead + step))
            means.append(window.mean("lead_time", skipna=False, keepdims=True))
        return xr.concat(means, dim="lead_time")

    return coupling
