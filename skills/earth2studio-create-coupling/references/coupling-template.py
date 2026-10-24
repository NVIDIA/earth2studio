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

"""Template: a pair-specific coupling that belongs in ``coupler/couplings.py``."""

import numpy as np
import xarray as xr

from earth2studio.coupler import Coupling, Outputs, window_mean


def atmos_to_ocean(atmos: str, step: np.timedelta64) -> Coupling:
    """Coupling forcing an ocean model with the atmosphere's mean surface fields.

    Parameters
    ----------
    atmos : str
        Name of the atmosphere model in the runner; list it before the ocean
    step : np.timedelta64
        The ocean model's step, which is the averaging window

    Returns
    -------
    Coupling
        Mean of the atmosphere over ``(lead, lead + step]``, with the variables
        the ocean forces on, in K converted to degC
    """
    mean = window_mean(atmos, step)

    def coupling(outputs: Outputs, leads: np.ndarray) -> xr.DataArray:
        x = mean(outputs, leads)
        x = x.sel(variable=["t2m", "u10m"])  # The slot's variables, in its order.
        # Unit conversion and regridding go here, on the way out.
        return x.where(x["variable"] != "t2m", x - 273.15)

    return coupling
