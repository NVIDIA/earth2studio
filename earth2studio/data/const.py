# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from earth2studio.data.utils import prep_data_inputs, prep_forecast_inputs
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray


class Constant:
    """A constant data source. Primarily useful for testing.

    Parameters
    ----------
    domain_coords: OrderedDict[str, np.ndarray]
        Domain coordinates that the random data will assume (such as lat, lon).
    value : int, optional
        Value to set data to, by default 1
    """

    def __init__(self, domain_coords: OrderedDict[str, np.ndarray], value: int = 1):
        self.domain_coords = domain_coords
        self.value = value

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve constant data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Constant data array
        """

        time, variable = prep_data_inputs(time, variable)

        shape = [len(time), len(variable)]
        coords = {"time": time, "variable": variable}

        for key, value in self.domain_coords.items():
            shape.append(len(value))
            coords[key] = value

        da = xr.DataArray(
            data=self.value * np.ones(shape), dims=list(coords), coords=coords
        )

        return da


class Constant_FX:
    """A constant forecast data source. Primarily useful for testing.

    Parameters
    ----------
    domain_coords: OrderedDict[str, np.ndarray]
        Domain coordinates that the random data will assume (such as lat, lon).
    value : int, optional
        Value to set data to, by default 1
    """

    def __init__(self, domain_coords: OrderedDict[str, np.ndarray], value: int = 1):
        self.domain_coords = domain_coords
        self.value = value

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve constant data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Constant data array
        """

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        shape = [len(time), len(lead_time), len(variable)]
        coords = {"time": time, "lead_time": lead_time, "variable": variable}

        for key, value in self.domain_coords.items():
            shape.append(len(value))
            coords[key] = value

        da = xr.DataArray(
            data=self.value * np.ones(shape), dims=list(coords), coords=coords
        )

        return da
