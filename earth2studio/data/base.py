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

from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray


@runtime_checkable
class DataSource(Protocol):
    """Data source interface."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....]. The coords
            should be provided. Time coordinate should be a datetime array and the
            variable coordinate should be array of strings with Earth2Studio variable
            ids.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data. Async data sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, ....]. The coords
            should be provided. Time coordinate should be a datetime array and the
            variable coordinate should be array of strings with Earth2Studio variable
            ids.
        """
        pass


@runtime_checkable
class ForecastSource(Protocol):
    """Forecast source interface"""

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, lead_time, ...].
            The coords should be provided. Time coordinate should be a TimeArray,
            lead time coordinate a LeadTimeArray and the variable coordinate should be
            an array of strings with Earth2Studio variable ids.
        """
        pass

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data. Async forecast sources support this.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return data for.
        lead_time: timedelta | list[timedelta], LeadTimeArray
            Timedelta, list of timedeltas or array of np.timedelta that refers to the
            forecast lead time to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return.

        Returns
        -------
        xr.DataArray
            An xarray data-array with the dimensions [time, variable, lead_time, ...].
            The coords should be provided. Time coordinate should be a TimeArray,
            lead time coordinate a LeadTimeArray and the variable coordinate should be
            an array of strings with Earth2Studio variable ids.
        """
        pass
