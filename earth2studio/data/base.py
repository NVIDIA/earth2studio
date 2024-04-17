# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from datetime import datetime
from typing import Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import TimeArray, VariableArray


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
            An xarray data-array with the dimensions [time, channel, ....]. The coords
            should be provided. Time coordinate should be a datetime array and the
            channel coordinate should be array of strings with Earth2Studio channel ids.
        """
        pass
