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

from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem, LeadTimeArray, TimeArray, VariableArray


def fetch_data(
    source: DataSource,
    time: TimeArray,
    variable: VariableArray,
    lead_time: LeadTimeArray = np.array([np.timedelta64(0, "h")]),
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, CoordSystem]:
    """Utility function to fetch data for models and load data on the target device.

    Parameters
    ----------
    source : DataSource
        The data source to fetch from
    time : TimeArray
        Timestamps to return data for (UTC).
    variable : VariableArray
        Strings or list of strings that refer to variables to return
    lead_time : LeadTimeArray, optional
        Lead times to fetch for each provided time, by default
        np.array(np.timedelta64(0, "h"))
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    da = []
    for lead in lead_time:
        adjust_times = np.array([t + lead for t in time], dtype="datetime64[ns]")
        da0 = source(adjust_times, variable)
        da0 = da0.expand_dims(dim={"lead_time": 1}, axis=1)
        da0 = da0.assign_coords(lead_time=np.array([lead], dtype="timedelta64[ns]"))
        da0 = da0.assign_coords(time=time)
        da.append(da0)

    return prep_data_array(xr.concat(da, "lead_time"), device=device)


def prep_data_array(
    da: xr.DataArray,
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, CoordSystem]:
    """Prepares a data array from a data source for inference workflows by converting
    the data array to a torch tensor and the coordinate system to an OrderedDict.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict
    """

    out = torch.Tensor(da.values).to(device)

    out_coords = OrderedDict()
    for dim in da.coords.dims:
        out_coords[dim] = np.array(da.coords[dim])

    return out, out_coords


def prep_data_inputs(
    time: datetime | list[datetime] | TimeArray,
    variable: str | list[str] | VariableArray,
) -> tuple[list[datetime], list[str]]:
    """Simple method to pre-process data source inputs into a common form

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Datetime, list of datetimes or array of np.datetime64 to fetch
    variable : str | list[str] | VariableArray
        String, list of strings or array of strings that refer to variables

    Returns
    -------
    tuple[list[datetime], list[str]]
        Time and variable lists
    """
    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    if isinstance(time, np.ndarray):  # np.datetime64 -> datetime
        time = timearray_to_datetime(time)

    return time, variable
