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

import os
from datetime import datetime
from typing import Any

import xarray as xr
from numpy import ndarray
from pandas import to_datetime

from earth2studio.utils.type import TimeArray, VariableArray


class DataArrayFile:
    """A local xarray dataarray file data source. This file should be compatable with
    xarray. For example, a netCDF file.

    Parameters
    ----------
    file_path : str
        Path to xarray data array compatible file.
    """

    def __init__(self, file_path: str, **xr_args: Any):
        self.file_path = file_path
        self.da = xr.open_dataarray(self.file_path, **xr_args)
        # self.da = xr.open_dataarray(self.file_path, **xr_args)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        return self.da.sel(time=time, variable=variable)


class DataSetFile:
    """A local xarray dataset file data source. This file should be compatable with
    xarray. For example, a netCDF file.

    Parameters
    ----------
    file_path : str
        Path to xarray dataset compatible file.
    array_name : str
        Data array name in xarray dataset
    """

    def __init__(self, file_path: str, array_name: str, **xr_args: Any):
        self.file_path = file_path
        self.da = xr.open_dataset(self.file_path, **xr_args)[array_name]

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        return self.da.sel(time=time, variable=variable)


class DataArrayDirectory:
    """A local xarray dataarray directory data source. This file should be compatable with
    xarray. For example, a netCDF file. the structure of the directory should be like
    path/to/monthly/files
    |___2020
    |   |___2020_01.nc
    |   |___2020_02.nc
    |   |___ ...
    |
    |___2021
        |___2021_01.nc
        |___...

    Parameters
    ----------
    file_path : str
        Path to xarray data array compatible file.
    xr_args : Any
        Keyword arguments to send to the xarray opening method.
    """

    def __init__(self, dir_path: str, **xr_args: Any):
        self.dir_path = dir_path
        self.das: dict[str, dict[str, xr.DataArray]] = {}
        for yr in os.listdir(self.dir_path):
            yr_dir = os.path.join(self.dir_path, yr)
            if os.path.isdir(yr_dir):
                self.das[yr] = {}
                for fl in os.listdir(yr_dir):
                    pth = os.path.join(yr_dir, fl)
                    if os.path.isfile(pth):
                        try:
                            arr = xr.open_dataarray(pth, **xr_args)
                        except:  # noqa
                            continue
                        mon = fl.split(".")[0].split("_")[-1]
                        self.das[yr][mon] = arr

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        if not (isinstance(time, list) or isinstance(time, ndarray)):
            time = [time]
        if not (isinstance(variable, list) or isinstance(variable, ndarray)):
            variable = [variable]

        arrs = []
        for tt in time:
            yr = str(to_datetime(tt).year)
            mon = str(to_datetime(tt).month).zfill(2)
            arrs.append(self.das[yr][mon].sel(time=tt, variable=variable))

        return xr.concat(arrs, dim="time")
