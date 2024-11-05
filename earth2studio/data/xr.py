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

import os
from datetime import datetime
from typing import Any
from numpy import ndarray
from pandas import to_datetime
from datetime import datetime

import xarray as xr

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


class DataVarFile:
    """A local xarray dataset file data source where the underlying data
    has been saved with variable names as xarray `Data variables`.

    This is essentially a utility function for users that want to convert
    the data structure into something usable by earth2studio.

    Parameters
    ----------
    file_path : str
        A file path or directory containing the xarray compatible file(s).
        If this is a directory, we find all files using os.path.listdir and
        open with xarray.open_mfdataset(files, **xr_args)
    xr_args : Any
        Keyword arguments to send to the xarray opening method.
    """
    def __init__(self, file_path: str, **xr_args: Any):
        self.file_path = file_path

        if os.path.isdir(file_path):
            import numpy as np
            files = sorted([
                os.path.join(
                    file_path, file
                ) for file in os.listdir(file_path)
            ])

            ds = []
            ds0 = []
            count = 0
            for f in files:
                if len(ds0) == 0:
                    t0 = f.split('/')[-1].split('_pkg')[0].split('_')[-1][:13] # TODO move out of loop, use time defined in file
                    temp_ds = xr.open_dataset(f, chunks = {'lead_time': 1})
                    temp_ds['ensemble'] = np.array([2*count, 2*count + 1])
                    ds0.append(temp_ds)
                else:
                    t0 = f.split('/')[-1].split('_pkg')[0].split('_')[-1][:13]
                    if t0 == t1:
                        temp_ds = xr.open_dataset(f, chunks = {'lead_time': 1})
                        temp_ds['ensemble'] = np.array([2*count, 2*count + 1])
                        ds0.append(temp_ds)
                    else:
                        count = 0
                        ds0 = xr.concat(ds0, 'ensemble')
                        ds.append(ds0)
                        temp_ds = xr.open_dataset(f, chunks = {'lead_time': 1})
                        temp_ds['ensemble'] = np.array([2*count, 2*count + 1])
                        ds0 = [temp_ds]
                t1 = t0
                count += 1

            ds.append(xr.concat(ds0, 'ensemble'))
            ds = xr.concat(ds, 'time')
        else:
            ds = xr.open_dataset(
                file_path, **xr_args
            )

        tr_dim = ['time', 'lead_time', 'variable', 'lat', 'lon']
        if 'ensemble' in ds:
            tr_dim = ['ensemble',] + tr_dim

        self.da = ds.to_array(
                dim = 'variable'
            ).transpose(
            *tr_dim
        )

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
        self.das = {}
        for yr in os.listdir(self.dir_path):
            yr_dir = os.path.join(self.dir_path, yr)
            if os.path.isdir(yr_dir):
                self.das[yr] = {}
                for fl in os.listdir(yr_dir):
                    pth = os.path.join(yr_dir, fl)
                    if os.path.isfile(pth):
                        try:
                            arr = xr.open_dataarray(pth, **xr_args)
                        except:
                            continue
                        mon = fl.split('.')[0].split('_')[-1]
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

        arrs = []
        for tt in time:
            yr = str(to_datetime(tt).year)
            mon = str(to_datetime(tt).month).zfill(2)
            arrs.append(self.das[yr][mon].sel(time=tt, variable=variable))

        return xr.concat(arrs, dim='time')
