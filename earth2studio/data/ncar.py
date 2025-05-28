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

import asyncio
import calendar
import concurrent.futures
import hashlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import NCAR_ERA5Lexicon
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class NCARAsyncTask:
    """Small helper struct for Async tasks"""

    ncar_file_uri: str
    ncar_data_variable: str
    # Dictionary mapping time index -> time id
    ncar_time_indices: dict[int, datetime]
    # Dictionary mapping level index -> varaible id
    ncar_level_indices: dict[int, str]


class NCAR_ERA5:
    """ERA5 data provided by NSF NCAR via the AWS Open Data Sponsorship Program. ERA5
    is the fifth generation of the ECMWF global reanalysis and available on a 0.25
    degree WGS84 grid at hourly intervals spanning from 1940 to the present.

    Parameters
    ----------
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
    cache : bool, optional
            Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Timeout in seconds for async operations, by default 600


    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional resources:
    https://registry.opendata.aws/nsf-ncar-era5/
    """

    NCAR_EAR5_LAT = np.linspace(90, -90, 721)
    NCAR_EAR5_LON = np.linspace(0, 360, 1440, endpoint=False)

    def __init__(
        self,
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._max_workers = max_workers
        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the NCAR lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from NCAR ERA5
        """

        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Modify the worker amount
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the ARCO lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Create tasks and group based on variable
        data_arrays: dict[str, list[xr.DataArray]] = {}
        async_tasks = []
        for task in self._create_tasks(time, variable).values():
            future = self.fetch_wrapper(task)
            async_tasks.append(future)

        # Now wait
        results = await tqdm.gather(
            *async_tasks, desc="Fetching NCAR ERA5 data", disable=(not self._verbose)
        )
        # Group based on variable
        for result in results:
            key = str(result.coords["variable"])
            if key not in data_arrays:
                data_arrays[key] = []
            data_arrays[key].append(result)

        # Concat times for same variable groups
        array_list = [xr.concat(arrs, dim="time") for arrs in data_arrays.values()]
        # Now concat varaibles
        res = xr.concat(array_list, dim="variable")
        res.name = None  # remove name, which is kept from one of the arrays

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return res.sel(time=time, variable=variable)

    def _create_tasks(
        self, time: list[datetime], variable: list[str]
    ) -> dict[str, NCARAsyncTask]:
        """Create download tasks, each corresponding to one file on S3. The H5 file
        stored in the dataset contains

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        variables : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[dict]
            List of download tasks.
        """
        tasks: dict[str, NCARAsyncTask] = {}  # group pressure-level variables

        s3_pattern = "s3://nsf-ncar-era5/{product}/{year}{month:02}/{product}.{variable}.{grid}.{year}{month:02}{daystart:02}00_{year}{month:02}{dayend:02}23.nc"
        for i, t in enumerate(time):
            for j, v in enumerate(variable):
                ncar_name, _ = NCAR_ERA5Lexicon[v]

                product = ncar_name.split("::")[0]
                variable_name = ncar_name.split("::")[1]
                grid = ncar_name.split("::")[2]
                level_index = int(ncar_name.split("::")[3])
                data_variable = f"{variable_name.split('_')[-1].upper()}"

                # Pressure is held in daily nc files
                if product == "e5.oper.an.pl":
                    daystart = t.day
                    dayend = t.day
                    time_index = t.hour
                # Surface held in monthly
                else:
                    daystart = 1
                    dayend = calendar.monthrange(t.year, t.month)[-1]
                    time_index = int(
                        (t - datetime(t.year, t.month, 1)).total_seconds() / 3600
                    )

                file_name = s3_pattern.format(
                    product=product,
                    variable=variable_name,
                    grid=grid,
                    year=t.year,
                    month=t.month,
                    daystart=daystart,
                    dayend=dayend,
                )

                if file_name in tasks:
                    tasks[file_name].ncar_time_indices[time_index] = t
                    tasks[file_name].ncar_level_indices[level_index] = v
                else:
                    tasks[file_name] = NCARAsyncTask(
                        ncar_file_uri=file_name,
                        ncar_data_variable=data_variable,
                        ncar_time_indices={time_index: t},
                        ncar_level_indices={level_index: v},
                    )

        return tasks

    async def fetch_wrapper(
        self,
        task: NCARAsyncTask,
    ) -> xr.DataArray:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(
            task.ncar_file_uri,
            task.ncar_data_variable,
            list(task.ncar_time_indices.keys()),
            list(task.ncar_level_indices.keys()),
        )

        # Rename levels coord to variable
        out = out.rename({"level": "variable", "longitude": "lon", "latitude": "lat"})
        out = out.assign_coords(variable=list(task.ncar_level_indices.values()))
        # Shouldnt be needed but just in case
        out = out.assign_coords(time=list(task.ncar_time_indices.values()))
        return out

    async def fetch_array(
        self,
        nc_file_uri: str,
        data_variable: str,
        time_idx: list[int],
        level_idx: list[int],
    ) -> xr.DataArray:
        """Fetches requested array from remote store

        Parameters
        ----------
        nc_file_uri : str
            S3 URI to NetCDF file
        data_variable : str
            Data variable name of the array to use in the NetCDF file
        time_idx : list[int]
            Time indexes (hours since start time of file)
        level_idx : list[int]
            Pressure level indices if applicable, should be same length as time_idx

        Returns
        -------
        np.ndarray
            Data
        """
        logger.debug(
            f"Fetching NCAR ERA5 variable: {data_variable} in file {nc_file_uri}"
        )
        # Here we manually cache the data arrays, this is because fsspec caches the
        # extracted NetCDF file. Not super optimal, can have some repeat storage given
        # different level / time indexes
        sha = hashlib.sha256(
            (
                str(nc_file_uri) + str(data_variable) + str(time_idx) + str(level_idx)
            ).encode()
        )
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if os.path.exists(cache_path):
            ds = await asyncio.to_thread(
                xr.open_dataarray, cache_path, engine="h5netcdf", cache=False
            )
        else:
            # New fs every call so we dont block, netcdf reads seems to not support
            # open_async -> S3AsyncStreamedFile (big sad)
            fs = s3fs.S3FileSystem(anon=True, asynchronous=False)
            with fs.open(nc_file_uri, "rb", block_size=4 * 1400 * 720) as f:
                ds = await asyncio.to_thread(
                    xr.open_dataset, f, engine="h5netcdf", cache=False
                )
                # Sometimes data field have VAR_ prepended
                if f"VAR_{data_variable}" in ds:
                    data_variable = f"VAR_{data_variable}"

                if data_variable not in ds:
                    raise ValueError(
                        f"Variable '{data_variable}' or 'VAR_{data_variable}' from task not found in dataset. "
                        + f"Available variables: {list(ds.keys())}."
                    )

                # Pressure level variable
                if "level" in ds.dims:
                    ds = ds.isel(time=list(time_idx), level=list(level_idx))[
                        data_variable
                    ]
                # Other product
                else:
                    ds = ds.isel(time=list(time_idx))[data_variable]
                    ds = ds.expand_dims({"level": [0]}, axis=1)
                # Load the data, this is the actual download
                ds = await asyncio.to_thread(ds.load)
                # Cache nc file if present
                if self._cache:
                    await asyncio.to_thread(ds.to_netcdf, cache_path, engine="h5netcdf")

        return ds

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify that date time is valid for ERA5 based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        """
        for time in times:
            if time < datetime(1940, 1, 1):
                raise ValueError(
                    f"Requested date time {time} must be after January 1st, 1940 for NCAR ERA5"
                )
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for NCAR ERA5"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "ncar_era5")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_ncar_era5")
        if not os.path.exists(cache_location):
            os.makedirs(cache_location, exist_ok=True)
        return cache_location
