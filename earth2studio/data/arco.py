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
import functools
import inspect
import os
import pathlib
import shutil
from datetime import datetime
from importlib.metadata import version

import fsspec
import gcsfs
import nest_asyncio
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    AsyncCachingFileSystem,
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import ARCOLexicon
from earth2studio.utils.type import TimeArray, VariableArray


class ARCO:
    """Analysis-Ready, Cloud Optimized (ARCO) is a data store of ERA5 re-analysis data
    currated by Google. This data is stored in Zarr format and contains 31 surface and
    pressure level variables (for 37 pressure levels)  on a 0.25 degree lat lon grid.
    Temporal resolution is 1 hour.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://cloud.google.com/storage/docs/public-datasets/era5
    """

    ARCO_LAT = np.linspace(90, -90, 721)
    ARCO_LON = np.linspace(0, 359.75, 1440)

    def __init__(
        self, cache: bool = True, verbose: bool = True, async_timeout: int = 600
    ):
        # Check Zarr version and use appropriate method
        try:
            zarr_version = version("zarr")
            self.zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            # Fallback to older method if version check fails
            self.zarr_major_version = 2  # Assume older version if we can't determine

        self._cache = cache
        self._verbose = verbose

        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        fs = gcsfs.GCSFileSystem(
            cache_timeout=-1,
            token="anon",  # noqa: S106 # nosec B106
            access="read_only",
            block_size=8**20,
            asynchronous=(self.zarr_major_version == 3),
            loop=loop,
        )

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            if self.zarr_major_version == 3:
                fs = AsyncCachingFileSystem(fs=fs, **cache_options, asynchronous=True)
            else:
                fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

        if self.zarr_major_version >= 3:
            # Zarr 3.0+ method
            zstore = zarr.storage.FsspecStore(
                fs,
                path="/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            )
            self.zarr_group = zarr.api.asynchronous.open(store=zstore, mode="r")
            self.level_coords = None
        else:
            # Legacy method for Zarr < 3.0
            logger.warning(
                "Using Zarr 2.0 method for ARCO, this can be extremely slow with caching!"
            )
            fs_map = fsspec.FSMap(
                "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", fs
            )
            self.zarr_group = zarr.open(fs_map, mode="r")
            self.level_coords = self.zarr_group["level"][:]

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
            return. Must be in the ARCO lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO
        """

        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

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
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), len(self.ARCO_LAT), len(self.ARCO_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.ARCO_LAT,
                "lon": self.ARCO_LON,
            },
        )

        args = [
            (t, i, v, j) for j, v in enumerate(variable) for i, t in enumerate(time)
        ]
        func_map = map(functools.partial(self.fetch_wrapper, xr_array=xr_array), args)

        # Before anything wait until the group gets opened
        if self.zarr_major_version >= 3:
            if inspect.isawaitable(self.zarr_group):
                self.zarr_group = await self.zarr_group
            self.level_coords = await (await self.zarr_group.get("level")).getitem(
                slice(None)
            )
        # Launch all fetch requests
        await tqdm.gather(
            *func_map, desc="Fetching ARCO data", disable=(not self._verbose)
        )
        return xr_array

    async def fetch_wrapper(
        self,
        e: tuple[datetime, int, str, int],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(e[0], e[2])
        xr_array[e[1], e[3]] = out

    async def fetch_array(self, time: datetime, variable: str) -> np.ndarray:
        """Fetches requested array from remote store

        Parameters
        ----------
        time : datetime
            Time to fetch
        variable : str
            Variable to fetch

        Returns
        -------
        np.ndarray
            Data
        """
        # Get time index (vanilla zarr doesnt support date indices)
        time_index = self._get_time_index(time)
        logger.debug(
            f"Fetching ARCO zarr array for variable: {variable} at {time.isoformat()}"
        )
        try:
            arco_name, modifier = ARCOLexicon[variable]
        except KeyError as e:
            logger.error(f"variable id {variable} not found in ARCO lexicon")
            raise e

        arco_variable, level = arco_name.split("::")
        if self.zarr_major_version >= 3:
            zarr_array = await self.zarr_group.get(arco_variable)
            shape = zarr_array.shape
            # Static variables
            if len(shape) == 2:
                data = await zarr_array.getitem(slice(None))
                output = modifier(data)
            # Surface variable
            elif len(shape) == 3:
                data = await zarr_array.getitem(time_index)
                output = modifier(data)
            # Atmospheric variable
            else:
                # Load levels coordinate system from Zarr store and check
                level_index = np.searchsorted(self.level_coords, int(level))
                data = await zarr_array.getitem((time_index, level_index))
                output = modifier(data)
        else:
            # Zarr 2.0 fall back
            shape = self.zarr_group[arco_variable].shape
            # Static variables
            if len(shape) == 2:
                output = modifier(self.zarr_group[arco_variable][:])
            # Surface variable
            elif len(shape) == 3:
                output = modifier(self.zarr_group[arco_variable][time_index])
            # Atmospheric variable
            else:
                level_index = np.where(self.level_coords == int(level))[0][0]
                output = modifier(
                    self.zarr_group[arco_variable][time_index, level_index]
                )

        return output

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "arco")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_arco")
        return cache_location

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for ARCO

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ARCO"
                )

            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for ARCO"
                )

            if time >= datetime(year=2023, month=11, day=10):
                raise ValueError(
                    f"Requested date time {time} needs to be before November 10th, 2023 for ARCO"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in ARCO")

    @classmethod
    def _get_time_index(cls, time: datetime) -> int:
        """Small little index converter to go from datetime to integer index.
        We don't need to do with with xarray, but since we are vanilla zarr for speed
        this conversion must be manual.

        Parameters
        ----------
        time : datetime
            Input date time

        Returns
        -------
        int
            Time coordinate index of ARCO data
        """
        start_date = datetime(year=1900, month=1, day=1)
        duration = time - start_date
        return int(divmod(duration.total_seconds(), 3600)[0])

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Checks if given date time is avaliable in the ARCO data source

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access

        Returns
        -------
        bool
            If date time is avaiable
        """
        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.utcfromtimestamp(float((time - _unix) / _ds))

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        gcs = gcsfs.GCSFileSystem(cache_timeout=-1)

        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            zarr_major_version = 2

        if zarr_major_version >= 3:
            gcstore = zarr.storage.FsspecStore(
                gcs,
                path="/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            )
        else:
            gcstore = gcsfs.GCSMap(
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                gcs=gcs,
            )

        zarr_group = zarr.open(gcstore, mode="r")
        # Load time coordinate system from Zarr store and check
        time_index = cls._get_time_index(time)
        max_index = zarr_group["time"][-1]
        return time_index >= 0 and time_index <= max_index
