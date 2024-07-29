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

import asyncio
import os
import pathlib
import shutil
import threading
from datetime import datetime

import fsspec
import gcsfs
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    unordered_generator,
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

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose

        fs = gcsfs.GCSFileSystem(
            cache_timeout=-1,
            token="anon",  # noqa: S106 # nosec B106
            access="read_only",
            block_size=2**20,
        )

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

        fs_map = fsspec.FSMap(
            "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", fs
        )
        self.zarr_group = zarr.open(fs_map, mode="r")
        self.async_timeout = 600
        self.async_process_limit = 4

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

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

        # This makes this function safe in existing async io loops
        # I.e. runnable in Jupyter notebooks
        xr_array = None

        def thread_func() -> None:
            """Function to call in seperate thread"""
            nonlocal xr_array
            loop = asyncio.new_event_loop()
            xr_array = loop.run_until_complete(
                asyncio.wait_for(
                    self.create_data_array(time, variable), timeout=self.async_timeout
                )
            )

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array

    async def create_data_array(
        self, time: list[datetime], variable: list[str]
    ) -> xr.DataArray:
        """Async function that creates and populates an xarray data array with requested
        ARCO data. Asyncio tasks are created for each data array enabling concurrent
        fetching.

        Parameters
        ----------
        time : list[datetime]
            Time list to fetch
        variable : list[str]
            Variable list to fetch

        Returns
        -------
        xr.DataArray
            Xarray data array
        """
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

        async def fetch_wrapper(
            e: tuple[datetime, int, str, int]
        ) -> tuple[int, int, np.ndarray]:
            """Small wrapper that is awaitable for async generator"""
            return e[1], e[3], self.fetch_array(e[0], e[2])

        args = [
            (t, i, v, j) for j, v in enumerate(variable) for i, t in enumerate(time)
        ]
        func_map = map(fetch_wrapper, args)

        pbar = tqdm(
            total=len(args), desc="Fetching ARCO data", disable=(not self._verbose)
        )
        # Mypy will struggle here because the async generator uses a generic type
        async for t, v, data in unordered_generator(  # type: ignore[misc,unused-ignore]
            func_map, limit=self.async_process_limit
        ):
            xr_array[t, v] = data  # type: ignore[has-type,unused-ignore]
            pbar.update(1)

        return xr_array

    def fetch_array(self, time: datetime, variable: str) -> np.ndarray:
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
        # Load levels coordinate system from Zarr store and check
        level_coords = self.zarr_group["level"][:]
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

        shape = self.zarr_group[arco_variable].shape
        # Static variables
        if len(shape) == 2:
            output = modifier(self.zarr_group[arco_variable][:])
        # Surface variable
        elif len(shape) == 3:
            output = modifier(self.zarr_group[arco_variable][time_index])
        # Atmospheric variable
        else:
            level_index = np.where(level_coords == int(level))[0][0]
            output = modifier(self.zarr_group[arco_variable][time_index, level_index])

        return output

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "arco")
        if not self._cache:
            if not DistributedManager.is_initialized():
                DistributedManager.initialize()
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
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
        gcstore = gcsfs.GCSMap(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            gcs=gcs,
        )
        zarr_group = zarr.open(gcstore, mode="r")
        # Load time coordinate system from Zarr store and check
        time_index = cls._get_time_index(time)
        max_index = zarr_group["time"][-1]
        return time_index >= 0 and time_index <= max_index
