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
import hashlib
import os
import pathlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from fsspec.implementations.ftp import FTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import GFSLexicon
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class GFSAsyncTask:
    """Small helper struct for Async tasks"""

    data_array_indices: tuple[int, int, int]
    gfs_file_uri: str
    gfs_byte_offset: int
    gfs_byte_length: int
    gfs_modifier: Callable


class GFS:
    """The global forecast service (GFS) initial state data source provided on an
    equirectangular grid. GFS is a weather forecast model developed by NOAA. This data
    source is provided on a 0.25 degree lat lon grid at 6-hour intervals spanning from
    Feb 26th 2021 to present date.

    Parameters
    ----------
    source: str, optional
        Data store location to pull from. Options are [aws, ncep], by default aws
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
    This data source only fetches the initial state of GFS and does not fetch an
    predicted time steps. See :class:`~earth2studio.data.GFS_FX` for fetching predicted
    data from this forecast system.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://registry.opendata.aws/noaa-gfs-bdp-pds/
    - https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php
    """

    GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
    MAX_BYTE_SIZE = 5000000

    GFS_LAT = np.linspace(90, -90, 721)
    GFS_LON = np.linspace(0, 359.75, 1440)

    def __init__(
        self,
        source: str = "aws",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._cache = cache
        self._verbose = verbose

        if source == "aws":
            self.uri_prefix = "noaa-gfs-bdp-pds"
            # Check to see if there is a running loop (initialized in async)
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(self._async_init())
            except RuntimeError:
                # Else we assume that async calls will be used which in that case
                # we will init the group in the call function when we have the loop
                self.fs = None

            # To update search "gfs." at https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
            # They are slowly adding more data
            def _range(time: datetime) -> None:
                if time < datetime(year=2021, month=1, day=1):
                    raise ValueError(
                        f"Requested date time {time} needs to be after January 1st, 2021 for GFS on AWS"
                    )

            self._history_range = _range
        elif source == "ncep":
            # Could use http location, but using ftp since better for larger data
            # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
            self.uri_prefix = "pub/data/nccf/com/gfs/prod/"
            self.fs = FTPFileSystem(host="ftpprd.ncep.noaa.gov")  # Not async

            def _range(time: datetime) -> None:
                if time + timedelta(days=10) < datetime.today():
                    raise ValueError(
                        f"Requested date time {time} needs to be within past 10 days for GFS NCEP source"
                    )

            self._history_range = _range
        else:
            raise ValueError(f"Invalid GFS source {source}")

        self.async_timeout = async_timeout

    async def _async_init(self) -> None:
        """Async initialization of zarr group

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GFS initial state / analysis data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GFS lexicon.

        Returns
        -------
        xr.DataArray
            GFS weather data array
        """
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
        )

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
            return. Must be in the GFS lexicon.

        Returns
        -------
        xr.DataArray
            GFS weather data array
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session()
        else:
            session = None

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with, compared to something seen in the
        # NCAR data source.
        xr_array = xr.DataArray(
            data=np.zeros(
                (len(time), 1, len(variable), len(self.GFS_LAT), len(self.GFS_LON))
            ),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "lead_time": [timedelta(hours=0)],
                "variable": variable,
                "lat": self.GFS_LAT,
                "lon": self.GFS_LON,
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, [timedelta(hours=0)], variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching GFS data", disable=(not self._verbose)
        )

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array.isel(lead_time=0)

    async def _create_tasks(
        self, time: list[datetime], lead_time: list[timedelta], variable: list[str]
    ) -> list[GFSAsyncTask]:
        """Create download tasks, each corresponding to one grib byte range on S3

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        variables : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[dict]
            List of download tasks
        """
        tasks: list[GFSAsyncTask] = []  # group pressure-level variables

        # Start with fetching all index files for each time / lead time
        args = [self._grib_index_uri(t, lt) for t in time for lt in lead_time]
        func_map = map(self._fetch_index, args)
        results = await tqdm.gather(
            *func_map, desc="Fetching GFS index files", disable=True
        )
        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                # Get index file dictionary
                index_file = results.pop(0)
                for k, v in enumerate(variable):
                    # sphinx - lexicon start
                    try:
                        gfs_name, modifier = GFSLexicon[v]
                    except KeyError:
                        logger.warning(
                            f"variable id {v} not found in GFS lexicon, good luck"
                        )
                        gfs_name = v

                        def modifier(x: np.array) -> np.array:
                            """Modify data (if necessary)."""
                            return x

                    byte_offset = None
                    byte_length = None
                    for key, value in index_file.items():
                        if gfs_name in key:
                            byte_offset = value[0]
                            byte_length = value[1]
                            break

                    if byte_length is None or byte_offset is None:
                        logger.warning(
                            f"Variable {v} not found in index file for time {t} at {lt}, values will be unset"
                        )
                        continue
                    # sphinx - lexicon end
                    tasks.append(
                        GFSAsyncTask(
                            data_array_indices=(i, j, k),
                            gfs_file_uri=self._grib_uri(t, lt),
                            gfs_byte_offset=byte_offset,
                            gfs_byte_length=byte_length,
                            gfs_modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: GFSAsyncTask,
        xr_array: xr.DataArray,
    ) -> xr.DataArray:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(
            task.gfs_file_uri,
            task.gfs_byte_offset,
            task.gfs_byte_length,
            task.gfs_modifier,
        )
        i, j, k = task.data_array_indices
        xr_array[i, j, k] = out

    async def fetch_array(
        self,
        grib_uri: str,
        byte_offset: int,
        byte_length: int,
        modifier: Callable,
    ) -> np.ndarray:
        """Fetch GFS data array. This will first fetch the index file to get byte range
        of the needed data, fetch the respective grib files and lastly combining grib
        files into single data array.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        lead_time : timedelta
            Forecast lead time to fetch
        variables : list[str]
            Variables to fetch

        Returns
        -------
        xr.DataArray
            FS data array for given time and lead time
        """
        logger.debug(f"Fetching GFS grib file: {grib_uri} {byte_offset}-{byte_length}")
        # Download the grib file to cache
        grib_file = await self._fetch_remote_file(
            grib_uri,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )
        # Open into xarray data-array
        da = xr.open_dataarray(
            grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        return modifier(da.values)

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for GFS based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for GFS"
                )
            # Check history range for given source
            self._history_range(time)

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int]]:
        """Fetch GFS atmospheric index file

        Parameters
        ----------
        index_uri : str
            URI to grib index file to download

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GFS vairables (byte offset, byte length)
        """
        # Grab index file
        index_file = await self._fetch_remote_file(index_uri)
        with open(index_file) as file:
            index_lines = [line.rstrip() for line in file]

        index_table = {}
        # Note we actually drop the last variable here (Vertical Speed Shear)
        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[0]}::{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        # Pop place holder
        return index_table

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetches remote file into cache"""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            if self.fs.async_impl:
                if byte_length:
                    byte_length = int(byte_offset + byte_length)
                data = await self.fs._cat_file(path, start=byte_offset, end=byte_length)
            else:
                data = await asyncio.to_thread(
                    self.fs.read_block, path, offset=byte_offset, length=byte_length
                )
            with open(cache_path, "wb") as file:
                await asyncio.to_thread(file.write, data)

        return cache_path

    def _grib_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS grib files"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}"
        )
        return os.path.join(self.uri_prefix, file_name)

    def _grib_index_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS index grib files"""
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}.idx"
        )
        return os.path.join(self.uri_prefix, file_name)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gfs")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_gfs")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the GFS object store. Uses S3 store

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
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        fs = s3fs.S3FileSystem(anon=True)
        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        s3_uri = f"s3://{cls.GFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class GFS_FX(GFS):
    """The global forecast service (GFS) forecast source provided on an equirectangular
    grid. GFS is a weather forecast model developed by NOAA. This data source is on a
    0.25 degree lat lon grid at 6-hour intervals spanning from Feb 26th 2021 to present
    date. Each forecast provides hourly predictions up to 16 days (384 hours).

    Parameters
    ----------
    source: str, optional
        Data store location to pull from. Options are [aws, ncep], by default aws
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

    - https://registry.opendata.aws/noaa-gfs-bdp-pds/
    - https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php
    """

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GFS forecast data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GFS lexicon.

        Returns
        -------
        xr.DataArray
            GFS weather data array
        """
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, lead_time, variable), timeout=self.async_timeout
            )
        )

        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GFS lexicon.

        Returns
        -------
        xr.DataArray
            GFS weather data array
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session()
        else:
            session = None

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with, compared to something seen in the
        # NCAR data source.
        xr_array = xr.DataArray(
            data=np.zeros(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(self.GFS_LAT),
                    len(self.GFS_LON),
                )
            ),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "lat": self.GFS_LAT,
                "lon": self.GFS_LON,
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching GFS data", disable=(not self._verbose)
        )

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client if s3fs
        # https://github.com/fsspec/s3fs/issues/943
        # https://github.com/zarr-developers/zarr-python/issues/2901
        if isinstance(self.fs, s3fs.S3FileSystem):
            await self.fs.set_session()  # Make sure the session was actually initalized
            s3fs.S3FileSystem.close_session(asyncio.get_event_loop(), self.fs.s3)

        return xr_array

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for GFS based on offline knowledge

        Parameters
        ----------
        lead_times : list[timedelta]
            list of lead times to fetch data
        """
        for delta in lead_times:
            if not delta.total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested lead time {delta} needs to be 1 hour interval for GFS"
                )
            # To update search "gfs." at https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
            # They are slowly adding more data
            hours = int(delta.total_seconds() // 3600)
            if hours > 384 or hours < 0:
                raise ValueError(
                    f"Requested lead time {delta} can only be a max of 384 hours for GFS"
                )
