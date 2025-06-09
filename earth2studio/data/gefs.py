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
import concurrent.futures
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
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_forecast_inputs,
)
from earth2studio.lexicon import GEFSLexicon, GEFSLexiconSel
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class GEFSAsyncTask:
    """Small helper struct for Async tasks"""

    data_array_indices: tuple[int, int, int]
    gefs_file_uri: str
    gefs_byte_offset: int
    gefs_byte_length: int | None
    gefs_modifier: Callable


class GEFS_FX:
    """The Global Ensemble Forecast System (GEFS) forecast source is a 30 member
    ensemble forecast provided on an 0.5 degree equirectangular grid.  GEFS is a weather
    forecast model developed by  National Centers for Environmental Prediction (NCEP).
    This forecast source has data at 6-hour intervals spanning from Sept 23rd 2020 to
    present date. Each forecast provides 3-hourly predictions up to 10 days (240 hours)
    and 6 hourly predictions for another 6 days (384 hours).

    Parameters
    ----------
    member : str, optional
        GEFS member. Options are: control gec00 (control), gepNN (forecast member NN,
        e.g. gep01, gep02,...), by default "gec00"
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
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

    Warning
    -------
    Some variables in the GEFS lexicon may not be available at lead time 0. Consult GEFS
    documentation for additional information.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://registry.opendata.aws/noaa-gefs/
    - https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/
    """

    GEFS_BUCKET_NAME = "noaa-gefs-pds"
    MAX_BYTE_SIZE = 5000000

    GEFS_LAT = np.linspace(90, -90, 361)
    GEFS_LON = np.linspace(0, 359.5, 720)

    GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    GEFS_CHECK_PRODUCT = "pgrb2bp5"

    def __init__(
        self,
        member: str = "gec00",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers

        if member not in self.GEFS_MEMBERS:
            raise ValueError(f"Invalid GEFS member {member}")

        self._product_class = "atmos"
        self._product_resolution = "0p50"  # 0p50 or 0p25
        self._member = member
        self.async_timeout = async_timeout
        self.lexicon = GEFSLexicon

        # Check to see if there is a running loop (initialized in async)
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            # Else we assume that async calls will be used which in that case
            # we will init the group in the call function when we have the loop
            self.fs = None

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
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GEFS forecast data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GEFS lexicon.

        Returns
        -------
        xr.DataArray
            GEFS forecast data array
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

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, lead_time, variable), timeout=self.async_timeout
            )
        )

        return xr_array

    async def fetch(
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
            return. Must be in the GEFS FX lexicon.

        Returns
        -------
        xr.DataArray
            GEFS forecast data array
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
            data=np.empty(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(self.GEFS_LAT),
                    len(self.GEFS_LON),
                )
            ),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "lat": self.GEFS_LAT,
                "lon": self.GEFS_LON,
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching GEFS data", disable=(not self._verbose)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return xr_array

    async def _create_tasks(
        self, time: list[datetime], lead_time: list[timedelta], variable: list[str]
    ) -> list[GEFSAsyncTask]:
        """Create download tasks, each corresponding to one grib byte range

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
        tasks: list[GEFSAsyncTask] = []  # group pressure-level variables

        # Start with fetching all index files for each time / lead time
        products = set()
        for v in variable:
            gefs_name_str, _ = self.lexicon[v]  # type: ignore
            products.add(gefs_name_str.split("::")[0])
        args = [
            self._grib_index_uri(t, lt, p)
            for t in time
            for lt in lead_time
            for p in products
        ]
        func_map = map(self._fetch_index, args)
        results = await tqdm.gather(
            *func_map, desc="Fetching GEFS index files", disable=True
        )
        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                # Get index file dictionaries for each possible product
                index_files = {p: results.pop(0) for p in products}
                for k, v in enumerate(variable):
                    try:
                        gefs_name_str, modifier = self.lexicon[v]  # type: ignore
                        gefs_name = gefs_name_str.split("::") + ["", ""]
                        product = gefs_name[0]
                        variable_name = gefs_name[1]
                        level = gefs_name[2]

                        # Create index key to find byte range
                        gefs_key = f"{variable_name}::{level}"

                    except KeyError as e:
                        logger.error(
                            f"variable id {variable} not found in GEFS lexicon"
                        )
                        raise e

                    # Get byte range from index
                    byte_offset = None
                    byte_length = None
                    for key, value in index_files[product].items():
                        if gefs_key in key:
                            byte_offset = value[0]
                            byte_length = (
                                None if value[1] < 0 else value[1]
                            )  # Negative means to eof
                            break

                    if byte_offset is None:
                        logger.warning(
                            f"Variable {v} not found in index file for time {t} at {lt}, values will be unset"
                        )
                        continue

                    tasks.append(
                        GEFSAsyncTask(
                            data_array_indices=(i, j, k),
                            gefs_file_uri=self._grib_uri(t, lt, product),
                            gefs_byte_offset=byte_offset,
                            gefs_byte_length=byte_length,
                            gefs_modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: GEFSAsyncTask,
        xr_array: xr.DataArray,
    ) -> xr.DataArray:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(
            task.gefs_file_uri,
            task.gefs_byte_offset,
            task.gefs_byte_length,
            task.gefs_modifier,
        )
        i, j, k = task.data_array_indices
        xr_array[i, j, k] = out

    async def fetch_array(
        self,
        grib_uri: str,
        byte_offset: int,
        byte_length: int | None,
        modifier: Callable,
    ) -> np.ndarray:
        """Fetch GEFS data array. This will first fetch the index file to get byte range
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
        logger.debug(f"Fetching GEFS grib file: {grib_uri} {byte_offset}-{byte_length}")
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
        """Verify if date time is valid for GEFS based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for GEFS"
                )
            # Brute forces checked this, older dates missing data
            # Open an issue if incorrect, may of gotten updated
            if time < datetime(year=2020, month=9, day=23):
                raise ValueError(
                    f"Requested date time {time} needs to be after Sept 23rd, 2020 for GEFS"
                )

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for GEFS based on offline knowledge"""
        for delta in lead_times:
            # To update search "gefs." at https://noaa-gefs-pds.s3.amazonaws.com/index.html
            hours = int(delta.total_seconds() // 3600)
            if hours > 384 or hours < 0:
                raise ValueError(
                    f"Requested lead time {delta} can only be a max of 384 hours for GEFS"
                )

            # 3-hours supported for first 10 days
            if delta.total_seconds() // 3600 <= 240:
                if not delta.total_seconds() % 10800 == 0:
                    raise ValueError(
                        f"Requested lead time {delta} needs to be 3 hour interval for first 10 days in GEFS"
                    )
            # 6 hours for rest
            else:
                if not delta.total_seconds() % 21600 == 0:
                    raise ValueError(
                        f"Requested lead time {delta} needs to be 6 hour interval for last 6 days in GEFS"
                    )

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int]]:
        """Fetch GEFS atmospheric index file

        Parameters
        ----------
        index_uri : str
            URI to grib index file to download

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GEFS vairables (byte offset, byte length)
        """
        # Grab index file
        index_file = await self._fetch_remote_file(index_uri)
        with open(index_file) as file:
            index_lines = [line.rstrip() for line in file]
        # Add dummy variable at end of file with max offset so algo below works
        index_lines.append("xx:-1:d=xx:NULL:NULL:NULL:NULL")
        index_table: dict[str, tuple[int, int]] = {}

        index_table = {}
        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[0]}::{lsplit[3]}::{lsplit[4]}::{lsplit[5]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

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

    def _grib_uri(
        self, time: datetime, lead_time: timedelta, product: str = "pgrb2a"
    ) -> str:
        """Generates the URI for GEFS grib files"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        p_res = self._product_resolution.replace("0", "")
        file_name = os.path.join(file_name, f"{self._product_class}/{product}{p_res}")
        file_name = os.path.join(
            file_name,
            f"{self._member}.t{time.hour:0>2}z.{product}.{self._product_resolution}.f{lead_hour:03d}",
        )
        return os.path.join(self.GEFS_BUCKET_NAME, file_name)

    def _grib_index_uri(
        self, time: datetime, lead_time: timedelta, product: str = "pgrb2a"
    ) -> str:
        """Generates the URI for GEFS index grib files"""
        # https://noaa-gefs-pds.s3.amazonaws.com/index.html#gefs.20250513/00/atmos/pgrb2ap5/
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        p_res = self._product_resolution.replace("0", "")
        file_name = os.path.join(file_name, f"{self._product_class}/{product}{p_res}")
        file_name = os.path.join(
            file_name,
            f"{self._member}.t{time.hour:0>2}z.{product}.{self._product_resolution}.f{lead_hour:03d}.idx",
        )
        return os.path.join(self.GEFS_BUCKET_NAME, file_name)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gefs")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_gefs")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the GEFS object store. Uses S3
        store

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
        # Just picking the first variable to look for
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/atmos/{cls.GEFS_CHECK_PRODUCT}/"
        s3_uri = f"s3://{cls.GEFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class GEFS_FX_721x1440(GEFS_FX):
    """The Global Ensemble Forecast System (GEFS) forecast source is a 30 member
    ensemble forecast provided on an 0.25 degree equirectangular grid. GEFS is a
    weather forecast model developed by  National Centers for Environmental Prediction
    (NCEP). This data source provides the select variables of GEFS served on a higher
    resolution grid t 6-hour intervals spanning from Sept 23rd 2020 to present date.
    Each forecast provides 3-hourly predictions up to 10 days (240 hours) and 6 hourly
    predictions for another 6 days (384 hours).

    Parameters
    ----------
    member : str, optional
        GEFS member. Options are: control gec00 (control), gepNN (forecast member NN,
        e.g. gep01, gep02,...), by default "gec00"
    max_workers : int, optional
        Max works in async io thread pool. Only applied when using sync call function
        and will modify the default async loop if one exists, by default 24
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

    Warning
    -------
    Some variables in the GEFS lexicon may not be available at lead time 0. Consult GEFS
    documentation for additional information.

    Note
    ----
    NCEP only provides a small subset of variables on the higher resoluton 0.25 degree
    grid. For a larger selection, use the standard GEFS data source.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://registry.opendata.aws/noaa-gefs/
    - https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast
    - https://www.nco.ncep.noaa.gov/pmb/products/gens/
    """

    GEFS_LAT = np.linspace(90, -90, 721)
    GEFS_LON = np.linspace(0, 359.75, 1440)

    GEFS_CHECK_PRODUCT = "pgrb2sp25"

    def __init__(
        self,
        member: str = "gec00",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            member=member,
            max_workers=max_workers,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )
        self._product_resolution = "0p25"
        self.lexicon = GEFSLexiconSel  # type: ignore

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GEFS forecast data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GEFS FX select lexicon.

        Returns
        -------
        xr.DataArray
            GEFS select forecast data array
        """
        return super().__call__(time, lead_time, variable)

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for GEFS based on offline knowledge"""
        for delta in lead_times:
            # To update search "gefs." at https://noaa-gefs-pds.s3.amazonaws.com/index.html
            hours = int(delta.total_seconds() // 3600)
            if hours > 240 or hours < 0:
                raise ValueError(
                    f"Requested lead time {delta} can only be a max of 240 hours for GEFS 0.25 degree data"
                )

            # 3-hours supported for first 10 days
            if delta.total_seconds() // 3600 <= 240:
                if not delta.total_seconds() % 10800 == 0:
                    raise ValueError(
                        f"Requested lead time {delta} needs to be 3 hour interval for first 10 days in GEFS 0.25 degree data"
                    )
