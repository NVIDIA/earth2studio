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

import gcsfs
import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

try:
    import pyproj
except ImportError:
    pyproj = None

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import HRRRFXLexicon, HRRRLexicon
from earth2studio.utils import check_extra_imports
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class HRRRAsyncTask:
    """Small helper struct for Async tasks"""

    data_array_indices: tuple[int, int, int]
    hrrr_file_uri: str
    hrrr_byte_offset: int
    hrrr_byte_length: int
    hrrr_modifier: Callable


@check_extra_imports("data", [pyproj])
class HRRR:
    """High-Resolution Rapid Refresh (HRRR) data source provides hourly North-American
    weather analysis data developed by NOAA (used to initialize the HRRR forecast
    model). This data source is provided on a Lambert conformal 3km grid at 1-hour
    intervals. The spatial dimensionality of HRRR data is [1059, 1799].

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
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

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://aws.amazon.com/marketplace/pp/prodview-yd5ydptv3vuz2#resources
    - https://hrrrzarr.s3.amazonaws.com/index.html
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    HRRR_BUCKET_NAME = "noaa-hrrr-bdp-pds"
    MAX_BYTE_SIZE = 5000000

    HRRR_X = np.arange(1799)
    HRRR_Y = np.arange(1059)

    def __init__(
        self,
        source: str = "aws",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._source = source
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers

        self.lexicon = HRRRLexicon
        self.async_timeout = async_timeout

        if self._source == "aws":
            self.uri_prefix = "noaa-hrrr-bdp-pds"

            # To update look at https://aws.amazon.com/marketplace/pp/prodview-yd5ydptv3vuz2#resources
            def _range(time: datetime) -> None:
                # sfc goes back to 2016 for anl, limit based on pressure
                # frst starts on on the same date pressure starts
                if time < datetime(year=2018, month=7, day=12, hour=13):
                    raise ValueError(
                        f"Requested date time {time} needs to be after July 12th, 2018 13:00 for HRRR"
                    )

            self._history_range = _range
        elif self._source == "google":
            self.uri_prefix = "high-resolution-rapid-refresh"

            # To update look at https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
            # Needs confirmation
            def _range(time: datetime) -> None:
                # sfc goes back to 2016 for anl, limit based on pressure
                # frst starts on on the same date pressure starts
                if time < datetime(year=2018, month=7, day=12, hour=13):
                    raise ValueError(
                        f"Requested date time {time} needs to be after July 12th, 2018 13:00 for HRRR"
                    )

            self._history_range = _range
        elif self._source == "azure":
            raise NotImplementedError(
                "Azure data source not implemented yet, open an issue if needed"
            )
        elif self._source == "nomads":
            # https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
            self.uri_prefix = (
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/"
            )

            def _range(time: datetime) -> None:
                if time + timedelta(days=2) < datetime.today():
                    raise ValueError(
                        f"Requested date time {time} needs to be within past 2 days for HRRR nomads source"
                    )

            self._history_range = _range
        else:
            raise ValueError(f"Invalid HRRR source { self._source}")

        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            # Else we assume that async calls will be used which in that case
            # we will init the group in the call function when we have the loop
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of fsspec file stores

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        if self._source == "aws":
            self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)
        elif self._source == "google":
            self.fs = gcsfs.GCSFileSystem(
                cache_timeout=-1,
                token="anon",  # noqa: S106 # nosec B106
                access="read_only",
                block_size=8**20,
            )
        elif self._source == "azure":
            raise NotImplementedError(
                "Azure data source not implemented yet, open an issue if needed"
            )
        elif self._source == "nomads":
            # HTTP file system, tried FTP but didnt work
            self.fs = HTTPFileSystem(asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR analysis data (lead time 0)

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR weather data array
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
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR weather data array
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

        # Generate HRRR lat-lon grid to append onto data array
        lat, lon = self.grid()
        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with
        xr_array = xr.DataArray(
            data=np.zeros(
                (
                    len(time),
                    1,
                    len(variable),
                    len(self.HRRR_Y),
                    len(self.HRRR_X),
                )
            ),
            dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
            coords={
                "time": time,
                "lead_time": [timedelta(hours=0)],
                "variable": variable,
                "hrrr_x": self.HRRR_X,
                "hrrr_y": self.HRRR_Y,
                "lat": (("hrrr_y", "hrrr_x"), lat),
                "lon": (("hrrr_y", "hrrr_x"), lon),
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, [timedelta(hours=0)], variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching HRRR data", disable=(not self._verbose)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        xr_array = xr_array.isel(lead_time=0)
        del xr_array.coords["lead_time"]
        return xr_array

    async def _create_tasks(
        self, time: list[datetime], lead_time: list[timedelta], variable: list[str]
    ) -> list[HRRRAsyncTask]:
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
        tasks: list[HRRRAsyncTask] = []  # group pressure-level variables

        # Start with fetching all index files for each time / lead time
        # TODO: Update so only needed products (can tell from parsing variables), for
        # now fetch all index files because its cheap and easier
        products = ["wrfsfc", "wrfprs", "wrfnat"]
        args = [
            self._grib_index_uri(t, lt, p)
            for t in time
            for lt in lead_time
            for p in products
        ]
        func_map = map(self._fetch_index, args)
        results = await tqdm.gather(
            *func_map, desc="Fetching HRRR index files", disable=True
        )
        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                # Get index file dictionaries for each possible product
                index_files = {p: results.pop(0) for p in products}
                for k, v in enumerate(variable):
                    try:
                        hrrr_name_str, modifier = self.lexicon[v]  # type: ignore
                        hrrr_name = hrrr_name_str.split("::") + ["", ""]
                        product = hrrr_name[0]
                        variable_name = hrrr_name[1]
                        level = hrrr_name[2]
                        forecastvld = hrrr_name[3]
                        index = hrrr_name[4]

                        # Create index key to find byte range
                        hrrr_key = f"{variable_name}::{level}"
                        if forecastvld:
                            hrrr_key = f"{hrrr_key}::{forecastvld}"
                        else:
                            if lt.total_seconds() == 0:
                                hrrr_key = f"{hrrr_key}::anl"
                            else:
                                hrrr_key = f"{hrrr_key}::{int(lt.total_seconds() // 3600):d} hour fcst"
                        if index and index.isnumeric():
                            hrrr_key = index

                        # Special cases
                        # could do this better with templates, but this is single instance
                        if variable_name == "APCP":
                            hours = int(lt.total_seconds() // 3600)
                            hrrr_key = f"{variable_name}::{level}::{hours-1:d}-{hours:d} hour acc fcst"

                    except KeyError as e:
                        logger.error(
                            f"variable id {variable} not found in HRRR lexicon"
                        )
                        raise e

                    # Get byte range from index
                    byte_offset = None
                    byte_length = None
                    for key, value in index_files[product].items():
                        if hrrr_key in key:
                            byte_offset = value[0]
                            byte_length = value[1]
                            break

                    if byte_length is None or byte_offset is None:
                        logger.warning(
                            f"Variable {v} not found in index file for time {t} at {lt}, values will be unset"
                        )
                        continue

                    tasks.append(
                        HRRRAsyncTask(
                            data_array_indices=(i, j, k),
                            hrrr_file_uri=self._grib_uri(t, lt, product),
                            hrrr_byte_offset=byte_offset,
                            hrrr_byte_length=byte_length,
                            hrrr_modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: HRRRAsyncTask,
        xr_array: xr.DataArray,
    ) -> xr.DataArray:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(
            task.hrrr_file_uri,
            task.hrrr_byte_offset,
            task.hrrr_byte_length,
            task.hrrr_modifier,
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
        """Fetch HRRR data array. This will first fetch the index file to get byte range
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
        logger.debug(f"Fetching HRRR grib file: {grib_uri} {byte_offset}-{byte_length}")
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
        """Verify if date time is valid for HRRR based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for HRRR"
                )
            # Check history range for given source
            self._history_range(time)

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int]]:
        """Fetch HRRR atmospheric index file

        Parameters
        ----------
        index_uri : str
            URI to grib index file to download

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of HRRR vairables (byte offset, byte length)
        """
        # Grab index file
        index_file = await self._fetch_remote_file(index_uri)
        with open(index_file) as file:
            index_lines = [line.rstrip() for line in file]

        index_table = {}
        # Note we actually drop the last variable here because its easier (SBT114)
        # GEFS has a solution for this if needed that involves appending a dummy line
        # Example of row: "1:0:d=2021111823:REFC:entire atmosphere:795 min fcst:"
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
        self, time: datetime, lead_time: timedelta, product: str = "wrfsfc"
    ) -> str:
        """Generates the URI for HRRR grib files"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"hrrr.{time.year}{time.month:0>2}{time.day:0>2}/conus/"
        file_name = os.path.join(
            file_name, f"hrrr.t{time.hour:0>2}z.{product}f{lead_hour:02d}.grib2"
        )
        return os.path.join(self.uri_prefix, file_name)

    def _grib_index_uri(
        self, time: datetime, lead_time: timedelta, product: str
    ) -> str:
        """Generates the URI for HRRR index grib files"""
        # https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"hrrr.{time.year}{time.month:0>2}{time.day:0>2}/conus/"
        file_name = os.path.join(
            file_name, f"hrrr.t{time.hour:0>2}z.{product}f{lead_hour:02d}.grib2.idx"
        )
        return os.path.join(self.uri_prefix, file_name)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "hrrr")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_hrrr")
        return cache_location

    @classmethod
    def grid(cls) -> tuple[np.array, np.array]:
        """Generates the HRRR lambert conformal projection grid coordinates. Creates the
        HRRR grid using single parallel lambert conformal mapping

        Note
        ----
        For more information about the HRRR grid see:

        - https://ntrs.nasa.gov/api/citations/20160009371/downloads/20160009371.pdf

        Returns
        -------
        Returns:
            tuple: (lat, lon) in degrees
        """
        # a, b is radius of globe 6371229
        p1 = pyproj.CRS(
            "proj=lcc lon_0=262.5 lat_0=38.5 lat_1=38.5 lat_2=38.5 a=6371229 b=6371229"
        )
        p2 = pyproj.CRS("latlon")
        transformer = pyproj.Transformer.from_proj(p2, p1)
        itransformer = pyproj.Transformer.from_proj(p1, p2)

        # Start with getting grid bounds based on lat / lon box (SW-NW-NE-SE)
        # Reference seems a bit incorrect from the actual data, grabbed from S3 HRRR gribs
        # Perhaps cell points? IDK
        lat = np.array(
            [21.138123, 47.83862349881542, 47.84219502248866, 21.140546625419148]
        )
        lon = np.array(
            [237.280472, 225.90452026573686, 299.0828072281622, 287.71028150897075]
        )

        easting, northing = transformer.transform(lat, lon)
        E, N = np.meshgrid(
            np.linspace(easting[0], easting[2], 1799),
            np.linspace(northing[0], northing[1], 1059),
        )
        lat, lon = itransformer.transform(E, N)
        lon = np.where(lon < 0, lon + 360, lon)
        return lat, lon

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the HRRR object store. Uses S3
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
        file_name = f"hrrr.{time.year}{time.month:0>2}{time.day:0>2}/conus"
        file_name = f"{file_name}/hrrr.t{time.hour:0>2}z.wrfnatf00.grib2.idx"
        s3_uri = f"s3://{cls.HRRR_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class HRRR_FX(HRRR):
    """High-Resolution Rapid Refresh (HRRR) forecast source provides a North-American
    weather forecasts with hourly forecast runs developed by NOAA. This forecast source
    has hourly forecast steps up to a lead time of 48 hours. Data is provided on a
    Lambert conformal 3km grid at 1-hour intervals. The spatial dimensionality of HRRR
    data is [1059, 1799].

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
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

    Note
    ----
    48 hour forecasts are provided on 6 hour intervals. 18 hour forecasts are generated
    hourly.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __init__(
        self,
        source: str = "aws",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            source=source,
            max_workers=max_workers,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )
        self.lexicon = HRRRFXLexicon  # type: ignore

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR forecast data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR forecast data array
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
            return. Must be in the HRRR FX lexicon.

        Returns
        -------
        xr.DataArray
            HRRR forecast data array
        """
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
                    len(self.HRRR_Y),
                    len(self.HRRR_X),
                )
            ),
            dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
            coords={
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "hrrr_x": self.HRRR_X,
                "hrrr_y": self.HRRR_Y,
            },
        )

        async_tasks = []
        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching HRRR data", disable=(not self._verbose)
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return xr_array

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for HRRR based on offline knowledge

        Parameters
        ----------
        lead_times : list[timedelta]
            list of lead times to fetch data
        """
        for delta in lead_times:
            if not delta.total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested lead time {delta} needs to be 1 hour interval for HRRR"
                )
            hours = int(delta.total_seconds() // 3600)
            # Note, one forecasts every 6 hours have 2 day lead times, others only have 18 hours
            if hours > 48 or hours < 0:
                raise ValueError(
                    f"Requested lead time {delta} can only be between [0,48] hours for HRRR forecast"
                )
