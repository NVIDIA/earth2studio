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
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import gcsfs
import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.ftp import FTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon.hrrr import HRRRFXLexicon, HRRRLexicon, HRRRLexiconNew
from earth2studio.utils.imports import check_extra_imports
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


# No more herbie, we can be faster
class HRRRNew:
    """High-Resolution Rapid Refresh (HRRR) data source provides hourly North-American
    weather analysis data developed by NOAA (used to initialize the HRRR forecast
    model). This data source is provided on a Lambert conformal 3km grid at 1-hour
    intervals. The spatial dimensionality of HRRR data is [1059, 1799].

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
    max_workers : int, optional
        Maximum number of concurrent downloads, potentially not thread safe with Herbie,
        by default 1
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
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._cache = cache
        self._verbose = verbose

        # Silence cfgrib warning, TODO Remove this
        warnings.simplefilter(action="ignore", category=FutureWarning)

        if source == "aws":
            self.uri_prefix = "noaa-hrrr-bdp-pds"
            self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)

            # To update look at https://aws.amazon.com/marketplace/pp/prodview-yd5ydptv3vuz2#resources
            def _range(time: datetime) -> None:
                if time < datetime(year=2021, month=1, day=1):
                    raise ValueError(
                        f"Requested date time {time} needs to be after January 1st, 2021 for GFS on AWS"
                    )

            self._history_range = _range
        elif source == "google":
            self.uri_prefix = "high-resolution-rapid-refresh"
            self.fs = gcsfs.GCSFileSystem(
                cache_timeout=-1,
                token="anon",  # noqa: S106 # nosec B106
                access="read_only",
                block_size=8**20,
            )

            # To update look at https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
            def _range(time: datetime) -> None:
                if time < datetime(year=2014, month=7, day=30):
                    raise ValueError(
                        f"Requested date time {time} needs to be after July 30th, 2014 for HRRR on GCS"
                    )

            self._history_range = _range
        elif source == "azure":
            raise NotImplementedError(
                "Azure data source not implemented yet, open an issue if needed"
            )
        elif source == "nomads":
            # Could use http location, but using ftp since better for larger data
            # https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
            self.uri_prefix = "pub/data/nccf/com/hrrr/prod/"
            self.fs = FTPFileSystem(host="nomads.ncep.noaa.gov")

            def _range(time: datetime) -> None:
                if time + timedelta(days=2) < datetime.today():
                    raise ValueError(
                        f"Requested date time {time} needs to be within past 2 days for HRRR nomads source"
                    )

            self._history_range = _range
        else:
            raise ValueError(f"Invalid HRRR source {source}")

        self.async_timeout = async_timeout

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR initial data to be used for initial conditions for the given
        time, variable information, and optional history.

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
            GFS weather data array
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
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with
        xr_array = xr.DataArray(
            data=np.empty(
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

        return xr_array.isel(lead_time=0)

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
                        hrrr_name, modifier = HRRRLexiconNew[v]  # type: ignore
                        hrrr_name = hrrr_name.split("::") + [None, None]
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
                            if lt == 0:
                                hrrr_key = f"{hrrr_key}::anl"
                            else:
                                hrrr_key = f"{hrrr_key}::{lt.seconds // 60} min fcst"
                        if index and index.isnumeric():
                            hrrr_key = index
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
                        raise Exception()

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
        xr_array[*task.data_array_indices] = out

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
        logger.debug(f"Fetching GRS grib file: {grib_uri} {byte_offset}-{byte_length}")
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
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for HRRR"
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
        # Note we actually drop the last variable here because its easier (SBT114)
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

        # Pop place holder
        return index_table

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetches remote file into cache"""
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
        file_name = f"hrrr.{time.year}{time.month:0>2}{time.day:0>2}/hrrr.t{time.hour:0>2}z.wrfnatf00.grib2.idx"
        s3_uri = f"s3://{cls.HRRR_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


@check_extra_imports("data", ["herbie"])
class _HRRRBase:

    HRRR_BUCKET_NAME = "noaa-hrrr-bdp-pds"
    HRRR_X = np.arange(1799)
    HRRR_Y = np.arange(1059)
    MAX_BYTE_SIZE = 5000000

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        source: str = "aws",
        max_workers: int = 1,
    ):
        self._cache = cache
        self._verbose = verbose
        self._source = source
        self._max_workers = max_workers

        self.fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=2**20,
            client_kwargs={},
        )

        # Doesnt work with read block, only use with index file
        cache_options = {
            "cache_storage": self.cache,
            "expiry_time": 31622400,  # 1 year
        }
        self.fsc = WholeFileCacheFileSystem(fs=self.fs, **cache_options)

        # Initialize Herbie client
        try:
            from herbie import Herbie

            # Silence cfgrib warning
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.Herbie = Herbie
        except ImportError:
            raise ImportError(
                "Some data dependencies are missing (Herbie). Please install them using 'pip install earth2studio[data]'"
            )

    def fetch(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> xr.DataArray:
        """Function to retrieve HRRR forecast data into a single Xarray data array

        Parameters
        ----------
        time : list[datetime]
            Timestamps to return data for (UTC).
        lead_time: list[timedelta]
            List of forecast lead times to fetch
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR weather data array
        """

        hrrr_da = xr.DataArray(
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

        args = [
            (t, i, ld, j, v, k)
            for k, v in enumerate(variable)
            for j, ld in enumerate(lead_time)
            for i, t in enumerate(time)
        ]
        pbar = tqdm(
            total=len(args), desc="Fetching HRRR data", disable=(not self._verbose)
        )

        # Use ThreadPoolExecutor to parallelize Herbie calls
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures = []

            for t, i, ld, j, v, k in args:
                try:
                    # Set lexicon based on lead
                    lexicon: type[HRRRLexicon] | type[HRRRFXLexicon] = HRRRLexicon
                    if ld.total_seconds() != 0:
                        lexicon = HRRRFXLexicon
                    hrrr_str, modifier = lexicon[v]  # type: ignore
                    hrrr_class, hrrr_product, hrrr_level, hrrr_var = hrrr_str.split(
                        "::"
                    )
                except KeyError as e:
                    logger.error(f"variable id {v} not found in HRRR lexicon")
                    raise e

                # Submit the Herbie task to the thread pool
                future = executor.submit(
                    self._fetch_herbie_data,
                    t,
                    ld,
                    hrrr_class,
                    hrrr_level,
                    hrrr_var,
                    modifier,
                    (i, j, k),
                )
                futures.append((future, i, j, k))

            # Process completed futures as they finish
            for future in concurrent.futures.as_completed([f[0] for f in futures]):
                data, coords, indices = future.result()
                hrrr_da[tuple(indices)] = data

                # Add lat/lon coordinates if not present
                if "lat" not in hrrr_da.coords:
                    hrrr_da.coords["lat"] = coords["lat"]
                    hrrr_da.coords["lon"] = coords["lon"]
                pbar.update(1)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return hrrr_da

    def _fetch_herbie_data(
        self,
        time: datetime,
        lead_time: timedelta,
        hrrr_class: str,
        hrrr_level: str,
        hrrr_var: str,
        modifier: Callable,
        indices: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict, tuple[int, int, int]]:
        """Helper method to fetch data using Herbie"""
        # Create Herbie object for this timestamp and forecast
        H = self.Herbie(
            date=time,
            model="hrrr",
            product=hrrr_class,
            fxx=int(lead_time.total_seconds() // 3600),
            priority=[self._source],
            verbose=False,
            save_dir=self.cache,
        )

        # Construct search string for the variable
        if hrrr_level == "surface":
            search_term = f"{hrrr_var}:surface"
        else:
            search_term = f"{hrrr_var}:{hrrr_level}"

        # Read the data using Herbie
        # Keep grib files cached
        data = H.xarray(search_term, remove_grib=False)
        data = list(data.values())[0]

        # Return both the modified data and the coordinates
        coords = {
            "lat": (["hrrr_y", "hrrr_x"], data.coords["latitude"].values),
            "lon": (["hrrr_y", "hrrr_x"], data.coords["longitude"].values),
        }

        return modifier(data.values), coords, indices

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for HRRR

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
            # sfc goes back to 2016 for anl, limit based on pressure
            # frst starts on on the same date pressure starts
            if time < datetime(year=2018, month=7, day=12, hour=13):
                raise ValueError(
                    f"Requested date time {time} needs to be after July 12th, 2018 13:00 for HRRR"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "hrrr")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_hrrr")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the HRRR store

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

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        fs = s3fs.S3FileSystem(anon=True)

        # Object store directory for given date
        date_group = time.strftime("%Y%m%d")
        forcast_hour = time.strftime("%H")
        s3_uri = f"{cls.HRRR_BUCKET_NAME}/hrrr.{date_group}/conus/hrrr.t{forcast_hour}z.wrfsfcf00.grib2.idx"

        return fs.exists(s3_uri)


# Leaving in here to dream, no natural levels in this data source
# class _HRRRZarrBase(_HRRRBase):
#     # Not used but keeping here for now for future reference
#     HRRR_BUCKET_NAME = "hrrrzarr"
#     HRRR_X = np.arange(1799)
#     HRRR_Y = np.arange(1059)

#     def __init__(
#         self,
#         lexicon: HRRRLexicon | HRRRFXLexicon,
#         cache: bool = True,
#         verbose: bool = True,
#     ):
#         self._cache = cache
#         self._lexicon = lexicon
#         self._verbose = verbose

#         fs = s3fs.S3FileSystem(
#             anon=True,
#             default_block_size=2**20,
#             client_kwargs={},
#         )

#         if self._cache:
#             cache_options = {
#                 "cache_storage": self.cache,
#                 "expiry_time": 31622400,  # 1 year
#             }
#             fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

#         fs_map = fsspec.FSMap(f"s3://{self.HRRR_BUCKET_NAME}", fs)
#         self.zarr_group = zarr.open(fs_map, mode="r")

#     async def async_fetch(
#         self,
#         time: list[datetime],
#         lead_time: list[timedelta],
#         variable: list[str],
#     ) -> xr.DataArray:
#         """Async function to retrieve HRRR forecast data into a single Xarray data array

#         Parameters
#         ----------
#         time : list[datetime]
#             Timestamps to return data for (UTC).
#         lead_time: list[timedelta]
#             List of forecast lead times to fetch
#         variable : str | list[str] | VariableArray
#             String, list of strings or array of strings that refer to variables to
#             return. Must be in the HRRR lexicon.

#         Returns
#         -------
#         xr.DataArray
#             HRRR weather data array
#         """

#         hrrr_da = xr.DataArray(
#             data=np.empty(
#                 (
#                     len(time),
#                     len(lead_time),
#                     len(variable),
#                     len(self.HRRR_Y),
#                     len(self.HRRR_X),
#                 )
#             ),
#             dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
#             coords={
#                 "time": time,
#                 "lead_time": lead_time,
#                 "variable": variable,
#                 "hrrr_x": self.HRRR_X,
#                 "hrrr_y": self.HRRR_Y,
#                 "lat": (
#                     ["hrrr_y", "hrrr_x"],
#                     self.zarr_group["grid"]["HRRR_chunk_index.zarr"]["latitude"][:],
#                 ),
#                 "lon": (
#                     ["hrrr_y", "hrrr_x"],
#                     self.zarr_group["grid"]["HRRR_chunk_index.zarr"]["longitude"][:]
#                     + 360,
#                 ),  # Change to [0,360]
#             },
#         )
#         # Banking on async calls in zarr 3.0
#         for i, t in enumerate(time):
#             for j, ld in enumerate(lead_time):
#                 # Set lexicon based on lead
#                 lexicon: type[HRRRLexicon] | type[HRRRFXLexicon] = HRRRLexicon
#                 if ld.total_seconds() != 0:
#                     lexicon = HRRRFXLexicon
#                 for k, v in enumerate(variable):
#                     try:
#                         hrrr_str, modifier = lexicon[v]
#                         hrrr_class, hrrr_product, hrrr_level, hrrr_var = hrrr_str.split(
#                             "::"
#                         )
#                     except KeyError as e:
#                         logger.error(f"variable id {v} not found in HRRR lexicon")
#                         raise e

#                     date_group = t.strftime("%Y%m%d")
#                     time_group = t.strftime(f"%Y%m%d_%Hz_{hrrr_product}.zarr")

#                     logger.debug(
#                         f"Fetching HRRR {hrrr_product} variable {v} at {t.isoformat()}"
#                     )

#                     data = self.zarr_group[hrrr_class][date_group][time_group][
#                         hrrr_level
#                     ][hrrr_var][hrrr_level][hrrr_var]
#                     if hrrr_product == "fcst":
#                         # Minus 1 here because index 0 is forecast with leadtime 1hr
#                         # forecast_period coordinate system tells what the lead times are in hours
#                         lead_index = int(ld.total_seconds() // 3600) - 1
#                         data = data[lead_index]

#                     hrrr_da[i, j, k] = modifier(data)

#         # Delete cache if needed
#         if not self._cache:
#             shutil.rmtree(self.cache)

#         return hrrr_da


class HRRR(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) data source provides hourly North-American
    weather analysis data developed by NOAA (used to initialize the HRRR forecast
    model). This data source is provided on a Lambert conformal 3km grid at 1-hour
    intervals. The spatial dimensionality of HRRR data is [1059, 1799]. This data source
    pulls data from the HRRR zarr bucket on S3.

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
    max_workers : int, optional
        Maximum number of concurrent downloads, potentially not thread safe with Herbie,
        by default 1
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

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://hrrrzarr.s3.amazonaws.com/index.html
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __init__(
        self,
        source: str = "aws",
        max_workers: int = 1,
        cache: bool = True,
        verbose: bool = True,
    ):
        super().__init__(cache, verbose, source, max_workers)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR initial data to be used for initial conditions for the given
        time, variable information, and optional history.

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
            HRRR analysis data array
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        data_array = self.fetch(time, [timedelta(hours=0)], variable)
        return data_array.isel(lead_time=0).drop_vars("lead_time")


class HRRR_FX(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) forecast source provides a North-American
    weather forecasts with hourly forecast runs developed by NOAA. This forecast source
    has hourly forecast steps up to a lead time of 48 hours. Data is provided on a
    Lambert conformal 3km grid at 1-hour intervals. The spatial dimensionality of HRRR
    data is [1059, 1799]. This data source pulls data from the HRRR zarr bucket on S3.

    Parameters
    ----------
    source : str, optional
        Data source to use ('aws', 'google', 'azure', 'nomads'), by default 'aws'
    max_workers : int, optional
        Maximum number of concurrent downloads, by default 4
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
        max_workers: int = 4,
        cache: bool = True,
        verbose: bool = True,
    ):
        super().__init__(cache, verbose, source, max_workers)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR forecast data.

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
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)
        return self.fetch(time, lead_time, variable)

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
