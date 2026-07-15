# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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

import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import obstore as obs
import pygrib
import xarray as xr
from loguru import logger
from obstore.store import ObjectStore
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    cancellable_to_thread,
    datasource_cache_root,
    gather_with_concurrency,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_forecast_inputs,
    resolve_async_workers,
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
        Deprecated, use async_workers instead. Kept for API compatibility, by
        default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent downloads. By default None, which autoscales
        to the number of download tasks (capped at 32)
    retries : int, optional
        Number of retries for each download task on transient errors, by default 3

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

    Badges
    ------
    region:global dataclass:simulation product:wind product:precip product:temp product:atmos
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
        async_workers: int | None = None,
        retries: int = 3,
    ):
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None

        if member not in self.GEFS_MEMBERS:
            raise ValueError(f"Invalid GEFS member {member}")

        self._product_class = "atmos"
        self._product_resolution = "0p50"  # 0p50 or 0p25
        self._member = member
        self.async_timeout = async_timeout
        self.lexicon = GEFSLexicon

        self.store: ObjectStore | None = None

    async def _async_init(self) -> None:
        """Async initialization of the object store

        Note
        ----
        Unlike async fsspec filesystems, obstore stores are event-loop
        independent and could be built in ``__init__``; kept as a lazy async
        method to preserve the initialization seam.
        """
        self.store = obstore_store_from_url(
            f"s3://{self.GEFS_BUCKET_NAME}",
            max_pool_connections=self._async_workers or 32,
        )

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
        try:
            xr_array = _sync_async(
                self.fetch, time, lead_time, variable, timeout=self.async_timeout
            )
        finally:
            # Delete cache if needed
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

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
        # Lazily initialize the object store on first use
        if self.store is None:
            await self._async_init()

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)

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

        async_tasks = await self._create_tasks(time, lead_time, variable)
        coros = [self.fetch_wrapper(task, xr_array=xr_array) for task in async_tasks]

        await gather_with_concurrency(
            coros,
            max_workers=resolve_async_workers(self._async_workers, len(coros)),
            task_timeout=120.0,
            desc="Fetching GEFS data",
            verbose=(not self._verbose),
        )

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
        results = await gather_with_concurrency(
            [self._fetch_index(uri) for uri in args],
            max_workers=resolve_async_workers(self._async_workers, len(args)),
            task_timeout=60.0,
            desc="Fetching GEFS index files",
            verbose=True,
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

                    # Get byte range from index. A byte length of None means
                    # the record runs to the end of the file (read to EOF).
                    byte_offset = None
                    byte_length = None
                    for key, value in index_files[product].items():
                        if gefs_key in key:
                            byte_offset = value[0]
                            byte_length = value[1]
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
        out = await async_retry(
            self.fetch_array,
            task.gefs_file_uri,
            task.gefs_byte_offset,
            task.gefs_byte_length,
            task.gefs_modifier,
            retries=self._retries,
            backoff=1.0,
            task_timeout=60.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
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
        np.ndarray
            GEFS array for given time and lead time
        """
        logger.debug(f"Fetching GEFS grib file: {grib_uri} {byte_offset}-{byte_length}")
        # Download the grib file to cache
        grib_file = await self._fetch_remote_file(
            grib_uri,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )
        # pygrib decode is blocking and GIL-bound; run in a thread with timeout
        values = await cancellable_to_thread(_decode_gefs_grib, grib_file, timeout=30.0)
        return modifier(values)

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

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int | None]]:
        """Fetch GEFS atmospheric index file

        Parameters
        ----------
        index_uri : str
            URI to grib index file to download

        Returns
        -------
        dict[str, tuple[int, int | None]]
            Dictionary of GEFS variables (byte offset, byte length). The last
            record has a byte length of ``None``, meaning read to end of file.
        """
        # Grab index file
        index_file = await self._fetch_remote_file(index_uri)
        with open(index_file) as file:
            records = [
                lsplit for line in file if len(lsplit := line.rstrip().split(":")) >= 7
            ]

        index_table: dict[str, tuple[int, int | None]] = {}
        for i, lsplit in enumerate(records):
            byte_offset = int(lsplit[1])
            # The final record runs to the end of the file (no next offset to
            # bound it), signalled with a byte length of None (read to EOF).
            byte_length: int | None
            if i + 1 < len(records):
                byte_length = int(records[i + 1][1]) - byte_offset
            else:
                byte_length = None
            key = f"{lsplit[0]}::{lsplit[3]}::{lsplit[4]}::{lsplit[5]}"
            if byte_length is not None and byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        return index_table

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetches remote file into cache"""
        if self.store is None:
            raise ValueError("Object store is not initialized")

        # Hash the bucket-prefixed path (not the store-relative key) so warm
        # caches populated before the obstore migration remain valid
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        key = path.removeprefix(self.GEFS_BUCKET_NAME + "/")
        return await obstore_fetch_to_cache(
            self.store,
            key,
            self.cache,
            byte_offset=byte_offset,
            byte_length=byte_length,
            cache_key=filename,
        )

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
            if self._tmp_cache_hash is None:
                # First access for temp cache: create a random suffix to avoid collisions
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gefs_{self._tmp_cache_hash}"
            )
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

        store = obstore_store_from_url(f"s3://{cls.GEFS_BUCKET_NAME}")
        # Object store directory for given time
        # Just picking the first product to look for
        prefix = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/atmos/{cls.GEFS_CHECK_PRODUCT}/"
        # obs.list yields chunks (lists) of entries; one entry proves existence
        chunk: list = next(iter(obs.list(store, prefix=prefix, chunk_size=1)), [])

        return len(chunk) > 0


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
        Deprecated, use async_workers instead. Kept for API compatibility, by
        default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent downloads. By default None, which autoscales
        to the number of download tasks (capped at 32)
    retries : int, optional
        Number of retries for each download task on transient errors, by default 3

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

    Badges
    ------
    region:global dataclass:simulation product:wind product:precip product:temp product:atmos
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
        async_workers: int | None = None,
        retries: int = 3,
    ):
        super().__init__(
            member=member,
            max_workers=max_workers,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
            async_workers=async_workers,
            retries=retries,
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


def _decode_gefs_grib(grib_file: str) -> np.ndarray:
    """Decode a single-message GEFS grib file into a numpy array.

    Module-level so it can be dispatched to a worker thread and patched in
    offline tests. Uses pygrib, which is faster and lower memory than
    xarray/cfgrib for single-message slices.

    Parameters
    ----------
    grib_file : str
        Path to local grib file holding one message

    Returns
    -------
    np.ndarray
        Decoded field values
    """
    try:
        grbs = pygrib.open(grib_file)
    except Exception:
        logger.error(f"Failed to open grib file {grib_file}")
        raise
    try:
        return grbs[1].values
    except Exception:
        logger.error(f"Failed to read grib file {grib_file}")
        raise
    finally:
        grbs.close()
