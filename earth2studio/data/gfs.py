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

import asyncio
import functools
import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import obstore as obs
import xarray as xr
from loguru import logger
from obstore.store import ObjectStore
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    as_datetime,
    as_timedelta,
    async_retry,
    cancellable_to_thread,
    datasource_cache_root,
    decode_grib_message,
    gather_with_concurrency,
    map_with_concurrency,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
    prep_forecast_inputs,
    resolve_async_workers,
    statistic_modifiers,
)
from earth2studio.lexicon import GFSLexicon
from earth2studio.utils.time_statistics import source_lead_times, source_times
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class GFSAsyncTask:
    """Small helper struct for Async tasks"""

    output_modifiers: list[Callable]
    gfs_file_uri: str
    gfs_byte_offset: int | None
    gfs_byte_length: int | None
    gfs_modifier: Callable
    # Extra bucket records used only for reconstructed six-hour precipitation.
    corrections: list[tuple[int, "GFSAsyncTask"]] = field(default_factory=list)


class GFS:
    """The global forecast service (GFS) initial state data source provided on an
    equirectangular grid. GFS is a weather forecast model developed by NOAA. This data
    source is provided on a 0.25 degree lat lon grid at 6-hour intervals spanning from
    Jan 1st 2021 to present date.

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
    async_workers : int, optional
        Maximum number of concurrent downloads. By default None, which autoscales
        to the number of download tasks (capped at 64)
    retries : int, optional
        Number of retries for each download task on transient errors, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    This data source fetches the initial state of GFS. The derived ``tp06`` variable
    uses the preceding cycle's six-hour forecast accumulation ending at the requested
    time, in metres. See [`GFS_FX`][earth2studio.data.GFS_FX] for forecast lead times.

    Note
    ----
    Temporal statistics: ``mean``, ``sum``, ``min``, and ``max`` accept labels
    ``variable:method:window`` or ``variable:method:start:end``. Windows are
    left-closed/right-open, sampled every six hours relative to the analysis time.
    ``t2m:mean:24h`` averages the preceding four analyses, excluding that time.
    ``tp06`` represents trailing six-hour accumulations; sums are totals only
    when those intervals do not overlap.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://registry.opendata.aws/noaa-gfs-bdp-pds/
    - https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php

    Badges
    ------
    region:global dataclass:analysis dataset:gfs product:wind product:precip product:temp product:atmos provider:noaa
    """

    GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
    MAX_BYTE_SIZE = 5000000

    GFS_LAT = np.linspace(90, -90, 721)
    GFS_LON = np.linspace(0, 359.75, 1440)
    lexicon = GFSLexicon

    def __init__(
        self,
        source: str = "aws",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int | None = None,
        retries: int = 3,
    ):
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self._inflight_downloads: dict[str, asyncio.Future] = {}

        self.store: ObjectStore | None = None
        if source == "aws":
            self.uri_prefix = "noaa-gfs-bdp-pds"
            self._store_url = f"s3://{self.GFS_BUCKET_NAME}"

            # To update search "gfs." at https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
            # They are slowly adding more data
            def _range(time: datetime) -> None:
                if time < datetime(year=2021, month=1, day=1):
                    raise ValueError(
                        f"Requested date time {time} needs to be after January 1st, 2021 for GFS on AWS"
                    )

            self._history_range = _range
        elif source == "ncep":
            # NOMADS HTTPS mirror of the NCEP production feed, supports the
            # ranged GETs obstore's HTTPStore issues for byte-range fetches
            # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
            self.uri_prefix = "pub/data/nccf/com/gfs/prod/"
            self._store_url = "https://nomads.ncep.noaa.gov"

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
        """Async initialization of the object store

        Note
        ----
        Unlike async fsspec filesystems, obstore stores are event-loop
        independent and could be built in ``__init__``; kept as a lazy async
        method to preserve the initialization seam.
        """
        self.store = obstore_store_from_url(
            self._store_url,
            max_pool_connections=self._async_workers or 64,
        )

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
        try:
            xr_array = _sync_async(
                self.fetch, time, variable, timeout=self.async_timeout
            )
        finally:
            # Delete cache if needed
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

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
        # Lazily initialize the object store on first use
        if self.store is None:
            await self._async_init()

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Note, this could be more memory efficient and avoid pre-allocation of the array
        # but this is much much cleaner to deal with, compared to something seen in the
        # NCAR data source.
        xr_array = xr.DataArray(
            data=np.full(
                (len(time), 1, len(variable), len(self.GFS_LAT), len(self.GFS_LON)),
                np.nan,
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

        async_tasks, attrs, finalize = await self._create_tasks(
            time, [timedelta(hours=0)], variable, xr_array, forecast=False
        )
        await map_with_concurrency(
            async_tasks,
            self.fetch_wrapper,
            async_workers=self._async_workers,
            task_timeout=120.0,
            desc="Fetching GFS data",
            verbose=self._verbose,
        )
        finalize()
        xr_array.attrs.update(attrs)

        return xr_array.isel(lead_time=0)

    def _lookup_record(
        self,
        index_file: dict[str, tuple[int, int]],
        v: str,
        t: datetime,
        lt: timedelta,
    ) -> tuple[int, int, Callable] | None:
        """Find the grib byte range and modifier of a variable in an index file

        Parameters
        ----------
        index_file : dict[str, tuple[int, int]]
            Parsed grib index of one (time, lead time) file
        v : str
            Earth2Studio variable name
        t : datetime
            Forecast initialization time
        lt : timedelta
            Forecast lead time

        Returns
        -------
        tuple[int, int, Callable] | None
            Byte offset, byte length and modifier, None if the record is missing
        """
        # --8<-- [start:gfs-lexicon-lookup]
        try:
            gfs_name, modifier = GFSLexicon[v]
        except KeyError:
            logger.warning(f"variable id {v} not found in GFS lexicon, good luck")
            gfs_name = v

            def modifier(x: np.array) -> np.array:
                """Modify data (if necessary)."""
                return x

        lead_hour = int(lt.total_seconds() // 3600)
        if v == "tp06":
            # Only a matching six-hour record can satisfy this label directly.
            gfs_name = f"{gfs_name}::{lead_hour - 6}-{lead_hour} hour acc fcst"
        elif v == "tp" and lead_hour > 0:
            # GFS restarts precipitation accumulation every six hours. For
            # leads 1-6, duplicate descriptions resolve to the first (recent
            # accumulation window) record in the index.
            start_hour = 6 * ((lead_hour - 1) // 6)
            gfs_name = f"{gfs_name}::{start_hour}-{lead_hour} hour acc fcst"

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
            return None
        # --8<-- [end:gfs-lexicon-lookup]
        return byte_offset, byte_length, modifier

    async def _create_tasks(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
        xr_array: xr.DataArray,
        forecast: bool,
    ) -> tuple[list[GFSAsyncTask], dict, Callable]:
        """Create GFS byte-range tasks, adding statistical output modifiers."""
        samples: dict[tuple[datetime, timedelta, str], list[Callable]] = {}
        finalizers = []
        attrs: dict = {}
        # Bind forecast reducers separately per initialization: equal valid times
        # from different forecast cycles must never contribute to the same window.
        for i in range(len(time) if forecast else 1):
            modifiers, attrs, finalize = statistic_modifiers(
                variable,
                xr_array.values[i] if forecast else xr_array.values[:, 0],
                supported=self.lexicon.TIME_STATISTICS,
                source=type(self).__name__,
                verbose=self._verbose,
            )
            finalizers.append(finalize)
            for v, statistic, add in modifiers:
                coords = (
                    np.asarray(lead_time, dtype="timedelta64[ns]")
                    if forecast
                    else np.asarray(time, dtype="datetime64[ns]")
                )
                required = (
                    coords[:, None]
                    if statistic is None
                    else (
                        source_lead_times(statistic, coords, np.timedelta64(1, "h"))
                        if forecast
                        else source_times(statistic, coords, np.timedelta64(6, "h"))
                    )
                )
                for j, row in enumerate(required):
                    for coord in row:
                        t, lt = (
                            (time[i], as_timedelta(coord))
                            if forecast
                            else (as_datetime(coord), timedelta(0))
                        )
                        self._validate_time([t])
                        if isinstance(self, GFS_FX):
                            self._validate_leadtime([lt])
                        if v == "tp06":
                            if forecast and lt < timedelta(hours=6):
                                raise ValueError(
                                    "GFS tp06 requires forecast lead times of at least 6 hours"
                                )
                            if not forecast:
                                t, lt = t - timedelta(hours=6), timedelta(hours=6)
                                self._validate_time([t])
                        samples.setdefault((t, lt, v), []).append(
                            functools.partial(add, j)
                        )
        if self._verbose and any(v == "tp06" for _, _, v in samples):
            logger.info(
                "GFS tp06: six-hour precipitation ending at each valid time, in metres"
            )
        if self._verbose and any(v == "tp" for _, _, v in samples):
            logger.warning(
                "GFS tp is accumulated over a 6 hour bucket that restarts, "
                "statistics of tp combine these bucket values"
            )
        # One index per file, including the records needed to reconstruct tp06.
        files = dict.fromkeys((t, lt) for t, lt, _ in samples)
        for t, lt, v in samples:
            hours = int(lt.total_seconds() // 3600)
            if v == "tp06" and hours % 6:
                files[t, timedelta(hours=6 * ((hours - 1) // 6))] = None
                files[t, lt - timedelta(hours=6)] = None
        results = await gather_with_concurrency(
            [self._fetch_index(self._grib_index_uri(t, lt)) for t, lt in files],
            max_workers=resolve_async_workers(self._async_workers, len(files)),
            task_timeout=60.0,
            desc="Fetching GFS index files",
            verbose=True,
        )
        indices = dict(zip(files, results))

        def make_task(
            t: datetime, lt: timedelta, v: str, outputs: list[Callable]
        ) -> GFSAsyncTask:
            found = self._lookup_record(indices[t, lt], v, t, lt)
            offset, length, modifier = (
                found if found is not None else (None, None, lambda x: x)
            )
            return GFSAsyncTask(
                outputs, self._grib_uri(t, lt), offset, length, modifier
            )

        tasks = []
        for (t, lt, v), outputs in samples.items():
            task = make_task(t, lt, "tp" if v == "tp06" else v, outputs)
            hours = int(lt.total_seconds() // 3600)
            if v == "tp06" and hours % 6:
                # P(6,7) + P(0,6) - P(0,1) gives the trailing total at lead 7.
                task.corrections = [
                    (
                        1,
                        make_task(t, timedelta(hours=6 * ((hours - 1) // 6)), "tp", []),
                    ),
                    (-1, make_task(t, lt - timedelta(hours=6), "tp", [])),
                ]
            tasks.append(task)

        def finalize_all() -> None:
            for finalize in finalizers:
                finalize()

        return tasks, attrs, finalize_all

    async def fetch_wrapper(self, task: GFSAsyncTask) -> None:
        """Small wrapper to pack arrays into the DataArray via output modifiers."""

        async def fetch(task: GFSAsyncTask) -> np.ndarray:
            if task.gfs_byte_offset is None or task.gfs_byte_length is None:
                return np.full((len(self.GFS_LAT), len(self.GFS_LON)), np.nan)
            return await async_retry(
                self.fetch_array,
                task.gfs_file_uri,
                task.gfs_byte_offset,
                task.gfs_byte_length,
                task.gfs_modifier,
                retries=self._retries,
                backoff=1.0,
                task_timeout=60.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )

        out = await fetch(task)
        for sign, correction in task.corrections:
            sample = await fetch(correction)
            if sign > 0:
                out += sample
            else:
                out -= sample
        for modifier in task.output_modifiers:
            modifier(out)

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
        np.ndarray
            GFS array for given time and lead time
        """
        logger.debug(f"Fetching GFS grib file: {grib_uri} {byte_offset}-{byte_length}")
        # Download the grib file to cache
        grib_file = await self._fetch_remote_file(
            grib_uri,
            byte_offset=byte_offset,
            byte_length=byte_length,
        )
        # pygrib decode is blocking and GIL-bound; run in a thread with timeout
        values = await cancellable_to_thread(
            decode_grib_message, grib_file, timeout=30.0
        )
        return modifier(values)

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
        try:
            index_file = await self._fetch_remote_file(index_uri)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The specified data index, {index_uri}, does not exist. Data seems to be missing."
            )
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
        # Hash the bucket-prefixed path (not the store-relative key) so warm
        # caches populated before the obstore migration remain valid
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        if self.store is None:
            raise ValueError("Object store is not initialized")

        # aws paths are bucket-prefixed; ncep paths are already store-relative
        key = path.removeprefix(self.GFS_BUCKET_NAME + "/")

        # Concurrent samples (e.g. tp06 at neighbouring leads) can request the same
        # byte range. Share one in-flight download per cache file; the cache file
        # itself is published atomically so readers never see a partial file.
        inflight = self._inflight_downloads
        task = inflight.get(filename)
        if task is None:
            task = asyncio.ensure_future(
                obstore_fetch_to_cache(
                    self.store,
                    key,
                    self.cache,
                    byte_offset=byte_offset,
                    byte_length=byte_length,
                    cache_key=filename,
                    atomic=True,
                )
            )
            inflight[filename] = task
            task.add_done_callback(lambda _t: inflight.pop(filename, None))
        # shield so one cancelled/timed-out sample does not abort the shared download
        return await asyncio.shield(task)

    def _grib_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS grib files"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        if time < datetime(2021, 3, 22, 12):
            file_name = os.path.join(
                file_name, f"gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}"
            )
        else:
            file_name = os.path.join(
                file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}"
            )
        return os.path.join(self.uri_prefix, file_name)

    def _grib_index_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS index grib files"""
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        # Directory structure changed March 22, 2021 at 12z
        if time < datetime(2021, 3, 22, 12):
            file_name = os.path.join(
                file_name, f"gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}.idx"
            )
        else:
            file_name = os.path.join(
                file_name,
                f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}.idx",
            )
        return os.path.join(self.uri_prefix, file_name)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gfs")
        if not self._cache:
            if self._tmp_cache_hash is None:
                # First access for temp cache: create a random suffix to avoid collisions
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gfs_{self._tmp_cache_hash}"
            )
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

        store = obstore_store_from_url(f"s3://{cls.GFS_BUCKET_NAME}")
        # Object store directory for given time
        # Should contain two keys: atmos and wave
        prefix = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        # obs.list yields chunks (lists) of entries; one entry proves existence
        chunk: list = next(iter(obs.list(store, prefix=prefix, chunk_size=1)), [])

        return len(chunk) > 0


class GFS_FX(GFS):
    """The global forecast service (GFS) forecast source provided on an equirectangular
    grid. GFS is a weather forecast model developed by NOAA. This data source is on a
    0.25 degree lat lon grid at 6-hour intervals spanning from Feb 26th 2021 to present
    date. Each forecast provides hourly predictions up to 16 days (384 hours).

    ``tp06`` is the trailing six-hour precipitation total in metres, reconstructed
    across accumulation-bucket resets within the same forecast initialization.
    All requested ``tp06`` sample lead times must be at least six hours. Temporal
    statistics of ``tp06`` reduce these six-hour totals; overlapping totals in a
    sum count overlapping precipitation more than once.

    Note
    ----
    Temporal statistics: ``mean``, ``sum``, ``min``, and ``max`` accept labels
    ``variable:method:window`` or ``variable:method:start:end``. Windows are
    left-closed/right-open, sampled hourly over lead times of one initialization.
    ``t2m:mean:6h`` at lead six hours averages leads zero through five hours.
    Negative or unavailable leads are rejected. Raw ``tp`` accumulation buckets
    reset every six hours; summing them is not a precipitation total.

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

    Badges
    ------
    region:global dataclass:simulation dataset:gfs product:wind product:precip product:temp product:atmos provider:noaa
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
        try:
            xr_array = _sync_async(
                self.fetch, time, lead_time, variable, timeout=self.async_timeout
            )
        finally:
            # Delete cache if needed
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

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
            data=np.full(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(self.GFS_LAT),
                    len(self.GFS_LON),
                ),
                np.nan,
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

        async_tasks, attrs, finalize = await self._create_tasks(
            time, lead_time, variable, xr_array, forecast=True
        )
        await map_with_concurrency(
            async_tasks,
            self.fetch_wrapper,
            async_workers=self._async_workers,
            task_timeout=120.0,
            desc="Fetching GFS data",
            verbose=self._verbose,
        )
        finalize()
        xr_array.attrs.update(attrs)

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
