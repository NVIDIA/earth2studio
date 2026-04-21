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
import hashlib
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.nclimgrid import NClimGridLexicon
from earth2studio.utils.type import TimeArray, VariableArray


@dataclass
class NClimGridDailyAsyncTask:
    """Async task for a single NClimGridDaily fetch operation."""

    time_index: int
    variable_index: int
    variable_id: str
    native_key: str
    modifier: Callable
    nc_uri: str
    target_date: np.datetime64


class NClimGridDaily:
    """NOAA NClimGrid daily gridded climate data source.

    NClimGrid provides daily CONUS gridded temperature and precipitation data
    on a ~1/24 degree (~0.0417°) latitude/longitude grid. Data spans from
    1951 to the present and is stored as monthly NetCDF files on AWS S3.

    Available variables: t2m_max, t2m_min, t2m (daily avg), tp (precipitation).

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository:

    - https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily
    - https://registry.opendata.aws/noaa-nclimgrid/

    Badges
    ------
    region:na dataclass:observation product:temp product:precip
    """

    NCLIMGRID_BUCKET_NAME = "noaa-nclimgrid-daily-pds"

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=False, skip_instance_cache=True
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get NClimGrid data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Canonical Earth2Studio variable name(s). Must be in the
            NClimGridLexicon.

        Returns
        -------
        xr.DataArray
            NClimGrid weather data array with dimensions [time, variable, lat, lon].
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        try:
            xr_array = loop.run_until_complete(
                asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get NClimGrid data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Canonical Earth2Studio variable name(s). Must be in the
            NClimGridLexicon.

        Returns
        -------
        xr.DataArray
            NClimGrid weather data array with dimensions [time, variable, lat, lon].
        """
        time_list, variable_list = prep_data_inputs(time, variable)

        # NClimGrid is daily data — truncate to day resolution so sub-day
        # timestamps (e.g. 07:30) map to the correct daily value.
        # Deduplicate since multiple sub-day times map to the same day.
        time_list = list(
            dict.fromkeys(datetime(t.year, t.month, t.day) for t in time_list)
        )

        Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time_list)

        tasks = self._create_tasks(time_list, variable_list)

        async_tasks = [self.fetch_wrapper(task) for task in tasks]
        results = await tqdm.gather(
            *async_tasks,
            desc="Fetching NClimGrid data",
            disable=(not self._verbose),
        )

        # Group results by variable, then concat times
        data_arrays: dict[str, list[xr.DataArray]] = {}
        for result in results:
            key = str(result.coords["variable"].values)
            if key not in data_arrays:
                data_arrays[key] = []
            data_arrays[key].append(result)

        array_list = []
        for arrs in data_arrays.values():
            if len(arrs) > 1:
                array_list.append(xr.concat(arrs, dim="time"))
            else:
                array_list.append(arrs[0])

        res = xr.concat(array_list, dim="variable", coords="minimal")
        res.name = None

        return res.sel(time=time_list, variable=variable_list)

    def _create_tasks(
        self, time: list[datetime], variable: list[str]
    ) -> list[NClimGridDailyAsyncTask]:
        """Create download tasks for parallel execution.

        Parameters
        ----------
        time : list[datetime]
            Timestamps to download.
        variable : list[str]
            Variables to download.

        Returns
        -------
        list[NClimGridDailyAsyncTask]
            List of async task requests.
        """
        tasks: list[NClimGridDailyAsyncTask] = []
        for i, t in enumerate(time):
            for j, v in enumerate(variable):
                try:
                    native_key, modifier = NClimGridLexicon[v]  # type: ignore[misc]
                except KeyError:
                    logger.warning(
                        f"Variable id {v} not found in NClimGridLexicon, skipping"
                    )
                    continue

                nc_uri = self._monthly_nc_uri(t)
                # NClimGrid is daily data — truncate to day resolution
                # so that sub-day times (e.g. 07:30) match the file timestamps
                target_date = np.datetime64(t).astype("datetime64[D]")
                tasks.append(
                    NClimGridDailyAsyncTask(
                        time_index=i,
                        variable_index=j,
                        variable_id=v,
                        native_key=native_key,
                        modifier=modifier,
                        nc_uri=nc_uri,
                        target_date=target_date,
                    )
                )
        return tasks

    async def fetch_wrapper(
        self,
        task: NClimGridDailyAsyncTask,
    ) -> xr.DataArray:
        """Unpack task, fetch data, and return as a single-element DataArray.

        Parameters
        ----------
        task : NClimGridDailyAsyncTask
            Task with all fetch metadata.

        Returns
        -------
        xr.DataArray
            Single time/variable slice with dims [time, variable, lat, lon].
        """
        data = await self.fetch_array(task)
        return data

    async def fetch_array(
        self,
        task: NClimGridDailyAsyncTask,
    ) -> xr.DataArray:
        """Fetch a single variable/time slice from remote store.

        Parameters
        ----------
        task : NClimGridDailyAsyncTask
            Task with all fetch metadata.

        Returns
        -------
        xr.DataArray
            Data array with dimensions [time, variable, lat, lon].
        """
        logger.debug(
            f"Fetching NClimGrid {task.native_key} for {task.target_date} "
            f"from {task.nc_uri}"
        )

        # Build a cache key from the URI, variable, and date
        sha = hashlib.sha256(
            (str(task.nc_uri) + str(task.native_key) + str(task.target_date)).encode()
        )
        cache_path = os.path.join(self.cache, sha.hexdigest())

        if os.path.exists(cache_path):
            ds = xr.open_dataarray(cache_path, engine="h5netcdf", cache=False)
        else:
            if self.fs is None:
                raise ValueError(
                    "File store is not initialized! If calling fetch directly "
                    "make sure the data source is initialized inside the async loop."
                )
            with self.fs.open(task.nc_uri, "rb") as f:
                dataset = await asyncio.to_thread(
                    xr.open_dataset, f, engine="h5netcdf", cache=False
                )
                da = dataset[task.native_key].sel(time=str(task.target_date))
                da = await asyncio.to_thread(da.load)

            # Apply lexicon modifier (unit conversion)
            values = task.modifier(da.values)
            values = np.asarray(values, dtype=np.float32)

            lat = da.coords["lat"].values.astype(np.float32)
            lon = da.coords["lon"].values.astype(np.float32)

            ds = xr.DataArray(
                data=values[np.newaxis, np.newaxis, :, :],
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": [task.target_date],
                    "variable": [task.variable_id],
                    "lat": lat,
                    "lon": lon,
                },
            )

            if self._cache:
                ds.to_netcdf(cache_path, engine="h5netcdf")

        return ds

    def _monthly_nc_uri(self, t: datetime) -> str:
        """Build S3 URI for a monthly NetCDF file.

        Parameters
        ----------
        t : datetime
            Timestamp within the target month.

        Returns
        -------
        str
            S3 URI to monthly NetCDF file.
        """
        return (
            f"s3://{self.NCLIMGRID_BUCKET_NAME}/access/grids/"
            f"{t.year}/ncdd-{t.year}{t.month:02d}-grd-scaled.nc"
        )

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify that date times are valid for NClimGrid.

        Parameters
        ----------
        times : list[datetime]
            Date times to validate.
        """
        for time in times:
            if time < datetime(1951, 1, 1):
                raise ValueError(
                    f"Requested date time {time} must be after January 1st, 1951 "
                    f"for NClimGrid"
                )
            # NClimGrid is daily — time of day is ignored during selection,
            # but we accept any datetime.

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            If date time is available.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    @property
    def cache(self) -> str:
        """Get the appropriate cache location.

        Returns
        -------
        str
            Path to the cache directory.
        """
        cache_location = os.path.join(datasource_cache_root(), "nclimgrid")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nclimgrid_{self._tmp_cache_hash}"
            )
        return cache_location
