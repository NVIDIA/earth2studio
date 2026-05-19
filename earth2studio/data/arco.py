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
import os
import pathlib
import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

import numpy as np
import xarray as xr
import zarr
from gcsfs import GCSFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    AsyncCachingFileSystem,
    _sync_async,
    datasource_cache_root,
    get_msc_filesystem,
    prep_data_inputs,
)
from earth2studio.lexicon import ARCOLexicon
from earth2studio.utils.type import TimeArray, VariableArray


class ARCO:
    """Analysis-Ready, Cloud Optimized (ARCO) is a data store of ERA5 re-analysis data
    currated by Google. This data is stored in Zarr format and contains 31 surface
    variables, pressure level variables defined on 37 pressure levels, and model level
    variables defined on the 137 native ERA5 vertical levels, all on a 0.25 degree lat
    lon grid. Temporal resolution is 1 hour.

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

    The data source will automatically use Multi-Storage Client (MSC) if available,
    otherwise it will fallback to using gcsfs directly. MSC can provide better
    performance for cloud storage access.

    Badges
    ------
    region:global dataclass:reanalysis product:wind product:precip product:temp product:atmos
    """

    ARCO_LAT = np.linspace(90, -90, 721)
    ARCO_LON = np.linspace(0, 359.75, 1440)
    ARCO_PATH = "/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ARCO_TIME_STOP = datetime(year=2023, month=11, day=11)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):

        self._cache = cache
        self._verbose = verbose

        self.zarr_group: zarr.core.group.AsyncGroup | None = None
        self.level_coords = None
        # Model-level store
        self.ml_zarr_group: zarr.core.group.AsyncGroup | None = None
        self.ml_level_coords = None

        self.async_timeout = async_timeout

    async def _async_init(self) -> None:
        """Async initialization of zarr group

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        # Common filesystem configuration parameters
        fs_config = {
            "cache_timeout": -1,
            "token": "anon",  # noqa: S106 # nosec B106
            "access": "read_only",
            "block_size": 8**20,
            "asynchronous": True,
            "skip_instance_cache": True,
        }

        # Try to use Multi-Storage Client if available, otherwise fallback to gcsfs
        MSCFileSystem = get_msc_filesystem()
        if MSCFileSystem:
            logger.debug("Using Multi-Storage Client for ARCO data access")
            fs = MSCFileSystem(**fs_config)
        else:
            fs = GCSFileSystem(**fs_config)

        # Need to manually set this here, the reason being that when the file system
        # defines the weak ref of the client, it needs the loop used to create it.
        # Otherwise it will try to kill the client with another loop, throwing an error
        # at the end of the script
        fs._loop = asyncio.get_event_loop()

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            fs = AsyncCachingFileSystem(fs=fs, **cache_options, asynchronous=True)

        # Pressure/surface store
        zstore = zarr.storage.FsspecStore(
            fs,
            path=self.ARCO_PATH,
        )
        self.zarr_group = await zarr.api.asynchronous.open(store=zstore, mode="r")

        if "valid_time_stop" in self.zarr_group.attrs:
            ARCO.ARCO_TIME_STOP = datetime.strptime(
                self.zarr_group.attrs["valid_time_stop"], "%Y-%m-%d"
            )

        self.level_coords = await (await self.zarr_group.get("level")).getitem(
            slice(None)
        )
        # Model-level store
        ml_zstore = zarr.storage.FsspecStore(
            fs,
            path="/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
        )
        self.ml_zarr_group = await zarr.api.asynchronous.open(store=ml_zstore, mode="r")
        self.ml_level_coords = await (await self.ml_zarr_group.get("hybrid")).getitem(
            slice(None)
        )

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
            return. Must be in the ARCO lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO
        """
        if self.zarr_group is None:
            await self._async_init()

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

        groups: dict[tuple[str, bool], list[tuple[str, int, str, Callable]]] = (
            defaultdict(list)
        )
        for j, v in enumerate(variable):
            try:
                arco_name, modifier = ARCOLexicon[v]
            except KeyError:
                logger.error(f"variable id {v} not found in ARCO lexicon")
                raise
            parts = arco_name.split("::")
            arco_variable = parts[0]
            level = parts[1] if len(parts) > 1 else ""
            is_mdl = self._is_mdl_level(v)
            groups[(arco_variable, is_mdl)].append((v, j, level, modifier))

        # Build one task per unique (time, zarr_array) combination
        args = [
            (t, i, arco_variable, is_mdl, var_entries)
            for (arco_variable, is_mdl), var_entries in groups.items()
            for i, t in enumerate(time)
        ]
        func_map = map(
            functools.partial(self.fetch_chunk_group, xr_array=xr_array), args
        )

        await tqdm.gather(
            *func_map, desc="Fetching ARCO data", disable=(not self._verbose)
        )
        return xr_array

    async def fetch_chunk_group(
        self,
        e: tuple[datetime, int, str, bool, list],
        xr_array: xr.DataArray,
    ) -> None:
        """Fetch a Zarr chunk once and distribute slices to all variables
        that share the same underlying array.

        Parameters
        ----------
        e : tuple
            (time, time_index, arco_variable, is_mdl, var_entries) where
            var_entries is [(var_name, var_idx, level, modifier), ...]
        xr_array : xr.DataArray
            Output array to write into
        """
        t, time_idx, arco_variable, is_mdl, var_entries = e

        if self.zarr_group is None or self.ml_zarr_group is None:
            raise ValueError("Zarr group is not initialized")

        time_index = self._get_time_index(t)

        if is_mdl:
            zarr_group = self.ml_zarr_group
            level_coords = self.ml_level_coords
        else:
            zarr_group = self.zarr_group
            level_coords = self.level_coords

        zarr_array = await zarr_group.get(arco_variable)
        shape = zarr_array.shape

        if len(shape) == 2:
            # static variable
            data = await zarr_array.getitem(slice(None))
            for var_name, var_idx, level, modifier in var_entries:
                xr_array[time_idx, var_idx] = modifier(data)
        elif len(shape) == 3:
            # surface variable
            data = await zarr_array.getitem(time_index)
            for var_name, var_idx, level, modifier in var_entries:
                xr_array[time_idx, var_idx] = modifier(data)
        else:
            # atmospheric variable : fetch all needed levels at once
            level_indices = []
            for var_name, var_idx, level, modifier in var_entries:
                level_indices.append(np.searchsorted(level_coords, int(level)))

            # Fetch all levels in a single chunk read
            all_levels_data = await zarr_array.getitem(time_index)
            for k, (var_name, var_idx, level, modifier) in enumerate(var_entries):
                xr_array[time_idx, var_idx] = modifier(
                    all_levels_data[level_indices[k]]
                )

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
        if self.zarr_group is None or self.ml_zarr_group is None:
            raise ValueError("Zarr group is not initialized")
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

        parts = arco_name.split("::")
        arco_variable, level = parts[0], (parts[1] if len(parts) > 1 else "")
        if self._is_mdl_level(variable):
            zarr_group = self.ml_zarr_group
            level_coords = self.ml_level_coords
        else:
            zarr_group = self.zarr_group
            level_coords = self.level_coords

        zarr_array = await zarr_group.get(arco_variable)
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
            level_index = np.searchsorted(level_coords, int(level))
            data = await zarr_array.getitem((time_index, level_index))
            output = modifier(data)

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

            if time > cls.ARCO_TIME_STOP:
                raise ValueError(
                    f"Requested date time {time} needs to be on or before {cls.ARCO_TIME_STOP.strftime('%B %d, %Y')} for ARCO"
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
    def _is_mdl_level(cls, variable: str) -> bool:
        """Checks if given variable is a model level variable based on the lexicon pattern.

        Parameters
        ----------
        variable : str
            Variable to check

        Returns
        -------
        bool
            If variable is a model level variable
        """
        return bool(re.match(r"^[a-zA-Z0-9]+[0-9]+k$", variable))

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

        # TODO: FIX THIS, FOR ZARR 3.0 THIS IS DANGEROUS NON-ASYNC
        # Try to use Multi-Storage Client if available, otherwise fallback to gcsfs
        MSCFileSystem = get_msc_filesystem()
        if MSCFileSystem:
            fs = MSCFileSystem(cache_timeout=-1)
        else:
            fs = GCSFileSystem(cache_timeout=-1)

        gcstore = zarr.storage.FsspecStore(
            fs,
            path=cls.ARCO_PATH,
        )

        zarr_group = zarr.open(gcstore, mode="r")
        # Load time coordinate system from Zarr store and check
        time_index = cls._get_time_index(time)
        max_index = zarr_group["time"][-1]
        return time_index >= 0 and time_index <= max_index
