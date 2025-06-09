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
from typing import Literal

import gcsfs
import nest_asyncio
import numpy as np
import xarray as xr
import zarr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    AsyncCachingFileSystem,
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import WB2ClimatetologyLexicon, WB2Lexicon
from earth2studio.utils.type import TimeArray, VariableArray


class _WB2Base:
    """Base class for weather bench 2 ERA5 datasets"""

    WB2_ERA5_LAT = np.empty(0)
    WB2_ERA5_LON = np.empty(0)

    def __init__(
        self,
        wb2_zarr_store: str,
        wb2_product: str = "era5",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):

        self._zarr_store_name = wb2_zarr_store
        self._product = wb2_product

        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout

        # Check Zarr version and use appropriate method
        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            # Fallback to older method if version check fails
            zarr_major_version = 2  # Assume older version if we can't determine
        # Only zarr 3.0 support
        if zarr_major_version < 3:
            raise ModuleNotFoundError("Zarr 3.0 and above support only")

        # Check to see if there is a running loop (initialized in async)
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            # Else we assume that async calls will be used which in that case
            # we will init the group in the call function when we have the loop
            self.zarr_group = None
            self.level_coords = None

    async def _async_init(self) -> None:
        """Async initialization of zarr group

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        fs = gcsfs.GCSFileSystem(
            cache_timeout=-1,
            token="anon",  # noqa: S106 # nosec B106
            access="read_only",
            block_size=8**20,
            asynchronous=True,
        )

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            fs = AsyncCachingFileSystem(fs=fs, **cache_options, asynchronous=True)

        zstore = zarr.storage.FsspecStore(
            fs,
            path=f"/weatherbench2/datasets/{self._product}/{self._zarr_store_name}",
        )
        self.zarr_group = await zarr.api.asynchronous.open(store=zstore, mode="r")
        self.level_coords = await (await self.zarr_group.get("level")).getitem(  # type: ignore
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
            return. Must be in the WB2 lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from weather bench 2
        """

        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.zarr_group is None:
            loop.run_until_complete(self._async_init())

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
            return. Must be in the WB2 lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from weather bench 2
        """
        if self.zarr_group is None:
            raise ValueError(
                "Zarr group is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        xr_array = xr.DataArray(
            data=np.empty(
                (
                    len(time),
                    len(variable),
                    len(self.WB2_ERA5_LAT),
                    len(self.WB2_ERA5_LON),
                )
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.WB2_ERA5_LAT,
                "lon": self.WB2_ERA5_LON,
            },
        )

        args = [
            (t, i, v, j) for j, v in enumerate(variable) for i, t in enumerate(time)
        ]
        func_map = map(functools.partial(self.fetch_wrapper, xr_array=xr_array), args)

        # Launch all fetch requests
        await tqdm.gather(
            *func_map, desc="Fetching WB2 data", disable=(not self._verbose)
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
        if self.zarr_group is None:
            raise ValueError("Zarr group is not initialized")
        # Get time index (vanilla zarr doesnt support date indices)
        time_index = self._get_time_index(time)
        logger.debug(
            f"Fetching WB2 zarr array for variable: {variable} at {time.isoformat()}"
        )
        try:
            wb2_name, modifier = WB2Lexicon[variable]  # type: ignore
        except KeyError as e:
            logger.error(f"variable id {variable} not found in WB2 lexicon")
            raise e

        wb2_name, level = wb2_name.split("::")

        zarr_array = await self.zarr_group.get(wb2_name)
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

        # Some WB2 data Zarr stores are saved [lon, lat] with lat flipped
        # Namely its the lower resolutions ones with this issue
        if output.shape[0] > output.shape[1]:
            output = np.flip(output, axis=-1).T

        return output

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "wb2era5")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_wb2era5")
        return cache_location

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for Weatherbench 2 ERA5

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for Weatherbench2 ERA5"
                )

            if time < datetime(year=1959, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1959 for Weatherbench2 ERA5"
                )

            if time > datetime(year=2023, month=1, day=10, hour=18):
                raise ValueError(
                    f"Requested date time {time} needs to be before January 11th, 2023  for Weatherbench2 ERA5"
                )

    @classmethod
    def _get_time_index(cls, time: datetime) -> int:
        """Little index converter to go from datetime to integer index for hour
        and day of year.

        Parameters
        ----------
        time : datetime
            Input date time

        Returns
        -------
        int
            hour coordinate index of data
        int
            day_of_year coordinate index of data
        """
        start_date = datetime(year=1959, month=1, day=1)
        duration = time - start_date
        return int(divmod(duration.total_seconds(), 21600)[0])


class WB2ERA5(_WB2Base):
    """
    ERA5 reanalysis data with several derived variables on a 0.25 degree lat-lon grid
    from 1959 to 2023 (incl) to 6 hour intervals on 13 pressure levels. Provided by the
    WeatherBench2 data repository.

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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5
    - https://arxiv.org/abs/2308.15560
    """

    WB2_ERA5_LAT = np.linspace(90, -90, 721)
    WB2_ERA5_LON = np.linspace(0, 359.75, 1440)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            wb2_zarr_store="1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )


class WB2ERA5_121x240(_WB2Base):
    """
    ERA5 reanalysis data with several derived variables down sampled to a 1.5 degree
    lat-lon grid from 1959 to 2023 (incl) to 6 hour intervals on 13 pressure levels.
    Provided by the WeatherBench2 data repository.

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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5
    - https://arxiv.org/abs/2308.15560
    """

    WB2_ERA5_LAT = np.linspace(90, -90, 121)
    WB2_ERA5_LON = np.linspace(0, 359.5, 240)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            wb2_zarr_store="1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )


class WB2ERA5_32x64(_WB2Base):
    """
    ERA5 reanalysis data with several derived variables down sampled to a 5.625 degree
    lat-lon grid from 1959 to 2023 (incl) to 6 hour intervals on 13 pressure levels.
    Provided by the WeatherBench2 data repository.

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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5
    - https://arxiv.org/abs/2308.15560
    """

    WB2_ERA5_LAT = np.linspace(-87.1875, 87.1875, 32)
    WB2_ERA5_LON = np.linspace(0, 360, 64, endpoint=False)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )


ClimatologyZarrStore = Literal[
    "1990-2017_6h_1440x721.zarr",
    "1990-2017_6h_512x256_equiangular_conservative.zarr",
    "1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr",
    "1990-2017_6h_64x32_equiangular_conservative.zarr",
    "1990-2019_6h_1440x721.zarr",
    "1990-2019_6h_512x256_equiangular_conservative.zarr",
    "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr",
    "1990-2019_6h_64x32_equiangular_conservative.zarr",
]


class WB2Climatology(_WB2Base):
    """
    Climatology provided by WeatherBench2,

    |    A climatology is used for e.g. computing anomaly metrics such as the ACC.
    |    For WeatherBench 2, the climatology was computed using a running window for
    |    smoothing (see paper and script) for each day of year and sixth hour of day.
    |    We have computed climatologies for 1990-2017 and 1990-2019.

    Parameters
    ----------
    climatology_zarr_store : ClimatologyZarrStore, optional
        Stores within `gs://weatherbench2/datasets/era5-hourly-climatology/` to select
        As of 05/03/2024 this is the following list of available files:

        - 1990-2017_6h_1440x721.zarr
        - 1990-2017_6h_512x256_equiangular_conservative.zarr
        - 1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr
        - 1990-2017_6h_64x32_equiangular_conservative.zarr
        - 1990-2019_6h_1440x721.zarr
        - 1990-2019_6h_512x256_equiangular_conservative.zarr
        - 1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr
        - 1990-2019_6h_64x32_equiangular_conservative.zarr

        by default `1990-2019_6h_1440x721.zarr`
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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5-climatology
    - https://arxiv.org/abs/2308.15560
    """

    def __init__(
        self,
        climatology_zarr_store: ClimatologyZarrStore = "1990-2017_6h_1440x721.zarr",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            climatology_zarr_store,
            wb2_product="era5-hourly-climatology",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )

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
            return. Must be in the WB2 Climatology lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from weather bench 2
        """
        if self.zarr_group is None:
            raise ValueError(
                "Zarr group is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Before anything wait until the group gets opened
        if inspect.isawaitable(self.zarr_group):
            self.zarr_group = await self.zarr_group

        WB2_CLIMATE_LAT = await (await self.zarr_group.get("latitude")).getitem(
            slice(None)
        )
        WB2_CLIMATE_LON = await (await self.zarr_group.get("longitude")).getitem(
            slice(None)
        )

        xr_array = xr.DataArray(
            data=np.empty(
                (len(time), len(variable), len(WB2_CLIMATE_LAT), len(WB2_CLIMATE_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": WB2_CLIMATE_LAT[:],
                "lon": WB2_CLIMATE_LON[:],
            },
        )

        args = [
            (t, i, v, j) for j, v in enumerate(variable) for i, t in enumerate(time)
        ]
        func_map = map(functools.partial(self.fetch_wrapper, xr_array=xr_array), args)

        self.level_coords = await (await self.zarr_group.get("level")).getitem(
            slice(None)
        )
        # Launch all fetch requests
        await tqdm.gather(
            *func_map, desc="Fetching WB2 climatology data", disable=(not self._verbose)
        )
        return xr_array

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
        if self.zarr_group is None:
            raise ValueError("Zarr group is not initialized")

        # Get time index (vanilla zarr doesnt support date indices)
        hour_index, day_of_year_index = self._get_time_index(time)
        logger.debug(
            f"Fetching WB2 climatology zarr array for variable: {variable} at {time.isoformat()}"
        )
        try:
            wb2_name, modifier = WB2ClimatetologyLexicon[variable]  # type: ignore
        except KeyError as e:
            logger.error(f"variable id {variable} not found in WB2 lexicon")
            raise e

        wb2_name, level = wb2_name.split("::")

        zarr_array = await self.zarr_group.get(wb2_name)
        shape = zarr_array.shape

        # Surface variable [hour idx (6 hour), day index, lat, lon]
        if len(shape) == 4:
            data = await zarr_array.getitem((hour_index, day_of_year_index))
            output = modifier(data)
        # Atmospheric variable [hour idx (6 hour), day index, level lat, lon]
        else:
            # Load levels coordinate system from Zarr store and check
            level_index = np.searchsorted(self.level_coords, int(level))
            data = await zarr_array.getitem(
                (hour_index, day_of_year_index, level_index)
            )
            output = modifier(data)

        return output

    @classmethod
    def _get_time_index(cls, time: datetime) -> tuple[int, int]:  # type: ignore[override]
        """Little index converter to go from datetime to integer index for hour
        and day of year.

        Parameters
        ----------
        time : datetime
            Input date time

        Returns
        -------
        int
            hour coordinate index of data
        int
            day_of_year coordinate index of data
        """
        tt = time.timetuple()
        return tt.tm_hour // 6, tt.tm_yday - 1
