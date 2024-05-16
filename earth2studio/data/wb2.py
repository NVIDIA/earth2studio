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

import os
import pathlib
import shutil
from datetime import datetime
from typing import Literal

import fsspec
import gcsfs
import numpy as np
import xarray as xr
import zarr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon import WB2Lexicon
from earth2studio.utils.type import TimeArray, VariableArray

LOCAL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")


class WB2ERA5:
    """
    ERA5 reanalysis data provided by WeatherBench2 downloaded from from the Copernicus
    Climate Data Store from 1959 to 2023 (incl) to 6 hour intervals on 13 pressure
    levels. This data is stored in Zarr format.

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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5
    - https://arxiv.org/abs/2308.15560
    """

    WB2_ERA5_LAT = np.linspace(90, -90, 721)
    WB2_ERA5_LON = np.linspace(0, 359.75, 1440)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
    ):

        self._cache = cache
        self._verbose = verbose

        if self._cache:
            gcstore = fsspec.get_mapper(
                "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
                target_protocol="gs",
                cache_storage=self.cache,
                target_options={"anon": True, "default_block_size": 2**20},
            )
        else:
            gcs = gcsfs.GCSFileSystem(cache_timeout=-1)
            gcstore = gcsfs.GCSMap(
                "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
                gcs=gcs,
            )
        self.zarr_group = zarr.open(gcstore, mode="r", consolidated=True)

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
            return. Must be in the WB2Lexicon lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from WB2Lexicon
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_wb2_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_wb2_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives WeatherBench2 data array for given date time by downloading a lat
        lon array from the Zarr store

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in WeatherBench2 lexicon

        Returns
        -------
        xr.DataArray
            WeatherBench2 data array for given date time
        """
        wb2da = xr.DataArray(
            data=np.empty(
                (1, len(variables), len(self.WB2_ERA5_LAT), len(self.WB2_ERA5_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.WB2_ERA5_LON,
                "lon": self.WB2_ERA5_LON,
            },
        )

        # Load levels coordinate system from Zarr store and check
        level_coords = self.zarr_group["level"][:]
        # Get time index (vanilla zarr doesnt support date indices)
        time_index = self._get_time_index(time)

        # TODO: Add MP here
        for i, variable in enumerate(
            tqdm(
                variables,
                desc=f"Fetching WB2 ERA5 for {time}",
                disable=(not self._verbose),
            )
        ):
            logger.debug(
                f"Fetching WB2 ERA5 zarr array for variable: {variable} at {time.isoformat()}"
            )
            try:
                wb2_name, modifier = WB2Lexicon[variable]
            except KeyError as e:
                logger.error(f"variable id {variable} not found in WB2 lexicon")
                raise e

            wb2_name, level = wb2_name.split("::")

            shape = self.zarr_group[wb2_name].shape
            # Static variables
            if len(shape) == 2:
                wb2da[0, i] = modifier(self.zarr_group[wb2_name][:])
            # Surface variable
            elif len(shape) == 3:
                wb2da[0, i] = modifier(self.zarr_group[wb2_name][time_index])
            # Atmospheric variable
            else:
                level_index = np.where(level_coords == int(level))[0][0]
                wb2da[0, i] = modifier(
                    self.zarr_group[wb2_name][time_index, level_index]
                )

        return wb2da

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "wb2era5")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
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

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Checks if given date time is avaliable in the WeatherBench2 data source

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
            time = datetime.utcfromtimestamp((time - _unix) / _ds)

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        gcs = gcsfs.GCSFileSystem(cache_timeout=-1)
        gcstore = gcsfs.GCSMap(
            "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
            gcs=gcs,
        )
        zarr_group = zarr.open(gcstore, mode="r")
        # Load time coordinate system from Zarr store and check
        time_index = cls._get_time_index(time)
        max_index = zarr_group["time"][-1]
        return time_index >= 0 and time_index <= max_index


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


class WB2Climatology:
    """
    Climatology provided by WeatherBench2,

    |    A climatology is used for e.g. computing anomaly metrics such as the ACC.
    |    For WeatherBench 2, the climatology was computed using a running window for
    |    smoothing (see paper and script) for each day of year and sixth hour of day.
    |    We have computed climatologies for 1990-2017 and 1990-2019.

    Parameters
    ----------
    ClimatologyZarrStore : Literal, optional
        File within gs://weatherbench2/datasets/era5-hourly-climatology/ to select,
        by default 1990-2019_6h_1440x721.zarr
        As of 05/03/2024 this is the following list of available files:
            1990-2017_6h_1440x721.zarr
            1990-2017_6h_512x256_equiangular_conservative.zarr
            1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr
            1990-2017_6h_64x32_equiangular_conservative.zarr
            1990-2019_6h_1440x721.zarr
            1990-2019_6h_512x256_equiangular_conservative.zarr
            1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr
            1990-2019_6h_64x32_equiangular_conservative.zarr
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

    - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5-climatology
    - https://arxiv.org/abs/2308.15560
    """

    def __init__(
        self,
        climatology_zarr_store: ClimatologyZarrStore = "1990-2017_6h_1440x721.zarr",
        cache: bool = True,
        verbose: bool = True,
    ):

        self._cache = cache
        self._verbose = verbose

        if self._cache:
            gcstore = fsspec.get_mapper(
                "gs://weatherbench2/datasets/era5-hourly-climatology/"
                + climatology_zarr_store,
                target_protocol="gs",
                cache_storage=self.cache,
                target_options={"anon": True, "default_block_size": 2**20},
            )
        else:
            gcs = gcsfs.GCSFileSystem(cache_timeout=-1)
            gcstore = gcsfs.GCSMap(
                "gs://weatherbench2/datasets/era5-hourly-climatology/"
                + climatology_zarr_store,
                gcs=gcs,
            )
        self.zarr_group = zarr.open(gcstore, mode="r")

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
            return. Must be in the WeatherBench2 lexicon.

        Returns
        -------
        xr.DataArray
            climatology data from WeatherBench climatology
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_wb_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_wb_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives WeatherBench data array for given date time by downloading a
        lat lon array from the Zarr store

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch.

        Returns
        -------
        xr.DataArray
            WeatherBench data array for given date time
        """
        LAT = self.zarr_group["latitude"][:]
        LON = self.zarr_group["longitude"][:]
        wbda = xr.DataArray(
            data=np.empty((1, len(variables), len(LAT), len(LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": LAT,
                "lon": LON,
            },
        )

        # Load levels coordinate system from Zarr store and check
        level_coords = self.zarr_group["level"][:]
        # Get time index (vanilla zarr doesnt support date indices)
        hour, day_of_year = self._get_time_index(time)

        # TODO: Add MP here
        for i, variable in enumerate(
            tqdm(
                variables,
                desc=f"Fetching WeatherBench Climatology for {time}",
                disable=(not self._verbose),
            )
        ):
            logger.debug(
                f"Fetching WeatherBench Climatology zarr array for variable: {variable} at {time.isoformat()}"
            )

            try:
                wb2_name, modifier = WB2Lexicon[variable]
            except KeyError as e:
                logger.error(
                    f"variable id {variable} not found in WeatherBench lexicon"
                )
                raise e

            wb2_variable, level = wb2_name.split("::")

            if len(level) > 0:
                wb2_variable, level = wb2_name.split("::")
                level_index = np.where(level_coords == int(level))[0][0]
                wbda[0, i] = modifier(
                    self.zarr_group[wb2_variable][hour, day_of_year, level_index]
                )

            else:
                wb2_variable = wb2_name.split("::")[0]
                wbda[0, i] = modifier(self.zarr_group[wb2_variable][hour, day_of_year])

        return wbda

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "wb2")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    @classmethod
    def _get_time_index(cls, time: datetime) -> tuple[int, int]:
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
