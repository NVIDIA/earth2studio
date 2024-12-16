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

import asyncio
import hashlib
import os
import pathlib
import shutil
from datetime import datetime, timedelta

import fsspec
import numpy as np
import s3fs
import xarray as xr
import zarr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import HRRRFXLexicon, HRRRLexicon
from earth2studio.lexicon.base import LexiconType
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class _HRRRBase:

    HRRR_BUCKET_NAME = "noaa-hrrr-bdp-pds"
    HRRR_X = np.arange(1799)
    HRRR_Y = np.arange(1059)
    MAX_BYTE_SIZE = 5000000

    def __init__(self, lexicon: LexiconType, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._lexicon = lexicon
        self._verbose = verbose

        self.fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=2**20,
            client_kwargs={},
        )

        # Doesnt work with read block
        # if self._cache:
        #     cache_options = {
        #         "cache_storage": self.cache,
        #         "expiry_time": 31622400,  # 1 year
        #     }
        #     self.fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

    async def async_fetch(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> xr.DataArray:
        """Async function to retrieve HRRR forecast data into a single Xarray data array

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

        for i, t in enumerate(time):
            for j, ld in enumerate(lead_time):
                for k, v in enumerate(variable):
                    try:
                        hrrr_str, modifier = self._lexicon[v]
                        hrrr_class, hrrr_product, hrrr_level, hrrr_var = hrrr_str.split(
                            "::"
                        )
                    except KeyError as e:
                        logger.error(f"variable id {v} not found in HRRR lexicon")
                        raise e

                    date_group = t.strftime("%Y%m%d")
                    forcast_hour = t.strftime("%H")
                    lead_index = int(ld.total_seconds() // 3600)
                    s3_grib_uri = f"{self.HRRR_BUCKET_NAME}/hrrr.{date_group}/conus/hrrr.t{forcast_hour}z.wrf{hrrr_class}f{lead_index:02}.grib2"

                    # Download the grib index file and parse
                    with self.fs.open(f"{s3_grib_uri}.idx") as file:
                        index_lines = [line.decode("utf-8").rstrip() for line in file]
                    # Add dummy variable at end of file with max offset
                    index_lines.append(
                        f"xx:{self.fs.size(s3_grib_uri)}:d=xx:NULL:NULL:NULL:NULL"
                    )

                    byte_offset = None
                    byte_length = -1
                    for line_index, line in enumerate(index_lines[:-1]):
                        lsplit = line.split(":")
                        if len(lsplit) < 7:
                            continue
                        # If match get in byte offset and length
                        if lsplit[3] == hrrr_var and lsplit[4] == hrrr_level:
                            nlsplit = index_lines[line_index + 1].split(":")
                            byte_length = int(nlsplit[1]) - int(lsplit[1])
                            byte_offset = int(lsplit[1])
                            key = f"{lsplit[3]}::{lsplit[4]}"
                            if byte_length > self.MAX_BYTE_SIZE:
                                raise ValueError(
                                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                                )
                    # If byte offset is not raise error
                    if byte_offset is None or byte_length is None:
                        raise KeyError(
                            f"Could not find variable {hrrr_var} level {hrrr_level} in index file"
                        )
                    # Read grib block into cache location
                    sha = hashlib.sha256(
                        (s3_grib_uri + str(byte_offset) + str(byte_length)).encode()
                    )
                    filename = sha.hexdigest()
                    cache_path = os.path.join(self.cache, filename)
                    if not pathlib.Path(cache_path).is_file():
                        grib_buffer = self.fs.read_block(
                            s3_grib_uri, offset=byte_offset, length=byte_length
                        )
                        with open(cache_path, "wb") as file:
                            file.write(grib_buffer)

                    da = xr.open_dataarray(
                        cache_path,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": ""},
                    )
                    hrrr_da[i, j, k] = modifier(da.values)
                    if "lat" not in hrrr_da.coords:
                        hrrr_da.coords["lat"] = (
                            ["hrrr_y", "hrrr_x"],
                            da.coords["latitude"].values,
                        )
                        hrrr_da.coords["lon"] = (
                            ["hrrr_y", "hrrr_x"],
                            da.coords["longitude"].values,
                        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return hrrr_da

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
            if not DistributedManager.is_initialized():
                DistributedManager.initialize()
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
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
            time = datetime.utcfromtimestamp((time - _unix) / _ds)

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


class _HRRR_ZarrBase:

    HRRR_BUCKET_NAME = "hrrrzarr"
    HRRR_X = np.arange(1799)
    HRRR_Y = np.arange(1059)

    def __init__(self, lexicon: LexiconType, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._lexicon = lexicon
        self._verbose = verbose

        fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=2**20,
            client_kwargs={},
        )

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

        fs_map = fsspec.FSMap(f"s3://{self.HRRR_BUCKET_NAME}", fs)
        self.zarr_group = zarr.open(fs_map, mode="r")

    async def async_fetch(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> xr.DataArray:
        """Async function to retrieve HRRR forecast data into a single Xarray data array

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
                "lat": (
                    ["hrrr_y", "hrrr_x"],
                    self.zarr_group["grid"]["HRRR_chunk_index.zarr"]["latitude"][:],
                ),
                "lon": (
                    ["hrrr_y", "hrrr_x"],
                    self.zarr_group["grid"]["HRRR_chunk_index.zarr"]["longitude"][:]
                    + 360,
                ),  # Change to [0,360]
            },
        )
        # Banking on async calls in zarr 3.0
        for i, t in enumerate(time):
            for j, ld in enumerate(lead_time):
                for k, v in enumerate(variable):
                    try:
                        hrrr_str, modifier = self._lexicon[v]
                        hrrr_class, hrrr_product, hrrr_level, hrrr_var = hrrr_str.split(
                            "::"
                        )
                    except KeyError as e:
                        logger.error(f"variable id {v} not found in HRRR lexicon")
                        raise e

                    date_group = t.strftime("%Y%m%d")
                    time_group = t.strftime(f"%Y%m%d_%Hz_{hrrr_product}.zarr")

                    logger.debug(
                        f"Fetching HRRR {hrrr_product} variable {v} at {t.isoformat()}"
                    )

                    data = self.zarr_group[hrrr_class][date_group][time_group][
                        hrrr_level
                    ][hrrr_var][hrrr_level][hrrr_var]
                    if hrrr_product == "fcst":
                        # Minus 1 here because index 0 is forecast with leadtime 1hr
                        # forecast_period coordinate system tells what the lead times are in hours
                        lead_index = int(ld.total_seconds() // 3600) - 1
                        data = data[lead_index]

                    hrrr_da[i, j, k] = modifier(data)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return hrrr_da

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
            if not DistributedManager.is_initialized():
                DistributedManager.initialize()
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
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
            time = datetime.utcfromtimestamp((time - _unix) / _ds)

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        fs = s3fs.S3FileSystem(anon=True)

        # Object store directory for given date
        date_group = time.strftime("%Y%m%d")
        time_group = time.strftime("%Y%m%d_%Hz_anl.zarr")
        s3_uri = f"s3://{cls.HRRR_BUCKET_NAME}/sfc/{date_group}/{time_group}"
        exists = fs.exists(s3_uri)

        return exists


class HRRR(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) data source provides hourly North-American
    weather analysis data developed by NOAA (used to initialize the HRRR forecast
    model). This data source is provided on a Lambert conformal 3km grid at 1-hour
    intervals. The spatial dimensionality of HRRR data is [1059, 1799]. This data source
    pulls data from the HRRR zarr bucket on S3.

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

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://hrrrzarr.s3.amazonaws.com/index.html
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __init__(self, cache: bool = True, verbose: bool = True):
        super().__init__(HRRRLexicon, cache, verbose)

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

        data_array = asyncio.get_event_loop().run_until_complete(
            self.async_fetch(time, [timedelta(hours=0)], variable)
        )
        return data_array.isel(lead_time=0).drop_vars("lead_time")


class HRRR_FX(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) forecast source provides a North-American
    weather forecasts with hourly forecast runs developed by NOAA. This forecast source
    has hourly forecast steps up to a lead time of 48 hours. Data is provided on a
    Lambert conformal 3km grid at 1-hour intervals. The spatial dimensionality of HRRR
    data is [1059, 1799]. This data source pulls data from the HRRR zarr bucket on S3.

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

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __init__(self, cache: bool = True, verbose: bool = True):
        super().__init__(HRRRFXLexicon, cache, verbose)

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

        return asyncio.get_event_loop().run_until_complete(
            self.async_fetch(time, lead_time, variable)
        )

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
            if hours > 48 or hours < 1:
                raise ValueError(
                    f"Requested lead time {delta} can only be between [1,48] hours for HRRR forecast"
                )
