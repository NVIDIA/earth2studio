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

import hashlib
import os
import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import s3fs
import xarray as xr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from modulus.distributed.manager import DistributedManager
from s3fs.core import S3FileSystem
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_forecast_inputs,
)
from earth2studio.lexicon import GEFSLexicon, GEFSLexiconSel
from earth2studio.lexicon.base import LexiconType
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class _GEFSBase:

    GEFS_BUCKET_NAME: str
    GEFS_CHECK_CLASS: str
    GEFS_LAT: np.ndarray
    GEFS_LON: np.ndarray

    MAX_BYTE_SIZE: int

    _cache: bool
    _verbose: bool
    lexicon: LexiconType
    fs: s3fs.S3FileSystem
    s3fs: s3fs.S3FileSystem

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GEFS ensemble forecast data

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
            GEFS weather data array
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        # Create data array of forecast data
        xr_array = self.create_data_array(time, lead_time, variable)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array

    def create_data_array(
        self, time: list[datetime], lead_time: list[timedelta], variable: list[str]
    ) -> xr.DataArray:
        """Function that creates and populates an xarray data array with requested
        GEFS data.

        Parameters
        ----------
        time : list[datetime]
            Time list to fetch
        lead_time: list[timedelta]
            Lead time list to getch
        variable : list[str]
            Variable list to fetch

        Returns
        -------
        xr.DataArray
            Xarray data array
        """
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

        args = [
            (t, i, l, j, v, k)
            for k, v in enumerate(variable)
            for j, l in enumerate(lead_time)  # noqa
            for i, t in enumerate(time)
        ]

        pbar = tqdm(
            total=len(args), desc="Fetching GEFS data", disable=(not self._verbose)
        )
        for (t, i, l, j, v, k) in args:  # noqa
            data = self.fetch_array(t, l, v)
            xr_array[i, j, k] = data
            pbar.update(1)

        return xr_array

    def fetch_array(
        self, time: datetime, lead_time: timedelta, variable: str
    ) -> np.ndarray:
        """Fetches requested array from remote store

        Parameters
        ----------
        time : datetime
            Time to fetch
        lead_time: timedelta
            Lead time to fetch
        variable : str
            Variable to fetch

        Returns
        -------
        np.ndarray
            Data
        """
        logger.debug(
            f"Fetching GEFS data for variable: {variable} at {time.isoformat()} lead time {lead_time}"
        )
        try:
            gefs_name, modifier = self.lexicon[variable]
            gefs_grib, gefs_var, gefs_level = gefs_name.split("::")
        except KeyError as e:
            logger.error(f"variable id {variable} not found in GEFS lexicon")
            raise e

        file_name = self._get_grid_name(time, lead_time, gefs_grib)
        s3_index_uri = os.path.join(self.GEFS_BUCKET_NAME, file_name + ".idx")
        s3_grib_uri = os.path.join(self.GEFS_BUCKET_NAME, file_name)

        # Download the grib index file and parse
        with self.fs.open(s3_index_uri) as file:
            index_lines = [line.decode("utf-8").rstrip() for line in file]
        # Add dummy variable at end of file with max offset
        index_lines.append(f"xx:{self.fs.size(s3_grib_uri)}:d=xx:NULL:NULL:NULL:NULL")
        index_table = {}

        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        gefs_name = f"{gefs_var}::{gefs_level}"
        if gefs_name not in index_table:
            raise KeyError(f"Could not find variable {gefs_name} in index file")
        byte_offset = index_table[gefs_name][0]
        byte_length = index_table[gefs_name][1]

        # CFGrib requires file to be downloaded locally, no one like grib
        # Because we need to read a byte range we need to use the s3fs store directly
        # so manual caching it is. Literally NO ONE likes grib
        sha = hashlib.sha256(
            (s3_grib_uri + str(byte_offset) + str(byte_length)).encode()
        )
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            grib_buffer = self.s3fs.read_block(
                s3_grib_uri, offset=byte_offset, length=byte_length
            )
            with open(cache_path, "wb") as file:
                file.write(grib_buffer)

        da = xr.open_dataarray(
            cache_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        return modifier(da.values)

    def _get_grid_name(
        self, time: datetime, lead_time: timedelta, grid_class: str
    ) -> str:
        """Returns gribfile name to fetch, should override in child

        Parameters
        ----------
        time : datetime
            Time to fetch
        lead_time: timedelta
            Lead time to fetch
        grid_class : str
            Grib classification in GEFS (e.g. pgrb2a, pgrb2b, pgrb2s)

        Returns
        -------
        str
            File name of Grib file in S3 bucket
        """
        raise NotImplementedError("Child class needs to implement this")

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
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
            if time < datetime(year=2020, month=9, day=23):
                raise ValueError(
                    f"Requested date time {time} needs to be after Sept 23rd, 2020 for GEFS"
                )

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for GEFS based on offline knowledge

        Parameters
        ----------
        lead_times : list[timedelta]
            list of lead times to fetch data
        """
        raise NotImplementedError("Child class needs to implement this")

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gfs")
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
        """Checks if given date time is avaliable in the GEFS object store

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

        fs = S3FileSystem(anon=True)

        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/atmos/{cls.GEFS_CHECK_CLASS}/"
        s3_uri = f"s3://{cls.GEFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class GEFS_FX(_GEFSBase):
    """The Global Ensemble Forecast System (GEFS) forecast source is a 30 member
    ensemble forecast provided on an 0.5 degree equirectangular grid.  GEFS is a weather
    forecast model developed by  National Centers for Environmental Prediction (NCEP).
    This forecast source has data at 6-hour intervals spanning from Sept 23rd 2020 to
    present date. Each forecast provides 3-hourly predictions up to 10 days (240 hours)
    and 6 hourly predictions for another 6 days (384 hours).

    Parameters
    ----------
    product : str, optional
        GEFS product. Options are: control gec00 (control), gepNN (forecast member NN,
        e.g. gep01, gep02,...), by default "gec00"
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

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

    GEFS_PRODUCTS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    GEFS_CHECK_CLASS = "pgrb2bp5"

    def __init__(
        self,
        product: str = "gec00",
        cache: bool = True,
        verbose: bool = True,
    ):

        if product not in self.GEFS_PRODUCTS:
            raise ValueError(f"Invalid GEFS product {product}")

        self._cache = cache
        self._verbose = verbose
        self._product = product
        self.s3fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=2**20,
            client_kwargs={},
        )

        # if self._cache:
        cache_options = {
            "cache_storage": self.cache,
            "expiry_time": 31622400,  # 1 year
        }
        self.fs = WholeFileCacheFileSystem(fs=self.s3fs, **cache_options)
        self.lexicon = GEFSLexicon

    def _get_grid_name(
        self, time: datetime, lead_time: timedelta, grid_class: str
    ) -> str:
        """Return grib file name"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        return os.path.join(
            file_name,
            f"atmos/{grid_class}p5/{self._product}.t{time.hour:0>2}z.{grid_class}.0p50.f{lead_hour:03d}",
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


class GEFS_FX_721x1440(_GEFSBase):
    """The Global Ensemble Forecast System (GEFS) forecast source is a 30 member
    ensemble forecast provided on an 0.25 degree equirectangular grid. GEFS is a
    weather forecast model developed by  National Centers for Environmental Prediction
    (NCEP). This data source provides the select variables of GEFS served on a higher
    resolution grid t 6-hour intervals spanning from Sept 23rd 2020 to present date.
    Each forecast provides 3-hourly predictions up to 10 days (240 hours) and 6 hourly
    predictions for another 6 days (384 hours).

    Parameters
    ----------
    product : str, optional
        GEFS product. Options are: control gec00 (control), gepNN (forecast member NN,
        e.g. gep01, gep02,...), by default "gec00"
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

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

    GEFS_BUCKET_NAME = "noaa-gefs-pds"
    MAX_BYTE_SIZE = 5000000

    GEFS_LAT = np.linspace(90, -90, 721)
    GEFS_LON = np.linspace(0, 359.75, 1440)

    GEFS_PRODUCTS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    GEFS_CHECK_CLASS = "pgrb2sp25"

    def __init__(
        self,
        product: str = "gec00",
        cache: bool = True,
        verbose: bool = True,
    ):

        if product not in self.GEFS_PRODUCTS:
            raise ValueError(f"Invalid GEFS select variable product {product}")

        self._cache = cache
        self._verbose = verbose
        self._product = product
        self.s3fs = s3fs.S3FileSystem(
            anon=True,
            default_block_size=2**20,
            client_kwargs={},
        )

        # if self._cache:
        cache_options = {
            "cache_storage": self.cache,
            "expiry_time": 31622400,  # 1 year
        }
        self.fs = WholeFileCacheFileSystem(fs=self.s3fs, **cache_options)
        self.lexicon = GEFSLexiconSel

    def _get_grid_name(
        self, time: datetime, lead_time: timedelta, grid_class: str
    ) -> str:
        """Return grib file name"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gefs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        return os.path.join(
            file_name,
            f"atmos/{grid_class}p25/{self._product}.t{time.hour:0>2}z.{grid_class}.0p25.f{lead_hour:03d}",
        )

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
