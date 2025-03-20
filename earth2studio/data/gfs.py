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
from fsspec.implementations.ftp import FTPFileSystem
from loguru import logger
from physicsnemo.distributed.manager import DistributedManager
from s3fs.core import S3FileSystem
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import GFSLexicon
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class GFS:
    """The global forecast service (GFS) initial state data source provided on an
    equirectangular grid. GFS is a weather forecast model developed by NOAA. This data
    source is provided on a 0.25 degree lat lon grid at 6-hour intervals spanning from
    Feb 26th 2021 to present date.

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
    This data source only fetches the initial state of GFS and does not fetch an
    predicted time steps. See :class:`~earth2studio.data.GFS_FX` for fetching predicted
    data from this forecast system.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://registry.opendata.aws/noaa-gfs-bdp-pds/
    - https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php
    """

    GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
    MAX_BYTE_SIZE = 5000000

    GFS_LAT = np.linspace(90, -90, 721)
    GFS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, source: str = "aws", cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose

        if source == "aws":
            self.uri_prefix = "noaa-gfs-bdp-pds"
            self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
            # To update search "gfs." at https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
            # They are slowly adding more data
            def _range(time: datetime) -> None:
                if time < datetime(year=2021, month=1, day=1):
                    raise ValueError(
                        f"Requested date time {time} needs to be after January 1st, 2021 for GFS on AWS"
                    )

            self._history_range = _range
        elif source == "ncep":
            # Could use http location, but using ftp since better for larger data
            # https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
            self.uri_prefix = "pub/data/nccf/com/gfs/prod/"
            self.fs = FTPFileSystem(host="ftpprd.ncep.noaa.gov")

            def _range(time: datetime) -> None:
                if time + timedelta(days=10) < datetime.today():
                    raise ValueError(
                        f"Requested date time {time} needs to be within past 10 days for GFS NCEP source"
                    )

            self._history_range = _range
        else:
            raise ValueError(f"Invalid GFS source {source}")

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve GFS initial data to be used for initial conditions for the given
        time, variable information, and optional history.

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

        # Fetch index file for requested time
        # Should really async this stuff
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives GFS initial state data array for given date time

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            List of atmosphric variables to fetch. Must be supported in GFS lexicon

        Returns
        -------
        xr.DataArray
            GFS data array for given date time

        Raises
        ------
        KeyError
            Un supported variable.
        """
        da = self._fetch_gfs_dataarray(time, timedelta(hours=0), variables)
        return da.isel(lead_time=0)

    def _fetch_gfs_dataarray(
        self,
        time: datetime,
        lead_time: timedelta,
        variables: list[str],
    ) -> xr.DataArray:
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
        logger.debug(f"Fetching GFS index file: {time} lead {lead_time}")
        index_file = self._fetch_index(self._grib_index_uri(time, lead_time))

        gfsda = xr.DataArray(
            data=np.empty((1, 1, len(variables), len(self.GFS_LAT), len(self.GFS_LON))),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "lead_time": [lead_time],
                "variable": variables,
                "lat": self.GFS_LAT,
                "lon": self.GFS_LON,
            },
        )

        # TODO: Add MP here
        for i, variable in enumerate(
            tqdm(
                variables, desc=f"Fetching GFS for {time}", disable=(not self._verbose)
            )
        ):
            # Convert from Earth2Studio variable ID to GFS id and modifier
            # sphinx - lexicon start
            try:
                gfs_name, modifier = GFSLexicon[variable]
            except KeyError:
                logger.warning(
                    f"variable id {variable} not found in GFS lexicon, good luck"
                )
                gfs_name = variable

                def modifier(x: np.array) -> np.array:
                    """Modify data (if necessary)."""
                    return x

            byte_offset = None
            byte_length = None
            for key, value in index_file.items():
                if gfs_name in key:
                    byte_offset = value[0]
                    byte_length = value[1]
                    break

            if byte_offset is None:
                raise KeyError(f"Could not find variable {gfs_name} in index file")
            # Download the grib file to cache
            logger.debug(
                f"Fetching GFS grib file for variable: {variable} at {time}_{lead_time}"
            )
            grib_file = self._fetch_remote_file(
                self._grib_uri(time, lead_time),
                byte_offset=byte_offset,
                byte_length=byte_length,
            )
            # Open into xarray data-array
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            gfsda[0, 0, i] = modifier(da.values)
            # sphinx - lexicon end

        return gfsda

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

    def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int]]:
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
        index_file = self._fetch_remote_file(index_uri)
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
            key = f"{lsplit[0]}::{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        # Pop place holder
        return index_table

    def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int = None
    ) -> str:
        """Fetches remote file into cache"""
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            data = self.fs.read_block(path, offset=byte_offset, length=byte_length)
            with open(cache_path, "wb") as file:
                file.write(data)

        return cache_path

    def _grib_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS grib files"""
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}"
        )
        return os.path.join(self.uri_prefix, file_name)

    def _grib_index_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Generates the URI for GFS index grib files"""
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        lead_hour = int(lead_time.total_seconds() // 3600)
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f{lead_hour:03d}.idx"
        )
        return os.path.join(self.uri_prefix, file_name)

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
            time = datetime.utcfromtimestamp((time - _unix) / _ds)

        # Offline checks
        # try:
        #     cls._validate_time([time])
        # except ValueError:
        #     return False

        fs = S3FileSystem(anon=True)

        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        s3_uri = f"s3://{cls.GFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists


class GFS_FX(GFS):
    """The global forecast service (GFS) forecast source provided on an equirectangular
    grid. GFS is a weather forecast model developed by NOAA. This data source is on a
    0.25 degree lat lon grid at 6-hour intervals spanning from Feb 26th 2021 to present
    date. Each forecast provides hourly predictions up to 16 days (384 hours).

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
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        # Fetch index file for requested time
        # Should really async this stuff
        data_arrays = []
        for t0 in time:
            lead_arrays = []
            for l0 in lead_time:
                data_array = self.fetch_dataarray(t0, l0, variable)
                lead_arrays.append(data_array)

            data_arrays.append(xr.concat(lead_arrays, dim="lead_time"))

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_dataarray(  # type: ignore[override]
        self,
        time: datetime,
        lead_time: timedelta,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives GFS data array for given date time by fetching the index file,
        fetching variable grib files and lastly combining grib files into single data
        array.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        lead_time : timedelta
            Forecast lead time to fetch
        variables : list[str]
            List of atmosphric variables to fetch. Must be supported in GFS lexicon

        Returns
        -------
        xr.DataArray
            GFS data array for given date time

        Raises
        ------
        KeyError
            Un supported variable.
        """
        return self._fetch_gfs_dataarray(time, lead_time, variables)

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
