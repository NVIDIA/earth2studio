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
from datetime import datetime

import boto3
import botocore
import numpy as np
import s3fs
import xarray as xr
from botocore import UNSIGNED
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon import GFSLexicon
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

LOCAL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")


class GFS:
    """The global forecast service (GFS) initial state data source provided on an
    equirectangular grid. GFS is a weather forecast model developed by NOAA. This data
    source is provided on a 0.25 degree lat lon grid at 6-hour intervals spanning from
    Feb 26th 2021 to present date.

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
    This data source only fetches the initial state of GFS and does not fetch an
    predicted time steps.

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

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose

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
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_gfs_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_gfs_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives GFS data array for given date time by fetching the index file,
        fetching variable grib files and lastly combining grib files into single data
        array.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in GFS lexicon

        Returns
        -------
        xr.DataArray
            GFS data array for given date time

        Raises
        ------
        KeyError
            Un supported variable.
        """
        logger.debug(f"Fetching GFS index file: {time}")
        index_file = self._fetch_index(time)

        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        # Would need to update "f000" for getting forecast steps
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f000"
        )
        grib_file_name = os.path.join(self.GFS_BUCKET_NAME, file_name)

        gfsda = xr.DataArray(
            data=np.empty((1, len(variables), len(self.GFS_LAT), len(self.GFS_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
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

            if gfs_name not in index_file:
                raise KeyError(f"Could not find variable {gfs_name} in index file")

            byte_offset = index_file[gfs_name][0]
            byte_length = index_file[gfs_name][1]
            # Download the grib file to cache
            logger.debug(f"Fetching GFS grib file for variable: {variable} at {time}")
            grib_file = self._download_s3_grib_cached(
                grib_file_name, byte_offset=byte_offset, byte_length=byte_length
            )
            # Open into xarray data-array
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            gfsda[0, i] = modifier(da.values)

        return gfsda

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
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
            # To update search "gfs." at https://noaa-gfs-bdp-pds.s3.amazonaws.com/index.html
            # They are slowly adding more data
            if time < datetime(year=2021, month=2, day=17):
                raise ValueError(
                    f"Requested date time {time} needs to be after February 17th, 2021 for GFS"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in GFS")

    def _fetch_index(self, time: datetime) -> dict[str, tuple[int, int]]:
        """Fetch GFS atmospheric index file

        Parameters
        ----------
        time : datetime
            Date time to fetch

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GFS vairables (byte offset, byte length)
        """
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f000.idx"
        )
        s3_uri = os.path.join(self.GFS_BUCKET_NAME, file_name)
        # Grab index file
        index_file = self._download_s3_index_cached(s3_uri)
        with open(index_file, "r") as file:
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
            key = f"{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        # Pop place holder
        return index_table

    def _download_s3_index_cached(self, path: str) -> str:
        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
        fs.get_file(path, cache_path)

        return cache_path

    def _download_s3_grib_cached(
        self, path: str, byte_offset: int = 0, byte_length: int = None
    ) -> str:
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        fs = s3fs.S3FileSystem(anon=True, client_kwargs={})
        if not pathlib.Path(cache_path).is_file():
            data = fs.read_block(path, offset=byte_offset, length=byte_length)
            with open(cache_path, "wb") as file:
                file.write(data)

        return cache_path

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "gfs")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the GFS object store

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

        s3 = boto3.client(
            "s3", config=botocore.config.Config(signature_version=UNSIGNED)
        )
        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        try:
            resp = s3.list_objects_v2(
                Bucket=cls.GFS_BUCKET_NAME, Prefix=file_name, Delimiter="/", MaxKeys=1
            )
        except botocore.exceptions.ClientError as e:
            logger.error("Failed to access from GFS S3 bucket")
            raise e

        return "KeyCount" in resp and resp["KeyCount"] > 0
