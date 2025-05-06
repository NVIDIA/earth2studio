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

import hashlib
import os
import pathlib
import shutil
from datetime import datetime

try:
    import ecmwf.opendata as opendata
except ImportError:
    opendata = None
import numpy as np
import xarray as xr
from loguru import logger
from s3fs.core import S3FileSystem
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import IFSLexicon
from earth2studio.utils import check_extra_imports
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@check_extra_imports("data", [opendata])
class IFS:
    """The integrated forecast system (IFS) initial state data source provided on an
    equirectangular grid. This data is part of ECMWF's open data project on AWS. This
    data source is provided on a 0.25 degree lat lon grid at 6-hour intervals for the
    most recent 4 days.

    Parameters
    ----------
    source : str, optional
        Data source to fetch data from. For possible options refer to ECMWF's open data
        Python SDK, by default "aws".
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
    This data source only fetches the initial state of control forecast of IFS and does
    not fetch an predicted time steps.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts
    - https://registry.opendata.aws/ecmwf-forecasts/
    - https://console.cloud.google.com/storage/browser/ecmwf-open-data/
    """

    IFS_BUCKET_NAME = "ecmwf-forecasts"
    IFS_LAT = np.linspace(90, -90, 721)
    IFS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, source: str = "aws", cache: bool = True, verbose: bool = True):
        # Optional import not installed error
        if opendata is None:
            raise ImportError(
                "ecmwf-opendata is not installed, install manually or using `pip install earth2studio[data]`"
            )

        self._cache = cache
        self._verbose = verbose
        self.client = opendata.Client(source=source)

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
            return. Must be in IFS lexicon.

        Returns
        -------
        xr.DataArray
            IFS weather data array
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_ifs_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_ifs_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives IFS data array for given date time by fetching variable grib files
        using the ecmwf opendata package and combining grib files into a data array.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in IFS lexicon

        Returns
        -------
        xr.DataArray
            IFS data array for given date time
        """
        ifsda = xr.DataArray(
            data=np.empty((1, len(variables), len(self.IFS_LAT), len(self.IFS_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.IFS_LAT,
                "lon": self.IFS_LON,
            },
        )

        # TODO: Add MP here, can further optimize by combining pressure levels
        # Not doing until tested.
        for i, variable in enumerate(
            tqdm(
                variables, desc=f"Fetching IFS for {time}", disable=(not self._verbose)
            )
        ):
            # Convert from Earth2Studio variable ID to GFS id and modifier
            try:
                ifs_name, modifier = IFSLexicon[variable]
            except KeyError as e:
                logger.error(f"Variable id {variable} not found in IFS lexicon")
                raise e

            variable, levtype, level = ifs_name.split("::")

            logger.debug(f"Fetching IFS grib file for variable: {variable} at {time}")
            grib_file = self._download_ifs_grib_cached(variable, levtype, level, time)
            # Open into xarray data-array
            # Provided [-180, 180], roll to [0, 360]
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            ).roll(longitude=-len(self.IFS_LON) // 2, roll_coords=True)
            ifsda[0, i] = modifier(da.values)

        return ifsda

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for IFS

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for IFS"
                )

            if (datetime.now() - time).days > 4:
                raise ValueError(
                    f"Requested date time {time} needs to be within the past 4 days for IFS"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in IFS")

    def _download_ifs_grib_cached(
        self,
        variable: str,
        levtype: str,
        level: str | list[str],
        time: datetime,
    ) -> str:
        if isinstance(level, str):
            level = [level]

        sha = hashlib.sha256(f"{variable}_{levtype}_{'_'.join(level)}_{time}".encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            request = {
                "date": time,
                "type": "fc",
                "param": variable,
                # "levtype": levtype, # NOTE: Commenting this out fixes what seems to be a bug with Opendata API on soil levels
                "step": 0,  # Would change this for forecasts
                "target": cache_path,
            }
            if levtype == "pl" or levtype == "sl":  # Pressure levels or soil levels
                request["levelist"] = level
            # Download
            self.client.retrieve(**request)

        return cache_path

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "ifs")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_ifs")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the IFS AWS data store

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

        file_name = f"{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}z/"
        s3_uri = f"s3://{cls.IFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists
