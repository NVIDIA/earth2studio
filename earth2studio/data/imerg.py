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

import os
import pathlib
import shutil
from datetime import datetime, timedelta

import aiohttp
import numpy as np
import requests
import xarray as xr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import IMERGLexicon
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class IMERG:
    """The Integrated Multi-satellitE Retrievals (IMERG) for GPM. IMERG is a NASA
    product that estimates the global surface precipitation rates at a high resolution
    of 0.1 degree every half-hour beginning in 2000. Provides total (m/hr), probability
    and index precipitation fields for the past half hour.

    Note
    ----
    This data source requires users to register for NASA's Earthdata portal:
    https://urs.earthdata.nasa.gov/. Users must supply their username and password
    for AuthN when using this data source. Users should be sure "NASA GESDISC DATA
    ARCHIVE" is an approved application on their profile:
    https://urs.earthdata.nasa.gov/profile

    Parameters
    ----------
    auth : aiohttp.BasicAuth | None
        BasicAuth object with user's EarthData username and password for basic HTTP
        authentication. If none is provided, one will be constructed using the
        `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables, by default
        None
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

    - https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary
    - https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07
    """

    # PYDap doesnt work: https://github.com/pydap/pydap/issues/188

    IMERG_LAT = np.linspace(89.95, -89.95, 1800)
    IMERG_LON = np.linspace(0.05, 359.95, 3600)

    def __init__(
        self,
        auth: aiohttp.BasicAuth | None = None,
        cache: bool = True,
        verbose: bool = True,
    ):
        self._cache = cache
        self._verbose = verbose

        cache_options: dict[str, str | int] = {}
        cache_options["cache_storage"] = self.cache
        if cache:
            cache_options["expiry_time"] = 31622400
        else:
            cache_options["expiry_time"] = 30

        if auth is None:
            if (
                "EARTHDATA_USERNAME" not in os.environ
                or "EARTHDATA_PASSWORD" not in os.environ
            ):
                raise ValueError(
                    "Both environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set to NASA Earthdata credentials for IMERG data source."
                )
            auth = aiohttp.BasicAuth(
                os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"]
            )
        fs = HTTPFileSystem(block_size=2**20, client_kwargs={"auth": auth})
        self.fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve IMERG data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the IMERG lexicon.

        Returns
        -------
        xr.DataArray
            IMERG data array
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:  # TODO: Tqdm
            data_array = self.fetch_imerg_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_imerg_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives IMERG H5 file for time and moves it into an xarray data array. This
        will download all variables / coords for this specific time since OpenDAP isn't
        working.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in IMERG lexicon

        Returns
        -------
        xr.DataArray
            IMERG data array for given date time

        Raises
        ------
        KeyError
            Unsupported variable
        """
        logger.debug(f"Fetching IMERG H5 file for: {time}")
        h5_file = self.fs.open(self.get_file_url(time)).name

        # Roll raw data from [-180,180] to [0,360] and reverse lat from [-90,90] to [90,90]
        raw_ds = (
            xr.open_dataset(h5_file, group="Grid")
            .roll(lon=-len(self.IMERG_LON) // 2, roll_coords=True)
            .isel(lat=slice(None, None, -1))
        )

        imergda = xr.DataArray(
            data=np.empty(
                (1, len(variables), len(self.IMERG_LAT), len(self.IMERG_LON))
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.IMERG_LAT,
                "lon": self.IMERG_LON,
            },
        )

        for i, variable in enumerate(variables):
            # Convert from Earth2Studio variable ID to IMERG id and modifier
            try:
                imerg_name, modifier = IMERGLexicon[variable]
            except KeyError as e:
                logger.error(f"variable id {variable} not found in IMERG lexicon.")
                raise KeyError(str(e))

            data = np.nan_to_num(raw_ds[imerg_name].isel(time=0).values.T)
            imergda[0, i] = modifier(data)

        return imergda

    @classmethod
    def get_file_url(cls, time: datetime) -> str:
        """Returns the IMERG H5 file for a requested date

        Note
        ----
        See repository for more information on what files are available:
        https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/

        Parameters
        ----------
        time : datetime
            Date time of data to get

        Returns
        -------
        str
            Data file url
        """
        dataset_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"

        start_time = time - timedelta(minutes=30)
        end_time = time - timedelta(seconds=1)

        date_str = start_time.strftime("%Y%m%d")
        start_str = start_time.strftime("%H%M%S")
        end_str = end_time.strftime("%H%M%S")
        min_idx = 60 * start_time.hour + start_time.minute

        year = start_time.year
        yrday = start_time.timetuple().tm_yday

        file_name = f"3B-HHR.MS.MRG.3IMERG.{date_str}-S{start_str}-E{end_str}.{min_idx:04d}.V07B.HDF5"
        return f"{dataset_url}/{year}/{yrday:03d}/{file_name}"

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date times are valid for IMERG based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 1800 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 30 minute interval for IMERG"
                )

            if time < datetime(year=2000, month=6, day=1, minute=30):
                raise ValueError(
                    f"Requested date time {time} needs to be after June 1st, 2000 for IMERG"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "imerg")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_imerg")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in IMERG

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

        response = requests.get(cls.get_file_url(time), timeout=5)
        # OK to 403, but 404 means file doesnt exist
        return response.status_code < 404
