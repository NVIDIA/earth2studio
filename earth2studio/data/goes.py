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
import hashlib
import os
import pathlib
import shutil
from datetime import datetime, timezone

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import GOESLexicon
from earth2studio.utils.type import TimeArray, VariableArray


class GOES:
    """GOES (Geostationary Operational Environmental Satellite) data source.

    This data source provides access to GOES-16 and GOES-18 satellite data from AWS S3.
    The data is exclusively ABI (Advanced Baseline Imager) data for now.

    Parameters
    ----------
    satellite : str, optional
        Which GOES satellite to use ('goes16' or 'goes18'), by default 'goes16'
    scan_mode : str, optional
        For ABI: Scan mode ('F' for Full Disk, 'C' for Continental US)
        Mesoscale data is currently not supported due to the changing scan position.
    max_workers : int, optional
        Maximum number of workers for parallel downloads, by default 24
    cache : bool, optional
        Whether to cache downloaded files, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    async_timeout : int, optional
        Timeout for async operations in seconds, by default 600

    Note
    ----
    Beginners Guide to GOES-R Series Data:
    https://noaa-goes16.s3.amazonaws.com/Beginners_Guide_to_GOES-R_Series_Data.pdf

    AWS S3 Bucket:
    https://aws.amazon.com/marketplace/pp/prodview-ngejrbcumyjtu#usage

    ABI Data:
    - 16 spectral bands (C01-C16):
        - C01, C02 (Visible)
        - C03, C04, C05, C06 (Near IR)
        - C07-C16 (IR)
    - Scan modes:
        - Full Disk (F): Entire Earth view
        - Continental US (C): Continental US (20°N-50°N, 125°W-65°W)
    """

    SCAN_TIME_FREQUENCY = {
        "F": 600,
        "C": 300,
    }  # Scan time frequency in seconds
    SCAN_DIMENSIONS = {
        "F": (5424, 5424),
        "C": (1500, 2500),
    }
    VALID_SCAN_MODES = {
        "goes16": ["F", "C"],
        "goes17": ["F", "C"],
        "goes18": ["F", "C"],
        "goes19": ["F", "C"],
    }
    GOES_HISTORY_RANGE = {
        "goes16": (
            datetime(2017, 12, 18),
            datetime(2025, 4, 7),
        ),  # GOES-16 operational from Dec 18, 2017
        "goes17": (
            datetime(2019, 2, 12),
            datetime(2023, 1, 4),
        ),  # GOES-17 operational from Feb 12, 2019
        "goes18": (datetime(2023, 1, 4), None),  # GOES-18 operational from Jan 4, 2023
        "goes19": (datetime(2025, 4, 7), None),  # GOES-19 operational from Apr 7, 2025
    }
    PERSPECTIVE_POINT_HEIGHT = 35786023.0
    SEMI_MAJOR_AXIS = 6378137.0
    SEMI_MINOR_AXIS = 6356752.31414
    LATITUDE_OF_PROJECTION_ORIGIN = 0.0
    LONGITUDE_OF_PROJECTION_ORIGIN = {
        "goes16": -75.0,
        "goes17": -137.0,
        "goes18": -137.0,
        "goes19": -75.0,
    }  # https://www.ospo.noaa.gov/operations/goes/east/fd-img16.html
    FULL_DISK_YX = (
        np.linspace(
            -0.15184399485588074,
            0.15184399485588074,
            5424,
        )[::-1],
        np.linspace(
            -0.15184399485588074,
            0.15184399485588074,
            5424,
        ),
    )
    CONTINENTAL_US_YX = {
        "goes16": (
            np.linspace(
                0.04426800459623337,
                0.12821200489997864,
                1500,
            )[::-1],
            np.linspace(
                -0.10133200138807297,
                0.038612000644207,
                2500,
            ),
        ),
        "goes17": (
            np.linspace(
                0.04426800459623337,
                0.12821200489997864,
                1500,
            )[::-1],
            np.linspace(
                -0.06997200101613998,
                0.06997200101613998,
                2500,
            ),
        ),
        "goes18": (
            np.linspace(
                0.04426800459623337,
                0.12821200489997864,
                1500,
            )[::-1],
            np.linspace(
                -0.06997200101613998,
                0.06997200101613998,
                2500,
            ),
        ),
        "goes19": (
            np.linspace(
                0.04426800459623337,
                0.12821200489997864,
                1500,
            )[::-1],
            np.linspace(
                -0.10133200138807297,
                0.038612000644207,
                2500,
            ),
        ),
    }
    BASE_URL = "s3://noaa-{satellite}/ABI-L2-MCMIP{scan_mode}/{year:04d}/{day_of_year:03d}/{hour:02d}/"

    def __init__(
        self,
        satellite: str = "goes16",
        scan_mode: str = "F",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self._satellite = satellite.lower()
        self._scan_mode = scan_mode.upper()
        self._max_workers = max_workers
        self._cache = cache
        self._verbose = verbose
        self._async_timeout = async_timeout

        # Validate satellite and scan mode
        self._validate_satellite_scan_mode(self._satellite, self._scan_mode)

        # Set up S3 filesystem
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem"""
        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)

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
            return. Must be in the GOES lexicon.

        Returns
        -------
        xr.DataArray
            Data array containing the requested GOES data
        """
        nest_asyncio.apply()  # Patch asyncio to work in notebooks
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self._async_timeout)
        )

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
            Timestamps to return data for
        variable : str | list[str] | VariableArray
            Variables to return using standardized names

        Returns
        -------
        xr.DataArray
            GOES data array
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesn't exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        session = await self.fs.set_session()

        # Create DataArray with appropriate dimensions
        if self._scan_mode == "F":
            y_coords, x_coords = self.FULL_DISK_YX
        else:
            y_coords, x_coords = self.CONTINENTAL_US_YX[self._satellite]
        xr_array = xr.DataArray(
            data=np.zeros(
                (len(time), len(variable), *self.SCAN_DIMENSIONS[self._scan_mode])
            ),
            dims=["time", "variable", "y", "x"],
            coords={
                "time": time,
                "variable": variable,
                "y": y_coords,
                "x": x_coords,
            },
        )

        # Create download tasks
        async_tasks = [(i, t, variable) for i, t in enumerate(time)]
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map, desc="Fetching GOES data", disable=(not self._verbose)
        )

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        # Close aiohttp client
        # https://github.com/fsspec/s3fs/issues/943
        # https://github.com/zarr-developers/zarr-python/issues/2901
        await self.fs.set_session()  # Make sure the session was actually initalized
        s3fs.S3FileSystem.close_session(asyncio.get_event_loop(), self.fs.s3)

        return xr_array

    async def fetch_wrapper(
        self,
        e: tuple[int, datetime, list[str]],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        out = await self.fetch_array(
            time=e[1],
            variable=e[2],
        )
        xr_array[e[0]] = out

    async def fetch_array(
        self,
        time: datetime,
        variable: list[str],
    ) -> np.ndarray:
        """Fetch GOES data array

        Parameters
        ----------
        time : datetime
            Time to get data for
        variable : str
            Variable to get

        Returns
        -------
        np.ndarray
            GOES data array
        """

        # Get the S3 path for the GOES data file
        goes_uri = await self._get_s3_path(time)
        logger.debug(f"Fetching GOES file: {goes_uri}")

        # Download the file to cache
        goes_file = await self._fetch_remote_file(goes_uri)
        # Open into xarray data-array
        da = xr.open_dataset(goes_file)
        x = np.zeros((len(variable), *self.SCAN_DIMENSIONS[self._scan_mode]))

        # Pre-process lexicon lookups to avoid try-except in loop
        variable_mappings = []
        for v in variable:
            if v in GOESLexicon.VOCAB:
                goes_name, modifier = GOESLexicon[v]
                variable_mappings.append((v, goes_name, modifier))
            else:
                logger.warning(f"Variable {v} not found in GOES lexicon, using as is")
                variable_mappings.append((v, v, lambda x: x))

        for i, (v, goes_name, modifier) in enumerate(variable_mappings):
            if modifier is not None:
                x[i] = modifier(da[goes_name].values)
            else:
                x[i] = da[goes_name].values

        return x

    async def _get_s3_path(self, time: datetime) -> str:
        """Get the S3 path for the GOES data file"""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        # Get needed date components
        year = time.year
        day_of_year = time.timetuple().tm_yday
        hour = time.hour

        base_url = self.BASE_URL.format(
            satellite=self._satellite,
            scan_mode=self._scan_mode[0:1],
            year=year,
            day_of_year=day_of_year,
            hour=hour,
        )

        # List files in the directory to find the most recent one
        files = await self.fs._ls(base_url)

        # Filter for files matching the product and scan mode (M1, and M2 will be in the same directory for example)
        pattern = f"OR_ABI-L2-MCMIP{self._scan_mode}"
        matching_files = [f for f in files if pattern in f]

        # Get time stamps from file names
        def get_time(file_name):
            start_str = file_name.split("/")[-1].split("_")[-3][1:-1]
            return datetime.strptime(start_str, "%Y%j%H%M%S")

        time_stamps = [get_time(f) for f in matching_files]

        # Get the index of the file that is the closest to the requested time
        # NOTE: Some of the M1 and M2 files seem to have ~10 min gaps here and there.
        # This fixes this issue by just taking the closest file. Still, some caution
        # is advised. Currently we only support F and C scan modes and those do not
        # have any gaps. Keeping this here for future reference though.
        file_index = np.argmin(np.abs(np.array(time_stamps) - time))

        # Get the file name
        file_name = matching_files[file_index]

        return file_name

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for GOES

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data
        """
        for time in times:
            # Check scan frequency interval
            if (
                not (time - datetime(1900, 1, 1)).total_seconds()
                % self.SCAN_TIME_FREQUENCY[self._scan_mode]
                == 0
            ):
                raise ValueError(
                    f"Requested date time {time} needs to be {self.SCAN_TIME_FREQUENCY[self._scan_mode]} second interval for GOES with scan mode {self._scan_mode}"
                )

            start_date, end_date = self.GOES_HISTORY_RANGE[self._satellite]
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} is before {self._satellite} became operational ({start_date})"
                )
            if end_date and time > end_date:
                raise ValueError(
                    f"Requested date time {time} is after {self._satellite} was retired ({end_date})"
                )

    async def _fetch_remote_file(self, path: str) -> str:
        """Fetches remote file into cache"""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            if self.fs.async_impl:
                data = await self.fs._cat_file(path)
            else:
                data = await asyncio.to_thread(self.fs.read_block, path)
            with open(cache_path, "wb") as file:
                await asyncio.to_thread(file.write, data)

        return cache_path

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "goes")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_goes")
        return cache_location

    @staticmethod
    def _validate_satellite_scan_mode(satellite: str, scan_mode: str) -> None:
        """Validate satellite and scan mode combination.

        Parameters
        ----------
        satellite : str
            Satellite name to validate
        scan_mode : str
            Scan mode to validate

        Raises
        ------
        ValueError
            If satellite or scan mode is invalid
        """
        if satellite not in GOES.VALID_SCAN_MODES:
            raise ValueError(f"Invalid satellite {satellite}")
        if scan_mode not in GOES.VALID_SCAN_MODES[satellite]:
            if scan_mode == "M1" or scan_mode == "M2":
                raise ValueError(
                    f"Mesoscale data ({scan_mode}) is currently not supported by this data source due to the changing scan position."
                )
            else:
                raise ValueError(f"Invalid scan mode {scan_mode} for {satellite}")

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        satellite: str = "goes16",
        scan_mode: str = "F",
    ) -> bool:
        """Checks if given date time is available in the GOES object store

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access
        satellite : str, optional
            Which GOES satellite to check, by default "goes16"
        scan_mode : str, optional
            Which scan mode to check, by default "F"

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        # Validate satellite and scan mode
        cls._validate_satellite_scan_mode(satellite, scan_mode)

        # Check if data exists in S3
        fs = s3fs.S3FileSystem(anon=True)

        # Get needed date components
        year = time.year
        day_of_year = time.timetuple().tm_yday
        hour = time.hour

        # Construct the base URL
        base_url = cls.BASE_URL.format(
            satellite=satellite,
            scan_mode=scan_mode[0:1],
            year=year,
            day_of_year=day_of_year,
            hour=hour,
        )

        # List files in the directory
        try:
            files = fs.ls(base_url)
        except FileNotFoundError:
            return False

        # Filter for files matching the product and scan mode
        pattern = f"OR_ABI-L2-MCMIP{scan_mode}"
        matching_files = [f for f in files if pattern in f]

        if not matching_files:
            return False

        # Sort files by time (same logic as _get_s3_path)
        def get_time(file_name: str) -> datetime:
            start_str = file_name.split("/")[-1].split("_")[-3][1:-1]
            t = datetime.strptime(start_str, "%Y%j%H%M%S")
            if time.tzinfo is not None:
                t = t.replace(tzinfo=timezone.utc)
            return t

        time_stamps = [get_time(f) for f in matching_files]

        # Get the index of the file that is the closest to the requested time
        file_index = np.argmin(np.abs(np.array(time_stamps) - time))

        # Check if the specific file exists
        try:
            matching_files[file_index]
            return True
        except IndexError:
            return False

    @classmethod
    def grid(
        cls, satellite: str = "goes16", scan_mode: str = "F"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lat, lon) in degrees for the native GOES grid.

        Parameters
        ----------
        satellite : str, optional
            Which GOES satellite to use, by default "goes16"
        scan_mode : str, optional
            Scan mode ('F' for Full Disk, 'C' for Continental US), by default "F"

        Note
        ----
        This function is based on the GOES ABI fixed grid projection variables and constants.
        The projection comes from the recommended NOAA documentation:
        https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (lat, lon) in degrees
        """

        # Validate satellite and scan mode
        cls._validate_satellite_scan_mode(satellite, scan_mode)

        # Read in GOES ABI fixed grid projection variables and constants
        if scan_mode == "F":
            x_coordinate_1d = GOES.FULL_DISK_YX[1]  # E/W scanning angle in radians
            y_coordinate_1d = GOES.FULL_DISK_YX[0]  # N/S elevation angle in radians
        else:
            x_coordinate_1d = GOES.CONTINENTAL_US_YX[satellite][
                1
            ]  # E/W scanning angle in radians
            y_coordinate_1d = GOES.CONTINENTAL_US_YX[satellite][
                0
            ]  # N/S elevation angle in radians
        lon_origin = GOES.LONGITUDE_OF_PROJECTION_ORIGIN[satellite]
        H = GOES.PERSPECTIVE_POINT_HEIGHT + GOES.SEMI_MAJOR_AXIS
        r_eq = GOES.SEMI_MAJOR_AXIS
        r_pol = GOES.SEMI_MINOR_AXIS

        # Create 2D coordinate matrices from 1D coordinate vectors
        x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

        # Equations to calculate latitude and longitude
        # Use errstate context manager to suppress invalid operations (e.g., sqrt of negative numbers)
        with np.errstate(invalid="ignore"):
            lambda_0 = (lon_origin * np.pi) / 180.0
            a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
                np.power(np.cos(x_coordinate_2d), 2.0)
                * (
                    np.power(np.cos(y_coordinate_2d), 2.0)
                    + (
                        ((r_eq * r_eq) / (r_pol * r_pol))
                        * np.power(np.sin(y_coordinate_2d), 2.0)
                    )
                )
            )
            b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
            c_var = (H**2.0) - (r_eq**2.0)
            r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (
                2.0 * a_var
            )
            s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
            s_y = -r_s * np.sin(x_coordinate_2d)
            s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

            abi_lat = (180.0 / np.pi) * (
                np.arctan(
                    ((r_eq * r_eq) / (r_pol * r_pol))
                    * (s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y)))
                )
            )
            abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

        return abi_lat, abi_lon
