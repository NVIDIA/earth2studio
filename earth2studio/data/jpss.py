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

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil
from datetime import datetime, timedelta, timezone

import h5py  # type: ignore
import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.jpss import JPSSLexicon
from earth2studio.utils.type import TimeArray, VariableArray


class JPSS:
    """JPSS VIIRS data source for NOAA-20, NOAA-21, and Suomi-NPP supporting both SDR (L1) and EDR (L2) products.

    Parameters
    ----------
    satellite : str, optional
        One of {"noaa-20", "noaa-21", "snpp"}. Default "noaa-20".
    product_type : str, optional
        Product type: "I" for imagery bands (375m), "M" for moderate bands (750m),
        or "L2" for Level 2 EDR products (750m). Default "I".
    max_workers : int, optional
        Maximum concurrent fetch tasks. Default 24.
    cache : bool, optional
        Cache downloaded files under a deterministic hash. Default True.
    verbose : bool, optional
        Show progress bars. Default True.
    async_timeout : int, optional
        Timeout for async operations in seconds. Default 600.

    Note
    ----
    The returned data automatically includes '_lat' and '_lon' variables containing
    latitude and longitude coordinates for each pixel. These are appended to the last
    two dimensions of the data arrays variables as '_lat' and '_lon' variables.

    For EDR (L2) products, not all products are available but generally include:
    - Land Surface Temperature, Surface Albedo, Snow Cover
    - Surface Reflectance, Active Fire Detection
    - Cloud properties (mask, phase, height, optical thickness)
    - Aerosol detection and Volcanic ash products
    """

    BASE_URL = "s3://{bucket}/{product}/{year:04d}/{month:02d}/{day:02d}/"
    SATELLITE_BUCKETS: dict[str, str] = {
        "noaa-20": "noaa-nesdis-n20-pds",
        "noaa-21": "noaa-nesdis-n21-pds",
        "snpp": "noaa-nesdis-snpp-pds",
    }
    VALID_SATELLITES = ["noaa-20", "noaa-21", "snpp"]
    VALID_PRODUCT_TYPES = ["I", "M", "L2"]
    PRODUCT_DIMENSIONS = {
        "I": (1536, 6400),  # I-bands: 1536 x 6400 pixels (375m resolution)
        "M": (768, 3200),  # M-bands: 768 x 3200 pixels (750m resolution)
        "L2": (768, 3200),  # L2 EDR products: 768 x 3200 pixels (750m resolution)
    }
    GEOLOCATION_NAME = {  # Use terrain corrected geolocation files (TC)
        "I": "VIIRS-IMG-GEO-TC",
        "M": "VIIRS-MOD-GEO-TC",
        "L2": "VIIRS-MOD-GEO-TC",
    }

    def __init__(
        self,
        satellite: str = "noaa-20",
        product_type: str = "I",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:

        self._validate_satellite_and_product_type(satellite, product_type)

        self._satellite: str = satellite
        self._product_type: str = product_type
        self._max_workers: int = max_workers
        self._cache: bool = cache
        self._verbose: bool = verbose
        self._async_timeout: int = async_timeout

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
            return. Must be in the JPSS lexicon.

        Returns
        -------
        xr.DataArray
            Data array containing the requested JPSS data
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self._async_timeout)
        )

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
            return. Must be in the JPSS lexicon.

        Returns
        -------
        xr.DataArray
            Data array containing the requested JPSS data
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this function directly, make sure the data source is initialized inside the async loop!"
            )

        # Prepare data inputs
        time, variable = prep_data_inputs(time, variable)

        # Validate variables
        JPSS._validate_variables(variable, self._product_type)

        # Create cache directory if it doesn't exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Add lat/lon to the variable list
        extended_variables = list(variable) + ["_lat", "_lon"]

        session = await self.fs.set_session()

        # Determine array dimensions
        y_size, x_size = self.PRODUCT_DIMENSIONS[self._product_type]

        # Create array with extended variables (original + lat + lon)
        xr_array = xr.DataArray(
            data=np.zeros(
                (len(time), len(extended_variables), y_size, x_size), dtype=np.float32
            ),
            dims=["time", "variable", "y", "x"],
            coords={
                "time": time,
                "variable": extended_variables,
                "y": np.arange(y_size),
                "x": np.arange(x_size),
            },
            attrs={
                "time_data": {},  # Will store {time_index: actual_timestamp}
                "time_geo": {},  # Will store {time_index: actual_timestamp}
                "file_data": {},  # Will store {time_index: filename}
                "file_geo": {},  # Will store {time_index: filename}
            },
        )

        # Fetch data
        async_tasks = [
            (i, t, j, v) for i, t in enumerate(time) for j, v in enumerate(variable)
        ]
        await tqdm.gather(
            *map(
                lambda args: self.fetch_data_wrapper(args, xr_array=xr_array),
                async_tasks,
            ),
            desc="Fetching VIIRS data",
            disable=(not self._verbose),
        )

        # Fetch geolocation data
        async_tasks = [(i, t) for i, t in enumerate(time)]
        await tqdm.gather(
            *map(
                lambda args: self.fetch_geolocation_wrapper(args, xr_array=xr_array),
                async_tasks,
            ),
            desc="Fetching VIIRS geolocation data",
            disable=(not self._verbose),
        )

        # Validate that data and geolocation timestamps match are within 1 second
        # NOTE: This should never happen but I will feel better leaving the check here.
        # Leaving file names in xarray attrs for any future debugging purposes
        for i in range(len(time)):
            time_diff = abs(
                xr_array.attrs["time_data"][i] - xr_array.attrs["time_geo"][i]
            )
            if time_diff > timedelta(seconds=1):
                raise ValueError(
                    f"Data and geolocation timestamps do not match. Something is going wrong with this request! {time_diff}"
                )

        if session:
            await session.close()

        # Close aiohttp client cleanly as in GOES implementation
        await self.fs.set_session()
        s3fs.S3FileSystem.close_session(asyncio.get_event_loop(), self.fs.s3)

        return xr_array

    async def fetch_data_wrapper(
        self,
        e: tuple[int, datetime, int, str],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        # Fetch data variables
        data_out, timestamp, filename = await self.fetch_data(time=e[1], variable=e[3])
        xr_array[e[0], e[2]] = data_out
        xr_array.attrs["time_data"][e[0]] = timestamp
        xr_array.attrs["file_data"][e[0]] = filename

    async def fetch_data(
        self,
        time: datetime,
        variable: str,
    ) -> tuple[np.ndarray, datetime, str]:
        """Fetch VIIRS data array

        Parameters
        ----------
        time : datetime
            Time to get data for
        variable : str
            Variable name to fetch

        Returns
        -------
        tuple[np.ndarray, datetime, str]
            VIIRS data array, timestamp, and filename
        """

        # Map standardized variables to (product_type, folder, dataset, modifier)
        product_type, folder, dataset_name, modifier = JPSSLexicon.get_item(variable)

        # Find file path on S3 for this time and product
        s3_uri, timestamp = await self._get_s3_path(
            time=time, variable=variable, geolocation=False
        )

        # Extract filename from S3 URI
        filename = s3_uri.split("/")[-1]

        # Fetch the file from S3
        local_file = await self._fetch_remote_file(s3_uri)

        # NetCDF file - use xarray
        with h5py.File(local_file, "r") as h5:
            # Get dataset from HDF5 file
            if self._product_type in ["I", "M"]:
                ds = h5["All_Data"][folder + "_All"][dataset_name]
            else:
                ds = h5[dataset_name]

            # Get data from dataset
            data = ds[...]

            # Filter VIIRS fill values BEFORE converting to float32 to preserve original dtype
            filtered_data = self._filter_fill_values(data, ds)

            # Apply modifier
            processed_data = modifier(filtered_data)

        return processed_data, timestamp, filename

    async def fetch_geolocation_wrapper(
        self,
        e: tuple[int, datetime],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        lat, lon, timestamp, filename = await self.fetch_geolocation(time=e[1])
        xr_array[e[0], -2] = lat
        xr_array[e[0], -1] = lon
        xr_array.attrs["time_geo"][e[0]] = timestamp
        xr_array.attrs["file_geo"][e[0]] = filename

    async def fetch_geolocation(
        self, time: datetime
    ) -> tuple[np.ndarray, np.ndarray, datetime, str]:
        """Fetch geolocation data from VIIRS geolocation files."""

        # Determine geolocation product based on product type
        geo_product = self.GEOLOCATION_NAME[self._product_type]

        # Find file path on S3 for this time and product
        s3_uri, timestamp = await self._get_s3_path(
            time=time, variable=geo_product, geolocation=True
        )

        # Extract filename from S3 URI
        filename = s3_uri.split("/")[-1]

        # Fetch the file from S3
        local_file = await self._fetch_remote_file(s3_uri)

        with h5py.File(local_file, "r") as h5:
            lat_ds = h5["All_Data"][list(h5["All_Data"].keys())[0]][
                "Latitude"
            ]  # Kind of a hacky way to get the lat/lon data
            lon_ds = h5["All_Data"][list(h5["All_Data"].keys())[0]]["Longitude"]
            lat = np.asarray(lat_ds[...], dtype=np.float32)
            lon = np.asarray(lon_ds[...], dtype=np.float32)

        return lat, lon, timestamp, filename

    def _filter_fill_values(
        self, data: np.ndarray, dataset: h5py.Dataset
    ) -> np.ndarray:
        """Filter VIIRS fill values and convert them to NaN.

        VIIRS uses integer fill values instead of NaN for invalid pixels (bow-tie deletions,
        truncated scans, quality failures, etc.). This method converts those sentinel values to
        NaN for downstream processing.

        References
        ----------
        - NOAA NESDIS. (2023). *JPSS Algorithm Specification, Volume II: Data Dictionary
          for the VIIRS RDR/SDR* (474-00448-02-06, Rev. L). Retrieved September 29, 2025,
          from https://www.nesdis.noaa.gov/s3/2024-01/474-00448-02-06_JPSS-VIIRS-SDR-DD-Part-6_L.pdf

        Parameters
        ----------
        data : np.ndarray
            The data array to filter
        dataset : h5py.Dataset
            The HDF5 dataset (used to check for fill value attributes)

        Returns
        -------
        np.ndarray
            Data array with fill values converted to NaN
        """

        # Make a copy and convert to float to allow NaN assignment
        filtered_data = data.astype(np.float32)

        # Filter fill values
        if self._product_type in ["I", "M"]:
            filter_threshold = 65529
            filtered_data[filtered_data >= filter_threshold] = np.nan
            filtered_data[filtered_data < 0] = np.nan  # Issue with data
        elif self._product_type == "L2":
            fill_value = dataset.attrs["_FillValue"].item()
            filtered_data[filtered_data == fill_value] = np.nan

        return filtered_data

    @staticmethod
    def _validate_satellite_and_product_type(satellite: str, product_type: str) -> None:
        """Validate that requested satellite and product type are valid

        Parameters
        ----------
        satellite : str
            Satellite name to validate
        product_type : str
            Product type to validate

        Raises
        ------
        ValueError
            If satellite or product type is invalid
        """
        if satellite not in JPSS.VALID_SATELLITES:
            raise ValueError(
                f"Invalid satellite {satellite}. Must be one of {JPSS.VALID_SATELLITES}"
            )
        if product_type not in JPSS.VALID_PRODUCT_TYPES:
            raise ValueError(
                f"Invalid product_type {product_type}. Must be one of {JPSS.VALID_PRODUCT_TYPES}"
            )

    @staticmethod
    def _validate_variables(variables: list[str], product_type: str = None) -> None:
        """Validate that requested variables are compatible with the configured product type.

        Parameters
        ----------
        variables : list[str]
            List of variable names to validate
        product_type : str, optional
            Product type to validate against. If None, only checks if variables exist.

        Raises
        ------
        ValueError
            If variables have incompatible product types or invalid format
        """
        unknown_vars = []
        non_product_vars = []

        for var in variables:
            # Check if variable exists in lexicon
            if var not in JPSSLexicon.VOCAB:
                unknown_vars.append(var)
                continue

            # Determine variable type if product_type is provided
            if (
                product_type is not None
                and JPSSLexicon.get_item(var)[0] != product_type
            ):
                non_product_vars.append(var)

        # Raise error for unknown variables
        if unknown_vars:
            raise ValueError(
                f"Unknown VIIRS variables: {unknown_vars}. "
                f"Please check the JPSS lexicon for available variables."
            )

        # Raise error for non-product variables
        if non_product_vars:
            raise ValueError(
                f"Variables {non_product_vars} are incompatible with the configured product type {product_type}"
            )

    async def _get_s3_path(
        self, time: datetime, variable: str, geolocation: bool = False
    ) -> tuple[str, datetime]:
        """Get the S3 path for a given time and product code.

        Returns
        -------
        tuple[str, datetime]
            S3 file path and actual file timestamp
        """

        # Get the bucket for the satellite
        bucket = self.SATELLITE_BUCKETS[self._satellite]

        # Determine folder
        if not geolocation:
            _, folder, _, _ = JPSSLexicon.get_item(variable)
        else:
            folder = self.GEOLOCATION_NAME[self._product_type]

        # Get the base URL
        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )

        # List files in the directory and choose the closest timestamp file
        if self.fs is None:  # type: ignore
            raise ValueError("File system is not initialized")  # type: ignore
        files = await self.fs._ls(base_url)

        # Filter for data files
        data_files = [f for f in files if f.endswith((".h5", ".nc"))]

        # Raise error if no matching files are found
        if not data_files:
            raise FileNotFoundError(
                f"No VIIRS {self._product_type} files found at {base_url} for {variable}, desired time {time}"
            )

        # Get time stamp from filename
        def get_time(path: str) -> datetime:
            """Get observation start time from filename."""
            filename = path.split("/")[-1]
            parts = filename.split("_")

            # Yup, L1 and L2 have completely different filename formats
            # Find the start time part (starts with 't' for L1 products)
            for part in parts:
                if part.startswith("t") and len(part) >= 8:
                    time_str = part[
                        1:8
                    ]  # Remove 't' and take first 7 digits (HHMMSS + fractional)
                    # Convert to HHMMSS format
                    hours = time_str[:2]
                    minutes = time_str[2:4]
                    seconds = time_str[4:6]

                    # Get date from 'd' part
                    date_part = None
                    for p in parts:
                        if p.startswith("d") and len(p) >= 9:
                            date_part = p[1:9]  # Remove 'd' and get YYYYMMDD
                            break

                    if date_part:
                        datetime_str = f"{date_part}{hours}{minutes}{seconds}"
                        return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")

            # Fallback to old method if parsing fails (L2 products)
            start_str = path.split("/")[-1].split("_")[-3][1:]
            trimmed = start_str[:14]
            return datetime.strptime(trimmed, "%Y%m%d%H%M%S")

        time_stamps = [get_time(f) for f in data_files]

        # Get the index of the file that is the closest to the requested time
        idx = int(np.argmin(np.abs(np.array(time_stamps) - time)))

        # Get the file name
        return data_files[idx], time_stamps[idx]

    async def _fetch_remote_file(self, path: str) -> str:
        """Fetch remote file into cache"""
        if self.fs is None:
            raise ValueError("File system is not initialized")
        sha = hashlib.sha256(path.encode()).hexdigest()
        cache_path = os.path.join(self.cache, sha)
        if not pathlib.Path(cache_path).is_file():
            if self.fs.async_impl:
                data = await self.fs._cat_file(path)
            else:
                data = await asyncio.to_thread(self.fs.read_block, path)
            with open(cache_path, "wb") as f:
                await asyncio.to_thread(f.write, data)
        return cache_path

    @property
    def cache(self) -> str:
        """Get the cache location"""
        cache_location = os.path.join(datasource_cache_root(), "jpss")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_jpss")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        variable: str,
        satellite: str = "noaa-20",
        product_type: str = "I",
    ) -> bool:
        """Checks if given date time is avaliable in JPSS

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access
        variable : str
            Variable to check
        satellite : str, optional
            Satellite to check
        product_type : str, optional
            Product type to check

        Returns
        -------
        bool
            If date time is avaiable
        """
        # Convert np.datetime64 to datetime
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        # Validate satellite and product type
        cls._validate_satellite_and_product_type(satellite, product_type)

        # Validate variable
        JPSS._validate_variables([variable], product_type)

        # Get the bucket for the satellite
        bucket = cls.SATELLITE_BUCKETS[satellite]

        # Determine folder
        _, folder, _, _ = JPSSLexicon.get_item(variable)
        geolocation_folder = cls.GEOLOCATION_NAME[product_type]

        # Get the base URL
        base_url = cls.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )
        geolocation_base_url = cls.BASE_URL.format(
            bucket=bucket,
            product=geolocation_folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )

        # List files in the directory and choose the closest timestamp file
        try:
            fs = s3fs.S3FileSystem(anon=True)
            _ = fs.ls(base_url)
            _ = fs.ls(geolocation_base_url)
        except FileNotFoundError:
            return False

        return True
