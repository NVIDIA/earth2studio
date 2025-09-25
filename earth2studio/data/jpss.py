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
from datetime import datetime, timezone


import h5py  # type: ignore
import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from loguru import logger
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
    band_type : str, optional
        Band resolution type: "I" for imagery bands (375m), "M" for moderate bands (750m), 
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
    PRODUCT_FOLDER = {
        "I": "VIIRS-I{band_num}-SDR",
        "M": "VIIRS-M{band_num}-SDR",
        "L2": "VIIRS-MOD-L2",
    }
    GEOLOCATION_NAME = {
        "I": "VIIRS-IMG-GEO",
        "M": "VIIRS-MOD-GEO",
        "L2": "VIIRS-MOD-GEO",
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
        if satellite not in self.VALID_SATELLITES:
            raise ValueError(f"Invalid satellite {satellite}. Must be one of {self.VALID_SATELLITES}")
        if product_type not in self.VALID_PRODUCT_TYPES:
            raise ValueError(f"Invalid product_type {product_type}. Must be one of {self.VALID_PRODUCT_TYPES}")

        self._satellite: str = satellite
        self._product_type: str = product_type
        self._max_workers = max_workers
        self._cache = cache
        self._verbose = verbose
        self._async_timeout = async_timeout

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

        # Validate product types
        self._validate_product_types(variable)

        # Create cache directory if it doesn't exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Add lat/lon to the variable list
        extended_variables = list(variable) + ["_lat", "_lon"]

        session = await self.fs.set_session()

        # Determine array dimensions
        y_size, x_size = self.PRODUCT_DIMENSIONS[self._product_type]
        
        # Create array with extended variables (original + lat + lon)
        xr_array = xr.DataArray(
            data=np.zeros((len(time), len(extended_variables), y_size, x_size), dtype=np.float32),
            dims=["time", "variable", "y", "x"],
            coords={
                "time": time,
                "variable": extended_variables,
                "y": np.arange(y_size),
                "x": np.arange(x_size),
            },
        )

        # Fetch data
        async_tasks = [(i, t, j, v) for i, t in enumerate(time) for j, v in enumerate(variable)]
        await tqdm.gather(
            *map(lambda args: self.fetch_data_wrapper(args, xr_array=xr_array), async_tasks),
            desc="Fetching VIIRS data",
            disable=(not self._verbose),
        )

        # Fetch geolocation data
        async_tasks = [(i, t) for i, t in enumerate(time)]
        await tqdm.gather(
            *map(lambda args: self.fetch_geolocation_wrapper(args, xr_array=xr_array), async_tasks),
            desc="Fetching VIIRS geolocation data",
            disable=(not self._verbose),
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
        data_out = await self.fetch_data(time=e[1], variable=e[3])
        xr_array[e[0], e[2]] = data_out

    async def fetch_data(
        self,
        time: datetime,
        variable: str,
    ) -> np.ndarray:
        """Fetch VIIRS data array

        Parameters
        ----------
        time : datetime
            Time to get data for
        variable : str
            Variable name to fetch

        Returns
        -------
        np.ndarray
            VIIRS data array
        """

        # Map standardized variables to (product_code, dataset, modifier)
        product_identifier, modifier, _ = JPSSLexicon[variable]
        product_code, dataset_name = product_identifier.split("/")

        # Find file path on S3 for this time and product
        s3_uri = await self._get_s3_path(time=time, product_code=product_code)

        # Fetch the file from S3
        local_file = await self._fetch_remote_file(s3_uri)

        # NetCDF file - use xarray
        with h5py.File(local_file, "r") as h5:
            # Get dataset from HDF5 file
            if self._product_type in ["I", "M"]:
                for _, grp in h5["All_Data"].items():
                    if isinstance(grp, h5py.Group) and dataset_name in grp:
                        ds = grp[dataset_name]
            else:
                ds = h5[dataset_name]

            # Get data from dataset
            data = ds[...]

            # Filter VIIRS fill values BEFORE converting to float32 to preserve original dtype
            filtered_data = self._filter_fill_values(data, ds)

            # Apply modifier
            processed_data = modifier(filtered_data)

        return processed_data

    async def fetch_geolocation_wrapper(
        self,
        e: tuple[int, datetime],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        geolocation_out = await self.fetch_geolocation(time=e[1])
        xr_array[e[0], -2:] = geolocation_out

    async def fetch_geolocation(self, time: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Fetch geolocation data from VIIRS geolocation files."""

        # Determine geolocation product based on product type
        geo_product = self.GEOLOCATION_NAME[self._product_type]

        # Find file path on S3 for this time and product
        s3_uri = await self._get_s3_path(time=time, product_code=geo_product, geolocation=True)

        # Fetch the file from S3
        local_file = await self._fetch_remote_file(s3_uri)

        # Download and read the geolocation file
        local_geo_file = await self._fetch_remote_file(local_file)
        
        with h5py.File(local_geo_file, "r") as h5:
            # Try to find latitude and longitude datasets within the file
            # They're often in geolocation groups or alongside the data
            lat_ds = None
            lon_ds = None

            #
            
            # Common locations for geolocation in VIIRS SDR files
            geo_search_patterns = [
                "Latitude",
                "Longitude", 
                "Navigation/Latitude",
                "Navigation/Longitude",
                "Geolocation/Latitude",
                "Geolocation/Longitude"
            ]
            
            # Search for lat/lon datasets
            for pattern in geo_search_patterns:
                try:
                    if pattern == "Latitude":
                        lat_ds = self._find_dataset(h5, "Latitude")
                    elif pattern == "Longitude":
                        lon_ds = self._find_dataset(h5, "Longitude")
                    else:
                        # Try path-based access
                        if pattern in h5:
                            if "Latitude" in pattern:
                                lat_ds = h5[pattern]
                            else:
                                lon_ds = h5[pattern]
                except (KeyError, ValueError):
                    continue
            
            # If not found in common locations, do exhaustive search
            if lat_ds is None or lon_ds is None:
                def find_geo_datasets(name, obj):
                    nonlocal lat_ds, lon_ds
                    if isinstance(obj, h5py.Dataset):
                        if "latitude" in name.lower() and lat_ds is None:
                            lat_ds = obj
                        elif "longitude" in name.lower() and lon_ds is None:
                            lon_ds = obj
                
                h5.visititems(find_geo_datasets)
            
            if lat_ds is None or lon_ds is None:
                # Fallback: create dummy coordinates based on array indices
                logger.warning("Could not find geolocation data in SDR file, creating dummy coordinates")
                # Get data shape from any dataset in the file
                data_shape = None
                def find_data_shape(name, obj):
                    nonlocal data_shape
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2 and data_shape is None:
                        data_shape = obj.shape
                
                h5.visititems(find_data_shape)
                
                if data_shape is None:
                    raise ValueError("Could not determine data dimensions from SDR file")
                
                # Create dummy lat/lon grids
                lat = np.full(data_shape, np.nan, dtype=np.float32)
                lon = np.full(data_shape, np.nan, dtype=np.float32)
                return lat, lon
            
            lat = np.asarray(lat_ds[...], dtype=np.float32)
            lon = np.asarray(lon_ds[...], dtype=np.float32)
            
            return lat, lon

    def _filter_fill_values(self, data: np.ndarray, dataset: h5py.Dataset) -> np.ndarray:
        """Filter VIIRS fill values and convert them to NaN.
        
        VIIRS uses fill values instead of NaN for invalid pixels (bow-tie effect, etc).
        This method identifies and converts these to NaN for proper handling.
        
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
        
        # Determine fill value
        if self._product_type in ["I", "M"]:
            fill_value = 65535
        elif self._product_type == "L2":
            fill_value = dataset.attrs['_FillValue'].item()
        
        # Apply explicit fill value
        filtered_data[filtered_data == fill_value] = np.nan
        
        return filtered_data

    def _validate_product_types(self, variables: list[str]) -> None:
        """Validate that requested variables are compatible with the configured product type.
        
        Parameters
        ----------
        variables : list[str]
            List of variable names to validate
            
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
                
            # Determine variable type
            if JPSSLexicon.get_item(var)[2] != self._product_type:
                non_product_vars.append(var)

        # Raise error for unknown variables
        if unknown_vars:
            raise ValueError(
                f"Unknown VIIRS variables: {unknown_vars}. "
                f"Please check the JPSS lexicon for available variables."
            )

        # Raise error for non-product variables
        if non_product_vars:
            raise ValueError(f"Variables {non_product_vars} are incompatible with the configured product type {self._product_type}")

    async def _get_s3_path(self, time: datetime, product_code: str, geolocation: bool = False) -> str:
        """Get the S3 path for a given time and product code."""

        # Get the bucket for the satellite
        bucket = self.SATELLITE_BUCKETS[self._satellite]

        # Determine folder name based on product type
        if self._product_type == "I":
            # SDR I-band products
            band_num = product_code[-2:]
            folder = f"VIIRS-I{int(band_num)}-SDR" if not geolocation else product_code
        elif self._product_type == "M":
            # SDR M-band products
            band_num = product_code[-2:]
            folder = f"VIIRS-M{int(band_num)}-SDR" if not geolocation else product_code
        elif self._product_type == "L2":
            # L2 EDR products - use the product code as the folder name
            folder = product_code
        else:
            raise ValueError(f"Invalid product type {self._product_type}")

        # Get the base URL
        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )

        # List files in the directory and choose the closest timestamp file
        files = await self.fs._ls(base_url)

        # Filter for data files
        data_files = [f for f in files if f.endswith((".h5", ".nc"))]

        # Match by product code in filename
        matching_files = [f for f in data_files if f"/{product_code}_" in f]

        # Raise error if no matching files are found
        if not matching_files:
            raise FileNotFoundError(f"No VIIRS {self._product_type} files found at {base_url} for {product_code}, desired time {time}")

        # Get time stamp from filename
        def get_time(path: str) -> datetime:
            """Get time stamp from filename."""
            start_str = path.split("/")[-1].split("_")[-3][1:]
            trimmed = start_str[:14] # Trim any fractional seconds beyond the first 14 characters
            return datetime.strptime(trimmed, "%Y%m%d%H%M%S")
        time_stamps = [get_time(f) for f in matching_files]

        # Get the index of the file that is the closest to the requested time
        idx = int(np.argmin(np.abs(np.array(time_stamps) - time)))

        # Get the file name
        return matching_files[idx]

    async def _fetch_remote_file(self, path: str) -> str:
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
        cache_location = os.path.join(datasource_cache_root(), "jpss")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_jpss")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        satellite: str = "noaa-20",
        product_type: str = "I",
        variable: str = "viirs1i",
    ) -> bool:
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        if satellite not in cls.VALID_SATELLITES:
            raise ValueError(f"Invalid satellite {satellite}. Must be one of {cls.VALID_SATELLITES}")

        try:
            product_identifier, _ = JPSSLexicon[variable]
            product_code = product_identifier.split("/")[0]
        except (KeyError, ValueError):
            return False

        bucket = cls.SATELLITE_BUCKETS[satellite]
        
        # Determine folder name based on product type
        if product_code.startswith("SVI"):
            # SDR I-band products
            band_num = product_code[-2:]
            folder = f"VIIRS-I{int(band_num)}-SDR"
        elif product_code.startswith("SVM"):
            # SDR M-band products
            band_num = product_code[-2:]
            folder = f"VIIRS-M{int(band_num)}-SDR"
        else:
            # L2 EDR products - use the product code as the folder name
            folder = product_code

        fs = s3fs.S3FileSystem(anon=True)
        base_url = cls.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )
        try:
            files = fs.ls(base_url)
        except FileNotFoundError:
            return False

        # Filter for data files based on product type
        if product_code.startswith(("SVI", "SVM")):
            # SDR products use HDF5 format
            data_files = [f for f in files if f.endswith(".h5")]
            if not data_files:
                return False
            # Match by product code in filename
            matching_files = [f for f in data_files if f"/{product_code}_" in f]
            return len(matching_files) > 0
        else:
            # L2 EDR products may use HDF5 or NetCDF format
            data_files = [f for f in files if f.endswith((".h5", ".nc"))]
            return len(data_files) > 0


