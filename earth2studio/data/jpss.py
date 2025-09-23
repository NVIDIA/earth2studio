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
        "M": (1600, 2712),  # M-bands: 1600 x 2712 pixels (750m resolution)
        "L2": (1600, 2712),  # L2 EDR products: 1600 x 2712 pixels (750m resolution)
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
        e: tuple[int, datetime, list[str]],
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
        product_identifier, modifier = JPSSLexicon[variable]
        product_code, dataset_name = product_identifier.split("/")

        # Find file path on S3 for this time and product
        s3_uri = await self._get_s3_path(time=time, product_code=product_code)

        # Fetch the file from S3
        local_file = await self._fetch_remote_file(s3_uri)

        # NetCDF file - use xarray
        with xr.open_dataset(local_file) as ds:
            # Load the data
            data = ds[dataset_name].values

            # Filter VIIRS fill values
            filtered_data = self._filter_fill_values(data, ds)
            
            # Apply modifier and convert to float32
            processed_data = np.asarray(modifier(filtered_data), dtype=np.float32)

        return processed_data

    async def fetch_geolocation_wrapper(
        self,
        e: tuple[int, datetime],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        geolocation_out = await self.fetch_geolocation(time=e[1])
        xr_array[e[0]] = geolocation_out

    async def fetch_geolocation(self, time: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Fetch geolocation data from VIIRS geolocation files."""
        # Determine geolocation product based on band type
        # From S3 exploration: VIIRS-IMG-GEO for I-bands, VIIRS-MOD-GEO for M-bands
        # For L2 products, try to use M-band geolocation as default (can be overridden later if needed)
        if self._product_type == "I":
            geo_product = "VIIRS-IMG-GEO"
        elif self._product_type == "M":
            geo_product = "VIIRS-MOD-GEO"
        else:  # L2 products
            geo_product = "VIIRS-MOD-GEO"  # Default to M-band resolution for L2 products

        # Find file path on S3 for this time and product
        bucket = self.SATELLITE_BUCKETS[self._satellite]
        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=geo_product,
            year=time.year,
            month=time.month,
            day=time.day,
        )
        
        try:
            files = await self.fs._ls(base_url)
        except Exception:
            raise FileNotFoundError(f"No {geo_product} geolocation files found for {time}")

        # Find matching geolocation file
        matching_files = [f for f in files if f.endswith(".h5")]
        if not matching_files:
            raise FileNotFoundError(f"No {geo_product} geolocation files found")
            
        def parse_start_dt(path: str) -> datetime:
            fname = path.split("/")[-1]
            parts = fname.split("_")
            # Look for date and time parts in filename
            for part in parts:
                if part.startswith("d") and len(part) == 9:  # dYYYYMMDD
                    date_str = part[1:]
                elif part.startswith("t") and len(part) >= 8:  # tHHMMSSs
                    time_str = part[1:7]
            try:
                dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                return dt.replace(tzinfo=timezone.utc)
            except:
                # Fallback: use file modification time or current time
                return time
                
        time_stamps = [parse_start_dt(f) for f in matching_files]
        idx = int(np.argmin(np.abs(np.array(time_stamps, dtype="datetime64[s]") - np.datetime64(time))))
        geo_file_path = matching_files[idx]
        
        # Download and read the geolocation file
        local_geo_file = await self._fetch_remote_file(geo_file_path)
        
        with h5py.File(local_geo_file, "r") as h5:
            # Try to find latitude and longitude datasets within the file
            # They're often in geolocation groups or alongside the data
            lat_ds = None
            lon_ds = None
            
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
        
        # First, check if the dataset has explicit fill value attributes
        fill_value = None
        for attr_name in ['_FillValue', 'missing_value', 'fill_value']:
            if attr_name in dataset.attrs:
                fill_value = dataset.attrs[attr_name]
                if isinstance(fill_value, np.ndarray):
                    fill_value = fill_value.item()
                logger.debug(f"Found fill value attribute {attr_name}: {fill_value}")
                break
        
        # Apply explicit fill value if found
        if fill_value is not None:
            filtered_data[filtered_data == fill_value] = np.nan
        
        # Apply common VIIRS fill value patterns
        if data.dtype in [np.uint16, np.uint8]:
            # For unsigned integer data, values near maximum are typically fill
            if data.dtype == np.uint16:
                # VIIRS commonly uses these specific fill values
                common_fill_values = [65535, 65534, 65533, 65532, 65531, 65530, 65529]  # Check specific values first
                for fv in common_fill_values:
                    filtered_data[filtered_data == fv] = np.nan
                
                # For VIIRS, if we see uniform values across entire array, it's likely all fill
                if not np.all(np.isnan(filtered_data)):
                    valid_data = filtered_data[~np.isnan(filtered_data)]
                    if len(valid_data) > 0:
                        # Check if all values are the same (indicating fill data)
                        unique_vals = np.unique(valid_data)
                        if len(unique_vals) == 1 and unique_vals[0] >= 60000:

                            filtered_data[:] = np.nan
                        else:
                            # Use 99.5th percentile as upper threshold - anything above is likely fill
                            fill_threshold = np.percentile(valid_data, 99.5)
                            # But don't go below a reasonable minimum threshold
                            fill_threshold = max(fill_threshold, 60000)

                            filtered_data[filtered_data >= fill_threshold] = np.nan
            else:  # uint8
                common_fill_values = [255, 254, 253]
                for fv in common_fill_values:
                    filtered_data[filtered_data == fv] = np.nan
            
        elif data.dtype in [np.int16, np.int8]:
            # For signed integer data, check both extreme values
            if data.dtype == np.int16:
                # Common VIIRS signed fill values
                fill_values = [-32768, -32767, 32767]
            else:  # int8
                fill_values = [-128, -127, 127]
            
            for fv in fill_values:
                filtered_data[filtered_data == fv] = np.nan
                
        # Additional checks for obviously invalid data
        # Negative radiance values are often invalid (TODO: Handle reflectance when added)
        filtered_data[filtered_data < 0] = np.nan
        
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
        var_types = []
        
        for var in variables:
            # Check if variable exists in lexicon
            if var not in JPSSLexicon.VOCAB:
                unknown_vars.append(var)
                continue
                
            # Determine variable type
            if var.startswith("viirs") and var.endswith("i"):
                var_types.append(("I", var))
            elif var.startswith("viirs") and var.endswith("m"):
                var_types.append(("M", var))
            elif var.startswith("viirs_"):
                var_types.append(("L2", var))
            else:
                # Handle any other patterns that might be added in the future
                var_types.append(("UNKNOWN", var))
        
        # Raise error for unknown variables
        if unknown_vars:
            raise ValueError(
                f"Unknown VIIRS variables: {unknown_vars}. "
                f"Please check the JPSS lexicon for available variables."
            )
        
        # Group variables by type
        i_vars = [var for vtype, var in var_types if vtype == "I"]
        m_vars = [var for vtype, var in var_types if vtype == "M"] 
        l2_vars = [var for vtype, var in var_types if vtype == "L2"]
        unknown_type_vars = [var for vtype, var in var_types if vtype == "UNKNOWN"]
        
        if unknown_type_vars:
            raise ValueError(f"Cannot determine product type for variables: {unknown_type_vars}")
        
        # Validate based on configured band_type
        if self._band_type == "I":
            if m_vars or l2_vars:
                mixed_vars = m_vars + l2_vars
                raise ValueError(
                    f"Instance configured for I-bands but received incompatible variables: {mixed_vars}. "
                    f"Use band_type='L2' for mixed I/M/L2 requests or create separate instances."
                )
        elif self._band_type == "M":
            if i_vars or l2_vars:
                mixed_vars = i_vars + l2_vars
                raise ValueError(
                    f"Instance configured for M-bands but received incompatible variables: {mixed_vars}. "
                    f"Use band_type='L2' for mixed I/M/L2 requests or create separate instances."
                )
        elif self._band_type == "L2":
            # L2 mode allows mixing of I, M, and L2 products
            # Just warn about potential resolution differences
            if i_vars and m_vars:
                logger.warning(
                    f"Mixing I-band variables {i_vars} with M-band variables {m_vars}. "
                    f"Note that I-bands (375m) and M-bands (750m) have different resolutions."
                )

    async def _fetch_geolocation_data(self, time: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Fetch geolocation data from VIIRS geolocation files."""
        # Determine geolocation product based on band type
        # From S3 exploration: VIIRS-IMG-GEO for I-bands, VIIRS-MOD-GEO for M-bands
        # For L2 products, try to use M-band geolocation as default (can be overridden later if needed)
        if self._product_type == "I":
            geo_product = "VIIRS-IMG-GEO"
        elif self._product_type == "M":
            geo_product = "VIIRS-MOD-GEO"
        else:  # L2 products
            geo_product = "VIIRS-MOD-GEO"  # Default to M-band resolution for L2 products
        
        bucket = self.SATELLITE_BUCKETS[self._satellite]
        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=geo_product,
            year=time.year,
            month=time.month,
            day=time.day,
        )
        
        try:
            files = await self.fs._ls(base_url)
        except Exception:
            raise FileNotFoundError(f"No {geo_product} geolocation files found for {time}")
            
        # Find matching geolocation file
        matching_files = [f for f in files if f.endswith(".h5")]
        if not matching_files:
            raise FileNotFoundError(f"No {geo_product} geolocation files found")
            
        def parse_start_dt(path: str) -> datetime:
            fname = path.split("/")[-1]
            parts = fname.split("_")
            # Look for date and time parts in filename
            for part in parts:
                if part.startswith("d") and len(part) == 9:  # dYYYYMMDD
                    date_str = part[1:]
                elif part.startswith("t") and len(part) >= 8:  # tHHMMSSs
                    time_str = part[1:7]
            try:
                dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                return dt.replace(tzinfo=timezone.utc)
            except:
                # Fallback: use file modification time or current time
                return time
                
        time_stamps = [parse_start_dt(f) for f in matching_files]
        idx = int(np.argmin(np.abs(np.array(time_stamps, dtype="datetime64[s]") - np.datetime64(time))))
        geo_file_path = matching_files[idx]
        
        # Download and read the geolocation file
        local_geo_file = await self._fetch_remote_file(geo_file_path)
        
        with h5py.File(local_geo_file, "r") as h5:
            # Try to find latitude and longitude datasets within the file
            # They're often in geolocation groups or alongside the data
            lat_ds = None
            lon_ds = None
            
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

    def _find_dataset(self, h5: h5py.File, dataset_name: str):  # type: ignore[override]
        # Try common locations first to be efficient
        if "All_Data" in h5:
            for gname, grp in h5["All_Data"].items():
                if isinstance(grp, h5py.Group) and dataset_name in grp:
                    return grp[dataset_name]
        
        # Fallback: exhaustive search
        found = None
        def visitor(name, obj):
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset):
                # Check if dataset name matches exactly or ends with the target name
                if name.split("/")[-1] == dataset_name or name.endswith("/" + dataset_name):
                    found = obj
        
        h5.visititems(visitor)
        if found is not None:
            return found
        
        # If still not found, try case-insensitive search for common L2 patterns
        def case_insensitive_visitor(name, obj):
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset):
                obj_name = name.split("/")[-1].lower()
                target_name = dataset_name.lower()
                if obj_name == target_name or target_name in obj_name:
                    found = obj
        
        h5.visititems(case_insensitive_visitor)
        if found is not None:
            return found
            
        raise KeyError(f"Dataset {dataset_name} not found in file")

    async def _get_s3_path(self, time: datetime, product_code: str) -> str:
        """Get the S3 path for a given time and product code."""

        # Get the bucket for the satellite
        bucket = self.SATELLITE_BUCKETS[self._satellite]

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

        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )

        # List files in the directory and choose the closest timestamp file
        files = await self.fs._ls(base_url)

        # Filter for data files - SDR products use .h5, L2 products may use .h5 or .nc
        if product_code.startswith(("SVI", "SVM")):
            # SDR products use HDF5 format
            data_files = [f for f in files if f.endswith(".h5")]
            if not data_files:
                raise FileNotFoundError(f"No VIIRS SDR files found at {base_url} for {product_code}")
            # Match by product code in filename
            matching_files = [f for f in data_files if f"/{product_code}_" in f]
            if not matching_files:
                raise FileNotFoundError(f"No VIIRS SDR files found at {base_url} for {product_code}")
        else:
            # L2 EDR products may use HDF5 or NetCDF format
            data_files = [f for f in files if f.endswith((".h5", ".nc"))]
            if not data_files:
                raise FileNotFoundError(f"No VIIRS L2 files found at {base_url} for {product_code}")
            # For L2 EDR products, all data files in the folder are candidates
            matching_files = data_files

        def parse_start_dt(path: str) -> datetime:
            """Parse start datetime from filename."""
            fname = path.split("/")[-1]
            
            # Handle different filename formats
            if (fname.startswith("LST_") or fname.startswith("JRR-")) and "_s" in fname:
                # L2 product formats: 
                # LST_v2r2_j01_s202501012358307_e202501012359552_c202501020100321.nc
                # JRR-CloudHeight_v3r2_j01_s202501012357067_e202501012358294_c202501020043237.nc
                parts = fname.split("_")
                for part in parts:
                    if part.startswith("s") and len(part) >= 14:  # s + YYYYMMDDHHMMSSS
                        datetime_str = part[1:15]  # Extract YYYYMMDDHHMMSSS (14 chars)
                        try:
                            dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
                            return dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            # Try with milliseconds
                            try:
                                dt = datetime.strptime(datetime_str[:12], "%Y%m%d%H%M")
                                return dt.replace(tzinfo=timezone.utc)
                            except ValueError:
                                pass
            else:
                # Standard VIIRS naming: find dYYYYMMDD and tHHMMSSs parts
                parts = fname.split("_")
                date_str = None
                time_str = None
                
                for part in parts:
                    if part.startswith("d") and len(part) >= 9:
                        date_str = part[1:9]  # Extract YYYYMMDD
                    elif part.startswith("t") and len(part) >= 8:
                        time_str = part[1:7]  # Extract HHMMSS
                
                if date_str and time_str:
                    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                    return dt.replace(tzinfo=timezone.utc)
            
            # Fallback: use a neutral time if we can't parse the filename
            logger.warning(f"Could not parse datetime from filename {fname}, using requested time")
            return time

        time_stamps = [parse_start_dt(f) for f in matching_files]
        idx = int(np.argmin(np.abs(np.array(time_stamps, dtype="datetime64[s]") - np.datetime64(time))))
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
        band_type: str = "I",
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


