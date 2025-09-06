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
    """JPSS VIIRS SDR (I and M bands) data source for NOAA-20, NOAA-21, and Suomi-NPP.

    Parameters
    ----------
    satellite : str, optional
        One of {"noaa-20", "noaa-21", "snpp"}. Default "noaa-20".
    band_type : str, optional
        Band resolution type: "I" for imagery bands (375m) or "M" for moderate bands (750m). Default "I".
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
    latitude and longitude coordinates for each pixel. All requested variables must
    be of the same band type (I or M) to ensure consistent spatial resolution.
    
    - I-bands: 3200 x 5424 pixels (375m resolution)
    - M-bands: 1600 x 2712 pixels (750m resolution)
    
    Currently only "Radiance" data is extracted from VIIRS SDR files. Future versions
    may add support for "Reflectance" data for applicable bands.
    """

    BASE_URL = "s3://{bucket}/{product}/{year:04d}/{month:02d}/{day:02d}/"
    SATELLITE_BUCKETS: dict[str, str] = {
        "noaa-20": "noaa-nesdis-n20-pds",
        "noaa-21": "noaa-nesdis-n21-pds",
        "snpp": "noaa-nesdis-snpp-pds",
    }
    VALID_SATELLITES = ["noaa-20", "noaa-21", "snpp"]
    VALID_BAND_TYPES = ["I", "M"]
    
    # VIIRS band dimensions (standard granule sizes)
    # I-bands: 375m resolution, M-bands: 750m resolution
    BAND_DIMENSIONS = {
        "I": (1536, 6400),  # I-bands: ~3200 x 5424 pixels
        "M": (1600, 2712),  # M-bands: ~1600 x 2712 pixels (half resolution)
    }

    def __init__(
        self,
        satellite: str = "noaa-20",
        band_type: str = "I",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        if satellite not in self.VALID_SATELLITES:
            raise ValueError(f"Invalid satellite {satellite}. Must be one of {self.VALID_SATELLITES}")
        if band_type not in self.VALID_BAND_TYPES:
            raise ValueError(f"Invalid band_type {band_type}. Must be one of {self.VALID_BAND_TYPES}")

        self._satellite: str = satellite
        self._band_type: str = band_type
        self._max_workers = max_workers
        self._cache = cache
        self._verbose = verbose
        self._async_timeout = async_timeout
        # TODO: Add support for "Reflectance" field in future versions
        # Currently only "Radiance" is supported for all VIIRS bands

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

        # Validate band types before fetching data
        self._validate_band_types(variable)

        # Create cache directory if it doesn't exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Add lat/lon to the variable list
        extended_variables = list(variable) + ["_lat", "_lon"]

        session = await self.fs.set_session()

        # Use fixed dimensions based on band type
        y_size, x_size = self.BAND_DIMENSIONS[self._band_type]
        
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

        async_tasks = [(i, t, variable) for i, t in enumerate(time)]
        await tqdm.gather(
            *map(lambda args: self.fetch_wrapper(args, xr_array=xr_array), async_tasks),
            desc="Fetching VIIRS data",
            disable=(not self._verbose),
        )

        if session:
            await session.close()

        # Close aiohttp client cleanly as in GOES implementation
        await self.fs.set_session()
        s3fs.S3FileSystem.close_session(asyncio.get_event_loop(), self.fs.s3)

        return xr_array

    async def fetch_wrapper(
        self,
        e: tuple[int, datetime, list[str]],
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray"""
        # Fetch data variables
        data_out = await self.fetch_array(time=e[1], variable=e[2])
        
        # Fetch geolocation data (lat/lon)
        lat, lon = await self._fetch_geolocation_for_band_type(time=e[1])
        
        # Combine data and geolocation
        combined_out = np.concatenate([data_out, np.expand_dims(lat, axis=0), np.expand_dims(lon, axis=0)], axis=0)

        # Add to xr_array
        xr_array[e[0]] = combined_out

    async def fetch_array(
        self,
        time: datetime,
        variable: list[str],
    ) -> np.ndarray:
        """Fetch VIIRS data array

        Parameters
        ----------
        time : datetime
            Time to get data for
        variable : list[str]
            List of variable names to fetch

        Returns
        -------
        np.ndarray
            VIIRS data array
        """

        # Validate file system (probably not needed)
        if self.fs is None:
            raise ValueError("File system is not initialized")
        
        # Map standardized variables to (product_code, dataset, modifier)
        mappings: list[tuple[str, str, str, callable]] = []
        for v in variable:
            if v in JPSSLexicon.VOCAB:
                product_identifier, modifier = JPSSLexicon[v]
                # Parse "SVI01/Radiance" format
                product_code, dataset_name = product_identifier.split("/")
                mappings.append((v, product_code, dataset_name, modifier))
            else:
                logger.warning(f"Variable {v} not found in VIIRS lexicon; skipping.")

        # Build arrays per variable
        arrays: list[np.ndarray] = []
        for _, product_code, dataset_name, modifier in mappings:
            # Find file path on S3 for this time and product
            s3_uri = await self._get_s3_path(time=time, product_code=product_code)
            local_file = await self._fetch_remote_file(s3_uri)

            # Read the dataset from HDF5
            with h5py.File(local_file, "r") as h5:
                # Get dataset from HDF5 file
                ds = self._find_dataset(h5, dataset_name)
                data = ds[...]
                
                # Filter VIIRS fill values BEFORE converting to float32 to preserve original dtype
                filtered_data = self._filter_fill_values(data, ds)
                
                # Apply modifier and convert to float32 for consistency
                processed_data = np.asarray(modifier(filtered_data), dtype=np.float32)
                
                arrays.append(processed_data)

        out = np.stack(arrays, axis=0)
        
        return out

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

    def _validate_band_types(self, variables: list[str]) -> None:
        """Validate that all requested variables match the configured band type.
        
        Parameters
        ----------
        variables : list[str]
            List of variable names to validate
            
        Raises
        ------
        ValueError
            If any variable doesn't match the configured band type or has invalid format
        """
        mismatched_vars = []
        invalid_format_vars = []
        
        for var in variables:
            # Extract band type from variable name (viirs1i -> I, viirs5m -> M)
            if var.startswith("viirs") and var.endswith("i"):
                var_band_type = "I"
            elif var.startswith("viirs") and var.endswith("m"):
                var_band_type = "M"
            else:
                invalid_format_vars.append(var)
                continue
            
            if var_band_type != self._band_type:
                mismatched_vars.append((var, var_band_type))
        
        # Raise descriptive errors
        if invalid_format_vars:
            raise ValueError(
                f"Invalid VIIRS variable format: {invalid_format_vars}. "
                f"Expected format: 'viirs{{number}}i' for I-bands or 'viirs{{number}}m' for M-bands"
            )
        
        if mismatched_vars:
            var_list = [f"'{var}' ({band_type}-band)" for var, band_type in mismatched_vars]
            raise ValueError(
                f"Band type mismatch: {', '.join(var_list)} cannot be used with this JPSS instance "
                f"configured for {self._band_type}-bands. Create separate instances for I-bands and M-bands "
                f"due to different spatial resolutions (I-bands: 375m, M-bands: 750m)."
            )

    async def _fetch_geolocation_for_band_type(self, time: datetime) -> tuple[np.ndarray, np.ndarray]:
        """Fetch geolocation data from VIIRS geolocation files."""
        # Determine geolocation product based on band type
        # From S3 exploration: VIIRS-IMG-GEO for I-bands, VIIRS-MOD-GEO for M-bands
        geo_product = "VIIRS-IMG-GEO" if self._band_type == "I" else "VIIRS-MOD-GEO"
        
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
        ## Fallback: exhaustive search
        #found = None
        #def visitor(name, obj):
        #    nonlocal found
        #    if found is not None:
        #        return
        #    if isinstance(obj, h5py.Dataset) and name.endswith("/" + dataset_name):
        #        found = obj
        #h5.visititems(visitor)
        #if found is not None:
        #    return found
        #raise KeyError(f"Dataset {dataset_name} not found in file")

    async def _get_s3_path(self, time: datetime, product_code: str) -> str:
        if self.fs is None:
            raise ValueError("File system is not initialized")

        bucket = self.SATELLITE_BUCKETS[self._satellite]
        # Derive folder name from product code
        band_num = product_code[-2:]
        if product_code.startswith("SVI"):
            folder = f"VIIRS-I{int(band_num)}-SDR"
        elif product_code.startswith("SVM"):
            folder = f"VIIRS-M{int(band_num)}-SDR"
        else:
            raise ValueError(f"Unrecognized VIIRS product code {product_code}")

        base_url = self.BASE_URL.format(
            bucket=bucket,
            product=folder,
            year=time.year,
            month=time.month,
            day=time.day,
        )

        # List files in the directory and choose the closest timestamp file
        files = await self.fs._ls(base_url)

        # VIIRS filenames example:
        # SVI01_npp_d20230303_t0001080_e0002321_b58787_c20230303004405425469_oeac_ops.h5
        # we match by product_code and date, selecting nearest start time to requested time
        matching_files = [f for f in files if f.endswith(".h5") and f"/{product_code}_" in f]
        if not matching_files:
            raise FileNotFoundError(f"No VIIRS SDR files found at {base_url} for {product_code}")

        def parse_start_dt(path: str) -> datetime:
            fname = path.split("/")[-1]
            parts = fname.split("_")
            # parts like: [SVI01, npp, dYYYYMMDD, tHHMMSSs, e..., b..., c..., ...]
            date_str = next(p[1:] for p in parts if p.startswith("d"))
            time_str = next(p[1:7] for p in parts if p.startswith("t"))
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=timezone.utc)

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
        # Derive folder name from product code
        band_num = product_code[-2:]
        if product_code.startswith("SVI"):
            folder = f"VIIRS-I{int(band_num)}-SDR"
        elif product_code.startswith("SVM"):
            folder = f"VIIRS-M{int(band_num)}-SDR"
        else:
            return False

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

        # Filter for product_code files
        matching_files = [f for f in files if f.endswith(".h5") and f"/{product_code}_" in f]
        if not matching_files:
            return False
        return True


