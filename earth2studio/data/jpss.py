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
from datetime import datetime, timezone
from typing import Literal

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


SatelliteName = Literal["noaa-20", "noaa-21", "snpp"]
BandType = Literal["I", "M"]


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
    field : str, optional
        HDF5 dataset to extract from VIIRS SDR files. Default "Radiance".
    
    Note
    ----
    The returned data automatically includes '_lat' and '_lon' variables containing
    latitude and longitude coordinates for each pixel. All requested variables must
    be of the same band type (I or M) to ensure consistent spatial resolution.
    """

    BASE_URL = "s3://{bucket}/{product}/{year:04d}/{month:02d}/{day:02d}/"
    SATELLITE_BUCKETS: dict[SatelliteName, str] = {
        "noaa-20": "noaa-nesdis-n20-pds",
        "noaa-21": "noaa-nesdis-n21-pds",
        "snpp": "noaa-nesdis-snpp-pds",
    }

    def __init__(
        self,
        satellite: SatelliteName = "noaa-20",
        band_type: BandType = "I",
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        field: str = "Radiance",
    ) -> None:
        if satellite not in self.SATELLITE_BUCKETS:
            raise ValueError(f"Invalid satellite {satellite}")
        if band_type not in ["I", "M"]:
            raise ValueError(f"Invalid band_type {band_type}. Must be 'I' or 'M'")

        self._satellite: SatelliteName = satellite
        self._band_type: BandType = band_type
        self._max_workers = max_workers
        self._cache = cache
        self._verbose = verbose
        self._async_timeout = async_timeout
        self._field = field

        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

    async def _async_init(self) -> None:
        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
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
            import shutil

            shutil.rmtree(self.cache, ignore_errors=True)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this function directly, make sure the data source is initialized inside the async loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate that all requested variables match the band type
        self._validate_band_types(variable)

        # Add lat/lon to the variable list
        extended_variables = list(variable) + ["_lat", "_lon"]

        session = await self.fs.set_session()

        # Discover shape using the first timestamp
        first_arr = await self.fetch_array(time=time[0], variable=variable)
        y_size, x_size = first_arr.shape[1], first_arr.shape[2]
        
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
        # Fetch data variables
        data_out = await self.fetch_array(time=e[1], variable=e[2])
        
        # Fetch geolocation data (lat/lon)
        try:
            lat, lon = await self._fetch_geolocation_for_band_type(time=e[1])
        except FileNotFoundError as e:
            logger.warning(f"Geolocation data not found: {e}")
            # Create dummy lat/lon grids with proper shape
            data_shape = data_out.shape[1:]  # (y, x) from data_out which is (n_vars, y, x)
            lat = np.full(data_shape, np.nan, dtype=np.float32)
            lon = np.full(data_shape, np.nan, dtype=np.float32)
        
        # Combine data and geolocation
        # data_out shape: (n_variables, y, x)
        # lat, lon shape: (y, x)
        combined_out = np.zeros((data_out.shape[0] + 2, data_out.shape[1], data_out.shape[2]), dtype=np.float32)
        combined_out[:data_out.shape[0]] = data_out  # Data variables
        combined_out[-2] = lat  # _lat
        combined_out[-1] = lon  # _lon
        
        xr_array[e[0]] = combined_out

    async def fetch_array(
        self,
        time: datetime,
        variable: list[str],
    ) -> np.ndarray:
        if self.fs is None:
            raise ValueError("File system is not initialized")

        # Map standardized variables to (product_code, dataset, modifier)
        mappings: list[tuple[str, str, str, callable]] = []
        for v in variable:
            try:
                product_code, dataset_name, modifier = JPSSLexicon[v]
            except KeyError:
                logger.warning(f"Variable {v} not found in VIIRS lexicon; skipping.")
                continue
            # Allow class-level override of dataset field
            dataset = self._field if self._field else dataset_name
            mappings.append((v, product_code, dataset, modifier))

        if len(mappings) == 0:
            raise ValueError("No valid VIIRS variables provided.")

        # Build arrays per variable
        arrays: list[np.ndarray] = []
        for _, product_code, dataset_name, modifier in mappings:
            # Find file path on S3 for this time and product
            s3_uri = await self._get_s3_path(time=time, product_code=product_code)
            local_file = await self._fetch_remote_file(s3_uri)

            # Read the dataset from HDF5
            with h5py.File(local_file, "r") as h5:
                # Dataset names in VIIRS SDR files typically include "Data" group.
                # Common paths: e.g., "/All_Data/VIIRS-XYZ-SDR_All/{dataset}"
                ds = self._find_dataset(h5, dataset_name)
                data = ds[...]
                arrays.append(np.asarray(modifier(data)))

        out = np.stack(arrays, axis=0)
        return out

    def _validate_band_types(self, variables: list[str]) -> None:
        """Validate that all requested variables match the configured band type."""
        for var in variables:
            try:
                product_code, _, _ = JPSSLexicon[var]
                var_band_type = "I" if product_code.startswith("SVI") else "M"
                if var_band_type != self._band_type:
                    raise ValueError(
                        f"Variable '{var}' is a {var_band_type}-band but this JPSS instance is configured for {self._band_type}-bands. "
                        f"Create separate instances for I-bands and M-bands due to different spatial resolutions."
                    )
            except KeyError:
                raise ValueError(f"Variable '{var}' not found in JPSS lexicon")

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
        # Fallback: exhaustive search
        found = None
        def visitor(name, obj):
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset) and name.endswith("/" + dataset_name):
                found = obj
        h5.visititems(visitor)
        if found is not None:
            return found
        raise KeyError(f"Dataset {dataset_name} not found in file")

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
        satellite: SatelliteName = "noaa-20",
        band_type: BandType = "I",
        variable: str = "i1",
    ) -> bool:
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        if satellite not in cls.SATELLITE_BUCKETS:
            raise ValueError(f"Invalid satellite {satellite}")

        try:
            product_code, _, _ = JPSSLexicon[variable]
        except KeyError:
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


