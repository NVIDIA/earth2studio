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
import concurrent.futures
import gzip
import hashlib
import os
import pathlib
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import nest_asyncio
import numpy as np
import pygrib
import s3fs
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MRMSLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    # Ensure the optional eccodes dependency group is flagged if missing at use time
    import eccodes  # noqa: F401
except ImportError:
    OptionalDependencyFailure("data")
    eccodes = None  # type: ignore[assignment]


@check_optional_dependencies("data")
class MRMS:
    """NOAA Multi-Radar/Multi-Sensor (MRMS) products via AWS S3.

    This data source downloads MRMS GRIB2 files (gzipped) from the NOAA MRMS
    public S3 bucket, decompresses them into the Earth2Studio cache, and opens
    the result with pygrib. Initially, only the composite reflectivity
    product is supported, exposed via the Earth2Studio variable id ``refc``.

    Parameters
    ----------
    max_offset_minutes : float, optional
        Time tolerance in minutes to search for the nearest available MRMS
        file to the requested timestamp, by default 10 minutes.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    verbose : bool, optional
        Print basic progress logs, by default True.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.

    Warning
    -------
    This is a remote data source and can download sizable files depending on
    request volume and frequency.

    Note
    ----
    Additional information:

    - https://registry.opendata.aws/noaa-mrms-pds/
    - https://noaa-mrms-pds.s3.amazonaws.com/index.html
    """

    MRMS_BUCKET_NAME = "noaa-mrms-pds"
    MRMS_REGION = "CONUS"
    EARLIEST_AVAILABLE = datetime(2020, 10, 14, tzinfo=timezone.utc)

    def __init__(
        self,
        max_offset_minutes: float = 10,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        async_timeout: int = 600,
    ):

        if max_offset_minutes < 0:
            raise ValueError("max_offset_minutes must be non-negative")

        self.max_offset_minutes = max_offset_minutes
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        # Set up S3 filesystem
        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem"""
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve MRMS data for given times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to fetch (UTC). Exact times must match available MRMS files.
        variable : str | list[str] | VariableArray
            Earth2Studio variable ids to return. Must be in the MRMS lexicon.
            Currently only ``refc`` is supported (composite reflectivity).

        Returns
        -------
        xr.DataArray
            Data with dimensions ``[time, variable, y, x]``. Coordinates include
            ``lat`` and ``lon`` on the same ``[y, x]`` grid when provided in the
            source file.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        xr_array = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
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
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            MRMS weather data array
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        # Validate requested times are within MRMS availability window
        self._validate_time(time)

        # Create cache dir if needed
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        session = await self.fs.set_session(refresh=True)

        # Group variables by MRMS product and keep (var_index, modifier)
        product_to_vars: dict[str, list[tuple[int, Callable]]] = {}
        for idx, v in enumerate(variable):
            try:
                product, modifier = MRMSLexicon[v]
            except KeyError as e:
                raise KeyError(f"Variable id {v} not found in MRMS lexicon.") from e
            product_to_vars.setdefault(product, []).append((idx, modifier))

        # Build tasks for each (time, product)
        tasks = [
            self._fetch_task(ti, t0, product, idx_mods)
            for ti, t0 in enumerate(time)
            for product, idx_mods in product_to_vars.items()
        ]

        # Execute tasks concurrently
        results = await tqdm.gather(
            *tasks, desc="Fetching MRMS data", disable=(not self._verbose)
        )

        # Determine grid from first successful result and validate others
        first = next((r for r in results if r is not None), None)
        if first is None:
            raise RuntimeError("All MRMS fetches failed; no data available.")
        lat = first["lat"]
        lon = first["lon"]
        dtype = np.asarray(first["field_values"]).dtype

        # Pre-allocate output; keep requested times as coordinate
        out = xr.DataArray(
            data=np.full(
                (len(time), len(variable), lat.shape[0], lon.shape[0]),
                np.nan,
                dtype=dtype,
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time, "variable": variable, "lat": lat, "lon": lon},
        )

        # Fill output
        for res in results:
            if res is None:
                continue
            if res["lat"].shape != lat.shape or res["lon"].shape != lon.shape:
                raise ValueError(
                    "Requested variables map to MRMS products with different grids; "
                    "split requests by product."
                )
            ti = res["time_index"]
            field_values = res["field_values"]
            for var_index, modifier in res["idx_mods"]:
                out[ti, var_index] = modifier(field_values)

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return out

    async def _fetch_task(
        self,
        time_index: int,
        time: datetime,  # tz aware time
        product: str,
        idx_mods: list[tuple[int, Callable]],
    ) -> dict | None:
        """Internal coroutine to fetch a single MRMS product field for a time."""
        # Resolve to an exact available S3 object within tolerance in a thread
        resolved = await self._resolve_s3_time(
            self.fs, time, product, self.max_offset_minutes
        )
        if resolved is None:
            logger.warning(
                f"No MRMS file found near requested time {time.isoformat()} "
                f"within ±{self.max_offset_minutes} minutes for product {product}."
                f"Consider increasing the max_offset_minutes parameter."
            )
            return None
        _, s3_uri = resolved

        if self._verbose:
            logger.info(f"Fetching MRMS file: {s3_uri}")

        # Download and decompress using async APIs
        grib_file = await self._download_and_decompress_async(s3_uri)

        try:
            grbs = pygrib.open(grib_file)
        except Exception as e:
            logger.error(f"Failed to open grib file {grib_file}")
            raise e
        try:
            grb = grbs[1]
            values = grb.values  # (ny, nx)
            lats, lons = grb.latlons()
            lat = lats[:, 0]
            lon = lons[0, :]
        except Exception as e:
            logger.error(f"Failed to read grib file {grib_file}")
            raise e
        finally:
            grbs.close()

        return {
            "time_index": time_index,
            "idx_mods": idx_mods,
            "lat": lat,
            "lon": lon,
            "field_values": values,
        }

    async def _download_and_decompress_async(self, s3_uri: str) -> str:
        """Async download of gzipped GRIB2 from S3 and decompress into cache; return path."""
        # Cache filenames derived from key
        key_hash = hashlib.sha256(s3_uri.encode()).hexdigest()
        grib_path = os.path.join(self.cache, f"{key_hash}.grib2")

        # Download gz and decompress if not present
        if not pathlib.Path(grib_path).is_file():
            # Read gzipped payload into memory and decompress to GRIB
            data = await self.fs._cat_file(s3_uri)  # type: ignore[attr-defined]
            decompressed = gzip.decompress(data)
            with open(grib_path, "wb") as fout:
                fout.write(decompressed)

        return grib_path

    @classmethod
    def _s3_key(cls, time: datetime, product: str) -> str:
        """Build the S3 object key for a given time and product."""
        date_str = time.strftime("%Y%m%d")
        hms = time.strftime("%H%M%S")
        filename = f"MRMS_{product}_{date_str}-{hms}.grib2.gz"
        return f"{cls.MRMS_REGION}/{product}/{date_str}/{filename}"

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify requested times are within MRMS availability window.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        now_utc = datetime.now(timezone.utc)
        for dt in times:
            if dt.tzinfo is None:
                # Enforce UTC timezone
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < self.EARLIEST_AVAILABLE:
                raise ValueError(
                    f"Requested date time {dt} needs to be after {self.EARLIEST_AVAILABLE.date()} for MRMS"
                )
            if dt > now_utc:
                raise ValueError(
                    f"Requested date time {dt} must not be in the future for MRMS"
                )

    @classmethod
    async def _resolve_s3_time(
        cls,
        fs: s3fs.S3FileSystem,
        time: datetime,
        product: str,
        max_offset_minutes: float = 10,
    ) -> tuple[datetime, str] | None:
        """Find nearest-available S3 object within the configured minute tolerance.

        Returns the resolved timestamp and full S3 URI if found, else None.
        """
        # Exact match fast path
        key = cls._s3_key(time, product)
        s3_uri = f"s3://{cls.MRMS_BUCKET_NAME}/{key}"
        if await fs._exists(s3_uri):
            return time, s3_uri

        # List candidate days (can cross day boundary within tolerance)
        t_min = time - timedelta(minutes=max_offset_minutes)
        t_max = time + timedelta(minutes=max_offset_minutes)
        candidate_dates = {
            t_min.strftime("%Y%m%d"),
            time.strftime("%Y%m%d"),
            t_max.strftime("%Y%m%d"),
        }

        pattern = re.compile(
            rf"^MRMS_{re.escape(product)}_(\d{{8}})-(\d{{6}})\.grib2\.gz$"
        )
        best_dt: datetime | None = None
        best_uri: str | None = None
        best_diff: float = float("inf")

        for date_str in sorted(candidate_dates):
            dir_uri = (
                f"s3://{cls.MRMS_BUCKET_NAME}/{cls.MRMS_REGION}/{product}/{date_str}/"
            )
            try:
                keys = await fs._ls(dir_uri)
            except FileNotFoundError:
                continue
            for key_path in keys:
                fname = os.path.basename(key_path)
                m = pattern.match(fname)
                if not m:
                    continue
                ds, hms = m.group(1), m.group(2)
                ts = datetime(
                    year=int(ds[0:4]),
                    month=int(ds[4:6]),
                    day=int(ds[6:8]),
                    hour=int(hms[0:2]),
                    minute=int(hms[2:4]),
                    second=int(hms[4:6]),
                )
                diff = abs((ts - time).total_seconds())
                if diff <= max_offset_minutes * 60 and diff < best_diff:
                    best_diff = diff
                    best_dt = ts
                    # key_path can be either with or without s3:// prefix; construct URI
                    if key_path.startswith("s3://"):
                        best_uri = key_path
                    else:
                        best_uri = f"s3://{key_path}"

        if best_dt is None or best_uri is None:
            return None
        return best_dt, best_uri

    @property
    def cache(self) -> str:
        """Return MRMS cache directory."""
        cache_location = os.path.join(datasource_cache_root(), "mrms")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_mrms")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        variable: str | list[str] | VariableArray,
        max_offset_minutes: int = 10,
    ) -> bool:
        """Check if an MRMS file exists for a given time. Only supports exact match.

        Parameters
        ----------
        time : datetime | np.datetime64
            Time to query (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s). The corresponding MRMS product is derived
            from the lexicon. All variables must map to the same product.
        max_offset_minutes : int, optional
            Time tolerance in minutes to search for the nearest available MRMS
            file to the requested timestamp, by default 0 (exact match only).

        Returns
        -------
        bool
            True if the object exists in S3, else False.
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        # Offline time bounds check
        if time.replace(tzinfo=timezone.utc) < cls.EARLIEST_AVAILABLE:
            return False
        if time.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc):
            return False

        # Derive MRMS product from requested variables
        if isinstance(variable, (list, tuple, np.ndarray)):
            variables = list(variable)
        else:
            variables = [variable]
        products: set[str] = set()
        for v in variables:
            try:
                prod, _ = MRMSLexicon[v]
            except KeyError as e:
                raise KeyError(f"Variable id {v} not found in MRMS lexicon.") from e
            products.add(prod)
        if len(products) != 1:
            raise ValueError(
                "Requested variables map to multiple MRMS products; split requests by product."
            )
        product = next(iter(products))

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _resolve_helper() -> bool:
            fs = s3fs.S3FileSystem(
                anon=True,
                client_kwargs={},
                asynchronous=True,
                skip_instance_cache=False,
            )
            return (
                await cls._resolve_s3_time(fs, time, product, max_offset_minutes)
                is not None
            )

        return loop.run_until_complete(_resolve_helper())
