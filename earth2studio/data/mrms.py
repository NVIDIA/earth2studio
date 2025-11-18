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

import gzip
import os
import pathlib
import re
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import s3fs
import xarray as xr
from loguru import logger

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
    """NOAA Multi-Radar/Multi-Sensor (MRMS) CONUS products via AWS S3.

    This data source downloads MRMS GRIB2 files (gzipped) from the NOAA MRMS
    public S3 bucket, decompresses them into the Earth2Studio cache, and opens
    the result with Xarray/cfgrib. Initially, only the composite reflectivity
    product is supported, exposed via the Earth2Studio variable id ``refc``.

    Parameters
    ----------
    product : str, optional
            MRMS product name (folder and filename stem) to fetch, by default
            ``MergedReflectivityQCComposite_00.50``.
    max_offset_minutes : int, optional
            Time tolerance in minutes to search for the nearest available MRMS
            file to the requested timestamp, by default 0 (exact match only).
    cache : bool, optional
            Cache data source in local filesystem cache, by default True.
    verbose : bool, optional
            Print basic progress logs, by default True.

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
    EARLIEST_AVAILABLE = datetime(2020, 10, 14, tzinfo=timezone.utc)

    def __init__(
        self,
        product: str = "MergedReflectivityQCComposite_00.50",
        max_offset_minutes: int = 0,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.product = product
        self.max_offset_minutes = max_offset_minutes
        self._cache = cache
        self._verbose = verbose
        self.fs = s3fs.S3FileSystem(anon=True)

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
        time, variable = prep_data_inputs(time, variable)

        # Validate requested times are within MRMS availability window
        self._validate_time(time)

        # Create cache dir if needed
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        data_arrays: list[xr.DataArray] = []
        for t0 in time:
            da = self._fetch_mrms_dataarray(t0, variable)
            data_arrays.append(da)

        # Delete cache if requested
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def _fetch_mrms_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Download, decompress, and load MRMS GRIB2 data into an Xarray DataArray.

        Parameters
        ----------
        time : datetime
                Exact timestamp of the MRMS file to fetch (UTC).
        variables : list[str]
                List of Earth2Studio variable ids. Must be supported in MRMS lexicon.

        Returns
        -------
        xr.DataArray
                Data array for the specified time with dimensions ``[time, variable, y, x]``.

        Raises
        ------
        FileNotFoundError
                If the requested remote file does not exist.
        KeyError
                If a requested variable is not present in the MRMS lexicon.
        """
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        # Resolve to an exact available S3 object within tolerance
        resolved = self._resolve_s3_time(time)
        if resolved is None:
            raise FileNotFoundError(
                f"No MRMS file found near requested time {time.isoformat()} "
                f"within ±{self.max_offset_minutes} minutes for product {self.product}"
            )
        actual_time, s3_uri = resolved

        if self._verbose:
            logger.info(f"Fetching MRMS file: {s3_uri}")

        grib_path = self._download_and_decompress(s3_uri)

        # Load with cfgrib.
        ds = xr.open_dataset(
            grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
        )
        if len(ds.data_vars) == 0:
            raise RuntimeError(f"No data variables found in MRMS file: {s3_uri}")
        data_var_name = list(ds.data_vars)[0]
        field = ds[data_var_name].rename({"latitude": "lat", "longitude": "lon"})

        # Build output DataArray with [time, variable, lat, lon]
        # Assumes lat and lon are 1D in source grib2 files
        lat_dim, lon_dim = field.dims[-2], field.dims[-1]
        lat_size, lon_size = field.sizes[lat_dim], field.sizes[lon_dim]

        out = xr.DataArray(
            data=np.empty((1, len(variables), lat_size, lon_size), dtype=field.dtype),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [actual_time],
                "variable": variables,
                "lat": field.coords["lat"],
                "lon": field.coords["lon"],
            },
        )

        # Map each requested variable via MRMS lexicon; MRMS data contains the product already
        for i, v in enumerate(variables):
            # Translate to ensure variable is valid for MRMS
            try:
                _, modifier = MRMSLexicon[v]
            except KeyError as e:
                raise KeyError(f"Variable id {v} not found in MRMS lexicon.") from e
            out[0, i] = modifier(field.values)

        return out

    def _download_and_decompress(self, s3_uri: str) -> str:
        """Download gzipped GRIB2 from S3 and decompress into cache; return path."""
        # Cache filenames derived from key
        key_hash = (
            s3_uri.replace("s3://", "").replace("/", "_").replace(".grib2.gz", "")
        )
        gz_path = os.path.join(self.cache, f"{key_hash}.grib2.gz")
        grib_path = os.path.join(self.cache, f"{key_hash}.grib2")

        # Download gz if not present
        if not pathlib.Path(gz_path).is_file():
            with self.fs.open(s3_uri, mode="rb") as fsrc, open(gz_path, "wb") as fdst:
                fdst.write(fsrc.read())

        # Decompress to .grib2 if needed
        if not pathlib.Path(grib_path).is_file():
            with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)

        return grib_path

    def _s3_key(self, time: datetime) -> str:
        """Build the S3 object key for a given time."""
        date_str = time.strftime("%Y%m%d")
        hms = time.strftime("%H%M%S")
        filename = f"MRMS_{self.product}_{date_str}-{hms}.grib2.gz"
        return f"CONUS/{self.product}/{date_str}/{filename}"

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify requested times are within MRMS availability window.

        Parameters
        ----------
        times : list[datetime]
                List of date times to fetch data for.
        """
        now_utc = datetime.now(timezone.utc)
        for t in times:
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t < self.EARLIEST_AVAILABLE:
                raise ValueError(
                    f"Requested date time {t} needs to be after {self.EARLIEST_AVAILABLE.date()} for MRMS"
                )
            if t > now_utc:
                raise ValueError(
                    f"Requested date time {t} must not be in the future for MRMS"
                )

    def _resolve_s3_time(self, time: datetime) -> tuple[datetime, str] | None:
        """Find nearest-available S3 object within the configured minute tolerance.

        Returns the resolved timestamp and full S3 URI if found, else None.
        """
        # 1) Exact match fast path
        key = self._s3_key(time)
        s3_uri = f"s3://{self.MRMS_BUCKET_NAME}/{key}"
        if self.fs.exists(s3_uri):
            return time, s3_uri

        # 2) If no tolerance, stop early
        if self.max_offset_minutes <= 0:
            return None

        # 3) List candidate days (can cross day boundary within tolerance)
        t_min = time - timedelta(minutes=self.max_offset_minutes)
        t_max = time + timedelta(minutes=self.max_offset_minutes)
        candidate_dates = {
            t_min.strftime("%Y%m%d"),
            time.strftime("%Y%m%d"),
            t_max.strftime("%Y%m%d"),
        }

        pattern = re.compile(
            rf"^MRMS_{re.escape(self.product)}_(\d{{8}})-(\d{{6}})\.grib2\.gz$"
        )
        best_dt: datetime | None = None
        best_uri: str | None = None
        best_diff: float = float("inf")

        for date_str in sorted(candidate_dates):
            dir_uri = f"s3://{self.MRMS_BUCKET_NAME}/CONUS/{self.product}/{date_str}/"
            try:
                keys = self.fs.ls(dir_uri)
            except FileNotFoundError:
                continue
            for key_path in keys:
                fname = os.path.basename(key_path)
                m = pattern.match(fname)
                if not m:
                    continue
                ds, hms = m.group(1), m.group(2)
                try:
                    ts = datetime(
                        year=int(ds[0:4]),
                        month=int(ds[4:6]),
                        day=int(ds[6:8]),
                        hour=int(hms[0:2]),
                        minute=int(hms[2:4]),
                        second=int(hms[4:6]),
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    continue
                diff = abs((ts - time).total_seconds())
                if diff <= self.max_offset_minutes * 60 and diff < best_diff:
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
        product: str = "MergedReflectivityQCComposite_00.50",
        max_offset_minutes: int = 0,
    ) -> bool:
        """Check if an MRMS file exists for a given time.

        Parameters
        ----------
        time : datetime | np.datetime64
                Time to query (UTC).
        product : str, optional
                Specific MRMS product to check, by default ``MergedReflectivityQCComposite_00.50``.
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
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        if time < cls.EARLIEST_AVAILABLE:
            return False
        if time > datetime.now(timezone.utc):
            return False

        fs = s3fs.S3FileSystem(anon=True)
        date_str = time.strftime("%Y%m%d")
        hms = time.strftime("%H%M%S")
        key = f"CONUS/{product}/{date_str}/MRMS_{product}_{date_str}-{hms}.grib2.gz"
        s3_uri = f"s3://{cls.MRMS_BUCKET_NAME}/{key}"
        if fs.exists(s3_uri):
            return True

        if max_offset_minutes <= 0:
            return False

        # Check nearby timestamps by listing candidate days
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
        for d in sorted(candidate_dates):
            dir_uri = f"s3://{cls.MRMS_BUCKET_NAME}/CONUS/{product}/{d}/"
            try:
                keys = fs.ls(dir_uri)
            except FileNotFoundError:
                continue
            for key_path in keys:
                fname = os.path.basename(key_path)
                m = pattern.match(fname)
                if not m:
                    continue
                ds, hms = m.group(1), m.group(2)
                try:
                    ts = datetime(
                        year=int(ds[0:4]),
                        month=int(ds[4:6]),
                        day=int(ds[6:8]),
                        hour=int(hms[0:2]),
                        minute=int(hms[2:4]),
                        second=int(hms[4:6]),
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    continue
                if abs((ts - time).total_seconds()) <= max_offset_minutes * 60:
                    return True
        return False
