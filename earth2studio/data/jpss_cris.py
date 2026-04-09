# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import h5py
import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger

from earth2studio.data.utils import (
    datasource_cache_root,
    gather_with_concurrency,
    prep_data_inputs,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.jpss import JPSSCrISLexicon
from earth2studio.utils.time import TimeTolerance, normalize_time_tolerance
from earth2studio.utils.type import TimeArray, VariableArray

# ---------------------------------------------------------------------------
# NOAA CrIS S3 bucket layout
# s3://noaa-nesdis-n20-pds/CrIS-FS-SDR/<YYYY>/<MM>/<DD>/SCRIF_*.h5
# s3://noaa-nesdis-n20-pds/CrIS-SDR-GEO/<YYYY>/<MM>/<DD>/GCRSO_*.h5
# ---------------------------------------------------------------------------

# S3 bucket per satellite short-name
_SAT_BUCKET_MAP: dict[str, str] = {
    "n20": "noaa-nesdis-n20-pds",
    "n21": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}

# Platform identifier in filenames
_SAT_PLATFORM_MAP: dict[str, str] = {
    "n20": "j01",
    "n21": "j02",
    "npp": "npp",
}

# Reverse mapping: platform code -> satellite short-name
_PLATFORM_SAT_MAP: dict[str, str] = {v: k for k, v in _SAT_PLATFORM_MAP.items()}

# Earliest date with CrIS FSR data on S3 per satellite
_SAT_START_DATE: dict[str, datetime] = {
    "npp": datetime(2022, 1, 1),
    "n20": datetime(2022, 1, 1),
    "n21": datetime(2022, 1, 1),
}

# ---------------------------------------------------------------------------
# CrIS instrument constants
# ---------------------------------------------------------------------------
_CRIS_NUM_CHANNELS_LW: int = 717
_CRIS_NUM_CHANNELS_MW: int = 869
_CRIS_NUM_CHANNELS_SW: int = 637
_CRIS_NUM_CHANNELS: int = (
    _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW + _CRIS_NUM_CHANNELS_SW
)  # 2223

_CRIS_NUM_FOR: int = 30  # Fields of Regard per scan line
_CRIS_NUM_FOV: int = 9  # Fields of View per FOR (3x3 detector array)

# IET epoch: 1958-01-01T00:00:00 UTC (International Atomic Time reference)
# Offset from Unix epoch (1970-01-01) to IET epoch (1958-01-01) in microseconds
_IET_EPOCH = datetime(1958, 1, 1)

# HDF5 dataset paths in SDR files (CrIS-FS-SDR)
_SDR_GROUP = "All_Data/CrIS-FS-SDR_All"
_SDR_RADIANCE_KEYS = {
    "LW": f"{_SDR_GROUP}/ES_RealLW",  # shape (n_scan, 30, 9, 717)
    "MW": f"{_SDR_GROUP}/ES_RealMW",  # shape (n_scan, 30, 9, 869)
    "SW": f"{_SDR_GROUP}/ES_RealSW",  # shape (n_scan, 30, 9, 637)
}

# Quality flag datasets
_SDR_QF_KEYS = {
    "QF1": f"{_SDR_GROUP}/QF1_SCAN_CRISSDR",  # shape (n_scan,)
    "QF2": f"{_SDR_GROUP}/QF2_CRISSDR",  # shape (n_scan, 9, 3)
    "QF3": f"{_SDR_GROUP}/QF3_CRISSDR",  # shape (n_scan, 30, 9, 3)
}

# HDF5 dataset paths in GEO files (CrIS-SDR-GEO)
_GEO_GROUP = "All_Data/CrIS-SDR-GEO_All"
_GEO_KEYS = {
    "lat": f"{_GEO_GROUP}/Latitude",  # shape (n_scan, 30, 9)
    "lon": f"{_GEO_GROUP}/Longitude",  # shape (n_scan, 30, 9)
    "height": f"{_GEO_GROUP}/Height",  # shape (n_scan, 30, 9)
    "sat_za": f"{_GEO_GROUP}/SatelliteZenithAngle",  # shape (n_scan, 30, 9)
    "sat_aza": f"{_GEO_GROUP}/SatelliteAzimuthAngle",  # shape (n_scan, 30, 9)
    "sol_za": f"{_GEO_GROUP}/SolarZenithAngle",  # shape (n_scan, 30, 9)
    "sol_aza": f"{_GEO_GROUP}/SolarAzimuthAngle",  # shape (n_scan, 30, 9)
    "for_time": f"{_GEO_GROUP}/FORTime",  # shape (n_scan, 30) IET µs
}


def _iet_to_datetime(iet_us: int) -> datetime:
    """Convert IET microseconds since 1958-01-01 to a Python datetime.

    Parameters
    ----------
    iet_us : int
        Microseconds since 1958-01-01T00:00:00 UTC.

    Returns
    -------
    datetime
        Corresponding UTC datetime.
    """
    return _IET_EPOCH + timedelta(microseconds=int(iet_us))


@dataclass
class _CrISAsyncTask:
    """Metadata for a single CrIS granule download task (SDR + GEO pair)."""

    sdr_uri: str
    geo_uri: str
    datetime_min: datetime
    datetime_max: datetime
    satellite: str
    variable: str
    modifier: Callable[[Any], Any]


class JPSS_CRIS:
    """JPSS CrIS (Cross-track Infrared Sounder) Full Spectral Resolution (FSR)
    Level 1 spectral radiance observations served from NOAA Open Data on AWS.

    Each HDF5 granule contains a small number of scan lines, each with 30
    Fields of Regard (FOR) and 9 Fields of View (FOV) per FOR (3x3 detector
    array).  In FSR mode the instrument produces 2223 spectral channels:

    - **LWIR** (9.14--15.38 µm, 650--1095 cm^-1): 717 channels at 0.625 cm^-1
    - **MWIR** (5.71--8.26 µm, 1210--1750 cm^-1): 869 channels at 0.625 cm^-1
    - **SWIR** (3.92--4.64 µm, 2155--2550 cm^-1): 637 channels at 0.625 cm^-1

    The returned :class:`~pandas.DataFrame` has one row per FOV per channel,
    following the same convention as :class:`~earth2studio.data.JPSS_ATMS`.
    The ``channel_index`` column uses 0-based sequential indexing across all
    three bands (LW: 0--716, MW: 717--1585, SW: 1586--2222).

    Data is stored as paired HDF5 files on S3:

    - **SDR** (``SCRIF_*.h5``): spectral radiance arrays
    - **GEO** (``GCRSO_*.h5``): geolocation (lat, lon, angles, time)

    Parameters
    ----------
    satellites : list[str] | None, optional
        Satellite short-names to query.  Valid values are ``"n20"``
        (NOAA-20), ``"n21"`` (NOAA-21), and ``"npp"`` (Suomi NPP).
        By default ``None``, which queries all valid satellites.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    cache : bool, optional
        Cache downloaded HDF5 files locally, by default True
    verbose : bool, optional
        Show download progress bars, by default True
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600
    max_workers : int, optional
        Maximum number of concurrent S3 fetch tasks, by default 24
    retries : int, optional
        Per-file retry count on transient I/O failures, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.  Each CrIS granule pair
    (SDR + GEO) is approximately 16 MB.

    Note
    ----
    Additional information on the data repository:

    - https://registry.opendata.aws/noaa-jpss/
    - https://www.star.nesdis.noaa.gov/jpss/CrIS.php
    - https://www.nesdis.noaa.gov/current-satellite-missions/currently-flying/joint-polar-satellite-system

    CrIS SDR format and channel specification:

    - https://ncc.nesdis.noaa.gov/documents/documentation/viirs-users-guide-tech-report-142a-v1.3.pdf

    Badges
    ------
    region:global dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.JPSS_CRIS"
    VALID_SATELLITES = frozenset(["n20", "n21", "npp"])

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("class"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            pa.field(
                "scan_angle",
                pa.float32(),
                nullable=True,
                metadata={"description": "SatelliteZenithAngle from CrIS GEO file"},
            ),
            E2STUDIO_SCHEMA.field("channel_index"),
            E2STUDIO_SCHEMA.field("solza"),
            E2STUDIO_SCHEMA.field("solaza"),
            E2STUDIO_SCHEMA.field("satellite_za"),
            E2STUDIO_SCHEMA.field("satellite_aza"),
            E2STUDIO_SCHEMA.field("quality"),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        satellites: list[str] | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        max_workers: int = 24,
        retries: int = 3,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid options: {sorted(self.VALID_SATELLITES)}"
                )
        self._satellites = satellites
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs: s3fs.S3FileSystem | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------
    async def _async_init(self) -> None:
        """Initialise the async S3 filesystem."""
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={},
            asynchronous=True,
            skip_instance_cache=True,
        )

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch CrIS FSR spectral radiance observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return (e.g. ``["crisfsr"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None (all).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with one row per FOV per channel.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        try:
            df = loop.run_until_complete(
                asyncio.wait_for(
                    self.fetch(time, variable, fields), timeout=self.async_timeout
                )
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async implementation of the data fetch.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialised! If calling this function "
                "directly, make sure the data source is initialised inside "
                "the async loop!"
            )

        session = await self.fs.set_session(refresh=True)

        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)
        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate variables
        for v in variable_list:
            try:
                JPSSCrISLexicon[v]  # type: ignore
            except KeyError:
                logger.error(f"Variable id {v} not found in JPSS CrIS lexicon")
                raise

        # Discover and download HDF5 file pairs within tolerance windows
        tasks = await self._create_tasks(time_list, variable_list)

        # Deduplicate by S3 URI (both SDR and GEO)
        uri_set: set[str] = set()
        for t in tasks:
            uri_set.add(t.sdr_uri)
            uri_set.add(t.geo_uri)

        fetch_jobs = [self._fetch_remote_file(uri) for uri in uri_set]
        await gather_with_concurrency(
            fetch_jobs,
            max_workers=self._max_workers,
            desc="Fetching CrIS HDF5 files",
            verbose=(not self._verbose),
        )

        if session:
            await session.close()

        # Decode and compile
        df = self._compile_dataframe(tasks, schema)
        return df

    # ------------------------------------------------------------------
    # Task creation -- discover CrIS granules in S3
    # ------------------------------------------------------------------
    async def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[_CrISAsyncTask]:
        """Build download tasks by listing the S3 day-directory.

        For each requested time +/- tolerance we list the relevant day
        directories on each satellite bucket and select SDR files whose
        embedded start-timestamp falls within the tolerance window.  The
        corresponding GEO file is paired by matching the common filename
        fields (platform, date, start time, end time, orbit).
        """
        tasks: list[_CrISAsyncTask] = []

        for v in variable_list:
            _, modifier = JPSSCrISLexicon[v]  # type: ignore

            for sat in self._satellites:
                bucket = _SAT_BUCKET_MAP[sat]

                for t in time_list:
                    tmin = t + self._tolerance_lower
                    tmax = t + self._tolerance_upper

                    # Iterate over calendar days covered by the window
                    day = tmin.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_day = tmax.replace(hour=0, minute=0, second=0, microsecond=0)

                    while day <= end_day:
                        sdr_prefix = (
                            f"{bucket}/CrIS-FS-SDR/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )
                        try:
                            listing = await self.fs._ls(sdr_prefix, detail=False)  # type: ignore[union-attr]
                        except FileNotFoundError:
                            logger.warning(f"No CrIS data at s3://{sdr_prefix}")
                            day += timedelta(days=1)
                            continue

                        # List the matching GEO directory and build a
                        # lookup keyed by (platform, date, start, end, orbit).
                        geo_prefix = (
                            f"{bucket}/CrIS-SDR-GEO/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )
                        geo_lookup: dict[str, str] = {}
                        try:
                            geo_listing = await self.fs._ls(geo_prefix, detail=False)  # type: ignore[union-attr]
                            for gpath in geo_listing:
                                gname = gpath.rsplit("/", 1)[-1]
                                if gname.startswith("GCRSO_"):
                                    geo_lookup[self._granule_key(gname)] = gpath
                        except FileNotFoundError:
                            logger.warning(f"No CrIS GEO data at s3://{geo_prefix}")

                        for path in listing:
                            fname = path.rsplit("/", 1)[-1]
                            if not fname.startswith("SCRIF_"):
                                continue

                            file_time = self._parse_filename_time(fname)
                            if file_time is None:
                                continue
                            if tmin <= file_time <= tmax:
                                sdr_key = self._granule_key(fname)
                                if sdr_key not in geo_lookup:
                                    logger.warning(f"No matching GEO file for {fname}")
                                    continue
                                tasks.append(
                                    _CrISAsyncTask(
                                        sdr_uri=f"s3://{path}",
                                        geo_uri=f"s3://{geo_lookup[sdr_key]}",
                                        datetime_min=tmin,
                                        datetime_max=tmax,
                                        satellite=sat,
                                        variable=v,
                                        modifier=modifier,
                                    )
                                )

                        day += timedelta(days=1)

        return tasks

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, s3_uri: str) -> None:
        """Download a single HDF5 file to local cache (with retry)."""
        local_path = self._cache_path(s3_uri)
        if pathlib.Path(local_path).is_file():
            return

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                data = await self.fs._cat_file(s3_uri.replace("s3://", "", 1))  # type: ignore[union-attr]
                with open(local_path, "wb") as fh:
                    fh.write(data)
                return
            except (OSError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        logger.warning(f"Failed to fetch {s3_uri} after {self._retries} retries")
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # HDF5 decoding & DataFrame compilation
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        tasks: list[_CrISAsyncTask],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode cached HDF5 files and assemble the output DataFrame."""
        frames: list[pd.DataFrame] = []

        for task in tasks:
            sdr_path = self._cache_path(task.sdr_uri)
            geo_path = self._cache_path(task.geo_uri)

            if not pathlib.Path(sdr_path).is_file():
                logger.warning(f"Cached SDR file missing for {task.sdr_uri}")
                continue
            if not pathlib.Path(geo_path).is_file():
                logger.warning(f"Cached GEO file missing for {task.geo_uri}")
                continue

            try:
                df = self._decode_hdf5(sdr_path, geo_path, task)
            except Exception:
                logger.warning(f"Failed to decode {task.sdr_uri}", exc_info=True)
                continue

            if df.empty:
                continue

            # Filter by time tolerance window
            mask = (df["time"] >= task.datetime_min) & (df["time"] <= task.datetime_max)
            df = df.loc[mask]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=schema.names)

        result = pd.concat(frames, ignore_index=True)

        # Deduplicate in case overlapping tolerance windows hit the same granule
        dedup_cols = [
            c
            for c in (
                "time",
                "lat",
                "lon",
                "channel_index",
                "satellite",
                "variable",
            )
            if c in result.columns
        ]
        if dedup_cols:
            result = result.drop_duplicates(subset=dedup_cols, ignore_index=True)

        result.attrs["source"] = self.SOURCE_ID
        return result[[name for name in schema.names if name in result.columns]]

    def _decode_hdf5(
        self,
        sdr_path: str,
        geo_path: str,
        task: _CrISAsyncTask,
    ) -> pd.DataFrame:
        """Decode a CrIS SDR + GEO HDF5 file pair into a DataFrame.

        The SDR file contains spectral radiance arrays with shape
        ``(n_scan, 30, 9, n_channels)`` for each band (LW, MW, SW).
        The GEO file contains geolocation with shape
        ``(n_scan, 30, 9)`` and time with shape ``(n_scan, 30)``.

        Each combination of (scan, FOR, FOV, channel) produces one row.
        """
        rows: list[dict] = []

        with h5py.File(sdr_path, "r") as sdr, h5py.File(geo_path, "r") as geo:
            # Read radiance arrays
            rad_lw = sdr[_SDR_RADIANCE_KEYS["LW"]][:]  # (n_scan, 30, 9, 717)
            rad_mw = sdr[_SDR_RADIANCE_KEYS["MW"]][:]  # (n_scan, 30, 9, 869)
            rad_sw = sdr[_SDR_RADIANCE_KEYS["SW"]][:]  # (n_scan, 30, 9, 637)

            n_scan = rad_lw.shape[0]

            # Read quality flags
            try:
                # QF3: per-FOR-per-FOV per-band quality, (n_scan, 30, 9, 3)
                qf3 = sdr[_SDR_QF_KEYS["QF3"]][:]
            except KeyError:
                qf3 = np.zeros(
                    (n_scan, _CRIS_NUM_FOR, _CRIS_NUM_FOV, 3), dtype=np.uint8
                )

            # Read GEO arrays
            lat = geo[_GEO_KEYS["lat"]][:]  # (n_scan, 30, 9)
            lon = geo[_GEO_KEYS["lon"]][:]  # (n_scan, 30, 9)
            sat_za = geo[_GEO_KEYS["sat_za"]][:]  # (n_scan, 30, 9)
            sat_aza = geo[_GEO_KEYS["sat_aza"]][:]  # (n_scan, 30, 9)
            sol_za = geo[_GEO_KEYS["sol_za"]][:]  # (n_scan, 30, 9)
            sol_aza = geo[_GEO_KEYS["sol_aza"]][:]  # (n_scan, 30, 9)
            for_time = geo[_GEO_KEYS["for_time"]][:]  # (n_scan, 30) IET µs

        # Concatenate radiance across bands: (n_scan, n_for, n_fov, n_channels)
        radiance = np.concatenate([rad_lw, rad_mw, rad_sw], axis=-1)
        n_for = radiance.shape[1]
        n_fov = radiance.shape[2]
        n_channels = radiance.shape[3]

        # Combine quality flags: OR the three band-specific QF3 flags into
        # a single uint16 per (scan, FOR, FOV).  Shift band flags into
        # distinct bit positions: LW=bits 0-3, MW=bits 4-7, SW=bits 8-11.
        qf_combined = (
            qf3[:, :, :, 0].astype(np.uint16)
            | (qf3[:, :, :, 1].astype(np.uint16) << 4)
            | (qf3[:, :, :, 2].astype(np.uint16) << 8)
        )  # (n_scan, n_for, n_fov)

        # Build rows vectorised per scan line
        for s in range(n_scan):
            for f in range(n_for):
                # Time from FORTime (shared across all FOVs in this FOR)
                iet_val = int(for_time[s, f])
                if iet_val <= 0:
                    continue
                obs_time = _iet_to_datetime(iet_val)

                for v in range(n_fov):
                    lat_val = float(lat[s, f, v])
                    lon_val = float(lon[s, f, v])

                    # Skip fill values (typically -999.x)
                    if lat_val < -90.0 or lat_val > 90.0:
                        continue
                    if lon_val < -180.1 or lon_val > 360.1:
                        continue

                    # Normalise longitude to [0, 360)
                    lon_val = lon_val % 360.0

                    sat_za_val = float(sat_za[s, f, v])
                    sat_aza_val = float(sat_aza[s, f, v])
                    sol_za_val = float(sol_za[s, f, v])
                    sol_aza_val = float(sol_aza[s, f, v])
                    qf_val = int(qf_combined[s, f, v])

                    for ch in range(n_channels):
                        raw_val = float(radiance[s, f, v, ch])

                        # Skip missing / fill values
                        if raw_val < -1e6 or raw_val > 1e6:
                            continue

                        val = float(task.modifier(raw_val))

                        rows.append(
                            {
                                "time": obs_time,
                                "class": "rad",
                                "lat": lat_val,
                                "lon": lon_val,
                                "scan_angle": sat_za_val,
                                "channel_index": ch,
                                "solza": sol_za_val,
                                "solaza": sol_aza_val,
                                "satellite_za": sat_za_val,
                                "satellite_aza": sat_aza_val,
                                "quality": qf_val,
                                "satellite": task.satellite,
                                "observation": val,
                                "variable": task.variable,
                            }
                        )

        if not rows:
            return pd.DataFrame(columns=self.SCHEMA.names)

        df = pd.DataFrame(rows)
        # Enforce schema dtypes
        df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ms]")
        df["lat"] = df["lat"].astype(np.float32)
        df["lon"] = df["lon"].astype(np.float32)
        df["scan_angle"] = df["scan_angle"].astype(np.float32)
        df["channel_index"] = df["channel_index"].astype(np.uint16)
        df["solza"] = df["solza"].astype(np.float32)
        df["solaza"] = df["solaza"].astype(np.float32)
        df["satellite_za"] = df["satellite_za"].astype(np.float32)
        df["satellite_aza"] = df["satellite_aza"].astype(np.float32)
        df["quality"] = df["quality"].astype(np.uint16)
        df["observation"] = df["observation"].astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Filename parsing and pairing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_filename_time(filename: str) -> datetime | None:
        """Extract the granule start time from a CrIS SDR filename.

        Expected pattern::

            SCRIF_{platform}_d{YYYYMMDD}_t{HHMMSSF}_e{HHMMSSF}_b{orbit}_c{creation}_oebc_ops.h5

        The ``t`` field encodes the start time as HHMMSS followed by a
        tenths-of-second digit.

        Returns ``None`` if the filename does not match.
        """
        parts = filename.split("_")
        # Find the date part (dYYYYMMDD) and time part (tHHMMSSF)
        date_str: str | None = None
        time_str: str | None = None

        for part in parts:
            if part.startswith("d") and len(part) == 9:
                date_str = part[1:]  # YYYYMMDD
            elif part.startswith("t") and len(part) == 8:
                time_str = part[1:7]  # HHMMSS (ignore tenths digit)

        if date_str is None or time_str is None:
            return None

        try:
            return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _granule_key(filename: str) -> str:
        """Extract the matching key from an SDR or GEO filename.

        Both SDR (SCRIF) and GEO (GCRSO) files share a common key
        formed by the platform, date, start-time, end-time, and orbit
        fields.  The creation timestamp (``_c`` field) differs between
        the two products, so it is excluded from the key.

        Parameters
        ----------
        filename : str
            Filename (basename only) of an SDR or GEO HDF5 file.

        Returns
        -------
        str
            Key string ``"{platform}_{date}_{start}_{end}_{orbit}"``.
        """
        parts = filename.split("_")
        # parts: [prefix, platform, d<date>, t<start>, e<end>, b<orbit>, ...]
        if len(parts) >= 6:
            return "_".join(parts[1:6])  # platform_d*_t*_e*_b*
        return filename  # fallback – should not happen for valid files

    # ------------------------------------------------------------------
    # resolve_fields / cache / available
    # ------------------------------------------------------------------
    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert *fields* parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification.

        Returns
        -------
        pa.Schema
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not in SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(field.name).type
                if field.type != expected:
                    raise TypeError(
                        f"Field '{field.name}' type {field.type} != {expected}"
                    )
            return fields

        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in SCHEMA. Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "jpss_cris")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_jpss_cris_{self._tmp_cache_hash}"
            )
        return cache_location

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic local cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode())
        return os.path.join(self.cache, sha.hexdigest())

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are within the data range.

        Parameters
        ----------
        times : list[datetime]
            Date-times to validate.
        """
        start_date = min(_SAT_START_DATE.values())
        for t in times:
            if t < start_date:
                raise ValueError(
                    f"Requested date time {t} needs to be after "
                    f"{start_date} for JPSS CrIS"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether data is available for a given time.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date-time to check.

        Returns
        -------
        bool
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
