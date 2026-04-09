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
from concurrent.futures import ThreadPoolExecutor
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
    "npp": datetime(2023, 9, 6),
    "n20": datetime(2023, 9, 6),
    "n21": datetime(2023, 9, 6),
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


@dataclass
class _CrISDecodedGranule:
    """Compact decoded data from a single CrIS granule.

    Stores spatial arrays (one value per FOV) and the 2-D radiance
    matrix.  The expensive channel-expansion into long-format rows is
    deferred until :py:meth:`JPSS_CRIS._compile_dataframe` where all
    granules are expanded in a single batch.
    """

    lat: np.ndarray  # (n_valid,) float32
    lon: np.ndarray  # (n_valid,) float32
    sat_za: np.ndarray  # (n_valid,) float32
    sat_aza: np.ndarray  # (n_valid,) float32
    sol_za: np.ndarray  # (n_valid,) float32
    sol_aza: np.ndarray  # (n_valid,) float32
    quality: np.ndarray  # (n_valid,) uint16
    times: np.ndarray  # (n_valid,) datetime64[ms]
    radiance: np.ndarray  # (n_valid, n_channels) float32
    satellite: str
    variable: str


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
    subsample : int, optional
        Temporal sub-sampling factor applied at the granule level.  Each
        CrIS granule covers roughly 32 seconds of observations; setting
        ``subsample=N`` selects every *N*-th granule from the time-ordered
        list, reducing data volume by approximately that factor.  By
        default 1 (no sub-sampling).
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
        subsample: int = 1,
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
        self._subsample = max(1, int(subsample))
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

        SDR and GEO directory listings are issued concurrently.
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
                        geo_prefix = (
                            f"{bucket}/CrIS-SDR-GEO/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )

                        # Issue both listings concurrently
                        sdr_coro = self.fs._ls(sdr_prefix, detail=False)  # type: ignore[union-attr]
                        geo_coro = self.fs._ls(geo_prefix, detail=False)  # type: ignore[union-attr]

                        sdr_listing: list[str] = []
                        geo_listing: list[str] = []
                        try:
                            sdr_listing, geo_listing = await asyncio.gather(
                                sdr_coro, geo_coro
                            )
                        except FileNotFoundError:
                            # One or both directories missing — try individually
                            try:
                                sdr_listing = await self.fs._ls(  # type: ignore[union-attr]
                                    sdr_prefix, detail=False
                                )
                            except FileNotFoundError:
                                logger.warning(f"No CrIS data at s3://{sdr_prefix}")
                            try:
                                geo_listing = await self.fs._ls(  # type: ignore[union-attr]
                                    geo_prefix, detail=False
                                )
                            except FileNotFoundError:
                                logger.warning(f"No CrIS GEO data at s3://{geo_prefix}")

                        if not sdr_listing:
                            day += timedelta(days=1)
                            continue

                        # Build GEO lookup keyed by granule key
                        geo_lookup: dict[str, str] = {}
                        for gpath in geo_listing:
                            gname = gpath.rsplit("/", 1)[-1]
                            if gname.startswith("GCRSO_"):
                                geo_lookup[self._granule_key(gname)] = gpath

                        for path in sdr_listing:
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
        """Decode cached HDF5 files and assemble the output DataFrame.

        HDF5 decoding is parallelised across threads.  Both h5py I/O and
        numpy compute release the GIL, so threads give effective speedup
        without the serialisation overhead of multiprocessing.

        Each granule is decoded into a compact :class:`_CrISDecodedGranule`
        (spatial arrays + 2-D radiance matrix).  The expensive expansion
        to long-format (one row per channel per FOV) is done once at the
        end over all granules combined, avoiding intermediate DataFrame
        allocation per granule.
        """

        def _decode_one(task: _CrISAsyncTask) -> _CrISDecodedGranule | None:
            sdr_path = self._cache_path(task.sdr_uri)
            geo_path = self._cache_path(task.geo_uri)

            if not pathlib.Path(sdr_path).is_file():
                logger.warning(f"Cached SDR file missing for {task.sdr_uri}")
                return None
            if not pathlib.Path(geo_path).is_file():
                logger.warning(f"Cached GEO file missing for {task.geo_uri}")
                return None

            try:
                return self._decode_hdf5(sdr_path, geo_path, task)
            except Exception:
                logger.warning(f"Failed to decode {task.sdr_uri}", exc_info=True)
                return None

        # Sub-sample granules (temporal sub-sampling) before decoding
        if self._subsample > 1:
            tasks = tasks[:: self._subsample]

        # Decode all granules in parallel using threads
        n_workers = min(len(tasks), self._max_workers) if tasks else 1
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_decode_one, tasks))

        granules = [g for g in results if g is not None]

        if not granules:
            return pd.DataFrame(columns=schema.names)

        # --- Batch: concatenate compact spatial arrays across granules ---
        all_lat = np.concatenate([g.lat for g in granules])
        all_lon = np.concatenate([g.lon for g in granules])
        all_sat_za = np.concatenate([g.sat_za for g in granules])
        all_sat_aza = np.concatenate([g.sat_aza for g in granules])
        all_sol_za = np.concatenate([g.sol_za for g in granules])
        all_sol_aza = np.concatenate([g.sol_aza for g in granules])
        all_quality = np.concatenate([g.quality for g in granules])
        all_times = np.concatenate([g.times for g in granules])
        all_radiance = np.concatenate([g.radiance for g in granules])  # (N, n_channels)

        n_total = len(all_lat)
        n_channels = all_radiance.shape[1]

        # Build satellite/variable arrays for dedup support
        # (each granule may have different satellite)
        sat_pieces = [np.broadcast_to(g.satellite, g.lat.shape[0]) for g in granules]
        var_pieces = [np.broadcast_to(g.variable, g.lat.shape[0]) for g in granules]
        all_sat = np.concatenate(sat_pieces)
        all_var = np.concatenate(var_pieces)

        # Free granule list — data is now in the concatenated arrays
        del granules

        # --- Deduplicate at spatial level (before channel expansion) ---
        # Build a structured key: (time_ms, lat_int, lon_int)
        # Using int32 quantization for lat/lon gives ~0.01° precision.
        # Satellite is intentionally excluded — collocated observations
        # from different platforms at the same (time, lat, lon) are treated
        # as duplicates to avoid double-counting in assimilation.
        time_as_i8 = all_times.view(np.int64)
        lat_i = (all_lat * 100).astype(np.int32)
        lon_i = (all_lon * 100).astype(np.int32)

        # Unique spatial points — use numpy lexsort for fast dedup
        order = np.lexsort((lon_i, lat_i, time_as_i8))
        sorted_t = time_as_i8[order]
        sorted_lat = lat_i[order]
        sorted_lon = lon_i[order]
        diffs = (
            (sorted_t[1:] != sorted_t[:-1])
            | (sorted_lat[1:] != sorted_lat[:-1])
            | (sorted_lon[1:] != sorted_lon[:-1])
        )
        unique_mask: np.ndarray = np.empty(n_total, dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = diffs
        keep_idx = order[unique_mask]
        keep_idx.sort()  # preserve original order

        if len(keep_idx) < n_total:
            all_lat = all_lat[keep_idx]
            all_lon = all_lon[keep_idx]
            all_sat_za = all_sat_za[keep_idx]
            all_sat_aza = all_sat_aza[keep_idx]
            all_sol_za = all_sol_za[keep_idx]
            all_sol_aza = all_sol_aza[keep_idx]
            all_quality = all_quality[keep_idx]
            all_times = all_times[keep_idx]
            all_radiance = all_radiance[keep_idx]
            all_sat = all_sat[keep_idx]
            all_var = all_var[keep_idx]
            n_total = len(keep_idx)

        # --- Expand to long-format using PyArrow for efficiency ---
        n_rows = n_total * n_channels

        arrs: dict[str, pa.Array] = {
            "time": pa.array(np.repeat(all_times, n_channels)),
            "class": pa.DictionaryArray.from_arrays(
                np.zeros(n_rows, dtype=np.int8), ["rad"]
            ),
            "lat": pa.array(np.repeat(all_lat, n_channels), type=pa.float32()),
            "lon": pa.array(np.repeat(all_lon, n_channels), type=pa.float32()),
            "scan_angle": pa.array(
                np.repeat(all_sat_za, n_channels), type=pa.float32()
            ),
            "channel_index": pa.array(
                np.tile(np.arange(n_channels, dtype=np.uint16), n_total),
                type=pa.uint16(),
            ),
            "solza": pa.array(np.repeat(all_sol_za, n_channels), type=pa.float32()),
            "solaza": pa.array(np.repeat(all_sol_aza, n_channels), type=pa.float32()),
            "satellite_za": pa.array(
                np.repeat(all_sat_za, n_channels), type=pa.float32()
            ),
            "satellite_aza": pa.array(
                np.repeat(all_sat_aza, n_channels), type=pa.float32()
            ),
            "quality": pa.array(np.repeat(all_quality, n_channels), type=pa.uint16()),
            "observation": pa.array(all_radiance.ravel(), type=pa.float32()),
        }

        # Build satellite and variable using DictionaryArray for memory
        unique_sats = list(dict.fromkeys(all_sat))
        sat_codes = {s: i for i, s in enumerate(unique_sats)}
        sat_indices = np.repeat(
            np.array([sat_codes[s] for s in all_sat], dtype=np.int8), n_channels
        )
        arrs["satellite"] = pa.DictionaryArray.from_arrays(sat_indices, unique_sats)

        unique_vars = list(dict.fromkeys(all_var))
        var_codes = {v: i for i, v in enumerate(unique_vars)}
        var_indices = np.repeat(
            np.array([var_codes[v] for v in all_var], dtype=np.int8), n_channels
        )
        arrs["variable"] = pa.DictionaryArray.from_arrays(var_indices, unique_vars)

        # Select only schema-requested columns
        schema_names = [n for n in schema.names if n in arrs]
        tbl = pa.table({n: arrs[n] for n in schema_names})
        result = tbl.to_pandas()

        result.attrs["source"] = self.SOURCE_ID
        return result

    def _decode_hdf5(
        self,
        sdr_path: str,
        geo_path: str,
        task: _CrISAsyncTask,
    ) -> _CrISDecodedGranule | None:
        """Decode a CrIS SDR + GEO HDF5 file pair into compact arrays.

        Returns a :class:`_CrISDecodedGranule` containing spatial arrays
        (one element per valid FOV) and a 2-D radiance matrix
        ``(n_valid, n_channels)``.  The expensive channel-expansion into
        long-format rows is deferred to
        :py:meth:`_compile_dataframe`.

        Scan lines whose FORTime falls entirely outside the tolerance
        window are skipped before reading the (much larger) radiance
        arrays.

        Parameters
        ----------
        sdr_path : str
            Local path to the CrIS SDR HDF5 file.
        geo_path : str
            Local path to the CrIS GEO HDF5 file.
        task : _CrISAsyncTask
            Task metadata (tolerance bounds, satellite, variable, modifier).
        """
        # IET epoch for vectorized time conversion
        iet_epoch = np.datetime64("1958-01-01T00:00:00", "us")
        tmin_dt64 = np.datetime64(task.datetime_min, "ms")
        tmax_dt64 = np.datetime64(task.datetime_max, "ms")

        # --- Phase 1: Read GEO time first (tiny) to filter scan lines ---
        with h5py.File(geo_path, "r") as geo:
            for_time = geo[_GEO_KEYS["for_time"]][:]  # (n_scan, 30) IET µs

        # Convert FORTime to datetime64 for each (scan, FOR)
        time_dt64 = (iet_epoch + for_time.astype("timedelta64[us]")).astype(
            "datetime64[ms]"
        )
        # A scan line is relevant if ANY FOR in that scan is within window
        scan_has_valid_time = np.any(
            (time_dt64 >= tmin_dt64) & (time_dt64 <= tmax_dt64) & (for_time > 0),
            axis=1,
        )  # (n_scan,)

        if not scan_has_valid_time.any():
            return None

        valid_scans = np.where(scan_has_valid_time)[0]

        # --- Phase 2: Read only the scan lines we need ---
        with h5py.File(sdr_path, "r") as sdr, h5py.File(geo_path, "r") as geo:
            # HDF5 fancy indexing with sorted indices (efficient for contiguous)
            rad_lw = sdr[_SDR_RADIANCE_KEYS["LW"]][valid_scans]
            rad_mw = sdr[_SDR_RADIANCE_KEYS["MW"]][valid_scans]
            rad_sw = sdr[_SDR_RADIANCE_KEYS["SW"]][valid_scans]

            try:
                qf3 = sdr[_SDR_QF_KEYS["QF3"]][valid_scans]
            except KeyError:
                qf3 = np.zeros(
                    (len(valid_scans), _CRIS_NUM_FOR, _CRIS_NUM_FOV, 3),
                    dtype=np.uint8,
                )

            lat = geo[_GEO_KEYS["lat"]][valid_scans]
            lon = geo[_GEO_KEYS["lon"]][valid_scans]
            sat_za = geo[_GEO_KEYS["sat_za"]][valid_scans]
            sat_aza = geo[_GEO_KEYS["sat_aza"]][valid_scans]
            sol_za = geo[_GEO_KEYS["sol_za"]][valid_scans]
            sol_aza = geo[_GEO_KEYS["sol_aza"]][valid_scans]

        # Use the pre-read time but only for valid scans
        for_time = for_time[valid_scans]

        n_scan = len(valid_scans)

        # Concatenate radiance: (n_scan, n_for, n_fov, n_channels)
        radiance = np.concatenate([rad_lw, rad_mw, rad_sw], axis=-1)
        del rad_lw, rad_mw, rad_sw  # free memory early
        n_for = radiance.shape[1]
        n_fov = radiance.shape[2]
        n_channels = radiance.shape[3]
        n_spatial = n_scan * n_for * n_fov

        # Combine QF3 band flags into single uint16 for the quality column.
        # These flags are exposed as metadata; filtering is left to the caller.
        qf_combined = (
            qf3[:, :, :, 0].astype(np.uint16)
            | (qf3[:, :, :, 1].astype(np.uint16) << 4)
            | (qf3[:, :, :, 2].astype(np.uint16) << 8)
        )
        del qf3

        # Flatten spatial dims
        lat_flat = lat.reshape(-1).astype(np.float32)
        lon_flat = lon.reshape(-1).astype(np.float32)

        # Expand for_time to (n_scan, n_for, n_fov)
        for_time_3d = np.broadcast_to(
            for_time[:, :, np.newaxis], (n_scan, n_for, n_fov)
        )
        time_flat = for_time_3d.reshape(-1)

        # Convert times for tolerance filtering
        times_dt64 = (iet_epoch + time_flat.astype("timedelta64[us]")).astype(
            "datetime64[ms]"
        )

        # Valid spatial mask: good lat/lon, positive time, AND within tolerance
        valid_spatial = (
            (lat_flat >= -90.0)
            & (lat_flat <= 90.0)
            & (lon_flat >= -180.1)
            & (lon_flat <= 360.1)
            & (time_flat > 0)
            & (times_dt64 >= tmin_dt64)
            & (times_dt64 <= tmax_dt64)
        )

        if not valid_spatial.any():
            return None

        # Apply spatial mask
        lat_valid = lat_flat[valid_spatial]
        lon_valid = lon_flat[valid_spatial] % 360.0
        times_valid = times_dt64[valid_spatial]

        sat_za_valid = sat_za.reshape(-1)[valid_spatial].astype(np.float32)
        sat_aza_valid = sat_aza.reshape(-1)[valid_spatial].astype(np.float32)
        sol_za_valid = sol_za.reshape(-1)[valid_spatial].astype(np.float32)
        sol_aza_valid = sol_aza.reshape(-1)[valid_spatial].astype(np.float32)
        qf_valid = qf_combined.reshape(-1)[valid_spatial].astype(np.uint16)

        # Radiance: (n_valid, n_channels)
        radiance_valid = radiance.reshape(n_spatial, n_channels)[valid_spatial]
        del radiance  # free the large array

        # Apply modifier
        radiance_valid = task.modifier(radiance_valid).astype(np.float32)

        # Mask out fill values: set them to NaN (kept for now, could filter)
        bad = (radiance_valid <= -1e6) | (radiance_valid >= 1e6)
        if bad.any():
            radiance_valid = radiance_valid.copy()
            radiance_valid[bad] = np.float32("nan")

        return _CrISDecodedGranule(
            lat=lat_valid,
            lon=lon_valid,
            sat_za=sat_za_valid,
            sat_aza=sat_aza_valid,
            sol_za=sol_za_valid,
            sol_aza=sol_aza_valid,
            quality=qf_valid,
            times=times_valid,
            radiance=radiance_valid,
            satellite=task.satellite,
            variable=task.variable,
        )

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
