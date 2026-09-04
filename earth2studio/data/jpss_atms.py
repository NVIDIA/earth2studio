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
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import (
    AsyncListableStore,
    _sync_async,
    datasource_cache_root,
    gather_with_concurrency,
    obstore_list_prefix,
    obstore_read_range,
    obstore_store_from_url,
    prep_data_inputs,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.jpss import (
    JPSSATMSLexicon,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import TimeTolerance, normalize_time_tolerance
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import eccodes
except ImportError:
    OptionalDependencyFailure("data")
    eccodes = None  # type: ignore[assignment]

# eccodes keeps global state in its underlying C library and is NOT
# thread-safe.  Decoding runs in worker threads (via asyncio.to_thread) so
# it can overlap with in-flight downloads, but this lock serializes the
# actual eccodes decode calls: only one BUFR file is decoded at a time.
_ECCODES_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# NOAA-20 S3 bucket layout
# s3://noaa-nesdis-n20-pds/ATMS_BUFR/<YYYY>/<MM>/<DD>/*.bufr
# ---------------------------------------------------------------------------

# Satellite identifier mapping (WMO code 001007)
_SAT_ID_MAP: dict[int, str] = {
    224: "npp",  # Suomi NPP
    225: "n20",  # NOAA-20 / JPSS-1
    226: "n21",  # NOAA-21 / JPSS-2
}

# S3 bucket per satellite short-name
_SAT_BUCKET_MAP: dict[str, str] = {
    "n20": "noaa-nesdis-n20-pds",
    "n21": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}

# Earliest date with ATMS BUFR data on S3 per satellite
_SAT_START_DATE: dict[str, datetime] = {
    "npp": datetime(2023, 9, 6),
    "n20": datetime(2023, 9, 6),
    "n21": datetime(2023, 9, 6),
}

# ---------------------------------------------------------------------------
# ATMS cross-track geometry and scan timing constants
#
# Source: JPSS ATMS SDR Algorithm Theoretical Basis Document (ATBD),
#   D0001-M01-S01-001_JPSS_ATBD_ATMS-SDR_B, Version 1, 2022-04-27,
#   Section 3 ("Instrument Description"), pp. 8-9.
#   https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-001_JPSS_ATBD_ATMS_SDR_D.pdf
#
# The antenna completes 3 revolutions in 8 seconds (scan period = 8/3 s).
# Each scan cycle samples 96 Earth-scene FOVs at ~18 ms integration each,
# with an angular sampling interval of 1.11°.  The total angular range
# from FOV-1 center to FOV-96 center is 95 × 1.11° = 105.45° (±52.725°
# from nadir).  The scan speed is ~61.6°/s.
#
# Integration time per FOV (18 ms) also confirmed by:
#   Weng et al. (2012), "Introduction to Suomi NPP ATMS for NWP and
#   tropical cyclone applications", J. Geophys. Res., 117, D19112,
#   doi:10.1029/2012JD018144, Section 2.
# ---------------------------------------------------------------------------
_ATMS_NUM_FOVS: int = 96
_ATMS_TOTAL_SCAN_DEG: float = 105.45  # 95 × 1.11°, per ATBD §3
_ATMS_DEG_PER_FOV: float = _ATMS_TOTAL_SCAN_DEG / (_ATMS_NUM_FOVS - 1)  # 1.11°

# Scan period: 3 revolutions in 8 seconds → 8/3 s per scan line (ATBD §3).
_ATMS_SCAN_PERIOD_S: float = 8.0 / 3.0  # 2.667 s

# Each FOV is sampled for ~18 ms with scan speed ~61.6°/s (ATBD §3).
# The 96 FOVs span ~1.73 s of the 2.667 s scan cycle (~65% duty cycle).
_ATMS_FOV_DWELL_S: float = 0.018  # 18 ms per FOV

# ATMS channel center frequencies (GHz), 1-indexed → 0-indexed array.
# Source: JPSS ATMS SDR ATBD, Table 2-1.
_ATMS_CHANNEL_FREQ_GHZ: np.ndarray = np.array(
    [
        23.8,
        31.4,
        50.3,
        51.76,
        52.8,
        53.596,
        54.40,
        54.94,
        55.50,
        57.29,
        57.29,
        57.29,
        57.29,
        57.29,
        57.29,
        88.20,
        165.5,
        183.31,
        183.31,
        183.31,
        183.31,
        183.31,
    ],
    dtype=np.float64,
)

# Speed of light in cm/s for frequency (GHz) → wavenumber (cm^-1) conversion.
_C_CM_S: float = 2.99792458e10
_ATMS_CHANNEL_WAVENUMBER: np.ndarray = _ATMS_CHANNEL_FREQ_GHZ * 1e9 / _C_CM_S


def _fov_to_scan_angle(fov: float | np.ndarray) -> float | np.ndarray:
    """Convert a 1-indexed field-of-view number to scan angle in degrees.

    Parameters
    ----------
    fov : float | np.ndarray
        Field-of-view number (1–96), scalar or per-FOV array.

    Returns
    -------
    float | np.ndarray
        Scan angle in degrees (negative = left of nadir, positive = right).
    """
    return (fov - (_ATMS_NUM_FOVS + 1) / 2.0) * _ATMS_DEG_PER_FOV


def _fov_to_time_offset(fov: float) -> timedelta:
    """Compute the sub-second time offset for a given FOV within a scan line.

    ATMS samples 96 FOVs sequentially at ~18 ms per step.  The BUFR time
    fields only carry integer-second precision, so this offset is added to
    recover approximate sub-second timing.  FOV 1 is the first sample
    (offset = 0); FOV 96 is the last (offset ≈ 1.71 s).

    Source: ATMS SDR ATBD (D0001-M01-S01-001), §3, p. 8-9.

    Parameters
    ----------
    fov : float
        Field-of-view number (1–96, 1-indexed).

    Returns
    -------
    timedelta
        Sub-second offset from the start of the scan line.
    """
    return timedelta(seconds=(fov - 1) * _ATMS_FOV_DWELL_S)


@dataclass
class _ATMSAsyncTask:
    """Metadata for a single BUFR file download task."""

    s3_uri: str
    datetime_min: datetime
    datetime_max: datetime
    satellite: str
    variable: str
    bufr_key: str
    modifier: Callable[[Any], Any]


@check_optional_dependencies()
class JPSS_ATMS:
    """JPSS ATMS (Advanced Technology Microwave Sounder) Level 1 BUFR
    brightness-temperature observations served from NOAA Open Data on AWS.

    Each BUFR file contains a single scan line with 96 cross-track
    field-of-view (FOV) positions and 22 microwave channels.
    The returned :class:`~pandas.DataFrame` has one row per FOV per channel,
    following the same convention as [`UFSObsSat`][earth2studio.data.UFSObsSat].

    ATMS has 22 channels spanning 23.8--183.31 GHz.  The ``sensor_index``
    column (1--22) identifies each channel:

    .. list-table:: ATMS Channel Specification
       :header-rows: 1
       :widths: 8 15 35

       * - Channel
         - Frequency (GHz)
         - Primary Sensitivity
       * - 1
         - 23.8
         - Window / water-vapour (surface)
       * - 2
         - 31.4
         - Window (surface emissivity, cloud liquid water)
       * - 3
         - 50.3
         - Oxygen (lower troposphere temperature)
       * - 4
         - 51.76
         - Oxygen (lower troposphere temperature)
       * - 5
         - 52.8
         - Oxygen (troposphere temperature)
       * - 6
         - 53.596
         - Oxygen (troposphere temperature)
       * - 7
         - 54.40
         - Oxygen (mid-troposphere temperature)
       * - 8
         - 54.94
         - Oxygen (mid-troposphere temperature)
       * - 9
         - 55.50
         - Oxygen (upper troposphere temperature)
       * - 10
         - 57.29 (fO1)
         - Oxygen (tropopause temperature)
       * - 11
         - 57.29 (fO2)
         - Oxygen (lower stratosphere temperature)
       * - 12
         - 57.29 (fO3)
         - Oxygen (stratosphere temperature)
       * - 13
         - 57.29 (fO4)
         - Oxygen (stratosphere temperature)
       * - 14
         - 57.29 (fO5)
         - Oxygen (upper stratosphere temperature)
       * - 15
         - 57.29 (fO6)
         - Oxygen (upper stratosphere temperature)
       * - 16
         - 88.20
         - Window (precipitation, sea ice)
       * - 17
         - 165.5
         - Window (precipitation, ice cloud)
       * - 18
         - 183.31 (fH1)
         - Water-vapour (upper troposphere humidity)
       * - 19
         - 183.31 (fH2)
         - Water-vapour (mid-troposphere humidity)
       * - 20
         - 183.31 (fH3)
         - Water-vapour (mid-troposphere humidity)
       * - 21
         - 183.31 (fH4)
         - Water-vapour (lower troposphere humidity)
       * - 22
         - 183.31 (fH5)
         - Water-vapour (lower troposphere humidity)

    Channels 10--15 share the 57.29 GHz oxygen-absorption line but use
    different passband offsets (fO1--fO6), giving each a distinct altitude
    weighting function.  Channels 18--22 similarly share 183.31 GHz with
    different offsets (fH1--fH5) for water-vapour profiling at different
    altitudes.

    Parameters
    ----------
    satellites : list[str] | None, optional
        Satellite short-names to query.  Valid values are ``"n20"``
        (NOAA-20), ``"n21"`` (NOAA-21), and ``"npp"`` (Suomi NPP).
        By default ``None``, which queries all valid satellites.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    cache : bool, optional
        Cache downloaded BUFR files locally, by default True
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
    of data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository:

    - https://registry.opendata.aws/noaa-jpss/
    - https://www.star.nesdis.noaa.gov/jpss/ATMS.php
    - https://www.nesdis.noaa.gov/current-satellite-missions/currently-flying/joint-polar-satellite-system

    ATMS channel specification and BUFR SDR format:

    - https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-001_JPSS_ATBD_ATMS_SDR_D.pdf


    Badges
    ------
    region:global dataclass:observation product:sat provider:noaa
    """

    SOURCE_ID = "earth2studio.data.JPSS_ATMS"
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
                metadata={"bufr_name": "fieldOfViewNumber (converted to degrees)"},
            ),
            E2STUDIO_SCHEMA.field("sensor_index"),
            E2STUDIO_SCHEMA.field("wavenumber"),
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

        # Object stores (one per satellite bucket) are lazily initialized
        # on first call
        self.stores: dict[str, AsyncListableStore] | None = None
        # Memoized S3 day-directory listings, one cache dict per bucket
        # (the same bucket-relative prefix exists in multiple buckets).
        # Only day directories that can no longer gain files are cached.
        self._listing_caches: dict[str, dict[str, list[str]]] = {}

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------
    async def _async_init(self) -> None:
        """Async initialization of the per-bucket object stores

        Note
        ----
        Unlike async fsspec filesystems, obstore stores are event-loop
        independent and could be built in ``__init__``; kept as a lazy async
        method to preserve the initialization seam.
        """
        buckets = {_SAT_BUCKET_MAP[sat] for sat in self._satellites}
        self.stores = {
            bucket: obstore_store_from_url(
                f"s3://{bucket}", max_pool_connections=self._max_workers
            )
            for bucket in buckets
        }
        self._listing_caches = {bucket: {} for bucket in buckets}

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch ATMS brightness-temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return (e.g. ``["atms"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None (all).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with one row per FOV per channel.
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
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
        if self.stores is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)
        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate variables
        for v in variable_list:
            try:
                JPSSATMSLexicon[v]  # type: ignore
            except KeyError:
                logger.error(f"Variable id {v} not found in JPSS ATMS lexicon")
                raise

        # Discover and download BUFR files within tolerance windows
        tasks = await self._create_tasks(time_list, variable_list)

        # Deduplicate decode work: multiple tasks can reference the same
        # file when tolerance windows overlap.  The URI already encodes
        # bucket (satellite) and file; the variable is included in the key
        # for safety since _decode_bufr uses task.bufr_key/modifier/variable.
        decode_tasks: dict[tuple[str, str], _ATMSAsyncTask] = {}
        for task in tasks:
            decode_tasks.setdefault((task.s3_uri, task.variable), task)

        # Pipeline: decode each file (in a worker thread, serialized by
        # _ECCODES_LOCK) as soon as its download lands, while other
        # downloads continue in flight.
        decoded: dict[tuple[str, str], pd.DataFrame] = {}

        async def _fetch_and_decode(key: tuple[str, str], task: _ATMSAsyncTask) -> None:
            await self._fetch_remote_file(task.s3_uri)
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                # Missing-file warning is emitted per task in
                # _compile_dataframe
                return

            def _decode() -> pd.DataFrame:
                # eccodes has global state and is not thread-safe: hold the
                # lock for the whole decode so only one file decodes at a
                # time (still overlapping with network I/O).
                with _ECCODES_LOCK:
                    return self._decode_bufr(local_path, task)

            try:
                decoded[key] = await asyncio.to_thread(_decode)
            except Exception:
                logger.warning(f"Failed to decode {task.s3_uri}", exc_info=True)

        await gather_with_concurrency(
            [_fetch_and_decode(key, task) for key, task in decode_tasks.items()],
            max_workers=self._max_workers,
            desc="Fetching ATMS BUFR files",
            verbose=(not self._verbose),
        )

        # Compile the decoded frames in task order
        df = self._compile_dataframe(tasks, schema, decoded)
        return df

    # ------------------------------------------------------------------
    # Task creation – discover BUFR granules in S3
    # ------------------------------------------------------------------
    async def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[_ATMSAsyncTask]:
        """Build download tasks by listing the S3 day-directory.

        For each requested time ± tolerance we list the relevant day
        directories on each satellite bucket and select files whose
        embedded start-timestamp falls within the tolerance window.
        """
        tasks: list[_ATMSAsyncTask] = []

        for v in variable_list:
            bufr_key, modifier = JPSSATMSLexicon[v]  # type: ignore

            for sat in self._satellites:
                bucket = _SAT_BUCKET_MAP[sat]

                for t in time_list:
                    tmin = t + self._tolerance_lower
                    tmax = t + self._tolerance_upper

                    # Iterate over calendar days covered by the window
                    day = tmin.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_day = tmax.replace(hour=0, minute=0, second=0, microsecond=0)

                    while day <= end_day:
                        if self.stores is None:
                            raise ValueError("Object stores are not initialized")
                        day_prefix = (
                            f"ATMS_BUFR/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )
                        # Day directories that can still gain files (today,
                        # allowing an hour of upload latency) bypass the
                        # per-bucket listing memoization
                        listing = await obstore_list_prefix(
                            self.stores[bucket],
                            day_prefix,
                            cache=self._listing_caches.setdefault(bucket, {}),
                            cacheable=day + timedelta(days=1, hours=1)
                            <= datetime.now(timezone.utc).replace(tzinfo=None),
                        )
                        if not listing:
                            logger.warning(
                                f"No ATMS data at s3://{bucket}/{day_prefix}"
                            )
                            day += timedelta(days=1)
                            continue

                        for key in listing:
                            fname = key.rsplit("/", 1)[-1]
                            file_time = self._parse_filename_time(fname)
                            if file_time is None:
                                continue
                            if tmin <= file_time <= tmax:
                                tasks.append(
                                    _ATMSAsyncTask(
                                        # Keys are bucket-relative; the full
                                        # s3://bucket/key form matches the
                                        # historical cache-key scheme
                                        s3_uri=f"s3://{bucket}/{key}",
                                        datetime_min=tmin,
                                        datetime_max=tmax,
                                        satellite=sat,
                                        variable=v,
                                        bufr_key=bufr_key,
                                        modifier=modifier,
                                    )
                                )

                        day += timedelta(days=1)

        return tasks

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, s3_uri: str) -> None:
        """Download a single BUFR file to local cache (with retry)."""
        local_path = self._cache_path(s3_uri)
        if pathlib.Path(local_path).is_file():
            return

        if self.stores is None:
            raise ValueError("Object stores are not initialized")
        bucket, key = s3_uri.removeprefix("s3://").split("/", 1)

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                data = await obstore_read_range(self.stores[bucket], key)
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
    # BUFR decoding & DataFrame compilation
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        tasks: list[_ATMSAsyncTask],
        schema: pa.Schema,
        decoded: dict[tuple[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        """Assemble the output DataFrame from pre-decoded per-file frames.

        ``decoded`` maps ``(s3_uri, variable)`` to the DataFrame decoded from
        that file; entries are absent when the decode failed (already warned
        during the fetch/decode pipeline).
        """
        frames: list[pd.DataFrame] = []

        for task in tasks:
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {task.s3_uri}")
                continue

            df = decoded.get((task.s3_uri, task.variable))
            if df is None or df.empty:
                # None: decode failed (warning already emitted); skip
                continue

            # Filter by time tolerance window
            mask = (df["time"] >= task.datetime_min) & (df["time"] <= task.datetime_max)
            df = df.loc[mask]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=schema.names)

        result = pd.concat(frames, ignore_index=True)

        # When multiple requested times have overlapping tolerance windows
        # the same BUFR file may appear in more than one task.  Downloads
        # and decodes are deduplicated by (uri, variable), but each task
        # still contributes its own window-filtered slice of the shared
        # frame, so identical observations can end up in ``frames`` twice.
        # Drop exact duplicates to prevent this.
        dedup_cols = [
            c
            for c in (
                "time",
                "lat",
                "lon",
                "sensor_index",
                "satellite",
                "variable",
            )
            if c in result.columns
        ]
        if dedup_cols:
            result = result.drop_duplicates(subset=dedup_cols, ignore_index=True)

        result.attrs["source"] = self.SOURCE_ID
        return result[[name for name in schema.names if name in result.columns]]

    def _decode_bufr(
        self,
        path: str,
        task: _ATMSAsyncTask,
    ) -> pd.DataFrame:
        """Decode a single ATMS BUFR file into a DataFrame.

        Each BUFR message contains *N* subsets (FOVs).  For each subset the
        brightness temperature array has *C* channel values, yielding
        ``N * C`` rows in the output.
        """
        tables: list[pa.Table] = []

        with open(path, "rb") as fh:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(fh)
                if msgid is None:
                    break
                try:
                    eccodes.codes_set(msgid, "unpack", 1)

                    n_subsets = eccodes.codes_get(msgid, "numberOfSubsets")

                    # Extract per-FOV arrays (length = n_subsets)
                    lat = eccodes.codes_get_array(msgid, "latitude")
                    lon = eccodes.codes_get_array(msgid, "longitude")
                    fov = eccodes.codes_get_array(msgid, "fieldOfViewNumber")
                    solza = eccodes.codes_get_array(msgid, "solarZenithAngle")
                    solaza = eccodes.codes_get_array(msgid, "solarAzimuth")
                    sat_za = eccodes.codes_get_array(msgid, "satelliteZenithAngle")
                    sat_aza = eccodes.codes_get_array(msgid, "bearingOrAzimuth")

                    # Brightness temperature array is channel-major:
                    # [ch1_fov0, ch1_fov1, ..., ch1_fovN, ch2_fov0, ...]
                    # i.e. shape (n_channels, n_fov) when reshaped, then
                    # transposed to (n_fov, n_channels) for row iteration.
                    bt_flat = eccodes.codes_get_array(msgid, task.bufr_key)
                    n_channels = JPSSATMSLexicon.ATMS_NUM_CHANNELS
                    n_fov = n_subsets

                    if bt_flat.size != n_fov * n_channels:
                        logger.warning(
                            f"Unexpected BT array size {bt_flat.size} in {path}, "
                            f"expected {n_fov}×{n_channels}. Skipping message."
                        )
                        continue

                    bt = bt_flat.reshape(n_channels, n_fov).T

                    # Per-channel quality flags (shape n_channels, one per channel)
                    try:
                        cqf_raw = eccodes.codes_get_array(
                            msgid, "channelDataQualityFlags"
                        )
                        if cqf_raw.size == n_channels:
                            # One flag per channel (shared across all FOVs)
                            cqf = cqf_raw.astype(np.uint16)
                        elif cqf_raw.size == n_fov * n_channels:
                            # Per-FOV per-channel: reshape to (n_channels, n_fov)
                            # and transpose to (n_fov, n_channels) so we can
                            # index cqf_per_fov[i, ch] later.
                            cqf = cqf_raw.reshape(n_channels, n_fov).T.astype(np.uint16)
                        elif cqf_raw.size >= n_channels:
                            # Unexpected size — take first n_channels entries
                            logger.debug(
                                f"channelDataQualityFlags unexpected size "
                                f"{cqf_raw.size}, using first {n_channels}"
                            )
                            cqf = cqf_raw[:n_channels].astype(np.uint16)
                        else:
                            cqf = np.zeros(n_channels, dtype=np.uint16)
                    except Exception:
                        cqf = np.zeros(n_channels, dtype=np.uint16)

                    # Time fields — may be scalars or per-subset arrays
                    # depending on the BUFR producer.  Use codes_get_array
                    # for all of them and broadcast scalars to length n_fov.
                    def _get_time_array(key: str) -> np.ndarray:
                        try:
                            arr = eccodes.codes_get_array(msgid, key)
                        except Exception:
                            arr = np.zeros(n_fov)
                        if arr.size == 1:
                            arr = np.full(n_fov, arr[0])
                        return arr.astype(int)

                    years = _get_time_array("year")
                    months = _get_time_array("month")
                    days = _get_time_array("day")
                    hours = _get_time_array("hour")
                    minutes = _get_time_array("minute")
                    seconds = _get_time_array("second")

                    # Satellite id
                    try:
                        sat_id = int(eccodes.codes_get(msgid, "satelliteIdentifier"))
                        sat_name = _SAT_ID_MAP.get(sat_id, task.satellite)
                    except Exception:
                        sat_name = task.satellite

                    # Build rows vectorized: one per (FOV, channel), FOV-major
                    # so the row order matches the historical nested loop.

                    # Base integer-second timestamps per FOV; invalid
                    # component combinations coerce to NaT (the row loop
                    # skipped these FOVs via ValueError/OverflowError)
                    base_time = pd.to_datetime(
                        {
                            "year": years,
                            "month": months,
                            "day": days,
                            "hour": hours,
                            "minute": minutes,
                            "second": seconds,
                        },
                        errors="coerce",
                    ).to_numpy()

                    # Add sub-second offset based on FOV position in the
                    # scan line.  BUFR only carries integer-second
                    # timestamps; the offset recovers ~18 ms per-FOV
                    # timing from the ATMS scan geometry (ATBD §3).
                    offsets = np.array(
                        [_fov_to_time_offset(float(f)) for f in fov],
                        dtype="timedelta64[us]",
                    )
                    obs_time = base_time + offsets

                    # Fill-value test kept verbatim from the row loop: only
                    # values > 1e6 or < 0 are dropped (NaN fails both
                    # comparisons and is therefore kept)
                    bt_flat = bt.reshape(-1)
                    keep = ~((bt_flat > 1e6) | (bt_flat < 0))
                    keep &= np.repeat(~np.isnat(obs_time), n_channels)
                    if not keep.any():
                        continue

                    if cqf.ndim == 2:
                        quality = cqf.reshape(-1)
                    else:
                        quality = np.tile(cqf, n_fov)

                    n_out = int(keep.sum())
                    columns = {
                        "time": np.repeat(obs_time, n_channels)[keep].astype(
                            "datetime64[ms]"
                        ),
                        "class": np.full(n_out, "rad", dtype=object),
                        "lat": np.repeat(lat, n_channels)[keep].astype(np.float32),
                        "lon": np.repeat(lon % 360.0, n_channels)[keep].astype(
                            np.float32
                        ),
                        "scan_angle": np.repeat(
                            _fov_to_scan_angle(fov.astype(np.float64)), n_channels
                        )[keep].astype(np.float32),
                        "sensor_index": np.tile(
                            np.arange(1, n_channels + 1, dtype=np.uint16), n_fov
                        )[keep],
                        "wavenumber": np.tile(
                            _ATMS_CHANNEL_WAVENUMBER[:n_channels], n_fov
                        )[keep],
                        "solza": np.repeat(solza, n_channels)[keep].astype(np.float32),
                        "solaza": np.repeat(solaza, n_channels)[keep].astype(
                            np.float32
                        ),
                        "satellite_za": np.repeat(sat_za, n_channels)[keep].astype(
                            np.float32
                        ),
                        "satellite_aza": np.repeat(sat_aza, n_channels)[keep].astype(
                            np.float32
                        ),
                        "quality": quality[keep].astype(np.uint16),
                        "satellite": np.full(n_out, sat_name, dtype=object),
                        "observation": np.asarray(task.modifier(bt_flat[keep])).astype(
                            np.float32
                        ),
                        "variable": np.full(n_out, task.variable, dtype=object),
                    }
                    tables.append(
                        pa.table(
                            {
                                name: pa.array(columns[name])
                                for name in self.SCHEMA.names
                            }
                        )
                    )
                finally:
                    eccodes.codes_release(msgid)

        if not tables:
            return pd.DataFrame(columns=self.SCHEMA.names)

        # Numpy columns above already carry the schema dtypes; one Arrow
        # concat + to_pandas replaces per-row dict assembly and the
        # column-by-column astype passes
        return pa.concat_tables(tables).to_pandas()

    # ------------------------------------------------------------------
    # File-name timestamp parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_filename_time(filename: str) -> datetime | None:
        """Extract the scan start time from an ATMS BUFR filename.

        Expected pattern::

            ATMS_v1r0_j01_s<YYYYMMDDHHMMSSF>_e..._c....bufr

        Returns ``None`` if the filename does not match.
        """
        parts = filename.split("_")
        for part in parts:
            if part.startswith("s") and len(part) >= 15:
                try:
                    return datetime.strptime(part[1:15], "%Y%m%d%H%M%S")
                except ValueError:
                    return None
        return None

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
        cache_location = os.path.join(datasource_cache_root(), "jpss_atms")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_jpss_atms_{self._tmp_cache_hash}"
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
        # Use the earliest S3 start date across all satellites
        start_date = min(_SAT_START_DATE.values())
        for t in times:
            if t < start_date:
                raise ValueError(
                    f"Requested date time {t} needs to be after "
                    f"{start_date} for JPSS ATMS"
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
