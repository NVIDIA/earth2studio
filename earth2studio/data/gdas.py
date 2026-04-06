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
import struct
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import (
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    prep_data_inputs,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.gdas import GDASObsConvLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance, timearray_to_datetime
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    from pybufrkit.decoder import Decoder as BufrDecoder
    from pybufrkit.tables import TableGroupCacheManager
except ImportError:
    OptionalDependencyFailure("data")
    BufrDecoder = None  # type: ignore[assignment,misc]
    TableGroupCacheManager = None  # type: ignore[assignment,misc]

NOMADS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"

# Mapping from BUFR section-1 dataCategory (internal NCEP encoding) to
# the human-readable PrepBUFR message-type class string.
# Ref: NCEP PrepBUFR documentation and embedded DX Table A.
PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper air: radiosondes, pilot balloons, dropsondes
    104: "AIRCFT",  # Aircraft: AIREP, PIREP, AMDAR, TAMDAR
    105: "SATWND",  # Satellite-derived winds
    107: "VADWND",  # VAD (NEXRAD) winds
    109: "ADPSFC",  # Surface land: METAR, synoptic
    110: "SFCSHP",  # Surface marine: ships, buoys, C-MAN
    112: "GPSIPW",  # GPS precipitable water
    113: "SYNDAT",  # Synthetic bogus data
    119: "RASSDA",  # RASS virtual temperature
    121: "ASCATW",  # ASCAT scatterometer winds
}

# Maximum age of data on NOMADS (approximate – production dir retains ~2 days)
_MAX_AGE_DAYS = 2

# PrepBUFR descriptor IDs for header fields
_HDR_SID = 1194  # Station ID (CCITT IA5, 64 bits)
_HDR_XOB = 6240  # Longitude (degrees east)
_HDR_YOB = 5002  # Latitude (degrees north)
_HDR_DHR = 4215  # Obs time minus cycle time (hours)
_HDR_ELV = 10199  # Station elevation (m)
_HDR_TYP = 55007  # Report type code
_HDR_T29 = 55008  # Data dump report type code

# PrepBUFR descriptor IDs for observation categories and levels
_OBS_CAT = 8193  # Observation category code
_OBS_POB = 7245  # Pressure observation (MB)
_OBS_PQM = 7246  # Pressure quality mark
_OBS_ZOB = 10007  # Height observation (m)
_OBS_TOB = 12245  # Temperature observation (DEG C)
_OBS_TQM = 12246  # Temperature quality mark
_OBS_QOB = 13245  # Specific humidity (MG/KG)
_OBS_QQM = 13246  # Moisture quality mark
_OBS_UOB = 11003  # U-wind component (M/S)
_OBS_VOB = 11004  # V-wind component (M/S)
_OBS_WQM = 11240  # Wind quality mark
_OBS_DDO = 11001  # Wind direction (DEGREES TRUE)
_OBS_FFO = 11252  # Wind speed (KNOTS)
_OBS_PWO = 13193  # Precipitable water (KG/M**2)
_OBS_PMO = 10243  # Mean sea level pressure (MB)
_OBS_TDO = 12244  # Dewpoint temperature (DEG C)

# Set of all observation-level descriptor IDs we care about
_OBSERVATION_DESCR_IDS: set[int] = {
    _OBS_POB,
    _OBS_PQM,
    _OBS_ZOB,
    _OBS_TOB,
    _OBS_TQM,
    _OBS_QOB,
    _OBS_QQM,
    _OBS_UOB,
    _OBS_VOB,
    _OBS_WQM,
    _OBS_DDO,
    _OBS_FFO,
    _OBS_PWO,
    _OBS_PMO,
    _OBS_TDO,
}

# Header descriptor IDs
_HEADER_DESCR_IDS: set[int] = {
    _HDR_SID,
    _HDR_XOB,
    _HDR_YOB,
    _HDR_DHR,
    _HDR_ELV,
    _HDR_TYP,
    _HDR_T29,
}


@dataclass
class _GDASAsyncTask:
    """Async task for fetching a single PrepBUFR cycle file."""

    url: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    variables: list[str]


@check_optional_dependencies()
class NomadsGDASObsConv:
    """Real-time GDAS conventional observation data from NOAA NOMADS PrepBUFR.

    Provides near-real-time access to quality-controlled conventional
    (in-situ) observations from the NOAA Global Data Assimilation System
    (GDAS). Data is sourced from PrepBUFR files on NOMADS, updated 4 times
    daily (00z, 06z, 12z, 18z) with approximately 6-10 hours latency.

    Observation types include radiosondes (ADPUPA), surface stations (ADPSFC),
    aircraft (AIRCAR/AIRCFT), ships and buoys (SFCSHP), wind profilers
    (PROFLR), satellite-derived winds (SATWND), and GPS precipitable water
    (GPSIPW).

    The output schema matches :class:`UFSObsConv` with the addition of a
    ``quality`` field containing the PrepBUFR quality control marker.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default np.timedelta64(10, "m").
    max_workers : int, optional
        Maximum concurrent async download tasks, by default 4.
    decode_workers : int, optional
        Number of parallel processes for BUFR message decoding.  Higher values
        speed up decoding of large PrepBUFR files at the cost of more memory.
        Set to 1 to disable multiprocessing, by default 8.
    cache : bool, optional
        Cache downloaded PrepBUFR files locally, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch, by default 600.
    retries : int, optional
        Number of retry attempts per failed download with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large
    amount of data to your local machine for large requests. Each 6-hourly
    PrepBUFR file is approximately 60-70 MB.

    Note
    ----
    Additional information on the data:

    - https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm
    - https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php

    Data is retained on the NOMADS production server for approximately
    2 days. Older data should be retrieved from the UFS GEFSv13 Replay
    dataset via :class:`UFSObsConv`.

    Badges
    ------
    region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "NomadsGDASObsConv"

    SCHEMA: pa.Schema = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            pa.field(
                "pres",
                pa.float32(),
                nullable=True,
                metadata={"description": "Observation pressure level (Pa)"},
            ),
            pa.field(
                "elev",
                pa.float32(),
                nullable=True,
                metadata={"description": "Observation height / elevation (m)"},
            ),
            pa.field(
                "type",
                pa.uint16(),
                nullable=True,
                metadata={
                    "description": (
                        "PrepBUFR observation type code (NCEP Table 2). "
                        "See https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_2.htm"
                    )
                },
            ),
            E2STUDIO_SCHEMA.field("class"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("station"),
            E2STUDIO_SCHEMA.field("station_elev"),
            E2STUDIO_SCHEMA.field("quality"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
        ]
    )

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        max_workers: int = 4,
        decode_workers: int = 8,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        retries: int = 3,
    ) -> None:
        self._tolerance_lower, self._tolerance_upper = normalize_time_tolerance(
            time_tolerance
        )
        # Convert to pytimedelta for datetime arithmetic
        self._tolerance_lower = pd.to_timedelta(self._tolerance_lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(self._tolerance_upper).to_pytimedelta()
        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout
        self._max_workers = max_workers
        self._decode_workers = max(1, decode_workers)
        self._retries = retries
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None  # type: ignore[assignment]

    async def _async_init(self) -> None:
        """Initialize async HTTP filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        from fsspec.implementations.http import HTTPFileSystem

        self.fs = HTTPFileSystem(asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch conventional observation data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in
            :class:`GDASObsConvLexicon`.
        fields : str | list[str] | pa.Schema | None, optional
            Schema fields to include in output. None returns all fields.

        Returns
        -------
        pd.DataFrame
            Observation data matching the requested time/variable window.

        Raises
        ------
        KeyError
            If a variable is not found in the lexicon.
        ValueError
            If requested time is out of valid range.
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
                    self.fetch(time, variable, fields),
                    timeout=self.async_timeout,
                )
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async fetch of conventional observation data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.
        fields : str | list[str] | pa.Schema | None, optional
            Schema fields to include in output.

        Returns
        -------
        pd.DataFrame
            Observation data.
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this "
                "function directly make sure the data source is initialized "
                "inside the async loop!"
            )

        time_list, variable_list = prep_data_inputs(time, variable)
        output_fields = self.resolve_fields(fields)

        self._validate_time(time_list)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Build tasks (one per 6h cycle needed)
        tasks = self._create_tasks(time_list, variable_list)

        # Deduplicate by URL (multiple time/var combos may share a cycle)
        unique_urls: dict[str, _GDASAsyncTask] = {}
        for task in tasks:
            if task.url in unique_urls:
                # Merge variables and widen time window
                existing = unique_urls[task.url]
                existing.variables = list(set(existing.variables) | set(task.variables))
                existing.datetime_min = min(existing.datetime_min, task.datetime_min)
                existing.datetime_max = max(existing.datetime_max, task.datetime_max)
            else:
                unique_urls[task.url] = task

        # Download all unique PrepBUFR files
        fetch_tasks = list(unique_urls.values())

        coros = [self._fetch_wrapper(t.url) for t in fetch_tasks]
        await gather_with_concurrency(
            coros,
            max_workers=self._max_workers,
            desc="Fetching GDAS PrepBUFR",
            verbose=(not self._verbose),
        )

        # Decode and compile
        df = self._compile_dataframe(fetch_tasks, variable_list)

        # Select output columns
        if output_fields is not None:
            df = df[[f for f in output_fields if f in df.columns]]

        df.attrs["source"] = self.SOURCE_ID
        return df

    def _create_tasks(
        self,
        times: list[datetime],
        variables: list[str],
    ) -> list[_GDASAsyncTask]:
        """Build download tasks for required 6h PrepBUFR cycles.

        Parameters
        ----------
        times : list[datetime]
            Requested timestamps.
        variables : list[str]
            Requested variable names.

        Returns
        -------
        list[_GDASAsyncTask]
            List of tasks, one per required cycle.
        """
        # Validate all variables first
        for v in variables:
            if v not in GDASObsConvLexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in GDASObsConvLexicon. "
                    f"Available: {list(GDASObsConvLexicon.VOCAB.keys())}"
                )

        tasks: list[_GDASAsyncTask] = []
        seen_cycles: set[datetime] = set()

        for t in times:
            dt_min = t + self._tolerance_lower
            dt_max = t + self._tolerance_upper

            # Find all 6h cycles that could contain obs in this window
            # Cycles are 00, 06, 12, 18
            cycle_start = dt_min.replace(
                hour=(dt_min.hour // 6) * 6, minute=0, second=0, microsecond=0
            )
            cycle = cycle_start
            while cycle <= dt_max:
                if cycle not in seen_cycles:
                    seen_cycles.add(cycle)
                    url = self._build_url(cycle)
                    tasks.append(
                        _GDASAsyncTask(
                            url=url,
                            datetime_file=cycle,
                            datetime_min=dt_min,
                            datetime_max=dt_max,
                            variables=list(variables),
                        )
                    )
                cycle += timedelta(hours=6)

        return tasks

    async def _fetch_wrapper(self, url: str) -> str:
        """Fetch a single PrepBUFR file with retry logic.

        Wraps :meth:`_fetch_remote_file` with :func:`async_retry` for
        automatic exponential backoff on transient errors.

        Parameters
        ----------
        url : str
            Full URL to the PrepBUFR file.

        Returns
        -------
        str
            Local path to the cached file.
        """
        return await async_retry(
            self._fetch_remote_file,
            url,
            retries=self._retries,
            backoff=1.0,
            task_timeout=120.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )

    async def _fetch_remote_file(self, url: str) -> str:
        """Download a PrepBUFR file from NOMADS.

        Parameters
        ----------
        url : str
            Full URL to the PrepBUFR file.

        Returns
        -------
        str
            Local path to the cached file.
        """
        cache_path = self._cache_path(url)

        if os.path.exists(cache_path):
            logger.debug(f"Cache hit: {cache_path}")
            return cache_path

        data = await self.fs._cat_file(url)  # type: ignore[attr-defined]
        with open(cache_path, "wb") as f:
            f.write(data)
        logger.debug(f"Downloaded {url} -> {cache_path}")
        return cache_path

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate requested times against NOMADS availability.

        Parameters
        ----------
        times : list[datetime]
            Times to validate.

        Raises
        ------
        ValueError
            If time is in the future or older than retention window.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        oldest = now - timedelta(days=_MAX_AGE_DAYS)

        for t in times:
            if t > now + timedelta(hours=12):
                raise ValueError(
                    f"Requested time {t} is in the future. "
                    f"NOMADS GDAS data has ~6-10h latency."
                )
            if t < oldest:
                raise ValueError(
                    f"Requested time {t} is older than ~{_MAX_AGE_DAYS} days. "
                    f"The NOMADS production directory retains only the most "
                    f"recent ~2 days of data. "
                    f"Use UFSObsConv for historical data."
                )

    def _compile_dataframe(
        self,
        tasks: list[_GDASAsyncTask],
        variables: list[str],
    ) -> pd.DataFrame:
        """Decode PrepBUFR files and compile into a single DataFrame.

        Parameters
        ----------
        tasks : list[_GDASAsyncTask]
            Completed download tasks.
        variables : list[str]
            Requested variable names.

        Returns
        -------
        pd.DataFrame
            Combined observation DataFrame.
        """
        all_frames: list[pd.DataFrame] = []

        for task in tasks:
            cache_path = self._cache_path(task.url)
            if not os.path.exists(cache_path):
                logger.warning(f"Missing cached file for {task.url}, skipping")
                continue

            try:
                df = self._decode_prepbufr(
                    cache_path,
                    task.variables,
                    task.datetime_min,
                    task.datetime_max,
                )
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                logger.warning(f"Error decoding {cache_path}: {e}, skipping")

        if not all_frames:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                {
                    name: pd.Series(dtype=self._pa_to_pandas_dtype(field.type))
                    for name, field in zip(self.SCHEMA.names, self.SCHEMA)
                }
            )

        df = pd.concat(all_frames, ignore_index=True)
        return df

    def _decode_prepbufr(
        self,
        path: str,
        variables: list[str],
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        """Decode a PrepBUFR file and extract requested variables.

        Uses pybufrkit with custom NCEP descriptor tables extracted from the
        DX table messages embedded at the start of each PrepBUFR file.
        Messages are decoded in parallel using a process pool for performance.

        Parameters
        ----------
        path : str
            Path to the local PrepBUFR file.
        variables : list[str]
            Earth2Studio variable names to extract.
        dt_min : datetime
            Minimum observation time (inclusive).
        dt_max : datetime
            Maximum observation time (inclusive).

        Returns
        -------
        pd.DataFrame
            Decoded observations with schema-conformant columns.
        """
        var_plan = self._build_extraction_plan(variables)

        with open(path, "rb") as fh:
            file_data = fh.read()

        # Extract NCEP-local descriptor tables from DX messages and
        # split file into individual BUFR message byte strings.
        table_b, table_d, messages = self._parse_prepbufr_messages(file_data)

        # Filter to known message types before decoding
        work_items: list[tuple[bytes, str]] = []
        for msg_bytes, data_cat in messages:
            if data_cat in PREPBUFR_OBS_TYPES:
                work_items.append((msg_bytes, PREPBUFR_OBS_TYPES[data_cat]))

        if not work_items:
            return self._empty_schema_df()

        all_rows: list[dict[str, Any]] = []

        if self._decode_workers > 1 and len(work_items) > 1:
            # Parallel decode using process pool
            with ProcessPoolExecutor(
                max_workers=self._decode_workers,
                initializer=_init_decode_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _decode_message_worker,
                        msg_bytes,
                        obs_class_str,
                        variables,
                        dt_min,
                        dt_max,
                    )
                    for msg_bytes, obs_class_str in work_items
                ]
                for future in futures:
                    try:
                        rows = future.result()
                        if rows:
                            all_rows.extend(rows)
                    except Exception:
                        logger.debug("Worker failed to decode a BUFR message")
        else:
            # Sequential decode (single worker or single message)
            decoder = self._create_decoder(table_b, table_d)
            for msg_bytes, obs_class_str in work_items:
                rows = _decode_message(
                    decoder, msg_bytes, obs_class_str, var_plan, dt_min, dt_max
                )
                all_rows.extend(rows)

        if not all_rows:
            return self._empty_schema_df()

        df = pd.DataFrame(all_rows)

        # Apply variable-specific modifiers (wind decomposition, etc.)
        result_frames: list[pd.DataFrame] = []
        for var_name, plan in var_plan.items():
            _, modifier = plan
            var_df = df[df["variable"] == var_name].copy()
            if not var_df.empty:
                var_df = modifier(var_df)
                result_frames.append(var_df)

        if not result_frames:
            return self._empty_schema_df()

        df = pd.concat(result_frames, ignore_index=True)

        # Ensure only schema columns remain (drop internal columns)
        schema_cols = set(self.SCHEMA.names)
        for col in list(df.columns):
            if col not in schema_cols:
                df = df.drop(columns=[col])

        # Enforce dtypes
        df["time"] = pd.to_datetime(df["time"])
        for name, field in zip(self.SCHEMA.names, self.SCHEMA):
            if name in df.columns and name != "time":
                try:
                    dtype = self._pa_to_pandas_dtype(field.type)
                    if field.nullable and df[name].isna().any():
                        if dtype == np.uint16:
                            df[name] = df[name].astype("UInt16")
                        elif dtype == np.float32:
                            df[name] = df[name].astype("Float32")
                    else:
                        df[name] = df[name].astype(dtype)
                except (ValueError, TypeError):
                    pass

        # Reorder columns to match schema
        df = df[[c for c in self.SCHEMA.names if c in df.columns]]

        return df

    @staticmethod
    def _parse_prepbufr_messages(
        file_data: bytes,
    ) -> tuple[
        dict[int, tuple[Any, ...]],
        dict[int, tuple[Any, ...]],
        list[tuple[bytes, int]],
    ]:
        """Split a PrepBUFR byte stream into messages and extract DX tables.

        The first several messages in a PrepBUFR file are DX table messages
        (dataCategory=11) containing NCEP-local BUFR Table B and Table D
        definitions needed to decode subsequent data messages.

        Parameters
        ----------
        file_data : bytes
            Entire PrepBUFR file contents.

        Returns
        -------
        tuple[dict, dict, list[tuple[bytes, int]]]
            (table_b_dict, table_d_dict, data_messages) where the dicts
            are in pybufrkit ``add_extra_entries`` format and
            data_messages is a list of (message_bytes, data_category) tuples
            for all non-DX messages.
        """
        table_b: dict[int, tuple[Any, ...]] = {}
        table_d: dict[int, tuple[Any, ...]] = {}
        data_messages: list[tuple[bytes, int]] = []
        dx_messages: list[bytes] = []

        # Split into individual BUFR messages
        pos = 0
        while pos < len(file_data):
            idx = file_data.find(b"BUFR", pos)
            if idx == -1:
                break
            # BUFR edition 3/4: message length in bytes 5-7 (3 bytes, big-endian)
            msg_len = struct.unpack(">I", b"\x00" + file_data[idx + 4 : idx + 7])[0]
            msg_bytes = file_data[idx : idx + msg_len]

            # Quick check of dataCategory in section 1.
            # BUFR edition 3: section 0 is 8 bytes, section 1 octet 9
            # (0-indexed byte 8) holds the data category.
            # Absolute offset from message start = 8 + 8 = 16.
            data_cat = file_data[idx + 16] if idx + 16 < len(file_data) else 0
            if data_cat == 11:
                dx_messages.append(msg_bytes)
            else:
                data_messages.append((msg_bytes, data_cat))

            pos = idx + msg_len

        # Decode DX table messages using pybufrkit (they use standard descriptors)
        if dx_messages:
            try:
                dx_decoder = BufrDecoder()
                for dx_bytes in dx_messages:
                    try:
                        dx_msg = dx_decoder.process(dx_bytes)
                    except Exception:  # noqa: S112
                        logger.debug("Skipping unparseable DX table message")
                        continue

                    td = dx_msg.template_data.value
                    dvas = td.decoded_values_all_subsets
                    if not dvas:
                        continue

                    flat = dvas[0]
                    _extract_dx_tables(flat, table_b, table_d)

            except Exception as e:
                logger.warning(f"Failed to extract DX tables: {e}")

        return table_b, table_d, data_messages

    @staticmethod
    def _create_decoder(
        table_b: dict[int, tuple[Any, ...]],
        table_d: dict[int, tuple[Any, ...]],
    ) -> Any:
        """Register custom NCEP tables and create a pybufrkit decoder.

        Parameters
        ----------
        table_b : dict
            NCEP Table B entries.
        table_d : dict
            NCEP Table D entries.

        Returns
        -------
        pybufrkit.decoder.Decoder
            Configured decoder instance.
        """
        TableGroupCacheManager.clear_extra_entries()
        TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
        if table_b or table_d:
            TableGroupCacheManager.add_extra_entries(table_b, table_d)
        return BufrDecoder()

    @staticmethod
    def _extract_subset(
        descs: list[Any],
        vals: list[Any],
        base_time: datetime,
        obs_class_str: str,
        var_plan: dict[str, tuple[str, Callable[..., pd.DataFrame]]],
        dt_min: datetime,
        dt_max: datetime,
    ) -> list[dict[str, Any]]:
        """Extract observation rows from a single decoded PrepBUFR subset.

        Walks descriptor-value pairs, collecting header fields and then
        yielding one row per (observation level, requested variable) pair.

        Parameters
        ----------
        descs : list
            Decoded descriptor objects for this subset.
        vals : list
            Decoded values for this subset.
        base_time : datetime
            Message base time (from section 1).
        obs_class_str : str
            PREPBUFR observation class string (e.g., "ADPUPA").
        var_plan : dict
            Extraction plan from ``_build_extraction_plan``.
        dt_min : datetime
            Minimum observation time (inclusive).
        dt_max : datetime
            Maximum observation time (inclusive).

        Returns
        -------
        list[dict]
            Observation rows suitable for DataFrame construction.
        """
        rows: list[dict[str, Any]] = []

        # ── Pass 1: extract header fields (first ~15 descriptors) ──
        header: dict[str, Any] = {
            "sid": "",
            "xob": None,
            "yob": None,
            "dhr": 0.0,
            "elv": None,
            "typ": None,
            "t29": None,
        }
        header_done = False
        for i, (d, v) in enumerate(zip(descs, vals)):
            did = d.id
            if did == _HDR_SID:
                header["sid"] = (
                    v.decode("ascii", errors="replace").strip()
                    if isinstance(v, bytes)
                    else str(v).strip()
                )
            elif did == _HDR_XOB:
                header["xob"] = v
            elif did == _HDR_YOB:
                header["yob"] = v
            elif did == _HDR_DHR:
                header["dhr"] = v if v is not None else 0.0
            elif did == _HDR_ELV:
                header["elv"] = v
            elif did == _HDR_TYP:
                header["typ"] = v
            elif did == _HDR_T29:
                header["t29"] = v
                header_done = True
            # Once we hit the first CAT, header is done
            if did == _OBS_CAT:
                header_done = True
                break
            if header_done:
                break

        lat = header["yob"]
        lon = header["xob"]
        if lat is None or lon is None:
            return rows
        if lat < -90.0 or lat > 90.0:
            return rows

        # Compute observation time = base_time + DHR (hours)
        try:
            dhr_hours = float(header["dhr"])
            obs_time = base_time + timedelta(hours=dhr_hours)
        except (ValueError, OverflowError, TypeError):
            obs_time = base_time

        # Time filter
        if obs_time < dt_min or obs_time > dt_max:
            return rows

        # Normalize longitude to [0, 360)
        lon_360 = float(lon) % 360.0

        # Build requested mnemonic IDs set
        needed_ids: dict[str, int] = {}
        for var_name, (bufr_key, _) in var_plan.items():
            desc_id = _MNEMONIC_TO_DESCR.get(bufr_key)
            if desc_id is not None:
                needed_ids[var_name] = desc_id

        # Wind variables need special handling (u/v from UOB/VOB or DDO/FFO)
        need_wind = any(
            bufr_key.startswith("wind") for _, (bufr_key, _) in var_plan.items()
        )

        # ── Pass 2: walk observation levels ──
        # PrepBUFR repeats CAT blocks; within each CAT block the levels
        # are delimited by a pressure observation (POB) descriptor.
        current_level: dict[int, Any] = {}
        in_obs = False

        for i in range(len(descs)):
            did = descs[i].id
            val = vals[i]

            # Start of a new observation level: POB
            if did == _OBS_POB:
                # Flush previous level
                if in_obs and current_level:
                    _emit_rows(
                        rows,
                        current_level,
                        header,
                        obs_time,
                        obs_class_str,
                        lat,
                        lon_360,
                        var_plan,
                        needed_ids,
                        need_wind,
                    )
                current_level = {_OBS_POB: val}
                in_obs = True
            elif in_obs and did in _OBSERVATION_DESCR_IDS:
                # Only store first occurrence per descriptor per level
                if did not in current_level:
                    current_level[did] = val

        # Flush last level
        if in_obs and current_level:
            _emit_rows(
                rows,
                current_level,
                header,
                obs_time,
                obs_class_str,
                lat,
                lon_360,
                var_plan,
                needed_ids,
                need_wind,
            )

        return rows

    @staticmethod
    def _build_extraction_plan(
        variables: list[str],
    ) -> dict[str, tuple[str, Callable[..., pd.DataFrame]]]:
        """Build extraction plan mapping variable names to PrepBUFR info.

        Parameters
        ----------
        variables : list[str]
            Earth2Studio variable names.

        Returns
        -------
        dict[str, tuple[str, Callable]]
            Map of var_name -> (prepbufr_mnemonic, modifier_function).
        """
        plan: dict[str, tuple[str, Callable[..., pd.DataFrame]]] = {}
        for var in variables:
            bufr_key, modifier = GDASObsConvLexicon.get_item(var)
            plan[var] = (bufr_key, modifier)
        return plan

    def _empty_schema_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema types.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame matching ``self.SCHEMA``.
        """
        return pd.DataFrame(
            {
                name: pd.Series(dtype=self._pa_to_pandas_dtype(field.type))
                for name, field in zip(self.SCHEMA.names, self.SCHEMA)
            }
        )

    @staticmethod
    def _pa_to_pandas_dtype(pa_type: pa.DataType) -> object:
        """Convert PyArrow type to pandas/numpy dtype.

        Parameters
        ----------
        pa_type : pa.DataType
            PyArrow data type.

        Returns
        -------
        object
            Corresponding pandas/numpy dtype.
        """
        if pa.types.is_timestamp(pa_type):
            return "datetime64[ns]"
        elif pa.types.is_float32(pa_type):
            return np.float32
        elif pa.types.is_uint16(pa_type):
            return np.uint16
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return object
        else:
            return object

    @staticmethod
    def _build_url(cycle: datetime) -> str:
        """Build the NOMADS URL for a PrepBUFR cycle.

        Parameters
        ----------
        cycle : datetime
            Cycle datetime (must be 00/06/12/18z).

        Returns
        -------
        str
            Full URL to the PrepBUFR .nr file.
        """
        date_str = cycle.strftime("%Y%m%d")
        hour_str = f"{cycle.hour:02d}"
        return f"{NOMADS_BASE_URL}/gdas.{date_str}/gdas.t{hour_str}z.prepbufr.nr"

    def _cache_path(self, url: str) -> str:
        """Compute deterministic cache path for a URL.

        Parameters
        ----------
        url : str
            URL to compute cache path for.

        Returns
        -------
        str
            Local file path for cached data.
        """
        sha = hashlib.sha256(url.encode()).hexdigest()[:16]
        return os.path.join(self.cache, f"prepbufr_{sha}.bin")

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gdas_prepbufr")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gdas_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if date time is available on NOMADS.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            True if the time falls within the NOMADS retention window.
        """
        if isinstance(time, np.datetime64):
            _time = timearray_to_datetime([time])[0]
        else:
            _time = time

        try:
            cls._validate_time([_time])
        except ValueError:
            return False
        return True

    @classmethod
    def resolve_fields(
        cls,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> list[str] | None:
        """Resolve and validate the requested output fields.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output. None returns all fields.

        Returns
        -------
        list[str] | None
            Validated list of field names, or None for all fields.

        Raises
        ------
        KeyError
            If a field is not in the schema.
        TypeError
            If fields is an unsupported type.
        """
        if fields is None:
            return None
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            fields = fields.names

        schema_names = set(cls.SCHEMA.names)
        for f in fields:
            if f not in schema_names:
                raise KeyError(
                    f"Field '{f}' not in {cls.__name__} SCHEMA. "
                    f"Available: {cls.SCHEMA.names}"
                )
        return list(fields)


# ── Module-level helper functions for PrepBUFR decoding ──────────────


# Mapping from lexicon mnemonic strings to PrepBUFR descriptor IDs
_MNEMONIC_TO_DESCR: dict[str, int] = {
    "TOB": _OBS_TOB,  # Temperature (DEG C)
    "QOB": _OBS_QOB,  # Specific humidity (MG/KG)
    "POB": _OBS_POB,  # Pressure (MB)
    "PWO": _OBS_PWO,  # Precipitable water (KG/M**2)
    "ZOB": _OBS_ZOB,  # Height (m)
    "PMO": _OBS_PMO,  # Mean sea level pressure (MB)
    "TDO": _OBS_TDO,  # Dewpoint temperature (DEG C)
    "DDO": _OBS_DDO,  # Wind direction (DEGREES TRUE)
    "FFO": _OBS_FFO,  # Wind speed (KNOTS)
    "UOB": _OBS_UOB,  # U-wind (M/S)
    "VOB": _OBS_VOB,  # V-wind (M/S)
}


# ── Process-pool worker functions for parallel decode ────────────────

# Module-level decoder for worker processes, set by _init_decode_worker.
_worker_decoder: Any = None


def _init_decode_worker(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Initializer for process pool workers.

    Registers NCEP-local descriptor tables with pybufrkit in each
    worker process and creates a reusable decoder instance stored
    as a module-level global.
    """
    global _worker_decoder  # noqa: PLW0603
    TableGroupCacheManager.clear_extra_entries()
    TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
    if table_b or table_d:
        TableGroupCacheManager.add_extra_entries(table_b, table_d)
    _worker_decoder = BufrDecoder()


def _decode_message_worker(
    msg_bytes: bytes,
    obs_class_str: str,
    variables: list[str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single BUFR message in a worker process.

    Uses the decoder created by :func:`_init_decode_worker`.

    Parameters
    ----------
    msg_bytes : bytes
        Raw BUFR message bytes.
    obs_class_str : str
        Observation class string (e.g. ``"ADPSFC"``).
    variables : list[str]
        Earth2Studio variable names to extract.
    dt_min : datetime
        Minimum observation time.
    dt_max : datetime
        Maximum observation time.

    Returns
    -------
    list[dict]
        Observation rows for this message.
    """
    var_plan = NomadsGDASObsConv._build_extraction_plan(variables)
    return _decode_message(
        _worker_decoder, msg_bytes, obs_class_str, var_plan, dt_min, dt_max
    )


def _decode_message(
    decoder: Any,
    msg_bytes: bytes,
    obs_class_str: str,
    var_plan: dict[str, tuple[str, Callable[..., pd.DataFrame]]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single BUFR message and extract observation rows.

    Parameters
    ----------
    decoder : pybufrkit.decoder.Decoder
        Configured pybufrkit decoder.
    msg_bytes : bytes
        Raw BUFR message bytes.
    obs_class_str : str
        Observation class string (e.g. ``"ADPSFC"``).
    var_plan : dict
        Variable extraction plan.
    dt_min : datetime
        Minimum observation time.
    dt_max : datetime
        Maximum observation time.

    Returns
    -------
    list[dict]
        Observation rows for this message.
    """
    try:
        msg = decoder.process(msg_bytes)
    except Exception:
        return []

    n_subsets = msg.n_subsets.value
    if n_subsets == 0:
        return []

    td = msg.template_data.value
    ddas = td.decoded_descriptors_all_subsets
    dvas = td.decoded_values_all_subsets

    # Message-level date from section 1.
    # BUFR edition 3 stores a 2-digit year (year of century),
    # so we must expand it to a 4-digit year.
    msg_year = msg.year.value
    if msg_year < 100:
        msg_year += 2000 if msg_year < 70 else 1900

    try:
        base_time = datetime(
            msg_year,
            msg.month.value,
            msg.day.value,
            msg.hour.value,
            msg.minute.value,
        )
    except (ValueError, OverflowError):
        return []

    rows: list[dict[str, Any]] = []
    for subset_idx in range(n_subsets):
        rows.extend(
            NomadsGDASObsConv._extract_subset(
                ddas[subset_idx],
                dvas[subset_idx],
                base_time,
                obs_class_str,
                var_plan,
                dt_min,
                dt_max,
            )
        )
    return rows


def _emit_rows(
    rows: list[dict[str, Any]],
    level: dict[int, Any],
    header: dict[str, Any],
    obs_time: datetime,
    obs_class_str: str,
    lat: float,
    lon_360: float,
    var_plan: dict[str, tuple[str, Callable[..., pd.DataFrame]]],
    needed_ids: dict[str, int],
    need_wind: bool,
) -> None:
    """Emit observation rows for a single pressure level.

    Mutates ``rows`` in place by appending one dict per requested variable
    that has a valid (non-None) observation at this level.

    Parameters
    ----------
    rows : list[dict]
        Accumulator list to append rows to.
    level : dict[int, Any]
        Descriptor-ID -> value mapping for the current level.
    header : dict
        Subset header fields (sid, xob, yob, dhr, elv, typ, t29).
    obs_time : datetime
        Observation time for this subset.
    obs_class_str : str
        PREPBUFR observation class string.
    lat : float
        Latitude.
    lon_360 : float
        Longitude in [0, 360).
    var_plan : dict
        Variable extraction plan.
    needed_ids : dict[str, int]
        Map of variable name to descriptor ID for non-wind variables.
    need_wind : bool
        Whether any wind variables are requested.
    """
    # Pressure for this level (MB -> Pa: multiply by 100)
    pob_val = level.get(_OBS_POB)
    pres_pa = np.float32(pob_val * 100.0) if pob_val is not None else None

    # Quality mark: use the first available quality mark in the level
    quality_val = None
    for qm_id in (_OBS_PQM, _OBS_TQM, _OBS_WQM, _OBS_QQM):
        qv = level.get(qm_id)
        if qv is not None:
            quality_val = np.uint16(int(qv))
            break

    # Common row template
    base_row: dict[str, Any] = {
        "time": obs_time,
        "lat": np.float32(lat),
        "lon": np.float32(lon_360),
        "pres": pres_pa,
        "elev": None,
        "type": np.uint16(int(header["typ"])) if header["typ"] is not None else None,
        "class": obs_class_str if obs_class_str else None,
        "station": header["sid"] if header["sid"] else None,
        "station_elev": (
            np.float32(header["elv"]) if header["elv"] is not None else None
        ),
        "quality": quality_val,
    }

    # Non-wind variables
    for var_name, desc_id in needed_ids.items():
        val = level.get(desc_id)
        if val is None:
            continue

        row = base_row.copy()
        row["variable"] = var_name
        row["observation"] = np.float32(val)
        rows.append(row)

    # Wind variables (u/v from UOB/VOB)
    if need_wind:
        uob = level.get(_OBS_UOB)
        vob = level.get(_OBS_VOB)

        if uob is not None and vob is not None:
            for var_name, (bufr_key, _) in var_plan.items():
                if bufr_key == "wind::u":
                    row = base_row.copy()
                    row["variable"] = var_name
                    row["observation"] = np.float32(uob)
                    rows.append(row)
                elif bufr_key == "wind::v":
                    row = base_row.copy()
                    row["variable"] = var_name
                    row["observation"] = np.float32(vob)
                    rows.append(row)


def _extract_dx_tables(
    flat: list[Any],
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Extract NCEP Table B and D entries from a DX message value list.

    DX table messages (dataCategory=11) contain embedded NCEP-local BUFR
    descriptor definitions.  The flat decoded values follow the layout
    produced by the NCEP BUFRLIB DX table encoding:

    - n_table_a, [table_a entries (3 fields each)…]
    - n_table_b, [table_b entries (11 fields each)…]
    - n_table_d, [table_d entries (variable-length)…]

    Each Table B entry has 11 fields:
        F, X, Y, mnemonic(32), desc_cont(32), unit(24),
        sign_scale(1), scale(3), sign_ref(1), reference(10), width(3)

    Each Table D entry has:
        F, X, Y, mnemonic(32), desc_cont(32), n_members,
        [member F, X, Y, …]

    Parameters
    ----------
    flat : list
        Decoded values from a single DX message subset.
    table_b : dict
        Accumulator dict to update with Table B entries.
    table_d : dict
        Accumulator dict to update with Table D entries.
    """
    idx = 0
    n = len(flat)

    # Helper: safely read string
    def _str(v: Any) -> str:
        if isinstance(v, bytes):
            return v.decode("ascii", errors="replace").strip()
        if v is None:
            return ""
        return str(v).strip()

    def _safe_int(v: Any) -> int:
        """Convert value to int, handling bytes and string types."""
        if isinstance(v, (int, float)):
            return int(v)
        s = _str(v)
        if not s:
            return 0
        try:
            return int(s)
        except ValueError:
            return 0

    def _fxy_to_id(f: Any, x: Any, y: Any) -> int:
        """Convert F, X, Y fields to an integer descriptor ID."""
        return _safe_int(f) * 100000 + _safe_int(x) * 1000 + _safe_int(y)

    # ── Table A section ──
    if idx >= n:
        return
    n_a = _safe_int(flat[idx])
    idx += 1
    # Table A entries: 3 fields each (type_number, desc_part1, desc_part2)
    idx += n_a * 3
    if idx >= n:
        return

    # ── Table B section ──
    n_b = _safe_int(flat[idx])
    idx += 1

    for _ in range(n_b):
        if idx + 10 >= n:
            return
        f_val = flat[idx]
        x_val = flat[idx + 1]
        y_val = flat[idx + 2]
        mnemonic = _str(flat[idx + 3])
        # flat[idx + 4] is description continuation
        unit = _str(flat[idx + 5])
        sign_scale = _str(flat[idx + 6])
        scale_s = _str(flat[idx + 7])
        sign_ref = _str(flat[idx + 8])
        ref_s = _str(flat[idx + 9])
        width_s = _str(flat[idx + 10])
        idx += 11

        desc_id = _fxy_to_id(f_val, x_val, y_val)
        if desc_id == 0:
            continue

        scale = _safe_int(scale_s)
        if sign_scale == "-":
            scale = -scale
        reference = _safe_int(ref_s)
        if sign_ref == "-":
            reference = -reference
        width = _safe_int(width_s)

        # Build pybufrkit Table B entry tuple:
        # (name, unit, scale, refval, nbits, crex_unit, crex_scale, crex_nchars)
        entry = (
            mnemonic,
            unit,
            scale,
            reference,
            width,
            unit,  # crex_unit = same as unit
            scale,  # crex_scale = same as scale
            max(1, (width + 3) // 4),  # crex_nchars approximation
        )
        table_b[desc_id] = entry

    # ── Table D section ──
    if idx >= n:
        return
    n_d = _safe_int(flat[idx])
    idx += 1

    for _ in range(n_d):
        if idx + 3 >= n:
            return
        # Table D header: F, X, Y, mnemonic(64)
        f_val = flat[idx]
        x_val = flat[idx + 1]
        y_val = flat[idx + 2]
        seq_mnemonic = _str(flat[idx + 3])
        idx += 4

        seq_id = _fxy_to_id(f_val, x_val, y_val)
        if seq_id == 0:
            continue

        # Number of member descriptors
        if idx >= n:
            return
        n_members = _safe_int(flat[idx])
        idx += 1

        members: list[str] = []
        for _ in range(n_members):
            if idx >= n:
                break
            member_fxy = _str(flat[idx])
            idx += 1
            members.append(member_fxy)

        if members:
            table_d[seq_id] = (seq_mnemonic, members)

        # Build pybufrkit Table D entry: (name, [member_descriptor_ids])
        table_d[seq_id] = (seq_mnemonic, members)
