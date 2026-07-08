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

import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    prep_data_inputs,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    GPSRO_BNDA,
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    _empty_dataframe,
    _NCEPGpsroAdapter,
    _NCEPPrepbufrAdapter,
)
from earth2studio.lexicon.gdas import GDASObsConvLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import normalize_time_tolerance, timearray_to_datetime
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NOMADS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"

# Maximum age of data on NOMADS (approximate – production dir retains ~2 days)
_MAX_AGE_DAYS = 2


@dataclass
class _GDASAsyncTask:
    """Async task for fetching one conventional cycle file."""

    url: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    variables: list[str]
    route: Literal["prepbufr", "gpsro"]


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NomadsGDASObsConv:
    """Real-time GDAS conventional observations from NOAA NOMADS.

    Provides near-real-time access to quality-controlled conventional
    (in-situ) observations from the NOAA Global Data Assimilation System
    (GDAS). Data is sourced from merged PrepBUFR and separate GPSRO files on
    NOMADS, updated 4 times daily (00z, 06z, 12z, 18z) with approximately
    6-10 hours latency.

    Observation types include radiosondes (ADPUPA), surface stations (ADPSFC),
    aircraft (AIRCAR/AIRCFT), ships and buoys (SFCSHP), wind profilers
    (PROFLR), satellite-derived winds (SATWND), and GPS precipitable water
    (GPSIPW). The ``gps`` variable reads only the combined ionosphere-corrected
    bending-angle observation from the separate GPSRO dump.

    GPSRO rows use the shared columns with product-specific meanings:
    ``type`` is receiver ``SAID``, ``station`` combines receiver/transmitter
    identifiers, ``quality`` is the QFRO flag table, ``pres`` is null, and
    ``elev`` is impact parameter minus Earth radius of curvature.

    The output schema matches :class:`UFSObsConv` with additional PrepBUFR
    ``quality``, ``pressure_quality``, and ``level_cat`` metadata.

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
        Cache downloaded observation files locally, by default True.
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
    PrepBUFR file is approximately 60-100 MB; GPSRO is downloaded only when
    ``gps`` is requested.

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

    SCHEMA: pa.Schema = NCEP_CONVENTIONAL_PUBLIC_SCHEMA

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
        self.fs: Any = None
        self._prepbufr_adapter = _NCEPPrepbufrAdapter(self._decode_workers)
        self._gpsro_adapter = _NCEPGpsroAdapter(self._decode_workers)

    async def _async_init(self) -> None:
        """Initialize async HTTP filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        from fsspec.implementations.http import (  # type: ignore[import-untyped]
            HTTPFileSystem,
        )

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
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
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
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        output_fields = self.resolve_fields(fields)

        self._validate_time(time_list)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Task construction unions requested windows by cycle and file route.
        fetch_tasks = self._create_tasks(time_list, variable_list)

        coros = [self._fetch_wrapper(t.url) for t in fetch_tasks]
        await gather_with_concurrency(
            coros,
            max_workers=self._max_workers,
            desc="Fetching GDAS conventional observations",
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

        prepbufr_variables: list[str] = []
        gpsro_variables: list[str] = []
        for variable in variables:
            source_key, _modifier = GDASObsConvLexicon.get_item(variable)
            if source_key.startswith("gpsro::"):
                gpsro_variables.append(variable)
            else:
                prepbufr_variables.append(variable)

        tasks: dict[tuple[datetime, str], _GDASAsyncTask] = {}

        def add_task(
            cycle: datetime,
            route: Literal["prepbufr", "gpsro"],
            dt_min: datetime,
            dt_max: datetime,
            route_variables: list[str],
        ) -> None:
            key = (cycle, route)
            existing = tasks.get(key)
            if existing is not None:
                existing.datetime_min = min(existing.datetime_min, dt_min)
                existing.datetime_max = max(existing.datetime_max, dt_max)
                return
            url = self._build_url(cycle)
            if route == "gpsro":
                url = self._build_gpsro_url(cycle)
            tasks[key] = _GDASAsyncTask(
                url=url,
                datetime_file=cycle,
                datetime_min=dt_min,
                datetime_max=dt_max,
                variables=route_variables,
                route=route,
            )

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
                if prepbufr_variables:
                    add_task(cycle, "prepbufr", dt_min, dt_max, prepbufr_variables)
                if gpsro_variables:
                    add_task(cycle, "gpsro", dt_min, dt_max, gpsro_variables)
                cycle += timedelta(hours=6)

        return list(tasks.values())

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
                if task.route == "gpsro":
                    df = self._decode_gpsro(
                        cache_path,
                        task.variables,
                        task.datetime_min,
                        task.datetime_max,
                    )
                else:
                    df = self._decode_prepbufr(
                        cache_path,
                        task.variables,
                        task.datetime_min,
                        task.datetime_max,
                    )
                if not df.empty:
                    all_frames.append(df.reindex(columns=self.SCHEMA.names))
            except Exception as e:
                logger.warning(f"Error decoding {cache_path}: {e}, skipping")

        if not all_frames:
            return self._empty_schema_df()

        df = pd.concat(all_frames, ignore_index=True)
        return df

    def _decode_prepbufr(
        self,
        path: str,
        variables: list[str],
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        """Decode locally cached PrepBUFR bytes through the shared adapter."""
        frame = self._prepbufr_adapter.decode_file(
            path,
            self._build_extraction_plan(variables),
            dt_min,
            dt_max,
        )
        return frame[self.SCHEMA.names]

    def _decode_gpsro(
        self,
        path: str,
        variables: list[str],
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        """Decode locally cached GPSRO bytes through the shared adapter."""
        frame = self._gpsro_adapter.decode_file(
            path,
            self._build_gpsro_plan(variables),
            dt_min,
            dt_max,
        )
        return frame[self.SCHEMA.names]

    @staticmethod
    def _build_extraction_plan(
        variables: list[str],
    ) -> dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
        """Map public variable names to PrepBUFR extraction keys.

        Parameters
        ----------
        variables : list[str]
            Earth2Studio variable names.

        Returns
        -------
        dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]
            Map of variable name to (PrepBUFR mnemonic or wind key, modifier).
        """
        plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {}
        for var in variables:
            bufr_key, modifier = GDASObsConvLexicon.get_item(var)
            if bufr_key.startswith("gpsro::"):
                raise ValueError(f"Variable '{var}' is not a PrepBUFR variable")
            plan[var] = (bufr_key, modifier)
        return plan

    @staticmethod
    def _build_gpsro_plan(
        variables: list[str],
    ) -> dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]]:
        plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = {}
        for variable in variables:
            source_key, modifier = GDASObsConvLexicon.get_item(variable)
            route, separator, descriptor = source_key.partition("::")
            if not separator or route != "gpsro":
                raise ValueError(f"Variable '{variable}' is not a GPSRO variable")
            descriptor_id = int(descriptor)
            if descriptor_id != GPSRO_BNDA:
                raise ValueError(f"Unsupported GPSRO descriptor {descriptor_id}")
            plan[variable] = (descriptor_id, modifier)
        return plan

    def _empty_schema_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema types.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame matching ``self.SCHEMA``.
        """
        return _empty_dataframe(self.SCHEMA)

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

    @staticmethod
    def _build_gpsro_url(cycle: datetime) -> str:
        """Build the NOMADS URL for the matching GPSRO cycle dump."""
        date_str = cycle.strftime("%Y%m%d")
        hour_str = f"{cycle.hour:02d}"
        return (
            f"{NOMADS_BASE_URL}/gdas.{date_str}/gdas.t{hour_str}z.gpsro.tm00.bufr_d.nr"
        )

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
