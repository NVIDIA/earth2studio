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
import shutil
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_dir,
    gather_with_concurrency,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    NCEPObsTask,
    compile_dataframe,
    cycle_windows,
    decode_gpsro,
    decode_prepbufr,
    plan_conv_tasks,
)
from earth2studio.lexicon.gdas import GDASObsConvLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import normalize_time_tolerance, timearray_to_datetime
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NOMADS_STORE_URL = "https://nomads.ncep.noaa.gov"
NOMADS_PREFIX = "pub/data/nccf/com/obsproc/prod"

# Maximum age of data on NOMADS (approximate – production dir retains ~2 days)
_MAX_AGE_DAYS = 2


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
        self._tmp_cache_hash: str | None = uuid.uuid4().hex[:8] if not cache else None
        self._store = obstore_store_from_url(
            NOMADS_STORE_URL, anonymous=False, max_pool_connections=max_workers
        )

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
        time_list, variable_list = prep_data_inputs(time, variable)
        output_fields = self.resolve_fields(fields)

        self._validate_time(time_list)

        # Task construction unions requested windows by cycle and file route.
        tasks = self._create_tasks(time_list, variable_list)

        coros = [
            async_retry(
                self._fetch_remote_file,
                t.uri,
                retries=self._retries,
                backoff=1.0,
                task_timeout=120.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for t in tasks
        ]
        await gather_with_concurrency(
            coros,
            max_workers=self._max_workers,
            desc="Fetching GDAS conventional observations",
            verbose=(not self._verbose),
        )

        # Decode and compile
        df = compile_dataframe(
            tasks,
            self.SCHEMA,
            self.SOURCE_ID,
            self.local_path,
            self._decode_file,
        )

        # Select output columns
        if output_fields is not None:
            df = df[[f for f in output_fields if f in df.columns]]

        df.attrs["source"] = self.SOURCE_ID
        return df

    def _create_tasks(
        self,
        times: list[datetime],
        variables: list[str],
    ) -> list[NCEPObsTask]:
        """Build download tasks for required 6h PrepBUFR cycles."""
        return plan_conv_tasks(
            cycle_windows(times, self._tolerance_lower, self._tolerance_upper),
            variables,
            GDASObsConvLexicon,
            self._build_uri,
        )

    def _build_uri(self, route: str, cycle: datetime) -> str:
        """Build the NOMADS object key for a cycle file on the given route."""
        date_str = cycle.strftime("%Y%m%d")
        hour_str = f"{cycle.hour:02d}"
        if route == "gpsro":
            return (
                f"{NOMADS_PREFIX}/gdas.{date_str}/"
                f"gdas.t{hour_str}z.gpsro.tm00.bufr_d.nr"
            )
        return f"{NOMADS_PREFIX}/gdas.{date_str}/gdas.t{hour_str}z.prepbufr.nr"

    # ------------------------------------------------------------------
    # File fetch (obstore HTTP)
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, uri: str) -> str:
        """Download a BUFR file from NOMADS via obstore."""
        return await obstore_fetch_to_cache(
            self._store,
            uri,
            self.cache,
            cache_key=hashlib.sha256(uri.encode()).hexdigest(),
        )

    # ------------------------------------------------------------------
    # File decode (dispatch by task route)
    # ------------------------------------------------------------------
    def _decode_file(self, local_path: str, task: NCEPObsTask) -> pd.DataFrame:
        """Decode a cached BUFR file dispatching by route."""
        if task.route == "gpsro":
            frame = decode_gpsro(
                local_path,
                task.var_plan,
                task.datetime_min,
                task.datetime_max,
                decode_workers=self._decode_workers,
            )
            return frame[self.SCHEMA.names]
        frame = decode_prepbufr(
            local_path,
            task.var_plan,
            task.datetime_min,
            task.datetime_max,
            decode_workers=self._decode_workers,
        )
        return frame[self.SCHEMA.names]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate requested times against NOMADS availability."""
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

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for a URI."""
        sha = hashlib.sha256(uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        return datasource_cache_dir("gdas_prepbufr", self._cache, self._tmp_cache_hash)

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
