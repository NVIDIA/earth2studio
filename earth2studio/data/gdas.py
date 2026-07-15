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
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.ncep_obs import (
    _NCEPGpsroPlan,
    _NCEPObsSourceBase,
    _NCEPObsTask,
    _NCEPPrepbufrPlan,
)
from earth2studio.data.utils import (
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    _NCEPGpsroAdapter,
    _NCEPPrepbufrAdapter,
)
from earth2studio.lexicon.gdas import GDASObsConvLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import TimeTolerance

NOMADS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"

# Maximum age of data on NOMADS (approximate – production dir retains ~2 days)
_MAX_AGE_DAYS = 2


class _NomadsObsStore:
    """NOMADS HTTP transport and raw-file cache."""

    def __init__(
        self,
        cache: bool,
        verbose: bool,
        max_workers: int,
        retries: int,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self.fs: Any = None

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

    async def fetch_files(self, uris: Sequence[str]) -> None:
        """Download remote files into the NOMADS cache."""
        if self.fs is None:
            await self._async_init()

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        coros = [
            async_retry(
                self._fetch_remote_file,
                uri,
                retries=self._retries,
                backoff=1.0,
                task_timeout=120.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for uri in uris
        ]
        await gather_with_concurrency(
            coros,
            max_workers=self._max_workers,
            desc="Fetching GDAS conventional observations",
            verbose=(not self._verbose),
        )

    async def _fetch_remote_file(self, url: str) -> str:
        """Download one NOMADS file and return its local cache path."""
        cache_path = self.local_path(url)
        if os.path.exists(cache_path):
            logger.debug(f"Cache hit: {cache_path}")
            return cache_path

        data = await self.fs._cat_file(url)
        with open(cache_path, "wb") as f:
            f.write(data)
        logger.debug(f"Downloaded {url} -> {cache_path}")
        return cache_path

    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for a NOMADS URL."""
        sha = hashlib.sha256(uri.encode()).hexdigest()[:16]
        return os.path.join(self.cache, f"prepbufr_{sha}.bin")

    @property
    def cache(self) -> str:
        """Local cache directory for NOMADS observation files."""
        cache_location = os.path.join(datasource_cache_root(), "gdas_prepbufr")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gdas_{self._tmp_cache_hash}"
            )
        return cache_location

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NomadsGDASObsConv(_NCEPObsSourceBase):
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

    SOURCE_ID = "earth2studio.data.NomadsGDASObsConv"
    SCHEMA: pa.Schema = NCEP_CONVENTIONAL_PUBLIC_SCHEMA
    LEXICON = GDASObsConvLexicon
    _store: _NomadsObsStore

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
        super().__init__(
            store=_NomadsObsStore(
                cache=cache,
                verbose=verbose,
                max_workers=max_workers,
                retries=retries,
            ),
            time_tolerance=time_tolerance,
            verbose=verbose,
            async_timeout=async_timeout,
            decode_workers=decode_workers,
        )
        self._prepbufr_adapter = _NCEPPrepbufrAdapter(self._decode_workers)
        self._gpsro_adapter = _NCEPGpsroAdapter(self._decode_workers)

    @property
    def cache(self) -> str:
        """Local cache directory for this data source."""
        return self._store.cache

    def _cache_path(self, url: str) -> str:
        """Compute the deterministic cache path for a NOMADS URL."""
        return self._store.local_path(url)

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

    def _decode_file(self, local_path: str, task: _NCEPObsTask) -> pd.DataFrame:
        """Decode one locally cached file through its route-specific adapter."""
        if task.route == "gpsro":
            return self._decode_gpsro(
                local_path,
                task.gpsro_plan,
                task.datetime_min,
                task.datetime_max,
            )
        return self._decode_prepbufr(
            local_path,
            task.prepbufr_plan,
            task.datetime_min,
            task.datetime_max,
        )

    def _decode_prepbufr(
        self,
        path: str,
        plan: _NCEPPrepbufrPlan,
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        """Decode locally cached PrepBUFR bytes through the shared adapter."""
        frame = self._prepbufr_adapter.decode_file(
            path,
            plan,
            dt_min,
            dt_max,
        )
        return frame[self.SCHEMA.names]

    def _decode_gpsro(
        self,
        path: str,
        plan: _NCEPGpsroPlan,
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        """Decode locally cached GPSRO bytes through the shared adapter."""
        frame = self._gpsro_adapter.decode_file(
            path,
            plan,
            dt_min,
            dt_max,
        )
        return frame[self.SCHEMA.names]

    @staticmethod
    def _build_prepbufr_uri(cycle: datetime) -> str:
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
    def _build_gpsro_uri(cycle: datetime) -> str:
        """Build the NOMADS URL for the matching GPSRO cycle dump."""
        date_str = cycle.strftime("%Y%m%d")
        hour_str = f"{cycle.hour:02d}"
        return (
            f"{NOMADS_BASE_URL}/gdas.{date_str}/gdas.t{hour_str}z.gpsro.tm00.bufr_d.nr"
        )

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
