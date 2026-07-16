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

# NOAA-NASA Joint Archive (NNJA) of Observations for Earth System Reanalysis.
#
# Reference: https://psl.noaa.gov/data/nnja_obs/
# Public S3 bucket: s3://noaa-reanalyses-pds/observations/reanalysis/


from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import s3fs  # type: ignore[import-untyped]
from loguru import logger

from earth2studio.data.utils import (
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    NCEPObsRequestMixin,
    _NCEPConvTask,
    _NCEPGpsroAdapter,
    _NCEPObsTask,
    _NCEPPrepbufrAdapter,
    _plan_conv_tasks,
    map_aircraft_profile_types,
)
from earth2studio.lexicon import NNJAObsConvLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import TimeTolerance

NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


class NNJAObsMixin:
    """NNJA archive mechanics shared across NNJA observation products.

    Supplies the anonymous-S3 session with bounded concurrency and retries,
    deterministic raw-file caching, temporary-cache cleanup, six-hour cycle
    validation, availability, and the missing-file policy. Compose with
    :class:`NCEPObsRequestMixin`. The concrete source supplies ``MIN_DATE``.
    """

    MIN_DATE: datetime
    _verbose: bool  # set by NCEPObsRequestMixin._init_request

    def _handle_missing_file(self, path: str) -> None:
        """Warn instead of raising on missing NNJA cycle files.

        NNJA does not guarantee every cycle/sub-archive combination exists
        (e.g. the ``gps/gpsro/`` archive only goes back to the early 2000s, and
        individual cycles can be absent). Returning a partial DataFrame is more
        useful than aborting a multi-cycle request because of one missing file.
        """
        logger.warning(f"NNJA file {path} not found, skipping")

    def _init_transport(
        self,
        *,
        cache: bool,
        async_workers: int,
        retries: int,
    ) -> None:
        """Initialize NNJA transport state; call from the concrete constructor."""
        self._cache = cache
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self.fs: s3fs.S3FileSystem | None = None

    async def fetch_files(self, uris: Sequence[str]) -> None:
        """Download remote files into the NNJA cache."""
        if self.fs is None:
            self.fs = s3fs.S3FileSystem(
                anon=True,
                client_kwargs={},
                asynchronous=True,
                skip_instance_cache=True,
            )

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        async with managed_session(self.fs):
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
                max_workers=self._async_workers,
                desc="Fetching NNJA files",
                verbose=(not self._verbose),
            )

    async def _fetch_remote_file(self, path: str) -> None:
        """Download a single remote file into the cache directory."""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self.local_path(path)
        if pathlib.Path(cache_path).is_file():
            return
        try:
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as fh:
                fh.write(data)
        except FileNotFoundError:
            self._handle_missing_file(path)

    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for NNJA observation files."""
        cache_location = os.path.join(datasource_cache_root(), "nnja")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nnja_{self._tmp_cache_hash}"
            )
        return cache_location

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that times align to a 6-hour cycle and are in range."""
        for t in times:
            if t.minute != 0 or t.second != 0 or t.microsecond != 0:
                raise ValueError(
                    f"Requested datetime {t} must be on a whole hour "
                    f"(NNJA cycles are 6-hourly)."
                )
            if t.hour % 6 != 0:
                raise ValueError(
                    f"Requested datetime {t} must align to a 6-hour cycle "
                    f"(00, 06, 12, 18z)."
                )
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than {cls.__name__}.MIN_DATE "
                    f"({cls.MIN_DATE.isoformat()})."
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NNJAObsConv(NNJAObsMixin, NCEPObsRequestMixin):
    """NNJA conventional (in-situ + GPS RO) observational data source. NOAA-NASA Joint
    Archive (NNJA) of Observations for Earth System Reanalysis is an archive ideal for
    developing observation-driven weather forecasting tools, as it includes a wide
    cross-section of data from a plethora of sensing platforms (satellites, surface
    stations, weather balloons, and more) and features data from 1979 to the present.

    GPSRO rows use the shared columns with product-specific meanings:
    ``type`` is receiver ``SAID``, ``station`` combines receiver/transmitter
    identifiers, ``quality`` is the QFRO flag table, ``pres`` is null, and
    ``elev`` is impact parameter minus Earth radius of curvature.

    Parameters
    ----------
    source : {"prepbufr", "prepbufr.acft_profiles"}, optional
        Which encoding family of the NNJA conventional archive to read,
        by default ``"prepbufr"``. These sources are different stages of the
        NCEP observation-processing pipeline, not independent replacement
        datasets:

        - ``"convbufr"`` points at raw dump streams grouped by family, such as
          ``aircft``/``aircar``/``adpupa``/``adpsfc``. These files preserve
          source-native schemas and require family-specific decoding and QC
          before they resemble GSI-ready observations. The generic PrepBUFR
          decoder does not implement those raw family schemas, so this source
          raises ``NotImplementedError``.
        - ``"prepbufr"`` points at the merged PrepBUFR cycle file. This is the
          preferred source for GSI-like conventional observations because
          upstream obsproc has already merged dump families, standardized many
          mnemonics, and attached report types / quality marks.
        - ``"prepbufr.acft_profiles"`` points at an aircraft-only PrepBUFR
          profile product. It groups aircraft points into flight-level,
          ascending, and descending profile report types that GSI remaps back
          to ordinary aircraft report types during processing.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple ``(lower, upper)`` for
        asymmetric windows, by default ``np.timedelta64(0, 'm')``.
    cache : bool, optional
        Cache downloaded files in the local filesystem cache, by default True.
    verbose : bool, optional
        Show progress bars, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 24.
    decode_workers : int, optional
        Number of parallel processes for BUFR message decoding. Higher values
        speed up decoding of large PrepBUFR files at the cost of more memory.
        Set to 1 to disable multiprocessing, by default 8.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Requested times must align to a 6-hour cycle (00, 06, 12, 18z); the time
    tolerance brackets the cycle when selecting observations.

    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-pds/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm

    Badges
    ------
    region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAObsConv"
    SCHEMA = NCEP_CONVENTIONAL_PUBLIC_SCHEMA
    LEXICON = NNJAObsConvLexicon
    MIN_DATE = datetime(1979, 1, 1)

    VALID_SOURCES = frozenset(["prepbufr", "prepbufr.acft_profiles"])

    def __init__(
        self,
        source: str = "prepbufr",
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 24,
        decode_workers: int = 8,
        retries: int = 3,
    ) -> None:
        if source == "convbufr":
            raise NotImplementedError(
                "NNJAObsConv(source='convbufr') targets raw dump streams grouped "
                "by family (aircft/aircar/adpupa/adpsfc) with source-native "
                "schemas that require family-specific decoding and QC before they "
                "resemble GSI-ready observations; the generic PrepBUFR decoder "
                "does not implement those raw family schemas"
            )
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Valid sources: {sorted(self.VALID_SOURCES)}"
            )
        self._source = source
        # Internal switch for the special aircraft-profile product. Default
        # output maps profile-stage 33x/43x/53x report codes to the standard
        # GSI/PREPBUFR aircraft 23x codes in ``type``
        self._map_acft_profile_report_types = True
        self._init_request(
            time_tolerance=time_tolerance,
            verbose=verbose,
            async_timeout=async_timeout,
            decode_workers=decode_workers,
        )
        self._init_transport(
            cache=cache,
            async_workers=async_workers,
            retries=retries,
        )
        self._prepbufr_adapter = _NCEPPrepbufrAdapter(self._decode_workers)
        self._gpsro_adapter = _NCEPGpsroAdapter(self._decode_workers)

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_NCEPObsTask]:
        return _plan_conv_tasks(
            self._cycle_windows(time_list), variable, self.LEXICON, self._build_uri
        )

    def _build_uri(self, route: str, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a cycle file on the given route."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        if route == "prepbufr":
            archive_dir = self._source
            if self._source == "prepbufr.acft_profiles":
                archive_dir = "bufr"
            return (
                f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/conv/{self._source}/"
                f"{year_key}/{month_key}/{archive_dir}/"
                f"gdas.{date_key}.t{hour_key}z.{self._source}.nr"
            )
        if route == "gpsro":
            return (
                f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/gps/gpsro/"
                f"{year_key}/{month_key}/bufr/"
                f"gdas.{date_key}.t{hour_key}z.gpsro.tm00.bufr_d"
            )
        raise ValueError(f"Unsupported route '{route}'")

    # ------------------------------------------------------------------
    # File decode (dispatch by task route)
    # ------------------------------------------------------------------
    def _decode_file(self, local_path: str, task: _NCEPObsTask) -> pd.DataFrame:
        if not isinstance(task, _NCEPConvTask):
            raise TypeError(f"Expected a conventional task, got {type(task).__name__}")
        if task.route == "gpsro":
            frame = self._gpsro_adapter.decode_file(
                local_path, task.var_plan, task.datetime_min, task.datetime_max
            )
            return frame[self.SCHEMA.names]
        if task.route == "prepbufr":
            frame = self._prepbufr_adapter.decode_file(
                local_path, task.var_plan, task.datetime_min, task.datetime_max
            )
            if (
                self._source == "prepbufr.acft_profiles"
                and self._map_acft_profile_report_types
            ):
                frame = map_aircraft_profile_types(frame)
            return frame[self.SCHEMA.names]
        raise ValueError(f"Unsupported route '{task.route}'")
