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
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs  # type: ignore[import-untyped]
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    _empty_dataframe,
    _NCEPGpsroAdapter,
    _NCEPPrepbufrAdapter,
    map_aircraft_profile_types,
)
from earth2studio.lexicon import NNJAObsConvLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"

# ── Async-task dataclasses ──────────────────────────────────────────


@dataclass
class _NNJAConvTask:
    """Async task for a single PrepBUFR cycle file (route ``prepbufr``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


@dataclass
class _NNJAGpsRoTask:
    """Async task for a single gps/gpsro cycle BUFR file (route ``gpsro``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


class _NNJAObsBase:
    """Shared infrastructure for NNJA DataFrame data sources.

    Subclasses must define ``SOURCE_ID``, ``SCHEMA``, ``MIN_DATE``, and
    implement ``_create_tasks(time_list, variable)`` and
    ``_decode_file(local_path, task)``.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema
    MIN_DATE: datetime = datetime(1979, 1, 1)

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 24,
        decode_workers: int = 8,
        retries: int = 3,
    ) -> None:
        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._decode_workers = max(1, decode_workers)
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: s3fs.S3FileSystem | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem."""
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
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
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Cycle timestamps (UTC). Must align to a 6-hour cycle (00, 06,
            12, 18z); the time tolerance is used to bracket the cycle when
            selecting observations.
        variable : str | list[str] | VariableArray
            Variable ids defined in
            :py:class:`earth2studio.lexicon.NNJAObsConvLexicon`.
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` (default) returns all schema
            fields.

        Returns
        -------
        pd.DataFrame
            Observation DataFrame with columns matching the resolved schema.
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
    # Async fetch (downloads + decode)
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = list({task.s3_uri for task in async_tasks})

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
                for uri in file_uri_set
            ]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching NNJA files",
                verbose=(not self._verbose),
            )

        # Decoding is CPU-bound and internally blocks on pool futures; run it in
        # a worker thread so it does not stall the shared fsspec IO loop that
        # _sync_async dispatches every data source's fetch onto.
        df = await asyncio.to_thread(
            self._compile_dataframe, async_tasks, variable_list, schema
        )
        df.attrs["source"] = self.SOURCE_ID
        return df

    # ------------------------------------------------------------------
    # File fetch
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, path: str) -> None:
        """Download a single remote file into the cache directory."""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self._cache_path(path)
        if pathlib.Path(cache_path).is_file():
            return
        try:
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as fh:
                fh.write(data)
        except FileNotFoundError:
            self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Handle missing file during fetch. Override in subclasses if a
        warn-only behaviour is preferred."""
        logger.error(f"File {path} not found")
        raise FileNotFoundError(f"File {path} not found")

    # ------------------------------------------------------------------
    # Compile DataFrame
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        async_tasks: list,
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        n_tasks = len(async_tasks)
        compile_t0 = time.perf_counter()
        for idx, task in enumerate(async_tasks, start=1):
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {task.s3_uri}, skipping")
                continue
            short_uri = task.s3_uri.rsplit("/", 1)[-1]
            logger.info(f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} start: {short_uri}")
            t0 = time.perf_counter()
            try:
                df = self._decode_file(local_path, task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Failed to decode {local_path}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            if df is None or df.empty:
                logger.info(
                    f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} done : "
                    f"{short_uri} (empty) in {elapsed:.1f}s"
                )
                continue
            logger.info(
                f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} done : "
                f"{short_uri} ({len(df):,} rows) in {elapsed:.1f}s"
            )
            df.attrs["source"] = self.SOURCE_ID
            frames.append(df)

        logger.info(
            f"[{self.SOURCE_ID}] compile finished: {len(frames)} non-empty "
            f"frames, total {time.perf_counter() - compile_t0:.1f}s"
        )

        if not frames:
            return _empty_dataframe(self.SCHEMA)[schema.names]

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _create_tasks(self, time_list: list[datetime], variable: list[str]) -> list:
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    def _decode_file(self, local_path: str, task: Any) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _decode_file.")

    # ------------------------------------------------------------------
    # Cycle iteration shared by all NNJA subclasses
    # ------------------------------------------------------------------
    def _cycle_windows(
        self, time_list: list[datetime]
    ) -> dict[datetime, tuple[datetime, datetime]]:
        """Map each unique 6-hour cycle to the union of requested time windows.

        For each ``t`` in ``time_list`` we cover all 6-hour cycles
        whose synoptic time falls within ``[t + tol_lower, t + tol_upper]``.
        Multiple input times that map to the same cycle are merged by
        taking the union of their windows so the cycle file is fetched
        once while keeping observations valid for any of them.
        """
        windows: dict[datetime, tuple[datetime, datetime]] = {}
        for t in time_list:
            tmin = t + self._tolerance_lower
            tmax = t + self._tolerance_upper
            day = tmin.replace(minute=0, second=0, microsecond=0)
            day = day.replace(hour=(day.hour // 6) * 6)
            while day <= tmax:
                existing = windows.get(day)
                windows[day] = (
                    (min(existing[0], tmin), max(existing[1], tmax))
                    if existing is not None
                    else (tmin, tmax)
                )
                day += timedelta(hours=6)
        return windows

    # ------------------------------------------------------------------
    # Time validation / cache / fields
    # ------------------------------------------------------------------
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

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for this data source."""
        cache_location = os.path.join(datasource_cache_root(), "nnja")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nnja_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        if fields is None:
            return cls.SCHEMA
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            for f in fields:
                if f.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{f.name}' not in {cls.__name__} SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(f.name).type
                if f.type != expected:
                    raise TypeError(
                        f"Field '{f.name}' has type {f.type}, expected "
                        f"{expected} from class SCHEMA"
                    )
            return fields
        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in {cls.__name__} SCHEMA. "
                    f"Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NNJAObsConv(_NNJAObsBase):
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
        super().__init__(
            time_tolerance=time_tolerance,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
            async_workers=async_workers,
            decode_workers=decode_workers,
            retries=retries,
        )
        self._prepbufr_adapter = _NCEPPrepbufrAdapter(self._decode_workers)
        self._gpsro_adapter = _NCEPGpsroAdapter(self._decode_workers)

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(self, time_list: list[datetime], variable: list[str]) -> list:
        # Partition variables by lexicon route prefix:
        #   "prepbufr::..." -> conv/prepbufr/ tasks (PrepBUFR decoder)
        #   "gpsro::..."    -> gps/gpsro/ tasks (GPS RO BUFR decoder)
        prepbufr_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = (
            {}
        )
        gpsro_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = {}

        for v in variable:
            try:
                source_key, modifier = NNJAObsConvLexicon[v]  # type: ignore[misc]
            except KeyError:
                logger.error(f"Variable id '{v}' not found in NNJAObsConvLexicon")
                raise
            route, _, rest = source_key.partition("::")
            if route == "prepbufr":
                prepbufr_plan[v] = (rest, modifier)
            elif route == "gpsro":
                try:
                    desc_id = int(rest)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid gpsro lexicon entry '{source_key}' for {v}: "
                        f"expected an integer BUFR descriptor id"
                    ) from exc
                gpsro_plan[v] = (desc_id, modifier)
            else:
                raise ValueError(
                    f"Unknown route '{route}' in NNJAObsConvLexicon entry "
                    f"'{source_key}' for variable '{v}' (expected 'prepbufr' or 'gpsro')"
                )

        # Build one task per unique cycle file; when multiple requested
        # times map to the same cycle the task's window is the union of
        # those time windows (see ``_NNJAObsBase._cycle_windows``).
        windows = self._cycle_windows(time_list) if prepbufr_plan or gpsro_plan else {}
        tasks: list = []
        for cycle_dt, (tmin, tmax) in windows.items():
            if prepbufr_plan:
                tasks.append(
                    _NNJAConvTask(
                        s3_uri=self._build_prepbufr_uri(cycle_dt),
                        datetime_file=cycle_dt,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        var_plan=prepbufr_plan,
                    )
                )
            if gpsro_plan:
                tasks.append(
                    _NNJAGpsRoTask(
                        s3_uri=self._build_gpsro_uri(cycle_dt),
                        datetime_file=cycle_dt,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        var_plan=gpsro_plan,
                    )
                )
        return tasks

    def _build_prepbufr_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single PrepBUFR cycle."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        archive_dir = self._source
        if self._source == "prepbufr.acft_profiles":
            archive_dir = "bufr"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/conv/{self._source}/"
            f"{year_key}/{month_key}/{archive_dir}/"
            f"gdas.{date_key}.t{hour_key}z.{self._source}.nr"
        )

    def _build_gpsro_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single gps/gpsro cycle file."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/gps/gpsro/"
            f"{year_key}/{month_key}/bufr/"
            f"gdas.{date_key}.t{hour_key}z.gpsro.tm00.bufr_d"
        )

    # Back-compat alias used by tests that targeted the v1 method name.
    def _build_uri(self, cycle: datetime) -> str:
        return self._build_prepbufr_uri(cycle)

    def _handle_missing_file(self, path: str) -> None:
        """Warn instead of raising on missing NNJA cycle files.

        NNJA does not guarantee every cycle/sub-archive combination
        exists (e.g. the ``gps/gpsro/`` archive only goes back to the
        early 2000s, and individual cycles can be absent). Returning a
        partial DataFrame is more useful than aborting a multi-cycle
        request because of one missing file.
        """
        logger.warning(f"NNJA conventional file {path} not found, skipping")

    # ------------------------------------------------------------------
    # File decode (dispatch by task type)
    # ------------------------------------------------------------------
    def _decode_file(
        self, local_path: str, task: _NNJAConvTask | _NNJAGpsRoTask
    ) -> pd.DataFrame:
        if isinstance(task, _NNJAGpsRoTask):
            return self._decode_gpsro_file(local_path, task)
        return self._decode_prepbufr_file(local_path, task)

    def _decode_prepbufr_file(
        self, local_path: str, task: _NNJAConvTask
    ) -> pd.DataFrame:
        """Decode one locally cached PrepBUFR file through the shared adapter."""
        frame = self._prepbufr_adapter.decode_file(
            local_path,
            task.var_plan,
            task.datetime_min,
            task.datetime_max,
        )
        if (
            self._source == "prepbufr.acft_profiles"
            and self._map_acft_profile_report_types
        ):
            frame = map_aircraft_profile_types(frame)
        return frame[self.SCHEMA.names]

    def _decode_gpsro_file(self, local_path: str, task: _NNJAGpsRoTask) -> pd.DataFrame:
        """Decode one locally cached GPSRO file through the shared adapter."""
        frame = self._gpsro_adapter.decode_file(
            local_path,
            task.var_plan,
            task.datetime_min,
            task.datetime_max,
        )
        return frame[self.SCHEMA.names]
