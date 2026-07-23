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

import hashlib
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from obstore.store import S3Store

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_dir,
    gather_with_concurrency,
    obstore_fetch_to_cache,
    prep_data_inputs,
)
from earth2studio.data.utils_bufr import BUFR_DEPENDENCY_KEY
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    NCEP_MICROWAVE_OUTPUT_SCHEMA,
    NCEP_MICROWAVE_SATELLITES,
    NCEPObsTask,
    compile_dataframe,
    cycle_windows,
    decode_gpsro,
    decode_microwave,
    decode_prepbufr,
    map_aircraft_profile_types,
    plan_conv_tasks,
    resolve_output_schema,
)
from earth2studio.lexicon import NNJAObsConvLexicon, NNJAObsSatLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


@dataclass(frozen=True)
class _NNJASatProduct:
    prefix: str
    filename: str
    first_year: int


class _NNJAObsSatIncompleteError(RuntimeError):
    def __init__(self, reason: str, **context: object) -> None:
        self.context = {"reason": reason, **context}
        super().__init__(f"NNJAObsSat request incomplete: {self.context}")


_NNJA_SAT_PRODUCTS: dict[str, _NNJASatProduct] = {
    "atms": _NNJASatProduct("atms/atms", "atms", 2012),
    "mhs": _NNJASatProduct("mhs/1bmhs", "1bmhs", 2005),
    "amsua": _NNJASatProduct("amsua/1bamua", "1bamua", 1998),
    "amsub": _NNJASatProduct("amsub/1bamub", "1bamub", 1998),
}


@dataclass(frozen=True)
class _NNJASatTask:
    """Fetch/decode task for one aggregate microwave cycle file."""

    uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    sensor: str
    var_plan: dict[str, str] = field(default_factory=dict)


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NNJAObsConv:
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
        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._decode_workers = max(1, decode_workers)
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = uuid.uuid4().hex[:8] if not cache else None
        # Anonymous obstore S3 store for the public NNJA bucket.
        self._store = S3Store(
            NNJA_BUCKET,
            region="us-east-1",
            skip_signature=True,
            client_options={"pool_max_idle_per_host": str(async_workers)},
        )

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps."""
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            self.cleanup()

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)

        tasks = self._create_tasks(time_list, variable_list)
        await self.fetch_files(list({task.uri for task in tasks}))

        return compile_dataframe(
            tasks,
            schema,
            self.SOURCE_ID,
            self.local_path,
            self._decode_file,
        )

    async def fetch_files(self, uris: Sequence[str]) -> None:
        """Download remote files into the NNJA cache."""
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
        key = path.removeprefix(f"s3://{NNJA_BUCKET}/")
        try:
            await obstore_fetch_to_cache(
                self._store,
                key,
                self.cache,
                cache_key=hashlib.sha256(path.encode()).hexdigest(),
            )
        except FileNotFoundError:
            self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Warn instead of raising on missing NNJA cycle files.

        NNJA does not guarantee every cycle/sub-archive combination exists
        (e.g. the ``gps/gpsro/`` archive only goes back to the early 2000s, and
        individual cycles can be absent). Returning a partial DataFrame is more
        useful than aborting a multi-cycle request because of one missing file.
        """
        logger.warning(f"NNJA file {path} not found, skipping")

    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for NNJA observation files."""
        return datasource_cache_dir("nnja", self._cache, self._tmp_cache_hash)

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[NCEPObsTask]:
        return plan_conv_tasks(
            cycle_windows(time_list, self._tolerance_lower, self._tolerance_upper),
            variable,
            self.LEXICON,
            self._build_uri,
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
    def _decode_file(self, local_path: str, task: NCEPObsTask) -> pd.DataFrame:
        if task.route == "gpsro":
            frame = decode_gpsro(
                local_path,
                task.var_plan,
                task.datetime_min,
                task.datetime_max,
                decode_workers=self._decode_workers,
            )
            return frame[self.SCHEMA.names]
        if task.route == "prepbufr":
            frame = decode_prepbufr(
                local_path,
                task.var_plan,
                task.datetime_min,
                task.datetime_max,
                decode_workers=self._decode_workers,
            )
            if (
                self._source == "prepbufr.acft_profiles"
                and self._map_acft_profile_report_types
            ):
                frame = map_aircraft_profile_types(frame)
            return frame[self.SCHEMA.names]
        raise ValueError(f"Unsupported route '{task.route}'")

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are in the NNJA archive range."""
        for t in times:
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

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        return resolve_output_schema(cls.SCHEMA, fields, class_name=cls.__name__)


@check_optional_dependencies(BUFR_DEPENDENCY_KEY)
class NNJAObsSat:
    """NNJA historical NCEP aggregate microwave satellite observations.

    This source reads the NCEP satellite ATMS, MHS, AMSU-A, and AMSU-B
    BUFR products from the NNJA archive. It returns one long-format row per
    finite encoded channel value. ``sensor_index`` is the physical ``CHNM``
    channel number, not a dense index into a selected channel list.

    ``atms`` returns the encoded 22-channel ``TMBR`` scene brightness
    temperature. ``atms_antenna_temperature`` returns the corresponding
    encoded ``TMANT`` antenna temperature. No conversion is performed between
    those two ATMS products.

    ``mhs`` (5 channels), ``amsua`` (15 channels), and ``amsub`` (5 channels)
    return their normal-feed ``TMBR`` values as encoded by NCEP. For these
    legacy products the mnemonic does not always mean the same physical
    antenna-correction state. GSI treats normal-feed ``TMBR`` as antenna
    temperature for AMSU-A, AMSU-B, and MHS, except for NOAA-15/16, which its
    reader treats as already converted upstream. NOAA satingest independently
    confirms that exception for AMSU-A. The platform identity is retained so a
    downstream transform can apply the appropriate convention explicitly.

    ``scan_position`` preserves the encoded one-based FOV number.
    ``scan_angle`` is the signed nominal instrument look angle derived from
    that FOV; negative values are on the first half of the scan. The encoded
    ``satellite_za`` remains the unsigned Earth-view zenith magnitude.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric +/- window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(10, 'm').
    satellites : list[str] | None, optional
        Satellite platforms to include. ``None`` includes every platform in
        the requested aggregate files.
    cache : bool, optional
        Cache downloaded files in the local filesystem cache, by default True.
    verbose : bool, optional
        Show progress bars, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 8.
    decode_workers : int, optional
        Number of parallel processes for BUFR message decoding. Set to 1 to
        disable multiprocessing, by default 8.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    Aggregate cycle files contain millions of footprints. Broad long-format
    requests can require substantial memory. A finite archived value is not a
    QC decision: historical files may retain passive or degraded channels even
    when the aggregate carries no usable quality flag. Training pipelines
    should apply an explicit platform/channel validity policy.

    Note
    ----
    Additional information on the archive and microwave product semantics:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-pds/
    - https://github.com/NOAA-EMC/GSI/blob/860d13740352004fca0136a8c3d0ac9dea30e0da/src/gsi/read_bufrtovs.f90#L754-L823
    - https://github.com/NOAA-EMC/GSI/blob/860d13740352004fca0136a8c3d0ac9dea30e0da/src/gsi/radinfo.f90#L1523-L1643
    - https://github.com/NOAA-EMC/satingest/blob/3bb883d931d2cbdbd8c5871c30ac25941918c882/ush/ingest_script_atovs1b.sh#L188-L231
    - https://github.com/NOAA-EMC/satingest/blob/3bb883d931d2cbdbd8c5871c30ac25941918c882/sorc/bufr_tranamsua.fd/tranamsua.f#L887-L910
    - https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-001_JPSS_ATBD_ATMS-SDR_B.pdf
    - https://user.eumetsat.int/s3/ope-eup-strapi-media/ATOVS_Level_1b_Product_Guide_f89971ac20.pdf
    - https://www.ncei.noaa.gov/pub/data/cdo/documentation/podguides/N-15%20thru%20N-19/pdf/APPENDIX%20J%20Instrument%20Scan%20Properties.pdf

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.NNJAObsSat"
    SCHEMA = NCEP_MICROWAVE_OUTPUT_SCHEMA
    LEXICON = NNJAObsSatLexicon
    MIN_DATE = datetime(1998, 1, 1)
    VALID_SATELLITES = NCEP_MICROWAVE_SATELLITES

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 8,
        decode_workers: int = 8,
        retries: int = 3,
    ) -> None:
        if satellites is None:
            self._satellites: tuple[str, ...] | None = None
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {sorted(invalid)}. "
                    f"Valid options: {sorted(self.VALID_SATELLITES)}"
                )
            self._satellites = tuple(sorted(set(satellites)))

        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._decode_workers = max(1, decode_workers)
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = uuid.uuid4().hex[:8] if not cache else None
        # Anonymous obstore S3 store for the public NNJA bucket.
        self._store = S3Store(
            NNJA_BUCKET,
            region="us-east-1",
            skip_signature=True,
            client_options={"pool_max_idle_per_host": str(async_workers)},
        )

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps."""
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            self.cleanup()

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)

        tasks = self._create_tasks(time_list, variable_list)
        uris = list({task.uri for task in tasks})
        try:
            await self.fetch_files(uris)
        except Exception as exc:
            self._handle_fetch_failure(uris, exc)

        return compile_dataframe(
            tasks,
            schema,
            self.SOURCE_ID,
            self.local_path,
            self._decode_file,
            self._handle_incomplete_task,
        )

    async def fetch_files(self, uris: Sequence[str]) -> None:
        """Download remote files into the NNJA cache."""
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
            desc="Fetching NNJA satellite files",
            verbose=(not self._verbose),
        )

    async def _fetch_remote_file(self, path: str) -> None:
        """Download a single remote file into the cache directory."""
        key = path.removeprefix(f"s3://{NNJA_BUCKET}/")
        try:
            await obstore_fetch_to_cache(
                self._store,
                key,
                self.cache,
                cache_key=hashlib.sha256(path.encode()).hexdigest(),
            )
        except FileNotFoundError:
            self._handle_missing_file(path)

    def _handle_fetch_failure(self, uris: Sequence[str], cause: Exception) -> None:
        if isinstance(cause, _NNJAObsSatIncompleteError):
            raise cause
        raise _NNJAObsSatIncompleteError(
            "fetch_failure",
            requested_uri_count=len(uris),
            requested_uris=tuple(uris),
            cause_type=type(cause).__name__,
            cause_message=str(cause),
        ) from cause

    def _handle_incomplete_task(
        self,
        uri: str,
        task_index: int,
        task_count: int,
        cause: Exception,
    ) -> None:
        context: dict[str, object] = {
            "uri": uri,
            "task_index": task_index,
            "task_count": task_count,
            "cause_type": type(cause).__name__,
            "cause_message": str(cause),
        }
        cause_context = getattr(cause, "context", None)
        if isinstance(cause_context, dict):
            context["cause_context"] = cause_context
        raise _NNJAObsSatIncompleteError("task_failure", **context) from cause

    def _handle_missing_file(self, path: str) -> None:
        """Fail a request when an aggregate cycle file is absent."""
        raise _NNJAObsSatIncompleteError("remote_file_missing", uri=path)

    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for NNJA observation files."""
        return datasource_cache_dir("nnja", self._cache, self._tmp_cache_hash)

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_NNJASatTask]:
        variables_by_sensor: dict[str, dict[str, str]] = {}
        for variable_name in variable:
            source_key, _modifier = self.LEXICON[variable_name]
            sensor, separator, source_field = source_key.partition("::")
            if not separator or sensor not in _NNJA_SAT_PRODUCTS or not source_field:
                raise ValueError(f"Invalid NNJA satellite lexicon key: {source_key}")
            variables_by_sensor.setdefault(sensor, {})[variable_name] = source_field

        windows = cycle_windows(time_list, self._tolerance_lower, self._tolerance_upper)
        tasks: list[_NNJASatTask] = []
        for sensor, var_plan in variables_by_sensor.items():
            product = _NNJA_SAT_PRODUCTS[sensor]
            for cycle, (datetime_min, datetime_max) in sorted(windows.items()):
                if cycle.year < product.first_year:
                    uri = self._build_satellite_uri(cycle, sensor)
                    raise _NNJAObsSatIncompleteError(
                        "archive_unavailable",
                        uri=uri,
                        sensor=sensor,
                        cycle=cycle.isoformat(),
                        first_year=product.first_year,
                    )
                tasks.append(
                    _NNJASatTask(
                        uri=self._build_satellite_uri(cycle, sensor),
                        datetime_file=cycle,
                        datetime_min=datetime_min,
                        datetime_max=datetime_max,
                        sensor=sensor,
                        var_plan=var_plan,
                    )
                )
        return tasks

    @staticmethod
    def _build_satellite_uri(cycle: datetime, sensor: str) -> str:
        """Build the NNJA S3 URI for one aggregate microwave cycle."""
        product = _NNJA_SAT_PRODUCTS[sensor]
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/{product.prefix}/"
            f"{cycle:%Y/%m}/bufr/gdas.{cycle:%Y%m%d}.t{cycle:%H}z."
            f"{product.filename}.tm00.bufr_d"
        )

    def _decode_file(self, local_path: str, task: _NNJASatTask) -> pd.DataFrame:
        frame = decode_microwave(
            local_path,
            task.sensor,
            task.var_plan,
            task.datetime_min,
            task.datetime_max,
            self._satellites,
            decode_workers=self._decode_workers,
        )
        return frame[self.SCHEMA.names]

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are in the NNJA archive range."""
        for t in times:
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

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        return resolve_output_schema(cls.SCHEMA, fields, class_name=cls.__name__)
