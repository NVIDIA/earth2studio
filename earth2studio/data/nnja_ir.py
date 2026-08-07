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
"""NNJA hyperspectral IR sounder observation data source.

Reads AIRS, IASI, and CrIS observations from the NOAA-NASA Joint Archive
(NNJA) on the public AWS S3 bucket ``s3://noaa-reanalyses-pds``.

All sensors are returned in brightness temperature (K):
- AIRS  (Aqua, 2002–2023): directly stored as BT in the BUFR (``TMBR``).
- IASI  (MetOp-A/B/C, 2007–present): stored as integer-coded scaled radiance
  (``SCRA`` + per-band ``CHSF``); converted via Planck inversion.
- CrIS  (NPP/NOAA-20/NOAA-21, 2012–present): stored as float radiance
  (``SRAD``, W m⁻² sr⁻¹ (cm⁻¹)⁻¹); converted via Planck inversion.

Use together with :py:class:`earth2studio.data.NNJAObsSat` for microwave
sensors and :py:class:`earth2studio.data.RoutedObsSource` to combine them
into a single logical observation stream::

    sat_source = RoutedObsSource({
        ("atms", "mhs", "amsua", "amsub"): NNJAObsSat(),
        ("airs", "iasi", "cris"):          NNJAObsIRSat(ir_channels="ir32"),
    })
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
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
from earth2studio.data.utils_ir import ir_channel_preset
from earth2studio.data.utils_ncep import (
    NCEP_MICROWAVE_OUTPUT_SCHEMA,
    compile_dataframe,
    cycle_windows,
    decode_ir_sounder,
    resolve_output_schema,
)
from earth2studio.lexicon.nnja_ir import NNJAObsIRSatLexicon
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


@dataclass(frozen=True)
class _NNJAIRProduct:
    prefix: str  # path segment after NNJA_PREFIX, e.g. "airs/airsev"
    filename: str  # BUFR filename stem, e.g. "airsev"
    first_year: int


_NNJA_IR_PRODUCTS: dict[str, _NNJAIRProduct] = {
    "airs": _NNJAIRProduct("airs/airsev", "airsev", 2002),
    "iasi": _NNJAIRProduct("iasi/mtiasi", "mtiasi", 2007),
    "cris": _NNJAIRProduct("cris/cris", "cris", 2012),
}

# Valid satellite platforms per sensor (subset of NCEP_MICROWAVE_SATELLITES).
_IR_SATELLITES: dict[str, frozenset[str]] = {
    "airs": frozenset({"aqua"}),
    "iasi": frozenset({"metop-a", "metop-b", "metop-c"}),
    "cris": frozenset({"npp", "n20", "n21"}),
}

NNJA_IR_SATELLITES: frozenset[str] = frozenset().union(*_IR_SATELLITES.values())


@dataclass(frozen=True)
class _NNJAIRTask:
    uri: str
    sensor: str
    channels: frozenset[int] | None
    datetime_min: datetime
    datetime_max: datetime
    satellites: tuple[str, ...] | None


@check_optional_dependencies()
class NNJAObsIRSat:
    """NNJA hyperspectral IR sounder satellite observations (AIRS, IASI, CrIS).

    Reads from ``s3://noaa-reanalyses-pds/observations/reanalysis/`` and
    returns one long-format row per (footprint, channel) with the observation
    in brightness temperature (K).

    ``sensor_index`` is the instrument's own channel number. ``wavenumber`` is
    the channel centre wavenumber in cm⁻¹. ``observation`` is always
    brightness temperature in Kelvin.

    Parameters
    ----------
    ir_channels : str or dict[str, list[int]] or None
        Channel selection:

        - ``str``: a named preset from
          :py:attr:`earth2studio.data.utils_ir.IR_CHANNEL_PRESETS`
          (``"ir32"`` or ``"ir48"``). Applies the same subset to each sensor.
        - ``dict``: maps sensor names (``"airs"``, ``"iasi"``, ``"cris"``) to
          explicit channel number lists. Sensors absent from the dict read all
          channels.
        - ``None``: read every published channel (281 AIRS, up to 8461 IASI,
          up to 2211 CrIS). Very large; use only for diagnostics.

    time_tolerance : TimeTolerance, optional
        Symmetric or asymmetric window for observation filtering,
        by default ``np.timedelta64(10, "m")``.
    satellites : list[str] or None, optional
        Platform filter. ``None`` includes every platform for each sensor.
    cache : bool, optional
        Persist downloaded BUFR files locally, by default True.
    verbose : bool, optional
        Show download progress bars, by default True.
    async_timeout : int, optional
        Total async fetch timeout in seconds, by default 600.
    async_workers : int, optional
        Concurrent S3 fetch tasks, by default 8.
    decode_workers : int, optional
        BUFR decode worker processes (1 disables multiprocessing),
        by default 8.
    retries : int, optional
        Fetch retry attempts with exponential back-off, by default 3.

    Warning
    -------
    ``ir_channels=None`` yields very large DataFrames. Always specify a
    preset or explicit channel list for training and evaluation workloads.

    Note
    ----
    Archive coverage:

    - AIRS: Aqua, 2002–2023 (instrument failed January 2024).
    - IASI: MetOp-A 2007–2021, MetOp-B 2012–present, MetOp-C 2018–present.
    - CrIS: NPP 2012–present, NOAA-20 2017–present, NOAA-21 2022–present.

    References:

    - https://registry.opendata.aws/noaa-reanalyses-pds/
    - https://psl.noaa.gov/data/nnja_obs/

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.NNJAObsIRSat"
    SCHEMA = NCEP_MICROWAVE_OUTPUT_SCHEMA
    LEXICON = NNJAObsIRSatLexicon
    MIN_DATE = datetime(2002, 1, 1)
    VALID_SATELLITES = NNJA_IR_SATELLITES

    def __init__(
        self,
        ir_channels: str | dict[str, list[int]] | None = "ir32",
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 8,
        decode_workers: int = 8,
        retries: int = 3,
    ) -> None:
        # Resolve channel selection to per-sensor frozensets (or None = all)
        if ir_channels is None:
            self._channels: dict[str, frozenset[int] | None] = {
                s: None for s in _NNJA_IR_PRODUCTS
            }
        elif isinstance(ir_channels, str):
            preset = ir_channel_preset(ir_channels)
            self._channels = {s: frozenset(ch) for s, ch in preset.items()}
        elif isinstance(ir_channels, dict):
            self._channels = {}
            for s in _NNJA_IR_PRODUCTS:
                self._channels[s] = (
                    frozenset(ir_channels[s]) if s in ir_channels else None
                )
        else:
            raise TypeError(
                f"ir_channels must be a str preset, dict, or None; got {type(ir_channels)}"
            )

        if satellites is None:
            self._satellites: tuple[str, ...] | None = None
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {sorted(invalid)}. "
                    f"Valid: {sorted(self.VALID_SATELLITES)}"
                )
            self._satellites = tuple(sorted(set(satellites)))

        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._decode_workers = max(1, decode_workers)
        self._retries = retries
        self.async_timeout = async_timeout

        import uuid

        self._tmp_cache_hash: str | None = uuid.uuid4().hex[:8] if not cache else None

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
        """Fetch IR sounder observations for requested times and sensors."""
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
        """Async fetch implementation."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)

        tasks = self._create_tasks(time_list, variable_list)
        uris = list({task.uri for task in tasks})
        await self.fetch_files(uris)

        return compile_dataframe(
            tasks,
            schema,
            self.SOURCE_ID,
            self.local_path,
            self._decode_file,
        )

    async def fetch_files(self, uris: list[str]) -> None:
        coros = [
            async_retry(
                self._fetch_remote_file,
                uri,
                retries=self._retries,
                backoff=1.0,
                task_timeout=300.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for uri in uris
        ]
        await gather_with_concurrency(
            coros,
            max_workers=self._async_workers,
            desc="Fetching NNJA IR sounder files",
            verbose=(not self._verbose),
        )

    async def _fetch_remote_file(self, uri: str) -> None:
        key = uri.removeprefix(f"s3://{NNJA_BUCKET}/")
        await obstore_fetch_to_cache(
            self._store,
            key,
            self.cache,
            cache_key=hashlib.sha256(uri.encode()).hexdigest(),
        )

    def local_path(self, uri: str) -> str:
        return os.path.join(self.cache, hashlib.sha256(uri.encode()).hexdigest())

    @property
    def cache(self) -> str:
        return datasource_cache_dir("nnja_ir", self._cache, self._tmp_cache_hash)

    def cleanup(self) -> None:
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    def _decode_file(self, path: str, task: _NNJAIRTask) -> pd.DataFrame:
        return decode_ir_sounder(
            path,
            task.sensor,
            task.channels,
            task.datetime_min,
            task.datetime_max,
            task.satellites,
            self._decode_workers,
        )

    def _create_tasks(
        self, time_list: list[datetime], variable_list: list[str]
    ) -> list[_NNJAIRTask]:
        windows = cycle_windows(time_list, self._tolerance_lower, self._tolerance_upper)
        tasks: list[_NNJAIRTask] = []
        for sensor in variable_list:
            if sensor not in _NNJA_IR_PRODUCTS:
                raise ValueError(
                    f"NNJAObsIRSat: unknown variable {sensor!r}. "
                    f"Valid: {sorted(_NNJA_IR_PRODUCTS)}"
                )
            product = _NNJA_IR_PRODUCTS[sensor]
            sensor_sats = _IR_SATELLITES[sensor]
            sat_filter = (
                tuple(s for s in self._satellites if s in sensor_sats)
                if self._satellites is not None
                else None
            )
            for cycle, (dt_min, dt_max) in sorted(windows.items()):
                if cycle.year < product.first_year:
                    logger.warning(
                        f"NNJAObsIRSat: {sensor} archive starts {product.first_year}; "
                        f"skipping cycle {cycle.isoformat()}"
                    )
                    continue
                tasks.append(
                    _NNJAIRTask(
                        uri=self._build_ir_uri(cycle, sensor),
                        sensor=sensor,
                        channels=self._channels.get(sensor),
                        datetime_min=dt_min,
                        datetime_max=dt_max,
                        satellites=sat_filter,
                    )
                )
        return tasks

    @staticmethod
    def _build_ir_uri(cycle: datetime, sensor: str) -> str:
        product = _NNJA_IR_PRODUCTS[sensor]
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/{product.prefix}/"
            f"{cycle:%Y/%m}/bufr/gdas.{cycle:%Y%m%d}.t{cycle:%H}z."
            f"{product.filename}.tm00.bufr_d"
        )

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        for t in times:
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than "
                    f"{cls.__name__}.MIN_DATE ({cls.MIN_DATE.isoformat()})."
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Return True if the requested time is within the archive range."""
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve a field subset request against the output schema."""
        return resolve_output_schema(cls.SCHEMA, fields, class_name=cls.__name__)
