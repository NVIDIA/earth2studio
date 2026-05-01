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
#
# This module is intentionally self-contained: the PrepBUFR decoding
# helpers below duplicate concepts already present in
# ``earth2studio.data.gdas`` (which decodes the same NCEP PrepBUFR file
# format from NOMADS). A future PR may extract a shared ``_prepbufr``
# module; for now the duplication keeps this PR isolated and avoids
# changing GDAS behaviour.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import os
import pathlib
import shutil
import struct
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import NNJAObsConvLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    from pybufrkit.decoder import Decoder as BufrDecoder
    from pybufrkit.tables import TableGroupCacheManager
except ImportError:
    OptionalDependencyFailure("data")
    BufrDecoder = None  # type: ignore[assignment,misc]
    TableGroupCacheManager = None  # type: ignore[assignment,misc]


NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


@contextlib.contextmanager
def _silence_bufr_noise() -> Iterator[None]:
    """Suppress chatty C-library stderr from pybufrkit.

    pybufrkit writes informational messages straight to file
    descriptor 2 (e.g. ``Cannot find sub-centre 3 nor valid default``)
    when the file uses NCEP-local descriptors. We rely on the DX
    tables embedded in each NNJA file to decode those correctly, so
    these messages are spurious and would otherwise flood the log
    with one line per BUFR message.

    The redirect only covers C-level writes; Python ``print``,
    ``logger`` and exceptions still propagate normally. We also
    flush ``sys.stderr`` first so any pending Python-side stderr
    is preserved.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_fd)


# ── PrepBUFR descriptor IDs (NCEP-local) ─────────────────────────────
# Header field descriptors
_HDR_SID = 1194  # Station ID
_HDR_XOB = 6240  # Longitude (deg E)
_HDR_YOB = 5002  # Latitude (deg N)
_HDR_DHR = 4215  # Obs time minus cycle time (h)
_HDR_ELV = 10199  # Station elevation (m)
_HDR_TYP = 55007  # Report type code
_HDR_T29 = 55008  # Data dump report type code

# Observation field descriptors
_OBS_CAT = 8193  # Observation category code
_OBS_POB = 7245  # Pressure observation (MB)
_OBS_ZOB = 10007  # Height (m)
_OBS_TOB = 12245  # Temperature (DEG C)
_OBS_QOB = 13245  # Specific humidity (MG/KG)
_OBS_UOB = 11003  # U-wind component (m/s)
_OBS_VOB = 11004  # V-wind component (m/s)

_OBSERVATION_DESCR_IDS: set[int] = {
    _OBS_POB,
    _OBS_ZOB,
    _OBS_TOB,
    _OBS_QOB,
    _OBS_UOB,
    _OBS_VOB,
}

# Lexicon mnemonic -> descriptor ID for non-wind variables
_MNEMONIC_TO_DESCR: dict[str, int] = {
    "TOB": _OBS_TOB,
    "QOB": _OBS_QOB,
    "POB": _OBS_POB,
    "ZOB": _OBS_ZOB,
    "UOB": _OBS_UOB,
    "VOB": _OBS_VOB,
}

# PrepBUFR section-1 dataCategory -> NCEP message-type class string
_PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper air: radiosondes, pilot balloons, dropsondes
    104: "AIRCFT",  # Aircraft
    105: "SATWND",  # Satellite-derived winds
    107: "VADWND",  # VAD (NEXRAD) winds
    109: "ADPSFC",  # Surface land
    110: "SFCSHP",  # Surface marine
    112: "GPSIPW",  # GPS precipitable water
    113: "SYNDAT",  # Synthetic bogus data
    119: "RASSDA",  # RASS virtual temperature
    121: "ASCATW",  # ASCAT scatterometer winds
}


# ── GPS RO BUFR descriptor IDs (NCEP gpsro encoding) ─────────────────
# Header descriptors (per-occultation, scalar)
_GPSRO_SAID = 1007  # Satellite identifier (receiver)
_GPSRO_PTID = 1050  # Platform transmitter ID (GPS satellite)
_GPSRO_QFRO = 33039  # Quality flags for radio occultation
_GPSRO_LAT = 5001  # Latitude (deg)
_GPSRO_LON = 6001  # Longitude (deg)
_GPSRO_YEAR = 4001
_GPSRO_MONTH = 4002
_GPSRO_DAY = 4003
_GPSRO_HOUR = 4004
_GPSRO_MIN = 4005
_GPSRO_SEC = 4006

# Per-level descriptors
_GPSRO_IMPP = 7040  # Impact parameter (m), bending-angle level marker
_GPSRO_BNDA = 15037  # Bending angle (rad)
_GPSRO_HEIT = 7007  # Height (m), refractivity level marker
_GPSRO_ARFR = 15036  # Atmospheric refractivity
_GPSRO_GPHTST = 7009  # Geopotential height (m), retrieval level marker
_GPSRO_PRES = 10004  # Pressure (Pa)
_GPSRO_TEMP = 12001  # Air temperature (K)
_GPSRO_SPFH = 13001  # Specific humidity (kg/kg)

# Descriptor IDs the gpsro decoder pulls out as observations
_GPSRO_OBS_DESCRS: set[int] = {_GPSRO_BNDA, _GPSRO_TEMP, _GPSRO_SPFH}


# ── Schemas ─────────────────────────────────────────────────────────

_NNJA_CONV_SCHEMA = pa.schema(
    [
        pa.field("time", pa.timestamp("ns"), metadata={"nnja_name": "Time"}),
        pa.field(
            "pres",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Pressure"},
        ),
        pa.field(
            "elev",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Height"},
        ),
        pa.field(
            "type",
            pa.uint16(),
            nullable=True,
            metadata={"nnja_name": "Observation_Type"},
        ),
        pa.field(
            "class",
            pa.string(),
            nullable=True,
            metadata={"nnja_name": "Observation_Class"},
        ),
        pa.field("lat", pa.float32(), metadata={"nnja_name": "Latitude"}),
        pa.field("lon", pa.float32(), metadata={"nnja_name": "Longitude"}),
        pa.field(
            "station",
            pa.string(),
            nullable=True,
            metadata={"nnja_name": "Station_ID"},
        ),
        pa.field(
            "station_elev",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Station_Elevation"},
        ),
        pa.field("observation", pa.float32()),
        pa.field("variable", pa.string()),
    ]
)

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
    # Map var_name -> (bufr_descriptor_id, modifier)
    var_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────


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
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        self._verbose = verbose
        self._cache = cache
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

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
        # ``asyncio.get_event_loop()`` is deprecated in Python 3.10+ when
        # there is no current loop, so prefer ``get_running_loop`` and
        # fall back to creating a fresh loop ourselves.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        try:
            df = loop.run_until_complete(
                asyncio.wait_for(
                    self.fetch(time, variable, fields), timeout=self.async_timeout
                )
            )
        finally:
            # Ensure the temporary cache directory is cleaned up even if
            # ``fetch`` raises (e.g. ``asyncio.TimeoutError`` from
            # ``wait_for``); otherwise a failed call would leak a tmp
            # directory under the user's cache root.
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
            raise ValueError(
                "File store is not initialized! If calling this function "
                "directly make sure the data source is initialized inside "
                "the async loop."
            )

        session = await self.fs.set_session(refresh=True)

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = {task.s3_uri for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(uri) for uri in file_uri_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching NNJA files", disable=(not self._verbose)
        )

        if session:
            await session.close()

        df = self._compile_dataframe(async_tasks, variable_list, schema)
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
            logger.info(
                f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} start: {short_uri}"
            )
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
            return pd.DataFrame(
                {name: pd.Series(dtype=object) for name in self.SCHEMA.names}
            )[[name for name in schema.names if name in self.SCHEMA.names]]

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
        once but ``_extract_subset`` keeps observations valid for any
        of them.
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


# ─────────────────────────────────────────────────────────────────────
# Self-contained PrepBUFR decoder (used by NNJAObsConv)
# ─────────────────────────────────────────────────────────────────────


def _safe_int(v: Any) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, bytes):
        s = v.decode("ascii", errors="replace").strip()
    elif v is None:
        s = ""
    else:
        s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _parse_prepbufr_messages(
    file_data: bytes,
) -> tuple[
    dict[int, tuple[Any, ...]],
    dict[int, tuple[Any, ...]],
    list[tuple[bytes, int]],
]:
    """Split a PrepBUFR byte stream into messages and extract DX tables.

    The first several messages of a PrepBUFR file are DX-table messages
    (dataCategory=11) carrying the NCEP-local Table B / Table D
    descriptor definitions needed to decode subsequent data messages.
    """
    table_b: dict[int, tuple[Any, ...]] = {}
    table_d: dict[int, tuple[Any, ...]] = {}
    data_messages: list[tuple[bytes, int]] = []
    dx_messages: list[bytes] = []

    pos = 0
    while pos < len(file_data):
        idx = file_data.find(b"BUFR", pos)
        if idx == -1:
            break
        msg_len = struct.unpack(">I", b"\x00" + file_data[idx + 4 : idx + 7])[0]
        if msg_len < 8:
            pos = idx + 4
            continue
        msg_bytes = file_data[idx : idx + msg_len]

        # BUFR ed3/4: section-0 = 8 bytes, section-1 octet-9 (offset 16) is dataCategory
        data_cat = file_data[idx + 16] if idx + 16 < len(file_data) else 0
        if data_cat == 11:
            dx_messages.append(msg_bytes)
        else:
            data_messages.append((msg_bytes, data_cat))
        pos = idx + msg_len

    if dx_messages:
        with _silence_bufr_noise():
            try:
                dx_decoder = BufrDecoder()
                for dx_bytes in dx_messages:
                    try:
                        dx_msg = dx_decoder.process(dx_bytes)
                    except Exception:  # noqa: S112
                        logger.debug("Skipping unparseable NNJA DX-table message")
                        continue
                    td = dx_msg.template_data.value
                    dvas = td.decoded_values_all_subsets
                    if not dvas:
                        continue
                    _extract_dx_tables(dvas[0], table_b, table_d)
            except Exception as e:
                logger.warning(f"Failed to extract NNJA DX tables: {e}")

    return table_b, table_d, data_messages


def _extract_dx_tables(
    flat: list[Any],
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Extract NCEP Table B and D entries from a DX-message subset.

    Encoding layout (per NCEP BUFRLIB):

    - n_table_a, [table_a entries (3 fields each) ...]
    - n_table_b, [table_b entries (11 fields each) ...]
    - n_table_d, [table_d entries (variable length) ...]
    """

    def _str(v: Any) -> str:
        if isinstance(v, bytes):
            return v.decode("ascii", errors="replace").strip()
        if v is None:
            return ""
        return str(v).strip()

    def _fxy(f: Any, x: Any, y: Any) -> int:
        return _safe_int(f) * 100000 + _safe_int(x) * 1000 + _safe_int(y)

    n = len(flat)
    idx = 0
    if idx >= n:
        return
    n_a = _safe_int(flat[idx])
    idx += 1
    idx += n_a * 3
    if idx >= n:
        return

    # Table B
    n_b = _safe_int(flat[idx])
    idx += 1
    for _ in range(n_b):
        if idx + 10 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        mnemonic = _str(flat[idx + 3])
        unit = _str(flat[idx + 5])
        sign_scale = _str(flat[idx + 6])
        scale_s = _str(flat[idx + 7])
        sign_ref = _str(flat[idx + 8])
        ref_s = _str(flat[idx + 9])
        width_s = _str(flat[idx + 10])
        idx += 11

        desc_id = _fxy(f_v, x_v, y_v)
        if desc_id == 0:
            continue
        scale = _safe_int(scale_s)
        if sign_scale == "-":
            scale = -scale
        reference = _safe_int(ref_s)
        if sign_ref == "-":
            reference = -reference
        width = _safe_int(width_s)
        table_b[desc_id] = (
            mnemonic,
            unit,
            scale,
            reference,
            width,
            unit,
            scale,
            max(1, (width + 3) // 4),
        )

    # Table D
    if idx >= n:
        return
    n_d = _safe_int(flat[idx])
    idx += 1
    for _ in range(n_d):
        if idx + 3 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        seq_mnemonic = _str(flat[idx + 3])
        idx += 4
        seq_id = _fxy(f_v, x_v, y_v)
        if seq_id == 0:
            continue
        if idx >= n:
            return
        n_members = _safe_int(flat[idx])
        idx += 1
        members: list[str] = []
        for _ in range(n_members):
            if idx >= n:
                break
            members.append(_str(flat[idx]))
            idx += 1
        if members:
            table_d[seq_id] = (seq_mnemonic, members)


# Worker globals (set per-process by _init_worker)
_worker_decoder: Any = None


def _register_dx_tables(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Reset pybufrkit's table cache and (re-)register NCEP DX tables."""
    TableGroupCacheManager.clear_extra_entries()
    # ``_TABLE_GROUP_CACHE`` is a private attribute of pybufrkit's
    # ``TableGroupCacheManager``; reach into it to invalidate any
    # previously-built TableGroup that captured a stale set of extra
    # entries. If pybufrkit ever renames or restructures this cache we
    # log a warning and fall back to ``add_extra_entries`` alone, which
    # still works for the common case where the cache hasn't been
    # populated yet in the current process.
    try:
        TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
    except AttributeError as exc:
        logger.warning(
            f"pybufrkit TableGroupCacheManager._TABLE_GROUP_CACHE not available "
            f"({exc}); skipping cache invalidation"
        )
    if table_b or table_d:
        TableGroupCacheManager.add_extra_entries(table_b, table_d)


def _init_worker(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """ProcessPoolExecutor initializer: register NCEP DX tables, create a
    per-process pybufrkit decoder, and redirect the worker's stderr to
    /dev/null (the worker only decodes BUFR; real errors come back
    through exceptions, not stderr)."""
    global _worker_decoder  # noqa: PLW0603
    try:
        sys.stderr.flush()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
    except OSError:
        pass
    _register_dx_tables(table_b, table_d)
    _worker_decoder = BufrDecoder()


def _decode_message_worker(
    msg_bytes: bytes,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single PrepBUFR message in a worker process."""
    return _decode_message(_worker_decoder, msg_bytes, obs_class, var_keys, dt_min, dt_max)


def _decode_gpsro_message_worker(
    msg_bytes: bytes,
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single GPS-RO BUFR message in a worker process.

    Each NNJA gpsro message contains one occultation with several
    hundred delayed-replication levels; pybufrkit's pure-Python
    bitstream decoder is the bottleneck, so we parallelize across
    messages with a ProcessPoolExecutor (same pattern as PrepBUFR).
    """
    try:
        msg = _worker_decoder.process(msg_bytes)
        n_subsets = msg.n_subsets.value
    except Exception:
        return []
    if not n_subsets:
        return []
    td = msg.template_data.value
    ddas = td.decoded_descriptors_all_subsets
    dvas = td.decoded_values_all_subsets
    rows: list[dict[str, Any]] = []
    for s_idx in range(n_subsets):
        rows.extend(
            _extract_gpsro_subset(
                ddas[s_idx], dvas[s_idx], wanted_descrs, dt_min, dt_max
            )
        )
    return rows


def _decode_message(
    decoder: Any,
    msg_bytes: bytes,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single PrepBUFR message and emit observation rows.

    ``var_keys`` is a list of ``(var_name, lexicon_key)`` pairs where
    ``lexicon_key`` is one of ``TOB``, ``QOB``, ``POB``, ``ZOB``,
    ``wind::u``, ``wind::v``.
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

    msg_year = msg.year.value
    if msg_year < 100:
        msg_year += 2000 if msg_year < 70 else 1900
    try:
        base_time = datetime(
            msg_year, msg.month.value, msg.day.value, msg.hour.value, msg.minute.value
        )
    except (ValueError, OverflowError):
        return []

    rows: list[dict[str, Any]] = []
    for s_idx in range(n_subsets):
        rows.extend(
            _extract_subset(
                ddas[s_idx], dvas[s_idx], base_time, obs_class, var_keys, dt_min, dt_max
            )
        )
    return rows


def _extract_subset(
    descs: list[Any],
    vals: list[Any],
    base_time: datetime,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Extract observation rows from a single decoded PrepBUFR subset.

    The subset is a flat list of (descriptor, value) pairs.  We first
    walk the header (SID/XOB/YOB/DHR/ELV/TYP) and then iterate over the
    repeated CAT/POB level blocks, emitting one row per (level,
    requested variable) where the variable's descriptor has a
    non-missing value.
    """
    rows: list[dict[str, Any]] = []

    header: dict[str, Any] = {
        "sid": "",
        "xob": None,
        "yob": None,
        "dhr": 0.0,
        "elv": None,
        "typ": None,
    }
    for d, v in zip(descs, vals):
        did = d.id
        if did == _HDR_SID:
            header["sid"] = (
                v.decode("ascii", errors="replace").strip()
                if isinstance(v, bytes)
                else (str(v).strip() if v is not None else "")
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
        elif did == _OBS_CAT:
            break

    lat = header["yob"]
    lon = header["xob"]
    if lat is None or lon is None:
        return rows
    if lat < -90.0 or lat > 90.0:
        return rows

    try:
        obs_time = base_time + timedelta(hours=float(header["dhr"]))
    except (ValueError, OverflowError, TypeError):
        obs_time = base_time
    if obs_time < dt_min or obs_time > dt_max:
        return rows

    lon_360 = float(lon) % 360.0

    # Build the per-variable descriptor lookup once
    needed_ids: dict[str, int] = {}
    need_wind = False
    for var_name, key in var_keys:
        if key.startswith("wind::"):
            need_wind = True
        elif key in _MNEMONIC_TO_DESCR:
            needed_ids[var_name] = _MNEMONIC_TO_DESCR[key]

    base_row: dict[str, Any] = {
        "time": obs_time,
        "lat": np.float32(lat),
        "lon": np.float32(lon_360),
        "pres": None,
        "elev": None,
        "type": np.uint16(int(header["typ"])) if header["typ"] is not None else None,
        "class": obs_class if obs_class else None,
        "station": header["sid"] if header["sid"] else None,
        "station_elev": (
            np.float32(header["elv"]) if header["elv"] is not None else None
        ),
    }

    # Walk observation levels: a new POB starts a level
    current: dict[int, Any] = {}
    in_obs = False
    for d, v in zip(descs, vals):
        did = d.id
        if did == _OBS_POB:
            if in_obs and current:
                _emit_level_rows(
                    rows, current, base_row, needed_ids, need_wind, var_keys
                )
            current = {_OBS_POB: v}
            in_obs = True
        elif in_obs and did in _OBSERVATION_DESCR_IDS:
            if did not in current:
                current[did] = v
    if in_obs and current:
        _emit_level_rows(rows, current, base_row, needed_ids, need_wind, var_keys)

    return rows


def _extract_gpsro_subset(
    descs: list[Any],
    vals: list[Any],
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Extract observation rows from one GPS RO occultation subset.

    ``wanted_descrs`` maps BUFR descriptor id -> Earth2Studio variable
    name (e.g. ``{15037: "gps", 12001: "gps_t", 13001: "gps_q"}``). For
    each non-missing value of a wanted descriptor encountered in the
    subset's flat (descriptor, value) stream we emit one row.

    The NCEP gpsro encoding lays out the per-level data sequentially as
    three sub-profiles in this order:

    1. Bending-angle profile keyed on ``IMPP`` (descriptor 7040), with
       observation in ``BNDA`` (15037).
    2. Refractivity profile keyed on ``HEIT`` (7007), observation in
       ``ARFR`` (15036).
    3. 1D-Var retrieval profile keyed on ``GPHTST`` (7009), with
       ``PRES`` / ``TMDBST`` / ``SPFH`` (10004 / 12001 / 13001).
    """
    rows: list[dict[str, Any]] = []

    # Header pass
    sat_id: Any = None
    tx_id: Any = None
    qf: Any = None
    lat: float | None = None
    lon: float | None = None
    yyyy = mm = dd = hh = mi = None
    sec: float = 0.0
    for d, v in zip(descs, vals):
        did = d.id
        if did == _GPSRO_SAID:
            sat_id = v
        elif did == _GPSRO_PTID:
            tx_id = v
        elif did == _GPSRO_QFRO:
            qf = v
        elif did == _GPSRO_LAT and v is not None:
            lat = float(v)
        elif did == _GPSRO_LON and v is not None:
            lon = float(v)
        elif did == _GPSRO_YEAR and v is not None:
            yyyy = int(v)
        elif did == _GPSRO_MONTH and v is not None:
            mm = int(v)
        elif did == _GPSRO_DAY and v is not None:
            dd = int(v)
        elif did == _GPSRO_HOUR and v is not None:
            hh = int(v)
        elif did == _GPSRO_MIN and v is not None:
            mi = int(v)
        elif did == _GPSRO_SEC and v is not None:
            try:
                sec = float(v)
            except (TypeError, ValueError):
                sec = 0.0
        elif did == _GPSRO_IMPP:
            break

    if lat is None or lon is None or yyyy is None or mm is None or dd is None:
        return rows
    try:
        obs_time = datetime(yyyy, mm, dd, hh or 0, mi or 0, int(sec))
    except (ValueError, OverflowError):
        return rows
    if obs_time < dt_min or obs_time > dt_max:
        return rows

    lon_360 = lon % 360.0
    station_id = (
        f"{int(sat_id)}_{int(tx_id)}"
        if sat_id is not None and tx_id is not None
        else None
    )

    # Per-level pass
    cur_pres: float | None = None
    cur_height: float | None = None
    cur_impp: float | None = None

    for d, v in zip(descs, vals):
        did = d.id
        if v is None:
            if did == _GPSRO_IMPP:
                cur_impp = None
            elif did == _GPSRO_GPHTST or did == _GPSRO_HEIT:
                cur_height = None
                cur_pres = None
            elif did == _GPSRO_PRES:
                cur_pres = None
            continue

        if did == _GPSRO_IMPP:
            try:
                cur_impp = float(v)
            except (TypeError, ValueError):
                cur_impp = None
            continue
        if did == _GPSRO_GPHTST or did == _GPSRO_HEIT:
            try:
                cur_height = float(v)
            except (TypeError, ValueError):
                cur_height = None
            continue
        if did == _GPSRO_PRES:
            try:
                cur_pres = float(v)
            except (TypeError, ValueError):
                cur_pres = None
            continue

        if did not in wanted_descrs or did not in _GPSRO_OBS_DESCRS:
            continue
        try:
            obs_val = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(obs_val):
            continue

        var_name = wanted_descrs[did]
        if did == _GPSRO_BNDA:
            pres_val = None
            elev_val = np.float32(cur_impp) if cur_impp is not None else None
        else:
            pres_val = np.float32(cur_pres) if cur_pres is not None else None
            elev_val = np.float32(cur_height) if cur_height is not None else None

        rows.append(
            {
                "time": obs_time,
                "lat": np.float32(lat),
                "lon": np.float32(lon_360),
                "pres": pres_val,
                "elev": elev_val,
                "type": np.uint16(int(qf)) if qf is not None else None,
                "class": "GPSRO",
                "station": station_id,
                "station_elev": None,
                "observation": np.float32(obs_val),
                "variable": var_name,
            }
        )

    return rows


def _emit_level_rows(
    rows: list[dict[str, Any]],
    level: dict[int, Any],
    base_row: dict[str, Any],
    needed_ids: dict[str, int],
    need_wind: bool,
    var_keys: list[tuple[str, str]],
) -> None:
    """Append one row per requested variable for the current pressure level."""
    pob = level.get(_OBS_POB)
    pres_val = np.float32(pob) if pob is not None else None  # PrepBUFR mb (lexicon mod converts to Pa for `pres`)

    common = base_row.copy()
    common["pres"] = pres_val

    # Non-wind variables
    for var_name, desc_id in needed_ids.items():
        val = level.get(desc_id)
        if val is None:
            continue
        row = common.copy()
        row["variable"] = var_name
        row["observation"] = np.float32(val)
        rows.append(row)

    # Wind decomposition: u from UOB, v from VOB. Each component is
    # emitted independently so a level with only one of UOB/VOB still
    # yields a row for the requested component (PrepBUFR usually pairs
    # u/v but unpaired levels do occur).
    if need_wind:
        uob = level.get(_OBS_UOB)
        vob = level.get(_OBS_VOB)
        for var_name, key in var_keys:
            if key == "wind::u" and uob is not None:
                row = common.copy()
                row["variable"] = var_name
                row["observation"] = np.float32(uob)
                rows.append(row)
            elif key == "wind::v" and vob is not None:
                row = common.copy()
                row["variable"] = var_name
                row["observation"] = np.float32(vob)
                rows.append(row)


# ─────────────────────────────────────────────────────────────────────
# NNJAObsConv
# ─────────────────────────────────────────────────────────────────────


@check_optional_dependencies()
class NNJAObsConv(_NNJAObsBase):
    """NNJA conventional (in-situ + GPS RO) observations data source.

    Reads observations from two complementary NNJA archives based on
    the requested variable's lexicon route:

    - ``u, v, q, t, pres`` -> ``conv/<source>/`` PrepBUFR cycle files
      (default ``source="prepbufr"``).
    - ``gps, gps_t, gps_q`` -> ``gps/gpsro/`` BUFR cycle files
      (bending angle and 1D-Var retrieval temperature / specific
      humidity profiles).

    Returns a :class:`pandas.DataFrame` with one row per (cycle,
    observation level, requested variable). Variable routing is
    handled automatically through
    :py:class:`earth2studio.lexicon.NNJAObsConvLexicon`.

    Parameters
    ----------
    source : {"prepbufr", "convbufr", "prepbufr.acft_profiles"}, optional
        Which encoding family of the NNJA conventional archive to read,
        by default ``"prepbufr"``.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple ``(lower, upper)`` for
        asymmetric windows, by default ``np.timedelta64(0, 'm')``.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent S3 downloads,
        by default 24.
    cache : bool, optional
        Cache downloaded files in the local filesystem cache, by default
        ``True``.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    verbose : bool, optional
        Show progress bars, by default ``True``.

    Warning
    -------
    This is a remote data source and may download a large amount of data
    for large requests. A single PrepBUFR cycle file is approximately
    100 MB.

    Note
    ----
    NNJA observation data is distributed under CC BY 4.0; please cite
    https://psl.noaa.gov/data/nnja_obs/ when using this data.

    Additional resources:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timedelta
        from earth2studio.data import NNJAObsConv

        ds = NNJAObsConv(time_tolerance=timedelta(hours=1))
        df = ds(datetime(2024, 1, 1, 0), ["t", "u", "v"])

        # GPS RO bending angle + 1D-Var retrieval profiles
        df_gps = ds(datetime(2024, 1, 1, 0), ["gps", "gps_t", "gps_q"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAObsConv"
    SCHEMA = _NNJA_CONV_SCHEMA
    MIN_DATE = datetime(1979, 1, 1)

    VALID_SOURCES = frozenset(["prepbufr", "convbufr", "prepbufr.acft_profiles"])

    def __init__(
        self,
        source: str = "prepbufr",
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Valid sources: {sorted(self.VALID_SOURCES)}"
            )
        self._source = source
        super().__init__(
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list:
        # Partition variables by lexicon route prefix:
        #   "prepbufr::..." -> conv/prepbufr/ tasks (PrepBUFR decoder)
        #   "gpsro::..."    -> gps/gpsro/ tasks (GPS RO BUFR decoder)
        prepbufr_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {}
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
        windows = (
            self._cycle_windows(time_list) if prepbufr_plan or gpsro_plan else {}
        )
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
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/conv/{self._source}/"
            f"{year_key}/{month_key}/{self._source}/"
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

    # PyArrow-type → numpy/pandas dtype for the always-nullable
    # numeric columns we add when a frame is missing them. Using a
    # typed empty column (instead of object-dtype ``None``) keeps
    # ``pd.concat`` from emitting "all-NA columns" FutureWarnings
    # when frames from different sub-archives are concatenated.
    _NULL_COLUMN_DTYPES: dict[str, type] = {
        "pres": np.float32,
        "elev": np.float32,
        "station_elev": np.float32,
        "lat": np.float32,
        "lon": np.float32,
        "observation": np.float32,
    }

    def _finalize_decoded_df(
        self,
        all_rows: list[dict[str, Any]],
        var_plan: dict[str, tuple[Any, Callable[[pd.DataFrame], pd.DataFrame]]],
        *,
        convert_pres_mb_to_pa: bool,
    ) -> pd.DataFrame:
        """Apply per-variable modifiers, normalize dtypes, project to schema."""
        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        result_frames: list[pd.DataFrame] = []
        for var, (_key, modifier) in var_plan.items():
            sub = df[df["variable"] == var]
            if sub.empty:
                continue
            result_frames.append(modifier(sub.copy()))
        if not result_frames:
            return pd.DataFrame()
        df = pd.concat(result_frames, ignore_index=True)

        # PrepBUFR levels carry POB in mb; the schema-level pressure
        # column should be in Pa for consistency with the lexicon's
        # ``pres`` observation conversion.
        if convert_pres_mb_to_pa and "pres" in df.columns:
            df["pres"] = (df["pres"].astype(np.float32) * 100.0).astype(np.float32)

        df["time"] = pd.to_datetime(df["time"])
        for name in self.SCHEMA.names:
            if name in df.columns:
                continue
            null_dtype = self._NULL_COLUMN_DTYPES.get(name)
            if null_dtype is not None:
                df[name] = np.full(len(df), np.nan, dtype=null_dtype)
            else:
                df[name] = pd.Series([None] * len(df), dtype=object)
        return df[list(self.SCHEMA.names)]

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
    def _decode_file(self, local_path: str, task) -> pd.DataFrame:
        if isinstance(task, _NNJAGpsRoTask):
            return self._decode_gpsro_file(local_path, task)
        return self._decode_prepbufr_file(local_path, task)

    # Threshold above which to spin up a multiprocess decoder. Pool
    # startup costs a few hundred ms; for small files (and tests) we
    # decode in-process.
    _PREPBUFR_POOL_MIN_MESSAGES = 32

    def _decode_prepbufr_file(
        self, local_path: str, task: _NNJAConvTask
    ) -> pd.DataFrame:
        """Decode a PrepBUFR cycle file into a DataFrame."""
        with open(local_path, "rb") as fh:
            file_data = fh.read()

        table_b, table_d, messages = _parse_prepbufr_messages(file_data)
        var_keys: list[tuple[str, str]] = [
            (var, plan[0]) for var, plan in task.var_plan.items()
        ]

        work_items: list[tuple[bytes, str]] = [
            (msg_bytes, _PREPBUFR_OBS_TYPES[data_cat])
            for msg_bytes, data_cat in messages
            if data_cat in _PREPBUFR_OBS_TYPES
        ]
        if not work_items:
            return pd.DataFrame()

        all_rows: list[dict[str, Any]] = []
        use_pool = (
            self._max_workers > 1
            and len(work_items) >= self._PREPBUFR_POOL_MIN_MESSAGES
        )
        logger.info(
            f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"messages={len(work_items)} (parsed {len(messages)}, "
            f"DX-table entries: B={len(table_b)} D={len(table_d)}) "
            f"strategy={'pool' if use_pool else 'sequential'}"
        )
        decode_t0 = time.perf_counter()
        if use_pool:
            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _decode_message_worker,
                        msg_bytes,
                        obs_class,
                        var_keys,
                        task.datetime_min,
                        task.datetime_max,
                    )
                    for msg_bytes, obs_class in work_items
                ]
                for future in futures:
                    try:
                        rows = future.result()
                        if rows:
                            all_rows.extend(rows)
                    except Exception:
                        logger.debug("NNJA worker failed to decode a BUFR message")
        else:
            with _silence_bufr_noise():
                _register_dx_tables(table_b, table_d)
                decoder = BufrDecoder()
                for msg_bytes, obs_class in work_items:
                    all_rows.extend(
                        _decode_message(
                            decoder,
                            msg_bytes,
                            obs_class,
                            var_keys,
                            task.datetime_min,
                            task.datetime_max,
                        )
                    )

        logger.info(
            f"[NNJAObsConv prepbufr] cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"decoded {len(all_rows):,} raw rows in "
            f"{time.perf_counter() - decode_t0:.1f}s"
        )
        return self._finalize_decoded_df(
            all_rows, task.var_plan, convert_pres_mb_to_pa=True
        )

    # Threshold above which to spin up a multiprocess gpsro decoder.
    # NNJA gpsro cycle files contain thousands of occultation messages,
    # each with hundreds of delayed-replication levels — pybufrkit's
    # pure-Python decoder is the bottleneck, so we parallelize.
    _GPSRO_POOL_MIN_MESSAGES = 16

    def _decode_gpsro_file(
        self, local_path: str, task: _NNJAGpsRoTask
    ) -> pd.DataFrame:
        """Decode a single NNJA gps/gpsro cycle BUFR file into a DataFrame.

        The NNJA gpsro files use NCEP-local BUFR descriptors that the
        standard ECMWF eccodes tables do not include, so we decode them
        with pybufrkit using the DX tables embedded at the start of the
        file. Each occultation profile is one message with several
        hundred delayed-replication levels; we parallelize across
        messages via ``ProcessPoolExecutor`` (same pattern as the
        PrepBUFR decoder) since pybufrkit's pure-Python bitstream
        reader is too slow for sequential decode.
        """
        with open(local_path, "rb") as fh:
            file_data = fh.read()

        table_b, table_d, messages = _parse_prepbufr_messages(file_data)
        if not messages:
            return pd.DataFrame()

        wanted_descrs: dict[int, str] = {
            desc_id: var for var, (desc_id, _mod) in task.var_plan.items()
        }

        all_rows: list[dict[str, Any]] = []
        use_pool = (
            self._max_workers > 1
            and len(messages) >= self._GPSRO_POOL_MIN_MESSAGES
        )
        logger.info(
            f"[NNJAObsConv gpsro]    cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"messages={len(messages)} "
            f"strategy={'pool' if use_pool else 'sequential'}"
        )
        decode_t0 = time.perf_counter()
        if use_pool:
            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _decode_gpsro_message_worker,
                        msg_bytes,
                        wanted_descrs,
                        task.datetime_min,
                        task.datetime_max,
                    )
                    for msg_bytes, _data_cat in messages
                ]
                for future in futures:
                    try:
                        rows = future.result()
                        if rows:
                            all_rows.extend(rows)
                    except Exception:
                        logger.debug("NNJA worker failed to decode a gpsro message")
        else:
            with _silence_bufr_noise():
                _register_dx_tables(table_b, table_d)
                decoder = BufrDecoder()
                for msg_bytes, _data_cat in messages:
                    try:
                        msg = decoder.process(msg_bytes)
                        n_subsets = msg.n_subsets.value
                    except Exception:  # noqa: S112
                        continue
                    if not n_subsets:
                        continue
                    td = msg.template_data.value
                    ddas = td.decoded_descriptors_all_subsets
                    dvas = td.decoded_values_all_subsets
                    for s_idx in range(n_subsets):
                        all_rows.extend(
                            _extract_gpsro_subset(
                                ddas[s_idx],
                                dvas[s_idx],
                                wanted_descrs,
                                task.datetime_min,
                                task.datetime_max,
                            )
                        )

        logger.info(
            f"[NNJAObsConv gpsro]    cycle={task.datetime_file:%Y-%m-%d %H:%MZ} "
            f"decoded {len(all_rows):,} raw rows in "
            f"{time.perf_counter() - decode_t0:.1f}s"
        )
        return self._finalize_decoded_df(
            all_rows, task.var_plan, convert_pres_mb_to_pa=False
        )
