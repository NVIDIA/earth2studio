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

"""Shared earth2bufr helpers for the NCEP BUFR sources.

:mod:`earth2studio.data.gdas` and :mod:`earth2studio.data.nnja` decode NCEP BUFR
with earth2bufr, which returns Arrow record batches. The helpers here turn decoded
columns into numpy arrays with validity masks, build timestamps from BUFR time
parts, and assemble Arrow columns without per-value Python work.
"""

from __future__ import annotations

import functools
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, TypeVar

import numpy as np
import pyarrow as pa
from loguru import logger

from earth2studio.utils.imports import OptionalDependencyFailure

# Shared optional-dependency key for earth2bufr. NCEP sources decode through
# this module, so their public classes check this key.
BUFR_DEPENDENCY_KEY = "bufr"

try:
    import earth2bufr
except ImportError:
    OptionalDependencyFailure("data", BUFR_DEPENDENCY_KEY)
    earth2bufr = None  # type: ignore[assignment]

# PrepBUFR Section 1 data category -> message family. Keys are the post-2007
# category numbers, for reference; selection is by family name.
PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper-air: RAOB, PIBAL, RECCO, dropsonde
    103: "AIRCAR",  # Aircraft: MDCRS ACARS
    104: "AIRCFT",  # Aircraft: AIREP, PIREP, AMDAR, TAMDAR
    105: "SATWND",  # Satellite-derived winds
    106: "PROFLR",  # Wind profiler / SODAR reports
    107: "VADWND",  # VAD/NEXRAD winds
    108: "SATEMP",  # POES/TOVS sounding/retrieval/radiance data
    109: "ADPSFC",  # Surface land: SYNOP/METAR
    110: "SFCSHP",  # Surface marine: ship/buoy/C-MAN/tide gauge
    111: "SFCBOG",  # Mean sea-level pressure bogus reports
    112: "SPSSMI",  # DMSP SSM/I retrieval products
    113: "SYNDAT",  # Synthetic bogus data
    114: "ERS1DA",  # ERS scatterometer winds
    115: "GOESND",  # GOES sounding/retrieval/radiance data
    116: "QKSWND",  # QuikSCAT scatterometer winds
    117: "MSONET",  # Mesonet surface reports
    118: "GPSIPW",  # GPS integrated precipitable water / zenith delay
    119: "RASSDA",  # RASS virtual temperature
    120: "WDSATR",  # WindSat scatterometer winds
    121: "ASCATW",  # ASCAT scatterometer winds
}


_T = TypeVar("_T")
_R = TypeVar("_R")

# Fastest decode-pool size for the NNJA/GDAS pipeline on 16-112 cores
_DEFAULT_DECODE_THREADS = 8


@functools.cache
def default_decode_threads() -> None:
    """Size earth2bufr's decode pool once, unless ``EARTH2BUFR_NUM_THREADS`` is set."""
    if "EARTH2BUFR_NUM_THREADS" in os.environ:
        return
    try:
        cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - platform dependent
        cpus = os.cpu_count() or 1
    earth2bufr.set_num_threads(min(_DEFAULT_DECODE_THREADS, cpus))


def open_bufr(path: str) -> Any:
    """Open a BUFR file with earth2bufr."""
    return earth2bufr.open_bufr(path)


def iter_batches(source: Any, path: str, **options: Any) -> Iterator[pa.RecordBatch]:
    """``source.iter_batches(**options)``, skipping messages truncated before their
    template ends and logging a warning with how many were skipped."""
    batches = source.iter_batches(skip_truncated_messages=True, **options)
    yield from batches
    failures = batches.decode_failures
    if failures:
        logger.warning(
            f"{path}: skipped {len(failures)} truncated BUFR messages "
            f"(first: message {failures[0].source_message_index}, "
            f"{failures[0].reason})"
        )


def list_value_filter(field: str, key: str, values: Iterable[int]) -> Any:
    """earth2bufr filter keeping list entries whose ``key`` is in ``values``."""
    return earth2bufr.ListValueFilter(
        field=field, key=key, values=tuple(sorted(values))
    )


def numpy_with_mask(array: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Values and validity of a numeric column; values are arbitrary at null slots."""
    valid = (
        np.asarray(array.is_valid())
        if array.null_count
        else np.ones(len(array), dtype=bool)
    )
    if len(array) and (
        pa.types.is_floating(array.type) or pa.types.is_integer(array.type)
    ):
        values = np.frombuffer(array.buffers()[1], dtype=array.type.to_pandas_dtype())
        return values[array.offset : array.offset + len(array)], valid
    return array.fill_null(0).to_numpy(zero_copy_only=False), valid


def float_values(array: pa.Array) -> np.ndarray:
    """Numeric column as float64 with NaN at null slots; a float64 column without nulls
    is returned as a read-only view of the Arrow buffer."""
    values, valid = numpy_with_mask(array)
    if valid.all():
        return values.astype(np.float64, copy=False)
    return np.where(valid, values, np.nan)


def physical_values(
    values: pa.StructArray,
    name: str,
    index: np.ndarray | None = None,
    dtype: type = np.float64,
) -> np.ndarray:
    """Child field of a struct in physical units as ``dtype``, NaN where missing, at the
    ``index`` entries if given.

    Raw-code fields (``raw_code_paths``) are converted as
    ``(code + reference) / 10**scale`` from the field metadata, in float64 per element.
    """
    codes, valid = numpy_with_mask(values.field(name))
    if index is not None:
        codes, valid = codes[index], valid[index]
    meta = values.type.field(name).metadata or {}
    raw = meta.get(b"representation") == b"bufr_raw_code"
    reference = int(meta[b"reference"]) if raw else 0
    scale = int(meta[b"scale"]) if raw else 0
    if reference:
        codes = codes.astype(np.int64) + reference
    out = np.full(len(codes), np.nan, dtype=dtype)
    if scale:
        divisor = 1.0 * 10**scale
        np.divide(
            codes, divisor, out=out, where=valid, dtype=np.float64, casting="unsafe"
        )
    else:
        np.copyto(out, codes, casting="unsafe", where=valid)
    return out


def masked(
    values: np.ndarray, valid: np.ndarray | None, arrow_type: pa.DataType
) -> pa.Array:
    """Arrow array of ``values`` with nulls where ``valid`` is False."""
    if valid is None or valid.all():
        return pa.array(values, type=arrow_type)
    return pa.array(values, type=arrow_type, mask=~valid)


def float_with_nulls(values: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    """Float array with NaN slots as nulls."""
    return masked(values, ~np.isnan(values), arrow_type)


def uint_with_nulls(values: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    """Unsigned integer array from floats with NaN slots as nulls."""
    valid = np.isfinite(values)
    ints = np.where(valid, values, 0).astype(arrow_type.to_pandas_dtype())
    return masked(ints, valid, arrow_type)


def null_array(arrow_type: pa.DataType, n: int) -> pa.Array:
    """All-null fixed-width array whose validity and data share one zero buffer."""
    nbytes = max(n * max(arrow_type.bit_width, 1) // 8, (n + 7) // 8, 1)
    zeros = pa.py_buffer(np.zeros(nbytes, dtype=np.uint8))
    return pa.Array.from_buffers(arrow_type, n, [zeros, zeros], null_count=n)


def dictionary_column(codes: np.ndarray, names: list[str]) -> pa.Array:
    """``dictionary<int8, string>`` column from codes into ``names``."""
    return pa.DictionaryArray.from_arrays(
        pa.array(codes, type=pa.int8()), pa.array(names, type=pa.utf8())
    )


def constant_dictionary(value: str, n: int) -> pa.Array:
    """``dictionary<int8, string>`` column holding one value."""
    return dictionary_column(np.zeros(n, dtype=np.int8), [value])


def timestamps_ns(values: np.ndarray, valid: np.ndarray | None = None) -> pa.Array:
    """``timestamp[ns]`` column over int64 nanoseconds since the epoch."""
    values = np.ascontiguousarray(values, dtype=np.int64)
    validity = None if valid is None or valid.all() else pa.array(valid).buffers()[1]
    return pa.Array.from_buffers(
        pa.timestamp("ns"), len(values), [validity, pa.py_buffer(values)]
    )


def constant_timestamp(value: datetime, n: int) -> pa.Array:
    """``timestamp[ns]`` column holding one value."""
    stamp = np.datetime64(value, "ns").astype(np.int64)
    return timestamps_ns(np.full(n, stamp, dtype=np.int64))


def datetime_ns(value: datetime) -> int:
    """Nanoseconds since the epoch of a naive UTC datetime."""
    return int(np.datetime64(value, "ns").astype(np.int64))


_DAYS_IN_MONTH = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def civil_time_ns(
    year: np.ndarray,
    month: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    minute: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nanoseconds since the epoch from BUFR time parts given as floats (NaN = missing).

    A missing minute or second counts as zero. The result is valid where the year,
    month and day form a calendar date, the hour is present and in range, and the
    minute is in range.
    """
    finite = np.isfinite(year) & np.isfinite(month) & np.isfinite(day)
    y = np.where(finite, year, 1970).astype(np.int64)
    m = np.where(finite, month, 1).astype(np.int64)
    d = np.where(finite, day, 1).astype(np.int64)
    hh = np.nan_to_num(hour).astype(np.int64)
    mm = np.nan_to_num(minute).astype(np.int64)
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    month_ok = (m >= 1) & (m <= 12)
    days_in_month = _DAYS_IN_MONTH[np.where(month_ok, m, 1)] + ((m == 2) & leap)
    valid = (
        finite
        & np.isfinite(hour)
        & month_ok
        & (y >= 1)
        & (y <= 9999)
        & (d >= 1)
        & (d <= days_in_month)
        & (hh >= 0)
        & (hh <= 23)
        & (mm >= 0)
        & (mm <= 59)
    )
    # Days from the civil date (proleptic Gregorian)
    ya = y - (m <= 2)
    era = np.floor_divide(ya, 400)
    yoe = ya - era * 400
    doy = (153 * ((m + 9) % 12) + 2) // 5 + d - 1
    days = era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468
    minutes = days * 1440 + hh * 60 + mm
    seconds_ns = np.rint(np.nan_to_num(second) * 1e9).astype(np.int64)
    return np.where(valid, minutes * 60_000_000_000 + seconds_ns, 0), valid


def map_ordered(
    fn: Callable[[_T], _R], items: Iterable[_T], workers: int
) -> Iterator[_R]:
    """``fn`` over ``items`` on a thread pool, in input order, at most ``2 * workers`` in flight."""
    if workers <= 1:
        yield from (fn(item) for item in items)
        return
    pending: deque = deque()
    with ThreadPoolExecutor(workers) as pool:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
