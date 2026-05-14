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

"""Shared PrepBUFR parsing utilities for NCEP-format BUFR files.

This module provides common helpers for decoding NCEP PrepBUFR files
(DX table extraction, message splitting, and pybufrkit table
registration) used by :mod:`earth2studio.data.gdas` and
:mod:`earth2studio.data.nnja`.
"""

from __future__ import annotations

import contextlib
import os
import struct
import sys
from collections.abc import Iterator
from typing import Any

from loguru import logger

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
)

try:
    from pybufrkit.decoder import Decoder as BufrDecoder
    from pybufrkit.tables import TableGroupCacheManager
except ImportError:
    OptionalDependencyFailure("data")
    BufrDecoder = None  # type: ignore[assignment,misc]
    TableGroupCacheManager = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────
# PrepBUFR descriptor ID constants
# ─────────────────────────────────────────────────────────────────────

# Header field descriptors (per-subset, scalar)
HDR_SID = 1194  # Station ID (CCITT IA5, 64 bits)
HDR_XOB = 6240  # Longitude (degrees east)
HDR_YOB = 5002  # Latitude (degrees north)
HDR_DHR = 4215  # Obs time minus cycle time (hours)
HDR_ELV = 10199  # Station elevation (m)
HDR_TYP = 55007  # Report type code
HDR_T29 = 55008  # Data dump report type code

# Observation field descriptors
OBS_CAT = 8193  # Observation category code
OBS_POB = 7245  # Pressure observation (MB)
OBS_ZOB = 10007  # Height observation (m)
OBS_TOB = 12245  # Temperature observation (DEG C)
OBS_QOB = 13245  # Specific humidity (MG/KG)
OBS_UOB = 11003  # U-wind component (M/S)
OBS_VOB = 11004  # V-wind component (M/S)

# Quality mark descriptors
OBS_PQM = 7246  # Pressure quality mark
OBS_TQM = 12246  # Temperature quality mark
OBS_QQM = 13246  # Moisture quality mark
OBS_WQM = 11240  # Wind quality mark

# Set of all header descriptor IDs
HEADER_DESCR_IDS: set[int] = {
    HDR_SID,
    HDR_XOB,
    HDR_YOB,
    HDR_DHR,
    HDR_ELV,
    HDR_TYP,
    HDR_T29,
}

# Set of core observation-level descriptor IDs (obs + quality marks)
OBSERVATION_DESCR_IDS: set[int] = {
    OBS_POB,
    OBS_PQM,
    OBS_ZOB,
    OBS_TOB,
    OBS_TQM,
    OBS_QOB,
    OBS_QQM,
    OBS_UOB,
    OBS_VOB,
    OBS_WQM,
}

# Lexicon mnemonic -> descriptor ID for non-wind observation fields
MNEMONIC_TO_DESCR: dict[str, int] = {
    "TOB": OBS_TOB,
    "QOB": OBS_QOB,
    "POB": OBS_POB,
    "ZOB": OBS_ZOB,
    "UOB": OBS_UOB,
    "VOB": OBS_VOB,
}

# PrepBUFR section-1 dataCategory -> NCEP message-type class string
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

# Per-variable quality mark mapping: observation descriptor -> QM descriptor
OBS_QUALITY_MAP: dict[int, int] = {
    OBS_POB: OBS_PQM,
    OBS_TOB: OBS_TQM,
    OBS_QOB: OBS_QQM,
    OBS_UOB: OBS_WQM,
    OBS_VOB: OBS_WQM,
}


# ─────────────────────────────────────────────────────────────────────
# Silence noisy C-level stderr from pybufrkit
# ─────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def silence_bufr_noise() -> Iterator[None]:
    """Suppress chatty C-library stderr from pybufrkit.

    pybufrkit writes informational messages straight to file
    descriptor 2 (e.g. ``Cannot find sub-centre 3 nor valid default``)
    when the file uses NCEP-local descriptors. We rely on the DX
    tables embedded in each file to decode those correctly, so
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


# ─────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────


def safe_int(v: Any) -> int:
    """Convert a value to int, handling bytes, strings, None, and floats."""
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


def _str(v: Any) -> str:
    """Convert a value to a stripped ASCII string."""
    if isinstance(v, bytes):
        return v.decode("ascii", errors="replace").strip()
    if v is None:
        return ""
    return str(v).strip()


def fxy_to_id(f: Any, x: Any, y: Any) -> int:
    """Convert F, X, Y fields to an integer BUFR descriptor ID."""
    return safe_int(f) * 100000 + safe_int(x) * 1000 + safe_int(y)


# ─────────────────────────────────────────────────────────────────────
# DX table extraction
# ─────────────────────────────────────────────────────────────────────


def extract_dx_tables(
    flat: list[Any],
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Extract NCEP Table B and D entries from a DX-message subset.

    DX table messages (dataCategory=11) contain embedded NCEP-local BUFR
    descriptor definitions.  The flat decoded values follow the layout
    produced by the NCEP BUFRLIB DX table encoding:

    - n_table_a, [table_a entries (3 fields each)...]
    - n_table_b, [table_b entries (11 fields each)...]
    - n_table_d, [table_d entries (variable-length)...]

    Each Table B entry has 11 fields:
        F, X, Y, mnemonic(32), desc_cont(32), unit(24),
        sign_scale(1), scale(3), sign_ref(1), reference(10), width(3)

    Each Table D entry has:
        F, X, Y, mnemonic(64), n_members,
        [member F, X, Y, ...]

    Parameters
    ----------
    flat : list
        Decoded values from a single DX message subset.
    table_b : dict
        Accumulator dict to update with Table B entries (pybufrkit format).
    table_d : dict
        Accumulator dict to update with Table D entries.
    """
    n = len(flat)
    idx = 0
    if idx >= n:
        return
    n_a = safe_int(flat[idx])
    idx += 1
    idx += n_a * 3
    if idx >= n:
        return

    # Table B
    n_b = safe_int(flat[idx])
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

        desc_id = fxy_to_id(f_v, x_v, y_v)
        if desc_id == 0:
            continue
        scale = safe_int(scale_s)
        if sign_scale == "-":
            scale = -scale
        reference = safe_int(ref_s)
        if sign_ref == "-":
            reference = -reference
        width = safe_int(width_s)
        table_b[desc_id] = (
            mnemonic,
            unit,
            scale,
            reference,
            width,
            unit,  # crex_unit = same as unit
            scale,  # crex_scale = same as scale
            max(1, (width + 3) // 4),  # crex_nchars approximation
        )

    # Table D
    if idx >= n:
        return
    n_d = safe_int(flat[idx])
    idx += 1
    for _ in range(n_d):
        if idx + 3 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        seq_mnemonic = _str(flat[idx + 3])
        idx += 4
        seq_id = fxy_to_id(f_v, x_v, y_v)
        if seq_id == 0:
            continue
        if idx >= n:
            return
        n_members = safe_int(flat[idx])
        idx += 1
        members: list[str] = []
        for _ in range(n_members):
            if idx >= n:
                break
            members.append(_str(flat[idx]))
            idx += 1
        if members:
            table_d[seq_id] = (seq_mnemonic, members)


# ─────────────────────────────────────────────────────────────────────
# Table registration & decoder creation
# ─────────────────────────────────────────────────────────────────────


def register_dx_tables(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Reset pybufrkit's table cache and (re-)register NCEP DX tables.

    Parameters
    ----------
    table_b : dict
        NCEP Table B entries in pybufrkit format.
    table_d : dict
        NCEP Table D entries.
    """
    TableGroupCacheManager.clear_extra_entries()
    try:
        TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
    except AttributeError as exc:
        logger.warning(
            f"pybufrkit TableGroupCacheManager._TABLE_GROUP_CACHE not available "
            f"({exc}); skipping cache invalidation"
        )
    if table_b or table_d:
        TableGroupCacheManager.add_extra_entries(table_b, table_d)


def create_decoder(
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
    register_dx_tables(table_b, table_d)
    return BufrDecoder()


# ─────────────────────────────────────────────────────────────────────
# PrepBUFR message parsing
# ─────────────────────────────────────────────────────────────────────


def parse_prepbufr_messages(
    file_data: bytes,
    silence_noise: bool = True,
) -> tuple[
    dict[int, tuple[Any, ...]],
    dict[int, tuple[Any, ...]],
    list[tuple[bytes, int]],
]:
    """Split a PrepBUFR byte stream into messages and extract DX tables.

    The first several messages in a PrepBUFR file are DX-table messages
    (dataCategory=11) containing NCEP-local BUFR Table B and Table D
    definitions needed to decode subsequent data messages.

    Parameters
    ----------
    file_data : bytes
        Entire PrepBUFR file contents.
    silence_noise : bool
        If True (default), suppress noisy C-level stderr from pybufrkit
        during DX table decoding.

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
        if msg_len < 8:
            pos = idx + 4
            continue
        msg_bytes = file_data[idx : idx + msg_len]

        # BUFR ed3/4: section-0 = 8 bytes, section-1 octet-9 (offset 16)
        # is dataCategory
        data_cat = file_data[idx + 16] if idx + 16 < len(file_data) else 0
        if data_cat == 11:
            dx_messages.append(msg_bytes)
        else:
            data_messages.append((msg_bytes, data_cat))
        pos = idx + msg_len

    # Decode DX table messages using pybufrkit (they use standard descriptors)
    if dx_messages:
        ctx = silence_bufr_noise() if silence_noise else contextlib.nullcontext()
        with ctx:
            try:
                dx_decoder = BufrDecoder()
                for dx_bytes in dx_messages:
                    try:
                        dx_msg = dx_decoder.process(dx_bytes)
                    except Exception:  # noqa: S112
                        logger.debug("Skipping unparseable DX-table message")
                        continue
                    td = dx_msg.template_data.value
                    dvas = td.decoded_values_all_subsets
                    if not dvas:
                        continue
                    extract_dx_tables(dvas[0], table_b, table_d)
            except Exception as e:
                logger.warning(f"Failed to extract DX tables: {e}")

    return table_b, table_d, data_messages


# ─────────────────────────────────────────────────────────────────────
# Process-pool worker initialization
# ─────────────────────────────────────────────────────────────────────

# Module-level decoder for worker processes, set by init_decode_worker.
_worker_decoder: Any = None


def init_decode_worker(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Initializer for process pool workers.

    Registers NCEP-local descriptor tables with pybufrkit in each
    worker process and creates a reusable decoder instance stored
    as a module-level global (``_worker_decoder``).

    Parameters
    ----------
    table_b : dict
        NCEP Table B entries in pybufrkit format.
    table_d : dict
        NCEP Table D entries.
    """
    global _worker_decoder  # noqa: PLW0603
    register_dx_tables(table_b, table_d)
    _worker_decoder = BufrDecoder()


def get_worker_decoder() -> Any:
    """Return the module-level worker decoder instance.

    Returns
    -------
    pybufrkit.decoder.Decoder
        The decoder created by :func:`init_decode_worker`.
    """
    return _worker_decoder
