# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode NCEP ``satwnd`` atmospheric-motion-vector BUFR dumps.

The SATWND dump is the raw AMV stream (``NC005xxx`` subsets) before PrepBUFR
merges and thins it. Each subset is one wind: producer, satellite, channel,
computation method, a height assignment as pressure, direction/speed, and one or
more quality-indicator blocks. Layouts differ by producer and era and NCEP-local
descriptor ids drift between table versions, so fields are resolved by mnemonic
from each file's embedded Table B.

A field repeated within a template (several ``PRLC`` height assignments, the
five-slot wind layouts) is read from its first slot that is populated anywhere
in the message. All subsets of a message share one template, so this picks the
same slot for every wind; a wind missing that slot stays missing rather than
being filled from a different slot. No report typing, quality control or
thinning is applied.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils_bufr import (
    get_worker_decoder,
    init_decode_worker,
)
from earth2studio.data.utils_bufr import (
    parse_prepbufr_messages as _parse_prepbufr_messages,
)
from earth2studio.data.utils_bufr import silence_bufr_noise as _silence_bufr_noise
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    _finalize_rows,
    empty_dataframe,
)

# Standard WMO descriptor ids used when a mnemonic is absent from the file's
# Table B. NCEP-local ids (SWQM, CMCM) drift between table versions and are
# always taken from the file when present.
_STANDARD_IDS: dict[str, int] = {
    "SAID": 1007,
    "GNAP": 1032,
    "GNAPS": 1044,
    "SWCM": 2023,
    "HAMD": 2163,
    "YEAR": 4001,
    "MNTH": 4002,
    "DAYS": 4003,
    "HOUR": 4004,
    "MINU": 4005,
    "SECO": 4006,
    "CLATH": 5001,
    "CLAT": 5002,
    "CLONH": 6001,
    "CLON": 6002,
    "PRLC": 7004,
    "SAZA": 7024,
    "WDIR": 11001,
    "WSPD": 11002,
    "PCCF": 33007,
    "SWQM": 33216,
}
_LOCAL_MNEMONICS = ("CMCM",)
_QUALITY_MNEMONICS = ("GNAP", "GNAPS", "PCCF")
_SCALAR_MNEMONICS = tuple(
    name
    for name in (*_STANDARD_IDS, *_LOCAL_MNEMONICS)
    if name not in _QUALITY_MNEMONICS
)
_DECODE_BATCH_SIZE = 64


def resolve_mnemonics(table_b: Mapping[int, tuple[Any, ...]]) -> dict[str, int]:
    """Mnemonic -> descriptor id from an embedded NCEP Table B, with WMO fallbacks."""
    ids = dict(_STANDARD_IDS)
    known = {*_STANDARD_IDS, *_LOCAL_MNEMONICS}
    for descriptor_id, entry in table_b.items():
        mnemonic = str(entry[0]).split()[0] if entry and entry[0] else ""
        if mnemonic in known:
            ids[mnemonic] = int(descriptor_id)
    return ids


def bufr_local_subcategory(message: bytes) -> int:
    """Section-1 local data subcategory (the ``NC005xxx`` subset number)."""
    edition = message[7]
    return message[17] if edition == 3 else message[20]


# ──────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────

_QUALITY_LIST = pa.list_(
    pa.struct([("application", pa.uint16()), ("confidence", pa.float32())])
)

NCEP_SATWND_PUBLIC_SCHEMA = pa.schema(
    [
        *NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
        pa.field(
            "satellite_id",
            pa.uint16(),
            nullable=True,
            metadata={"description": "BUFR SAID satellite identifier"},
        ),
        pa.field(
            "subset",
            pa.string(),
            nullable=True,
            metadata={"description": "NCEP dump subset, e.g. NC005030"},
        ),
        pa.field(
            "wind_method",
            pa.uint8(),
            nullable=True,
            metadata={
                "description": (
                    "SWCM satellite-derived wind computation method: 1 IR, 2 VIS, "
                    "3 WV cloud top, 4-7 WV clear-sky/deep-layer"
                )
            },
        ),
        pa.field(
            "wind_method_local",
            pa.uint8(),
            nullable=True,
            metadata={
                "description": (
                    "CMCM, the NCEP-local computation method carried by templates "
                    "that predate SWCM"
                )
            },
        ),
        pa.field(
            "height_method",
            pa.uint8(),
            nullable=True,
            metadata={"description": "HAMD height assignment method"},
        ),
        pa.field(
            "satellite_za",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "SAZA satellite zenith angle (deg), signed for cross-track "
                    "scanners"
                )
            },
        ),
        pa.field(
            "quality_indicators",
            _QUALITY_LIST,
            nullable=True,
            metadata={
                "description": (
                    "Percent confidence (PCCF) per generating application (GNAP, "
                    "0-01-032); code meanings are producer specific"
                )
            },
        ),
        pa.field(
            "amv_quality_indicators",
            _QUALITY_LIST,
            nullable=True,
            metadata={
                "description": (
                    "Percent confidence (PCCF) per AMV quality-indicator application "
                    "(GNAPS, 0-01-044)"
                )
            },
        ),
    ]
)

_WIND_COLUMNS = (
    "time",
    "lat",
    "lon",
    "pres",
    "quality",
    "satellite_id",
    "subset",
    "wind_method",
    "wind_method_local",
    "height_method",
    "satellite_za",
    "quality_indicators",
    "amv_quality_indicators",
    "u",
    "v",
)


# ──────────────────────────────────────────────────────────────────────────
# Message decode
# ──────────────────────────────────────────────────────────────────────────


def _populated(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scan_subset(
    descriptors: list[Any], values: list[Any], ids: Mapping[str, int]
) -> tuple[dict[str, list[Any]], dict[int, float], dict[int, float]]:
    """Every occurrence of each scalar mnemonic, plus GNAP/GNAPS -> PCCF maps."""
    names = {ids[name]: name for name in _SCALAR_MNEMONICS if name in ids}
    id_gnap, id_gnaps, id_pccf = ids["GNAP"], ids["GNAPS"], ids["PCCF"]
    occurrences: dict[str, list[Any]] = {}
    gnap: dict[int, float] = {}
    gnaps: dict[int, float] = {}
    pending: dict[int, float] | None = None
    code = 0
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if descriptor_id in (id_gnap, id_gnaps):
            pending = None
            if value is not None:
                pending = gnap if descriptor_id == id_gnap else gnaps
                code = int(value)
            continue
        if descriptor_id == id_pccf:
            if pending is not None and value is not None:
                pending.setdefault(code, float(value))
            pending = None
            continue
        name = names.get(descriptor_id)
        if name is not None:
            occurrences.setdefault(name, []).append(value)
    return occurrences, gnap, gnaps


def _first_populated_slots(scans: list[dict[str, list[Any]]]) -> dict[str, int]:
    """Per mnemonic, the first slot populated in any subset of the message."""
    slots: dict[str, int] = {}
    for name in {name for scan in scans for name in scan}:
        depth = max(len(scan.get(name, ())) for scan in scans)
        slots[name] = next(
            (
                slot
                for slot in range(depth)
                if any(
                    slot < len(scan.get(name, ())) and _populated(scan[name][slot])
                    for scan in scans
                )
            ),
            0,
        )
    return slots


def _quality_list(codes: Mapping[int, float]) -> list[dict[str, float]] | None:
    return [
        {"application": code, "confidence": value} for code, value in codes.items()
    ] or None


def _decode_satwnd_message(
    decoder: Any,
    message_bytes: bytes,
    ids: Mapping[str, int],
    dt_min: datetime,
    dt_max: datetime,
) -> dict[str, list[Any]]:
    """Columnar winds for one BUFR message; empty lists when nothing decodes."""
    columns: dict[str, list[Any]] = {name: [] for name in _WIND_COLUMNS}
    message = decoder.process(message_bytes)
    if not message.n_subsets.value:
        return columns
    subset = f"NC005{bufr_local_subcategory(message_bytes):03d}"
    template_data = message.template_data.value
    scans = [
        _scan_subset(descriptors, values, ids)
        for descriptors, values in zip(
            template_data.decoded_descriptors_all_subsets,
            template_data.decoded_values_all_subsets,
        )
    ]
    slots = _first_populated_slots([occurrences for occurrences, _, _ in scans])
    # High-accuracy CLATH/CLONH when the template carries them, else CLAT/CLON.
    lat_name = "CLATH" if "CLATH" in slots else "CLAT"
    lon_name = "CLONH" if "CLONH" in slots else "CLON"

    for occurrences, gnap, gnaps in scans:

        def get(name: str) -> Any:
            values = occurrences.get(name, ())
            slot = slots.get(name, 0)
            return values[slot] if slot < len(values) else None

        lat, lon = get(lat_name), get(lon_name)
        direction, speed = get("WDIR"), get("WSPD")
        year, month, day = get("YEAR"), get("MNTH"), get("DAYS")
        if not all(
            _populated(v) for v in (lat, lon, direction, speed, year, month, day)
        ):
            continue
        second = get("SECO")
        try:
            obs_time = datetime(
                int(year),
                int(month),
                int(day),
                int(get("HOUR") or 0),
                int(get("MINU") or 0),
            ) + timedelta(seconds=float(second) if _populated(second) else 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if obs_time < dt_min or obs_time > dt_max:
            continue
        radians = math.radians(float(direction))
        pressure = get("PRLC")
        columns["time"].append(obs_time)
        columns["lat"].append(float(lat))
        columns["lon"].append(float(lon) % 360.0)
        columns["pres"].append(float(pressure) if _populated(pressure) else math.nan)
        columns["quality"].append(get("SWQM"))
        columns["satellite_id"].append(get("SAID"))
        columns["subset"].append(subset)
        columns["wind_method"].append(get("SWCM"))
        columns["wind_method_local"].append(get("CMCM"))
        columns["height_method"].append(get("HAMD"))
        zenith = get("SAZA")
        columns["satellite_za"].append(
            float(zenith) if _populated(zenith) else math.nan
        )
        columns["quality_indicators"].append(_quality_list(gnap))
        columns["amv_quality_indicators"].append(_quality_list(gnaps))
        columns["u"].append(-float(speed) * math.sin(radians))
        columns["v"].append(-float(speed) * math.cos(radians))
    return columns


def _decode_message_batch(
    arguments: tuple[list[bytes], Mapping[str, int], datetime, datetime],
) -> tuple[dict[str, list[Any]], int]:
    messages, ids, dt_min, dt_max = arguments
    columns: dict[str, list[Any]] = {name: [] for name in _WIND_COLUMNS}
    failures = 0
    with _silence_bufr_noise():
        for message_bytes in messages:
            try:
                decoded = _decode_satwnd_message(
                    get_worker_decoder(), message_bytes, ids, dt_min, dt_max
                )
            except Exception:  # noqa: BLE001 - one corrupt message is skipped
                failures += 1
                continue
            for name, values in decoded.items():
                columns[name].extend(values)
    return columns, failures


def _winds_to_rows(
    columns: Mapping[str, list[Any]], wanted: Mapping[str, str]
) -> pd.DataFrame:
    """One row per (wind, requested component), components in ``wanted`` order."""
    base = pd.DataFrame(
        {name: columns[name] for name in _WIND_COLUMNS if name not in ("u", "v")}
    )
    base["class"] = "SATWND"
    frames = []
    for component, variable in wanted.items():
        frame = base.copy()
        frame["observation"] = np.asarray(columns[component], dtype=np.float32)
        frame["variable"] = variable
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def decode_satwnd(
    path: str,
    plan: Mapping[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
    dt_min: datetime,
    dt_max: datetime,
    decode_workers: int = 8,
) -> pd.DataFrame:
    """Decode an NCEP ``satwnd`` AMV BUFR dump into a DataFrame.

    Parameters
    ----------
    path : str
        Local path to the ``gdas.*.satwnd.tm00.bufr_d`` file.
    plan : Mapping
        Variable decode plan ``{variable: (component, modifier)}`` where
        component is ``"u"`` or ``"v"``.
    dt_min, dt_max : datetime
        Time window for observation filtering.
    decode_workers : int
        Number of parallel decode processes (1 disables multiprocessing).

    Returns
    -------
    pd.DataFrame
        One row per (wind, component) in :data:`NCEP_SATWND_PUBLIC_SCHEMA`, in
        source-message order within each component.
    """
    started = time.perf_counter()
    wanted = {component: variable for variable, (component, _) in plan.items()}
    modifiers = {variable: modifier for variable, (_, modifier) in plan.items()}
    with open(path, "rb") as file:
        file_data = file.read()
    table_b, table_d, messages = _parse_prepbufr_messages(file_data, silence_noise=True)
    if not messages:
        return empty_dataframe(NCEP_SATWND_PUBLIC_SCHEMA)
    ids = resolve_mnemonics(table_b)
    message_bytes = [message for message, _data_category in messages]
    arguments: Iterable[tuple[list[bytes], Mapping[str, int], datetime, datetime]] = (
        (message_bytes[i : i + _DECODE_BATCH_SIZE], ids, dt_min, dt_max)
        for i in range(0, len(message_bytes), _DECODE_BATCH_SIZE)
    )
    n_batches = math.ceil(len(message_bytes) / _DECODE_BATCH_SIZE)
    columns: dict[str, list[Any]] = {name: [] for name in _WIND_COLUMNS}
    failures = 0
    workers = min(max(1, decode_workers), n_batches)
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_decode_worker,
            initargs=(table_b, table_d),
        ) as pool:
            # Executor.map keeps source-message order.
            results = pool.map(_decode_message_batch, arguments)
            for batch, batch_failures in results:
                for name, values in batch.items():
                    columns[name].extend(values)
                failures += batch_failures
    else:
        init_decode_worker(table_b, table_d)
        for argument in arguments:
            batch, batch_failures = _decode_message_batch(argument)
            for name, values in batch.items():
                columns[name].extend(values)
            failures += batch_failures
    logger.debug(
        f"Decoded {len(columns['time']):,} SATWND winds from {len(message_bytes):,} "
        f"messages in {time.perf_counter() - started:.1f}s"
    )
    if failures:
        logger.warning(
            f"{path}: skipped {failures} of {len(message_bytes)} undecodable "
            "SATWND messages"
        )
    if not columns["time"]:
        return empty_dataframe(NCEP_SATWND_PUBLIC_SCHEMA)
    return _finalize_rows(
        _winds_to_rows(columns, wanted),
        modifiers,
        convert_pres_mb_to_pa=False,
        schema=NCEP_SATWND_PUBLIC_SCHEMA,
    )
