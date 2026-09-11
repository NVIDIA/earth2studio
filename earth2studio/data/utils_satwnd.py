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
computation method, a final height assignment as pressure, direction/speed,
and one or more quality-indicator blocks. Layouts differ by producer and era
(GOES legacy, GOES-R, EUMETSAT, JMA, MODIS/AVHRR/VIIRS, LEO-GEO) and NCEP-local
descriptor ids drift between table versions, so fields are resolved by
mnemonic from each file's embedded Table B and read as first occurrence,
which is what GSI ``ufbint`` does.

GSI report types (240-260) are not in the file. GSI derives them from
``(subset, satellite id, computation method)`` through the ``sattabin`` table
in ``read_satwnd.f90``; :data:`SATWND_TYPE_TABLE` is a port of that table at
NOAA-EMC/GSI ``860d1374``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils_bufr import create_decoder as _create_decoder
from earth2studio.data.utils_bufr import (
    parse_prepbufr_messages as _parse_prepbufr_messages,
)
from earth2studio.data.utils_bufr import silence_bufr_noise as _silence_bufr_noise
from earth2studio.data.utils_ncep import (
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    _finalize_rows,
    empty_dataframe,
)

# ──────────────────────────────────────────────────────────────────────────
# GSI read_satwnd.f90 sattabin: (subset, SAID, SWCM) -> (report type, case)
# ──────────────────────────────────────────────────────────────────────────

_GOES = (731, 732, 733, 734, 735, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 270, 271, 272, 273)  # fmt: skip
_INSAT = (430, 431, 432, 450, 451, 452, 410, 470)
_HIMAWARI = (153, 154, 150, 151, 152, 171, 172, 173, 174, 253)
_METEOSAT = (58, 59, 50, 51, 52, 53, 54, 55, 56, 57, 70, 71)
_MODIS = (783, 784)
_LEOGEO = (854,)
_AVHRR = (3, 4, 5, 206, 207, 208, 209, 223, 225, 226)
_VIIRS = (224, 225, 226)

# (subset number, platforms, computation methods, report type, GSI case).
# GSI case -1 means the type is known but GSI never processes it (INSAT 256).
_TABLE_ROWS: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int, int], ...] = (
    (1, _GOES, (1,), 245, 6),
    (2, _GOES, (2,), 251, 6),
    (3, _GOES, (3,), 246, 6),
    (4, _GOES, (4,), 245, 6),
    (5, _GOES, (1,), 245, 6),
    (6, _GOES, (3,), 251, 6),
    (8, _GOES, (2,), 246, 6),
    (9, _GOES, (4,), 245, 6),
    (10, _GOES, (1,), 245, 7),
    (11, _GOES, (3,), 246, 7),
    (12, _GOES, (2,), 251, 7),
    (13, _GOES, (4,), 245, 7),
    (15, _GOES, (1,), 245, 7),
    (16, _GOES, (3,), 246, 7),
    (17, _GOES, (2,), 251, 7),
    (19, _GOES, (1,), 240, 11),
    (21, _INSAT, (1,), 256, -1),
    (22, _INSAT, (2,), 256, -1),
    (23, _INSAT, (3,), 256, -1),
    (24, _INSAT, (1,), 256, -1),
    (25, _INSAT, (2,), 256, -1),
    (26, _INSAT, (3,), 256, -1),
    (30, _GOES, (1,), 245, 15),
    (31, _GOES, (4, 5), 247, 19),
    (32, _GOES, (2,), 251, 17),
    (34, _GOES, (3,), 246, 18),
    (39, _GOES, (1,), 240, 16),
    (41, _HIMAWARI, (1,), 252, 3),
    (42, _HIMAWARI, (2,), 242, 3),
    (43, _HIMAWARI, (3, 4, 5), 250, 3),
    (44, _HIMAWARI, (1,), 252, 4),
    (45, _HIMAWARI, (2,), 242, 4),
    (46, _HIMAWARI, (3, 4, 5), 250, 4),
    (47, _HIMAWARI, (1,), 253, 5),
    (48, _HIMAWARI, (2,), 242, 5),
    (49, _HIMAWARI, (3, 4, 5), 250, 5),
    (52, _GOES, (1,), 245, 15),
    (53, _GOES, (4,), 247, 19),
    (54, _GOES, (2,), 251, 17),
    (55, _GOES, (3,), 246, 18),
    (56, _GOES, (1,), 240, 16),
    (61, _METEOSAT, (1,), 253, 0),
    (62, _METEOSAT, (2,), 243, 0),
    (63, _METEOSAT, (3, 4, 5), 254, 0),
    (64, _METEOSAT, (1,), 253, 1),
    (65, _METEOSAT, (2,), 243, 1),
    (66, _METEOSAT, (3, 4, 5), 254, 1),
    (67, _METEOSAT, (1,), 253, 2),
    (68, _METEOSAT, (2,), 243, 2),
    (69, _METEOSAT, (3, 4, 5), 254, 2),
    (70, _MODIS, (1,), 257, 8),
    (71, _MODIS, (3,), 258, 8),
    (71, _MODIS, (4, 5), 259, 8),
    (72, _LEOGEO, (1,), 255, 12),
    (80, _AVHRR, (1,), 244, 9),
    (81, _AVHRR, (1,), 244, 10),
    (90, _VIIRS, (1,), 260, 13),
    (91, _VIIRS, (1,), 260, 14),
    (99, _GOES, (1, 2, 3, 4, 5, 6), 241, 20),
    (99, _HIMAWARI, (1,), 241, 20),
    (99, _METEOSAT, (1,), 241, 20),
    (99, _VIIRS, (1,), 241, 20),
)


def _build_type_table() -> dict[tuple[int, int, int], tuple[int, int]]:
    table: dict[tuple[int, int, int], tuple[int, int]] = {}
    for subset, platforms, methods, report_type, case in _TABLE_ROWS:
        for said in platforms:
            for swcm in methods:
                table[(subset, said, swcm)] = (report_type, case)
    return table


SATWND_TYPE_TABLE: dict[tuple[int, int, int], tuple[int, int]] = _build_type_table()
"""``(NC005 subset number, SAID, SWCM) -> (GSI report type, GSI processing case)``."""

# GSI c_station_id prefixes by SWCM (satellite derived wind computation method).
SWCM_NAMES: dict[int, str] = {1: "IR", 2: "VI", 3: "CT", 4: "DL", 5: "D5", 6: "D6", 7: "D7"}  # fmt: skip

# Quality-indicator resolution per GSI processing case. GNAP-triplet families
# key on GNAP (0-01-032); AMVQIC families key on GNAPS (0-01-044). Values are
# ``(qi_without_forecast, qi_with_forecast, expected_error)`` codes.
_GNAP_QI_CODES: dict[int, tuple[int, int, int]] = {
    1: (2, 1, 3),  # EUMETSAT
    4: (102, 101, 103),  # JMA
    7: (1, 3, 4),  # NESDIS legacy
    8: (1, 3, 4),  # MODIS
    9: (1, 3, 4),  # AVHRR
    11: (1, 3, 4),  # NESDIS shortwave IR
    13: (1, 3, 4),  # VIIRS
}
_AMVQIC_CASES = frozenset({2, 5, 10, 14, 15, 16, 17, 18, 19, 20})
# GNAPS 5 is "QI without forecast" for every AMVQIC producer, which is also the
# occurrence GSI reads positionally (amvqic(2,2)). GSI's expected error is the
# fourth occurrence: GNAPS 7 in GOES-R/NESDIS products, GNAPS 2 in the
# EUMETSAT/JMA 310077 layout.
_AMVQIC_QIFN, _AMVQIC_QIFY, _AMVQIC_EE = 5, 6, (7, 2)

# Standard WMO descriptor ids used when a mnemonic is absent from the file's
# Table B. NCEP-local ids (SWQM, EHAM) drift between table versions and are
# always taken from the file when present.
_STANDARD_IDS: dict[str, int] = {
    "SAID": 1007,
    "GNAP": 1032,
    "GNAPS": 1044,
    "SWCM": 2023,
    "EHAM": 2162,
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


def resolve_mnemonics(table_b: Mapping[int, tuple[Any, ...]]) -> dict[str, int]:
    """Mnemonic -> descriptor id from an embedded NCEP Table B, with WMO fallbacks."""
    ids = dict(_STANDARD_IDS)
    for descriptor_id, entry in table_b.items():
        mnemonic = str(entry[0]).split()[0] if entry and entry[0] else ""
        if mnemonic in _STANDARD_IDS:
            ids[mnemonic] = int(descriptor_id)
    return ids


def bufr_local_subcategory(message: bytes) -> int:
    """Section-1 local data subcategory (the ``NC005xxx`` subset number)."""
    edition = message[7]
    return message[17] if edition == 3 else message[20]


# ──────────────────────────────────────────────────────────────────────────
# Height from pressure (US Standard Atmosphere 1976)
# ──────────────────────────────────────────────────────────────────────────

_USSA_H_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0])
_USSA_LAPSE_K_M = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
_USSA_T_K = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_USSA_P_PA = np.array(
    [
        101_325.0,
        22_632.06397346295,
        5_474.888669677785,
        868.0186847552303,
        110.9063055549665,
        66.93887311868764,
        3.9564204280407553,
    ]
)
_USSA_TOP_P_PA = 0.37338358997621796
_USSA_K = 8.31432 / (9.80665 * 0.0289644)


def pressure_to_height_m(pressure_pa: np.ndarray) -> np.ndarray:
    """USSA-1976 geopotential height (m) for a pressure (Pa), floored at 0.

    AMVs carry a pressure, not a height. This is a climatological pressure
    altitude for the schema's height column, not an observation; NaN where the
    pressure is not finite, not positive, or above the 84.852 km table top.
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    height = np.full(p.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(p) & (p >= _USSA_TOP_P_PA)
    for index in range(_USSA_P_PA.size):
        last = index + 1 == _USSA_P_PA.size
        top_p = _USSA_TOP_P_PA if last else _USSA_P_PA[index + 1]
        below = p <= _USSA_P_PA[index] if index else np.ones(p.shape, dtype=bool)
        above_top = p >= top_p if last else p > top_p
        layer = usable & below & above_top
        if not layer.any():
            continue
        ratio = p[layer] / _USSA_P_PA[index]
        lapse = _USSA_LAPSE_K_M[index]
        if lapse:
            height[layer] = _USSA_H_M[index] + _USSA_T_K[index] / lapse * (
                np.power(ratio, -lapse * _USSA_K) - 1.0
            )
        else:
            height[layer] = _USSA_H_M[index] - _USSA_K * _USSA_T_K[index] * np.log(
                ratio
            )
    return np.maximum(height, 0.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────

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
                    "SWCM computation method: 1 IR, 2 VIS, 3 WV cloud top, "
                    "4-7 WV clear-sky/deep-layer"
                )
            },
        ),
        pa.field(
            "height_method",
            pa.uint8(),
            nullable=True,
            metadata={"description": "HAMD or EHAM height assignment method"},
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
            "qi",
            pa.float32(),
            nullable=True,
            metadata={
                "description": "Quality indicator without forecast (%), GSI resolution"
            },
        ),
        pa.field(
            "qi_forecast",
            pa.float32(),
            nullable=True,
            metadata={"description": "Quality indicator with forecast (%)"},
        ),
        pa.field(
            "expected_error",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "Expected error as encoded by the producer (percent "
                    "confidence or m/s)"
                )
            },
        ),
        pa.field(
            "gsi_case",
            pa.uint8(),
            nullable=True,
            metadata={
                "description": (
                    "GSI read_satwnd processing case (istype); null when GSI "
                    "does not process the subset"
                )
            },
        ),
    ]
)


# ──────────────────────────────────────────────────────────────────────────
# Subset decode
# ──────────────────────────────────────────────────────────────────────────


def _first_values(
    descriptors: list[Any], values: list[Any], ids: Mapping[str, int]
) -> tuple[dict[int, Any], dict[int, float], dict[int, float]]:
    """First non-missing value per descriptor id, plus GNAP/GNAPS -> PCCF maps."""
    first: dict[int, Any] = {}
    gnap: dict[int, float] = {}
    gnaps: dict[int, float] = {}
    pending: tuple[dict[int, float], int] | None = None
    id_gnap, id_gnaps, id_pccf = ids["GNAP"], ids["GNAPS"], ids["PCCF"]
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if value is None:
            if descriptor_id in (id_gnap, id_gnaps):
                pending = None
            continue
        if descriptor_id == id_gnap:
            pending = (gnap, int(value))
            continue
        if descriptor_id == id_gnaps:
            pending = (gnaps, int(value))
            continue
        if descriptor_id == id_pccf:
            if pending is not None:
                target, code = pending
                target.setdefault(code, float(value))
                pending = None
            continue
        first.setdefault(descriptor_id, value)
    return first, gnap, gnaps


def resolve_quality(
    case: int, gnap: Mapping[int, float], gnaps: Mapping[int, float]
) -> tuple[float, float, float]:
    """``(qi_without_forecast, qi_with_forecast, expected_error)`` per GSI case."""
    if case in _GNAP_QI_CODES:
        qifn_code, qify_code, ee_code = _GNAP_QI_CODES[case]
        return (
            gnap.get(qifn_code, math.nan),
            gnap.get(qify_code, math.nan),
            gnap.get(ee_code, math.nan),
        )
    if case in _AMVQIC_CASES:
        ee = math.nan
        for code in _AMVQIC_EE:
            if code in gnaps:
                ee = gnaps[code]
                break
        return gnaps.get(_AMVQIC_QIFN, math.nan), gnaps.get(_AMVQIC_QIFY, math.nan), ee
    return math.nan, math.nan, math.nan


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _extract_satwnd_subset(
    descriptors: list[Any],
    values: list[Any],
    subset: int,
    ids: Mapping[str, int],
    wanted: Mapping[str, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Rows (one per requested wind component) for one AMV subset."""
    first, gnap, gnaps = _first_values(descriptors, values, ids)

    def get(mnemonic: str) -> Any:
        return first.get(ids[mnemonic])

    said = get("SAID")
    swcm = get("SWCM")
    if said is None or swcm is None:
        return []
    said, swcm = int(said), int(swcm)
    resolved = SATWND_TYPE_TABLE.get((subset, said, swcm))
    if resolved is None:
        return []
    report_type, case = resolved

    lat = get("CLATH") if get("CLATH") is not None else get("CLAT")
    lon = get("CLONH") if get("CLONH") is not None else get("CLON")
    year, month, day = get("YEAR"), get("MNTH"), get("DAYS")
    if lat is None or lon is None or year is None or month is None or day is None:
        return []
    second = _float_or_nan(get("SECO"))
    try:
        obs_time = datetime(
            int(year),
            int(month),
            int(day),
            int(get("HOUR") or 0),
            int(get("MINU") or 0),
        ) + timedelta(seconds=0.0 if math.isnan(second) else second)
    except (TypeError, ValueError, OverflowError):
        return []
    if obs_time < dt_min or obs_time > dt_max:
        return []

    pressure = _float_or_nan(get("PRLC"))
    direction = _float_or_nan(get("WDIR"))
    speed = _float_or_nan(get("WSPD"))
    if not (
        math.isfinite(pressure) and math.isfinite(direction) and math.isfinite(speed)
    ):
        return []
    radians = math.radians(direction)
    components = {"u": -speed * math.sin(radians), "v": -speed * math.cos(radians)}

    height_method = get("EHAM") if get("EHAM") is not None else get("HAMD")
    quality_mark = get("SWQM")
    qi, qi_forecast, expected_error = resolve_quality(case, gnap, gnaps)
    height = float(pressure_to_height_m(np.array([pressure]))[0])
    base = {
        "time": obs_time,
        "lat": np.float32(float(lat)),
        "lon": np.float32(float(lon) % 360.0),
        "pres": np.float32(pressure),
        "elev": np.float32(height),
        "type": np.uint16(report_type),
        "level_cat": None,
        "class": "SATWND",
        # GSI c_station_id: computation-method tag and zero-padded SAID.
        "station": f"{SWCM_NAMES.get(swcm, 'XX')}{said:03d}",
        "station_elev": None,
        "quality": np.uint16(int(quality_mark)) if quality_mark is not None else None,
        "pressure_quality": None,
        "satellite_id": np.uint16(said),
        "subset": f"NC005{subset:03d}",
        "wind_method": np.uint8(swcm),
        "height_method": (
            np.uint8(int(height_method)) if height_method is not None else None
        ),
        "satellite_za": np.float32(_float_or_nan(get("SAZA"))),
        "qi": np.float32(qi),
        "qi_forecast": np.float32(qi_forecast),
        "expected_error": np.float32(expected_error),
        "gsi_case": np.uint8(case) if case >= 0 else None,
    }
    rows = []
    for component, variable in wanted.items():
        rows.append(
            {
                **base,
                "observation": np.float32(components[component]),
                "variable": variable,
            }
        )
    return rows


def _decode_satwnd_message(
    decoder: Any,
    message_bytes: bytes,
    ids: Mapping[str, int],
    wanted: Mapping[str, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    try:
        message = decoder.process(message_bytes)
    except Exception:
        return []
    if not message.n_subsets.value:
        return []
    subset = bufr_local_subcategory(message_bytes)
    template_data = message.template_data.value
    rows: list[dict[str, Any]] = []
    for descriptors, values in zip(
        template_data.decoded_descriptors_all_subsets,
        template_data.decoded_values_all_subsets,
    ):
        rows.extend(
            _extract_satwnd_subset(
                descriptors, values, subset, ids, wanted, dt_min, dt_max
            )
        )
    return rows


_worker_decoder: Any = None


def _init_decode_worker(
    table_b: dict[int, tuple[Any, ...]], table_d: dict[int, tuple[Any, ...]]
) -> None:
    global _worker_decoder  # noqa: PLW0603
    _worker_decoder = _create_decoder(table_b, table_d)


def _satwnd_worker(
    message_bytes: bytes,
    ids: Mapping[str, int],
    wanted: Mapping[str, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    with _silence_bufr_noise():
        return _decode_satwnd_message(
            _worker_decoder, message_bytes, ids, wanted, dt_min, dt_max
        )


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
        One row per (wind, component) in :data:`NCEP_SATWND_PUBLIC_SCHEMA`.
        Winds whose ``(subset, SAID, SWCM)`` GSI does not type are dropped.
    """
    decode_workers = max(1, decode_workers)
    wanted = {component: variable for variable, (component, _) in plan.items()}
    modifiers = {variable: modifier for variable, (_, modifier) in plan.items()}
    with open(path, "rb") as file:
        file_data = file.read()
    table_b, table_d, messages = _parse_prepbufr_messages(file_data, silence_noise=True)
    if not messages:
        return empty_dataframe(NCEP_SATWND_PUBLIC_SCHEMA)
    ids = resolve_mnemonics(table_b)
    rows: list[dict[str, Any]] = []
    use_parallel = decode_workers > 1 and len(messages) >= 32
    if use_parallel:
        with ProcessPoolExecutor(
            max_workers=decode_workers,
            initializer=_init_decode_worker,
            initargs=(table_b, table_d),
        ) as pool:
            futures = [
                pool.submit(_satwnd_worker, message_bytes, ids, wanted, dt_min, dt_max)
                for message_bytes, _data_category in messages
            ]
            for future in futures:
                try:
                    rows.extend(future.result())
                except Exception as error:
                    logger.debug(f"SATWND worker failed: {error}")
    else:
        decoder = _create_decoder(table_b, table_d)
        for message_bytes, _data_category in messages:
            rows.extend(
                _decode_satwnd_message(
                    decoder, message_bytes, ids, wanted, dt_min, dt_max
                )
            )
    return _finalize_rows(
        rows,
        modifiers,
        convert_pres_mb_to_pa=False,
        schema=NCEP_SATWND_PUBLIC_SCHEMA,
    )
