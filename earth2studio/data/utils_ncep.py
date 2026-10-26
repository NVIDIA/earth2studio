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

"""NCEP BUFR observation decoding for the NNJA and GDAS sources.

Every stream decodes with earth2bufr into a native wide Arrow table: one row per
PrepBUFR report level or GPS-RO profile level, per satellite wind, or per satellite
footprint, with one ``observation_<variable>`` column per requested variable. The
long format (one row per observation) is produced from it by :func:`_to_long`.
"""

from __future__ import annotations

import pathlib
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from earth2studio.data.utils import radiance_to_bt
from earth2studio.data.utils_bufr import (
    PREPBUFR_OBS_TYPES,
    civil_time_ns,
    constant_dictionary,
    constant_timestamp,
    datetime_ns,
    default_decode_threads,
    dictionary_column,
    float_values,
    float_with_nulls,
    iter_batches,
    list_value_filter,
    map_ordered,
    masked,
    null_array,
    numpy_with_mask,
    open_bufr,
    physical_values,
    timestamps_ns,
    uint_with_nulls,
)
from earth2studio.data.utils_ir import (
    CRIS_BANDS,
    cris_radiance_mw,
    iasi_radiance_mw,
    wavenumber_cm_inverse,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA, LexiconType
from earth2studio.lexicon.nnja import NNJAObsSatLexicon

NCEP_CONVENTIONAL_PUBLIC_SCHEMA = pa.schema(
    [
        E2STUDIO_SCHEMA.field("time"),
        pa.field(
            "report_time",
            pa.timestamp("ns"),
            nullable=True,
            metadata={"description": "PrepBUFR report-header time (DHR)"},
        ),
        pa.field(
            "pres",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "Observation pressure coordinate (Pa); null for "
                    "source-native GPSRO rows"
                )
            },
        ),
        pa.field(
            "elev",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "Observation height (m); GPSRO bending-angle rows use impact "
                    "parameter minus Earth radius of curvature, refractivity rows "
                    "the level's geometric height (HEIT)"
                )
            },
        ),
        pa.field(
            "type",
            pa.uint16(),
            nullable=True,
            metadata={
                "description": (
                    "PrepBUFR report type for conventional rows; GPSRO "
                    "receiver satellite identifier (SAID)"
                )
            },
        ),
        pa.field(
            "level_cat",
            pa.uint16(),
            nullable=True,
            metadata={"description": "PrepBUFR CAT level category"},
        ),
        E2STUDIO_SCHEMA.field("class"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        pa.field(
            "report_lat",
            pa.float32(),
            nullable=True,
            metadata={"description": "PrepBUFR report-header latitude (YOB)"},
        ),
        pa.field(
            "report_lon",
            pa.float32(),
            nullable=True,
            metadata={"description": "PrepBUFR report-header longitude (XOB)"},
        ),
        E2STUDIO_SCHEMA.field("station"),
        E2STUDIO_SCHEMA.field("station_elev"),
        pa.field(
            "quality",
            pa.uint16(),
            nullable=True,
            metadata={
                "description": (
                    "PrepBUFR quality mark for conventional rows; QFRO flag "
                    "table for GPSRO rows"
                )
            },
        ),
        pa.field(
            "pressure_quality",
            pa.uint16(),
            nullable=True,
            metadata={"description": "PrepBUFR pressure quality mark (PQM)"},
        ),
        pa.field(
            "radius_curvature",
            pa.float64(),
            nullable=True,
            metadata={
                "description": (
                    "GPSRO local Earth radius of curvature (ELRC, m); null for "
                    "other rows"
                )
            },
        ),
        pa.field(
            "geoid_undulation",
            pa.float64(),
            nullable=True,
            metadata={
                "description": "GPSRO geoid undulation (GEODU, m); null for other rows"
            },
        ),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
        pa.field(
            "cycle_time",
            pa.timestamp("ns"),
            nullable=True,
            metadata={
                "description": (
                    "Cycle of the NCEP file the row was decoded from; an observation "
                    "at a file's window edge can appear in two files"
                )
            },
        ),
    ]
)


_ACFT_PROFILE_SCALAR_TYPE_MAP = {
    330: 130,
    430: 130,
    530: 130,
    331: 131,
    431: 131,
    531: 131,
    332: 132,
    432: 132,
    532: 132,
    333: 133,
    433: 133,
    533: 133,
    334: 134,
    434: 134,
    534: 134,
    335: 135,
    435: 135,
    535: 135,
}

_ACFT_PROFILE_WIND_TYPE_MAP = {
    source_type: target_type + 100
    for source_type, target_type in _ACFT_PROFILE_SCALAR_TYPE_MAP.items()
}


# PrepBUFR ``pres`` variable: the GSI-style pressure-observation population, not
# every level coordinate. Pressure-capable report types, CAT == 0, POB >= 500 hPa
# and a 0-15 quality mark (GSI's read-stage guard).
_PRESSURE_REPORT_TYPES = np.array([120, 180, 181, 187])
_PRESSURE_CLASSES = frozenset(["ADPUPA", "ADPSFC", "SFCSHP"])
_PRESSURE_MIN_HPA = np.float32(500.0)

# ──────────────────────────────────────────────────────────────────────────
# Shared NCEP observation request helpers. Keep these as stateless utilities:
# concrete data sources own transport/cache state and pass callables in where
# source-specific behavior is needed.
# ──────────────────────────────────────────────────────────────────────────

_NCEPObsModifier = Callable[[pd.DataFrame], pd.DataFrame]
# Decode plan: variable id -> (route-specific decode key, post-decode modifier).
# The key is the raw string after the lexicon "route::" prefix; each adapter
# interprets it (PrepBUFR mnemonic, or GPS RO descriptor id parsed to int).
_NCEPPlan = Mapping[str, tuple[str, _NCEPObsModifier]]


# Disjoint, sorted closed time intervals requested from one cycle file
Intervals = tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True)
class NCEPObsTask:
    """One conventional NCEP observation cycle file to fetch and decode.

    ``columns`` are the output columns requested of the decode; None requests all.
    """

    uri: str
    route: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: _NCEPPlan
    intervals: Intervals = ()
    columns: frozenset[str] | None = None


def plan_conv_tasks(
    windows: Mapping[datetime, Intervals],
    variable: list[str],
    lexicon: LexiconType,
    build_uri: Callable[[str, datetime], str],
) -> list[NCEPObsTask]:
    """Partition variables by lexicon route prefix into per-cycle conv tasks.

    Parameters
    ----------
    windows : Mapping[datetime, Intervals]
        Cycle datetime -> requested observation-time intervals (``cycle_windows``).
    variable : list[str]
        Requested variable ids, routed by their ``"route::key"`` lexicon prefix.
    lexicon : LexiconType
        Source lexicon mapping a variable id to ``("route::key", modifier)``.
    build_uri : Callable[[str, datetime], str]
        Source hook returning the archive URI for a ``(route, cycle)`` pair.

    Returns
    -------
    list[NCEPObsTask]
        One task per ``(route, cycle)``.
    """
    plans: dict[str, dict[str, tuple[str, _NCEPObsModifier]]] = {}
    for v in variable:
        try:
            source_key, modifier = lexicon[v]
        except KeyError:
            logger.error(f"Variable id '{v}' not found in {lexicon.__name__}")
            raise
        route, _, rest = source_key.partition("::")
        plans.setdefault(route, {})[v] = (rest, modifier)

    tasks: list[NCEPObsTask] = []
    for cycle, intervals in windows.items():
        for route, plan in plans.items():
            tasks.append(
                NCEPObsTask(
                    uri=build_uri(route, cycle),
                    route=route,
                    datetime_file=cycle,
                    datetime_min=intervals[0][0],
                    datetime_max=intervals[-1][1],
                    var_plan=plan,
                    intervals=intervals,
                )
            )
    return tasks


def observation_cycle_times(
    time: datetime,
    tolerance_lower: timedelta,
    tolerance_upper: timedelta,
    cadence: timedelta = timedelta(hours=6),
) -> list[datetime]:
    """Return the cycle file times a request reads.

    NCEP cycle files (PrepBUFR, GPS-RO, satellite dumps and the GSI diagnostics
    built from them) hold the reports whose report time lies within
    ``cadence / 2`` of their cycle ``T``. A request ``[tmin, tmax]`` reads every
    file whose span it overlaps in the interior,
    ``T - cadence / 2 < tmax`` and ``T + cadence / 2 > tmin``, and the decoders
    cut the rows to the closed window by report time.

    A request returns the contents of the cycles it overlaps, cut to its window.
    An observation exactly on a ``T ± cadence / 2`` boundary is returned from the
    files the request reads; rows only a neighbouring cycle holds at that instant
    are not returned. A zero-length request reads every file whose closed span
    holds its instant, so an instant on a boundary reads both files. Each row
    carries its file's ``cycle_time``. Drifting-sonde
    levels can carry times past their file's span; they follow their report time.

    Parameters
    ----------
    time : datetime
        Requested observation timestamp.
    tolerance_lower : timedelta
        Lower tolerance bound relative to ``time``.
    tolerance_upper : timedelta
        Upper tolerance bound relative to ``time``.
    cadence : timedelta, optional
        Cadence between published files, by default ``timedelta(hours=6)``.

    Returns
    -------
    list[datetime]
        Cadence-aligned file timestamps whose spans overlap the interior of
        ``[time + tolerance_lower, time + tolerance_upper]``.

    Raises
    ------
    ValueError
        If ``cadence`` is not positive.
    """
    if cadence <= timedelta(0):
        raise ValueError("cadence must be positive")

    first = time + tolerance_lower - cadence / 2
    last = time + tolerance_upper + cadence / 2
    # A zero-length request reads every file whose closed span holds the instant
    instant = tolerance_lower == tolerance_upper
    day_start = first.replace(hour=0, minute=0, second=0, microsecond=0)
    steps = (first - day_start) // cadence
    if not (instant and day_start + steps * cadence == first):
        steps += 1
    cycle = day_start + steps * cadence

    cycles: list[datetime] = []
    while cycle < last or (instant and cycle == last):
        cycles.append(cycle)
        cycle += cadence
    return cycles


def cycle_windows(
    time_list: list[datetime],
    tolerance_lower: timedelta,
    tolerance_upper: timedelta,
    cadence: timedelta = timedelta(hours=6),
) -> dict[datetime, Intervals]:
    """Map cadence file times to the observation-time intervals requested from them.

    Requests sharing a file keep their own intervals; overlapping or touching
    intervals are merged, so a file read for 23Z ±30 min and 02Z ±30 min returns
    no rows between 23:30 and 01:30.

    Parameters
    ----------
    time_list : list[datetime]
        Requested observation timestamps.
    tolerance_lower : timedelta
        Lower tolerance bound relative to each requested timestamp.
    tolerance_upper : timedelta
        Upper tolerance bound relative to each requested timestamp.
    cadence : timedelta, optional
        Cadence between published files, by default ``timedelta(hours=6)``.

    Returns
    -------
    dict[datetime, Intervals]
        Mapping from each required cadence file timestamp to its sorted, disjoint
        closed observation-time intervals.
    """
    requested: dict[datetime, list[tuple[datetime, datetime]]] = {}
    for t in time_list:
        for cycle in observation_cycle_times(
            t, tolerance_lower, tolerance_upper, cadence
        ):
            requested.setdefault(cycle, []).append(
                (t + tolerance_lower, t + tolerance_upper)
            )
    windows: dict[datetime, Intervals] = {}
    for cycle, intervals in requested.items():
        merged: list[tuple[datetime, datetime]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        windows[cycle] = tuple(merged)
    return windows


def filter_to_intervals(table: pa.Table, intervals: Intervals) -> pa.Table:
    """Keep the rows whose filter time lies in at least one closed interval.

    The filter time is the PrepBUFR report time where the table has one, else the
    row time, matching the decoders' window filter.
    """
    times = table["time"]
    if "report_time" in table.column_names:
        times = pc.coalesce(table["report_time"], times)
    keep = pa.array(np.zeros(table.num_rows, dtype=bool))
    for start, end in intervals:
        inside = pc.and_(
            pc.greater_equal(times, pa.scalar(start, times.type)),
            pc.less_equal(times, pa.scalar(end, times.type)),
        )
        keep = pc.or_(keep, inside)
    if pc.all(keep).as_py():
        return table
    return table.filter(keep)


def resolve_output_schema(
    schema: pa.Schema,
    fields: str | list[str] | pa.Schema | None,
    *,
    class_name: str,
) -> pa.Schema:
    """Resolve a requested field subset against a source schema."""
    if fields is None:
        return schema
    if isinstance(fields, str):
        fields = [fields]
    if isinstance(fields, pa.Schema):
        for f in fields:
            if f.name not in schema.names:
                raise KeyError(
                    f"Field '{f.name}' not in {class_name} SCHEMA. "
                    f"Available: {schema.names}"
                )
            expected = schema.field(f.name).type
            if f.type != expected:
                raise TypeError(
                    f"Field '{f.name}' has type {f.type}, expected "
                    f"{expected} from class SCHEMA"
                )
        return fields
    selected = []
    for name in fields:
        if name not in schema.names:
            raise KeyError(
                f"Field '{name}' not in {class_name} SCHEMA. Available: {schema.names}"
            )
        selected.append(schema.field(name))
    return pa.schema(selected)


# ──────────────────────────────────────────────────────────────────────────
# Aggregate satellite schema and constants
# ──────────────────────────────────────────────────────────────────────────

_C_CM_S = 2.99792458e10
# ATMS channel centre frequencies by channel as the NNJA atms aggregates encode them:
# SCCF (0-02-153) raw codes at scale -5, so the frequency in Hz is code x 10**5. The
# same for every platform; decode_satellite checks them against each file's SCCF.
# fmt: off
_ATMS_SCCF_CODE = np.array(
    [238000, 314000, 503000, 517600, 528000, 536000, 544000, 549400, 555000, 572900,
     575100, 576600, 576300, 576200, 576100, 882000, 1655000, 1903100, 1878000,
     1863000, 1851100, 1843100],
    dtype=np.float64,
)
# fmt: on
_ATMS_WAVENUMBER_CM = _ATMS_SCCF_CODE / (1.0 * 10**-5) / _C_CM_S

_NCEP_SATELLITE_NAME_BY_SAID: dict[int, str] = {
    3: "metop-b",
    4: "metop-a",
    5: "metop-c",
    206: "n15",
    207: "n16",
    208: "n17",
    209: "n18",
    223: "n19",
    224: "npp",
    225: "n20",
    226: "n21",
    # IR sounder platforms
    784: "aqua",  # Aqua AIRS
}
NCEP_MICROWAVE_SATELLITES = frozenset(_NCEP_SATELLITE_NAME_BY_SAID.values())

# Signed nominal cross-track geometry. These are the instrument defaults used
# by GSI's ``satstep`` routine, not experiment-specific ``scaninfo`` overrides:
# https://github.com/NOAA-EMC/GSI/blob/860d13740352004fca0136a8c3d0ac9dea30e0da/src/gsi/radinfo.f90#L1523-L1643
_NCEP_MICROWAVE_SCAN_GEOMETRY: dict[str, tuple[float, float]] = {
    "atms": (-52.725, 1.11),
    "amsua": (-48.0 - 1.0 / 3.0, 3.0 + 1.0 / 3.0),
    "amsub": (-48.95, 1.1),
    "mhs": (-445.0 / 9.0, 10.0 / 9.0),
}


NCEP_MICROWAVE_OUTPUT_SCHEMA = pa.schema(
    [
        E2STUDIO_SCHEMA.field("time"),
        E2STUDIO_SCHEMA.field("class"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        E2STUDIO_SCHEMA.field("elev"),
        E2STUDIO_SCHEMA.field("scan_angle"),
        pa.field(
            "scan_position",
            pa.uint16(),
            metadata={"description": "Encoded one-based field-of-view number"},
        ),
        pa.field("scan_line", pa.uint32(), nullable=True),
        pa.field(
            "detector",
            pa.uint16(),
            nullable=True,
            metadata={
                "description": (
                    "CrIS detector within the field of regard (FOVN, 1-9), which "
                    "with scan_position fixes the look angle; null for other sensors"
                )
            },
        ),
        E2STUDIO_SCHEMA.field("sensor_index"),
        E2STUDIO_SCHEMA.field("wavenumber"),
        E2STUDIO_SCHEMA.field("solza"),
        E2STUDIO_SCHEMA.field("solaza"),
        E2STUDIO_SCHEMA.field("satellite_za"),
        E2STUDIO_SCHEMA.field("satellite_aza"),
        pa.field(
            "quality",
            pa.uint32(),
            nullable=True,
            metadata={
                "description": (
                    "Encoded channel-quality flags; interpretation is sensor "
                    "and product specific. Microwave: per-channel 0-33-081 "
                    "flags as encoded. AIRS: 24-bit ACQF. CrIS: "
                    "NFQF (bits 0-18) | NCQF<<19 (bits 19-27) for the "
                    "channel's band. IASI: null (mtiasi carries no "
                    "per-channel flag)."
                )
            },
        ),
        pa.field(
            "scan_quality",
            pa.uint32(),
            nullable=True,
            metadata={
                "description": (
                    "Scan-level quality flags as encoded: ATMS 0-33-080, "
                    "CrIS NSQF 0-33-075; null for other sensors"
                )
            },
        ),
        pa.field(
            "granule_quality",
            pa.uint16(),
            nullable=True,
            metadata={
                "description": (
                    "Granule-level quality flags as encoded (ATMS 0-33-079); "
                    "null for other sensors"
                )
            },
        ),
        pa.field(
            "footprint_quality",
            pa.uint8(),
            nullable=True,
            metadata={
                "description": (
                    "IASI system quality flag QGFQ (0-33-060) as encoded; "
                    "null for other sensors"
                )
            },
        ),
        E2STUDIO_SCHEMA.field("satellite"),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
        pa.field(
            "cycle_time",
            pa.timestamp("ns"),
            nullable=True,
            metadata={
                "description": (
                    "Cycle of the NCEP file the row was decoded from; an observation "
                    "at a file's window edge can appear in two files"
                )
            },
        ),
    ]
)


# ──────────────────────────────────────────────────────────────────────────
# Satellite wind schema
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


# ──────────────────────────────────────────────────────────────────────────
# Wide schemas and the long conversion
# ──────────────────────────────────────────────────────────────────────────

PREPBUFR_VARIABLES = ("u", "v", "q", "t", "pres")
GPSRO_VARIABLES = ("gps", "gps_refractivity")
CONV_VARIABLES = (*PREPBUFR_VARIABLES, *GPSRO_VARIABLES)
SATWND_VARIABLES = ("u", "v")
SAT_VARIABLE_SENSOR: dict[str, str] = {
    variable: key.split("::")[0] for variable, key in NNJAObsSatLexicon.VOCAB.items()
}
SAT_SENSORS = tuple(dict.fromkeys(SAT_VARIABLE_SENSOR.values()))
# Channel numbers each NNJA aggregate carries, ascending: the channel axis of a wide
# satellite frame when no channels are requested. These are the channel sets of the
# HealDA NNJA satellite archive, one per sensor over the whole record.
SAT_CHANNELS: dict[str, tuple[int, ...]] = {
    "atms": tuple(range(1, 23)),
    "mhs": tuple(range(1, 6)),
    "amsua": tuple(range(1, 16)),
    "amsub": tuple(range(1, 6)),
    # fmt: off
    "airs": (
           1,    6,    7,   10,   11,   15,   16,   17,   20,   21,   22,   24,   27,   28,
          30,   36,   39,   40,   42,   51,   52,   54,   55,   56,   59,   62,   63,   68,
          69,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   82,   83,   84,
          86,   92,   93,   98,   99,  101,  104,  105,  108,  110,  111,  113,  116,  117,
         123,  124,  128,  129,  138,  139,  144,  145,  150,  151,  156,  157,  159,  162,
         165,  168,  169,  170,  172,  173,  174,  175,  177,  179,  180,  182,  185,  186,
         190,  192,  198,  201,  204,  207,  210,  215,  216,  221,  226,  227,  232,  252,
         253,  256,  257,  261,  262,  267,  272,  295,  299,  300,  305,  310,  321,  325,
         333,  338,  355,  362,  375,  453,  475,  484,  497,  528,  587,  672,  787,  791,
         843,  870,  914,  950, 1003, 1012, 1019, 1024, 1030, 1038, 1048, 1069, 1079, 1082,
        1083, 1088, 1090, 1092, 1095, 1104, 1111, 1115, 1116, 1119, 1120, 1123, 1130, 1138,
        1142, 1178, 1199, 1206, 1221, 1237, 1252, 1260, 1263, 1266, 1285, 1301, 1304, 1329,
        1371, 1382, 1415, 1424, 1449, 1455, 1466, 1477, 1500, 1519, 1538, 1545, 1565, 1574,
        1583, 1593, 1614, 1627, 1636, 1644, 1652, 1669, 1674, 1681, 1694, 1708, 1717, 1723,
        1740, 1748, 1751, 1756, 1763, 1766, 1771, 1777, 1780, 1783, 1794, 1800, 1803, 1806,
        1812, 1826, 1843, 1852, 1865, 1866, 1868, 1869, 1872, 1873, 1876, 1881, 1882, 1883,
        1911, 1917, 1918, 1924, 1928, 1937, 1941, 2099, 2100, 2101, 2103, 2104, 2106, 2107,
        2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121,
        2122, 2123, 2128, 2134, 2141, 2145, 2149, 2153, 2164, 2189, 2197, 2209, 2226, 2234,
        2280, 2318, 2321, 2325, 2328, 2333, 2339, 2348, 2353, 2355, 2357, 2363, 2370, 2371,
        2377,
    ),
    "iasi": (
          16,   29,   32,   35,   38,   41,   44,   47,   49,   50,   51,   53,   55,   56,
          57,   59,   61,   62,   63,   66,   68,   70,   72,   74,   76,   78,   79,   81,
          82,   83,   84,   85,   86,   87,   89,   92,   93,   95,   97,   99,  101,  103,
         104,  106,  109,  110,  111,  113,  116,  119,  122,  125,  128,  131,  133,  135,
         138,  141,  144,  146,  148,  150,  151,  154,  157,  159,  160,  161,  163,  167,
         170,  173,  176,  179,  180,  185,  187,  191,  193,  197,  199,  200,  202,  203,
         205,  207,  210,  212,  213,  214,  217,  218,  219,  222,  224,  225,  226,  228,
         230,  231,  232,  236,  237,  239,  243,  246,  249,  252,  254,  259,  260,  262,
         265,  267,  269,  275,  279,  282,  285,  294,  296,  299,  300,  303,  306,  309,
         313,  320,  323,  326,  327,  329,  332,  335,  345,  347,  350,  354,  356,  360,
         363,  366,  371,  372,  373,  375,  377,  379,  381,  383,  386,  389,  398,  401,
         404,  405,  407,  408,  410,  411,  414,  416,  418,  423,  426,  428,  432,  433,
         434,  439,  442,  445,  450,  457,  459,  472,  477,  483,  509,  515,  546,  552,
         559,  566,  571,  573,  578,  584,  594,  625,  646,  662,  668,  705,  739,  756,
         797,  867,  906,  921, 1027, 1046, 1090, 1098, 1121, 1133, 1173, 1191, 1194, 1222,
        1271, 1283, 1338, 1409, 1414, 1420, 1424, 1427, 1430, 1434, 1440, 1442, 1445, 1450,
        1454, 1460, 1463, 1469, 1474, 1479, 1483, 1487, 1494, 1496, 1502, 1505, 1509, 1510,
        1513, 1518, 1521, 1526, 1529, 1532, 1536, 1537, 1541, 1545, 1548, 1553, 1560, 1568,
        1574, 1579, 1583, 1585, 1587, 1606, 1626, 1639, 1643, 1652, 1658, 1659, 1666, 1671,
        1675, 1681, 1694, 1697, 1710, 1786, 1791, 1805, 1839, 1884, 1913, 1946, 1947, 1991,
        2019, 2094, 2119, 2213, 2239, 2271, 2289, 2321, 2333, 2346, 2349, 2352, 2359, 2367,
        2374, 2398, 2426, 2562, 2701, 2741, 2745, 2760, 2819, 2889, 2907, 2910, 2919, 2921,
        2939, 2944, 2945, 2948, 2951, 2958, 2971, 2977, 2985, 2988, 2990, 2991, 2993, 3002,
        3008, 3014, 3027, 3029, 3030, 3036, 3047, 3049, 3052, 3053, 3055, 3058, 3064, 3069,
        3087, 3093, 3098, 3105, 3107, 3110, 3116, 3127, 3129, 3136, 3146, 3151, 3160, 3165,
        3168, 3175, 3178, 3189, 3207, 3228, 3244, 3248, 3252, 3256, 3263, 3281, 3295, 3303,
        3309, 3312, 3322, 3326, 3354, 3366, 3375, 3378, 3411, 3416, 3432, 3438, 3440, 3442,
        3444, 3446, 3448, 3450, 3452, 3454, 3458, 3467, 3476, 3484, 3491, 3497, 3499, 3504,
        3506, 3509, 3518, 3527, 3555, 3575, 3577, 3580, 3582, 3586, 3589, 3599, 3610, 3626,
        3638, 3646, 3653, 3658, 3661, 3673, 3689, 3700, 3710, 3726, 3763, 3814, 3841, 3888,
        4032, 4059, 4068, 4082, 4095, 4160, 4234, 4257, 4411, 4498, 4520, 4552, 4567, 4608,
        4646, 4698, 4808, 4849, 4920, 4939, 4947, 4967, 4991, 4996, 5015, 5028, 5056, 5128,
        5130, 5144, 5170, 5178, 5183, 5188, 5191, 5368, 5371, 5379, 5381, 5383, 5397, 5399,
        5401, 5403, 5405, 5446, 5455, 5472, 5480, 5483, 5485, 5492, 5497, 5502, 5507, 5509,
        5517, 5528, 5558, 5697, 5714, 5749, 5766, 5785, 5798, 5799, 5801, 5817, 5833, 5834,
        5836, 5849, 5851, 5852, 5865, 5869, 5881, 5884, 5897, 5900, 5916, 5932, 5948, 5963,
        5968, 5978, 5988, 5992, 5994, 5997, 6003, 6008, 6023, 6026, 6039, 6053, 6056, 6067,
        6071, 6082, 6085, 6098, 6112, 6126, 6135, 6140, 6149, 6154, 6158, 6161, 6168, 6174,
        6182, 6187, 6205, 6209, 6213, 6317, 6339, 6342, 6366, 6381, 6391, 6489, 6962, 6966,
        6970, 6975, 6977, 6982, 6985, 6987, 6989, 6991, 6993, 6995, 6997, 6999, 7000, 7004,
        7008, 7013, 7016, 7021, 7024, 7027, 7029, 7032, 7038, 7043, 7046, 7049, 7069, 7072,
        7076, 7081, 7084, 7089, 7099, 7209, 7222, 7231, 7235, 7247, 7267, 7269, 7284, 7389,
        7419, 7423, 7424, 7426, 7428, 7431, 7436, 7444, 7475, 7549, 7584, 7665, 7666, 7831,
        7836, 7853, 7865, 7885, 7888, 7912, 7950, 7972, 7980, 7995, 8007, 8015, 8055, 8078,
    ),
    "cris": (
          19,   24,   26,   27,   28,   31,   32,   33,   37,   39,   42,   44,   47,   49,
          50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,
          64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,
          78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,
          92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,
         106,  107,  108,  109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,
         120,  121,  122,  123,  124,  125,  126,  127,  128,  129,  130,  131,  132,  133,
         134,  135,  136,  137,  138,  139,  140,  141,  142,  143,  144,  145,  146,  147,
         148,  149,  150,  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,  161,
         162,  163,  164,  165,  166,  167,  168,  169,  170,  171,  172,  173,  174,  175,
         176,  177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187,  188,  189,
         190,  191,  192,  193,  194,  195,  196,  197,  198,  199,  200,  208,  211,  216,
         224,  234,  236,  238,  239,  242,  246,  248,  255,  264,  266,  268,  275,  279,
         283,  285,  291,  295,  301,  305,  311,  332,  342,  389,  400,  402,  404,  406,
         410,  427,  439,  440,  441,  445,  449,  455,  458,  461,  464,  467,  470,  473,
         475,  482,  486,  487,  490,  493,  496,  499,  501,  503,  505,  511,  513,  514,
         518,  519,  520,  522,  529,  534,  563,  568,  575,  592,  594,  596,  598,  600,
         602,  604,  611,  614,  616,  618,  620,  622,  626,  631,  638,  646,  648,  652,
         659,  673,  675,  678,  684,  688,  694,  700,  707,  710,  713,  714,  718,  720,
         722,  725,  728,  735,  742,  748,  753,  762,  780,  784,  798,  849,  860,  862,
         866,  874,  882,  890,  898,  906,  907,  908,  914,  937,  972,  973,  978,  980,
         981,  988,  995,  998, 1000, 1003, 1008, 1009, 1010, 1014, 1017, 1018, 1020, 1022,
        1024, 1026, 1029, 1030, 1032, 1034, 1037, 1038, 1041, 1042, 1044, 1046, 1049, 1050,
        1053, 1054, 1058, 1060, 1062, 1064, 1066, 1069, 1076, 1077, 1080, 1086, 1091, 1095,
        1101, 1109, 1112, 1121, 1128, 1133, 1163, 1172, 1187, 1189, 1205, 1211, 1219, 1231,
        1245, 1271, 1289, 1300, 1313, 1316, 1325, 1329, 1346, 1347, 1473, 1474, 1491, 1499,
        1553, 1570, 1596, 1602, 1619, 1624, 1635, 1939, 1940, 1941, 1942, 1943, 1944, 1945,
        1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
        1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973,
        1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
        2119, 2140, 2143, 2147, 2153, 2158, 2161, 2168, 2171, 2175, 2182,
    ),
    # fmt: on
}

# Default of the ``layout`` parameter of the NNJA and GDAS sources
_DEFAULT_LAYOUT: Literal["long", "wide"] = "long"


def _validate_layout(layout: str) -> Literal["long", "wide"]:
    if layout not in ("long", "wide"):
        raise ValueError(f"layout must be 'long' or 'wide', got {layout!r}")
    return layout  # type: ignore[return-value]


def _fields(schema: pa.Schema, exclude: Collection[str]) -> list[pa.Field]:
    return [field for field in schema if field.name not in exclude]


def _observation_field(variable: str, arrow_type: pa.DataType) -> pa.Field:
    return pa.field(
        f"observation_{variable}",
        arrow_type,
        nullable=True,
        metadata={"description": f"Observation of variable {variable!r}"},
    )


def _conv_wide_schema(variables: Sequence[str], unfilled: Sequence[str]) -> pa.Schema:
    return pa.schema(
        [
            *_fields(
                NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
                ("quality", "observation", "variable", *unfilled),
            ),
            *[
                field
                for variable in variables
                for field in (
                    _observation_field(variable, pa.float32()),
                    pa.field(
                        f"quality_{variable}",
                        pa.uint16(),
                        nullable=True,
                        metadata={
                            "description": (
                                f"Quality mark of the {variable!r} observation"
                            )
                        },
                    ),
                )
            ],
        ]
    )


# One wide stream per decoder, each with the columns its decoder fills
_NCEP_PREPBUFR_WIDE_SCHEMA = _conv_wide_schema(
    PREPBUFR_VARIABLES, ("radius_curvature", "geoid_undulation")
)
_NCEP_GPSRO_WIDE_SCHEMA = _conv_wide_schema(
    GPSRO_VARIABLES,
    (
        "report_time",
        "pres",
        "level_cat",
        "report_lat",
        "report_lon",
        "station_elev",
        "pressure_quality",
    ),
)


def _conv_stream_schema(variables: Collection[str], class_name: str) -> pa.Schema:
    """The wide schema of the conventional stream (PrepBUFR or GPS-RO) holding
    ``variables``, with the observation and quality columns of ``variables`` only; a
    wide call returns one stream.

    Raises
    ------
    ValueError
        If ``variables`` span both streams
    """
    if set(variables) <= set(GPSRO_VARIABLES):
        return _wide_schema(_NCEP_GPSRO_WIDE_SCHEMA, variables)
    if set(variables) <= set(PREPBUFR_VARIABLES):
        return _wide_schema(_NCEP_PREPBUFR_WIDE_SCHEMA, variables)
    raise ValueError(
        f'{class_name}(layout="wide") returns one stream per call: request the '
        f"PrepBUFR variables {list(PREPBUFR_VARIABLES)} or the GPS-RO variables "
        f"{list(GPSRO_VARIABLES)}, not both"
    )


_SATWND_UNFILLED = (
    "report_time",
    "elev",
    "type",
    "level_cat",
    "report_lat",
    "report_lon",
    "station",
    "station_elev",
    "pressure_quality",
    "radius_curvature",
    "geoid_undulation",
)
_NCEP_SATWND_WIDE_SCHEMA = pa.schema(
    [
        *_fields(
            NCEP_SATWND_PUBLIC_SCHEMA, ("observation", "variable", *_SATWND_UNFILLED)
        ),
        *[_observation_field(v, pa.float32()) for v in SATWND_VARIABLES],
    ]
)

_CHANNELS_KEY = b"channels"
_SENSOR_KEY = b"sensor"


def _channel_field(
    name: str,
    value_type: pa.DataType,
    sensor: str,
    channels: Sequence[int],
    description: str,
) -> pa.Field:
    """A per-channel column with one slot per channel of ``channels``; the sensor and
    the channels are recorded in the field metadata."""
    return pa.field(
        name,
        pa.list_(value_type, len(channels)),
        nullable=True,
        metadata={
            "description": description,
            _SENSOR_KEY: sensor,
            _CHANNELS_KEY: ",".join(str(c) for c in channels),
        },
    )


def _field_channels(field: pa.Field) -> np.ndarray:
    return np.array(
        [int(c) for c in field.metadata[_CHANNELS_KEY].split(b",")], dtype=np.int64
    )


# Footprint columns only some sensors' templates fill: surface height (HOLS) for the
# ATOVS sensors and CrIS, the nominal scan angle for the microwave sensors, the CrIS
# detector, and each sensor's provider footprint flags
_SAT_SENSOR_FOOTPRINT = {
    "atms": ("scan_angle", "scan_quality", "granule_quality"),
    "mhs": ("elev", "scan_angle"),
    "amsua": ("elev", "scan_angle"),
    "amsub": ("elev", "scan_angle"),
    "airs": (),
    "iasi": ("footprint_quality",),
    "cris": ("elev", "detector", "scan_quality"),
}


def _sat_wide_schema(sensor: str, channels: Collection[int] | None = None) -> pa.Schema:
    """Wide schema of one sensor: the shared footprint columns, the sensor's own
    footprint columns, and ``quality`` and ``observation_<variable>`` over the channel
    axis ``channels`` (ascending), else ``SAT_CHANNELS[sensor]``."""
    axis = sorted(SAT_CHANNELS[sensor] if channels is None else channels)
    others = {
        name
        for s, names in _SAT_SENSOR_FOOTPRINT.items()
        if s != sensor
        for name in names
    } - set(_SAT_SENSOR_FOOTPRINT[sensor])
    footprint = _fields(
        NCEP_MICROWAVE_OUTPUT_SCHEMA,
        ("sensor_index", "quality", "wavenumber", "observation", "variable", *others),
    )
    return pa.schema(
        [
            *footprint,
            _channel_field(
                "quality",
                pa.uint32(),
                sensor,
                axis,
                f"{sensor} per-channel quality, one slot per channel",
            ),
            *[
                _channel_field(
                    f"observation_{variable}",
                    pa.float32(),
                    sensor,
                    axis,
                    f"Observation of variable {variable!r}, one slot per channel",
                )
                for variable, variable_sensor in SAT_VARIABLE_SENSOR.items()
                if variable_sensor == sensor
            ],
        ]
    )


def channel_axis(frame: pd.DataFrame, sensor: str) -> np.ndarray:
    """Channel numbers of the slots of a wide satellite frame's ``sensor`` columns.

    In a ``layout="wide"`` :class:`~earth2studio.data.NNJAObsSat` frame of
    ``sensor``, ``observation_<variable>`` and ``quality`` hold one value per slot;
    slot ``i`` is channel ``channel_axis(frame, sensor)[i]``.

    Parameters
    ----------
    frame : pd.DataFrame
        Wide satellite frame
    sensor : str
        Sensor name, e.g. ``"cris"``

    Returns
    -------
    np.ndarray
        Channel numbers, ascending
    """
    return np.array(frame.attrs["channels"][sensor], dtype=np.int64)


def _is_list(arrow_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
    )


def _single_chunk(column: pa.ChunkedArray) -> pa.Array:
    return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()


def _to_long(
    table: pa.Table,
    variables: Sequence[str],
    schema: pa.Schema,
) -> pa.Table:
    """Convert a wide NCEP observation table to the long format.

    Scalar ``observation_<variable>`` columns give one row per valid value,
    variable by variable in ``variables`` order, with ``quality`` from
    ``quality_<variable>`` when the table has it. Fixed-size list
    ``observation_<variable>`` columns (satellite footprints) give one row per
    slot of the sensor's channel axis, footprint by footprint, with the variables
    of one sensor interleaved per channel, ``sensor_index`` and ``wavenumber``
    from the axis and ``quality`` from ``quality_<sensor>``.

    Parameters
    ----------
    table : pa.Table
        Wide table
    variables : Sequence[str]
        Variables to emit, in output order; absent columns are skipped
    schema : pa.Schema
        Long schema; its columns missing from the result are filled with nulls

    Returns
    -------
    pa.Table
        Long table with the columns of ``schema``
    """
    present = [v for v in variables if f"observation_{v}" in table.column_names]
    if not present or table.num_rows == 0:
        return _conform([], schema)
    first_type = table.schema.field(f"observation_{present[0]}").type
    if pa.types.is_fixed_size_list(first_type):
        parts = _explode_channels(table, present)
    else:
        parts = _stack_variables(table, present)
    return _conform(parts, schema)


def _drop_wide(table: pa.Table, variables: Collection[str]) -> list[str]:
    dropped = {f"observation_{v}" for v in variables} | {
        f"quality_{v}" for v in variables
    }
    return [
        name
        for name in table.column_names
        if name not in dropped and not name.startswith("observation_")
    ]


def _stack_variables(table: pa.Table, variables: Sequence[str]) -> list[pa.Table]:
    shared = _drop_wide(table, CONV_VARIABLES)
    parts = []
    for variable in variables:
        observation = table.column(f"observation_{variable}")
        quality_name = f"quality_{variable}"
        names = [*shared, f"observation_{variable}"]
        if quality_name in table.column_names:
            names.append(quality_name)
        part = table.select(names)
        if observation.null_count:
            part = part.filter(pc.is_valid(observation))
        if part.num_rows == 0:
            continue
        renamed = [
            "observation" if n == f"observation_{variable}" else n for n in names
        ]
        renamed = ["quality" if n == quality_name else n for n in renamed]
        part = part.rename_columns(renamed)
        part = part.append_column(
            "variable", constant_dictionary(variable, part.num_rows)
        )
        parts.append(part)
    return parts


def _channel_wavenumber(
    sensor: str, channels: np.ndarray, table: pa.Table
) -> np.ndarray | None:
    if sensor == "atms":
        if len(channels) and (
            channels.min() < 1 or channels.max() > len(_ATMS_WAVENUMBER_CM)
        ):
            raise ValueError(f"atms channels outside 1-{len(_ATMS_WAVENUMBER_CM)}")
        return _ATMS_WAVENUMBER_CM[channels - 1]
    if sensor == "airs":
        metadata = table.schema.metadata or {}
        if _WAVENUMBER_BY_CHANNEL not in metadata:
            raise ValueError("AIRS table carries no wavenumber_by_channel metadata")
        return np.frombuffer(metadata[_WAVENUMBER_BY_CHANNEL])[channels]
    if sensor in _IR_SENSORS:
        return wavenumber_cm_inverse(sensor, channels)
    return None


def _explode_channels(table: pa.Table, variables: Sequence[str]) -> list[pa.Table]:
    sensors = {SAT_VARIABLE_SENSOR[v] for v in variables}
    if len(sensors) != 1:
        raise ValueError(f"a wide satellite table holds one sensor, got {sensors}")
    (sensor,) = sensors
    footprint = [
        name for name in _drop_wide(table, SAT_VARIABLE_SENSOR) if name != "quality"
    ]
    channels = _field_channels(table.schema.field(f"observation_{variables[0]}"))
    n_var = len(variables)
    n_rows = table.num_rows
    per_entry = np.repeat(np.arange(len(channels)), n_var)
    n_slots = n_rows * len(channels)
    index = np.repeat(np.arange(n_rows), len(channels) * n_var)
    columns = {name: table.column(name).take(index) for name in footprint}
    columns["sensor_index"] = pa.array(
        np.tile(channels[per_entry], n_rows), type=pa.uint16()
    )
    wavenumber = _channel_wavenumber(sensor, channels, table)
    if wavenumber is not None:
        columns["wavenumber"] = pa.array(np.tile(wavenumber[per_entry], n_rows))
    quality = pc.list_flatten(_single_chunk(table.column("quality")))
    if n_var > 1:
        quality = quality.take(np.repeat(np.arange(len(quality)), n_var))
    columns["quality"] = quality
    flat = [
        pc.list_flatten(_single_chunk(table.column(f"observation_{v}")))
        for v in variables
    ]
    if n_var > 1:
        stacked = np.stack(
            [v.to_numpy(zero_copy_only=False) for v in flat], axis=1
        ).ravel()
        columns["observation"] = float_with_nulls(stacked, pa.float32())
        codes = np.tile(np.arange(n_var, dtype=np.int8), n_slots)
        columns["variable"] = dictionary_column(codes, list(variables))
    else:
        columns["observation"] = flat[0]
        columns["variable"] = constant_dictionary(variables[0], n_slots)
    return [pa.table(columns)]


def _conform(parts: list[pa.Table], schema: pa.Schema) -> pa.Table:
    """Columns of ``schema`` in order: concatenated parts, nulls where a part lacks one."""
    if not parts:
        return schema.empty_table()
    aligned = []
    for part in parts:
        columns = []
        for field in schema:
            if field.name in part.column_names:
                columns.append(part.column(field.name))
            else:
                columns.append(
                    pa.chunked_array([null_array(field.type, part.num_rows)])
                    if pa.types.is_primitive(field.type)
                    else pa.chunked_array([pa.nulls(part.num_rows, field.type)])
                )
        aligned.append(pa.Table.from_arrays(columns, names=schema.names))
    return concat_tables(aligned)


def concat_tables(tables: Sequence[pa.Table]) -> pa.Table:
    """Concatenate tables; a column absent from some tables is null there.

    A column must have one type across the tables.
    """
    tables = [t for t in tables if t is not None]
    if len(tables) == 1:
        return tables[0]
    if tables and all(t.schema.equals(tables[0].schema) for t in tables[1:]):
        return pa.concat_tables(tables)
    names = list(dict.fromkeys(name for t in tables for name in t.column_names))
    target: dict[str, pa.DataType] = {}
    for name in names:
        types = {t.schema.field(name).type for t in tables if name in t.column_names}
        if len(types) != 1:
            raise TypeError(f"column {name!r} has incompatible types {types}")
        target[name] = types.pop()
    aligned = []
    for t in tables:
        columns = [
            (
                t.column(name)
                if name in t.column_names
                else pa.chunked_array([pa.nulls(t.num_rows, target[name])])
            )
            for name in names
        ]
        aligned.append(pa.Table.from_arrays(columns, names=names))
    return pa.concat_tables(aligned)


def map_aircraft_profile_types(table: pa.Table) -> pa.Table:
    """Map aircraft-profile report types (33x/43x/53x) to GSI aircraft types (13x/23x)."""
    if table.num_rows == 0:
        return table
    report_type = float_values(_single_chunk(table.column("type")))
    wind = pc.is_in(
        _single_chunk(table.column("variable")).cast(pa.string()),
        value_set=pa.array(["u", "v"]),
    ).to_numpy(zero_copy_only=False)
    mapped = report_type.copy()
    for mapping, rows in (
        (_ACFT_PROFILE_WIND_TYPE_MAP, wind),
        (_ACFT_PROFILE_SCALAR_TYPE_MAP, ~wind),
    ):
        for source_type, target_type in mapping.items():
            mapped[rows & (report_type == source_type)] = target_type
    index = table.schema.get_field_index("type")
    return table.set_column(
        index, "type", uint_with_nulls(mapped, table.schema.field("type").type)
    )


# Files decoded at once per source, shared by all its instances and calls; each also
# runs earth2bufr's threads and decode_workers batch threads
_FILES_IN_FLIGHT = 2
_DECODE_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_DECODE_SLOTS_LOCK = threading.Lock()


def _decode_slots(source_id: str) -> threading.BoundedSemaphore:
    with _DECODE_SLOTS_LOCK:
        return _DECODE_SLOTS.setdefault(
            source_id, threading.BoundedSemaphore(_FILES_IN_FLIGHT)
        )


def compile_tables(
    tasks: Sequence[Any],
    source_id: str,
    local_path: Callable[[str], str],
    decode_task: Callable[[str, Any], pa.Table],
    on_error: Callable[[str, int, int, Exception], None] | None = None,
    downloaded: threading.Event | None = None,
) -> list[tuple[Any, pa.Table]]:
    """Decode cached task files into wide tables, each with a ``cycle_time`` column.

    Each file is decoded over the hull of its requested intervals and its rows are
    then cut to the intervals themselves.

    Parameters
    ----------
    tasks : Sequence
        Task objects exposing ``uri``, ``datetime_file`` and ``intervals``
    source_id : str
        Source identifier for log messages
    local_path : Callable[[str], str]
        Maps a URI to its local cache path
    decode_task : Callable[[str, task], pa.Table]
        Decodes one local file given ``(path, task)``
    on_error : Callable[[str, int, int, Exception], None] | None, optional
        Called as ``on_error(uri, idx, n_tasks, exc)`` when ``decode_task``
        raises; the task is then skipped. If None the error propagates.
    downloaded : threading.Event | None, optional
        Set once the downloads have finished. Until then a task waits for its file
        to appear in the cache (written atomically) and is decoded as soon as it
        does. If None every file is expected to be cached already.

    Returns
    -------
    list[tuple[task, pa.Table]]
        Non-empty decoded tables in task order
    """
    default_decode_threads()
    slots = _decode_slots(source_id)
    pieces: list[tuple[Any, pa.Table]] = []
    n_tasks = len(tasks)
    compile_t0 = time.perf_counter()

    def decode(task: Any) -> tuple[pa.Table | Exception | None, float]:
        path = local_path(task.uri)
        if downloaded is not None:
            while not pathlib.Path(path).is_file() and not downloaded.wait(0.05):
                pass
        if not pathlib.Path(path).is_file():
            return None, 0.0
        try:
            with slots:
                t0 = time.perf_counter()
                table = decode_task(path, task)
                seconds = time.perf_counter() - t0
            if len(task.intervals) > 1:
                table = filter_to_intervals(table, task.intervals)
            return table, seconds
        except Exception as exc:
            return exc, 0.0

    decoded = map_ordered(decode, tasks, _FILES_IN_FLIGHT)
    for idx, (task, (table, seconds)) in enumerate(zip(tasks, decoded), start=1):
        uri = task.uri
        if table is None:
            logger.warning(f"Cached file missing for {uri}, skipping")
            continue
        if isinstance(table, Exception):
            # A skipped file is indistinguishable from missing observations, so
            # files are skipped only when the caller passes on_error.
            if on_error is None:
                raise table
            on_error(uri, idx, n_tasks, table)
            logger.error(f"Failed to decode {local_path(uri)}: {table}")
            continue
        logger.info(
            f"[{source_id}] decode {idx}/{n_tasks}: {uri.rsplit('/', 1)[-1]} "
            f"({table.num_rows:,} rows) in {seconds:.1f}s"
        )
        if table.num_rows:
            table = table.append_column(
                "cycle_time", constant_timestamp(task.datetime_file, table.num_rows)
            )
            pieces.append((task, table))
    logger.info(
        f"[{source_id}] decoded {len(pieces)} non-empty files in "
        f"{time.perf_counter() - compile_t0:.1f}s"
    )
    return pieces


def to_dataframe(table: pa.Table, schema: pa.Schema, source_id: str) -> pd.DataFrame:
    """The columns of ``schema`` as a DataFrame of ``pd.ArrowDtype`` columns (zero-copy).

    ``attrs["channels"]`` maps each sensor with per-channel columns in ``schema`` to
    its channel axis (see :func:`channel_axis`).
    """
    columns = []
    for field in schema:
        if field.name in table.column_names:
            columns.append(table.column(field.name))
        else:
            columns.append(pa.chunked_array([pa.nulls(table.num_rows, field.type)]))
    frame = pa.Table.from_arrays(columns, names=schema.names).to_pandas(
        types_mapper=pd.ArrowDtype
    )
    frame.attrs["source"] = source_id
    channels = {
        field.metadata[_SENSOR_KEY].decode(): _field_channels(field).tolist()
        for field in schema
        if field.metadata and _CHANNELS_KEY in field.metadata
    }
    if channels:
        frame.attrs["channels"] = channels
    return frame


# ──────────────────────────────────────────────────────────────────────────
# PrepBUFR
# ──────────────────────────────────────────────────────────────────────────

# variable -> (observation mnemonic, quality mark mnemonic)
_PREPBUFR_VARIABLES = {
    "u": ("UOB", "WQM"),
    "v": ("VOB", "WQM"),
    "q": ("QOB", "QQM"),
    "t": ("TOB", "TQM"),
    "pres": ("POB", "PQM"),
}
_PREPBUFR_HEADER = ("SID", "XOB", "YOB", "DHR", "ELV", "TYP")
_PREPBUFR_LEVEL = ("CAT", "POB", "PQM", "ZOB", "HRDR", "XDR", "YDR")
# Conversions to Earth2Studio units: TOB degC -> K, QOB mg/kg -> kg/kg, POB hPa -> Pa
_PREPBUFR_ADD = {"t": np.float32(273.15)}
_PREPBUFR_MULTIPLY = {"q": np.float32(1e-6), "pres": np.float32(100.0)}


def _round_half_away_div(numerator: np.ndarray, denominator: int) -> np.ndarray:
    quotient, remainder = np.divmod(np.abs(numerator), denominator)
    quotient += remainder * 2 >= denominator
    return np.where(numerator < 0, -quotient, quotient)


def _offset_microseconds(hours: np.ndarray, scale: int) -> np.ndarray:
    """Hour offsets encoded at ``scale`` decimals as microseconds, rounded half away."""
    ticks_per_hour = 10**scale
    ticks = np.rint(hours * ticks_per_hour).astype(np.int64)
    if 3_600_000_000 % ticks_per_hour == 0:
        return ticks * (3_600_000_000 // ticks_per_hour)
    return _round_half_away_div(ticks * 3_600_000_000, ticks_per_hour)


def _message_base_times(info: Any) -> np.ndarray:
    """Section 1 reference time in microseconds per source message index; invalid -> int64 min.

    Edition 3 stores the year of century (pivot 70; BUFR codes 2000 as 100) and no
    seconds.
    """
    header = np.array(
        [
            (
                m.source_message_index,
                m.edition,
                m.year,
                m.month,
                m.day,
                m.hour,
                m.minute,
                m.second,
            )
            for m in info.data_messages
        ],
        dtype=np.int64,
    ).reshape(-1, 8)
    index, edition, year, month, day, hour, minute, second = header.T
    edition3 = edition == 3
    year = np.where(
        edition3 & (year <= 100), year + np.where(year < 70, 2000, 1900), year
    )
    second = np.where(edition3, 0, second)
    stamp_ns, valid = civil_time_ns(
        year.astype(np.float64),
        month.astype(np.float64),
        day.astype(np.float64),
        hour.astype(np.float64),
        minute.astype(np.float64),
        second.astype(np.float64),
    )
    base = np.full(len(info.messages), np.iinfo(np.int64).min, dtype=np.int64)
    base[index] = np.where(valid, stamp_ns // 1000, np.iinfo(np.int64).min)
    return base


def _find_leaf_paths(
    schema: pa.Schema, leaves: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[tuple[str, ...]]] = {}

    def walk(fields: list[pa.Field], prefix: tuple[str, ...]) -> None:
        for field in fields:
            path = (*prefix, field.name)
            field_type = field.type
            if pa.types.is_list(field_type):
                field_type = field_type.value_type
            if pa.types.is_struct(field_type):
                walk(list(field_type), path)
            elif field.name in leaves:
                found.setdefault(field.name, []).append(path)

    walk(list(schema), ())
    duplicated = {name: paths for name, paths in found.items() if len(paths) > 1}
    if duplicated:
        raise ValueError(f"PrepBUFR mnemonics occur more than once: {duplicated}")
    return {name: paths[0] for name, paths in found.items()}


def _field_scale(schema: pa.Schema, path: tuple[str, ...]) -> int:
    container: Any = schema
    field: Any = None
    for name in path:
        field = container.field(name)
        field_type = field.type
        if pa.types.is_list(field_type):
            field_type = field_type.value_type
        container = field_type
    return int(field.metadata[b"bufr.scale"])


class _NestedGather:
    """Gathers nested leaves at rows ``index`` of ``container``.

    Each list on a path contributes its first (or last) entry; an empty list or a
    null leaf gives an invalid entry. Leaves under one struct share an index walk.
    """

    def __init__(
        self,
        container: pa.RecordBatch | pa.StructArray,
        index: np.ndarray,
        take_last: bool,
    ) -> None:
        self.container = container
        self.index = index.astype(np.int64, copy=False)
        self.take_last = take_last
        self._parents: dict[tuple[str, ...], tuple[Any, np.ndarray, np.ndarray]] = {}
        self._leaves: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    def _child(self, array: Any, name: str) -> pa.Array:
        if isinstance(array, pa.RecordBatch):
            return array.column(name)
        return array.field(name)

    def _parent(self, prefix: tuple[str, ...]) -> tuple[Any, np.ndarray, np.ndarray]:
        if prefix in self._parents:
            return self._parents[prefix]
        if not prefix:
            state = (self.container, self.index, np.ones(len(self.index), dtype=bool))
        else:
            array, index, valid = self._parent(prefix[:-1])
            if array is not None:
                array = self._child(array, prefix[-1])
                valid = valid.copy()
                if array.offset:
                    raise ValueError("sliced arrays are not supported")
                if pa.types.is_list(array.type):
                    offsets = array.offsets.to_numpy()
                    starts = offsets[index]
                    ends = offsets[index + 1]
                    valid &= ends > starts
                    if array.null_count:
                        valid &= np.asarray(array.is_valid())[index]
                    index = np.where(valid, ends - 1 if self.take_last else starts, 0)
                    array = array.values if valid.any() else None
                if array is not None and array.null_count:
                    valid &= np.asarray(array.is_valid())[index]
            state = (array, index, valid)
        self._parents[prefix] = state
        return state

    def __call__(self, path: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        path = tuple(path)
        if path not in self._leaves:
            array, index, valid = self._parent(path[:-1])
            if array is None:
                self._leaves[path] = (np.zeros(len(index)), valid)
            else:
                values, leaf_valid = numpy_with_mask(self._child(array, path[-1]))
                self._leaves[path] = (values[index], valid & leaf_valid[index])
        return self._leaves[path]


def _prepbufr_batch(
    batch: pa.RecordBatch,
    obs_class: str,
    variables: Sequence[str],
    base_us: np.ndarray,
    window_us: tuple[int, int],
    take_last: bool,
) -> pa.Table | None:
    """Wide rows (one per report level giving a requested variable) of one batch."""
    leaves = [*_PREPBUFR_HEADER, *_PREPBUFR_LEVEL]
    for variable in variables:
        leaves.extend(n for n in _PREPBUFR_VARIABLES[variable] if n not in leaves)
    paths = _find_leaf_paths(batch.schema, leaves)
    n = batch.num_rows
    subsets = np.arange(n)
    message = batch.column("_source_message_index").to_numpy().astype(np.int64)
    message_base = base_us[message]
    header_gather = _NestedGather(batch, subsets, take_last)

    def header(name: str) -> tuple[np.ndarray, np.ndarray]:
        return header_gather(paths[name])

    dhr, dhr_valid = header("DHR")
    dhr_scale = _field_scale(batch.schema, paths["DHR"])
    if not dhr_valid.all():
        raise ValueError(f"{int((~dhr_valid).sum())} {obs_class} reports with null DHR")
    header_us = message_base + _offset_microseconds(dhr, dhr_scale)
    keep = (header_us >= window_us[0]) & (header_us <= window_us[1])

    cat_path = paths["CAT"]
    container: pa.RecordBatch | pa.StructArray
    if len(cat_path) == 1:
        container = batch
        parent = np.flatnonzero(keep)
        level_index = parent
        level_paths = paths
    else:
        levels = batch.column(cat_path[0])
        offsets = levels.offsets.to_numpy()
        parent = np.repeat(subsets, np.diff(offsets))
        level_index = np.arange(offsets[0], offsets[-1])
        selected = keep[parent]
        parent, level_index = parent[selected], level_index[selected]
        container = levels.values
        level_paths = {}
        for name in leaves[len(_PREPBUFR_HEADER) :]:
            if name in paths:
                if paths[name][0] != cat_path[0]:
                    raise ValueError(
                        f"{name} at {paths[name]} is outside the level list "
                        f"{cat_path[0]}"
                    )
                level_paths[name] = paths[name][1:]

    def level(gather: _NestedGather, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in level_paths:
            size = len(gather.index)
            return np.zeros(size), np.zeros(size, dtype=bool)
        return gather(level_paths[name])

    candidates = _NestedGather(container, level_index, take_last)
    yob, yob_valid = header("YOB")
    xob, xob_valid = header("XOB")
    ydr, ydr_valid = level(candidates, "YDR")
    xdr, xdr_valid = level(candidates, "XDR")
    hrdr, hrdr_valid = level(candidates, "HRDR")
    report_lat = yob.astype(np.float32)
    report_lon = np.mod(xob, 360.0).astype(np.float32)
    lat = np.where(ydr_valid, ydr.astype(np.float32), report_lat[parent])
    lat_valid = ydr_valid | yob_valid[parent]
    lon = np.where(xdr_valid, np.mod(xdr, 360.0).astype(np.float32), report_lon[parent])
    lon_valid = xdr_valid | xob_valid[parent]
    located = (
        lat_valid
        & lon_valid
        & np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )
    parent, level_index = parent[located], level_index[located]
    lat, lon = lat[located], lon[located]
    hrdr, hrdr_valid = hrdr[located], hrdr_valid[located]
    levels_gather = _NestedGather(container, level_index, take_last)
    hrdr_scale = _field_scale(batch.schema, paths["HRDR"]) if "HRDR" in paths else 0
    level_us = np.where(
        hrdr_valid,
        message_base[parent]
        + _offset_microseconds(np.where(hrdr_valid, hrdr, 0.0), hrdr_scale),
        header_us[parent],
    )
    cat, cat_valid = level(levels_gather, "CAT")
    pob, pob_valid = level(levels_gather, "POB")
    pqm, pqm_valid = level(levels_gather, "PQM")
    zob, zob_valid = level(levels_gather, "ZOB")
    typ, typ_valid = header("TYP")

    picks = {}
    for variable in variables:
        obs_name, qm_name = _PREPBUFR_VARIABLES[variable]
        value, value_valid = level(levels_gather, obs_name)
        quality, quality_valid = level(levels_gather, qm_name)
        rows = value_valid
        if variable == "pres":
            if obs_class not in _PRESSURE_CLASSES:
                rows = np.zeros(len(value), dtype=bool)
            else:
                rows = (
                    rows
                    & (value.astype(np.float32) >= _PRESSURE_MIN_HPA)
                    & typ_valid[parent]
                    & np.isin(typ[parent], _PRESSURE_REPORT_TYPES)
                    & cat_valid
                    & (cat == 0)
                    & quality_valid
                    & (quality >= 0)
                    & (quality <= 15)
                )
        picks[variable] = (rows, value, quality, quality_valid)
    if not picks:
        return None
    idx = np.flatnonzero(np.logical_or.reduce([pick[0] for pick in picks.values()]))
    if len(idx) == 0:
        return None
    p = parent[idx]
    k = len(idx)
    elv, elv_valid = header("ELV")
    station = pc.utf8_trim_whitespace(batch.column(paths["SID"][0]))
    station = pc.if_else(pc.equal(station, ""), pa.scalar(None, pa.string()), station)
    station = station.dictionary_encode()

    def uint16(values: np.ndarray, valid: np.ndarray) -> pa.Array:
        return masked(np.where(valid, values, 0).astype(np.uint16), valid, pa.uint16())

    columns: dict[str, pa.Array] = {
        "time": timestamps_ns(level_us[idx] * 1000),
        "report_time": timestamps_ns(header_us[p] * 1000),
        "pres": masked(
            pob[idx].astype(np.float32) * np.float32(100.0),
            pob_valid[idx],
            pa.float32(),
        ),
        "elev": masked(zob[idx].astype(np.float32), zob_valid[idx], pa.float32()),
        "type": uint16(typ[p], typ_valid[p]),
        "level_cat": uint16(cat[idx], cat_valid[idx]),
        "class": constant_dictionary(obs_class, k),
        "lat": pa.array(lat[idx], type=pa.float32()),
        "lon": pa.array(lon[idx], type=pa.float32()),
        "report_lat": masked(report_lat[p], yob_valid[p], pa.float32()),
        "report_lon": masked(report_lon[p], xob_valid[p], pa.float32()),
        "station": station.take(pa.array(p)),
        "station_elev": masked(elv[p].astype(np.float32), elv_valid[p], pa.float32()),
        "pressure_quality": uint16(pqm[idx], pqm_valid[idx]),
    }
    for variable, (rows, value, quality, quality_valid) in picks.items():
        observation = value[idx].astype(np.float32)
        if variable in _PREPBUFR_ADD:
            observation = observation + _PREPBUFR_ADD[variable]
        if variable in _PREPBUFR_MULTIPLY:
            observation = observation * _PREPBUFR_MULTIPLY[variable]
        valid = rows[idx]
        columns[f"observation_{variable}"] = masked(observation, valid, pa.float32())
        columns[f"quality_{variable}"] = uint16(
            quality[idx], valid & quality_valid[idx]
        )
    return pa.table(columns)


_SEQUENCE_MNEMONICS: dict[tuple[str, str], str | None] = {}


def _sequence_mnemonic(source: Any, message: Any, descriptor: str) -> str | None:
    """Table D mnemonic of ``descriptor`` in the tables that decode ``message``; None
    if those tables do not define it. Cached per embedded DX table block."""
    block = source.info.table_epochs[message.table_epoch].embedded_table_sha256
    if block is None or (block, descriptor) not in _SEQUENCE_MNEMONICS:
        structure = source.source_structure(message.source_message_index)
        sequences = {
            s["descriptor"]: s["mnemonic"] for s in structure["active_sequences"]
        }
        if block is None:
            return sequences.get(descriptor)
        for key, mnemonic in sequences.items():
            _SEQUENCE_MNEMONICS[(block, key)] = mnemonic
        _SEQUENCE_MNEMONICS.setdefault((block, descriptor), None)
    return _SEQUENCE_MNEMONICS[(block, descriptor)]


def _prepbufr_message_types(source: Any) -> dict[int, str]:
    """Data category -> Table A mnemonic of each message family in a PrepBUFR file.

    The mnemonic is read from the file's own DX tables, so files that number the
    families differently (230-249 before 2007, 102-121 after) map to the same
    names as ``PREPBUFR_OBS_TYPES``. Categories whose Table A sequence is not one
    of those families are left out.
    """
    families = set(PREPBUFR_OBS_TYPES.values())
    first_message: dict[int, Any] = {}
    for message in source.info.data_messages:
        first_message.setdefault(message.data_category, message)
    types: dict[int, str] = {}
    for category, message in first_message.items():
        table_a = next((d for d in message.descriptors if d.startswith("3-")), None)
        name = None if table_a is None else _sequence_mnemonic(source, message, table_a)
        if name in families:
            types[category] = name
    return types


def decode_prepbufr(
    path: str,
    variables: Sequence[str],
    dt_min: datetime,
    dt_max: datetime,
    decode_workers: int = 8,
    exclude_message_types: Collection[str] = (),
    original_event: bool = False,
) -> pa.Table:
    """Decode a merged NCEP PrepBUFR file into a wide table.

    One row per report level that gives any of ``variables``, in file order, with
    ``observation_<variable>`` and ``quality_<variable>`` columns in Earth2Studio
    units (null where the level has no such observation).

    Parameters
    ----------
    path : str
        Local path to the PrepBUFR file
    variables : Sequence[str]
        Variables among ``u``, ``v``, ``q``, ``t`` and ``pres``
    dt_min, dt_max : datetime
        Closed observation time window. Reports are kept or dropped whole by
        header time, which is every observation's time except drifting sounding
        levels.
    decode_workers : int, optional
        Threads transforming decoded batches, by default 8
    exclude_message_types : Collection[str], optional
        PrepBUFR message families (e.g. ``"SATWND"``) skipped at decode
    original_event : bool, optional
        Take each observation's original report (the last event of its stack)
        instead of the latest, quality-controlled event (the first)

    Returns
    -------
    pa.Table
        Wide table

    Raises
    ------
    ValueError
        If the file has data messages but none of a known PrepBUFR message family,
        or a report has no DHR (observation time offset)
    """
    window_us = (
        datetime_ns(dt_min) // 1000,
        datetime_ns(dt_max) // 1000,
    )
    with open_bufr(path) as source:
        file_types = _prepbufr_message_types(source)
        if source.info.data_messages and not file_types:
            raise ValueError(
                f"{path}: no PrepBUFR message family found in the file's DX tables"
            )
        message_types = {
            category: name
            for category, name in sorted(file_types.items())
            if name not in exclude_message_types
        }
        if not message_types:
            return _empty_wide(_NCEP_PREPBUFR_WIDE_SCHEMA, variables)
        base_us = _message_base_times(source.info)
        batches = iter_batches(
            source,
            path,
            data_category_filter=list(message_types),
            fixed_replication="structural",
            target_decoded_bytes=128 << 20,
            prefetch_batches=2,
        )

        def transform(batch: pa.RecordBatch) -> pa.Table | None:
            category = int(batch.column("_data_category")[0].as_py())
            return _prepbufr_batch(
                batch,
                message_types[category],
                variables,
                base_us,
                window_us,
                original_event,
            )

        tables = [
            t for t in map_ordered(transform, batches, decode_workers) if t is not None
        ]
    if not tables:
        return _empty_wide(_NCEP_PREPBUFR_WIDE_SCHEMA, variables)
    return pa.concat_tables(tables, promote_options="permissive")


def _empty_wide(schema: pa.Schema, variables: Sequence[str]) -> pa.Table:
    """Empty wide table: shared columns plus the columns of ``variables``."""
    return _wide_schema(schema, variables).empty_table()


def _wide_schema(schema: pa.Schema, variables: Collection[str]) -> pa.Schema:
    """``schema``'s shared columns plus the observation and quality columns of
    ``variables``."""
    keep = []
    for field in schema:
        name = field.name
        if name.startswith("observation_") or (
            name.startswith("quality_")
            and name.removeprefix("quality_") in CONV_VARIABLES
        ):
            suffix = name.split("_", 1)[1]
            if suffix not in variables:
                continue
        keep.append(field)
    return pa.schema(keep)


# ──────────────────────────────────────────────────────────────────────────
# GPS radio occultation
# ──────────────────────────────────────────────────────────────────────────
# GPSRO is not a PrepBUFR conventional report, so conventional semantics do not
# apply: there is no TYP report type (the receiver SAID is stored in ``type``,
# matching GSI/UFS diagnostics) and no 0-15 quality mark (QFRO, a radio-occultation
# flag table, is stored in the quality column and must not be read as a QM).

_GPSRO_SCALARS = (
    "SAID",
    "PTID",
    "QFRO",
    "ELRC",
    "GEODU",
    "CLATH",
    "CLONH",
    "YEAR",
    "MNTH",
    "DAYS",
    "HOUR",
    "MINU",
    "SECO",
)
_GPSRO_PROJECTION = [
    *_GPSRO_SCALARS,
    "DRF16BIT.CLATH",
    "DRF16BIT.CLONH",
    "DRF16BIT.DRF8BIT.MEFR",
    "DRF16BIT.DRF8BIT.IMPP",
    "DRF16BIT.DRF8BIT.BNDA",
    "DRF16BIT_2.HEIT",
    "DRF16BIT_2.ARFR",
]


def _first_field(struct: pa.StructArray, name: str) -> tuple[np.ndarray, np.ndarray]:
    # Repeated names (BNDA, ARFR): the first is the value, the second its error
    return numpy_with_mask(struct.field(struct.type.get_all_field_indices(name)[0]))


def _list_entries(lists: pa.ListArray) -> tuple[pa.StructArray, np.ndarray]:
    offsets = lists.offsets.to_numpy()
    if lists.offset or offsets[0] != 0 or offsets[-1] != len(lists.values):
        raise ValueError("list does not span its values")
    lengths = np.diff(offsets)
    parent = np.repeat(np.arange(len(lists)), lengths)
    return lists.values, parent


def _gpsro_batch(
    batch: pa.RecordBatch, variables: Sequence[str], window_ns: tuple[int, int]
) -> pa.Table | None:
    """Wide rows of one batch: bending-angle levels, then refractivity levels."""
    scalar = {name: numpy_with_mask(batch.column(name)) for name in _GPSRO_SCALARS}
    level_struct, level_parent = _list_entries(batch.column("DRF16BIT"))
    frequencies, frequency_level = _list_entries(level_struct.field("DRF8BIT"))
    lat_ref, lat_ref_valid = scalar["CLATH"]
    lon_ref, lon_ref_valid = scalar["CLONH"]

    def part(name: str) -> np.ndarray:
        values, valid = scalar[name]
        return np.where(valid, values, np.nan).astype(np.float64)

    time_ns, time_valid = civil_time_ns(
        part("YEAR"),
        part("MNTH"),
        part("DAYS"),
        part("HOUR"),
        part("MINU"),
        part("SECO"),
    )
    subset_ok = (
        lat_ref_valid
        & lon_ref_valid
        & time_valid
        & (time_ns >= window_ns[0])
        & (time_ns <= window_ns[1])
    )
    said, said_valid = scalar["SAID"]
    ptid, ptid_valid = scalar["PTID"]
    elrc, elrc_valid = scalar["ELRC"]
    geodu, geodu_valid = scalar["GEODU"]
    qfro, qfro_valid = scalar["QFRO"]
    station = pc.binary_join_element_wise(
        pc.utf8_lpad(
            pc.cast(masked(said.astype(np.int64), said_valid, pa.int64()), pa.string()),
            4,
            "0",
        ),
        pc.utf8_lpad(
            pc.cast(masked(ptid.astype(np.int64), ptid_valid, pa.int64()), pa.string()),
            4,
            "0",
        ),
        "",
    ).dictionary_encode()

    tables = []
    for variable in variables:
        if variable == "gps":
            mefr, mefr_valid = _first_field(frequencies, "MEFR")
            impp, impp_valid = _first_field(frequencies, "IMPP")
            value, value_valid = _first_field(frequencies, "BNDA")
            subset = level_parent[frequency_level]
            # One entry per level: its last ionosphere-corrected entry (mean
            # frequency 0), else its last entry
            entry = np.arange(len(frequencies))
            last = np.full(len(level_struct), -1, dtype=np.int64)
            np.maximum.at(last, frequency_level, entry)
            corrected = np.full(len(level_struct), -1, dtype=np.int64)
            is_corrected = mefr_valid & (mefr == 0)
            np.maximum.at(corrected, frequency_level[is_corrected], entry[is_corrected])
            primary = np.where(corrected >= 0, corrected, last)
            rows = primary[primary >= 0]
            rows = rows[
                impp_valid[rows]
                & value_valid[rows]
                & elrc_valid[subset[rows]]
                & subset_ok[subset[rows]]
            ]
            subset = subset[rows]
            level_index = frequency_level[rows]
            level_lat, level_lat_valid = numpy_with_mask(level_struct.field("CLATH"))
            level_lon, level_lon_valid = numpy_with_mask(level_struct.field("CLONH"))
            lat = np.where(
                level_lat_valid[level_index], level_lat[level_index], lat_ref[subset]
            )
            lon = np.where(
                level_lon_valid[level_index], level_lon[level_index], lon_ref[subset]
            )
            elev = (impp[rows] - elrc[subset]).astype(np.float32)
            value = value[rows]
        else:
            refractivity, refractivity_subset = _list_entries(
                batch.column("DRF16BIT_2")
            )
            heit, heit_valid = _first_field(refractivity, "HEIT")
            value, value_valid = _first_field(refractivity, "ARFR")
            rows = np.flatnonzero(
                heit_valid & value_valid & subset_ok[refractivity_subset]
            )
            subset = refractivity_subset[rows]
            lat = lat_ref[subset]
            lon = lon_ref[subset]
            elev = heit[rows].astype(np.float32)
            value = value[rows]
        k = len(rows)
        if k == 0:
            continue
        columns: dict[str, pa.Array] = {
            "time": timestamps_ns(time_ns[subset]),
            "elev": pa.array(elev, type=pa.float32()),
            "type": masked(
                np.where(said_valid, said, 0).astype(np.uint16)[subset],
                said_valid[subset],
                pa.uint16(),
            ),
            "class": constant_dictionary("GPSRO", k),
            "lat": pa.array(lat.astype(np.float32), type=pa.float32()),
            "lon": pa.array(np.mod(lon, 360.0).astype(np.float32), type=pa.float32()),
            "station": station.take(pa.array(subset)),
            "radius_curvature": masked(elrc[subset], elrc_valid[subset], pa.float64()),
            "geoid_undulation": masked(
                geodu[subset], geodu_valid[subset], pa.float64()
            ),
        }
        quality = masked(
            np.where(qfro_valid, qfro, 0).astype(np.uint16)[subset],
            qfro_valid[subset],
            pa.uint16(),
        )
        for other in variables:
            if other == variable:
                columns[f"observation_{other}"] = pa.array(
                    value.astype(np.float32), type=pa.float32()
                )
                columns[f"quality_{other}"] = quality
            else:
                columns[f"observation_{other}"] = null_array(pa.float32(), k)
                columns[f"quality_{other}"] = null_array(pa.uint16(), k)
        tables.append(pa.table(columns))
    if not tables:
        return None
    return pa.concat_tables(tables)


def decode_gpsro(
    path: str,
    variables: Sequence[str],
    dt_min: datetime,
    dt_max: datetime,
    decode_workers: int = 8,
) -> pa.Table:
    """Decode an NCEP GPSRO BUFR file into a wide table.

    One row per bending-angle level (``observation_gps``, the level's last
    ionosphere-corrected entry, else its last entry) or refractivity level
    (``observation_gps_refractivity``).

    Parameters
    ----------
    path : str
        Local path to the GPSRO BUFR file
    variables : Sequence[str]
        Variables among ``gps`` and ``gps_refractivity``
    dt_min, dt_max : datetime
        Closed observation time window
    decode_workers : int, optional
        Threads transforming decoded batches, by default 8

    Returns
    -------
    pa.Table
        Wide table
    """
    window_ns = (datetime_ns(dt_min), datetime_ns(dt_max))
    with open_bufr(path) as source:
        batches = iter_batches(
            source,
            path,
            projection=_GPSRO_PROJECTION,
            fixed_replication="structural",
            target_decoded_bytes=32 << 20,
            prefetch_batches=2,
        )
        tables = [
            t
            for t in map_ordered(
                lambda b: _gpsro_batch(b, variables, window_ns), batches, decode_workers
            )
            if t is not None
        ]
    if not tables:
        return _empty_wide(_NCEP_GPSRO_WIDE_SCHEMA, variables)
    return pa.concat_tables(tables, promote_options="permissive")


# ──────────────────────────────────────────────────────────────────────────
# Satellite winds
# ──────────────────────────────────────────────────────────────────────────
# Each mnemonic is read from its first top-level occurrence that is populated in
# some wind of the same template in the file, for every wind of that template.
# Later occurrences hold other quantities: alternate height assignments, model
# wind levels, the pressure uncertainty (PRLC), the contributing satellite
# (SAID, SAZA) or intermediate image-pair vectors (WDIR, WSPD).

_SATWND_TIME = ("YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO")
_SATWND_SCALARS = (
    "SAID",
    "SWCM",
    "HAMD",
    "PRLC",
    "SAZA",
    "WDIR",
    "WSPD",
    "SWQM",
    "CMCM",
    "CLATH",
    "CLAT",
    "CLONH",
    "CLON",
)
_SATWND_QUALITY_CODES = {
    "GNAP": "quality_indicators",
    "GNAPS": "amv_quality_indicators",
}
_SATWND_QUALITY_FIELDS = ["GNAP", "GNAPS", "PCCF"]
_SATWND_PROJECTION = [*_SATWND_TIME, *_SATWND_SCALARS]
_SATWND_INTS = {
    "satellite_id": ("SAID", pa.uint16()),
    "quality": ("SWQM", pa.uint16()),
    "wind_method": ("SWCM", pa.uint8()),
    "wind_method_local": ("CMCM", pa.uint8()),
    "height_method": ("HAMD", pa.uint8()),
}
_SATWND_FLOATS = {"pres": "PRLC", "satellite_za": "SAZA"}


@dataclass(frozen=True)
class _SatwndPlan:
    columns: dict[str, list[int]]
    pairs: dict[str, list[tuple[int, int]]]


def _satwnd_plan(schema: pa.Schema) -> _SatwndPlan:
    """Column indices of each mnemonic's top-level occurrences and the
    GNAP/GNAPS -> PCCF pairs of a template."""
    columns: dict[str, list[int]] = {}
    for index, field in enumerate(schema):
        if field.name in (*_SATWND_SCALARS, *_SATWND_TIME) and not pa.types.is_nested(
            field.type
        ):
            columns.setdefault(field.name, []).append(index)
    events = [
        (f.name, i) for i, f in enumerate(schema) if f.name in ("GNAP", "GNAPS", "PCCF")
    ]
    pairs = {
        code: [
            (events[i][1], events[i + 1][1])
            for i in range(len(events) - 1)
            if events[i][0] == code and events[i + 1][0] == "PCCF"
        ]
        for code in _SATWND_QUALITY_CODES
    }
    return _SatwndPlan(columns, pairs)


@dataclass(frozen=True)
class _SatwndSlots:
    """Per mnemonic, the first populated occurrence (None: none is)."""

    scalars: dict[str, int | None]

def _satwnd_slots(batch: pa.RecordBatch, plan: _SatwndPlan) -> _SatwndSlots:
    n = batch.num_rows
    return _SatwndSlots(
        {
            name: next(
                (k for k, i in enumerate(indices) if batch.column(i).null_count < n),
                None,
            )
            for name, indices in plan.columns.items()
        }
    )


def _merge_slots(a: _SatwndSlots, b: _SatwndSlots) -> _SatwndSlots:
    """Choices over the winds of both batches of one template."""
    return _SatwndSlots(
        {
            name: min(
                (k for k in (a.scalars[name], b.scalars[name]) if k is not None),
                default=None,
            )
            for name in a.scalars
        }
    )


def _satwnd_reslot(
    table: pa.Table, plan: _SatwndPlan, slots: _SatwndSlots, file_slots: _SatwndSlots
) -> pa.Table | None:
    """``_satwnd_batch``'s table of a batch as ``file_slots`` give it. A batch's slot is
    the first occurrence it populates, so an earlier file slot reads an occurrence the
    batch leaves empty: the columns read from it are null, and with a required field
    empty no wind is kept."""
    moved = {
        name
        for name, k in slots.scalars.items()
        if k is not None and file_slots.scalars[name] != k
    }
    lat = "CLATH" if "CLATH" in plan.columns else "CLAT"
    lon = "CLONH" if "CLONH" in plan.columns else "CLON"
    if moved & {lat, lon, "WDIR", "WSPD", *_SATWND_TIME}:
        return None
    sources = {**_SATWND_FLOATS, **{out: n for out, (n, _) in _SATWND_INTS.items()}}
    for out, name in sources.items():
        if name in moved:
            index = table.schema.get_field_index(out)
            table = table.set_column(
                index, out, pa.nulls(table.num_rows, table.schema.field(out).type)
            )
    return table


def _satwnd_quality(
    batch: pa.RecordBatch, pairs: list[tuple[int, int]], keep: np.ndarray
) -> pa.Array:
    """Per wind, the (application, confidence) pairs; a repeated application keeps its first."""
    n = int(keep.sum())
    if not pairs:
        return pa.nulls(n, _QUALITY_LIST)
    codes = np.empty((n, len(pairs)), dtype=np.int64)
    confidence = np.empty((n, len(pairs)), dtype=np.float32)
    valid = np.empty((n, len(pairs)), dtype=bool)
    for k, (code_column, pccf_column) in enumerate(pairs):
        code, code_valid = numpy_with_mask(batch.column(code_column))
        pccf, pccf_valid = numpy_with_mask(batch.column(pccf_column))
        codes[:, k] = code[keep]
        confidence[:, k] = pccf[keep]
        valid[:, k] = (code_valid & pccf_valid)[keep]
    for k in range(1, len(pairs)):
        valid[:, k] &= ~(valid[:, :k] & (codes[:, :k] == codes[:, k : k + 1])).any(
            axis=1
        )
    counts = valid.sum(axis=1)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    entries = pa.StructArray.from_arrays(
        [pa.array(codes[valid].astype(np.uint16)), pa.array(confidence[valid])],
        fields=list(_QUALITY_LIST.value_type),
    )
    return pa.ListArray.from_arrays(
        pa.array(offsets), entries, type=_QUALITY_LIST, mask=pa.array(counts == 0)
    )


def _satwnd_templates(source: Any) -> tuple[np.ndarray, np.ndarray, pa.Array]:
    """Per message, its template index and its dump subset code, and the subset
    names. A template is a (descriptors, table epoch); its subset is the Table D
    mnemonic of its root descriptor, from the file's DX tables."""
    template = np.zeros(len(source.info.messages), dtype=np.int64)
    subset = np.zeros(len(source.info.messages), dtype=np.int16)
    names: dict[str, int] = {}
    templates: dict[tuple, tuple[int, int]] = {}
    for message in source.info.data_messages:
        descriptors = tuple(message.descriptors)
        key = (descriptors, message.table_epoch)
        if key not in templates:
            root = descriptors[1] if len(descriptors) > 1 else descriptors[0]
            mnemonic = _sequence_mnemonic(source, message, root)
            if mnemonic is None:
                raise ValueError(f"Root descriptor {root} is not in the DX tables")
            templates[key] = (len(templates), names.setdefault(mnemonic, len(names)))
        template[message.source_message_index], subset[message.source_message_index] = (
            templates[key]
        )
    return template, subset, pa.array(list(names), type=pa.string())


def _satwnd_batch(
    batch: pa.RecordBatch,
    plan: _SatwndPlan,
    slots: _SatwndSlots,
    subset_code: np.ndarray,
    subset_names: pa.Array,
    variables: Sequence[str],
    window_ns: tuple[int, int],
) -> pa.Table | None:
    n = batch.num_rows
    column = {
        name: batch.column(indices[slots.scalars[name] or 0])
        for name, indices in plan.columns.items()
    }

    def floats(name: str) -> np.ndarray:
        array = column.get(name)
        return np.full(n, np.nan) if array is None else float_values(array)

    lat = floats("CLATH" if "CLATH" in column else "CLAT")
    lon = floats("CLONH" if "CLONH" in column else "CLON")
    wdir, wspd = floats("WDIR"), floats("WSPD")
    parts = [floats(name) for name in _SATWND_TIME]
    stamp, ok = civil_time_ns(*parts)
    keep = ok & (stamp >= window_ns[0]) & (stamp <= window_ns[1])
    for required in (lat, lon, wdir, wspd):
        keep &= np.isfinite(required)
    if not keep.any():
        return None
    k = int(keep.sum())
    arrow_keep = pa.array(keep)
    radians = wdir[keep] * (np.pi / 180.0)
    speed = -wspd[keep]
    columns: dict[str, pa.Array] = {
        "time": timestamps_ns(stamp[keep]),
        "lat": pa.array(lat[keep].astype(np.float32)),
        "lon": pa.array(np.mod(lon[keep], 360.0).astype(np.float32)),
        "class": constant_dictionary("SATWND", k),
    }
    for out, name in _SATWND_FLOATS.items():
        array = column.get(name)
        columns[out] = (
            null_array(pa.float32(), k)
            if array is None
            else pc.cast(array, pa.float32()).filter(arrow_keep)
        )
    # A non-positive assigned pressure is missing
    columns["pres"] = pc.if_else(
        pc.greater(columns["pres"], 0), columns["pres"], pa.scalar(None, pa.float32())
    )
    for out, (name, arrow_type) in _SATWND_INTS.items():
        array = column.get(name)
        columns[out] = (
            null_array(arrow_type, k)
            if array is None
            else pc.cast(array, arrow_type).filter(arrow_keep)
        )
    message = batch.column("_source_message_index").to_numpy()[keep]
    columns["subset"] = pa.DictionaryArray.from_arrays(
        pa.array(subset_code[message]), subset_names
    )
    for code, out in _SATWND_QUALITY_CODES.items():
        columns[out] = _satwnd_quality(batch, plan.pairs[code], keep)
    # Meteorological convention: the direction the wind blows from
    components = {
        "u": (speed * np.sin(radians)).astype(np.float32),
        "v": (speed * np.cos(radians)).astype(np.float32),
    }
    for variable in variables:
        columns[f"observation_{variable}"] = pa.array(components[variable])
    return pa.table(columns)


def decode_satwnd(
    path: str,
    variables: Sequence[str],
    dt_min: datetime,
    dt_max: datetime,
    decode_workers: int = 8,
    quality_indicators: bool = True,
) -> pa.Table:
    """Decode an NCEP satwnd dump into a wide table, one row per wind.

    Parameters
    ----------
    path : str
        Local path to the satwnd BUFR file
    variables : Sequence[str]
        Variables among ``u`` and ``v``
    dt_min, dt_max : datetime
        Closed observation time window
    decode_workers : int, optional
        Threads transforming decoded batches, by default 8
    quality_indicators : bool, optional
        Decode the GNAP/GNAPS quality indicators; when False their columns are
        null. By default True

    Returns
    -------
    pa.Table
        Wide table
    """
    window_ns = (datetime_ns(dt_min), datetime_ns(dt_max))

    def transform(batch: pa.RecordBatch) -> tuple:
        plan = _satwnd_plan(batch.schema)
        slots = _satwnd_slots(batch, plan)
        table = _satwnd_batch(
            batch, plan, slots, subset_code, subset_names, variables, window_ns
        )
        index = int(batch.column("_source_message_index")[0].as_py())
        return int(template[index]), slots, table, plan

    with open_bufr(path) as source:
        template, subset_code, subset_names = _satwnd_templates(source)
        # Fields differ by template; a template's missing fields are left out
        batches = iter_batches(
            source,
            path,
            projection=_SATWND_PROJECTION
            + (_SATWND_QUALITY_FIELDS if quality_indicators else []),
            projection_missing="skip",
            target_decoded_bytes=256 << 20,
            prefetch_batches=2,
        )
        results = list(map_ordered(transform, batches, decode_workers))
    file_slots: dict[int, _SatwndSlots] = {}
    for key, slots, _, _ in results:
        file_slots[key] = (
            _merge_slots(file_slots[key], slots) if key in file_slots else slots
        )
    tables = []
    for key, slots, table, plan in results:
        if table is not None and file_slots[key] != slots:
            table = _satwnd_reslot(table, plan, slots, file_slots[key])
        if table is not None:
            tables.append(table)
    if not tables:
        return _empty_wide(_NCEP_SATWND_WIDE_SCHEMA, variables)
    return pa.concat_tables(tables, promote_options="permissive")


# ──────────────────────────────────────────────────────────────────────────
# Aggregate satellite sounders (microwave and hyperspectral infrared)
# ──────────────────────────────────────────────────────────────────────────

_SAT_TIME = ["YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"]
_SAT_GEOMETRY = ["CLATH", "CLONH", "SAZA", "SOZA", "BEARAZ", "SOLAZI"]
_IR_SENSORS = frozenset({"airs", "iasi", "cris"})
_IR_SCALARS = ["SAID", *_SAT_TIME, "SLNM", "FOVN", *_SAT_GEOMETRY]
_IR_OBSERVATION = {"cris": "SRAD", "iasi": "SCRA", "airs": "TMBR"}
_IR_CHANNEL_LIMIT = {"airs": 2378, "iasi": 8461, "cris": 2211}
# ATMSGQ, ATMSSQ, ATMSCHQ: 0-33-079 granule, 0-33-080 scan, 0-33-081 channel quality
_ATMS_SCALARS = ["SAID", *_SAT_TIME, "SLNM", "FOVN", *_SAT_GEOMETRY, "ATMSGQ", "ATMSSQ"]
_CRIS_BAND_GROUP = "fixed_1_07_003__occ_0"
_CRIS_CHANNEL_BAND = np.full(CRIS_BANDS[-1][2] + 1, -1, dtype=np.int64)
for _band, (_, _first, _last) in enumerate(CRIS_BANDS):
    _CRIS_CHANNEL_BAND[_first : _last + 1] = _band
# Schema metadata of a wide AIRS table: the file's centre wavenumber in cm⁻¹ as
# float64 bytes indexed by channel number, NaN for channels the file lacks
_WAVENUMBER_BY_CHANNEL = b"wavenumber_by_channel"


def _airs_wavenumber(source: Any, path: str) -> np.ndarray:
    """Centre wavenumber (cm⁻¹) by channel number from the first footprint whose
    channels all carry LOGRCW; LOGRCW is constant per channel within a file.

    Raises
    ------
    ValueError
        If the file has footprints but none carries LOGRCW for all its channels
    """
    by_channel = np.full(_IR_CHANNEL_LIMIT["airs"] + 1, np.nan)
    footprints = 0
    batches = iter_batches(
        source,
        path,
        messages_per_batch=1,
        data_category_filter=[21],
        projection=["DRF16BIT.CHNM", "DRF16BIT.LOGRCW"],
    )
    for batch in batches:
        offsets, values = _list_layout(_channel_group(batch))
        footprints += len(offsets) - 1
        channels = physical_values(values, "CHNM")
        log10_m = physical_values(values, "LOGRCW")
        for start, end in zip(offsets[:-1], offsets[1:]):
            footprint = slice(start, end)
            if (
                end > start
                and np.isfinite(channels[footprint]).all()
                and np.isfinite(log10_m[footprint]).all()
            ):
                # LOGRCW is log10 of the centre wavenumber in m⁻¹
                by_channel[channels[footprint].astype(np.int64)] = (
                    10.0 ** log10_m[footprint] / 100.0
                )
                return by_channel
    if footprints:
        raise ValueError(f"{path}: no AIRS footprint carries LOGRCW for all channels")
    return by_channel


def _sat_decode_options(
    sensor: str, fields: Collection[str], channels: Collection[int] | None
) -> dict[str, Any]:
    """earth2bufr options: only the requested fields and channels are decoded."""
    if sensor == "atms":
        channel_fields = [
            f"DRF16BIT.{f}" for f in ("CHNM", "SCCF", "ATMSCHQ", *sorted(fields))
        ]
        return {
            "projection": _ATMS_SCALARS + channel_fields,
            "raw_code_paths": channel_fields,
        }
    if sensor not in _IR_SENSORS:
        # 1bamua / 1bmhs / 1bamub: one fixed channel group, found by its CHNM child
        return {"fixed_replication": "structural"}
    observation = f"DRF16BIT.{_IR_OBSERVATION[sensor]}"
    options: dict[str, Any] = {}
    if sensor == "airs":
        projection = _IR_SCALARS + [
            "DRF16BIT.CHNM",
            "DRF16BIT.ACQF",
            observation,
        ]
    else:
        options["raw_code_paths"] = ["DRF16BIT.CHNM", observation]
        if sensor == "iasi":
            projection = _IR_SCALARS + ["QGFQ", "STCH", "ENCH", "CHSF"]
        elif channels is None:
            projection = _IR_SCALARS + [
                "FORN",
                "HOLS",
                "NSQF",
                f"{_CRIS_BAND_GROUP}.NFQF",
                f"{_CRIS_BAND_GROUP}.NCQF",
            ]
            options["fixed_replication"] = "structural"
        else:
            projection = _IR_SCALARS + ["FORN", "HOLS", "NSQF", "NFQF", "NCQF"]
        projection = projection + ["DRF16BIT.CHNM", observation]
    if channels is not None:
        options["list_filters"] = [list_value_filter("DRF16BIT", "CHNM", channels)]
    return {"projection": projection, **options}


def _column(batch: pa.RecordBatch, name: str) -> np.ndarray | None:
    """First top-level occurrence of a field as float64 (NaN = missing); None if absent."""
    indices = batch.schema.get_all_field_indices(name)
    return float_values(batch.column(indices[0])) if indices else None


def _required_column(batch: pa.RecordBatch, name: str) -> np.ndarray:
    values = _column(batch, name)
    if values is None:
        raise ValueError(f"decoded batch lacks the {name} field")
    return values


def _footprint_latlon(batch: pa.RecordBatch) -> tuple[np.ndarray, np.ndarray]:
    """CLATH/CLONH where both are finite, else CLAT/CLON; templates carry one or both."""
    lat, lon = _column(batch, "CLATH"), _column(batch, "CLONH")
    coarse_lat, coarse_lon = _column(batch, "CLAT"), _column(batch, "CLON")
    if lat is None or lon is None:
        return _required_column(batch, "CLAT"), _required_column(batch, "CLON")
    if coarse_lat is None or coarse_lon is None:
        return lat, lon
    high = np.isfinite(lat) & np.isfinite(lon)
    return np.where(high, lat, coarse_lat), np.where(high, lon, coarse_lon)


def _columns_at(batch: pa.RecordBatch, name: str) -> list[np.ndarray]:
    return [
        float_values(batch.column(i)) for i in batch.schema.get_all_field_indices(name)
    ]


def _channel_group(batch: pa.RecordBatch) -> pa.Array:
    for index, field in enumerate(batch.schema):
        if _is_list(field.type) and pa.types.is_struct(field.type.value_type):
            if field.type.value_type.get_field_index("CHNM") >= 0:
                return batch.column(index)
    raise ValueError("no channel group with CHNM in the decoded batch")


def _list_layout(lists: pa.Array) -> tuple[np.ndarray, pa.StructArray]:
    if lists.offset:
        raise ValueError("sliced list arrays are not supported")
    if pa.types.is_fixed_size_list(lists.type):
        width = lists.type.list_size
        return np.arange(len(lists) + 1, dtype=np.int64) * width, lists.values
    offsets = lists.offsets.to_numpy()
    if offsets[0] != 0:
        raise ValueError("sliced list arrays are not supported")
    return offsets, lists.values


class _ChannelMatrix:
    """Channel entries of one batch's kept footprints on a fixed channel axis.

    ``index`` is, per kept footprint and slot, the entry in the list values, -1 where
    the footprint lacks the slot's channel; it is None when every footprint is kept
    and carries exactly the axis in order, so the entries are already footprint-major.
    A batch whose kept footprints each carry exactly the axis in order is a reshape of
    its entries; otherwise entries are placed by channel number. Footprints with no
    channel of the axis are not kept; ``off_axis`` counts the entries whose channel is
    not on the axis.
    """

    def __init__(self, lists: pa.Array, keep: np.ndarray, axis: np.ndarray) -> None:
        offsets, self.values = _list_layout(lists)
        lengths = np.diff(offsets)
        chnm = physical_values(self.values, "CHNM")
        self.axis = axis
        self.off_axis = 0
        width = len(axis)
        kept = np.flatnonzero(keep & (lengths > 0))
        shape = (len(kept), width)
        if (
            len(kept) == len(lengths)
            and len(chnm) == len(lengths) * width
            and np.all(lengths == width)
            and np.array_equal(chnm.reshape(shape), np.broadcast_to(axis, shape))
        ):
            self.kept, self.index, self.complete = kept, None, True
            return
        if np.all(lengths[kept] == width):
            index = offsets[kept][:, None] + np.arange(width)
            if np.array_equal(chnm[index], np.broadcast_to(axis, index.shape)):
                self.kept, self.index, self.complete = kept, index, True
                return
        parent = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        slot = np.minimum(np.searchsorted(axis, chnm), width - 1)
        on_axis = keep[parent] & (axis[slot] == chnm)
        self.off_axis = int((keep[parent] & np.isfinite(chnm) & ~on_axis).sum())
        entries = np.flatnonzero(on_axis)
        has_channel = np.zeros(len(lengths), dtype=bool)
        has_channel[parent[entries]] = True
        self.kept = np.flatnonzero(has_channel)
        row = np.cumsum(has_channel) - 1
        self.index = np.full((len(self.kept), width), -1, dtype=np.int64)
        self.index[row[parent[entries]], slot[entries]] = entries
        self.complete = False

    @property
    def shape(self) -> tuple[int, int]:
        """(kept footprints, channel axis)"""
        return len(self.kept), len(self.axis)

    def take(self, name: str, dtype: type = np.float64) -> np.ndarray:
        """Physical values of child ``name`` per slot, footprint-major; NaN where missing."""
        if self.index is None:
            return physical_values(self.values, name, None, dtype)
        index = self.index.ravel()
        if self.complete:
            return physical_values(self.values, name, index, dtype)
        missing = index < 0
        values = physical_values(self.values, name, np.where(missing, 0, index), dtype)
        values[missing] = np.nan
        return values

    def present(self) -> np.ndarray:
        """Per footprint and slot, whether the footprint carries the slot's channel."""
        if self.complete:
            return np.ones(self.shape, dtype=bool)
        return self.index >= 0

    def lists(self, flat: pa.Array) -> pa.Array:
        return pa.FixedSizeListArray.from_arrays(flat, len(self.axis))


def _cris_band_quality(batch: pa.RecordBatch) -> tuple[np.ndarray, np.ndarray]:
    """Per footprint and band slot i (LW, MW, SW): NFQF | NCQF << 19, valid where NFQF is set."""
    if _CRIS_BAND_GROUP in batch.schema.names:
        group = batch.column(_CRIS_BAND_GROUP).values
        nfqf = float_values(group.field("NFQF")).reshape(-1, 3)
        ncqf = float_values(group.field("NCQF")).reshape(-1, 3)
    else:
        nfqf = np.stack(_columns_at(batch, "NFQF"), axis=1)
        ncqf = np.stack(_columns_at(batch, "NCQF"), axis=1)
    valid = np.isfinite(nfqf)
    packed = np.where(valid, nfqf, 0.0).astype(np.uint32) | (
        np.where(np.isfinite(ncqf), ncqf, 0.0).astype(np.uint32) << 19
    )
    return packed, valid


def _iasi_scale_exponent(
    batch: pa.RecordBatch,
    rows: _ChannelMatrix,
    coded: np.ndarray,
    groups: dict[bytes, np.ndarray],
) -> np.ndarray:
    """CHSF per slot from the one STCH..ENCH group holding the slot's channel; NaN where
    that group's CHSF is null. ``groups`` caches each bounds layout's group per
    slot across a file's batches.

    Raises
    ------
    ValueError
        If a bound is null, a channel of the axis is not in exactly one group, or
        CHSF is null for a nonzero SCRA
    """
    # Scale groups precede the channel list; the AVHRR cluster CHSF fields follow it
    end = batch.schema.get_field_index("DRF16BIT")
    start, stop, scale = (
        np.stack(
            [
                float_values(batch.column(i))
                for i in batch.schema.get_all_field_indices(name)
                if i < end
            ],
            axis=1,
        )[rows.kept]
        for name in ("STCH", "ENCH", "CHSF")
    )
    if not (np.isfinite(start).all() and np.isfinite(stop).all()):
        raise ValueError("IASI scale-group bounds contain nulls")
    bounds = np.concatenate([start, stop], axis=1)
    # Footprints share few bounds layouts, usually one
    if (bounds == bounds[0]).all():
        layouts, layout_of = bounds[:1], np.zeros(len(bounds), dtype=np.int64)
    else:
        layouts, layout_of = np.unique(bounds, axis=0, return_inverse=True)
    group_of = np.empty((len(layouts), len(rows.axis)), dtype=np.int64)
    for i, layout in enumerate(layouts):
        key = layout.tobytes()
        if key not in groups:
            first, last = np.split(layout, 2)
            member = (rows.axis[None, :] >= first[:, None]) & (
                rows.axis[None, :] <= last[:, None]
            )
            if not (member.sum(axis=0) == 1).all():
                raise ValueError("IASI scale groups do not hold each channel once")
            groups[key] = member.argmax(axis=0)
        group_of[i] = groups[key]
    if len(layouts) == 1:
        exponent = scale[:, group_of[0]].ravel()
    else:
        exponent = np.take_along_axis(
            scale, group_of[layout_of.ravel()], axis=1
        ).ravel()
    unscaled = ~np.isfinite(exponent) & np.isfinite(coded) & (coded != 0.0)
    if unscaled.any():
        raise ValueError("IASI scale factor CHSF is null for a nonzero SCRA")
    return exponent


def _satellite_name(said: int) -> str:
    return _NCEP_SATELLITE_NAME_BY_SAID.get(int(said), f"satellite-{int(said)}")


def _satellite_column(said: np.ndarray) -> pa.Array:
    """Platform names as a dictionary of the SAIDs present in ``said``."""
    # A batch usually holds one platform
    if len(said) and (said == said[0]).all():
        return constant_dictionary(_satellite_name(said[0]), len(said))
    codes, index = np.unique(said.astype(np.int64), return_inverse=True)
    return dictionary_column(
        index.ravel().astype(np.int8), [_satellite_name(s) for s in codes]
    )


def _footprint_time_ns(parts: list[np.ndarray]) -> np.ndarray:
    """Footprint times from YEAR..SECO; 24:00:00 is midnight of the next day.

    Raises
    ------
    ValueError
        If a part is missing, a calendar part is not an integer, or the time is out
        of range
    """
    if not all(np.isfinite(part).all() for part in parts):
        raise ValueError("satellite footprint time has missing parts")
    if not all(np.array_equal(part, np.rint(part)) for part in parts[:5]):
        raise ValueError("satellite footprint time has non-integer parts")
    year, month, day, hour, minute, second = parts
    midnight = (hour == 24) & (minute == 0) & (second == 0)
    time_ns, ok = civil_time_ns(
        year, month, day, np.where(midnight, 0.0, hour), minute, second
    )
    if not (ok & (second >= 0.0) & (second < 61.0)).all():
        raise ValueError("satellite footprint time is out of range")
    return time_ns + np.where(midnight, 86_400 * 10**9, 0)


def _sat_batch(
    batch: pa.RecordBatch,
    sensor: str,
    plan: Mapping[str, str],
    window_ns: tuple[int, int],
    satellites: frozenset[str] | None,
    axis: np.ndarray,
    iasi_groups: dict[bytes, np.ndarray],
) -> pa.Table | None:
    """Wide rows (one per kept footprint) of one batch over the channel ``axis``.

    ``plan`` maps each requested variable to its BUFR field.
    """
    # Times decide the window, so every footprint's is checked; SAID and position only
    # where the window, then the satellites, keep the footprint
    time_ns = _footprint_time_ns([_required_column(batch, name) for name in _SAT_TIME])
    lat, lon = _footprint_latlon(batch)
    said = _required_column(batch, "SAID")
    is_cris = sensor == "cris"
    position = _required_column(batch, "FORN" if is_cris else "FOVN")
    keep = (time_ns >= window_ns[0]) & (time_ns <= window_ns[1]) & np.isfinite(position)
    if not np.isfinite(said[keep]).all():
        raise ValueError("satellite footprint has no SAID")
    if satellites is not None:
        said_codes, said_index = np.unique(
            np.where(keep, said, 0).astype(np.int64), return_inverse=True
        )
        keep &= np.array(
            [_satellite_name(s) in satellites for s in said_codes], dtype=bool
        )[said_index.ravel()]
    if not (np.isfinite(lon[keep]).all() and (np.abs(lat[keep]) <= 90.0).all()):
        raise ValueError(
            "satellite footprint latitude or longitude is missing or out of range"
        )
    rows = _ChannelMatrix(_channel_group(batch), keep, axis)
    if rows.off_axis:
        logger.warning(
            f"{sensor}: dropped {rows.off_axis} channel entries off the channel axis"
        )
    if not len(rows.kept):
        return None
    values = rows.values
    child_names = {values.type.field(i).name for i in range(values.type.num_fields)}
    # Values are footprint-major (kept footprints x axis); channel constants broadcast
    channels = rows.axis.astype(np.int64)
    slots = len(rows.kept) * len(channels)

    if sensor in _IR_SENSORS:
        scan_angle: np.ndarray | None = None
        elev = _column(batch, "HOLS") if is_cris else None
    else:
        start, step = _NCEP_MICROWAVE_SCAN_GEOMETRY[sensor]
        scan_angle = start + (position - 1.0) * step
        elev = _column(batch, "HOLS")

    observations: dict[str, np.ndarray] = {}
    if sensor == "airs":
        quality = rows.take("ACQF")
        for variable, field in plan.items():
            observations[variable] = rows.take(field, np.float32)
    elif sensor in ("iasi", "cris"):
        wavenumber = wavenumber_cm_inverse(sensor, channels)
        coded = rows.take(_IR_OBSERVATION[sensor])
        if is_cris:
            radiance = cris_radiance_mw(coded)
            packed, packed_ok = _cris_band_quality(batch)
            band = _CRIS_CHANNEL_BAND[channels]
            quality = np.where(
                packed_ok[rows.kept][:, band] & rows.present(),
                packed[rows.kept][:, band],
                np.nan,
            ).ravel()
        else:
            exponent = _iasi_scale_exponent(batch, rows, coded, iasi_groups)
            radiance = iasi_radiance_mw(coded, exponent)
            quality = np.full(slots, np.nan)
        bt = radiance_to_bt(radiance.reshape(rows.shape), wavenumber).ravel()
        for variable in plan:
            observations[variable] = bt
    else:
        if sensor == "atms":
            sccf = rows.take("SCCF").reshape(rows.shape)
            table = _ATMS_SCCF_CODE[channels - 1] / (1.0 * 10**-5)
            if not np.all((sccf == table) | np.isnan(sccf)):
                raise ValueError("ATMS SCCF differs from the channel frequency table")
        quality = (
            rows.take("ATMSCHQ") if "ATMSCHQ" in child_names else np.full(slots, np.nan)
        )
        for variable, field in plan.items():
            observations[variable] = rows.take(field, np.float32)

    kept = rows.kept

    def footprint_float(values_: np.ndarray | None) -> pa.Array:
        if values_ is None:
            return null_array(pa.float32(), len(kept))
        return float_with_nulls(values_[kept].astype(np.float32), pa.float32())

    def footprint_uint(values_: np.ndarray | None, arrow_type: pa.DataType) -> pa.Array:
        if values_ is None:
            return null_array(arrow_type, len(kept))
        return uint_with_nulls(values_[kept], arrow_type)

    columns: dict[str, pa.Array] = {
        "time": timestamps_ns(time_ns[kept]),
        "class": constant_dictionary("rad", len(kept)),
        "lat": pa.array(lat[kept].astype(np.float32)),
        "lon": pa.array(np.mod(lon[kept], 360.0).astype(np.float32)),
        "elev": footprint_float(elev),
        "scan_angle": footprint_float(scan_angle),
        "scan_position": footprint_uint(position, pa.uint16()),
        "scan_line": footprint_uint(_column(batch, "SLNM"), pa.uint32()),
        "detector": footprint_uint(
            _column(batch, "FOVN") if is_cris else None, pa.uint16()
        ),
        "solza": footprint_float(_column(batch, "SOZA")),
        "solaza": footprint_float(_column(batch, "SOLAZI")),
        "satellite_za": footprint_float(_column(batch, "SAZA")),
        "satellite_aza": footprint_float(_column(batch, "BEARAZ")),
        "scan_quality": footprint_uint(
            _column(batch, "NSQF" if is_cris else "ATMSSQ"), pa.uint32()
        ),
        "granule_quality": footprint_uint(_column(batch, "ATMSGQ"), pa.uint16()),
        "footprint_quality": footprint_uint(
            _column(batch, "QGFQ") if sensor == "iasi" else None, pa.uint8()
        ),
        "satellite": _satellite_column(said[kept]),
        "quality": rows.lists(uint_with_nulls(quality, pa.uint32())),
    }
    for variable, values_ in observations.items():
        columns[f"observation_{variable}"] = rows.lists(
            float_with_nulls(values_.astype(np.float32, copy=False), pa.float32())
        )
    return pa.table(columns)


def decode_satellite(
    path: str,
    sensor: str,
    plan: Mapping[str, str],
    dt_min: datetime,
    dt_max: datetime,
    satellites: Collection[str] | None = None,
    channels: Collection[int] | None = None,
    decode_workers: int = 8,
) -> pa.Table:
    """Decode an NCEP aggregate satellite file into a wide table, one row per footprint.

    Per-channel values are fixed-size lists over the channel axis, ``channels`` in
    ascending order or else ``SAT_CHANNELS[sensor]``: ``quality`` and one
    ``observation_<variable>`` per requested variable, null in a slot whose channel
    the footprint lacks. The axis is recorded in the field metadata of these
    columns. Footprint columns are those of the sensor's wide schema. IASI and CrIS radiances are returned as brightness temperature.

    Parameters
    ----------
    path : str
        Local path to the BUFR file
    sensor : str
        One of ``atms``, ``mhs``, ``amsua``, ``amsub``, ``airs``, ``iasi``, ``cris``
    plan : Mapping[str, str]
        Variable -> BUFR channel field (e.g. ``{"atms": "TMBR"}``)
    dt_min, dt_max : datetime
        Closed observation time window
    satellites : Collection[str] | None, optional
        Platforms to keep, by default all
    channels : Collection[int] | None, optional
        Channel numbers decoded for an IR sounder, by default ``SAT_CHANNELS[sensor]``
    decode_workers : int, optional
        Threads transforming decoded batches, by default 8

    Returns
    -------
    pa.Table
        Wide table
    """
    window_ns = (datetime_ns(dt_min), datetime_ns(dt_max))
    satellite_filter = None if satellites is None else frozenset(satellites)
    axis = sorted(SAT_CHANNELS[sensor] if channels is None else set(channels))
    schema = _sat_wide_schema(sensor, axis)
    axis_values = np.array(axis, dtype=np.float64)
    iasi_groups: dict[bytes, np.ndarray] = {}
    with open_bufr(path) as source:
        wavenumber = _airs_wavenumber(source, path) if sensor == "airs" else None
        batches = iter_batches(
            source,
            path,
            data_category_filter=[21],
            target_decoded_bytes=32 << 20,
            prefetch_batches=2,
            **_sat_decode_options(sensor, set(plan.values()), channels),
        )
        tables = [
            t
            for t in map_ordered(
                lambda b: _sat_batch(
                    b,
                    sensor,
                    plan,
                    window_ns,
                    satellite_filter,
                    axis_values,
                    iasi_groups,
                ),
                batches,
                decode_workers,
            )
            if t is not None
        ]
    if not tables:
        return _empty_wide(schema, list(plan))
    table = concat_tables(tables)
    # Only the sensor's columns; the per-channel fields carry the axis in their metadata
    names = [name for name in table.column_names if name in schema.names]
    fields = [
        schema.field(name) if name.startswith(("observation_", "quality")) else field
        for name, field in zip(names, table.select(names).schema)
    ]
    table = pa.Table.from_arrays(table.select(names).columns, schema=pa.schema(fields))
    if wavenumber is not None:
        table = table.replace_schema_metadata(
            {_WAVENUMBER_BY_CHANNEL: wavenumber.tobytes()}
        )
    return table


def assemble_frame(
    pieces: Sequence[tuple[Any, pa.Table]],
    layout: str,
    variables: Sequence[str],
    long_schema: pa.Schema,
    fields: pa.Schema,
    source_id: str,
    long_transform: Callable[[Any, pa.Table], pa.Table] | None = None,
) -> pd.DataFrame:
    """Per-file wide tables -> the requested layout as one DataFrame.

    Parameters
    ----------
    pieces : Sequence[tuple[task, pa.Table]]
        Decoded wide tables per task, from :func:`compile_tables`
    layout : str
        ``"long"`` converts each file's table with :func:`_to_long`; ``"wide"``
        returns the wide rows
    variables : Sequence[str]
        Requested variables, in long output order
    long_schema : pa.Schema
        The long schema of the source
    fields : pa.Schema
        Output columns
    source_id : str
        Stored in ``attrs["source"]``
    long_transform : Callable[[task, pa.Table], pa.Table] | None, optional
        Applied to each file's long table

    Returns
    -------
    pd.DataFrame
        Frame of ``pd.ArrowDtype`` columns
    """
    tables = []
    for task, piece in pieces:
        if layout == "long":
            piece = _to_long(piece, variables, long_schema)
            if long_transform is not None:
                piece = long_transform(task, piece)
        tables.append(piece)
    table = concat_tables(tables) if tables else fields.empty_table()
    return to_dataframe(table, fields, source_id)
