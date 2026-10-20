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
    dictionary_column,
    float_values,
    float_with_nulls,
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
from earth2studio.data.utils_ir import CRIS_BANDS, wavenumber_cm_inverse
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


@dataclass(frozen=True)
class NCEPObsTask:
    """One conventional NCEP observation cycle file to fetch and decode."""

    uri: str
    route: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: _NCEPPlan


def plan_conv_tasks(
    windows: Mapping[datetime, tuple[datetime, datetime]],
    variable: list[str],
    lexicon: LexiconType,
    build_uri: Callable[[str, datetime], str],
) -> list[NCEPObsTask]:
    """Partition variables by lexicon route prefix into per-cycle conv tasks.

    Parameters
    ----------
    windows : Mapping[datetime, tuple[datetime, datetime]]
        Cycle datetime -> (min, max) observation-time window, already merged.
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
    for cycle, (tmin, tmax) in windows.items():
        for route, plan in plans.items():
            tasks.append(
                NCEPObsTask(
                    uri=build_uri(route, cycle),
                    route=route,
                    datetime_file=cycle,
                    datetime_min=tmin,
                    datetime_max=tmax,
                    var_plan=plan,
                )
            )
    return tasks


def observation_cycle_times(
    time: datetime,
    tolerance_lower: timedelta,
    tolerance_upper: timedelta,
    cadence: timedelta = timedelta(hours=6),
) -> list[datetime]:
    """Return cadence file times whose observation windows overlap a request.

    Each file timestamp is treated as the end of its observation window, so a
    file at ``T`` may contain observations from ``(T - cadence, T]``. This means
    requests often need the next cycle file as well as the cycle at or before
    the requested time.

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
        Cadence-aligned file timestamps whose backward-looking observation
        windows overlap ``[time + tolerance_lower, time + tolerance_upper]``.

    Raises
    ------
    ValueError
        If ``cadence`` is not positive.
    """
    if cadence <= timedelta(0):
        raise ValueError("cadence must be positive")

    tmin = time + tolerance_lower
    tmax = time + tolerance_upper
    day_start = tmin.replace(hour=0, minute=0, second=0, microsecond=0)
    cycle = day_start + ((tmin - day_start) // cadence) * cadence
    if cycle < tmin:
        cycle += cadence

    cycles: list[datetime] = []
    while cycle < tmax + cadence:
        cycles.append(cycle)
        cycle += cadence
    return cycles


def cycle_windows(
    time_list: list[datetime],
    tolerance_lower: timedelta,
    tolerance_upper: timedelta,
    cadence: timedelta = timedelta(hours=6),
) -> dict[datetime, tuple[datetime, datetime]]:
    """Map cadence file times to merged requested observation windows.

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
    dict[datetime, tuple[datetime, datetime]]
        Mapping from each required cadence file timestamp to the merged
        observation-time window covered by requests for that file.
    """
    windows: dict[datetime, tuple[datetime, datetime]] = {}
    for t in time_list:
        tmin = t + tolerance_lower
        tmax = t + tolerance_upper
        for cycle in observation_cycle_times(
            t, tolerance_lower, tolerance_upper, cadence
        ):
            existing = windows.get(cycle)
            windows[cycle] = (
                (min(existing[0], tmin), max(existing[1], tmax))
                if existing is not None
                else (tmin, tmax)
            )
    return windows


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
# ATMS channel centre frequencies as encoded (SCCF, 0-02-153, in units of 10⁵ Hz) in the
# NNJA atms aggregates; the same for every platform
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

CONV_VARIABLES = ("u", "v", "q", "t", "pres", "gps", "gps_refractivity")
SATWND_VARIABLES = ("u", "v")
SAT_VARIABLE_SENSOR: dict[str, str] = {
    variable: key.split("::")[0] for variable, key in NNJAObsSatLexicon.VOCAB.items()
}
SAT_SENSORS = tuple(dict.fromkeys(SAT_VARIABLE_SENSOR.values()))
# Per-channel columns of a satellite footprint row, one set per sensor
SAT_CHANNEL_COLUMNS = ("sensor_index", "quality")

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


_NCEP_CONVENTIONAL_WIDE_SCHEMA = pa.schema(
    [
        *_fields(
            NCEP_CONVENTIONAL_PUBLIC_SCHEMA, ("quality", "observation", "variable")
        ),
        *[
            field
            for variable in CONV_VARIABLES
            for field in (
                _observation_field(variable, pa.float32()),
                pa.field(
                    f"quality_{variable}",
                    pa.uint16(),
                    nullable=True,
                    metadata={
                        "description": f"Quality mark of the {variable!r} observation"
                    },
                ),
            )
        ],
    ]
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

_SAT_CHANNEL_TYPES = {
    "sensor_index": pa.uint16(),
    "quality": pa.uint32(),
}
_NCEP_MICROWAVE_WIDE_SCHEMA = pa.schema(
    [
        *_fields(
            NCEP_MICROWAVE_OUTPUT_SCHEMA,
            (*SAT_CHANNEL_COLUMNS, "wavenumber", "observation", "variable"),
        ),
        *[
            pa.field(
                f"{name}_{sensor}",
                pa.list_(arrow_type),
                nullable=True,
                metadata={
                    "description": (
                        f"{sensor} per-channel {name}, one entry per channel; a "
                        "fixed-size list when every footprint carries the same "
                        "channels"
                    )
                },
            )
            for sensor in SAT_SENSORS
            for name, arrow_type in _SAT_CHANNEL_TYPES.items()
        ],
        *[_observation_field(v, pa.list_(pa.float32())) for v in SAT_VARIABLE_SENSOR],
    ]
)


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
    ``quality_<variable>`` when the table has it. List-valued
    ``observation_<variable>`` columns (satellite footprints) give one row per
    channel entry of the footprint, footprint by footprint, with the variables of
    one sensor interleaved per channel and the per-channel columns
    ``<name>_<sensor>`` as ``sensor_index``, ``wavenumber`` and ``quality``.

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
    if _is_list(first_type):
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


def _list_parent(lists: pa.Array) -> np.ndarray:
    lengths = pc.fill_null(pc.list_value_length(lists), 0).to_numpy()
    return np.repeat(np.arange(len(lists), dtype=np.int64), lengths)


def _channel_wavenumber(sensor: str, channels: np.ndarray) -> np.ndarray | None:
    if sensor == "atms":
        if len(channels) and (
            channels.min() < 1 or channels.max() > len(_ATMS_WAVENUMBER_CM)
        ):
            raise ValueError(f"atms channels outside 1-{len(_ATMS_WAVENUMBER_CM)}")
        return _ATMS_WAVENUMBER_CM[channels - 1]
    if sensor in _IR_SENSORS:
        return wavenumber_cm_inverse(sensor, channels)
    return None


def _explode_channels(table: pa.Table, variables: Sequence[str]) -> list[pa.Table]:
    by_sensor: dict[str, list[str]] = {}
    for variable in variables:
        by_sensor.setdefault(SAT_VARIABLE_SENSOR[variable], []).append(variable)
    channel_names = {
        f"{name}_{sensor}" for sensor in SAT_SENSORS for name in SAT_CHANNEL_COLUMNS
    }
    footprint = [
        name
        for name in _drop_wide(table, SAT_VARIABLE_SENSOR)
        if name not in channel_names
    ]
    parts = []
    for sensor, sensor_variables in by_sensor.items():
        rows = table
        first = table.column(f"observation_{sensor_variables[0]}")
        if first.null_count:
            rows = table.filter(pc.is_valid(first))
        if rows.num_rows == 0:
            continue
        observations = [
            _single_chunk(rows.column(f"observation_{v}")) for v in sensor_variables
        ]
        parent = _list_parent(observations[0])
        n_var = len(sensor_variables)
        index = np.repeat(parent, n_var) if n_var > 1 else parent
        columns = {name: rows.column(name).take(index) for name in footprint}
        for name in SAT_CHANNEL_COLUMNS:
            column_name = f"{name}_{sensor}"
            if column_name in rows.column_names:
                values = pc.list_flatten(_single_chunk(rows.column(column_name)))
                if n_var > 1:
                    values = values.take(
                        np.repeat(np.arange(len(values), dtype=np.int64), n_var)
                    )
                columns[name] = values
        if "sensor_index" in columns:
            channels = columns["sensor_index"].to_numpy(zero_copy_only=False)
            wavenumber = _channel_wavenumber(sensor, channels.astype(np.int64))
            if wavenumber is not None:
                columns["wavenumber"] = pa.array(wavenumber)
        flat = [pc.list_flatten(obs) for obs in observations]
        if n_var > 1:
            stacked = np.stack(
                [v.to_numpy(zero_copy_only=False) for v in flat], axis=1
            ).ravel()
            columns["observation"] = float_with_nulls(stacked, pa.float32())
            codes = np.tile(np.arange(n_var, dtype=np.int8), len(parent))
            columns["variable"] = dictionary_column(codes, list(sensor_variables))
        else:
            columns["observation"] = flat[0]
            columns["variable"] = constant_dictionary(sensor_variables[0], len(parent))
        parts.append(pa.table(columns))
    return parts


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
    """Concatenate tables, unifying list column types and filling missing columns.

    Fixed-size lists of different sizes become variable-size lists; a column
    absent from some tables is null there.
    """
    tables = [t for t in tables if t is not None]
    if len(tables) == 1:
        return tables[0]
    names = list(dict.fromkeys(name for t in tables for name in t.column_names))
    target: dict[str, pa.DataType] = {}
    for name in names:
        types = {t.schema.field(name).type for t in tables if name in t.column_names}
        if len(types) == 1:
            target[name] = types.pop()
            # A fixed-size list padded with nulls would store child values per row
            if pa.types.is_fixed_size_list(target[name]) and any(
                name not in t.column_names for t in tables
            ):
                target[name] = pa.list_(target[name].value_type)
        elif all(_is_list(t) for t in types):
            target[name] = pa.list_(next(iter(types)).value_type)
        else:
            raise TypeError(f"column {name!r} has incompatible types {types}")
    aligned = []
    for t in tables:
        columns = []
        for name in names:
            arrow_type = target[name]
            if name not in t.column_names:
                columns.append(pa.chunked_array([pa.nulls(t.num_rows, arrow_type)]))
            elif t.schema.field(name).type != arrow_type:
                columns.append(t.column(name).cast(arrow_type))
            else:
                columns.append(t.column(name))
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


# Files decoded at once by compile_tables; each also runs earth2bufr's threads and
# decode_workers batch threads
_FILES_IN_FLIGHT = 2


def compile_tables(
    tasks: Sequence[Any],
    source_id: str,
    local_path: Callable[[str], str],
    decode_task: Callable[[str, Any], pa.Table],
    on_error: Callable[[str, int, int, Exception], None] | None = None,
    downloaded: threading.Event | None = None,
) -> list[tuple[Any, pa.Table]]:
    """Decode cached task files into wide tables, each with a ``cycle_time`` column.

    Parameters
    ----------
    tasks : Sequence
        Task objects exposing ``uri`` and ``datetime_file``
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
        t0 = time.perf_counter()
        try:
            return decode_task(path, task), time.perf_counter() - t0
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
    """The columns of ``schema`` as a DataFrame of ``pd.ArrowDtype`` columns (zero-copy)."""
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

    Edition 3 stores a two-digit year (pivot 70) and no seconds.
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
        edition3 & (year < 100), year + np.where(year < 70, 2000, 1900), year
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
    header_us = message_base + _offset_microseconds(
        np.where(dhr_valid, dhr, 0.0), dhr_scale
    )
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
        "radius_curvature": null_array(pa.float64(), k),
        "geoid_undulation": null_array(pa.float64(), k),
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
        structure = source.source_structure(message.source_message_index)
        sequences = {
            s["descriptor"]: s["mnemonic"] for s in structure["active_sequences"]
        }
        table_a = next((d for d in message.descriptors if d.startswith("3-")), None)
        name = sequences.get(table_a)
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
        If the file has data messages but none of a known PrepBUFR message family
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
            return _empty_wide(_NCEP_CONVENTIONAL_WIDE_SCHEMA, variables)
        base_us = _message_base_times(source.info)
        batches = source.iter_batches(
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
        return _empty_wide(_NCEP_CONVENTIONAL_WIDE_SCHEMA, variables)
    return pa.concat_tables(tables, promote_options="permissive")


def _empty_wide(schema: pa.Schema, variables: Sequence[str]) -> pa.Table:
    """Empty wide table: shared columns plus the columns of ``variables``."""
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
    return pa.schema(keep).empty_table()


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


def _list_entries(lists: pa.ListArray) -> tuple[pa.StructArray, np.ndarray, np.ndarray]:
    offsets = lists.offsets.to_numpy()
    if lists.offset or offsets[0] != 0 or offsets[-1] != len(lists.values):
        raise ValueError("list does not span its values")
    lengths = np.diff(offsets)
    parent = np.repeat(np.arange(len(lists)), lengths)
    position = np.arange(offsets[-1]) - offsets[parent]
    return lists.values, parent, position


def _reference_coordinate(
    top: np.ndarray,
    top_valid: np.ndarray,
    levels: tuple[pa.StructArray, np.ndarray, np.ndarray],
    name: str,
    first_impp_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The occultation's CLATH/CLONH, else the first level value ahead of the first IMPP."""
    if top_valid.all():
        return top, top_valid
    struct, parent, position = levels
    values, valid = numpy_with_mask(struct.field(name))
    candidate = np.flatnonzero(valid & (position <= first_impp_position[parent]))
    first = np.full(len(top), len(values), dtype=np.int64)
    np.minimum.at(first, parent[candidate], candidate)
    found = first < len(values)
    fallback = values[np.where(found, first, 0)] if len(values) else np.zeros(len(top))
    return np.where(top_valid, top, fallback), top_valid | found


def _gpsro_batch(
    batch: pa.RecordBatch, variables: Sequence[str], window_ns: tuple[int, int]
) -> pa.Table | None:
    """Wide rows of one batch: bending-angle levels, then refractivity levels."""
    n = batch.num_rows
    scalar = {name: numpy_with_mask(batch.column(name)) for name in _GPSRO_SCALARS}
    levels = _list_entries(batch.column("DRF16BIT"))
    level_struct, level_parent, level_position = levels
    frequencies, frequency_level, _ = _list_entries(level_struct.field("DRF8BIT"))
    has_frequency = np.zeros(len(level_struct), dtype=bool)
    has_frequency[frequency_level] = True
    first_impp_position = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    with_impp = np.flatnonzero(has_frequency)
    np.minimum.at(
        first_impp_position, level_parent[with_impp], level_position[with_impp]
    )
    lat_ref, lat_ref_valid = _reference_coordinate(
        *scalar["CLATH"], levels, "CLATH", first_impp_position
    )
    lon_ref, lon_ref_valid = _reference_coordinate(
        *scalar["CLONH"], levels, "CLONH", first_impp_position
    )

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
            # Ionosphere-corrected bending angle: mean frequency 0
            rows = np.flatnonzero(
                mefr_valid
                & (np.abs(mefr) <= 0.5)
                & impp_valid
                & value_valid
                & elrc_valid[subset]
                & subset_ok[subset]
            )
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
            refractivity, refractivity_subset, _ = _list_entries(
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
            "report_time": null_array(pa.timestamp("ns"), k),
            "pres": null_array(pa.float32(), k),
            "elev": pa.array(elev, type=pa.float32()),
            "type": masked(
                np.where(said_valid, said, 0).astype(np.uint16)[subset],
                said_valid[subset],
                pa.uint16(),
            ),
            "level_cat": null_array(pa.uint16(), k),
            "class": constant_dictionary("GPSRO", k),
            "lat": pa.array(lat.astype(np.float32), type=pa.float32()),
            "lon": pa.array(np.mod(lon, 360.0).astype(np.float32), type=pa.float32()),
            "report_lat": null_array(pa.float32(), k),
            "report_lon": null_array(pa.float32(), k),
            "station": station.take(pa.array(subset)),
            "station_elev": null_array(pa.float32(), k),
            "pressure_quality": null_array(pa.uint16(), k),
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

    One row per bending-angle level (``observation_gps``, the ionosphere-corrected
    bending angle) or refractivity level (``observation_gps_refractivity``).

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
        batches = source.iter_batches(
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
        return _empty_wide(_NCEP_CONVENTIONAL_WIDE_SCHEMA, variables)
    return pa.concat_tables(tables, promote_options="permissive")


# ──────────────────────────────────────────────────────────────────────────
# Satellite winds
# ──────────────────────────────────────────────────────────────────────────
# Each mnemonic is read from its first top-level occurrence in the template.
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
_SATWND_PROJECTION = [*_SATWND_TIME, *_SATWND_SCALARS, "GNAP", "GNAPS", "PCCF"]
_SATWND_INTS = {
    "satellite_id": ("SAID", pa.uint16()),
    "quality": ("SWQM", pa.uint16()),
    "wind_method": ("SWCM", pa.uint8()),
    "wind_method_local": ("CMCM", pa.uint8()),
    "height_method": ("HAMD", pa.uint8()),
}
_SATWND_FLOATS = {"pres": "PRLC", "satellite_za": "SAZA"}
_SATWND_SUBSETS = pa.array([f"NC005{c:03d}" for c in range(256)])


@dataclass(frozen=True)
class _SatwndPlan:
    column: dict[str, int]
    pairs: dict[str, list[tuple[int, int]]]


def _satwnd_plan(schema: pa.Schema) -> _SatwndPlan:
    """Column index of each mnemonic's first top-level occurrence and the
    GNAP/GNAPS -> PCCF pairs of a template."""
    column: dict[str, int] = {}
    for index, field in enumerate(schema):
        if (
            field.name in (*_SATWND_SCALARS, *_SATWND_TIME)
            and field.name not in column
            and not pa.types.is_nested(field.type)
        ):
            column[field.name] = index
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
    return _SatwndPlan(column, pairs)


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


def _satwnd_batch(
    batch: pa.RecordBatch,
    subcategory: np.ndarray,
    variables: Sequence[str],
    window_ns: tuple[int, int],
) -> pa.Table | None:
    plan = _satwnd_plan(batch.schema)
    n = batch.num_rows

    def get(name: str) -> pa.Array | None:
        index = plan.column.get(name)
        return None if index is None else batch.column(index)

    def floats(name: str) -> np.ndarray:
        array = get(name)
        return np.full(n, np.nan) if array is None else float_values(array)

    lat = floats("CLATH" if "CLATH" in plan.column else "CLAT")
    lon = floats("CLONH" if "CLONH" in plan.column else "CLON")
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
        array = get(name)
        columns[out] = (
            null_array(pa.float32(), k)
            if array is None
            else pc.cast(array, pa.float32()).filter(arrow_keep)
        )
    for out, (name, arrow_type) in _SATWND_INTS.items():
        array = get(name)
        columns[out] = (
            null_array(arrow_type, k)
            if array is None
            else pc.cast(array, arrow_type).filter(arrow_keep)
        )
    message = batch.column("_source_message_index").to_numpy()[keep]
    columns["subset"] = pa.DictionaryArray.from_arrays(
        pa.array(subcategory[message].astype(np.int16)), _SATWND_SUBSETS
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

    Returns
    -------
    pa.Table
        Wide table
    """
    window_ns = (datetime_ns(dt_min), datetime_ns(dt_max))
    with open_bufr(path) as source:
        subcategory = np.zeros(len(source.info.messages), dtype=np.int64)
        for message in source.info.messages:
            subcategory[message.source_message_index] = message.local_sub_category
        # Fields differ by template; a template's missing fields are left out
        batches = source.iter_batches(
            projection=_SATWND_PROJECTION,
            projection_missing="skip",
            target_decoded_bytes=256 << 20,
            prefetch_batches=2,
        )
        tables = [
            t
            for t in map_ordered(
                lambda b: _satwnd_batch(b, subcategory, variables, window_ns),
                batches,
                decode_workers,
            )
            if t is not None
        ]
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


def _sat_decode_options(
    sensor: str, fields: Collection[str], channels: Collection[int] | None
) -> dict[str, Any]:
    """earth2bufr options: only the requested fields and channels are decoded."""
    if sensor == "atms":
        channel_fields = [f"DRF16BIT.{f}" for f in ("CHNM", "ATMSCHQ", *sorted(fields))]
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


class _ChannelRows:
    """Channel entries of the kept footprints of one batch, footprint-major.

    Uniform (``width`` set): every kept footprint carries the channel sequence
    ``channels``, so entries reshape to (footprints, width). Otherwise each kept
    footprint has ``counts`` entries at ``index`` in the list values.
    """

    def __init__(self, lists: pa.Array, keep: np.ndarray) -> None:
        offsets, self.values = _list_layout(lists)
        lengths = np.diff(offsets)
        self.kept = np.flatnonzero(keep & (lengths > 0))
        self.width = 0
        self.channels = np.empty(0)
        chnm = physical_values(self.values, "CHNM")
        if len(self.kept) and np.all(lengths[self.kept] == lengths[self.kept[0]]):
            width = int(lengths[self.kept[0]])
            starts = offsets[self.kept]
            index = (starts[:, None] + np.arange(width)).ravel()
            row = chnm[index[:width]]
            entries = chnm[index].reshape(-1, width)
            if (
                not np.isnan(row).any()
                and len(np.unique(row)) == width
                and np.all(entries == row)
            ):
                self.width = width
                self.channels = row
                self.index = index
                self.counts = np.full(len(self.kept), width, dtype=np.int64)
                return
        parent = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        entry_ok = keep[parent] & np.isfinite(chnm)
        self.index = np.flatnonzero(entry_ok)
        self.counts = np.bincount(parent[self.index], minlength=len(lengths))[self.kept]

    @property
    def n_entries(self) -> int:
        return len(self.index)

    # np.tile and np.repeat hold the GIL; broadcast copies of uniform rows do not
    def channel_numbers(self) -> np.ndarray:
        if self.width:
            return np.broadcast_to(self.channels, (len(self.kept), self.width)).ravel()
        return physical_values(self.values, "CHNM", self.index)

    def footprints(self) -> np.ndarray:
        """Batch row of each entry."""
        if self.width:
            return np.broadcast_to(
                self.kept[:, None], (len(self.kept), self.width)
            ).ravel()
        return np.repeat(self.kept, self.counts)

    def lists(self, flat: pa.Array) -> pa.Array:
        if self.width:
            return pa.FixedSizeListArray.from_arrays(flat, self.width)
        offsets = np.zeros(len(self.kept) + 1, dtype=np.int32)
        np.cumsum(self.counts, out=offsets[1:])
        return pa.ListArray.from_arrays(pa.array(offsets), flat)


def _compacted_band(flags: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    """Value of the band-th non-missing entry per row."""
    valid = np.isfinite(flags)
    rank = np.cumsum(valid, axis=1) - 1
    hit = valid & (rank == band)
    return np.where(hit, flags, 0.0).sum(axis=1), hit.any(axis=1)


def _cris_band_quality(batch: pa.RecordBatch) -> tuple[np.ndarray, np.ndarray]:
    """Per footprint and band (LW, MW, SW): NFQF | NCQF << 19 and whether NFQF is set.

    Bands are the non-missing NFQF/NCQF entries in order.
    """
    if _CRIS_BAND_GROUP in batch.schema.names:
        group = batch.column(_CRIS_BAND_GROUP).values
        nfqf = float_values(group.field("NFQF")).reshape(-1, 3)
        ncqf = float_values(group.field("NCQF")).reshape(-1, 3)
    else:
        nfqf = np.stack(_columns_at(batch, "NFQF"), axis=1)
        ncqf = np.stack(_columns_at(batch, "NCQF"), axis=1)
    packed = np.zeros(nfqf.shape, dtype=np.uint32)
    valid = np.zeros(nfqf.shape, dtype=bool)
    for band in range(3):
        nf, nf_ok = _compacted_band(nfqf, band)
        nc, nc_ok = _compacted_band(ncqf, band)
        packed[:, band] = nf.astype(np.uint32) | (
            np.where(nc_ok, nc, 0.0).astype(np.uint32) << 19
        )
        valid[:, band] = nf_ok
    return packed, valid


def _iasi_scale_factor(
    batch: pa.RecordBatch, rows: _ChannelRows, channels: np.ndarray
) -> np.ndarray:
    """10**-CHSF of each entry's scale group: the first STCH..ENCH group holding the
    channel, NaN outside every group. Groups are resolved once per distinct layout."""
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
        )
        for name in ("STCH", "ENCH", "CHSF")
    )
    usable = np.isfinite(start) & np.isfinite(stop) & np.isfinite(scale)
    groups = np.where(
        usable[:, :, None], np.stack([start, stop, scale], axis=2), -1.0
    ).reshape(len(start), -1)
    layout_of = np.full(len(groups), -1, dtype=np.int64)
    layouts: list[np.ndarray] = []
    while (todo := np.flatnonzero(layout_of < 0)).size:
        match = (groups[todo] == groups[todo[0]]).all(axis=1)
        layout_of[todo[match]] = len(layouts)
        layouts.append(groups[todo[0]])
    lut = np.full((len(layouts), _IR_CHANNEL_LIMIT["iasi"] + 1), np.nan)
    for i, layout in enumerate(layouts):
        for first, last, exponent in layout.reshape(-1, 3)[::-1]:
            if first >= 0:
                lut[i, int(first) : int(last) + 1] = exponent
    with np.errstate(invalid="ignore"):
        factor = np.power(10.0, -lut)
    footprint = rows.footprints()
    return factor[layout_of[footprint], channels.astype(np.int64)]


def _sat_batch(
    batch: pa.RecordBatch,
    sensor: str,
    plan: Mapping[str, str],
    window_ns: tuple[int, int],
    satellites: frozenset[str] | None,
) -> pa.Table | None:
    """Wide rows (one per kept footprint) of one batch.

    ``plan`` maps each requested variable to its BUFR field.
    """
    parts = [_required_column(batch, name) for name in _SAT_TIME]
    time_ns, time_ok = civil_time_ns(*parts)
    for part in parts:
        time_ok &= np.isfinite(part)
    time_ok &= (parts[5] >= 0.0) & (parts[5] < 61.0)
    lat, lon = _footprint_latlon(batch)
    said = _required_column(batch, "SAID")
    is_cris = sensor == "cris"
    position = _required_column(batch, "FORN" if is_cris else "FOVN")
    keep = (
        time_ok
        & np.isfinite(said)
        & (time_ns >= window_ns[0])
        & (time_ns <= window_ns[1])
        & np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & np.isfinite(position)
    )
    said_codes, said_index = np.unique(
        np.where(np.isfinite(said), said, -1).astype(np.int64), return_inverse=True
    )
    names = [
        _NCEP_SATELLITE_NAME_BY_SAID.get(int(s), f"satellite-{int(s)}")
        for s in said_codes
    ]
    if satellites is not None:
        keep &= np.array([name in satellites for name in names], dtype=bool)[said_index]
    rows = _ChannelRows(_channel_group(batch), keep)
    if not len(rows.kept):
        return None
    values = rows.values
    child_names = {values.type.field(i).name for i in range(values.type.num_fields)}
    channels = rows.channel_numbers()

    if sensor in _IR_SENSORS:
        limit = _IR_CHANNEL_LIMIT[sensor]
        if len(channels) and (np.nanmin(channels) < 1 or np.nanmax(channels) > limit):
            raise ValueError(f"{sensor} channels outside 1-{limit}")
        scan_angle: np.ndarray | None = None
        elev = _column(batch, "HOLS") if is_cris else None
    else:
        start, step = _NCEP_MICROWAVE_SCAN_GEOMETRY[sensor]
        scan_angle = start + (position - 1.0) * step
        elev = _column(batch, "HOLS")

    observations: dict[str, np.ndarray] = {}
    if sensor == "airs":
        quality = physical_values(values, "ACQF", rows.index)
        for variable, field in plan.items():
            observations[variable] = physical_values(
                values, field, rows.index, np.float32
            )
    elif sensor in ("iasi", "cris"):
        wavenumber = wavenumber_cm_inverse(sensor, channels.astype(np.int64))
        coded = physical_values(values, _IR_OBSERVATION[sensor], rows.index)
        if is_cris:
            # SRAD W m-2 sr-1 (cm-1)-1 -> mW
            radiance = coded * 1000.0
            packed, packed_ok = _cris_band_quality(batch)
            footprint = rows.footprints()
            band = _CRIS_CHANNEL_BAND[channels.astype(np.int64)]
            quality = np.where(
                packed_ok[footprint, band], packed[footprint, band], np.nan
            )
        else:
            # SCRA x 10^-CHSF W m-2 sr-1 (m-1)-1 -> mW m-2 sr-1 (cm-1)-1
            radiance = coded * _iasi_scale_factor(batch, rows, channels) * 1e5
            quality = np.full(len(channels), np.nan)
        bt = radiance_to_bt(radiance, wavenumber)
        for variable in plan:
            observations[variable] = bt
    else:
        quality = (
            physical_values(values, "ATMSCHQ", rows.index)
            if "ATMSCHQ" in child_names
            else np.full(len(channels), np.nan)
        )
        for variable, field in plan.items():
            observations[variable] = physical_values(
                values, field, rows.index, np.float32
            )

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
        "satellite": dictionary_column(said_index[kept].astype(np.int8), names),
        f"sensor_index_{sensor}": rows.lists(
            pa.array(channels.astype(np.uint16), type=pa.uint16())
        ),
        f"quality_{sensor}": rows.lists(uint_with_nulls(quality, pa.uint32())),
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

    Per-channel values are lists in channel order: ``sensor_index_<sensor>``,
    ``quality_<sensor>`` and one ``observation_<variable>`` per requested variable,
    fixed-size when every footprint of a batch carries the same channels. IASI and
    CrIS radiances are returned as brightness temperature.

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
        Channel numbers decoded for an IR sounder, by default all
    decode_workers : int, optional
        Threads transforming decoded batches, by default 8

    Returns
    -------
    pa.Table
        Wide table
    """
    window_ns = (datetime_ns(dt_min), datetime_ns(dt_max))
    satellite_filter = None if satellites is None else frozenset(satellites)
    with open_bufr(path) as source:
        batches = source.iter_batches(
            data_category_filter=[21],
            target_decoded_bytes=32 << 20,
            prefetch_batches=2,
            **_sat_decode_options(sensor, set(plan.values()), channels),
        )
        tables = [
            t
            for t in map_ordered(
                lambda b: _sat_batch(b, sensor, plan, window_ns, satellite_filter),
                batches,
                decode_workers,
            )
            if t is not None
        ]
    if not tables:
        return _empty_wide(_NCEP_MICROWAVE_WIDE_SCHEMA, list(plan))
    return concat_tables(tables)


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
