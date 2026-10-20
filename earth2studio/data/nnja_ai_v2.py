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

# NNJA-AI v2: Brightband's pre-decoded Parquet edition of the NOAA-NASA Joint
# Archive (NNJA) of Observations. Each BUFR stream is published as one Parquet
# file per UTC day, partitioned as
#
#   <group>/<sensor>/<MSG_TYPE>/OBS_DATE=YYYY-MM-DD/<sensor>.<MSG_TYPE>.parquet
#
# with the BUFR template preserved as nested Arrow columns plus normalized
# ``OBS_TIME``/``LAT``/``LON`` columns. These sources read those files
# directly; no BUFR decoding is performed.
#
# Reference: https://www.brightband.com/data/nnja-ai/

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_dir,
    gather_with_concurrency,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
)
from earth2studio.data.utils_ncep import (
    _C_CM_S,
    _NCEP_MICROWAVE_SCAN_GEOMETRY,
    _NCEP_SATELLITE_NAME_BY_SAID,
    NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
    NCEP_MICROWAVE_OUTPUT_SCHEMA,
    NCEP_MICROWAVE_SATELLITES,
    _finalize_rows,
    empty_dataframe,
    resolve_output_schema,
)
from earth2studio.lexicon import NNJAObsConvLexicon, NNJAObsSatLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_AI_V2_ROOT = "gs://nnja-ai/data/v2"

Window = tuple[datetime, datetime]


# ---------------------------------------------------------------------------
# Archive layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NNJAAIv2Product:
    """One NNJA-AI v2 stream: a daily-partitioned Parquet directory."""

    group: str
    sensor: str
    message_type: str
    first_year: int

    def partition_key(self, day: date) -> str:
        return (
            f"{self.group}/{self.sensor}/{self.message_type}/"
            f"OBS_DATE={day:%Y-%m-%d}/{self.sensor}.{self.message_type}.parquet"
        )


# Microwave sounders. ``channel_sequence`` is the BUFR replication sequence
# holding the per-channel fields (CHNM, TMBR, ...).
_SAT_PRODUCTS: dict[str, _NNJAAIv2Product] = {
    "amsua": _NNJAAIv2Product("amsua", "1bamua", "NC021023", 2008),
    "mhs": _NNJAAIv2Product("mhs", "1bmhs", "NC021027", 2005),
    "atms": _NNJAAIv2Product("atms", "atms", "NC021203", 2012),
}
_SAT_CHANNEL_SEQUENCE: dict[str, str] = {
    "amsua": "BRITCSTC",
    "mhs": "BRITCSTC",
    "atms": "ATMSCH",
}

# PrepBUFR message families published as separate streams under conv/prepbufr.
NNJA_AI_V2_PREPBUFR_FAMILIES: tuple[str, ...] = (
    "ADPSFC",
    "ADPUPA",
    "AIRCAR",
    "AIRCFT",
    "ASCATW",
    "PROFLR",
    "RASSDA",
    "SATWND",
    "SFCSHP",
    "SYNDAT",
    "VADWND",
)
_CONV_FIRST_YEAR = 2000

# PrepBUFR observation mnemonic and the event sequence holding its quality mark.
_CONV_VARIABLE_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "t": ("TOB", ("T___INFO", "T__EVENT", "TQM")),
    "q": ("QOB", ("Q___INFO", "Q__EVENT", "QQM")),
    "u": ("UOB", ("W___INFO", "W__EVENT", "WQM")),
    "v": ("VOB", ("W___INFO", "W__EVENT", "WQM")),
    "pres": ("POB", ("P___INFO", "P__EVENT", "PQM")),
}
_CONV_UNSUPPORTED_VARIABLES = frozenset(["gps"])
# Multi-level families carry balloon/aircraft drift per level; their
# per-level time can differ from the report header time by up to a few hours.
_CONV_DRIFT_PAD = timedelta(hours=3)


def _split_root(root: str) -> tuple[str, str]:
    """Split an archive root URL into an obstore URL and a key prefix."""
    root = root.rstrip("/")
    scheme, separator, rest = root.partition("://")
    if not separator:
        raise ValueError(f"NNJA-AI v2 root must be a URL, got {root!r}")
    if scheme == "file":
        return root, ""
    bucket, _, prefix = rest.partition("/")
    return f"{scheme}://{bucket}", prefix.strip("/")


def _merge_windows(
    time_list: Sequence[datetime], lower: timedelta, upper: timedelta
) -> list[Window]:
    """Merge per-timestamp tolerance windows into disjoint intervals."""
    intervals = sorted((t + lower, t + upper) for t in time_list)
    merged: list[Window] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _partition_days(windows: Sequence[Window], pad: timedelta) -> list[date]:
    """UTC days whose partition files can hold rows inside ``windows``."""
    days: set[date] = set()
    for start, end in windows:
        day = (start - pad).date()
        last = (end + pad).date()
        while day <= last:
            days.add(day)
            day += timedelta(days=1)
    return sorted(days)


def _utc_scalar(value: datetime) -> pa.Scalar:
    return pa.scalar(value.replace(tzinfo=timezone.utc))


def _window_filter(windows: Sequence[Window], pad: timedelta) -> pads.Expression:
    """Arrow filter selecting ``OBS_TIME`` inside any padded window."""
    expression: pads.Expression | None = None
    for start, end in windows:
        term = (pc.field("OBS_TIME") >= _utc_scalar(start - pad)) & (
            pc.field("OBS_TIME") <= _utc_scalar(end + pad)
        )
        expression = term if expression is None else expression | term
    if expression is None:
        raise ValueError("At least one observation window is required")
    return expression


def _in_windows(times: np.ndarray, windows: Sequence[Window]) -> np.ndarray:
    """Boolean mask of ``datetime64[ns]`` values falling inside any window."""
    mask: np.ndarray = np.zeros(len(times), dtype=bool)
    for start, end in windows:
        mask |= (times >= np.datetime64(start, "ns")) & (
            times <= np.datetime64(end, "ns")
        )
    return mask


def _has_path(schema: pa.Schema, path: Sequence[str]) -> bool:
    """Whether a (possibly nested) column path exists in ``schema``."""
    if path[0] not in schema.names:
        return False
    data_type = schema.field(path[0]).type
    for name in path[1:]:
        if not pa.types.is_struct(data_type) or data_type.get_field_index(name) < 0:
            return False
        data_type = data_type.field(name).type
    return True


def _to_numpy(array: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Convert an Arrow array to NumPy with nulls as NaN."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        return np.asarray(array.to_pylist(), dtype=object)
    if pa.types.is_timestamp(array.type):
        return array.cast(pa.timestamp("ns")).to_numpy(zero_copy_only=False)
    return array.to_numpy(zero_copy_only=False).astype(np.float64)


def _list_lengths(array: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return pc.fill_null(pc.list_value_length(array), 0).to_numpy(zero_copy_only=False)


def _event_edge(array: pa.ChunkedArray | pa.Array, newest: bool) -> np.ndarray:
    """First (newest) or last (oldest) element of each PrepBUFR event list.

    PrepBUFR stacks quality-control events newest first. Empty lists yield NaN.
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    lengths = _list_lengths(array)
    values = _to_numpy(pc.list_flatten(array))
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    index = offsets[:-1] if newest else offsets[1:] - 1
    result: np.ndarray = np.full(len(lengths), np.nan, dtype=np.float64)
    valid = lengths > 0
    result[valid] = values[index[valid]]
    return result


def _cycle_time(table: pa.Table) -> np.ndarray:
    """NCEP cycle each row was dumped in, as ``datetime64[ns]``.

    Parsed from ``SRC_FILENAME`` (``gdas.YYYYMMDD.tHHz....``); rows whose name
    does not parse fall back to the BUFR message date ``MSG_DATE``.
    """
    if "source" not in table.column_names and "cycle" not in table.column_names:
        raise ValueError("Neither SRC_FILENAME nor MSG_DATE was read")
    cycle: pa.Array | None = None
    if "source" in table.column_names:
        source = table["source"].combine_chunks()
        stamp = pc.utf8_slice_codeunits(source, 5, 17)
        parsed = pc.strptime(stamp, format="%Y%m%d.t%H", unit="s", error_is_null=True)
        cycle = parsed.cast(pa.timestamp("ns"))
    if "cycle" in table.column_names:
        message_date = table["cycle"].combine_chunks().cast(pa.timestamp("ns"))
        cycle = message_date if cycle is None else pc.fill_null(cycle, message_date)
    if cycle is None:  # pragma: no cover - guarded above
        raise ValueError("Neither SRC_FILENAME nor MSG_DATE was read")
    return cycle.to_numpy(zero_copy_only=False)


def _float_column(values: np.ndarray, data_type: pa.DataType) -> pa.Array:
    return pa.array(np.asarray(values, dtype=np.float64)).cast(data_type, safe=False)


def _uint_column(values: np.ndarray, data_type: pa.DataType) -> pa.Array:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    integers = np.where(valid, np.rint(values), 0).astype(np.int64)
    return pa.array(integers, mask=~valid).cast(data_type, safe=False)


def _microwave_frame(table: pa.Table) -> pd.DataFrame:
    """Match the pandas dtypes produced by ``NNJAObsSat``."""

    def types_mapper(data_type: pa.DataType) -> pd.ArrowDtype | None:
        if pa.types.is_unsigned_integer(data_type):
            return pd.ArrowDtype(data_type)
        return None

    return table.to_pandas(types_mapper=types_mapper)


# ---------------------------------------------------------------------------
# Partition decode
# ---------------------------------------------------------------------------


def decode_nnja_ai_v2_microwave(
    path: str,
    sensor: str,
    plan: Mapping[str, str],
    windows: Sequence[Window],
    satellites: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one NNJA-AI v2 microwave partition into a long-format DataFrame.

    Parameters
    ----------
    path : str
        Local path of the daily Parquet partition.
    sensor : str
        Sensor name (``"amsua"``, ``"mhs"`` or ``"atms"``).
    plan : Mapping[str, str]
        ``{variable: channel_field}`` where ``channel_field`` is the BUFR
        mnemonic inside the channel sequence (``TMBR`` or ``TMANT``).
    windows : Sequence[tuple[datetime, datetime]]
        Observation-time windows to keep.
    satellites : Sequence[str] | None, optional
        Platform filter, by default None (all platforms).

    Returns
    -------
    pd.DataFrame
        Rows in :data:`earth2studio.data.utils_ncep.NCEP_MICROWAVE_OUTPUT_SCHEMA`.
    """
    schema_out = NCEP_MICROWAVE_OUTPUT_SCHEMA
    dataset = pads.dataset(path, format="parquet")
    schema = dataset.schema
    sequence = _SAT_CHANNEL_SEQUENCE[sensor]
    if not _has_path(schema, (sequence, "CHNM")):
        raise ValueError(f"{path}: channel sequence {sequence}.CHNM not found")

    columns: dict[str, pads.Expression] = {
        "time": pc.field("OBS_TIME"),
        "lat": pc.field("LAT"),
        "lon": pc.field("LON"),
        "said": pc.field("SAID"),
        "fovn": pc.field("FOVN"),
        "saza": pc.field("SAZA"),
        "soza": pc.field("SOZA"),
        "solazi": pc.field("SOLAZI"),
        "bearaz": pc.field("BEARAZ"),
        "chnm": pc.field(sequence, "CHNM"),
    }
    optional = {
        "cycle": ("MSG_DATE",),
        "source": ("SRC_FILENAME",),
        "hols": ("HOLS",),
        "slnm": ("SLNM",),
        "scan_quality": ("ATMSSQ",),
        "granule_quality": ("ATMSGQ",),
        "sccf": (sequence, "SCCF"),
        "quality": (sequence, "ATMSCHQ"),
    }
    for name, field_path in optional.items():
        if _has_path(schema, field_path):
            columns[name] = pc.field(*field_path)
    for variable, channel_field in plan.items():
        if not _has_path(schema, (sequence, channel_field)):
            raise ValueError(
                f"{path}: field {sequence}.{channel_field} for variable "
                f"'{variable}' not found"
            )
        columns[f"obs::{variable}"] = pc.field(sequence, channel_field)

    table = dataset.to_table(
        columns=columns, filter=_window_filter(windows, timedelta())
    )
    if table.num_rows == 0:
        return _microwave_frame(schema_out.empty_table())

    # Footprint (scalar) fields.
    time_ns = _to_numpy(table["time"])
    lat = _to_numpy(table["lat"])
    lon = _to_numpy(table["lon"]) % 360.0
    said = _to_numpy(table["said"])
    fovn = _to_numpy(table["fovn"])
    satellite = np.array(
        [
            (
                _NCEP_SATELLITE_NAME_BY_SAID.get(int(s), f"satellite-{int(s)}")
                if np.isfinite(s)
                else None
            )
            for s in said
        ],
        dtype=object,
    )
    footprint_valid = (
        _in_windows(time_ns, windows)
        & np.isfinite(lat)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & np.isfinite(lon)
        & np.isfinite(said)
    )
    if satellites is not None:
        footprint_valid &= np.isin(satellite, list(satellites))
    start, step = _NCEP_MICROWAVE_SCAN_GEOMETRY[sensor]
    scan_angle = start + (fovn - 1.0) * step

    def footprint(name: str) -> np.ndarray | None:
        return _to_numpy(table[name]) if name in table.column_names else None

    elev = footprint("hols")
    slnm = footprint("slnm")
    cycle = _cycle_time(table)
    scan_quality = footprint("scan_quality")
    granule_quality = footprint("granule_quality")

    # Channel (replicated) fields, flattened with their footprint index.
    chnm = table["chnm"].combine_chunks()
    parent = pc.list_parent_indices(chnm).to_numpy()
    channel = _to_numpy(pc.list_flatten(chnm))
    wavenumber = (
        _to_numpy(pc.list_flatten(table["sccf"].combine_chunks())) / _C_CM_S
        if "sccf" in table.column_names
        else None
    )
    quality = (
        _to_numpy(pc.list_flatten(table["quality"].combine_chunks()))
        if "quality" in table.column_names
        else None
    )

    n_total = len(channel)
    nan_column: np.ndarray = np.full(n_total, np.nan, dtype=np.float64)
    tables: list[pa.Table] = []
    for variable in plan:
        observation = _to_numpy(
            pc.list_flatten(table[f"obs::{variable}"].combine_chunks())
        )
        if len(observation) != n_total:
            raise ValueError(
                f"{path}: {sequence} channel fields have inconsistent lengths"
            )
        keep = np.isfinite(observation) & np.isfinite(channel) & footprint_valid[parent]
        if not keep.any():
            continue
        index = parent[keep]
        n = int(keep.sum())

        def per_footprint(values: np.ndarray | None) -> np.ndarray:
            return values[index] if values is not None else nan_column[:n]

        def per_channel(values: np.ndarray | None) -> np.ndarray:
            return values[keep] if values is not None else nan_column[:n]

        numeric: dict[str, np.ndarray] = {
            "lat": lat[index],
            "lon": lon[index],
            "elev": per_footprint(elev),
            "scan_angle": scan_angle[index],
            "scan_position": fovn[index],
            "scan_line": per_footprint(slnm),
            "sensor_index": channel[keep],
            "wavenumber": per_channel(wavenumber),
            "solza": _to_numpy(table["soza"])[index],
            "solaza": _to_numpy(table["solazi"])[index],
            "satellite_za": _to_numpy(table["saza"])[index],
            "satellite_aza": _to_numpy(table["bearaz"])[index],
            "quality": per_channel(quality),
            "scan_quality": per_footprint(scan_quality),
            "granule_quality": per_footprint(granule_quality),
            "observation": observation[keep],
        }
        strings: dict[str, Any] = {
            "class": ["rad"] * n,
            "satellite": satellite[index],
            "variable": [variable] * n,
        }
        times: dict[str, np.ndarray | None] = {
            "time": time_ns[index],
            "cycle_time": cycle[index],
        }
        arrays: list[pa.Array] = []
        for field in schema_out:
            if field.name in numeric:
                values = numeric[field.name]
                if pa.types.is_integer(field.type):
                    arrays.append(_uint_column(values, field.type))
                else:
                    arrays.append(_float_column(values, field.type))
            elif field.name in strings:
                arrays.append(pa.array(strings[field.name], pa.string()))
            elif field.name in times and times[field.name] is not None:
                arrays.append(pa.array(times[field.name]).cast(field.type))
            else:
                # Fields this archive does not carry (e.g. CrIS detector).
                arrays.append(pa.nulls(n, field.type))
        tables.append(pa.Table.from_arrays(arrays, schema=schema_out))
    if not tables:
        return _microwave_frame(schema_out.empty_table())
    return _microwave_frame(pa.concat_tables(tables))


def decode_nnja_ai_v2_prepbufr(
    path: str,
    family: str,
    plan: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    windows: Sequence[Window],
    original_event: bool = False,
) -> pd.DataFrame:
    """Read one NNJA-AI v2 PrepBUFR family partition into a long DataFrame.

    Parameters
    ----------
    path : str
        Local path of the daily Parquet partition.
    family : str
        PrepBUFR message family, e.g. ``"ADPUPA"``; written to ``class``.
    plan : Mapping[str, Callable]
        ``{variable: modifier}`` where ``modifier`` converts raw PrepBUFR
        units to Earth2Studio units (from
        :class:`earth2studio.lexicon.NNJAObsConvLexicon`).
    windows : Sequence[tuple[datetime, datetime]]
        Observation-time windows to keep.
    original_event : bool, optional
        Emit each observation's original report (oldest event) instead of its
        latest quality-controlled event, by default False.

    Returns
    -------
    pd.DataFrame
        Rows in :data:`earth2studio.data.utils_ncep.NCEP_CONVENTIONAL_PUBLIC_SCHEMA`.
    """
    schema_out = NCEP_CONVENTIONAL_PUBLIC_SCHEMA
    dataset = pads.dataset(path, format="parquet")
    schema = dataset.schema
    suffix = "_RAW" if original_event else "_QC"
    if f"POB{suffix}" not in schema.names:
        raise ValueError(f"{path}: PrepBUFR pressure column POB{suffix} not found")
    # Level fields sit under a PRSLEVEL/PRSLEVLA sequence in profile families;
    # the flat POB/TOB/... columns are lists there and scalars for
    # single-level families (ADPSFC, SFCSHP, SATWND, ...).
    level_sequence = next(
        (name for name in ("PRSLEVEL", "PRSLEVLA") if name in schema.names), None
    )
    prefix: tuple[str, ...] = (level_sequence,) if level_sequence else ()
    multi_level = pa.types.is_list(schema.field(f"POB{suffix}").type)
    has_drift = _has_path(schema, (*prefix, "DRFTINFO", "HRDR"))
    pad = _CONV_DRIFT_PAD if has_drift else timedelta()

    columns: dict[str, pads.Expression] = {
        "time": pc.field("OBS_TIME"),
        "cycle": pc.field("MSG_DATE"),
        "source": pc.field("SRC_FILENAME"),
        "lat": pc.field("LAT"),
        "lon": pc.field("LON"),
        "station": pc.field("SID"),
        "type": pc.field("TYP"),
        "station_elev": pc.field("ELV"),
        "pres": pc.field(f"POB{suffix}"),
    }
    optional = {
        "elev": (f"ZOB{suffix}",),
        "level_cat": (*prefix, "CAT"),
        "pressure_quality": (*prefix, "P___INFO", "P__EVENT", "PQM"),
        "hrdr": (*prefix, "DRFTINFO", "HRDR"),
        "xdr": (*prefix, "DRFTINFO", "XDR"),
        "ydr": (*prefix, "DRFTINFO", "YDR"),
    }
    for name, field_path in optional.items():
        if _has_path(schema, field_path):
            columns[name] = pc.field(*field_path)
    variables: list[str] = []
    for variable in plan:
        mnemonic, quality_path = _CONV_VARIABLE_FIELDS[variable]
        if f"{mnemonic}{suffix}" not in schema.names:
            continue  # family does not report this variable (e.g. ASCATW TOB)
        variables.append(variable)
        columns[f"obs::{variable}"] = pc.field(f"{mnemonic}{suffix}")
        full_quality_path = (*prefix, *quality_path)
        if _has_path(schema, full_quality_path):
            columns[f"quality::{variable}"] = pc.field(*full_quality_path)
    if not variables:
        return empty_dataframe(schema_out)

    table = dataset.to_table(columns=columns, filter=_window_filter(windows, pad))
    if table.num_rows == 0:
        return empty_dataframe(schema_out)

    # Expand per-level list columns to one row per level; single-level
    # families (ADPSFC, SFCSHP, ...) are already one row per report.
    parent: np.ndarray
    if multi_level:
        reference = table["pres"].combine_chunks()
        lengths = _list_lengths(reference)
        parent = np.repeat(np.arange(len(lengths)), lengths)

        def level_values(name: str) -> np.ndarray:
            column = table[name].combine_chunks()
            if not np.array_equal(_list_lengths(column), lengths):
                raise ValueError(f"{path}: {name} is not aligned with POB levels")
            return _to_numpy(pc.list_flatten(column))

        def level_quality(name: str) -> np.ndarray:
            column = table[name].combine_chunks()
            if not np.array_equal(_list_lengths(column), lengths):
                raise ValueError(f"{path}: {name} is not aligned with POB levels")
            return _event_edge(pc.list_flatten(column), newest=not original_event)

    else:
        parent = np.arange(table.num_rows)

        def level_values(name: str) -> np.ndarray:
            return _to_numpy(table[name])

        def level_quality(name: str) -> np.ndarray:
            return _event_edge(table[name], newest=not original_event)

    report_time = _to_numpy(table["time"])[parent]
    report_lat = _to_numpy(table["lat"])[parent]
    report_lon = _to_numpy(table["lon"])[parent] % 360.0
    cycle = _cycle_time(table)[parent]
    time_ns = report_time
    lat = report_lat
    lon = report_lon
    if "hrdr" in table.column_names:
        hours = level_values("hrdr")
        drift = np.isfinite(hours)
        offset_ns = np.rint(np.nan_to_num(hours) * 3.6e12).astype(np.int64)
        time_ns = np.where(drift, cycle + offset_ns.astype("timedelta64[ns]"), time_ns)
    if "ydr" in table.column_names and "xdr" in table.column_names:
        ydr = level_values("ydr")
        xdr = level_values("xdr")
        lat = np.where(np.isfinite(ydr), ydr, lat)
        lon = np.where(np.isfinite(xdr), xdr, lon)
    lon = lon % 360.0

    keep_level = (
        _in_windows(time_ns, windows)
        & np.isfinite(lat)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & np.isfinite(lon)
    )
    n_levels = len(parent)
    nan_column: np.ndarray = np.full(n_levels, np.nan, dtype=np.float64)
    pres = level_values("pres")
    elev = level_values("elev") if "elev" in table.column_names else nan_column
    level_cat = (
        level_values("level_cat") if "level_cat" in table.column_names else nan_column
    )
    pressure_quality = (
        level_quality("pressure_quality")
        if "pressure_quality" in table.column_names
        else nan_column
    )
    station = _to_numpy(table["station"])[parent]
    report_type = _to_numpy(table["type"])[parent]
    station_elev = _to_numpy(table["station_elev"])[parent]

    frames: list[pd.DataFrame] = []
    for variable in variables:
        observation = level_values(f"obs::{variable}")
        quality = (
            level_quality(f"quality::{variable}")
            if f"quality::{variable}" in table.column_names
            else nan_column
        )
        keep = keep_level & np.isfinite(observation)
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "time": time_ns[keep],
                    "report_time": report_time[keep],
                    "cycle_time": cycle[keep],
                    "report_lat": report_lat[keep].astype(np.float32),
                    "report_lon": report_lon[keep].astype(np.float32),
                    "pres": pres[keep].astype(np.float32),
                    "elev": elev[keep].astype(np.float32),
                    "type": pd.array(report_type[keep], dtype="UInt16"),
                    "level_cat": pd.array(level_cat[keep], dtype="UInt16"),
                    "class": family,
                    "lat": lat[keep].astype(np.float32),
                    "lon": lon[keep].astype(np.float32),
                    "station": station[keep],
                    "station_elev": station_elev[keep].astype(np.float32),
                    "quality": pd.array(quality[keep], dtype="UInt16"),
                    "pressure_quality": pd.array(
                        pressure_quality[keep], dtype="UInt16"
                    ),
                    "observation": observation[keep].astype(np.float32),
                    "variable": variable,
                }
            )
        )
    if not frames:
        return empty_dataframe(schema_out)
    frame = pd.concat(frames, ignore_index=True)
    return _finalize_rows(
        frame,
        {variable: plan[variable] for variable in variables},
        convert_pres_mb_to_pa=True,
        schema=schema_out,
    )


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


class NNJAAIv2PartitionMissingError(RuntimeError):
    """A requested NNJA-AI v2 daily partition does not exist in the archive."""


@dataclass(frozen=True)
class _NNJAAIv2Task:
    """One daily partition to read, with the windows and variables wanted."""

    uri: str
    key: str
    product: _NNJAAIv2Product
    stream: str
    windows: tuple[Window, ...]
    var_plan: Mapping[str, Any]


class _NNJAAIv2Source:
    """Shared fetch and cache machinery for the NNJA-AI v2 sources."""

    SOURCE_ID: str
    SCHEMA: pa.Schema
    MIN_DATE: datetime

    def __init__(
        self,
        time_tolerance: TimeTolerance,
        root: str,
        cache: bool,
        verbose: bool,
        async_timeout: int,
        async_workers: int,
        retries: int,
    ) -> None:
        self._root = root.rstrip("/")
        self._store_url, self._prefix = _split_root(self._root)
        self._verbose = verbose
        self._cache = cache
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = uuid.uuid4().hex[:8] if not cache else None
        self._store: Any = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

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
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Long-format observation DataFrame.
        """
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
        """Async function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Long-format observation DataFrame.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        windows = _merge_windows(
            time_list, self._tolerance_lower, self._tolerance_upper
        )
        tasks = self._create_tasks(windows, variable_list)
        await self._fetch_files(tasks)
        return self._compile(tasks, schema)

    def _compile(
        self, tasks: Sequence[_NNJAAIv2Task], schema: pa.Schema
    ) -> pd.DataFrame:
        """Read every cached partition and concatenate the requested fields."""
        frames: list[pd.DataFrame] = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, start=1):
            path = self.local_path(task.uri)
            if not os.path.isfile(path):
                logger.debug(f"No cached partition for {task.uri}, skipping")
                continue
            t0 = time.perf_counter()
            try:
                frame = self._decode_file(path, task)
            except Exception as exc:
                raise RuntimeError(
                    f"{type(self).__name__} failed to read {task.uri} "
                    f"({index}/{len(tasks)}): {exc}"
                ) from exc
            logger.info(
                f"[{self.SOURCE_ID}] read {index}/{len(tasks)} "
                f"{task.uri.rsplit('/', 2)[-2]}/{task.uri.rsplit('/', 1)[-1]}: "
                f"{len(frame):,} rows in {time.perf_counter() - t0:.1f}s"
            )
            if not frame.empty:
                frames.append(frame)
        logger.info(
            f"[{self.SOURCE_ID}] {len(frames)} non-empty partitions in "
            f"{time.perf_counter() - started:.1f}s"
        )
        if frames:
            result = pd.concat(frames, ignore_index=True)[list(schema.names)]
        else:
            result = empty_dataframe(schema)
        result.attrs["source"] = self.SOURCE_ID
        return result

    # -- remote files -----------------------------------------------------
    def _make_task(
        self,
        product: _NNJAAIv2Product,
        stream: str,
        day: date,
        windows: Sequence[Window],
        var_plan: Mapping[str, Any],
    ) -> _NNJAAIv2Task:
        partition = product.partition_key(day)
        key = f"{self._prefix}/{partition}" if self._prefix else partition
        return _NNJAAIv2Task(
            uri=f"{self._root}/{partition}",
            key=key,
            product=product,
            stream=stream,
            windows=tuple(windows),
            var_plan=dict(var_plan),
        )

    async def _fetch_files(self, tasks: Sequence[_NNJAAIv2Task]) -> None:
        unique = {task.uri: task for task in tasks}
        if self._store is None:
            self._store = obstore_store_from_url(
                self._store_url, max_pool_connections=self._async_workers
            )
        coros = [
            async_retry(
                self._fetch_remote_file,
                task,
                retries=self._retries,
                backoff=1.0,
                task_timeout=600.0,
                exceptions=(OSError, TimeoutError, ConnectionError),
            )
            for task in unique.values()
        ]
        await gather_with_concurrency(
            coros,
            max_workers=self._async_workers,
            desc=f"Fetching {type(self).__name__} files",
            verbose=(not self._verbose),
        )

    async def _fetch_remote_file(self, task: _NNJAAIv2Task) -> None:
        try:
            await obstore_fetch_to_cache(
                self._store,
                task.key,
                self.cache,
                cache_key=hashlib.sha256(task.uri.encode()).hexdigest(),
            )
        except FileNotFoundError:
            self._handle_missing_file(task)

    def _handle_missing_file(self, task: _NNJAAIv2Task) -> None:
        raise NNJAAIv2PartitionMissingError(
            f"NNJA-AI v2 partition not found: {task.uri}"
        )

    def local_path(self, uri: str) -> str:
        """Return the deterministic cache path for a partition URI."""
        return os.path.join(self.cache, hashlib.sha256(uri.encode()).hexdigest())

    @property
    def cache(self) -> str:
        """Local cache directory for NNJA-AI v2 partition files."""
        return datasource_cache_dir("nnja_ai_v2", self._cache, self._tmp_cache_hash)

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    # -- subclass hooks ---------------------------------------------------
    def _create_tasks(
        self, windows: list[Window], variable: list[str]
    ) -> list[_NNJAAIv2Task]:
        raise NotImplementedError

    def _decode_file(self, local_path: str, task: _NNJAAIv2Task) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
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


class NNJAAIv2ObsSat(_NNJAAIv2Source):
    """NNJA-AI v2 microwave satellite observations (AMSU-A, MHS, ATMS).

    Reads Brightband's pre-decoded Parquet edition of the NNJA archive, one
    file per UTC day and sensor, instead of fetching and decoding six-hourly
    BUFR cycle files. The output schema and semantics match
    :py:class:`earth2studio.data.NNJAObsSat`: one row per finite channel
    value, ``sensor_index`` is the physical ``CHNM`` channel number,
    ``scan_position`` the one-based field of view and ``scan_angle`` the
    signed nominal look angle derived from it. Encoded ``TMBR`` is returned
    as-is (see ``NNJAObsSat`` for the antenna/brightness temperature caveats
    of the legacy sounders).

    Only the microwave sounders published in the v2 archive are supported:
    ``amsua``, ``mhs``, ``atms`` and ``atms_antenna_temperature``. AMSU-B
    and the hyperspectral IR sounders are not available through this source.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric +/- window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(10, 'm').
    satellites : list[str] | None, optional
        Satellite platforms to include. ``None`` includes every platform in
        the requested files, by default None.
    root : str, optional
        Archive root URL, by default ``"gs://nnja-ai/data/v2"``. ``gs://``,
        ``s3://`` and ``file://`` roots are supported.
    cache : bool, optional
        Cache downloaded partition files in the local filesystem cache, by
        default True. Partition files hold a whole UTC day, so a persistent
        cache is strongly recommended for repeated or multi-cycle requests.
    verbose : bool, optional
        Show progress bars, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 1200.
    async_workers : int, optional
        Maximum number of concurrent downloads, by default 8.
    retries : int, optional
        Number of retry attempts per failed download, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests. A daily ATMS partition is about
    1.3 GB; AMSU-A and MHS are 40-200 MB per day.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.NNJAAIv2ObsSat"
    SCHEMA = NCEP_MICROWAVE_OUTPUT_SCHEMA
    LEXICON = NNJAObsSatLexicon
    MIN_DATE = datetime(2005, 1, 1)
    VALID_SATELLITES = NCEP_MICROWAVE_SATELLITES
    VALID_VARIABLES = frozenset(
        variable
        for variable, key in NNJAObsSatLexicon.VOCAB.items()
        if key.partition("::")[0] in _SAT_PRODUCTS
    )

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        root: str = NNJA_AI_V2_ROOT,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 1200,
        async_workers: int = 8,
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
        super().__init__(
            time_tolerance, root, cache, verbose, async_timeout, async_workers, retries
        )

    def _create_tasks(
        self, windows: list[Window], variable: list[str]
    ) -> list[_NNJAAIv2Task]:
        plans: dict[str, dict[str, str]] = {}
        for name in variable:
            if name not in self.VALID_VARIABLES:
                raise ValueError(
                    f"Unsupported {type(self).__name__} variable '{name}'. "
                    f"Supported: {sorted(self.VALID_VARIABLES)}"
                )
            source_key, _ = self.LEXICON[name]
            sensor, _, channel_field = source_key.partition("::")
            plans.setdefault(sensor, {})[name] = channel_field

        tasks: list[_NNJAAIv2Task] = []
        for sensor, plan in plans.items():
            product = _SAT_PRODUCTS[sensor]
            for day in _partition_days(windows, timedelta()):
                if day.year < product.first_year:
                    raise ValueError(
                        f"NNJA-AI v2 {sensor} archive starts in "
                        f"{product.first_year}; requested {day.isoformat()}"
                    )
                tasks.append(self._make_task(product, sensor, day, windows, plan))
        return tasks

    def _decode_file(self, local_path: str, task: _NNJAAIv2Task) -> pd.DataFrame:
        frame = decode_nnja_ai_v2_microwave(
            local_path, task.stream, task.var_plan, task.windows, self._satellites
        )
        return frame[self.SCHEMA.names]


class NNJAAIv2ObsConv(_NNJAAIv2Source):
    """NNJA-AI v2 conventional (in-situ) observations from PrepBUFR families.

    Reads Brightband's pre-decoded Parquet edition of the NNJA merged
    PrepBUFR product, published as one file per UTC day and message family
    (``ADPUPA``, ``ADPSFC``, ``AIRCAR``, ...). The output schema and unit
    conventions match :py:class:`earth2studio.data.NNJAObsConv`: PrepBUFR
    report type, level category and quality marks are carried through,
    ``pres`` is in Pa, ``t`` in K and ``q`` in kg kg-1. Balloon and aircraft
    drift (``DRFTINFO``) is applied to per-level time and position where the
    family reports it.

    GPS radio occultation (``gps``) is not provided by this source; use
    :py:class:`earth2studio.data.NNJAObsConv` for it.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric +/- window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(0, 'm').
    exclude_message_types : Sequence[str], optional
        PrepBUFR message families whose partition files are skipped entirely
        (e.g. ``("SATWND", "AIRCAR")``), by default ().
    original_event : bool, optional
        Emit each observation's original report (oldest PrepBUFR event)
        instead of its latest quality-controlled event, by default False.
    root : str, optional
        Archive root URL, by default ``"gs://nnja-ai/data/v2"``. ``gs://``,
        ``s3://`` and ``file://`` roots are supported.
    cache : bool, optional
        Cache downloaded partition files in the local filesystem cache, by
        default True. Partition files hold a whole UTC day, so a persistent
        cache is strongly recommended for repeated or multi-cycle requests.
    verbose : bool, optional
        Show progress bars, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 1200.
    async_workers : int, optional
        Maximum number of concurrent downloads, by default 8.
    retries : int, optional
        Number of retry attempts per failed download, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests. One day of all PrepBUFR
    families is about 300 MB.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm

    Badges
    ------
    region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAAIv2ObsConv"
    SCHEMA = NCEP_CONVENTIONAL_PUBLIC_SCHEMA
    LEXICON = NNJAObsConvLexicon
    MIN_DATE = datetime(_CONV_FIRST_YEAR, 1, 1)
    VALID_MESSAGE_TYPES = frozenset(NNJA_AI_V2_PREPBUFR_FAMILIES)
    VALID_VARIABLES = frozenset(_CONV_VARIABLE_FIELDS)

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        exclude_message_types: Sequence[str] = (),
        original_event: bool = False,
        root: str = NNJA_AI_V2_ROOT,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 1200,
        async_workers: int = 8,
        retries: int = 3,
    ) -> None:
        unknown = set(exclude_message_types) - self.VALID_MESSAGE_TYPES
        if unknown:
            raise ValueError(
                f"Unknown PrepBUFR message type(s) {sorted(unknown)}. "
                f"Valid options: {sorted(self.VALID_MESSAGE_TYPES)}"
            )
        self._families = tuple(
            family
            for family in NNJA_AI_V2_PREPBUFR_FAMILIES
            if family not in set(exclude_message_types)
        )
        self._original_event = original_event
        super().__init__(
            time_tolerance, root, cache, verbose, async_timeout, async_workers, retries
        )

    def _create_tasks(
        self, windows: list[Window], variable: list[str]
    ) -> list[_NNJAAIv2Task]:
        unsupported = set(variable) & _CONV_UNSUPPORTED_VARIABLES
        if unsupported:
            raise ValueError(
                f"{type(self).__name__} does not provide {sorted(unsupported)}; "
                "GPS radio occultation is read by NNJAObsConv"
            )
        unknown = set(variable) - self.VALID_VARIABLES
        if unknown:
            raise ValueError(
                f"Unsupported {type(self).__name__} variable(s) {sorted(unknown)}. "
                f"Supported: {sorted(self.VALID_VARIABLES)}"
            )
        plan = {name: self.LEXICON[name][1] for name in variable}
        days = _partition_days(windows, _CONV_DRIFT_PAD)
        return [
            self._make_task(
                _NNJAAIv2Product("conv", "prepbufr", family, _CONV_FIRST_YEAR),
                family,
                day,
                windows,
                plan,
            )
            for family in self._families
            for day in days
        ]

    def _decode_file(self, local_path: str, task: _NNJAAIv2Task) -> pd.DataFrame:
        frame = decode_nnja_ai_v2_prepbufr(
            local_path,
            task.stream,
            task.var_plan,
            task.windows,
            original_event=self._original_event,
        )
        return frame[self.SCHEMA.names]

    def _handle_missing_file(self, task: _NNJAAIv2Task) -> None:
        # Not every family is published for every day (SYNDAT is often empty
        # or absent); a partial frame beats failing the whole request.
        logger.warning(f"NNJA-AI v2 partition {task.uri} not found, skipping")
