# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NCEP conventional PrepBUFR and GPSRO decode utilities."""

from __future__ import annotations

import asyncio
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import _sync_async, prep_data_inputs
from earth2studio.data.utils_bufr import (
    HDR_DHR,
    HDR_ELV,
    HDR_SID,
    HDR_TYP,
    HDR_XOB,
    HDR_YOB,
    MNEMONIC_TO_DESCR,
    OBS_CAT,
    OBS_HRDR,
    OBS_POB,
    OBS_PQM,
    OBS_QUALITY_MAP,
    OBS_UOB,
    OBS_VOB,
    OBS_WQM,
    OBS_XDR,
    OBS_YDR,
    OBS_ZOB,
    OBSERVATION_DESCR_IDS,
    PREPBUFR_OBS_TYPES,
)
from earth2studio.data.utils_bufr import create_decoder as _create_decoder
from earth2studio.data.utils_bufr import (
    parse_prepbufr_messages as _parse_prepbufr_messages,
)
from earth2studio.data.utils_bufr import silence_bufr_noise as _silence_bufr_noise
from earth2studio.lexicon.base import E2STUDIO_SCHEMA, LexiconType
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

# GPS RO BUFR descriptor IDs (NCEP gpsro encoding). GPSRO is not a PrepBUFR
# conventional report, so conventional semantics do not apply: there is no TYP
# report type (the decoder stores receiver SAID in the shared ``type`` column,
# matching GSI/UFS diagnostics) and no 0-15 quality mark (QFRO, a WMO/NCEP
# radio-occultation flag table, is stored in ``quality`` for schema uniformity
# and must not be read as a conventional QM).
GPSRO_SAID = 1007
GPSRO_PTID = 1050
GPSRO_QFRO = 33039
GPSRO_ELRC = 10035
GPSRO_LAT = 5001
GPSRO_LON = 6001
GPSRO_YEAR = 4001
GPSRO_MONTH = 4002
GPSRO_DAY = 4003
GPSRO_HOUR = 4004
GPSRO_MIN = 4005
GPSRO_SEC = 4006
GPSRO_MEFR = 2121
GPSRO_IMPP = 7040
GPSRO_BNDA = 15037


NCEP_CONVENTIONAL_PUBLIC_SCHEMA = pa.schema(
    [
        E2STUDIO_SCHEMA.field("time"),
        pa.field(
            "pres",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "Observation pressure coordinate (Pa); null for "
                    "source-native GPSRO bending-angle rows"
                )
            },
        ),
        pa.field(
            "elev",
            pa.float32(),
            nullable=True,
            metadata={
                "description": (
                    "Observation height (m); GPSRO uses impact parameter "
                    "minus Earth radius of curvature"
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
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
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

_NULL_FLOAT_DTYPES: dict[str, type[np.floating[Any]]] = {
    "pres": np.float32,
    "elev": np.float32,
    "station_elev": np.float32,
    "lat": np.float32,
    "lon": np.float32,
    "observation": np.float32,
}


def _message_base_time(message: Any) -> datetime | None:
    """Read a BUFR Section 1 reference date without edition-3 second leakage."""
    try:
        edition = int(message.edition.value)
        year = int(message.year.value)
        if edition == 3:
            # Edition 3 stores a 2-digit year (year of century); expand to
            # 4-digit, pivoting at 70 (>=70 -> 19xx, <70 -> 20xx).
            if year < 100:
                year += 2000 if year < 70 else 1900
            second = 0
        else:
            second = int(message.second.value)
        return datetime(
            year,
            int(message.month.value),
            int(message.day.value),
            int(message.hour.value),
            int(message.minute.value),
            second,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, with halves away from zero."""
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def _time_from_offset(
    base_time: datetime, offset_hours: Any, scale: int
) -> datetime | None:
    """Add a decoded hour offset at its embedded descriptor precision."""
    if offset_hours is None:
        return None
    try:
        offset = Decimal(str(offset_hours))
        if not offset.is_finite():
            return None
        ticks_per_hour = 10**scale
        ticks = int((offset * ticks_per_hour).to_integral_value(rounding=ROUND_HALF_UP))
        microseconds = _round_ratio(ticks * 3_600_000_000, ticks_per_hour)
        return base_time + timedelta(microseconds=microseconds)
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return None


def _descriptor_scale(
    table_b: dict[int, tuple[Any, ...]], descriptor_id: int, default: int
) -> int:
    entry = table_b.get(descriptor_id)
    if entry is None or len(entry) < 3:
        return default
    try:
        return int(entry[2])
    except (TypeError, ValueError):
        return default


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    return str(value).strip()


def _decode_prepbufr_message(
    decoder: Any,
    message_bytes: bytes,
    obs_class: str,
    var_keys: Sequence[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
    dhr_scale: int,
    hrdr_scale: int,
) -> list[dict[str, Any]]:
    try:
        message = decoder.process(message_bytes)
    except Exception:
        return []
    if not message.n_subsets.value:
        return []
    base_time = _message_base_time(message)
    if base_time is None:
        return []

    template_data = message.template_data.value
    rows: list[dict[str, Any]] = []
    for descriptors, values in zip(
        template_data.decoded_descriptors_all_subsets,
        template_data.decoded_values_all_subsets,
    ):
        rows.extend(
            _extract_prepbufr_subset(
                descriptors,
                values,
                base_time,
                obs_class,
                var_keys,
                dt_min,
                dt_max,
                dhr_scale,
                hrdr_scale,
            )
        )
    return rows


def _extract_prepbufr_subset(
    descriptors: list[Any],
    values: list[Any],
    base_time: datetime,
    obs_class: str,
    var_keys: Sequence[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
    dhr_scale: int = 5,
    hrdr_scale: int = 5,
) -> list[dict[str, Any]]:
    """Extract first-event observations from CAT-delimited physical levels."""
    header: dict[str, Any] = {
        "sid": "",
        "xob": None,
        "yob": None,
        "dhr": 0.0,
        "elv": None,
        "typ": None,
    }
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if descriptor_id == HDR_SID:
            header["sid"] = _decode_text(value)
        elif descriptor_id == HDR_XOB:
            header["xob"] = value
        elif descriptor_id == HDR_YOB:
            header["yob"] = value
        elif descriptor_id == HDR_DHR:
            header["dhr"] = value if value is not None else 0.0
        elif descriptor_id == HDR_ELV:
            header["elv"] = value
        elif descriptor_id == HDR_TYP:
            header["typ"] = value
        elif descriptor_id == OBS_CAT:
            break

    header_time = _time_from_offset(base_time, header["dhr"], dhr_scale) or base_time
    needed_ids: dict[str, int] = {}
    need_wind = False
    for variable, key in var_keys:
        if key.startswith("wind::"):
            need_wind = True
        elif key in MNEMONIC_TO_DESCR:
            needed_ids[variable] = MNEMONIC_TO_DESCR[key]

    base_row: dict[str, Any] = {
        "time": header_time,
        "lat": np.float32(header["yob"]) if header["yob"] is not None else None,
        "lon": (
            np.float32(float(header["xob"]) % 360.0)
            if header["xob"] is not None
            else None
        ),
        "pres": None,
        "elev": None,
        "type": np.uint16(int(header["typ"])) if header["typ"] is not None else None,
        "class": obs_class or None,
        "station": header["sid"] or None,
        "station_elev": (
            np.float32(header["elv"]) if header["elv"] is not None else None
        ),
        "quality": None,
        "pressure_quality": None,
    }

    # PrepBUFR repeats CAT blocks; each CAT starts a new physical level. Within a
    # level keep only the first occurrence of each descriptor (the observation);
    # later repeats are event-stack history, not new observations.
    rows: list[dict[str, Any]] = []
    level: dict[int, Any] = {}
    in_level = False
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if descriptor_id == OBS_CAT:
            if in_level:
                _emit_prepbufr_level(
                    rows,
                    level,
                    base_row,
                    base_time,
                    dt_min,
                    dt_max,
                    hrdr_scale,
                    needed_ids,
                    need_wind,
                    var_keys,
                )
            level = {OBS_CAT: value}
            in_level = True
        elif in_level and descriptor_id in OBSERVATION_DESCR_IDS:
            level.setdefault(descriptor_id, value)
    if in_level:
        _emit_prepbufr_level(
            rows,
            level,
            base_row,
            base_time,
            dt_min,
            dt_max,
            hrdr_scale,
            needed_ids,
            need_wind,
            var_keys,
        )
    return rows


def _emit_prepbufr_level(
    rows: list[dict[str, Any]],
    level: dict[int, Any],
    base_row: dict[str, Any],
    base_time: datetime,
    dt_min: datetime,
    dt_max: datetime,
    hrdr_scale: int,
    needed_ids: dict[str, int],
    need_wind: bool,
    var_keys: Sequence[tuple[str, str]],
) -> None:
    common = base_row.copy()
    level_time = _time_from_offset(base_time, level.get(OBS_HRDR), hrdr_scale)
    if level_time is not None:
        common["time"] = level_time
    if common["time"] < dt_min or common["time"] > dt_max:
        return

    level_lat = level.get(OBS_YDR)
    level_lon = level.get(OBS_XDR)
    if level_lat is not None:
        common["lat"] = np.float32(level_lat)
    if level_lon is not None:
        common["lon"] = np.float32(float(level_lon) % 360.0)
    if common["lat"] is None or common["lon"] is None:
        return
    lat = float(common["lat"])
    lon = float(common["lon"])
    if not np.isfinite(lat) or not np.isfinite(lon) or not -90.0 <= lat <= 90.0:
        return

    pressure = level.get(OBS_POB)
    common["pres"] = np.float32(pressure) if pressure is not None else None
    # ZOB is the observation level's height; header ELV remains station_elev.
    height = level.get(OBS_ZOB)
    if height is not None:
        common["elev"] = np.float32(height)
    pressure_quality = level.get(OBS_PQM)
    common["pressure_quality"] = (
        np.uint16(int(pressure_quality)) if pressure_quality is not None else None
    )
    level_cat = level.get(OBS_CAT)
    common["level_cat"] = np.uint16(int(level_cat)) if level_cat is not None else None

    for variable, descriptor_id in needed_ids.items():
        value = level.get(descriptor_id)
        if value is None:
            continue
        row = common.copy()
        row["variable"] = variable
        row["observation"] = np.float32(value)
        quality_id = OBS_QUALITY_MAP.get(descriptor_id)
        quality = level.get(quality_id) if quality_id is not None else None
        row["quality"] = np.uint16(int(quality)) if quality is not None else None
        rows.append(row)

    if not need_wind:
        return
    # Wind decomposition: u from UOB, v from VOB, sharing WQM. Each component is
    # emitted independently so a level with only one of UOB/VOB still yields a row.
    wind_quality_value = level.get(OBS_WQM)
    wind_quality = (
        np.uint16(int(wind_quality_value)) if wind_quality_value is not None else None
    )
    for variable, key in var_keys:
        descriptor_id = OBS_UOB if key == "wind::u" else OBS_VOB
        if key not in {"wind::u", "wind::v"}:
            continue
        value = level.get(descriptor_id)
        if value is None:
            continue
        row = common.copy()
        row["variable"] = variable
        row["observation"] = np.float32(value)
        row["quality"] = wind_quality
        rows.append(row)


def _decode_gpsro_message(
    decoder: Any,
    message_bytes: bytes,
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    try:
        message = decoder.process(message_bytes)
    except Exception:
        return []
    if not message.n_subsets.value:
        return []
    template_data = message.template_data.value
    rows: list[dict[str, Any]] = []
    for descriptors, values in zip(
        template_data.decoded_descriptors_all_subsets,
        template_data.decoded_values_all_subsets,
    ):
        rows.extend(
            _extract_gpsro_subset(descriptors, values, wanted_descrs, dt_min, dt_max)
        )
    return rows


def _extract_gpsro_subset(
    descriptors: list[Any],
    values: list[Any],
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Extract the MEFR=0 first-BNDA bending angle from one occultation."""
    sat_id: Any = None
    transmitter_id: Any = None
    quality_flag: Any = None
    radius: float | None = None
    lat: float | None = None
    lon: float | None = None
    year = month = day = hour = minute = None
    second = 0.0
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if descriptor_id == GPSRO_SAID:
            sat_id = value
        elif descriptor_id == GPSRO_PTID:
            transmitter_id = value
        elif descriptor_id == GPSRO_QFRO:
            quality_flag = value
        elif descriptor_id == GPSRO_ELRC and value is not None:
            radius = float(value)
        elif descriptor_id == GPSRO_LAT and value is not None:
            lat = float(value)
        elif descriptor_id == GPSRO_LON and value is not None:
            lon = float(value)
        elif descriptor_id == GPSRO_YEAR and value is not None:
            year = int(value)
        elif descriptor_id == GPSRO_MONTH and value is not None:
            month = int(value)
        elif descriptor_id == GPSRO_DAY and value is not None:
            day = int(value)
        elif descriptor_id == GPSRO_HOUR and value is not None:
            hour = int(value)
        elif descriptor_id == GPSRO_MIN and value is not None:
            minute = int(value)
        elif descriptor_id == GPSRO_SEC and value is not None:
            try:
                second = float(value)
            except (TypeError, ValueError):
                second = 0.0
        elif descriptor_id == GPSRO_IMPP:
            break

    if lat is None or lon is None or year is None or month is None or day is None:
        return []
    try:
        if not np.isfinite(second):
            return []
        obs_time = datetime(year, month, day, hour or 0, minute or 0) + timedelta(
            seconds=second
        )
    except (TypeError, ValueError, OverflowError):
        return []
    if obs_time < dt_min or obs_time > dt_max:
        return []

    # GSI setupref.f90 writes the GPSRO station id as ``(2(i4.4))``: zero-padded
    # receiver SAID followed by transmitter PTID.
    station = (
        f"{int(sat_id):04d}{int(transmitter_id):04d}"
        if sat_id is not None and transmitter_id is not None
        else None
    )
    current_impact: float | None = None
    current_frequency: float | None = None
    current_lat: float | None = lat
    current_lon: float | None = lon
    bnda_slot = 0
    rows: list[dict[str, Any]] = []

    # Each frequency block lays out MEFR -> IMPP -> two BNDA per frequency; count
    # slots, not values: BNDA #1 is the observation, BNDA #2 is its error.
    for descriptor, value in zip(descriptors, values):
        descriptor_id = descriptor.id
        if descriptor_id == GPSRO_BNDA:
            bnda_slot += 1
        if value is None:
            if descriptor_id == GPSRO_LAT:
                current_lat = None
            elif descriptor_id == GPSRO_LON:
                current_lon = None
            elif descriptor_id == GPSRO_IMPP:
                current_impact = None
            elif descriptor_id == GPSRO_MEFR:
                current_frequency = None
                bnda_slot = 0
            continue
        if descriptor_id == GPSRO_LAT:
            current_lat = float(value)
            continue
        if descriptor_id == GPSRO_LON:
            current_lon = float(value)
            continue
        if descriptor_id == GPSRO_MEFR:
            current_frequency = float(value)
            bnda_slot = 0
            continue
        if descriptor_id == GPSRO_IMPP:
            current_impact = float(value)
            continue
        if descriptor_id not in wanted_descrs or descriptor_id != GPSRO_BNDA:
            continue
        try:
            observation = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(observation):
            continue
        # MEFR == 0 is the ionosphere-corrected (frequency-combined) angle; take
        # only its first BNDA slot (the observation, not the error).
        if current_frequency is None or round(current_frequency) != 0 or bnda_slot != 1:
            continue
        if current_impact is None or radius is None:
            continue
        if current_lat is None or current_lon is None:
            continue

        rows.append(
            {
                "time": obs_time,
                "lat": np.float32(current_lat),
                "lon": np.float32(current_lon % 360.0),
                "pres": None,
                # Impact height = impact parameter - local radius of curvature.
                "elev": np.float32(current_impact - radius),
                # GPSRO has no conventional TYP; store receiver SAID in this
                # shared numeric type column (matches GSI/UFS diagnostics).
                "type": np.uint16(int(sat_id)) if sat_id is not None else None,
                "level_cat": None,
                "class": "GPSRO",
                "station": station,
                "station_elev": None,
                # QFRO is a GPSRO flag table stored in ``quality`` for a uniform
                # schema; it is not the conventional 0-15 QM scale.
                "quality": (
                    np.uint16(int(quality_flag)) if quality_flag is not None else None
                ),
                "pressure_quality": None,
                "observation": np.float32(observation),
                "variable": wanted_descrs[descriptor_id],
            }
        )
    return rows


def _empty_dataframe(
    schema: pa.Schema = NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
) -> pd.DataFrame:
    # Build typed empty columns (not object-dtype ``None``) so ``pd.concat`` does
    # not emit "all-NA columns" FutureWarnings when frames from different
    # sub-archives are concatenated.
    columns: dict[str, pd.Series[Any]] = {}
    for field in schema:
        if pa.types.is_timestamp(field.type):
            dtype: Any = "datetime64[ns]"
        elif pa.types.is_float32(field.type):
            dtype = np.float32
        elif pa.types.is_float64(field.type):
            dtype = np.float64
        elif pa.types.is_uint16(field.type):
            dtype = pd.ArrowDtype(field.type)
        else:
            dtype = object
        columns[field.name] = pd.Series(dtype=dtype)
    return pd.DataFrame(columns)


def _finalize_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
    variables: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]],
    *,
    convert_pres_mb_to_pa: bool,
    schema: pa.Schema = NCEP_CONVENTIONAL_PUBLIC_SCHEMA,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return _empty_dataframe(schema)
        frame = rows.copy()
    elif rows:
        frame = pd.DataFrame(rows)
    else:
        return _empty_dataframe(schema)

    result_frames: list[pd.DataFrame] = []
    for variable, modifier in variables.items():
        variable_frame = frame.loc[frame["variable"] == variable].copy()
        if not variable_frame.empty:
            result_frames.append(modifier(variable_frame))
    if not result_frames:
        return _empty_dataframe(schema)
    frame = pd.concat(result_frames, ignore_index=True)

    # PrepBUFR levels carry POB in mb; convert the schema pressure column to Pa
    # for consistency with the lexicon's ``pres`` observation conversion.
    if convert_pres_mb_to_pa and "pres" in frame:
        frame["pres"] = (frame["pres"].astype(np.float32) * 100.0).astype(np.float32)
    frame["time"] = pd.to_datetime(frame["time"])

    for field in schema:
        if field.name in frame:
            continue
        float_dtype = _NULL_FLOAT_DTYPES.get(field.name)
        if float_dtype is not None:
            frame[field.name] = np.full(len(frame), np.nan, dtype=float_dtype)
        elif pa.types.is_timestamp(field.type):
            frame[field.name] = pd.Series(
                pd.NaT, index=frame.index, dtype="datetime64[ns]"
            )
        elif pa.types.is_uint16(field.type):
            frame[field.name] = pd.Series(
                pd.NA, index=frame.index, dtype=pd.ArrowDtype(field.type)
            )
        else:
            frame[field.name] = pd.Series([None] * len(frame), dtype=object)

    for field in schema:
        if pa.types.is_timestamp(field.type):
            frame[field.name] = pd.to_datetime(frame[field.name])
        elif pa.types.is_uint16(field.type):
            frame[field.name] = pd.to_numeric(
                frame[field.name], errors="coerce"
            ).astype(pd.ArrowDtype(field.type))
        elif pa.types.is_float32(field.type):
            frame[field.name] = pd.to_numeric(
                frame[field.name], errors="coerce"
            ).astype(np.float32)
        elif pa.types.is_float64(field.type):
            frame[field.name] = pd.to_numeric(
                frame[field.name], errors="coerce"
            ).astype(np.float64)
    return frame[schema.names]


def map_aircraft_profile_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Map profile report types by scalar versus wind variable family."""
    if frame.empty:
        return frame
    output = frame.copy()
    wind = output["variable"].isin(["u", "v"])
    output.loc[wind, "type"] = output.loc[wind, "type"].replace(
        _ACFT_PROFILE_WIND_TYPE_MAP
    )
    output.loc[~wind, "type"] = output.loc[~wind, "type"].replace(
        _ACFT_PROFILE_SCALAR_TYPE_MAP
    )
    return output


_worker_decoder: Any = None


def _init_decode_worker(
    table_b: dict[int, tuple[Any, ...]], table_d: dict[int, tuple[Any, ...]]
) -> None:
    global _worker_decoder  # noqa: PLW0603
    _worker_decoder = _create_decoder(table_b, table_d)


def _prepbufr_worker(
    message_bytes: bytes,
    obs_class: str,
    var_keys: Sequence[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
    dhr_scale: int,
    hrdr_scale: int,
) -> list[dict[str, Any]]:
    with _silence_bufr_noise():
        return _decode_prepbufr_message(
            _worker_decoder,
            message_bytes,
            obs_class,
            var_keys,
            dt_min,
            dt_max,
            dhr_scale,
            hrdr_scale,
        )


def _gpsro_worker(
    message_bytes: bytes,
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    with _silence_bufr_noise():
        return _decode_gpsro_message(
            _worker_decoder, message_bytes, wanted_descrs, dt_min, dt_max
        )


class _NCEPPrepbufrAdapter:
    """Decode merged NCEP PrepBUFR bytes independently of their transport."""

    def __init__(self, decode_workers: int = 8) -> None:
        self.decode_workers = max(1, decode_workers)

    def decode_file(
        self,
        path: str,
        plan: Mapping[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        # Keys drive decode (pickled to workers); modifiers stay here for
        # finalize (closures cannot cross the process-pool boundary).
        var_keys = [(variable, key) for variable, (key, _) in plan.items()]
        modifiers = {variable: modifier for variable, (_, modifier) in plan.items()}
        with open(path, "rb") as file:
            file_data = file.read()
        table_b, table_d, messages = _parse_prepbufr_messages(
            file_data, silence_noise=True
        )
        dhr_scale = _descriptor_scale(table_b, HDR_DHR, 5)
        hrdr_scale = _descriptor_scale(table_b, OBS_HRDR, 5)
        work_items = [
            (message_bytes, PREPBUFR_OBS_TYPES[data_category])
            for message_bytes, data_category in messages
            if data_category in PREPBUFR_OBS_TYPES
        ]
        if not work_items:
            return _empty_dataframe()

        rows: list[dict[str, Any]] = []
        use_parallel = self.decode_workers > 1 and len(work_items) >= 32
        started = time.perf_counter()
        if use_parallel:
            with ProcessPoolExecutor(
                max_workers=self.decode_workers,
                initializer=_init_decode_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _prepbufr_worker,
                        message_bytes,
                        obs_class,
                        var_keys,
                        dt_min,
                        dt_max,
                        dhr_scale,
                        hrdr_scale,
                    )
                    for message_bytes, obs_class in work_items
                ]
                for future in futures:
                    try:
                        rows.extend(future.result())
                    except Exception as error:
                        logger.debug(f"PrepBUFR worker failed: {error}")
        else:
            decoder = _create_decoder(table_b, table_d)
            for message_bytes, obs_class in work_items:
                rows.extend(
                    _decode_prepbufr_message(
                        decoder,
                        message_bytes,
                        obs_class,
                        var_keys,
                        dt_min,
                        dt_max,
                        dhr_scale,
                        hrdr_scale,
                    )
                )
        logger.debug(
            f"Decoded {len(rows):,} PrepBUFR rows in "
            f"{time.perf_counter() - started:.1f}s"
        )
        return _finalize_rows(
            rows,
            modifiers,
            convert_pres_mb_to_pa=True,
        )


class _NCEPGpsroAdapter:
    """Decode the existing NCEP combined bending-angle product."""

    def __init__(self, decode_workers: int = 8) -> None:
        self.decode_workers = max(1, decode_workers)

    def decode_file(
        self,
        path: str,
        plan: Mapping[str, tuple[str | int, Callable[[pd.DataFrame], pd.DataFrame]]],
        dt_min: datetime,
        dt_max: datetime,
    ) -> pd.DataFrame:
        # Descriptors drive decode (pickled to workers); modifiers stay here for
        # finalize (closures cannot cross the process-pool boundary). Callers
        # pass the descriptor id as a string (shared planner) or int (GDAS until
        # it migrates). TODO(gdas migration): once GDAS uses the shared planner
        # (str keys), narrow the union to str and drop the int coercion.
        wanted_descrs = {int(key): variable for variable, (key, _) in plan.items()}
        modifiers = {variable: modifier for variable, (_, modifier) in plan.items()}
        with open(path, "rb") as file:
            file_data = file.read()
        table_b, table_d, messages = _parse_prepbufr_messages(
            file_data, silence_noise=True
        )
        if not messages:
            return _empty_dataframe()
        rows: list[dict[str, Any]] = []
        use_parallel = self.decode_workers > 1 and len(messages) >= 32
        if use_parallel:
            with ProcessPoolExecutor(
                max_workers=self.decode_workers,
                initializer=_init_decode_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _gpsro_worker,
                        message_bytes,
                        wanted_descrs,
                        dt_min,
                        dt_max,
                    )
                    for message_bytes, _data_category in messages
                ]
                for future in futures:
                    try:
                        rows.extend(future.result())
                    except Exception as error:
                        logger.debug(f"GPSRO worker failed: {error}")
        else:
            decoder = _create_decoder(table_b, table_d)
            for message_bytes, _data_category in messages:
                rows.extend(
                    _decode_gpsro_message(
                        decoder, message_bytes, wanted_descrs, dt_min, dt_max
                    )
                )
        return _finalize_rows(
            rows,
            modifiers,
            convert_pres_mb_to_pa=False,
        )


# ──────────────────────────────────────────────────────────────────────────
# Shared NCEP observation request orchestration: task types, the request
# lifecycle mixin (input handling, cycle-window planning, fetch, DataFrame
# assembly), consumed by the NNJA / NOMADS-GDAS / GFS observation sources.
# ──────────────────────────────────────────────────────────────────────────

_NCEPObsModifier = Callable[[pd.DataFrame], pd.DataFrame]
# Decode plan: variable id -> (route-specific decode key, post-decode modifier).
# The key is the raw string after the lexicon "route::" prefix; each adapter
# interprets it (PrepBUFR mnemonic, or GPS RO descriptor id parsed to int).
_NCEPPlan = Mapping[str, tuple[str, _NCEPObsModifier]]


@dataclass(frozen=True)
class _NCEPObsTask:
    """One NCEP observation cycle file to fetch and decode.

    Minimal base carrying only what the shared request lifecycle needs: the URI
    and the cycle's time window. Route/product-specific fields live on
    subclasses such as :class:`_NCEPConvTask`.
    """

    uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime


@dataclass(frozen=True)
class _NCEPConvTask(_NCEPObsTask):
    """A conventional (PrepBUFR / GPS RO) cycle file.

    ``route`` (``"prepbufr"`` / ``"gpsro"``) selects the decoder and URI layout;
    ``var_plan`` maps each requested variable to its route-specific decode key
    and post-decode modifier.
    """

    route: str
    var_plan: _NCEPPlan


def _plan_conv_tasks(
    windows: Mapping[datetime, tuple[datetime, datetime]],
    variable: list[str],
    lexicon: LexiconType,
    build_uri: Callable[[str, datetime], str],
) -> list[_NCEPObsTask]:
    """Partition variables by lexicon route prefix into per-cycle conv tasks.

    Parameters
    ----------
    windows : Mapping[datetime, tuple[datetime, datetime]]
        Cycle datetime -> (min, max) observation-time window, already merged
        across requested times (see ``NCEPObsRequestMixin._cycle_windows``).
    variable : list[str]
        Requested variable ids, routed by their ``"route::key"`` lexicon prefix.
    lexicon : LexiconType
        Source lexicon mapping a variable id to ``("route::key", modifier)``.
    build_uri : Callable[[str, datetime], str]
        Source hook returning the archive URI for a ``(route, cycle)`` pair.

    Returns
    -------
    list[_NCEPObsTask]
        One ``_NCEPConvTask`` (typed as the base) per ``(route, cycle)``.
        ``prepbufr`` uses the PrepBUFR decoder, ``gpsro`` the GPS RO BUFR
        decoder; the key after ``::`` is kept verbatim for the adapter.
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

    tasks: list[_NCEPObsTask] = []
    for cycle, (tmin, tmax) in windows.items():
        for route, plan in plans.items():
            tasks.append(
                _NCEPConvTask(
                    uri=build_uri(route, cycle),
                    datetime_file=cycle,
                    datetime_min=tmin,
                    datetime_max=tmax,
                    route=route,
                    var_plan=plan,
                )
            )
    return tasks


class NCEPObsRequestMixin:
    """Request-orchestration behavior for NCEP DataFrame observation sources.

    Provides the synchronous entry point, async fetch orchestration, six-hour
    cycle-window planning, output-schema resolution, and cached-file assembly.
    Compose with a transport mixin that supplies ``fetch_files``,
    ``local_path``, ``cleanup``, and ``_validate_time``, and a concrete source
    that supplies ``_create_tasks`` and ``_decode_file``. Conventional sources
    build ``_create_tasks`` from the shared :func:`plan_conv_tasks` helper;
    satellite sources implement their own per-sensor planner.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema

    def _init_request(
        self,
        *,
        time_tolerance: TimeTolerance,
        verbose: bool,
        async_timeout: int,
        decode_workers: int,
    ) -> None:
        """Initialize shared request state; call from the concrete constructor."""
        self._verbose = verbose
        self._decode_workers = max(1, decode_workers)
        self.async_timeout = async_timeout

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
            Variables defined by the concrete data source lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` returns all schema fields.

        Returns
        -------
        pd.DataFrame
            Observation DataFrame with columns matching the resolved schema.
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
        """Async function to get data."""
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self._resolve_output_schema(fields)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = list({task.uri for task in async_tasks})
        await self.fetch_files(file_uri_set)

        # Decoding is CPU-bound and internally blocks on pool futures; run it in
        # a worker thread so it does not stall the shared fsspec IO loop that
        # _sync_async dispatches every data source's fetch onto.
        df = await asyncio.to_thread(self._compile_dataframe, async_tasks, schema)
        df.attrs["source"] = self.SOURCE_ID
        return df

    def _compile_dataframe(
        self,
        async_tasks: Sequence[_NCEPObsTask],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        n_tasks = len(async_tasks)
        compile_t0 = time.perf_counter()
        for idx, task in enumerate(async_tasks, start=1):
            uri = task.uri
            local_path = self.local_path(uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {uri}, skipping")
                continue
            short_uri = uri.rsplit("/", 1)[-1]
            logger.info(f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} start: {short_uri}")
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
            return _empty_dataframe(self.SCHEMA)[schema.names]

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]

    def _cycle_windows(
        self, time_list: list[datetime]
    ) -> dict[datetime, tuple[datetime, datetime]]:
        """Map each unique 6-hour cycle to the union of requested time windows.

        For each ``t`` in ``time_list`` we cover all 6-hour cycles
        whose synoptic time falls within ``[t + tol_lower, t + tol_upper]``.
        Multiple input times that map to the same cycle are merged by
        taking the union of their windows so the cycle file is fetched
        once while keeping observations valid for any of them.
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

    @classmethod
    def _resolve_output_schema(
        cls, fields: str | list[str] | pa.Schema | None
    ) -> pa.Schema:
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

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        return cls._resolve_output_schema(fields)

    # Extension points supplied by the composed transport mixin and concrete
    # source. Keep the transport mixin ahead of this mixin in the MRO, e.g.
    # ``class Foo(SomeTransportMixin, NCEPObsRequestMixin)``, so its concrete
    # implementations override these fallbacks.
    async def fetch_files(self, uris: Sequence[str]) -> None:
        """Download the requested remote objects into the local cache."""
        raise NotImplementedError

    def local_path(self, uri: str) -> str:
        """Return the deterministic local cache path for a remote URI."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any temporary storage after a request."""
        raise NotImplementedError

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate requested times against product availability."""
        raise NotImplementedError

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_NCEPObsTask]:
        """Plan the per-file fetch/decode tasks for a request.

        Conventional sources delegate to :func:`plan_conv_tasks`; satellite
        sources implement their own per-sensor planner.
        """
        raise NotImplementedError

    def _decode_file(self, local_path: str, task: _NCEPObsTask) -> pd.DataFrame:
        """Decode one cached file into a DataFrame."""
        raise NotImplementedError
