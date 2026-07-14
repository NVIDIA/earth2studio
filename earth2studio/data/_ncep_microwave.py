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

"""Source-semantic decoder for NCEP aggregate microwave BUFR products."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils_bufr import (
    get_worker_decoder,
    init_decode_worker,
    parse_prepbufr_messages,
    silence_bufr_noise,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA

_C_CM_S = 2.99792458e10
_DECODE_BATCH_SIZE = 32

_SATELLITE_NAMES: dict[int, str] = {
    3: "metop-b",
    4: "metop-a",
    5: "metop-c",
    200: "n08",
    201: "n09",
    202: "n10",
    203: "n11",
    204: "n12",
    205: "n14",
    206: "n15",
    207: "n16",
    208: "n17",
    209: "n18",
    223: "n19",
    224: "npp",
    225: "n20",
    226: "n21",
}
_NCEP_MICROWAVE_SATELLITES = frozenset(_SATELLITE_NAMES.values())

# BUFR descriptors used by the NCEP aggregate microwave templates.
_SAID = 1007
_YEAR = 4001
_MONTH = 4002
_DAY = 4003
_HOUR = 4004
_MINUTE = 4005
_SECOND = 4006
_SCAN_LINE = 5041
_CHANNEL_NUMBER = 5042
_FOV_NUMBER = 5043
_LAT_HIGH = 5001
_LAT_COARSE = 5002
_LON_HIGH = 6001
_LON_COARSE = 6002
_SATELLITE_ZENITH = 7024
_SOLAR_ZENITH = 7025
_SURFACE_ELEVATION = 10001
_BEARING_OR_AZIMUTH = 5021
_SOLAR_AZIMUTH = 5022
_CHANNEL_FREQUENCY = 2153
_ANTENNA_TEMPERATURE = 12066
_BRIGHTNESS_TEMPERATURE = 12163
_CHANNEL_QUALITY = 33081

_SCALAR_DESCRIPTORS = {
    _SAID,
    _YEAR,
    _MONTH,
    _DAY,
    _HOUR,
    _MINUTE,
    _SECOND,
    _SCAN_LINE,
    _FOV_NUMBER,
    _LAT_HIGH,
    _LAT_COARSE,
    _LON_HIGH,
    _LON_COARSE,
    _SATELLITE_ZENITH,
    _SOLAR_ZENITH,
    _SURFACE_ELEVATION,
    _BEARING_OR_AZIMUTH,
    _SOLAR_AZIMUTH,
}

_CHANNEL_DESCRIPTORS = {
    _CHANNEL_FREQUENCY,
    _ANTENNA_TEMPERATURE,
    _BRIGHTNESS_TEMPERATURE,
    _CHANNEL_QUALITY,
}

_SOURCE_FIELD_DESCRIPTORS = {
    "TMANT": _ANTENNA_TEMPERATURE,
    "TMBR": _BRIGHTNESS_TEMPERATURE,
}


class _NCEPMicrowaveDecodeError(RuntimeError):
    def __init__(self, path: str, failed_messages: int, total_messages: int) -> None:
        self.context: dict[str, object] = {
            "path": path,
            "decoded_messages": total_messages - failed_messages,
            "failed_messages": failed_messages,
            "total_messages": total_messages,
        }
        super().__init__(f"Incomplete microwave BUFR decode: {self.context}")


_NCEP_MICROWAVE_PUBLIC_SCHEMA = pa.schema(
    [
        E2STUDIO_SCHEMA.field("time"),
        E2STUDIO_SCHEMA.field("class"),
        E2STUDIO_SCHEMA.field("lat"),
        E2STUDIO_SCHEMA.field("lon"),
        E2STUDIO_SCHEMA.field("elev"),
        pa.field(
            "scan_position",
            pa.uint16(),
            metadata={"description": "Encoded one-based field-of-view number"},
        ),
        pa.field("scan_line", pa.uint32(), nullable=True),
        E2STUDIO_SCHEMA.field("sensor_index"),
        E2STUDIO_SCHEMA.field("wavenumber"),
        E2STUDIO_SCHEMA.field("solza"),
        E2STUDIO_SCHEMA.field("solaza"),
        E2STUDIO_SCHEMA.field("satellite_za"),
        E2STUDIO_SCHEMA.field("satellite_aza"),
        pa.field(
            "quality",
            pa.uint16(),
            nullable=True,
            metadata={
                "description": (
                    "Encoded channel-quality flags; interpretation is sensor "
                    "and product specific"
                )
            },
        ),
        E2STUDIO_SCHEMA.field("satellite"),
        E2STUDIO_SCHEMA.field("observation"),
        E2STUDIO_SCHEMA.field("variable"),
    ]
)


def _as_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _as_optional_int(value: Any) -> int | None:
    number = _as_float(value)
    if not np.isfinite(number):
        return None
    return int(round(number))


def _observation_time(scalars: Mapping[int, Any]) -> np.datetime64 | None:
    required = (_YEAR, _MONTH, _DAY, _HOUR, _MINUTE, _SECOND)
    if any(descriptor not in scalars for descriptor in required):
        return None
    second = _as_float(scalars[_SECOND])
    if not np.isfinite(second) or second < 0.0 or second >= 61.0:
        return None
    try:
        minute = datetime(
            int(scalars[_YEAR]),
            int(scalars[_MONTH]),
            int(scalars[_DAY]),
            int(scalars[_HOUR]),
            int(scalars[_MINUTE]),
        )
    except (TypeError, ValueError, OverflowError):
        return None

    # Fractional SECO is source data; no per-FOV timing is synthesized.
    second_ns = int(np.rint(second * 1_000_000_000.0))
    return np.datetime64(minute, "ns") + np.timedelta64(second_ns, "ns")


def _decode_microwave_subset(
    descriptors: Sequence[Any],
    values: Sequence[Any],
    sensor: str,
    variable_fields: tuple[tuple[str, int], ...],
    datetime_min: datetime,
    datetime_max: datetime,
    satellites: frozenset[str] | None,
) -> list[dict[str, Any]]:
    """Decode one aggregate microwave subset directly to long rows."""
    scalars: dict[int, Any] = {}
    channels: dict[int, dict[int, Any]] = {}
    current_channel: int | None = None

    for descriptor, value in zip(descriptors, values):
        descriptor_id = int(descriptor.id)
        if descriptor_id == _CHANNEL_NUMBER:
            current_channel = _as_optional_int(value)
            if current_channel is not None:
                channels.setdefault(current_channel, {})
            continue
        if descriptor_id in _CHANNEL_DESCRIPTORS and current_channel is not None:
            channels[current_channel].setdefault(descriptor_id, value)
            continue
        if (
            descriptor_id in _SCALAR_DESCRIPTORS
            and value is not None
            and descriptor_id not in scalars
        ):
            scalars[descriptor_id] = value

    observation_time = _observation_time(scalars)
    satellite_id = _as_optional_int(scalars.get(_SAID))
    scan_position = _as_optional_int(scalars.get(_FOV_NUMBER))
    if (
        observation_time is None
        or satellite_id is None
        or scan_position is None
        or not channels
    ):
        return []
    if observation_time < np.datetime64(
        datetime_min, "ns"
    ) or observation_time > np.datetime64(datetime_max, "ns"):
        return []

    latitude = _as_float(scalars.get(_LAT_HIGH))
    longitude = _as_float(scalars.get(_LON_HIGH))
    if not np.isfinite(latitude) or not np.isfinite(longitude):
        latitude = _as_float(scalars.get(_LAT_COARSE))
        longitude = _as_float(scalars.get(_LON_COARSE))
    if (
        not np.isfinite(latitude)
        or latitude < -90.0
        or latitude > 90.0
        or not np.isfinite(longitude)
    ):
        return []

    satellite = _SATELLITE_NAMES.get(satellite_id, f"satellite-{satellite_id}")
    if satellites is not None and satellite not in satellites:
        return []

    scalar_values = {
        "time": observation_time,
        "class": "rad",
        "lat": latitude,
        "lon": longitude % 360.0,
        "elev": _as_float(scalars.get(_SURFACE_ELEVATION)),
        "scan_position": scan_position,
        "scan_line": _as_optional_int(scalars.get(_SCAN_LINE)),
        "solza": _as_float(scalars.get(_SOLAR_ZENITH)),
        "solaza": _as_float(scalars.get(_SOLAR_AZIMUTH)),
        "satellite_za": _as_float(scalars.get(_SATELLITE_ZENITH)),
        "satellite_aza": _as_float(scalars.get(_BEARING_OR_AZIMUTH)),
        "satellite": satellite,
    }

    rows: list[dict[str, Any]] = []
    # Dict insertion order is the encoded CHNM order.
    for channel_number in channels:
        channel = channels[channel_number]
        frequency = _as_float(channel.get(_CHANNEL_FREQUENCY))
        channel_values = {
            "sensor_index": channel_number,
            "wavenumber": frequency / _C_CM_S,
            "quality": _as_optional_int(channel.get(_CHANNEL_QUALITY)),
        }
        for variable, source_descriptor in variable_fields:
            observation = _as_float(channel.get(source_descriptor))
            if not np.isfinite(observation):
                continue
            rows.append(
                {
                    **scalar_values,
                    **channel_values,
                    "observation": observation,
                    "variable": variable,
                }
            )
    return rows


def _decode_message_batch(
    arguments: tuple[
        str,
        list[tuple[int, bytes]],
        tuple[tuple[str, int], ...],
        datetime,
        datetime,
        tuple[str, ...] | None,
    ],
) -> tuple[list[dict[str, Any]], int]:
    sensor, messages, variable_fields, datetime_min, datetime_max, satellite_names = (
        arguments
    )
    satellites = frozenset(satellite_names) if satellite_names is not None else None
    decoder = get_worker_decoder()
    if decoder is None:
        raise RuntimeError("BUFR decoder worker is not initialized")

    rows: list[dict[str, Any]] = []
    failures = 0
    with silence_bufr_noise():
        for _message_index, message_bytes in messages:
            try:
                message = decoder.process(message_bytes)
                template_data = message.template_data.value
                descriptors_all = template_data.decoded_descriptors_all_subsets
                values_all = template_data.decoded_values_all_subsets
            except Exception:
                failures += 1
                continue

            for descriptors, values in zip(descriptors_all, values_all):
                rows.extend(
                    _decode_microwave_subset(
                        descriptors,
                        values,
                        sensor,
                        variable_fields,
                        datetime_min,
                        datetime_max,
                        satellites,
                    )
                )
    return rows, failures


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pa.Table.from_pylist(rows, schema=_NCEP_MICROWAVE_PUBLIC_SCHEMA)

    def types_mapper(data_type: pa.DataType) -> pd.ArrowDtype | None:
        if pa.types.is_unsigned_integer(data_type):
            return pd.ArrowDtype(data_type)
        return None

    return table.to_pandas(types_mapper=types_mapper)


class _NCEPMicrowaveAdapter:
    """Decode NCEP aggregate microwave BUFR independently of its transport."""

    def __init__(self, decode_workers: int = 8) -> None:
        self.decode_workers = max(1, decode_workers)

    def decode_file(
        self,
        path: str,
        sensor: str,
        plan: Mapping[str, str],
        datetime_min: datetime,
        datetime_max: datetime,
        satellites: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Decode one local NCEP aggregate microwave BUFR file."""
        variable_fields = tuple(
            (variable, _SOURCE_FIELD_DESCRIPTORS[source_field])
            for variable, source_field in plan.items()
        )
        file_data = pathlib.Path(path).read_bytes()
        table_b, table_d, messages = parse_prepbufr_messages(file_data)
        if not table_b or not table_d:
            raise ValueError(f"Embedded NCEP BUFR tables are missing from {path}")
        indexed_messages = list(enumerate(message for message, _ in messages))
        batches = [
            indexed_messages[index : index + _DECODE_BATCH_SIZE]
            for index in range(0, len(indexed_messages), _DECODE_BATCH_SIZE)
        ]
        arguments = [
            (
                sensor,
                batch,
                variable_fields,
                datetime_min,
                datetime_max,
                satellites,
            )
            for batch in batches
        ]

        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        failures = 0
        if self.decode_workers > 1 and len(batches) > 1:
            with ProcessPoolExecutor(
                max_workers=min(self.decode_workers, len(batches)),
                initializer=init_decode_worker,
                initargs=(table_b, table_d),
            ) as pool:
                # Executor.map preserves input order, so batch assembly remains
                # in exact source-message order even when workers finish out of order.
                for batch_rows, batch_failures in pool.map(
                    _decode_message_batch, arguments
                ):
                    rows.extend(batch_rows)
                    failures += batch_failures
        else:
            init_decode_worker(table_b, table_d)
            for argument in arguments:
                batch_rows, batch_failures = _decode_message_batch(argument)
                rows.extend(batch_rows)
                failures += batch_failures

        if failures:
            raise _NCEPMicrowaveDecodeError(path, failures, len(messages))
        logger.debug(
            f"Decoded {len(rows):,} {sensor} channel rows in "
            f"{time.perf_counter() - started:.1f}s"
        )
        return _rows_to_dataframe(rows)
