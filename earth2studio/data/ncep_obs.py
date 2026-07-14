# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request orchestration and decoding for NCEP observation sources."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import _sync_async, prep_data_inputs
from earth2studio.data.utils_bufr import (
    get_worker_decoder,
    init_decode_worker,
    parse_prepbufr_messages,
    silence_bufr_noise,
)
from earth2studio.data.utils_ncep import _empty_dataframe
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


class _NCEPObsStore(Protocol):
    """Raw-object transport contract for NCEP observation sources.

    Implementations fetch requested remote objects, map each URI to a stable
    local path, and clean up temporary storage. Product-specific concerns such
    as URI construction, time availability, decoding, and output completeness
    remain owned by the source class.
    """

    async def fetch_files(self, uris: Sequence[str]) -> None: ...

    def local_path(self, uri: str) -> str: ...

    def cleanup(self) -> None: ...


class _NCEPObsSourceBase:
    """Orchestrate requests shared by NCEP DataFrame observation sources.

    The base normalizes inputs, validates product times and fields, fetches raw
    files through an injected :class:`_NCEPObsStore`, decodes tasks in source
    order, and assembles the requested DataFrame. Subclasses retain ownership
    of product schemas, task creation, URI selection, and file decoding.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema

    def __init__(
        self,
        store: _NCEPObsStore,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        verbose: bool = True,
        async_timeout: int = 600,
        decode_workers: int = 8,
    ) -> None:
        self._store = store
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
            self._store.cleanup()

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
        file_uri_set = list({self._task_uri(task) for task in async_tasks})
        try:
            await self._store.fetch_files(file_uri_set)
        except Exception as exc:
            self._handle_fetch_failure(file_uri_set, exc)
            raise

        df = self._compile_dataframe(async_tasks, schema)
        df.attrs["source"] = self.SOURCE_ID
        return df

    def _compile_dataframe(
        self,
        async_tasks: list[Any],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        n_tasks = len(async_tasks)
        compile_t0 = time.perf_counter()
        for idx, task in enumerate(async_tasks, start=1):
            uri = self._task_uri(task)
            local_path = self._store.local_path(uri)
            if not pathlib.Path(local_path).is_file():
                self._handle_incomplete_task(
                    uri=uri,
                    task_index=idx,
                    task_count=n_tasks,
                    cause=FileNotFoundError(local_path),
                )
                continue
            short_uri = uri.rsplit("/", 1)[-1]
            logger.info(f"[{self.SOURCE_ID}] decode {idx}/{n_tasks} start: {short_uri}")
            t0 = time.perf_counter()
            try:
                df = self._decode_file(local_path, task)
            except Exception as exc:  # pragma: no cover - defensive
                self._handle_incomplete_task(
                    uri=uri,
                    task_index=idx,
                    task_count=n_tasks,
                    cause=exc,
                )
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

    def _handle_fetch_failure(self, uris: Sequence[str], cause: Exception) -> None:
        return None

    def _handle_incomplete_task(
        self,
        *,
        uri: str,
        task_index: int,
        task_count: int,
        cause: Exception,
    ) -> None:
        if isinstance(cause, FileNotFoundError):
            logger.warning(f"Cached file missing for {uri}, skipping")
        else:
            logger.error(f"Failed to decode {self._store.local_path(uri)}: {cause}")

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[Any]:
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    def _decode_file(self, local_path: str, task: Any) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _decode_file.")

    def _task_uri(self, task: Any) -> str:
        raise NotImplementedError("Subclasses must implement _task_uri.")

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
    def _validate_time(cls, times: list[datetime]) -> None:
        raise NotImplementedError("Subclasses must implement _validate_time.")

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


# NCEP aggregate microwave BUFR decoding.
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
