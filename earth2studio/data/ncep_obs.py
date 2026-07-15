# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request orchestration for NCEP observation data sources."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Protocol, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import _sync_async, prep_data_inputs
from earth2studio.data.utils_ncep import GPSRO_BNDA, _empty_dataframe
from earth2studio.lexicon.base import LexiconType
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

_NCEPObsModifier = Callable[[pd.DataFrame], pd.DataFrame]
_NCEPPrepbufrPlan = Mapping[str, tuple[str, _NCEPObsModifier]]
_NCEPGpsroPlan = Mapping[str, tuple[int, _NCEPObsModifier]]


@dataclass(frozen=True)
class _NCEPObsTask:
    """Describe one route-specific NCEP observation file."""

    uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    route: Literal["prepbufr", "gpsro"]
    var_plan: _NCEPPrepbufrPlan | _NCEPGpsroPlan

    def __post_init__(self) -> None:
        key_type = str if self.route == "prepbufr" else int
        if any(not isinstance(key, key_type) for key, _ in self.var_plan.values()):
            raise TypeError(f"{self.route} plan contains an invalid source key")

    @property
    def prepbufr_plan(self) -> _NCEPPrepbufrPlan:
        if self.route != "prepbufr":
            raise ValueError(f"Task route is '{self.route}', not 'prepbufr'")
        return cast(_NCEPPrepbufrPlan, self.var_plan)

    @property
    def gpsro_plan(self) -> _NCEPGpsroPlan:
        if self.route != "gpsro":
            raise ValueError(f"Task route is '{self.route}', not 'gpsro'")
        return cast(_NCEPGpsroPlan, self.var_plan)


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
    order, and assembles the requested DataFrame. Subclasses retain ownership of
    product schemas, lexicons, route-specific URIs, and file decoding.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema
    LEXICON: LexiconType

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
        file_uri_set = list({task.uri for task in async_tasks})
        await self._store.fetch_files(file_uri_set)

        df = self._compile_dataframe(async_tasks, schema)
        df.attrs["source"] = self.SOURCE_ID
        return df

    def _compile_dataframe(
        self,
        async_tasks: list[_NCEPObsTask],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        n_tasks = len(async_tasks)
        compile_t0 = time.perf_counter()
        for idx, task in enumerate(async_tasks, start=1):
            uri = task.uri
            local_path = self._store.local_path(uri)
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

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_NCEPObsTask]:
        """Build tasks from prefixed or direct PrepBUFR lexicon keys."""
        prepbufr_plan: dict[str, tuple[str, _NCEPObsModifier]] = {}
        gpsro_plan: dict[str, tuple[int, _NCEPObsModifier]] = {}

        for variable_name in variable:
            source_key, modifier = self.LEXICON[variable_name]
            if source_key.startswith("gpsro::"):
                descriptor = source_key.removeprefix("gpsro::")
                try:
                    descriptor_id = int(descriptor)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid GPSRO source key '{source_key}' for "
                        f"variable '{variable_name}'"
                    ) from exc
                if descriptor_id != GPSRO_BNDA:
                    raise ValueError(f"Unsupported GPSRO descriptor {descriptor_id}")
                gpsro_plan[variable_name] = (descriptor_id, modifier)
            else:
                prepbufr_key = source_key.removeprefix("prepbufr::")
                if not prepbufr_key:
                    raise ValueError(
                        f"Invalid PrepBUFR source key '{source_key}' for "
                        f"variable '{variable_name}'"
                    )
                prepbufr_plan[variable_name] = (prepbufr_key, modifier)

        windows = self._cycle_windows(time_list) if prepbufr_plan or gpsro_plan else {}
        tasks: list[_NCEPObsTask] = []
        for cycle, (datetime_min, datetime_max) in windows.items():
            if prepbufr_plan:
                tasks.append(
                    _NCEPObsTask(
                        uri=self._build_prepbufr_uri(cycle),
                        datetime_file=cycle,
                        datetime_min=datetime_min,
                        datetime_max=datetime_max,
                        route="prepbufr",
                        var_plan=prepbufr_plan,
                    )
                )
            if gpsro_plan:
                tasks.append(
                    _NCEPObsTask(
                        uri=self._build_gpsro_uri(cycle),
                        datetime_file=cycle,
                        datetime_min=datetime_min,
                        datetime_max=datetime_max,
                        route="gpsro",
                        var_plan=gpsro_plan,
                    )
                )
        return tasks

    def _build_prepbufr_uri(self, cycle: datetime) -> str:
        raise NotImplementedError("Subclasses must implement _build_prepbufr_uri.")

    def _build_gpsro_uri(self, cycle: datetime) -> str:
        raise NotImplementedError("Subclasses must implement _build_gpsro_uri.")

    def _decode_file(self, local_path: str, task: _NCEPObsTask) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _decode_file.")

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
