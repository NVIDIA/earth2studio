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

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow as pa
from aiohttp import ClientResponseError
from fsspec.implementations.http import HTTPFileSystem

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon import IEM_ASOSLexicon
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

IEM_ASOS_ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_ASOS_STATIONS_ENDPOINT = (
    "https://mesonet.agron.iastate.edu/geojson/network.py?network=AZOS"
)


@dataclass
class _IEM_ASOSAsyncTask:
    datetime_min: datetime
    datetime_max: datetime
    remote_url: str
    local_path: str


class IEM_ASOS:
    """IEM archive of parsed ASOS, AWOS, and METAR station observations.

    This source uses the Iowa Environmental Mesonet bulk ASOS service. It
    requests only parsed observation fields and never requests or retains the
    unprocessed METAR report. IEM performs little quality control on this
    historical archive.

    Parameters
    ----------
    stations : str | list[str] | None, optional
        IEM station identifiers to query. None queries all stations allowed by
        the selected networks, by default None.
    networks : str | list[str] | None, optional
        IEM network identifiers, such as ``"IA_ASOS"``. None does not restrict
        the request by network, by default None.
    report_types : tuple[int, ...], optional
        IEM report types to include: 1 for HFMETAR, 3 for routine, and 4 for
        special reports, by default (3, 4).
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default np.timedelta64(10, "m").
    cache : bool, optional
        Cache parsed CSV responses in the local filesystem, by default True.
    verbose : bool, optional
        Show download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for an async fetch, by default 600.
    async_workers : int, optional
        Maximum number of scheduled async fetch tasks. Requests are serialized
        to comply with the IEM rate limit, by default 16.
    retries : int, optional
        Number of retries for transient download failures, by default 3.

    Warning
    -------
    Broad station or time selections can return large responses. IEM limits
    requests without explicit stations to 24 hours and applies an IP-based
    one-request-per-second throttle.

    Note
    ----
    Additional resources:

    - https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
    - https://mesonet.agron.iastate.edu/request/download.phtml

    Badges
    ------
    region:global dataclass:observation product:wind product:precip product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.IEM_ASOS"
    MIN_DATE = datetime(1900, 1, 1)
    MAX_REQUEST_SPAN = timedelta(hours=24)
    REQUEST_INTERVAL_SECONDS = 1.0
    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("elev"),
            E2STUDIO_SCHEMA.field("station"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
        ]
    )

    def __init__(
        self,
        stations: str | list[str] | None = None,
        networks: str | list[str] | None = None,
        report_types: tuple[int, ...] = (3, 4),
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ) -> None:
        self.stations = [stations] if isinstance(stations, str) else stations
        self.networks = [networks] if isinstance(networks, str) else networks
        invalid_report_types = set(report_types) - {1, 3, 4}
        if invalid_report_types:
            raise ValueError(
                f"Invalid IEM report types: {sorted(invalid_report_types)}. "
                "Valid report types are 1, 3, and 4."
            )
        if not report_types:
            raise ValueError("report_types must contain at least one report type")
        self.report_types = report_types

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: HTTPFileSystem | None = None
        self._request_lock: asyncio.Lock | None = None
        self._last_request_time: float | None = None

    async def _async_init(self) -> None:
        self.fs = HTTPFileSystem(asynchronous=True)
        self._request_lock = asyncio.Lock()

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Retrieve parsed IEM ASOS/AWOS observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return observations for in UTC.
        variable : str | list[str] | VariableArray
            Variables from :class:`IEM_ASOSLexicon` to return.
        fields : str | list[str] | pa.Schema | None, optional
            Output fields to include, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Parsed observations in Earth2Studio long-form schema.
        """
        try:
            frame = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)
        return frame

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Asynchronously retrieve parsed IEM ASOS/AWOS observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return observations for in UTC.
        variable : str | list[str] | VariableArray
            Variables from :class:`IEM_ASOSLexicon` to return.
        fields : str | list[str] | pa.Schema | None, optional
            Output fields to include, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Parsed observations in Earth2Studio long-form schema.
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        tasks = self._create_tasks(time_list, variable_list)

        if self.fs is None:
            raise ValueError("HTTP filesystem is not initialized")
        async with managed_session(self.fs):
            coros = [self.fetch_wrapper(task) for task in tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=120.0,
                desc="Fetching IEM ASOS data",
                verbose=(not self._verbose),
            )

        frame = self._compile_dataframe(tasks, time_list, variable_list, schema)
        frame.attrs["source"] = self.SOURCE_ID
        return frame

    def _create_tasks(
        self, time_list: list[datetime], variable_list: list[str]
    ) -> list[_IEM_ASOSAsyncTask]:
        remote_fields: list[str] = []
        for variable in variable_list:
            try:
                source_key, _ = IEM_ASOSLexicon[variable]  # type: ignore[misc]
            except KeyError as exc:
                raise KeyError(
                    f"Variable '{variable}' not found in IEM_ASOSLexicon. "
                    f"Available variables: {list(IEM_ASOSLexicon.VOCAB)}"
                ) from exc
            for remote_field in source_key.split("::"):
                if remote_field not in remote_fields:
                    remote_fields.append(remote_field)

        windows = sorted(
            (t + self._tolerance_lower, t + self._tolerance_upper) for t in time_list
        )
        merged: list[tuple[datetime, datetime]] = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        tasks: list[_IEM_ASOSAsyncTask] = []
        for window_start, window_end in merged:
            request_end = window_end + timedelta(minutes=1)
            chunk_start = window_start
            while chunk_start < request_end:
                chunk_end = min(chunk_start + self.MAX_REQUEST_SPAN, request_end)
                remote_url = self._build_url(chunk_start, chunk_end, remote_fields)
                digest = hashlib.sha256(remote_url.encode()).hexdigest()
                tasks.append(
                    _IEM_ASOSAsyncTask(
                        datetime_min=window_start,
                        datetime_max=window_end,
                        remote_url=remote_url,
                        local_path=os.path.join(self.cache, f"{digest}.csv"),
                    )
                )
                chunk_start = chunk_end
        return tasks

    async def fetch_wrapper(self, task: _IEM_ASOSAsyncTask) -> None:
        """Fetch and cache one parsed IEM CSV response.

        Parameters
        ----------
        task : _IEM_ASOSAsyncTask
            Bounded IEM request task.
        """
        if pathlib.Path(task.local_path).is_file():
            return
        payload = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=120.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        with open(task.local_path, "wb") as csv_file:
            csv_file.write(payload)

    async def fetch_array(self, task: _IEM_ASOSAsyncTask) -> bytes:
        """Download one parsed IEM CSV response.

        Parameters
        ----------
        task : _IEM_ASOSAsyncTask
            Bounded IEM request task.

        Returns
        -------
        bytes
            CSV payload containing parsed fields only.
        """
        if self.fs is None or self._request_lock is None:
            raise ValueError("HTTP filesystem is not initialized")

        async with self._request_lock:
            loop = asyncio.get_running_loop()
            if self._last_request_time is not None:
                delay = self.REQUEST_INTERVAL_SECONDS - (
                    loop.time() - self._last_request_time
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            try:
                payload = await self.fs._cat_file(task.remote_url)
            except ClientResponseError as exc:
                if exc.status not in {429, 503}:
                    raise
                raise ConnectionError(
                    f"IEM ASOS service returned transient HTTP {exc.status}"
                ) from exc
            self._last_request_time = loop.time()

        if payload.lstrip().startswith((b"ERROR", b"<!DOCTYPE", b"<html")):
            raise OSError("IEM ASOS service returned an error response")
        return payload

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        for time in times:
            if time < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {time} is earlier than "
                    f"IEM_ASOS.MIN_DATE ({cls.MIN_DATE.isoformat()})."
                )

    def _build_url(
        self, start: datetime, end: datetime, remote_fields: list[str]
    ) -> str:
        params: list[tuple[str, str]] = [
            *(("data", field) for field in remote_fields),
            *(("station", station) for station in self.stations or []),
            *(("network", network) for network in self.networks or []),
            ("sts", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("ets", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "yes"),
            ("elev", "yes"),
            ("missing", "empty"),
            ("trace", "0.0001"),
            *(("report_type", str(report_type)) for report_type in self.report_types),
        ]
        return f"{IEM_ASOS_ENDPOINT}?{urlencode(params)}"

    def _compile_dataframe(
        self,
        tasks: list[_IEM_ASOSAsyncTask],
        time_list: list[datetime],
        variable_list: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for task in tasks:
            if not pathlib.Path(task.local_path).is_file():
                continue
            frame = pd.read_csv(
                task.local_path,
                na_values=["M", "null"],
                keep_default_na=True,
                comment="#",
            )
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=schema.names)

        source = pd.concat(frames, ignore_index=True).drop_duplicates()
        required = {"station", "valid", "lon", "lat", "elevation"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(
                f"IEM response is missing required columns: {sorted(missing)}"
            )

        source["valid"] = pd.to_datetime(source["valid"], errors="coerce", utc=True)
        source["valid"] = source["valid"].dt.tz_localize(None)
        source["lon"] = (pd.to_numeric(source["lon"], errors="coerce") + 360.0) % 360.0

        requested_mask = pd.Series(False, index=source.index)
        for time in time_list:
            requested_mask |= source["valid"].between(
                time + self._tolerance_lower,
                time + self._tolerance_upper,
                inclusive="both",
            )
        source = source.loc[requested_mask]

        output_frames: list[pd.DataFrame] = []
        for variable in variable_list:
            _, modifier = IEM_ASOSLexicon[variable]  # type: ignore[misc]
            output = pd.DataFrame(
                {
                    "time": source["valid"],
                    "lat": pd.to_numeric(source["lat"], errors="coerce"),
                    "lon": source["lon"],
                    "elev": pd.to_numeric(source["elevation"], errors="coerce"),
                    "station": source["station"].astype("string"),
                    "observation": modifier(source),
                    "variable": variable,
                }
            )
            output_frames.append(output.dropna(subset=["time", "observation"]))

        if not output_frames:
            return pd.DataFrame(columns=schema.names)
        result = pd.concat(output_frames, ignore_index=True)
        for field in self.SCHEMA:
            if field.name in result and pa.types.is_floating(field.type):
                result[field.name] = result[field.name].astype(
                    field.type.to_pandas_dtype()
                )
        return result[schema.names]

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve requested output fields into a validated schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field names or schema to select. None selects the full schema.

        Returns
        -------
        pa.Schema
            Validated subset of :attr:`SCHEMA`.

        Raises
        ------
        KeyError
            If a field is not present in the source schema.
        TypeError
            If a supplied field has an incompatible type.
        """
        if fields is None:
            return cls.SCHEMA
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in IEM_ASOS.SCHEMA. "
                        f"Available fields: {cls.SCHEMA.names}"
                    )
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type}."
                    )
            return fields

        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in IEM_ASOS.SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @classmethod
    def get_station_metadata(cls) -> pd.DataFrame:
        """Fetch and cache the IEM ASOS-class station catalog.

        Returns
        -------
        pd.DataFrame
            Station metadata with columns ID, LAT, LON, ELEV, NAME, NETWORK,
            COUNTRY, and ONLINE.
        """
        cache_dir = os.path.join(datasource_cache_root(), "iem_asos")
        os.makedirs(cache_dir, exist_ok=True)
        stations_file = os.path.join(cache_dir, "azos.geojson")

        if not os.path.isfile(stations_file):
            fs = HTTPFileSystem()
            payload = fs.cat_file(IEM_ASOS_STATIONS_ENDPOINT)
            with open(stations_file, "wb") as geojson_file:
                geojson_file.write(payload)

        with open(stations_file, encoding="utf-8") as geojson_file:
            payload = json.load(geojson_file)

        records: list[dict[str, object]] = []
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            station_id = properties.get("sid") or feature.get("id")
            if station_id is None or len(coordinates) < 2:
                continue
            records.append(
                {
                    "ID": str(station_id),
                    "LAT": coordinates[1],
                    "LON": coordinates[0],
                    "ELEV": properties.get("elevation"),
                    "NAME": properties.get("sname"),
                    "NETWORK": properties.get("network"),
                    "COUNTRY": properties.get("country"),
                    "ONLINE": properties.get("online"),
                }
            )

        return pd.DataFrame.from_records(
            records,
            columns=[
                "ID",
                "LAT",
                "LON",
                "ELEV",
                "NAME",
                "NETWORK",
                "COUNTRY",
                "ONLINE",
            ],
        )

    @classmethod
    def get_stations_bbox(
        cls,
        lat_lon_bbox: tuple[float, float, float, float],
    ) -> list[str]:
        """Return IEM ASOS-class station IDs within a lat/lon bounding box.

        Parameters
        ----------
        lat_lon_bbox : tuple[float, float, float, float]
            Bounding box [lat_min, lon_min, lat_max, lon_max]. Both
            [-180, 180) and [0, 360) longitude conventions are accepted;
            [0, 360) is detected when lon_max >= 180.

        Returns
        -------
        list[str]
            IEM station IDs inside the bounding box, including its boundary.
        """
        lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
        frame = cls.get_station_metadata()
        frame["LAT"] = pd.to_numeric(frame["LAT"], errors="coerce")
        frame["LON"] = pd.to_numeric(frame["LON"], errors="coerce")
        frame = frame.dropna(subset=["LAT", "LON"])

        if lon_max >= 180:
            frame["LON"] = (frame["LON"] + 360.0) % 360.0
        frame = frame[(frame["LAT"] >= lat_min) & (frame["LAT"] <= lat_max)]
        frame = frame[(frame["LON"] >= lon_min) & (frame["LON"] <= lon_max)]

        return frame["ID"].tolist()

    @property
    def cache(self) -> str:
        """Return the local cache directory for parsed IEM CSV responses."""
        cache_location = os.path.join(datasource_cache_root(), "iem_asos")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_iem_asos_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether a timestamp is within the IEM archive range.

        Parameters
        ----------
        time : datetime | np.datetime64
            Timestamp to check.

        Returns
        -------
        bool
            True when the timestamp is not earlier than the archive start.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
