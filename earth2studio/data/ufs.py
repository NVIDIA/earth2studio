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

import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import h5netcdf
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
)
from earth2studio.data.utils_ncep import cycle_windows
from earth2studio.lexicon import GSIConventionalLexicon, GSISatelliteLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


@dataclass
class _GSIAsyncTask:
    """Small helper struct for Async tasks"""

    datetime_file: datetime
    datetime_max: datetime
    datetime_min: datetime
    gsi_obs_key: str
    gsi_modifier: Callable
    gsi_obs_name: str
    e2s_obs_name: str
    satellite: str | None = None


class _UFSObsBase:
    """Base class for GSI data sources.

    This abstract base class provides common functionality for reading NOAA UFS
    GEFS-v13 replay observation data from S3.
    """

    UFS_BUCKET = "noaa-ufs-gefsv13replay-pds"
    SOURCE_ID: str  # To be defined by subclasses
    SCHEMA: pa.Schema  # To be defined by subclasses

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        max_workers: int = 24,
        cycle_aware: bool = True,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        self.obs_type = "ges"
        self._verbose = verbose
        self._cache = cache
        self._cycle_aware = cycle_aware
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        # Anonymous obstore S3 store for the public NOAA UFS replay bucket.
        self._store = obstore_store_from_url(
            f"s3://{self.UFS_BUCKET}",
            max_pool_connections=self._max_workers,
            region=self._region,
        )

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    # NOAA UFS GEFSv13 replay archive is a public bucket in us-east-1.
    _region = "us-east-1"

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
            DataFrame column names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

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
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_key_set = {task.gsi_obs_key for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(key) for key in file_key_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching GSI files", disable=(not self._verbose)
        )

        df = self._compile_dataframe(async_tasks, variable_list, schema)

        return df

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        """Create async tasks for fetching data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    async def _fetch_remote_file(
        self,
        key: str,
        byte_offset: int = 0,
        byte_length: int | None = None,
    ) -> None:
        """Fetches a remote object (by key within UFS_BUCKET) into cache.

        Parameters
        ----------
        key : str
            Object key within UFS_BUCKET to fetch
        byte_offset : int, optional
            Byte offset to start reading from, by default 0
        byte_length : int | None, optional
            Number of bytes to read, by default None (read all)
        """
        cache_path = self.cache_path(key, byte_offset, byte_length)
        try:
            # cache_key keeps the historical sha256(key + offset + length)
            # naming so warm caches remain valid
            await obstore_fetch_to_cache(
                self._store,
                key,
                self.cache,
                byte_offset=byte_offset,
                byte_length=byte_length,
                cache_key=os.path.basename(cache_path),
            )
        except FileNotFoundError:
            self._handle_missing_file(key)

    def _handle_missing_file(self, key: str) -> None:
        """Warn and skip a missing diag file. Archive gaps are expected
        (e.g. various satellite/GPS outages), so shouldn't fully derail
        a call to fetch data.

        Can be overridden by subclasses that require stricter handling.
        """
        uri = f"s3://{self.UFS_BUCKET}/{key}"
        logger.warning(f"File {uri} not found")

    def _compile_dataframe(
        self,
        async_tasks: list[_GSIAsyncTask],
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Compile fetched data into a DataFrame."""
        # Identify schema fields that are per-channel (need Channel_Index lookup)
        channel_indexed_fields: dict[str, str] = {}
        for field in schema:
            if (
                field.metadata
                and b"channel_indexed" in field.metadata
                and b"gsi_name" in field.metadata
            ):
                gsi_name = field.metadata[b"gsi_name"].decode("utf-8")
                channel_indexed_fields[gsi_name] = field.name

        frames: list[pd.DataFrame] = []
        for task in async_tasks:
            # Overwrite obs column name (needed for uv)
            column_map = self._build_column_map(schema)
            column_map[task.gsi_obs_name] = "observation"
            local_path = self.cache_path(task.gsi_obs_key)
            if not pathlib.Path(local_path).is_file():
                logger.warning(
                    "Cached file missing for {}",
                    f"s3://{self.UFS_BUCKET}/{task.gsi_obs_key}",
                )
                continue
            try:
                with h5netcdf.File(local_path, "r") as ds:
                    data: dict[str, np.ndarray] = {}
                    channel_index_raw: np.ndarray | None = None
                    for name, dset in ds.variables.items():
                        if name not in column_map:
                            continue
                        # Skip channel-indexed fields; they are expanded below
                        if name in channel_indexed_fields:
                            continue
                        values = np.asarray(dset[:])
                        pa_type = self.SCHEMA.field(column_map[name]).type
                        # Convert char arrays into strings for DF
                        if values.dtype.kind == "S" and values.ndim == 2:
                            values = values.view(f"S{values.shape[1]}").ravel()
                            values = np.char.rstrip(
                                np.char.decode(values, "utf-8"), "\x00"
                            )
                        # Apply subclass-specific transformations
                        values = self._transform_column(name, values, task, ds)
                        data[name] = pa.array(values, type=pa_type)
                        # Stash raw Channel_Index for per-channel expansion
                        if name == "Channel_Index":
                            channel_index_raw = np.asarray(dset[:])

                    # Expand channel-indexed fields using Channel_Index as lookup
                    if channel_index_raw is not None:
                        idx: np.ndarray = channel_index_raw.astype(np.int32) - 1
                        for gsi_name, field_name in channel_indexed_fields.items():
                            if gsi_name in ds.variables:
                                lut = np.asarray(
                                    ds[gsi_name][:],
                                    dtype=schema.field(
                                        field_name
                                    ).type.to_pandas_dtype(),
                                )
                                data[gsi_name] = pa.array(
                                    lut[idx], type=schema.field(field_name).type
                                )

                df = pd.DataFrame(data)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to read {}: {}", local_path, exc)
                raise exc

            # Rename columns
            df.rename(columns=column_map, inplace=True)
            # Add e2s columns
            df["variable"] = task.e2s_obs_name
            df.attrs["source"] = self.SOURCE_ID
            self._add_task_columns(df, task)

            mask = (df["time"] >= task.datetime_min) & (df["time"] <= task.datetime_max)
            df = df.loc[mask]
            frames.append(task.gsi_modifier(df))

        if not frames:
            logger.warning(
                "No observation files were available for this request; "
                "returning an empty DataFrame."
            )
            return schema.empty_table().to_pandas()

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build mapping from GSI column names to schema column names."""
        column_map = {}
        for field in schema:
            if field.metadata is None or b"gsi_name" not in field.metadata:
                continue
            column_map[field.metadata[b"gsi_name"].decode("utf-8")] = field.name
        # Always include time field for filtering
        time_field = self.SCHEMA.field("time")
        column_map[time_field.metadata[b"gsi_name"].decode("utf-8")] = time_field.name
        return column_map

    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _transform_column.")

    def _add_task_columns(self, df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add task-specific columns to DataFrame. Override in subclasses."""
        pass

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification. Can be:
            - None: Returns the full class SCHEMA
            - str: Single field name to select from SCHEMA
            - list[str]: List of field names to select from SCHEMA
            - pa.Schema: Validated against class SCHEMA for compatibility

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields

        Raises
        ------
        KeyError
            If a requested field name is not found in the class SCHEMA
        TypeError
            If a field type in the provided schema doesn't match the class SCHEMA
        ValueError
            If required fields are missing
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            # Validate provided schema against class schema
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in class SCHEMA. "
                        f"Available fields: {cls.SCHEMA.names}"
                    )
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from class SCHEMA"
                    )
            return fields

        # fields is list[str] - select fields from class schema
        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for GSI based on offline knowledge

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            start_date = datetime(1980, 1, 1)
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} needs to be after {start_date} for UFS observations"
                )

    def cache_path(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Gets local cache path given s3 uri

        Parameters
        ----------
        path : str
            s3 uri
        byte_offset : int, optional
            Byte offset of file to read, by default 0
        byte_length : int | None, optional
            Byte length of file to read, by default None

        Returns
        -------
        str
            Local path of cached file
        """
        if not byte_length:
            byte_length = -1
        sha = hashlib.sha256((path + str(byte_offset) + str(byte_length)).encode())
        filename = sha.hexdigest()
        return os.path.join(self.cache, filename)

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "gsi")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_gsi_{self._tmp_cache_hash}"
            )
        return cache_location


class UFSObsConv(_UFSObsBase):
    """NOAA UFS GEFS-v13 replay observations in-situ data

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
    cycle_aware : bool, optional
        Exclude future cycle files relative to the upper tolerance bound, by default True.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        ds = UFSObsConv(tolerance=timedelta(hours=2))
        df = ds(datetime(2024, 1, 1, 20), ["u"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.UFSObsConv"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Time"}),
            pa.field(
                "pres", pa.float32(), nullable=True, metadata={"gsi_name": "Pressure"}
            ),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Height"}
            ),
            pa.field(
                "type",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Observation_Type"},
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("station", pa.string(), metadata={"gsi_name": "Station_ID"}),
            pa.field(
                "station_elev",
                pa.float32(),
                nullable=True,
                metadata={"gsi_name": "Station_Elevation"},
            ),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        windows = cycle_windows(
            time_list,
            self._tolerance_lower,
            self._tolerance_upper,
            cycle_aware=self._cycle_aware,
        )
        for v in variable:
            try:
                gsi_name, modifier = GSIConventionalLexicon[v]  # type: ignore
                gsi_platform, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
            except KeyError:
                if v in GSISatelliteLexicon:
                    logger.warning(
                        f"Variable id {v} is a UFS satellite variable, skipping in conventional fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for day, (tmin, tmax) in windows.items():
                year_key = day.strftime("%Y")
                month_key = day.strftime("%m")
                datetime_key = day.strftime("%Y%m%d%H")
                obs_key = f"{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_platform}_{gsi_sensor}_{gsi_product}.{datetime_key}_control.nc4"
                tasks.append(
                    _GSIAsyncTask(
                        datetime_file=day,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        gsi_obs_key=obs_key,
                        gsi_modifier=modifier,
                        gsi_obs_name=gsi_name,
                        e2s_obs_name=v,
                    )
                )
        return tasks

    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for conventional data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Time":
            values = pd.to_timedelta(values, unit="h") + task.datetime_file
        # GSI stores Pressure in hPa (mb), convert to Pa
        elif name == "Pressure":
            values = values * 100.0
        return values

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build column map including elev field required for modifiers."""
        column_map = super()._build_column_map(schema)
        # Required for modifier filtering
        elev_field = self.SCHEMA.field("elev")
        column_map[elev_field.metadata[b"gsi_name"].decode("utf-8")] = elev_field.name
        return column_map


class UFSObsSat(_UFSObsBase):
    """NOAA UFS GEFS-v13 replay observations satellite data

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    satellites : list[str], optional
        List of satellite platforms to include, by default includes all platforms.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
    cycle_aware : bool, optional
        Exclude future cycle files relative to the upper tolerance bound, by default True.
    cache : bool, optional
        Cache data source in local filesystem cache, by default True.
    async_timeout : int, optional
        Time in seconds after which the async fetch will be cancelled if not finished,
        by default 600.
    verbose : bool, optional
        Log basic progress information, by default True.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional resources:

    - https://registry.opendata.aws/noaa-ufs-gefsv13replay-pds/
    - https://psl.noaa.gov/data/ufs_replay/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Use all possible satellites
        ds = UFSObsSat(tolerance=timedelta(hours=2))
        df = ds(datetime(2024, 1, 1, 20), ["atms"])

        # Use specific satellite
        ds = UFSObsSat(tolerance=timedelta(hours=2), satellites=["n20"])
        df = ds(datetime(2024, 1, 1, 20), ["atms"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.UFSObsSat"
    VALID_SATELLITES = frozenset(
        [
            "aqua",
            "npp",
            "metop-a",
            "metop-b",
            "metop-c",
            "n15",
            "n16",
            "n17",
            "n18",
            "n19",
            "n20",
        ]
    )
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"gsi_name": "Obs_Time"}),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"gsi_name": "Elevation"}
            ),
            pa.field(
                "class",
                pa.string(),
                nullable=True,
                metadata={"gsi_name": "Observation_Class"},
            ),
            pa.field("lat", pa.float32(), metadata={"gsi_name": "Latitude"}),
            pa.field("lon", pa.float32(), metadata={"gsi_name": "Longitude"}),
            pa.field("scan_angle", pa.float32(), metadata={"gsi_name": "Scan_Angle"}),
            pa.field(
                "channel_index",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "Channel_Index"},
            ),
            pa.field(
                "sensor_index",
                pa.uint16(),
                nullable=True,
                metadata={"gsi_name": "sensor_chan", "channel_indexed": "true"},
            ),
            pa.field(
                "wavenumber",
                pa.float64(),
                nullable=True,
                metadata={"gsi_name": "wavenumber", "channel_indexed": "true"},
            ),
            pa.field("solza", pa.float32(), metadata={"gsi_name": "Sol_Zenith_Angle"}),
            pa.field(
                "solaza", pa.float32(), metadata={"gsi_name": "Sol_Azimuth_Angle"}
            ),
            pa.field(
                "satellite_za", pa.float32(), metadata={"gsi_name": "Sat_Zenith_Angle"}
            ),
            pa.field(
                "satellite_aza",
                pa.float32(),
                metadata={"gsi_name": "Sat_Azimuth_Angle"},
            ),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        max_workers: int = 24,
        cycle_aware: bool = True,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid satellites are: {sorted(self.VALID_SATELLITES)}"
                )
        self.satellites = satellites
        super().__init__(
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            cycle_aware=cycle_aware,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        windows = cycle_windows(
            time_list,
            self._tolerance_lower,
            self._tolerance_upper,
            cycle_aware=self._cycle_aware,
        )
        for v in variable:
            try:
                gsi_name, modifier = GSISatelliteLexicon[v]  # type: ignore
                gsi_platforms0, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
                gsi_platforms = [
                    p for p in gsi_platforms0.split(",") if p in self.satellites
                ]
            except KeyError:
                if v in GSIConventionalLexicon:
                    logger.warning(
                        f"Variable id {v} is a UFS conventional variable, skipping in satellite fetch"
                    )
                    continue
                logger.error(f"Variable id {v} not found in GSI lexicon")
                raise

            for gsi_platform in gsi_platforms:
                for day, (tmin, tmax) in windows.items():
                    year_key = day.strftime("%Y")
                    month_key = day.strftime("%m")
                    datetime_key = day.strftime("%Y%m%d%H")
                    obs_key = f"{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_key}_control.nc4"
                    tasks.append(
                        _GSIAsyncTask(
                            datetime_file=day,
                            datetime_min=tmin,
                            datetime_max=tmax,
                            gsi_obs_key=obs_key,
                            gsi_modifier=modifier,
                            gsi_obs_name=gsi_name,
                            e2s_obs_name=v,
                            satellite=gsi_platform,
                        )
                    )
        return tasks

    def _build_column_map(self, schema: pa.Schema) -> dict[str, str]:
        """Build column map, always including Channel_Index for channel-indexed fields."""
        column_map = super()._build_column_map(schema)
        # Channel_Index is required to expand any channel-indexed fields
        for field in schema:
            if field.metadata and b"channel_indexed" in field.metadata:
                ci_field = self.SCHEMA.field("channel_index")
                ci_gsi = ci_field.metadata[b"gsi_name"].decode("utf-8")
                column_map[ci_gsi] = ci_field.name
                break
        return column_map

    def _transform_column(
        self,
        name: str,
        values: np.ndarray,
        task: _GSIAsyncTask,
        ds: h5netcdf.File,
    ) -> np.ndarray:
        """Transform column values for satellite data."""
        # Convert hours offset to timedelta, and add to datetime of file
        if name == "Obs_Time":
            values = pd.to_timedelta(values, unit="h") + task.datetime_file
        return values

    def _add_task_columns(self, df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add satellite column."""
        df["satellite"] = task.satellite
