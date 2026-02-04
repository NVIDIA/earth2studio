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
import concurrent.futures
import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import h5netcdf
import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import GSIConventionalLexicon, GSISatelliteLexicon
from earth2studio.utils.type import TimeArray, VariableArray


@dataclass
class _GSIAsyncTask:
    """Small helper struct for Async tasks"""

    datetime_file: datetime
    datetime_max: datetime
    datetime_min: datetime
    gsi_file_uri: str
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
        tolerance: timedelta | np.timedelta64 = np.timedelta64(0),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        self.obs_type = "ges"
        self._verbose = verbose
        self._cache = cache
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

        if isinstance(tolerance, np.timedelta64):
            self.tolerance = pd.to_timedelta(tolerance).to_pytimedelta()
        else:
            self.tolerance = tolerance

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem"""
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

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
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        df = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable, fields), timeout=self.async_timeout
            )
        )

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
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this "
                "function directly make sure the data source is initialized inside the async loop!"
            )

        session = await self.fs.set_session(refresh=True)

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = {task.gsi_file_uri for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(uri) for uri in file_uri_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching GSI files", disable=(not self._verbose)
        )

        if session:
            await session.close()

        df = self._compile_dataframe(async_tasks, variable_list, schema)

        return df

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        """Create async tasks for fetching data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    async def _fetch_remote_file(
        self,
        path: str,
        byte_offset: int = 0,
        byte_length: int | None = None,
    ) -> None:
        """Fetches remote file into cache.

        Parameters
        ----------
        path : str
            S3 URI to fetch
        byte_offset : int, optional
            Byte offset to start reading from, by default 0
        byte_length : int | None, optional
            Number of bytes to read, by default None (read all)
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self.cache_path(path, byte_offset, byte_length)
        if not pathlib.Path(cache_path).is_file():
            if byte_length:
                byte_length = int(byte_offset + byte_length)
            try:
                data = await self.fs._cat_file(path, start=byte_offset, end=byte_length)
                with open(cache_path, "wb") as file:
                    file.write(data)
            except FileNotFoundError:
                self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Handle missing file during fetch. Can be overridden by subclasses."""
        logger.error(f"File {path} not found")
        raise FileNotFoundError(f"File {path} not found")

    def _compile_dataframe(
        self,
        async_tasks: list[_GSIAsyncTask],
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Compile fetched data into a DataFrame."""
        frames: list[pd.DataFrame] = []
        for task in async_tasks:
            # Overwrite obs column name (needed for uv)
            column_map = self._build_column_map(schema)
            column_map[task.gsi_obs_name] = "observation"
            local_path = self.cache_path(task.gsi_file_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning("Cached file missing for {}", task.gsi_file_uri)
                continue
            try:
                with h5netcdf.File(local_path, "r") as ds:
                    data: dict[str, np.ndarray] = {}
                    for name, dset in ds.variables.items():
                        if name not in column_map:
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
    tolerance : timedelta | np.timedelta64, optional
        Time tolerance; observations within +/- tolerance of any requested time are
        returned, by default np.timedelta64(0).
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
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
        for v in variable:
            try:
                gsi_name, modifier = GSIConventionalLexicon[v]  # type: ignore
                gsi_platform, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
            except KeyError as e:
                logger.error(f"Variable id {v} not found in GSI conventional lexicon")
                raise e

            for t in time_list:
                tmin = t - self.tolerance
                tmax = t + self.tolerance
                day = tmin.replace(minute=0, second=0, microsecond=0)
                day = day.replace(hour=(day.hour // 6) * 6)
                while day <= tmax:
                    year_key = day.strftime("%Y")
                    month_key = day.strftime("%m")
                    datetime_key = day.strftime("%Y%m%d%H")
                    s3_uri = f"s3://{self.UFS_BUCKET}/{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_platform}_{gsi_sensor}_{gsi_product}.{datetime_key}_control.nc4"
                    tasks.append(
                        _GSIAsyncTask(
                            datetime_file=day,
                            datetime_min=tmin,
                            datetime_max=tmax,
                            gsi_file_uri=s3_uri,
                            gsi_modifier=modifier,
                            gsi_obs_name=gsi_name,
                            e2s_obs_name=v,
                        )
                    )
                    day = day + timedelta(hours=6)
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
    tolerance : timedelta | np.timedelta64, optional
        Time tolerance; observations within +/- tolerance of any requested time are
        returned, by default np.timedelta64(0).
    satellites : list[str], optional
        List of satellite platforms to include, by default includes all platforms.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent downloads, by default 24.
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
    """

    SOURCE_ID = "earth2studio.data.UFSObsSat"
    VALID_SATELLITES = frozenset(
        [
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
        tolerance: timedelta | np.timedelta64 = np.timedelta64(0),
        satellites: list[str] | None = None,
        max_workers: int = 24,
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
            tolerance=tolerance,
            max_workers=max_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_GSIAsyncTask]:
        tasks: list[_GSIAsyncTask] = []
        for v in variable:
            try:
                gsi_name, modifier = GSISatelliteLexicon[v]  # type: ignore
                gsi_platforms0, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
                gsi_platforms = [
                    p for p in gsi_platforms0.split(",") if p in self.satellites
                ]
            except KeyError as e:
                logger.error(f"Variable id {v} not found in GSI satellite lexicon")
                raise e

            for gsi_platform in gsi_platforms:
                for t in time_list:
                    tmin = t - self.tolerance
                    tmax = t + self.tolerance
                    day = tmin.replace(minute=0, second=0, microsecond=0)
                    day = day.replace(hour=(day.hour // 6) * 6)
                    while day <= tmax:
                        year_key = day.strftime("%Y")
                        month_key = day.strftime("%m")
                        datetime_key = day.strftime("%Y%m%d%H")
                        s3_uri = f"s3://{self.UFS_BUCKET}/{year_key}/{month_key}/{datetime_key}/gsi/diag_{gsi_sensor}_{gsi_platform}_{gsi_product}.{datetime_key}_control.nc4"
                        tasks.append(
                            _GSIAsyncTask(
                                datetime_file=day,
                                datetime_min=tmin,
                                datetime_max=tmax,
                                gsi_file_uri=s3_uri,
                                gsi_modifier=modifier,
                                gsi_obs_name=gsi_name,
                                e2s_obs_name=v,
                                satellite=gsi_platform,
                            )
                        )
                        day = day + timedelta(hours=6)
        return tasks

    def _handle_missing_file(self, path: str) -> None:
        """Satellite data may have missing platforms, just warn instead of error."""
        logger.warning(f"File {path} not found")

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
        # Channel index actually seems to be a pointer to sensor channels
        if name == "Channel_Index":
            sensor_chan = ds["sensor_chan"][:].astype(np.uint16)
            values = sensor_chan[values.astype(np.uint16) - 1]
        return values

    def _add_task_columns(self, df: pd.DataFrame, task: _GSIAsyncTask) -> None:
        """Add satellite column for satellite data."""
        df["satellite"] = task.satellite


if __name__ == "__main__":

    ds = UFSObsSat(satellites=["npp"], tolerance=timedelta(hours=6))
    df = ds(datetime(2024, 2, 1), ["atms"], ["lon", "variable"])
    print(df)
