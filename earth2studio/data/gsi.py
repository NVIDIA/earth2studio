# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import nest_asyncio
import numpy as np
import pandas as pd
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import GSIConventionalLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import h5netcdf
except ImportError:
    OptionalDependencyFailure("data")
    h5netcdf = None  # type: ignore[assignment]

SATELLITE_COLUMNS = {
    "Latitude": "lat",
    "Longitude": "lon",
    "Observation": "observation",
    "Channel_Index": "channel_index",
    "Obs_Time": "obs_time",
    "Sat_Zenith_Angle": "sat_zenith_angle",
    "Sol_Zenith_Angle": "sol_zenith_angle",
    "Scan_Angle": "scan_angle",
    "QC_Flag": "qc_flag",
    "Obs_Minus_Forecast_adjusted": "obs_minus_forecast_adjusted",
    "Obs_Minus_Forecast_unadjusted": "obs_minus_forecast_unadjusted",
}

CONV_METADATA_COLUMNS = {
    "Latitude": "lat",
    "Longitude": "lon",
    "Time": "obs_time",
    "Pressure": "pressure",
    "Height": "height",
    "Observation_Type": "observation_type",
    "Analysis_Use_Flag": "analysis_use_flag",
    "Obs_Minus_Forecast_adjusted": "obs_minus_forecast_adjusted",
    "Obs_Minus_Forecast_unadjusted": "obs_minus_forecast_unadjusted",
    "u_Obs_Minus_Forecast_adjusted": "u_obs_minus_forecast_adjusted",
    "u_Obs_Minus_Forecast_unadjusted": "u_obs_minus_forecast_unadjusted",
    "v_Obs_Minus_Forecast_adjusted": "v_obs_minus_forecast_adjusted",
    "v_Obs_Minus_Forecast_unadjusted": "v_obs_minus_forecast_unadjusted",
}


@dataclass
class GSIAsyncTask:
    """Small helper struct for Async tasks"""

    datetime_max: datetime
    datetime_min: datetime
    gsi_file_uri: str
    gsi_modifier: Callable
    gsi_obs_name: str
    e2s_obs_name: str


@check_optional_dependencies()
class GSI_Conventional:
    """NOAA UFS GEFS-v13 replay observations (diag_*_*.nc4).

    This data source reads raw UFS GSI diagnostic NetCDF files and returns a pandas
    DataFrame filtered to the requested timestamps. The expected file structure is
    the raw observation layout described in the UFS replay ETL scripts, e.g.:

    ``{data_dir}/{year}/**/gsi/diag_<sensor>_<platform>_<ges|anl>.<yyyymmddhh>_control.nc4``

    Parameters
    ----------
    sensors : str | list[str], optional
        Sensor names to include. Use "all" for every sensor. Conventional sensors can
        be specified as "conv" or as "conv_<platform>" (e.g., "conv_uv").
    obs_type : {"ges", "anl"}, optional
        Observation type to read ("ges" or "anl"), by default "ges".
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
    """

    UFS_BUCKET = "noaa-ufs-gefsv13replay-pds"

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
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            DataFrame column names to return.
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
            asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
        )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> pd.DataFrame:
        """Async function to get data."""
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this "
                "function directly make sure the data source is initialized inside the async loop!"
            )

        time_list, variable_list = prep_data_inputs(time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = {task.gsi_file_uri for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(uri) for uri in file_uri_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching GSI files", disable=(not self._verbose)
        )

        return self._compile_dataframe(async_tasks, ["u10m"])
        # tasks = [self._read_file(path) for path in file_paths]
        # frames = await tqdm.gather(
        #     *tasks, desc="Fetching UFS data", disable=(not self._verbose)
        # )

        # frames = [
        #     self._filter_by_time(df, time_list, self.tolerance)
        #     for df in frames
        #     if not df.empty
        # ]
        # frames = [df for df in frames if not df.empty]

        # if len(frames) == 0:
        #     return pd.DataFrame(columns=variable_list)

        # out = pd.concat(frames, ignore_index=True)
        # for col in variable_list:
        #     if col not in out.columns:
        #         out[col] = np.nan
        # return out.loc[:, variable_list]

    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[GSIAsyncTask]:

        for v in variable:
            try:
                gsi_name, modifier = GSIConventionalLexicon[v]  # type: ignore
                gsi_platform, gsi_sensor, gsi_product, gsi_name = gsi_name.split("::")
            except KeyError as e:
                logger.error(f"Variable id {v} not found in GSI conventional lexicon")
                raise e

            tasks: list[GSIAsyncTask] = []
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
                        GSIAsyncTask(
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

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetches remote file into cache"""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self.cache_path(path, byte_offset, byte_length)
        if not pathlib.Path(cache_path).is_file():
            if byte_length:
                byte_length = int(byte_offset + byte_length)
            try:
                data = await self.fs._cat_file(path, start=byte_offset, end=byte_length)
            except FileNotFoundError as e:
                logger.error(f"File {path} not found")
                raise e
            with open(cache_path, "wb") as file:
                file.write(data)

    def _compile_dataframe(
        self,
        async_tasks: list[GSIAsyncTask],
        variables: list[str],
    ) -> pd.DataFrame:

        column_map = GSIConventionalLexicon.column_map()
        frames: list[pd.DataFrame] = []
        for task in async_tasks:
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
                        # Convert char arrays into strings for DF
                        values = np.asarray(dset[:])
                        if values.dtype.kind == "S" and values.ndim == 2:
                            values = np.apply_along_axis(
                                lambda row: b"".join(row)
                                .decode("utf-8")
                                .rstrip("\x00"),
                                1,
                                values,
                            )
                        data[name] = values
                df = pd.DataFrame(data)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to read {}: {}", local_path, exc)
                raise exc

            # Rename column
            df.rename(columns=column_map, inplace=True)
            df["variable"] = task.e2s_obs_name

            print(df)

            time_values = pd.to_datetime(df["Time"], errors="coerce")
            mask = (time_values >= task.datetime_min) & (
                time_values <= task.datetime_max
            )
            df = df.loc[mask]
            if df.empty:
                continue

            for col in variables:
                if col not in df.columns:
                    df[col] = None
            frames.append(df.loc[:, variables])

        return frames
        # if len(frames) == 0:
        #     return pd.DataFrame(columns=variables)
        # return pd.concat(frames, ignore_index=True)

    def _read_satellite(self, ds: h5netcdf.File) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        for src, dst in SATELLITE_COLUMNS.items():
            if src in ds:
                data[dst] = np.asarray(ds[src][:])
        if "channel_index" in data and "sensor_chan" in ds:
            sensor_chan = np.asarray(ds["sensor_chan"][:])
            channel_idx = data["channel_index"].astype(np.int64) - 1
            valid = (channel_idx >= 0) & (channel_idx < sensor_chan.size)
            raw = np.full(channel_idx.shape, np.nan, dtype=np.float32)
            raw[valid] = sensor_chan[channel_idx[valid]].astype(np.float32)
            data["raw_channel_id"] = raw
        return pd.DataFrame(data)

    def _read_conventional(self, ds: h5netcdf.File) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}
        for src, dst in CONV_METADATA_COLUMNS.items():
            if src in ds:
                data[dst] = np.asarray(ds[src][:])

        if "lat" not in data or "lon" not in data:
            return pd.DataFrame()

        n = len(data["lat"])
        excluded = set(CONV_METADATA_COLUMNS.keys())
        excluded.add("sensor_chan")
        for name, dset in ds.items():
            if name in excluded:
                continue
            if not hasattr(dset, "shape") or len(dset.shape) != 1:
                continue
            if len(dset) != n:
                continue
            data[name.lower()] = np.asarray(dset[:])
        return pd.DataFrame(data)

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
        sha = hashlib.sha256((path + str(byte_offset) + str(byte_offset)).encode())
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


if __name__ == "__main__":

    ds = GSI_Conventional(tolerance=timedelta(hours=6))
    da = ds([datetime(2024, 1, 1, 3)], ["u10m"])
