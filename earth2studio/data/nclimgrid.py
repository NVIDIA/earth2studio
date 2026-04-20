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

import asyncio
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
import xarray as xr
from loguru import logger
from tqdm.auto import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.nclimgrid import NClimGridLexicon
from earth2studio.utils.type import TimeArray, VariableArray


class NClimGrid:
    """Earth2Studio NClimGrid gridded datasource.

    Provides access to NOAA's NClimGrid daily gridded climate dataset from S3.
    Data is read from monthly NetCDF files and returned as a standardized DataFrame.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository:

    - https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00332
    - https://registry.opendata.aws/noaa-nclimgrid/
    """

    SOURCE_ID = "earth2studio.data.nclimgrid"
    NCLIMGRID_BUCKET_NAME = "noaa-nclimgrid-daily-pds"

    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("lat", pa.float32()),
            pa.field("lon", pa.float32()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of filesystem.

        Note
        ----
        Async fsspec expects initialization inside of the execution loop.
        """
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch NClimGrid data as a standardized DataFrame.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Canonical Earth2Studio variable name(s). Must be in the
            NClimGridLexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Output field subset, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            NClimGrid observation data.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        try:
            df = loop.run_until_complete(
                asyncio.wait_for(
                    self.fetch(time, variable, fields),
                    timeout=self.async_timeout,
                )
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
        """Async function to get NClimGrid data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Canonical Earth2Studio variable name(s). Must be in the
            NClimGridLexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Output field subset, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            NClimGrid observation data.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for var in variable_list:
            native, modifier = NClimGridLexicon.get_item(var)
            df_var = self._fetch_variable_dataframe(var, native, modifier, time_list)
            frames.append(df_var)

        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=self.SCHEMA.names)

        out.attrs["source"] = self.SOURCE_ID

        out = out[[f.name for f in schema]]
        return out

    def _monthly_nc_path(self, ts: pd.Timestamp) -> str:
        """Build S3 path for a monthly NetCDF file.

        Parameters
        ----------
        ts : pd.Timestamp
            Timestamp within the target month.

        Returns
        -------
        str
            S3 URI to monthly NetCDF file.
        """
        return (
            f"s3://{self.NCLIMGRID_BUCKET_NAME}/access/grids/"
            f"{ts.year}/ncdd-{ts.year}{ts.month:02d}-grd-scaled.nc"
        )

    def _dataarray_to_dataframe(
        self,
        da_t: xr.DataArray,
        var: str,
        modifier: Callable,
    ) -> pd.DataFrame:
        """Convert one timestep DataArray into a standardized DataFrame.

        Parameters
        ----------
        da_t : xr.DataArray
            Single-timestep data array from the NetCDF file.
        var : str
            Earth2Studio variable name.
        modifier : Callable
            Lexicon modifier function for unit conversion.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns matching SCHEMA.
        """
        da_t = da_t.load()

        values = modifier(da_t.values)
        values = np.asarray(values, dtype="float32")

        df = (
            xr.DataArray(
                values,
                coords=da_t.coords,
                dims=da_t.dims,
                name="observation",
            )
            .to_dataframe()
            .reset_index()
        )

        df["variable"] = var
        df = df[self.SCHEMA.names]

        df["lat"] = df["lat"].astype("float32")
        df["lon"] = df["lon"].astype("float32")
        df["observation"] = df["observation"].astype("float32")
        df["variable"] = df["variable"].astype("string")

        return df

    def _fetch_variable_dataframe(
        self,
        var: str,
        native: str,
        modifier: Callable,
        times: list[datetime],
    ) -> pd.DataFrame:
        """Fetch data for a single variable across the requested times.

        Parameters
        ----------
        var : str
            Earth2Studio variable name.
        native : str
            Native variable name in the NetCDF file.
        modifier : Callable
            Lexicon modifier function for unit conversion.
        times : list[datetime]
            Timestamps to fetch.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame for all timesteps.
        """
        selected_times = sorted({pd.Timestamp(t) for t in times})
        frames: list[pd.DataFrame] = []

        iterator: list[pd.Timestamp] | tqdm = selected_times
        if self._verbose and len(selected_times) > 1:
            iterator = tqdm(selected_times, desc=f"NClimGrid {native}", leave=False)

        current_month_ds: xr.Dataset | None = None
        current_month: tuple[int, int] | None = None

        for ts in iterator:
            ts = pd.Timestamp(ts)

            if current_month != (ts.year, ts.month):
                path = self._monthly_nc_path(ts)

                if self._verbose:
                    logger.info(f"Opening {path}")

                current_month_ds = xr.open_dataset(
                    path,
                    engine="h5netcdf",
                    storage_options={"anon": True},
                )
                current_month = (ts.year, ts.month)

            if current_month_ds is None:
                continue

            try:
                da_t = current_month_ds[native].sel(time=ts)
            except Exception as e:
                logger.debug(f"Skipping timestep {ts}: {e}")
                continue

            cache_file = self._cache_file_for_timestep(native, ts)

            if self._cache and cache_file.exists():
                df_t = pd.read_parquet(cache_file)
            else:
                df_t = self._dataarray_to_dataframe(da_t, var, modifier)

                if self._cache:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    df_t.to_parquet(cache_file)

            frames.append(df_t)

        if not frames:
            return pd.DataFrame(columns=self.SCHEMA.names)

        return pd.concat(frames, ignore_index=True)

    def _cache_file_for_timestep(self, native: str, ts: pd.Timestamp) -> pathlib.Path:
        """Get cache file path for a single native variable and timestep.

        Parameters
        ----------
        native : str
            Native variable name in the NetCDF file.
        ts : pd.Timestamp
            Timestamp for this observation.

        Returns
        -------
        pathlib.Path
            Path to the parquet cache file.
        """
        return pathlib.Path(self.cache) / (f"{native}_{ts.strftime('%Y%m%d')}.parquet")

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve requested output fields into a validated Arrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Fields to include in output. None returns all fields.

        Returns
        -------
        pa.Schema
            Validated schema for the requested fields.

        Raises
        ------
        KeyError
            If a requested field is not in the schema.
        TypeError
            If a pa.Schema field has a mismatched type.
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not in schema. "
                        f"Valid fields: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(field.name).type
                if field.type != expected:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected}"
                    )
            return fields

        selected = []
        for f in fields:
            if f not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{f}' not in schema. Valid fields: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(f))

        return pa.schema(selected)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location.

        Returns
        -------
        str
            Path to the cache directory.
        """
        cache_location = os.path.join(datasource_cache_root(), "nclimgrid")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nclimgrid_{self._tmp_cache_hash}"
            )
        return cache_location
