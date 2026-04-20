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

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
import xarray as xr
from tqdm.auto import tqdm

from earth2studio.lexicon.nclimgrid import NClimGridLexicon

logger = logging.getLogger("earth2studio.nclimgrid")
logger.setLevel(logging.INFO)


class NClimGrid:
    """
    Earth2Studio NClimGrid gridded datasource.

    Designed for large Zarr datasets on S3.

    Key features
    ------------
    - Canonical variable mapping via NClimGridLexicon
    - Strong schema validation
    - Input normalization for datetime, list-like, and slice time requests
    - Windowed parquet caching per variable × timestep
    - Chunk-stream DataFrame generation to avoid loading large time windows at once
    """

    SOURCE_ID = "earth2studio.data.nclimgrid"

    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("lat", pa.float32()),
            pa.field("lon", pa.float32()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    # -------------------------------------------------------
    # Schema utilities
    # -------------------------------------------------------

    @classmethod
    def resolve_fields(cls, fields: Any) -> pa.Schema:
        """
        Resolve requested output fields into a validated Arrow schema.
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
                        f"Field '{field.name}' has type {field.type}, expected {expected}"
                    )
            return fields

        selected = []
        for f in fields:
            if f not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{f}' not in schema. " f"Valid fields: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(f))

        return pa.schema(selected)

    # -------------------------------------------------------
    # Input normalization utilities
    # -------------------------------------------------------

    @staticmethod
    def _normalize_time(time: Any) -> Any:
        """
        Normalize time input into one of:
        - None
        - slice(pd.Timestamp, pd.Timestamp, step)
        - list[pd.Timestamp]
        """
        if time is None:
            return None

        if isinstance(time, slice):
            start = pd.to_datetime(time.start) if time.start is not None else None
            stop = pd.to_datetime(time.stop) if time.stop is not None else None
            return slice(start, stop, time.step)

        if isinstance(time, datetime):
            return [pd.Timestamp(time)]

        if isinstance(time, (list, tuple, np.ndarray, pd.DatetimeIndex, pd.Index)):
            return list(pd.to_datetime(time))

        raise TypeError(
            "Invalid time input. Expected None, datetime, slice, or list-like of datetimes."
        )

    @staticmethod
    def _normalize_variable(variable: Any) -> list[str]:
        """
        Normalize variable input into list[str].
        """
        if isinstance(variable, str):
            return [variable]

        if isinstance(variable, (list, tuple, np.ndarray, pd.Index)):
            return [str(v) for v in variable]

        raise TypeError("Invalid variable input. Expected str or list-like of strings.")

    @staticmethod
    def _resolve_requested_times(times: Any) -> list[pd.Timestamp]:
        """
        Resolve requested times from user input rather than from lazy dataset coordinates.

        This avoids forcing .values on a lazily selected DataArray just to enumerate times.
        """
        if times is None:
            raise ValueError(
                "Explicit time selection is required for NClimGrid streaming access."
            )

        if isinstance(times, slice):
            if times.start is None or times.stop is None:
                raise ValueError(
                    "Slice time selection must include both start and stop for streaming."
                )
            return list(pd.date_range(times.start, times.stop, freq="D"))

        # list-like path
        resolved = list(pd.to_datetime(times))
        # stable ordering, unique
        return sorted(pd.unique(pd.Index(resolved)))

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------

    def __init__(
        self,
        bucket: str = "noaa-nclimgrid-daily-pds",
        cache: bool = True,
        verbose: bool = False,
        cache_dir: str = "~/.earth2studio-cache/nclimgrid",
    ) -> None:
        self.bucket = bucket
        self.verbose = verbose
        self.cache = cache
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fs = s3fs.S3FileSystem(anon=True)
        self._call_cache: dict[Any, pd.DataFrame] = {}

    # -------------------------------------------------------
    # Constructor
    # -------------------------------------------------------

    def _monthly_nc_path(self, ts: Any) -> str:
        ts = pd.Timestamp(ts)
        return (
            f"s3://{self.bucket}/access/grids/"
            f"{ts.year}/ncdd-{ts.year}{ts.month:02d}-grd-scaled.nc"
        )

    # -------------------------------------------------------
    # Internal cache utilities
    # -------------------------------------------------------

    @staticmethod
    def _timestamp_cache_key(ts: Any) -> str:
        """
        Convert timestamp into stable YYYYMMDD cache key component.
        """
        return pd.Timestamp(ts).strftime("%Y%m%d")

    def _cache_file_for_timestep(self, native: str, ts: Any) -> Path:
        """
        One cache file per native variable per timestep.
        """
        return self.cache_dir / f"{native}_{self._timestamp_cache_key(ts)}.parquet"

    # -------------------------------------------------------
    # Internal dataframe construction
    # -------------------------------------------------------

    def _dataarray_to_dataframe(
        self,
        da_t: xr.DataArray,
        var: str,
        modifier: Any,
    ) -> pd.DataFrame:
        """
        Convert one timestep DataArray into standardized DataFrame.

        Uses explicit per-timestep materialization to keep memory bounded and
        avoid giant lazy graphs.
        """
        # Force only the current timestep into memory.
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

        # enforce schema column order
        df = df[self.SCHEMA.names]

        # enforce dtypes
        df["lat"] = df["lat"].astype("float32")
        df["lon"] = df["lon"].astype("float32")
        df["observation"] = df["observation"].astype("float32")
        df["variable"] = df["variable"].astype("string")

        return df

    def _fetch_variable_dataframe(
        self,
        var: str,
        native: str,
        modifier: Any,
        times: Any,
    ) -> pd.DataFrame:

        selected_times = self._resolve_requested_times(times)
        frames = []

        iterator = selected_times
        if getattr(self, "verbose", False) and len(selected_times) > 1:
            iterator = tqdm(iterator, desc=f"NClimGrid {native}", leave=False)

        current_month_ds = None
        current_month = None

        for ts in iterator:

            ts = pd.Timestamp(ts)

            # open new monthly file only when month changes
            if current_month != (ts.year, ts.month):

                path = self._monthly_nc_path(ts)

                if self.verbose:
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

            if self.cache and cache_file.exists():
                df_t = pd.read_parquet(cache_file)
            else:
                df_t = self._dataarray_to_dataframe(da_t, var, modifier)

                if self.cache:
                    df_t.to_parquet(cache_file)

            frames.append(df_t)

        if not frames:
            return pd.DataFrame(columns=self.SCHEMA.names)

        return pd.concat(frames, ignore_index=True)

    # -------------------------------------------------------
    # Core fetch logic
    # -------------------------------------------------------

    def __call__(
        self,
        time: Any = None,
        variable: Any = None,
        fields: Any = None,
    ) -> pd.DataFrame:
        """
        Fetch NClimGrid data as standardized DataFrame.

        Parameters
        ----------
        time : None | datetime | slice | list-like of datetime
            Requested timestep(s).
        variable : str | list[str]
            Canonical Earth2Studio variable name(s).
        fields : None | str | list[str] | pa.Schema
            Output field subset.

        Returns
        -------
        pd.DataFrame
        """
        if variable is None:
            raise ValueError("variable must be provided")
        if time is None:
            raise ValueError("time must be provided")

        times = self._normalize_time(time)
        variables = self._normalize_variable(variable)
        schema = self.resolve_fields(fields)

        times_key = tuple(self._resolve_requested_times(times))
        vars_key = tuple(sorted(variables))
        fields_key = tuple([f.name for f in schema])

        cache_key = (times_key, vars_key, fields_key)

        if self.cache and cache_key in self._call_cache:
            return self._call_cache[cache_key].copy()

        frames = []
        for var in variables:
            native, modifier = NClimGridLexicon.get_item(var)
            df_var = self._fetch_variable_dataframe(var, native, modifier, times)
            frames.append(df_var)

        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=self.SCHEMA.names)

        out.attrs["source"] = self.SOURCE_ID

        # field-aware output
        out = out[[f.name for f in schema]]
        if self.cache:
            self._call_cache[cache_key] = out
        return out
