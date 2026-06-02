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

import json
import os
import pathlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from netCDF4 import Dataset

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    prep_data_inputs,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.ibtracs import IBTrACSLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

# IBTrACS base URL
IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/netcdf"
)

# Valid region codes
VALID_REGIONS = frozenset(
    {
        "NA",
        "EP",
        "WP",
        "SP",
        "NI",
        "SI",
        "SA",
        "ALL",
        "since1980",
        "last3years",
        "ACTIVE",
    }
)

# Reference epoch for IBTrACS time variable (days since this date)
IBTRACS_TIME_EPOCH = datetime(1858, 11, 17)


@dataclass
class IBTrACSAsyncTask:
    """Async task for fetching a single IBTrACS region file."""

    region: str
    remote_url: str
    local_path: str


class IBTrACS:
    """International Best Track Archive for Climate Stewardship (IBTrACS) data source.

    IBTrACS provides global tropical cyclone best track data compiled from various
    regional agencies (NHC, JTWC, JMA, etc.) into a standardized format. This source
    returns track observations as a DataFrame with storm metadata.

    Parameters
    ----------
    region : str | list[str]
        IBTrACS region code(s) to fetch. Valid codes:
        - Basin codes: ``"NA"`` (North Atlantic), ``"EP"`` (Eastern Pacific),
          ``"WP"`` (Western Pacific), ``"SP"`` (South Pacific),
          ``"NI"`` (North Indian), ``"SI"`` (South Indian), ``"SA"`` (South Atlantic)
        - Combined datasets: ``"ALL"`` (all basins), ``"since1980"`` (satellite era),
          ``"last3years"`` (recent storms), ``"ACTIVE"`` (currently active)
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default np.timedelta64(0)
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and missing data warnings, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 4
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential backoff,
        by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests. The ``"ALL"`` region file is ~23 MB.

    Note
    ----
    IBTrACS files are updated frequently (especially ``"ACTIVE"``). This source
    checks the server's ``Last-Modified`` header and re-downloads when the remote
    file is newer than the cached version.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/international-best-track-archive
    - https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/doc/IBTrACS_version4_Technical_Details.pdf

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timedelta
        from earth2studio.data import IBTrACS

        # Fetch North Atlantic and Eastern Pacific storms
        ds = IBTrACS(region=["NA", "EP"], time_tolerance=timedelta(days=1))
        df = ds(datetime(2024, 9, 1), ["tcwnd", "mslp"])

        # Get active storms
        ds_active = IBTrACS(region="ACTIVE")
        df_active = ds_active(datetime.utcnow(), ["tcwnd", "mslp"])

    Badges
    ------
    region:global dataclass:observation product:wind product:atmos
    """

    SOURCE_ID = "earth2studio.data.ibtracs"
    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
            E2STUDIO_SCHEMA.field("track_id"),
            E2STUDIO_SCHEMA.field("storm_name"),
            E2STUDIO_SCHEMA.field("basin"),
            E2STUDIO_SCHEMA.field("season"),
        ]
    )

    def __init__(
        self,
        region: str | list[str] = "ALL",
        time_tolerance: TimeTolerance = np.timedelta64(0),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 4,
        retries: int = 3,
    ) -> None:
        # Normalize region to list
        if isinstance(region, str):
            region = [region]

        # Validate regions
        for r in region:
            if r not in VALID_REGIONS:
                raise ValueError(
                    f"Invalid region '{r}'. Must be one of: {sorted(VALID_REGIONS)}"
                )
        self.regions = region

        # Normalize tolerance to (lower, upper) python timedelta bounds
        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        # Async HTTP filesystem (initialized lazily)
        self.fs: HTTPFileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialization of HTTP filesystem."""
        self.fs = HTTPFileSystem(asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Retrieve tropical cyclone track observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names from the IBTrACS lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            IBTrACS track observations with columns matching the schema.
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
        """Async function to get tropical cyclone track data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names from the IBTrACS lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            IBTrACS track observations.
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in IBTrACSLexicon:
                raise KeyError(
                    f"Variable '{v}' not found in IBTrACS lexicon. "
                    f"Available: {list(IBTrACSLexicon.VOCAB.keys())}"
                )

        # Create tasks for each region file
        tasks = self._create_tasks()

        # Fetch region files with cache invalidation
        # HTTPFileSystem manages its own session lazily (no set_session needed)
        coros = [
            async_retry(
                self._fetch_region_file,
                task,
                retries=self._retries,
                backoff=1.0,
                task_timeout=120.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for task in tasks
        ]
        await gather_with_concurrency(
            coros,
            max_workers=self._async_workers,
            desc="Fetching IBTrACS data",
            verbose=(not self._verbose),
        )

        # Compile DataFrame from cached files
        df = self._compile_dataframe(tasks, time_list, variable_list, schema)
        df.attrs["source"] = self.SOURCE_ID

        return df

    def _create_tasks(self) -> list[IBTrACSAsyncTask]:
        """Create async tasks for fetching region files."""
        tasks = []
        for region in self.regions:
            filename = f"IBTrACS.{region}.v04r01.nc"
            remote_url = f"{IBTRACS_BASE_URL}/{filename}"
            local_path = os.path.join(self.cache, filename)
            tasks.append(
                IBTrACSAsyncTask(
                    region=region,
                    remote_url=remote_url,
                    local_path=local_path,
                )
            )
        return tasks

    async def _fetch_region_file(self, task: IBTrACSAsyncTask) -> None:
        """Fetch a region file with cache invalidation based on Last-Modified header.

        Parameters
        ----------
        task : IBTrACSAsyncTask
            Task containing region and URL info.
        """
        if self.fs is None:
            raise ValueError("Filesystem not initialized")

        meta_path = task.local_path + ".meta"

        # Check if we need to download
        need_download = True
        if self._cache and os.path.isfile(task.local_path):
            # Check Last-Modified header
            try:
                info = await self.fs._info(task.remote_url)
                server_mtime = info.get("Last-Modified") or info.get("last-modified")

                if server_mtime and os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        cached_meta = json.load(f)
                    if cached_meta.get("last_modified") == server_mtime:
                        need_download = False
                        if self._verbose:
                            logger.debug(f"Using cached {task.region} (up to date)")
            except (OSError, KeyError):
                # If HEAD request fails, just download
                pass

        if need_download:
            if self._verbose:
                logger.info(f"Downloading IBTrACS {task.region} region data...")

            # Download the file
            data = await self.fs._cat_file(task.remote_url)
            with open(task.local_path, "wb") as f:
                f.write(data)

            # Save metadata for cache invalidation
            if self._cache:
                try:
                    info = await self.fs._info(task.remote_url)
                    server_mtime = info.get("Last-Modified") or info.get(
                        "last-modified"
                    )
                    if server_mtime:
                        with open(meta_path, "w") as f:
                            json.dump({"last_modified": server_mtime}, f)
                except OSError:
                    pass

    def _compile_dataframe(
        self,
        tasks: list[IBTrACSAsyncTask],
        time_list: list[datetime],
        variable_list: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Compile observations from cached NetCDF files into a DataFrame.

        Parameters
        ----------
        tasks : list[IBTrACSAsyncTask]
            List of completed fetch tasks.
        time_list : list[datetime]
            Target times.
        variable_list : list[str]
            Variables to extract.
        schema : pa.Schema
            Output schema.

        Returns
        -------
        pd.DataFrame
            Compiled track observations.
        """
        # Calculate time bounds
        tmin = min(time_list) + self._tolerance_lower
        tmax = max(time_list) + self._tolerance_upper

        frames: list[pd.DataFrame] = []

        for task in tasks:
            if not os.path.isfile(task.local_path):
                if self._verbose:
                    logger.warning(f"Cached file missing for region {task.region}")
                continue

            # Read NetCDF and extract data
            df = self._read_netcdf_region(task.local_path, variable_list, tmin, tmax)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=schema.names)

        result = pd.concat(frames, ignore_index=True)

        # Return only requested fields
        return result[[name for name in schema.names if name in result.columns]]

    def _read_netcdf_region(
        self,
        filepath: str,
        variable_list: list[str],
        tmin: datetime,
        tmax: datetime,
    ) -> pd.DataFrame:
        """Read a single IBTrACS NetCDF file and extract track observations.

        Parameters
        ----------
        filepath : str
            Path to NetCDF file.
        variable_list : list[str]
            Variables to extract.
        tmin : datetime
            Minimum time bound.
        tmax : datetime
            Maximum time bound.

        Returns
        -------
        pd.DataFrame
            Track observations in long format.
        """
        with Dataset(filepath, "r") as ds:
            # Get dimensions
            n_storms = ds.dimensions["storm"].size
            n_times = ds.dimensions["date_time"].size

            # Read core variables
            # Time is in "days since 1858-11-17 00:00:00"
            time_var = ds.variables["time"][:]  # (storm, date_time)
            lat_var = ds.variables["lat"][:]  # (storm, date_time)
            lon_var = ds.variables["lon"][:]  # (storm, date_time)

            # Storm identifiers (need to decode from char arrays)
            sid_var = ds.variables["sid"][:]  # (storm, char)
            name_var = ds.variables["name"][:]  # (storm, char)
            basin_var = ds.variables["basin"][:]  # (storm, date_time, char) - can vary
            season_var = ds.variables["season"][:]  # (storm,)

            # Build records for each valid observation
            records: list[dict] = []

            for storm_idx in range(n_storms):
                # Decode storm ID and name
                track_id = self._decode_char_array(sid_var[storm_idx])
                storm_name = self._decode_char_array(name_var[storm_idx])
                season = int(season_var[storm_idx])

                for time_idx in range(n_times):
                    # Skip masked/missing time values
                    t = time_var[storm_idx, time_idx]
                    if np.ma.is_masked(t) or np.isnan(t):
                        continue

                    # Convert time from days since epoch
                    obs_time = IBTRACS_TIME_EPOCH + timedelta(days=float(t))

                    # Filter by time range
                    if obs_time < tmin or obs_time > tmax:
                        continue

                    # Get position
                    lat = lat_var[storm_idx, time_idx]
                    lon = lon_var[storm_idx, time_idx]

                    if np.ma.is_masked(lat) or np.ma.is_masked(lon):
                        continue

                    lat = float(lat)
                    lon = float(lon)

                    # Normalize longitude to [0, 360)
                    lon = (lon + 360.0) % 360.0

                    # Get basin (can vary per track point)
                    if basin_var.ndim == 3:
                        basin = self._decode_char_array(basin_var[storm_idx, time_idx])
                    else:
                        basin = self._decode_char_array(basin_var[storm_idx])

                    # Extract each variable
                    for var_name in variable_list:
                        source_key, modifier = IBTrACSLexicon[var_name]

                        value = self._extract_variable(
                            ds, source_key, storm_idx, time_idx
                        )

                        if value is None:
                            continue

                        # Apply unit conversion
                        converted = modifier(np.array([value]))
                        obs_value: float = float(converted[0])

                        records.append(
                            {
                                "time": obs_time,
                                "lat": lat,
                                "lon": lon,
                                "track_id": track_id,
                                "storm_name": storm_name,
                                "basin": basin,
                                "season": season,
                                "variable": var_name,
                                "observation": obs_value,
                            }
                        )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Convert dtypes
        df["time"] = pd.to_datetime(df["time"])
        df["lat"] = df["lat"].astype(np.float32)
        df["lon"] = df["lon"].astype(np.float32)
        df["observation"] = df["observation"].astype(np.float32)
        df["season"] = df["season"].astype(np.int32)

        return df

    def _extract_variable(
        self,
        ds: Dataset,
        source_key: str,
        storm_idx: int,
        time_idx: int,
    ) -> float | None:
        """Extract a single variable value from the NetCDF dataset.

        Parameters
        ----------
        ds : Dataset
            NetCDF dataset.
        source_key : str
            IBTrACS variable key (may contain :: for computed vars).
        storm_idx : int
            Storm index.
        time_idx : int
            Time index within storm track.

        Returns
        -------
        float | None
            Extracted value or None if missing.
        """
        # Handle computed variables (storm translation components)
        if source_key == "storm_speed::u":
            speed = ds.variables["storm_speed"][storm_idx, time_idx]
            direction = ds.variables["storm_dir"][storm_idx, time_idx]
            if np.ma.is_masked(speed) or np.ma.is_masked(direction):
                return None
            # Convert meteorological direction to u-component
            # Direction is where storm is heading, in degrees
            # u = speed * sin(dir), v = speed * cos(dir) for met convention
            speed_ms = float(speed) * IBTrACSLexicon._KTS_TO_MS
            dir_rad = np.deg2rad(float(direction))
            return speed_ms * np.sin(dir_rad)

        elif source_key == "storm_speed::v":
            speed = ds.variables["storm_speed"][storm_idx, time_idx]
            direction = ds.variables["storm_dir"][storm_idx, time_idx]
            if np.ma.is_masked(speed) or np.ma.is_masked(direction):
                return None
            speed_ms = float(speed) * IBTrACSLexicon._KTS_TO_MS
            dir_rad = np.deg2rad(float(direction))
            return speed_ms * np.cos(dir_rad)

        # Handle wind radii (average of quadrants)
        elif source_key in ("usa_r34", "usa_r50", "usa_r64"):
            var = ds.variables[source_key]
            # Shape is (storm, date_time, quadrant) where quadrant is 4
            vals = var[storm_idx, time_idx, :]
            # Check if all values are masked
            mask = np.ma.getmaskarray(vals)
            if mask.all():
                return None
            # Take mean of non-masked quadrants
            valid_vals = vals[~mask]
            if len(valid_vals) == 0:
                return None
            return float(np.mean(valid_vals))

        # Simple variables
        else:
            if source_key not in ds.variables:
                return None
            var = ds.variables[source_key]
            val = var[storm_idx, time_idx]
            if np.ma.is_masked(val):
                return None
            return float(val)

    @staticmethod
    def _decode_char_array(char_array: np.ndarray) -> str:
        """Decode a NetCDF character array to a Python string.

        Parameters
        ----------
        char_array : np.ndarray
            Character array (may be masked).

        Returns
        -------
        str
            Decoded string with trailing spaces removed.
        """
        if np.ma.is_masked(char_array):
            # Handle fully masked arrays
            if char_array.mask.all():
                return ""
            char_array = char_array.filled(b" ")

        # Handle bytes vs str
        if hasattr(char_array, "tobytes"):
            return char_array.tobytes().decode("utf-8", errors="ignore").strip()
        return "".join(str(c) for c in char_array).strip()

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate time inputs.

        Parameters
        ----------
        times : list[datetime]
            Times to validate.

        Raises
        ------
        ValueError
            If any time is before 1842 (IBTrACS start).
        """
        for t in times:
            if t < datetime(1842, 1, 1):
                raise ValueError(
                    f"Requested date time {t} needs to be after 1842-01-01 "
                    "for IBTrACS data source"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            If date time is available.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification.

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields.
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
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

        # fields is list[str]
        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "ibtracs")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_ibtracs_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def list_regions(cls) -> list[str]:
        """List available IBTrACS regions.

        Returns
        -------
        list[str]
            List of valid region codes.
        """
        return sorted(VALID_REGIONS)
