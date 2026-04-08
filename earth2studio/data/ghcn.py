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
import concurrent.futures
import hashlib
import os
import pathlib
import shutil
import uuid
from datetime import datetime

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon.ghcn import GHCN_ELEMENT_MAP, GHCNLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


class GHCN:
    """NOAA's Global Historical Climatology Network Daily (GHCN-D) is a dataset that
    contains daily climate summaries from land surface stations across the globe.

    Parameters
    ----------
    stations : list[str]
        GHCN station IDs (11-character strings, e.g. ``"USW00023273"``) to
        attempt to fetch data from.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default np.timedelta64(0)
    max_workers : int, optional
        Maximum number of threads for async file operations, by default 24
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and missing data warnings, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    To help get a list of possible station IDs, this class includes
    :py:meth:`GHCN.get_stations_bbox` which accepts a lat-lon bounding box and will
    return known station IDs. For more information on the stations, users should
    consult the ``ghcnd-stations.txt`` which can be accessed with
    :py:meth:`GHCN.get_station_metadata`.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
    - https://registry.opendata.aws/noaa-ghcn/
    - https://docs.opendata.aws/noaa-ghcn-pds/readme.html

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Southeast US, lat lon bounding box (lat min, lon min, lat max, lon max)
        stations = GHCN.get_stations_bbox((30, -90, 36, -80))
        ds = GHCN(stations, time_tolerance=timedelta(days=1))
        df = ds(datetime(2024, 1, 1), ["t2m_max", "tp"])

    Badges
    ------
    region:global dataclass:observation product:precip product:temp product:insitu
    """

    SOURCE_ID = "earth2studio.data.ghcn"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"ghcn_name": "DATE"}),
            pa.field("lat", pa.float32(), metadata={"ghcn_name": "LAT"}),
            pa.field("lon", pa.float32(), metadata={"ghcn_name": "LON"}),
            pa.field(
                "elev", pa.float32(), nullable=True, metadata={"ghcn_name": "ELEV"}
            ),
            pa.field("station", pa.string(), metadata={"ghcn_name": "ID"}),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    # S3 bucket for GHCN public data
    _S3_BUCKET = "noaa-ghcn-pds"

    def __init__(
        self,
        stations: list[str],
        time_tolerance: TimeTolerance = np.timedelta64(0),
        max_workers: int = 24,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self.stations = stations
        # Normalize tolerance to (lower, upper) python timedelta bounds
        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()
        self._cache = cache
        self._max_workers = max_workers
        self._tmp_cache_hash: str | None = None
        self._verbose = verbose

        # Station metadata (lat, lon, elev) loaded lazily
        self._station_meta: pd.DataFrame | None = None

        # Check to see if there is a running loop (initialized in async)
        try:
            nest_asyncio.apply()  # Monkey patch asyncio to work in notebooks
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

        self.async_timeout = async_timeout

    async def _async_init(self) -> None:
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={},
            asynchronous=True,
            skip_instance_cache=True,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the GHCN lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            GHCN data frame
        """
        # Run async path synchronously
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

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables (column
            ids) to return. Must be in the GHCN lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            GHCN data frame
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this "
                "function directly make sure the data source is initialized inside the "
                "async loop!"
            )

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        session = await self.fs.set_session(refresh=True)

        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)
        schema = self.resolve_fields(fields)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Check variables are valid and collect required GHCN elements
        elements: set[str] = set()
        for v in variable:
            try:
                GHCNLexicon[v]
            except KeyError as e:
                logger.error(f"variable id {v} not found in GHCN lexicon")
                raise e
            elements.add(GHCN_ELEMENT_MAP[v])

        # Load station metadata (lat, lon, elev) if not cached
        if self._station_meta is None:
            self._station_meta = await asyncio.to_thread(self.get_station_metadata)

        # Build unique (year, element) pairs needed
        year_element_pairs: set[tuple[int, str]] = set()
        for dt in time:
            for element in elements:
                year_element_pairs.add((dt.year, element))

        # Fetch all required parquet partitions in parallel
        func_map = []
        pair_list = sorted(year_element_pairs)
        for year, element in pair_list:
            func_map.append(self._fetch_year_element(year, element))

        partition_dfs = await tqdm.gather(
            *func_map, desc="Fetching NOAA GHCN data", disable=(not self._verbose)
        )

        # Index partitions by (year, element)
        partition_map: dict[tuple[int, str], pd.DataFrame] = {
            (year, element): df for (year, element), df in zip(pair_list, partition_dfs)
        }

        # Build reverse map: element code -> E2Studio variable name
        element_to_var: dict[str, str] = {}
        for v in variable:
            element_to_var[GHCN_ELEMENT_MAP[v]] = v

        # Filter by station, time tolerance, apply unit conversion per element.
        # Collect per-variable DataFrames separately, then concat within each
        # variable before merging across variables to avoid column name collisions.
        station_set = set(self.stations)
        var_frames: dict[str, list[pd.DataFrame]] = {v: [] for v in variable}

        for dt in time:
            tmin = dt + self._tolerance_lower
            tmax = dt + self._tolerance_upper

            for element in elements:
                df = partition_map.get((dt.year, element))
                if df is None or df.empty:
                    continue

                # Filter to requested stations
                mask = df["ID"].isin(station_set)
                # Filter by date range (DATE is string YYYYMMDD)
                date_min = tmin.strftime("%Y%m%d")
                date_max = tmax.strftime("%Y%m%d")
                mask = mask & (df["DATE"] >= date_min) & (df["DATE"] <= date_max)
                # Filter by quality flag: None/NaN means passed all QC checks
                mask = mask & df["Q_FLAG"].isna()

                df_window = df.loc[mask, ["ID", "DATE", "DATA_VALUE"]].copy()
                if df_window.empty:
                    continue

                # Apply unit conversion for this element's variable
                var_name = element_to_var[element]
                _, mod = GHCNLexicon[var_name]
                df_window[var_name] = mod(
                    pd.to_numeric(df_window["DATA_VALUE"], errors="coerce")
                )
                df_window = df_window.drop(columns=["DATA_VALUE"])
                var_frames[var_name].append(df_window)

        # Concat all rows per variable, then merge across variables
        per_var_dfs: list[pd.DataFrame] = [
            pd.concat(var_frames[v], ignore_index=True)
            for v in variable
            if var_frames[v]
        ]

        if len(per_var_dfs) == 0:
            return pd.DataFrame(columns=schema.names)

        df = per_var_dfs[0]
        for extra in per_var_dfs[1:]:
            df = df.merge(extra, on=["ID", "DATE"], how="outer")

        # Convert DATE (string YYYYMMDD) to datetime
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")

        # Join station metadata (lat, lon, elev)
        meta = self._station_meta
        meta_subset = meta[meta["ID"].isin(station_set)][
            ["ID", "LAT", "LON", "ELEV"]
        ].drop_duplicates(subset="ID")
        df = df.merge(meta_subset, on="ID", how="left")

        # Rename columns using schema metadata
        df = df.rename(columns=self.column_map())
        df["station"] = df["station"].astype(str)

        # Normalize longitude from [-180, 180) to [0, 360)
        if "lon" in df.columns:
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["lon"] = (df["lon"] + 360.0) % 360.0

        # Transform to long format (one observation per row)
        result = self._create_observation_dataframe(df, variable, schema)
        result.attrs["source"] = self.SOURCE_ID

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return result

    def _create_observation_dataframe(
        self, df: pd.DataFrame, variables: list[str], schema: pa.Schema
    ) -> pd.DataFrame:
        # Metadata columns to keep (fields with ghcn_name metadata)
        id_vars = [field.name for field in schema if field.name in df.columns]
        value_vars = [v for v in variables if v in df.columns]

        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="variable",
            value_name="observation",
        )
        df_long = df_long.dropna(subset=["observation"]).reset_index(drop=True)
        return df_long[[name for name in schema.names]]

    async def _fetch_year_element(self, year: int, element: str) -> pd.DataFrame:
        if self.fs is None:
            raise ValueError("File system is not initialized")

        s3_path = (
            f"{self._S3_BUCKET}/parquet/by_year/" f"YEAR={year}/ELEMENT={element}/"
        )
        # Hash the URL for cache file names
        cache_hash = hashlib.sha256(s3_path.encode()).hexdigest()
        parquet_path = os.path.join(self.cache, f"{cache_hash}.parquet")

        # Read from cached parquet if available
        if self._cache and os.path.isfile(parquet_path):
            df = await asyncio.to_thread(pd.read_parquet, parquet_path)
        else:
            try:

                def _read_parquet() -> pd.DataFrame:
                    sync_fs = s3fs.S3FileSystem(anon=True)
                    files = sync_fs.ls(s3_path)
                    dataset = pq.ParquetDataset(files, filesystem=sync_fs)
                    table = dataset.read(columns=["ID", "DATE", "DATA_VALUE", "Q_FLAG"])
                    return table.to_pandas()

                df = await asyncio.to_thread(_read_parquet)
                # Cache locally
                await asyncio.to_thread(df.to_parquet, parquet_path, index=False)

            except (FileNotFoundError, OSError, pa.ArrowInvalid):
                if self._verbose:
                    logger.warning(
                        f"GHCN data not found for YEAR={year}, ELEMENT={element}"
                    )
                return pd.DataFrame()

        return df

    def _validate_time(self, times: list[datetime]) -> None:
        for time in times:
            if time < datetime(1750, 1, 1):
                raise ValueError(
                    f"Requested date time {time} needs to be after "
                    f"1750-01-01 for GHCN data source"
                )

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

    @classmethod
    def column_map(cls) -> dict[str, str]:
        """Build column name mapping from GHCN source names to schema names.

        Returns
        -------
        dict[str, str]
            Mapping from GHCN column names to schema field names
        """
        mapping = {}
        for field in cls.SCHEMA:
            if field.metadata and b"ghcn_name" in field.metadata:
                ghcn_name = field.metadata[b"ghcn_name"].decode("utf-8")
                mapping[ghcn_name] = field.name
        return mapping

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "ghcn")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_ghcn_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def get_station_metadata(cls) -> pd.DataFrame:
        """Fetch and cache the GHCN station metadata file from S3.

        Reads ``ghcnd-stations.txt`` which is a fixed-width format file containing
        station ID, latitude, longitude, elevation, state, and name.

        Returns
        -------
        pd.DataFrame
            Station metadata with columns: ID, LAT, LON, ELEV, STATE, NAME, GSN,
            HCN, WMO
        """
        cache_dir = os.path.join(datasource_cache_root(), "ghcn")
        os.makedirs(cache_dir, exist_ok=True)
        stations_file = os.path.join(cache_dir, "ghcnd-stations.txt")

        if not os.path.isfile(stations_file):
            fs = s3fs.S3FileSystem(anon=True)
            fs.get(f"s3://{cls._S3_BUCKET}/ghcnd-stations.txt", stations_file)

        return pd.read_fwf(
            stations_file,
            widths=[11, 9, 10, 7, 3, 31, 4, 4, 6],
            names=["ID", "LAT", "LON", "ELEV", "STATE", "NAME", "GSN", "HCN", "WMO"],
        )

    @classmethod
    def get_stations_bbox(
        cls,
        lat_lon_bbox: tuple[float, float, float, float],
    ) -> list[str]:
        """Return GHCN station IDs within a lat/lon bounding box.

        Parameters
        ----------
        lat_lon_bbox : tuple[float, float, float, float]
            Latitude/Longitude bounding box [lat_min, lon_min, lat_max, lon_max]
            (in cardinal directions [lat_south, lon_west, lat_north, lon_east])

        Returns
        -------
        list[str]
            List of 11-character GHCN station IDs
        """
        lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
        df = cls.get_station_metadata()

        df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
        df = df.dropna(subset=["LAT", "LON"])

        # Normalize longitudes from [-180, 180) to [0, 360) if needed
        if lon_max >= 180:
            df["LON"] = (df["LON"] + 360) % 360

        df = df[(df["LAT"] >= lat_min) & (df["LAT"] <= lat_max)]
        df = df[(df["LON"] >= lon_min) & (df["LON"] <= lon_max)]

        return df["ID"].tolist()

    @classmethod
    def get_station_inventory(cls) -> pd.DataFrame:
        """Fetch and cache the GHCN station inventory file from S3.

        The inventory file contains station/element/date-range metadata showing
        which elements are available for each station and over what time period.

        Returns
        -------
        pd.DataFrame
            Inventory with columns: ID, LAT, LON, ELEMENT, FIRSTYEAR, LASTYEAR
        """
        cache_dir = os.path.join(datasource_cache_root(), "ghcn")
        os.makedirs(cache_dir, exist_ok=True)
        inventory_file = os.path.join(cache_dir, "ghcnd-inventory.txt")

        if not os.path.isfile(inventory_file):
            fs = s3fs.S3FileSystem(anon=True)
            fs.get(f"s3://{cls._S3_BUCKET}/ghcnd-inventory.txt", inventory_file)

        return pd.read_fwf(
            inventory_file,
            widths=[11, 9, 10, 5, 5, 5],
            names=["ID", "LAT", "LON", "ELEMENT", "FIRSTYEAR", "LASTYEAR"],
        )
