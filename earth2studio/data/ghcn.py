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
import hashlib
import io
import os
import pathlib
import shutil
import uuid
from abc import abstractmethod
from datetime import datetime

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon.ghcn import GHCNDailyLexicon, GHCNHourlyLexicon
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray


class _GHCNBase:
    """Internal base class shared by GHCNDaily and GHCNHourly.

    Subclasses must define the following class attributes:

    - ``SCHEMA`` : pa.Schema
    - ``SOURCE_ID`` : str
    - ``_CACHE_DIR`` : str  — subdirectory name under datasource_cache_root()
    - ``_SCHEMA_META_KEY`` : bytes  — metadata key used by column_map()
    """

    SCHEMA: pa.Schema
    SOURCE_ID: str
    _CACHE_DIR: str
    _SCHEMA_META_KEY: bytes

    def __init__(
        self,
        stations: list[str],
        time_tolerance: TimeTolerance,
        cache: bool,
        verbose: bool,
        async_timeout: int,
        async_workers: int,
        retries: int,
    ):
        self.stations = stations
        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()
        self._cache = cache
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self._verbose = verbose
        self.async_timeout = async_timeout

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
            String, list of strings or array of strings that refer to variables
            to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Data frame
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)
        return df

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), self._CACHE_DIR)
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_{self._CACHE_DIR}_{self._tmp_cache_hash}"
            )
        return cache_location

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
        """Build column name mapping from source names to schema names.

        Reads the ``_SCHEMA_META_KEY`` metadata on each schema field to discover
        the source column name.

        Returns
        -------
        dict[str, str]
            Mapping from source column names to schema field names
        """
        mapping = {}
        for field in cls.SCHEMA:
            if field.metadata and cls._SCHEMA_META_KEY in field.metadata:
                src_name = field.metadata[cls._SCHEMA_META_KEY].decode("utf-8")
                mapping[src_name] = field.name
        return mapping

    def _create_observation_dataframe(
        self, df: pd.DataFrame, variables: list[str], schema: pa.Schema
    ) -> pd.DataFrame:
        """Transform wide-format DataFrame to long format with observation/variable columns.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame with variable columns
        variables : list[str]
            List of variable names to include
        schema : pa.Schema
            Output schema defining column order

        Returns
        -------
        pd.DataFrame
            Long format DataFrame with observation and variable columns
        """
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

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if the given date time is available for this data source.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    @abstractmethod
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame: ...

    @classmethod
    @abstractmethod
    def _validate_time(cls, times: list[datetime]) -> None: ...

    @classmethod
    def get_station_metadata(cls) -> pd.DataFrame:
        """Fetch and cache the GHCN station metadata file from the GHCN-Daily S3 bucket.

        Both GHCNDaily and GHCNHourly use the same 11-character GHCN station ID
        format (e.g. ``USW00013874``), sourced from ``ghcnd-stations.txt``.

        Returns
        -------
        pd.DataFrame
            Station metadata with columns: ID, LAT, LON, ELEV, STATE, NAME, GSN,
            HCN, WMO
        """
        cache_dir = os.path.join(datasource_cache_root(), cls._CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        stations_file = os.path.join(cache_dir, "ghcnd-stations.txt")

        if not os.path.isfile(stations_file):
            fs = s3fs.S3FileSystem(anon=True)
            fs.get("s3://noaa-ghcn-pds/ghcnd-stations.txt", stations_file)

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
            Bounding box [lat_min, lon_min, lat_max, lon_max].
            Accepts both [-180, 180) and [0, 360) longitude conventions;
            [0, 360) is auto-detected when ``lon_max >= 180``.

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

        if lon_max >= 180:
            df["LON"] = (df["LON"] + 360) % 360
        df = df[(df["LAT"] >= lat_min) & (df["LAT"] <= lat_max)]
        df = df[(df["LON"] >= lon_min) & (df["LON"] <= lon_max)]

        return df["ID"].tolist()


class GHCNDaily(_GHCNBase):
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
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and missing data warnings, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 16
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential backoff,
        by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    To help get a list of possible station IDs, this class includes
    :py:meth:`GHCNDaily.get_stations_bbox` which accepts a lat-lon bounding box and will
    return known station IDs. For more information on the stations, users should
    consult the ``ghcnd-stations.txt`` which can be accessed with
    :py:meth:`GHCNDaily.get_station_metadata`.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
    - https://registry.opendata.aws/noaa-ghcn/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Southeast US, lat lon bounding box (lat min, lon min, lat max, lon max)
        stations = GHCNDaily.get_stations_bbox((30, -90, 36, -80))
        ds = GHCNDaily(stations, time_tolerance=timedelta(days=1))
        df = ds(datetime(2024, 1, 1), ["t2m_max", "tp"])

    Badges
    ------
    region:global dataclass:observation product:wind product:precip product:temp product:atmos product:solar product:insitu
    """

    SOURCE_ID = "earth2studio.data.ghcn"
    _CACHE_DIR = "ghcn"
    _SCHEMA_META_KEY = b"ghcn_name"
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
    _S3_BUCKET = "noaa-ghcn-pds"

    def __init__(
        self,
        stations: list[str],
        time_tolerance: TimeTolerance = np.timedelta64(0),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ):
        super().__init__(
            stations,
            time_tolerance,
            cache,
            verbose,
            async_timeout,
            async_workers,
            retries,
        )
        # Station metadata (lat, lon, elev) loaded lazily
        self._station_meta: pd.DataFrame | None = None
        self.fs: s3fs.S3FileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialization of filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={},
            asynchronous=True,
            skip_instance_cache=True,
        )

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
            ids) to return. Must be in the GHCNDaily lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            GHCN data frame
        """
        if self.fs is None:
            await self._async_init()

        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)
        schema = self.resolve_fields(fields)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Check variables are valid and collect required GHCN products
        products: set[str] = set()
        for v in variable:
            try:
                products.add(GHCNDailyLexicon[v][0])  # type: ignore[misc]
            except KeyError:
                raise KeyError(f"variable id {v} not found in GHCNDaily lexicon")

        async with managed_session(self.fs) as session:  # noqa: F841

            # Load station metadata (lat, lon, elev) if not cached.
            # Use asyncio.to_thread so the sync S3 download doesn't call
            # fsspec.asyn.sync() from within the already-running event loop.
            if self._station_meta is None:
                self._station_meta = await asyncio.to_thread(self.get_station_metadata)

            # Build unique (year, product) pairs needed. Tolerance windows can
            # span year boundaries, so enumerate every year in [tmin, tmax].
            year_product_pairs: set[tuple[int, str]] = set()
            for dt in time:
                tmin = dt + self._tolerance_lower
                tmax = dt + self._tolerance_upper
                for product in products:
                    for yr in range(tmin.year, tmax.year + 1):
                        year_product_pairs.add((yr, product))

            # Fetch all required parquet partitions in parallel
            pair_list = sorted(year_product_pairs)
            coros = [
                async_retry(
                    self._fetch_year_element,
                    year,
                    product,
                    retries=self._retries,
                    backoff=1.0,
                    task_timeout=60.0,
                    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
                )
                for year, product in pair_list
            ]

            partition_dfs = await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching NOAA GHCN data",
                verbose=(not self._verbose),
            )

            # Index partitions by (year, product)
            partition_map: dict[tuple[int, str], pd.DataFrame] = {
                (year, product): df
                for (year, product), df in zip(pair_list, partition_dfs)
            }

            # Build reverse map: product code -> E2Studio variable name
            product_to_var: dict[str, str] = {}
            for v in variable:
                product_to_var[GHCNDailyLexicon.VOCAB[v]] = v

            # Filter by station, time tolerance, apply unit conversion per product.
            # Collect per-variable DataFrames separately, then concat within each
            # variable before merging across variables to avoid column name collisions.
            station_set = set(self.stations)
            var_frames: dict[str, list[pd.DataFrame]] = {v: [] for v in variable}

            for dt in time:
                tmin = dt + self._tolerance_lower
                tmax = dt + self._tolerance_upper
                date_min = tmin.strftime("%Y%m%d")
                date_max = tmax.strftime("%Y%m%d")

                for product in products:
                    # Walk every year the tolerance window touches so cross-year
                    # windows (e.g. Dec 31 +/- 1 day) don't drop data.
                    for yr in range(tmin.year, tmax.year + 1):
                        df = partition_map.get((yr, product))
                        if df is None or df.empty:
                            continue

                        # Filter to requested stations
                        mask = df["ID"].isin(station_set)
                        # Filter by date range (DATE is string YYYYMMDD)
                        mask = (
                            mask & (df["DATE"] >= date_min) & (df["DATE"] <= date_max)
                        )
                        # Filter by quality flag: None/NaN means passed all QC
                        mask = mask & df["Q_FLAG"].isna()

                        df_window = df.loc[mask, ["ID", "DATE", "DATA_VALUE"]].copy()
                        if df_window.empty:
                            continue

                        # Apply unit conversion for this product's variable
                        var_name = product_to_var[product]
                        _, mod = GHCNDailyLexicon[var_name]  # type: ignore[misc]
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
            meta: pd.DataFrame = self._station_meta  # type: ignore[assignment]
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

        return result

    async def _fetch_year_element(self, year: int, element: str) -> pd.DataFrame:
        """Fetch a single year/element partition from the GHCN S3 bucket.

        Parameters
        ----------
        year : int
            Year partition to fetch
        element : str
            GHCN element code (e.g. TMAX, PRCP)

        Returns
        -------
        pd.DataFrame
            Parquet partition data with columns ID, DATE, DATA_VALUE, Q_FLAG
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        s3_path = f"{self._S3_BUCKET}/parquet/by_year/YEAR={year}/ELEMENT={element}/"
        # Hash the URL for cache file names
        cache_hash = hashlib.sha256(s3_path.encode()).hexdigest()
        parquet_path = os.path.join(self.cache, f"{cache_hash}.parquet")

        # Read from cached parquet if available
        if self._cache and os.path.isfile(parquet_path):
            df = await asyncio.to_thread(pd.read_parquet, parquet_path)
        else:
            try:
                # List parquet files in the partition directory using async fs
                files = await self.fs._ls(f"s3://{s3_path}", detail=False)
                # Read all parquet files using async byte reads
                frames: list[pd.DataFrame] = []
                for file_path in files:
                    if not file_path.endswith(".parquet"):
                        continue
                    data = await self.fs._cat_file(file_path)
                    table = pq.read_table(
                        io.BytesIO(data),
                        columns=["ID", "DATE", "DATA_VALUE", "Q_FLAG"],
                    )
                    frames.append(table.to_pandas())

                if not frames:
                    return pd.DataFrame()

                df = pd.concat(frames, ignore_index=True)
                # Cache locally
                await asyncio.to_thread(df.to_parquet, parquet_path, index=False)

            except (FileNotFoundError, OSError, pa.ArrowInvalid):
                if self._verbose:
                    logger.warning(
                        f"GHCN data not found for YEAR={year}, ELEMENT={element}"
                    )
                return pd.DataFrame()

        return df

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify date times are valid for this data source.

        Parameters
        ----------
        times : list[datetime]
            Date times to validate

        Raises
        ------
        ValueError
            If any time is before 1750-01-01
        """
        for time in times:
            if time < datetime(1750, 1, 1):
                raise ValueError(
                    f"Requested date time {time} needs to be after "
                    f"1750-01-01 for GHCNDaily data source"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available by verifying the partition exists on S3.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        if not super().available(time):
            return False

        # Check if the year partition exists on S3 (using TMAX as representative)
        s3_path = (
            f"s3://{cls._S3_BUCKET}/parquet/by_year/" f"YEAR={time.year}/ELEMENT=TMAX/"
        )
        try:
            fs = s3fs.S3FileSystem(anon=True)
            return fs.exists(s3_path)
        except OSError:
            return False

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


class GHCNHourly(_GHCNBase):
    """NOAA's Global Historical Climatology Network Hourly (GHCNh) is a global
    database of hourly surface observations that supersedes the Integrated Surface
    Database (ISD). It compiles observations from thousands of stations worldwide
    into a common data model with consistent CSV encoding.

    Parameters
    ----------
    stations : list[str]
        Station IDs in GHCN station format (11 characters), e.g. ``"USW00013874"``
        for Atlanta Hartsfield-Jackson. Use :py:meth:`GHCNHourly.get_stations_bbox`
        to discover IDs by geographic area.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric ± window) or a tuple (lower, upper) for asymmetric windows,
        by default np.timedelta64(10, 'm')
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and missing data warnings, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished
        successfully, by default 600
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 16
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential backoff,
        by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    To help get a list of possible station IDs, this class includes
    :py:meth:`GHCNHourly.get_stations_bbox` which accepts a lat-lon bounding box and
    will return known station IDs. For more information on the stations, users
    should consult the ``ghcnd-stations.txt`` file accessible via
    :py:meth:`GHCNHourly.get_station_metadata`.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
    - https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        # Atlanta Hartsfield-Jackson airport
        stations = GHCNHourly.get_stations_bbox((33, -85, 34, -84))
        ds = GHCNHourly(stations, time_tolerance=timedelta(hours=1))
        df = ds(datetime(2024, 1, 1, 12), ["t2m", "ws10m"])

    Badges
    ------
    region:global dataclass:observation product:wind product:precip product:temp
    product:insitu
    """

    SOURCE_ID = "earth2studio.data.ghcn_hourly"
    BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access"
    _CACHE_DIR = "ghcnh"
    _SCHEMA_META_KEY = b"ghcnh_name"

    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns"), metadata={"ghcnh_name": "DATE"}),
            pa.field("lat", pa.float32(), metadata={"ghcnh_name": "LATITUDE"}),
            pa.field("lon", pa.float32(), metadata={"ghcnh_name": "LONGITUDE"}),
            pa.field(
                "elev",
                pa.float32(),
                nullable=True,
                metadata={"ghcnh_name": "ELEVATION"},
            ),
            pa.field("station", pa.string(), metadata={"ghcnh_name": "STATION"}),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        stations: list[str],
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ):
        super().__init__(
            stations,
            time_tolerance,
            cache,
            verbose,
            async_timeout,
            async_workers,
            retries,
        )
        self.fs: fsspec.AbstractFileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialization of filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        # skip_instance_cache ensures each GHCNh instance owns its session.
        self.fs = fsspec.filesystem(
            "https", asynchronous=True, skip_instance_cache=True
        )

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
            String, list of strings or array of strings that refer to variables
            to return. Must be in the GHCNh lexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            GHCNh data frame
        """
        if self.fs is None:
            await self._async_init()

        time, variable = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        for v in variable:
            if v not in GHCNHourlyLexicon.VOCAB:
                raise KeyError(f"variable id {v} not found in GHCNHourly lexicon")

        async with managed_session(self.fs):
            # Build unique (station, year) pairs needed
            station_year_pairs = sorted(
                {(s, dt.year) for s in self.stations for dt in time}
            )

            coros = [
                async_retry(
                    self._fetch_station_year,
                    station_id,
                    year,
                    retries=self._retries,
                    backoff=1.0,
                    task_timeout=60.0,
                    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
                )
                for station_id, year in station_year_pairs
            ]

            station_year_dfs = await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching NOAA GHCNh data",
                verbose=(not self._verbose),
            )

        # Map results back to (station, year)
        partition_map: dict[tuple[str, int], pd.DataFrame] = {
            pair: df for pair, df in zip(station_year_pairs, station_year_dfs)
        }

        filtered_df = []
        for station in self.stations:
            for dt in time:
                tmin = dt + self._tolerance_lower
                tmax = dt + self._tolerance_upper

                df = partition_map.get((station, dt.year))
                if df is None or df.empty:
                    continue

                df_window = df[(df["DATE"] >= tmin) & (df["DATE"] <= tmax)]
                if not df_window.empty:
                    filtered_df.append(df_window)

        if len(filtered_df) == 0:
            return pd.DataFrame(columns=schema.names)

        df = pd.concat(filtered_df, ignore_index=True)

        if not df.empty:
            df = df.rename(columns=self.column_map())
            df["station"] = df["station"].astype(str)
            if "lon" in df.columns:
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
                df["lon"] = (df["lon"] + 360.0) % 360.0

        df = self._add_variables(df)

        result = self._create_observation_dataframe(df, variable, schema)
        result.attrs["source"] = self.SOURCE_ID

        return result

    # Columns read from each GHCNh parquet file
    _PARQUET_COLS = [
        "STATION",
        "DATE",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "temperature",
        "dew_point_temperature",
        "wind_speed",
        "wind_direction",
        "wind_gust",
        "precipitation",
        "sky_cover_layer_1",
        "sky_cover_layer_2",
        "sky_cover_layer_3",
        "sky_cover_layer_4",
    ]

    async def _fetch_station_year(self, station_id: str, year: int) -> pd.DataFrame:
        """Fetch and cache parquet for a single station and year from GHCNh.

        Parameters
        ----------
        station_id : str
            GHCNh station ID in GHCN format, e.g. ``USW00013874``
        year : int
            Year of data to fetch

        Returns
        -------
        pd.DataFrame
            Parsed parquet with a datetime DATE column, or empty DataFrame on 404
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        url = (
            f"{self.BASE_URL}/by-year/{year}/parquet/GHCNh_{station_id}_{year}.parquet"
        )
        cache_hash = hashlib.sha256(url.encode()).hexdigest()
        parquet_path = os.path.join(self.cache, f"{cache_hash}.parquet")

        if self._cache and os.path.isfile(parquet_path):
            df = await asyncio.to_thread(pd.read_parquet, parquet_path)
        else:
            try:
                data = await self.fs._cat_file(url)
                df = await asyncio.to_thread(
                    pd.read_parquet, io.BytesIO(data), columns=self._PARQUET_COLS
                )
                df["DATE"] = pd.to_datetime(df["DATE"])
                await asyncio.to_thread(df.to_parquet, parquet_path, index=False)
            except FileNotFoundError:
                if self._verbose:
                    logger.warning(
                        f"GHCNh: no data for station {station_id}, year {year}"
                    )
                return pd.DataFrame()
        return df

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify date times are valid for this data source.

        Parameters
        ----------
        times : list[datetime]
            Date times to validate

        Raises
        ------
        ValueError
            If any time is before 1901-01-01
        """
        for time in times:
            if time < datetime(1901, 1, 1):
                raise ValueError(
                    f"Requested date time {time} needs to be after "
                    f"1901-01-01 for GHCNh data source"
                )

    # ================
    # Variable computation
    # ================

    # Mapping from abbreviated sky cover codes to fractional cloud cover (0–1).
    # Mid-points of each okta range are used for FEW/SCT/BKN.
    _SKY_COVER_FRACTIONS: dict[str, float] = {
        "CLR": 0.0,
        "SKC": 0.0,
        "CAVOK": 0.0,
        "NSC": 0.0,
        "NCD": 0.0,
        "FEW": 0.1875,  # 1–2 oktas
        "SCT": 0.4375,  # 3–4 oktas
        "BKN": 0.75,  # 5–7 oktas
        "OVC": 1.0,  # 8 oktas
        "VV": 1.0,  # vertical visibility / sky obscured
        "OVX": 1.0,
    }

    def _add_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all earth2studio variable columns from GHCNh parquet columns.

        Simple variables are driven by the GHCNHourlyLexicon (column name +
        unit conversion). Derived variables (u10m, v10m, tcc) that require
        multiple source columns are handled below.
        """
        # Simple variables: column name and unit conversion from lexicon
        for var, (col, mod) in (
            (v, GHCNHourlyLexicon[v])  # type: ignore[misc]
            for v, src in GHCNHourlyLexicon.VOCAB.items()
            if src is not None
        ):
            src = df[col] if col in df.columns else np.nan
            df[var] = mod(pd.to_numeric(src, errors="coerce"))

        # u10m / v10m: meteorological convention, direction is where wind blows FROM
        wind_dir = pd.to_numeric(
            df["wind_direction"] if "wind_direction" in df.columns else np.nan,
            errors="coerce",
        ).where(lambda s: s != 999, np.nan)
        ws = df.get("ws10m", pd.Series(np.nan, index=df.index))
        rad = np.radians(wind_dir)
        valid = wind_dir.notna() & ws.notna()
        df["u10m"] = (-np.sin(rad) * ws).where(valid, np.nan)
        df["v10m"] = (-np.cos(rad) * ws).where(valid, np.nan)

        # tcc: maximum sky cover across layers
        tcc = pd.Series(np.nan, index=df.index, dtype=float)
        for col in [
            "sky_cover_layer_1",
            "sky_cover_layer_2",
            "sky_cover_layer_3",
            "sky_cover_layer_4",
        ]:
            if col not in df.columns:
                continue
            codes = df[col].astype(str).str.split(":", n=1, expand=True).iloc[:, 0]
            layer = codes.map(lambda x: self._SKY_COVER_FRACTIONS.get(x, np.nan))
            tcc = pd.concat([tcc, layer.astype(float)], axis=1).max(axis=1)
        df["tcc"] = tcc.where((tcc >= 0.0) & (tcc <= 1.0), np.nan)

        return df
