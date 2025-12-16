# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from dataclasses import dataclass
from datetime import datetime, timedelta

import nest_asyncio
import numpy as np
import pandas as pd
import s3fs
import xarray as xr  # noqa: F401  # kept in case of future extension; not currently used
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import ISDLexicon


@dataclass
class _StationData:
    station_id: str
    year: int
    dataframe: pd.DataFrame


class ISD:
    """NOAA's Integrated Surface Database (ISD) is a global database that consists of
    hourly and synoptic surface observations compiled from numerous sources into a
    common data model.

    Parameters
    ----------
    stations : list[str]
        Station IDs as the concatenation of USAF (6 chars) and WBAN (5 digits) to
        attempt to fetch data from.
    tolerance : timedelta | np.timedelta64, optional
        Time tolerance; nearest row within +/- tolerance is used per request, by default
        np.timedelta64(0)
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
    :py:meth:`ISD.get_stations_bbox` which accepts a lat-lon bounding box and will return
    known historical stations IDs. For more information on the stations, users should
    consult the `isd-history.csv` which can easily accessed with :py:meth:`ISD.get_station_history`

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
    - https://registry.opendata.aws/noaa-isd/
    - https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf

    Example
    -------
    .. highlight:: python
    .. code-block:: python
        # Bay area, lat lon bounding box (lat min, lon min, lat max, lon max)
        stations = ISD.get_stations_bbox((36, -124, 40, -120))
        ds = ISD(stations, tolerance=timedelta(hours=2))
        df = ds(datetime(2024, 1, 1, 20), ["station", "time", "lat", "lon", "t2m"])
    """

    def __init__(
        self,
        stations: list[str],
        tolerance: timedelta | np.timedelta64 = np.timedelta64(0),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        self.stations = stations
        # Normalize tolerance to python timedelta
        if isinstance(tolerance, np.timedelta64):
            self.tolerance = pd.to_timedelta(tolerance).to_pytimedelta()
        else:
            self.tolerance = tolerance
        self._cache = cache
        self._tmp_cache_hash: str | None = None
        self._verbose = verbose

        # Check to see if there is a running loop (initialized in async)
        try:
            nest_asyncio.apply()  # Monkey patch asyncio to work in notebooks
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

        self.async_timeout = async_timeout

    async def _async_init(self) -> None:
        """Async initialization of fs object

        Note
        ----
        Async fsspec expects initialization inside of the execution loop
        """
        self.fs = s3fs.S3FileSystem(anon=True, client_kwargs={}, asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | np.ndarray,
        variable: str | list[str] | np.ndarray,
    ) -> pd.DataFrame:
        """Function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the ISD lexicon.

        Returns
        -------
        pd.DataFrame
            ISD data frame
        """
        # Run async path synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        df = loop.run_until_complete(self.fetch(time, variable))

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | np.ndarray,
        variable: str | list[str] | np.ndarray,
    ) -> pd.DataFrame:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables (column
            ids) to return. Must be in the ISD lexicon.

        Returns
        -------
        pd.DataFrame
            ISD data frame
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this \
            function directly make sure the data source is initialized inside the async \
            loop!"
            )

        # https://filesystem-spec.readthedocs.io/en/latest/async.html#using-from-async
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session()
        else:
            session = None

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Check variables are valid
        for v in variable:
            try:
                ISDLexicon[v]
            except KeyError as e:
                logger.error(f"variable id {v} not found in ISD lexicon")
                raise e

        # Load dataframes for each station-year (cached parquet if available)
        func_map: list[asyncio.Task[_StationData]] = []
        for station in self.stations:
            for dt in time:
                func_map.append(  # noqa: PERF401
                    self._fetch_station_year(station, dt.year)
                )

        # Launch all fetch requests
        station_year_dfs = await tqdm.gather(
            *func_map, desc="Fetching NOAA ISD data", disable=(not self._verbose)
        )

        # Gather all dataframes by station and by year, keeping only those with DATE within requested time ± tolerance
        filtered_df = []
        index = 0
        for station in self.stations:
            for dt in time:
                df = station_year_dfs[index]
                index += 1

                tmin = dt - self.tolerance
                tmax = dt + self.tolerance

                if df.empty:
                    continue

                df_window = df[(df["DATE"] >= tmin) & (df["DATE"] <= tmax)]
                if not df_window.empty:
                    filtered_df.append(df_window)

        if len(filtered_df) == 0:
            return pd.DataFrame(columns=variable)

        df = pd.concat(filtered_df, ignore_index=True)

        # Standardize common metadata columns to lower case and normalize longitude
        if not df.empty:
            df = df.rename(
                columns={
                    "STATION": "station",
                    "DATE": "time",
                    "SOURCE": "source",
                    "LATITUDE": "lat",
                    "LONGITUDE": "lon",
                    "ELEVATION": "elev",
                }
            )
            df["station"] = df["station"].astype(str)
            # Normalize longitude from [-180, 180) to [0, 360)
            if "lon" in df.columns:
                df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
                df["lon"] = (df["lon"] + 360.0) % 360.0

        # Process columns
        df = self._extract_ws10m(df)
        df = self._extract_uv(df)
        df = self._extract_tp(df)
        df = self._extract_t2m(df)
        df = self._extract_fg10m(df)
        df = self._extract_d2m(df)
        df = self._extract_tcc(df)

        df = df.loc[:, variable]

        # Close aiohttp client if s3fs
        if session:
            await session.close()

        return df

    async def _fetch_station_year(self, station_id: str, year: int) -> pd.DataFrame:
        """Async method for fetching csv to given station

        Parameters
        ----------
        station_id : str
            ISD station ID
        year : int
            Year of the requested time

        Returns
        -------
        pd.DataFrame
            Pandas dataframe of CSV
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        s3_url = f"s3://noaa-global-hourly-pds/{year}/{station_id}.csv"
        # Hash the URL for cache file names
        cache_hash = hashlib.sha256(s3_url.encode()).hexdigest()
        parquet_path = os.path.join(self.cache, f"{cache_hash}.parquet")

        # Read from cached parquet if available
        if self._cache and os.path.isfile(parquet_path):
            df = await asyncio.to_thread(pd.read_parquet, parquet_path)
        else:
            # Download CSV via s3fs to cache, then read with pandas
            try:
                # file_butter = await self.fs._open(s3_url)
                async with await self.fs.open_async(s3_url, "rb") as f:
                    df = await asyncio.to_thread(
                        pd.read_csv,
                        io.BytesIO(await f.read()),
                        parse_dates=["DATE"],
                        low_memory=False,  # Mixed types
                    )
                    await asyncio.to_thread(df.to_parquet, parquet_path, index=False)
            except FileNotFoundError:
                # If that station does not have data for this year, return empty
                if self._verbose:
                    logger.warning(
                        f"Station {station_id} does not have any data for requested year {year}"
                    )
                return pd.DataFrame()

        return df

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "isd")
        if not self._cache:
            if self._tmp_cache_hash is None:
                # First access for temp cache: create a random suffix to avoid collisions
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_isd_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def get_station_history(
        cls,
    ) -> pd.DataFrame:
        """Fetches and caches the ISD station history data frame from S3. Useful for
        getting additional information on a specific station

        Returns
        -------
        pd.DataFrame
            Raw data frame of downloaded csv file
        """
        cache_dir = os.path.join(datasource_cache_root(), "isd")
        os.makedirs(cache_dir, exist_ok=True)
        catalog_csv = os.path.join(cache_dir, "isd-history.csv")

        if not os.path.isfile(catalog_csv):
            fs = s3fs.S3FileSystem(anon=True)
            fs.get("s3://noaa-isd-pds/isd-history.csv", catalog_csv)
        return pd.read_csv(catalog_csv)

    @classmethod
    def get_stations_bbox(
        cls,
        lat_lon_bbox: tuple[float, float, float, float],
    ) -> list[str]:
        """Return ISD station IDs within a lat/lon box. Station IDs are the combination
        of the USAF and WBAN IDs.

        Parameters
        ----------
        lat_lon_bbox : tuple[float, float, float, float]
            Latitude/Longitude bounding box to get stations [lat_min, lon_min, lat_max, lon_max]
            (in cardinal directions [lat_south, lon_west, lat_north, lon_east])

        Returns
        -------
        list[str]
            List of stations IDs
        """
        # Unpack bbox (lat_min, lon_min, lat_max, lon_max)
        lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
        df = cls.get_station_history()

        cols = {c.lower(): c for c in df.columns}
        lat_col = cols.get("lat")
        lon_col = cols.get("lon")
        usaf_col = cols.get("usaf")
        wban_col = cols.get("wban")

        df = df[[usaf_col, wban_col, lat_col, lon_col]].dropna(
            subset=[lat_col, lon_col]
        )
        # Ensure numeric lat/lon
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        # Normalize longitudes from [-180, 180) to [0, 360) if needed
        if lon_max >= 180:
            df[lon_col] = (df[lon_col] + 360) % 360
        df = df[(df[lat_col] >= lat_min) & (df[lat_col] <= lat_max)]
        df = df[(df[lon_col] >= lon_min) & (df[lon_col] <= lon_max)]
        df[wban_col] = (
            pd.to_numeric(df[wban_col], errors="coerce").fillna(0).astype(int)
        )
        df[usaf_col] = df[usaf_col].astype(str).str.zfill(6)
        return (df[usaf_col] + df[wban_col].map(lambda x: f"{x:05d}")).tolist()

    # ================
    # Field processing
    # ================

    def _extract_ws10m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 10m wind speed from ISD WND column into 'ws10m' (m/s).

        WND is a comma-separated field:
            (direction(deg), direction quality code, type_code, speed(0.1 m/s), speed quality code)
        9999 indicates missing.
        """
        if "WND" not in df:
            df["ws10m"] = np.nan
            return df
        s = df["WND"].astype(str)
        parts = s.str.split(",", expand=True)
        # parts[3] is speed scaled by 10
        spd = pd.to_numeric(parts[3], errors="coerce")
        spd = spd.where(spd != 9999, np.nan) / 10.0
        df["ws10m"] = spd
        return df

    def _extract_uv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract 10m wind components (u10m, v10m) from ISD WND column.

        WND is a comma-separated field:
            (direction(deg), direction quality code, type_code, speed(0.1 m/s), speed quality code)
        Components follow the meteorological convention (direction wind blows from):
        u = -speed * sin(dir), v = -speed * cos(dir)
        """
        if "WND" not in df:
            df["u10m"] = np.nan
            df["v10m"] = np.nan
            return df
        s = df["WND"].astype(str)
        parts = s.str.split(",", expand=True)
        # Require at least 4 parts to compute
        if parts.shape[1] < 4:
            df["u10m"] = np.nan
            df["v10m"] = np.nan
            return df

        direction = pd.to_numeric(parts[0], errors="coerce")
        type_code = parts[2].astype(str)
        speed = pd.to_numeric(parts[3], errors="coerce")
        # Missing speed handling and scale
        speed = speed.where(speed != 9999, np.nan) / 10.0
        # Calm condition: reported 'C' or reported 9 with non-nan speed
        calm_mask = (type_code == "C") | ((direction == 9) & type_code.notna())
        speed = speed.mask(calm_mask, 0.0)

        rad = np.radians(direction)
        u = -np.sin(rad) * speed
        v = -np.cos(rad) * speed
        # Where either speed or direction invalid, set NaN
        valid = speed.notna() & direction.notna()
        df["u10m"] = u.where(valid, np.nan)
        df["v10m"] = v.where(valid, np.nan)
        return df

    def _extract_tp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the hourly total precipitation in meters

        Liquid precipitation period quantity is indicated by column AA1 and is provided
        in mm at a scale factor of 10 (so 10^-5). 9999 is missing
        """
        if "AA1" not in df:
            df["tp"] = np.nan
            return df
        s = df["AA1"].astype(str)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] < 2:
            df["tp"] = np.nan
            return df
        amt = pd.to_numeric(parts[1], errors="coerce")
        # 9999 indicates missing
        amt = amt.where(amt != 9999.0, np.nan) / 10000.0
        df["tp"] = amt
        return df

    def _extract_t2m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the surface temperature in K

        Temperature is indicated by TMP and is provided in Celcius with scaling factor
        of 10, 9999 means missing, in the string field:
            (temp (0.1 C), air temp quality code)
        """
        if "TMP" not in df:
            df["t2m"] = np.nan
            return df
        s = df["TMP"].astype(str)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] < 1:
            df["t2m"] = np.nan
            return df
        temp = pd.to_numeric(parts[0], errors="coerce")
        # 9999 indicates missing; convert 0.1 C to K
        temp = temp.where(temp != 9999.0, np.nan) / 10.0 + 273.15
        df["t2m"] = temp
        return df

    def _extract_fg10m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the surface wind gust speed

        Wind gust observations is denoted by OC1 in m/s with a scaling factor of 10,
        9999 means missing. String field:
            (temp (0.1 m/s), quality code)
        """
        if "OC1" not in df:
            df["fg10m"] = np.nan
            return df
        s = df["OC1"].astype(str)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] < 1:
            df["fg10m"] = np.nan
            return df
        gust = pd.to_numeric(parts[0], errors="coerce")
        # 9999 indicates missing; convert 0.1 m/s to m/s
        gust = gust.where(gust != 9999.0, np.nan) / 10.0
        df["fg10m"] = gust
        return df

    def _extract_d2m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the surface dew point in Kelvin

        Dew point is denoted by DEW in C with a scaling factor of 10,
        9999 means missing. String field:
            (temp (0.1 C), quality code)
        """
        if "DEW" not in df:
            df["d2m"] = np.nan
            return df
        s = df["DEW"].astype(str)
        parts = s.str.split(",", expand=True)
        if parts.shape[1] < 1:
            df["d2m"] = np.nan
            return df
        dew = pd.to_numeric(parts[0], errors="coerce")
        # 9999 indicates missing; convert 0.1 C to K
        dew = dew.where(dew != 9999.0, np.nan) / 10.0 + 273.15
        df["d2m"] = dew
        return df

    def _extract_tcc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the total cloud cover (0 - 1)

        Cloud cover seems to denoted by a handful of identifies that are labeled as
        "domain specific", this will check to see if GA1, GD1 or GF1 are present. Cloud
        cover is described through a few different codes in a string field:
            (coverage code, coverage code #2, ...)
        The identifier codes include:
            For GA:
                00 = None, SKC or CLR
                01 = One okta - 1/10 or less but not zero
                02 = Two oktas - 2/10 - 3/10, or FEW
                03 = Three oktas - 4/10
                04 = Four oktas - 5/10, or SCT
                05 = Five oktas - 6/10
                06 = Six oktas - 7/10 - 8/10
                07 = Seven oktas - 9/10 or more but not 10/10, or BKN
                08 = Eight oktas - 10/10, or OVC
                09 = Sky obscured, or cloud amount cannot be estimated
                10 = Partial obscuration
                99 = Missing
            For GD (field 1)
                0 = Clear - No coverage
                1 = FEW - 2/8 or less coverage (not including zero)
                2 = SCATTERED - 3/8-4/8 coverage
                3 = BROKEN - 5/8-7/8 coverage
                4 = OVERCAST - 8/8 coverage
                5 = OBSCURED
                6 = PARTIALLY OBSCURED
                9 = MISSING
            For GF (field 1)
                00 = None, SKC or CLR
                01 = One okta - 1/10 or less but not zero
                02 = Two oktas - 2/10 - 3/10, or FEW
                03 = Three oktas - 4/10
                04 = Four oktas - 5/10, or SCT
                05 = Five oktas - 6/10
                06 = Six oktas - 7/10 - 8/10
                07 = Seven oktas - 9/10 or more but not 10/10, or BKN
                08 = Eight oktas - 10/10, or OVC
                09 = Sky obscured, or cloud amount cannot be estimated
                10 = Partial obscuration
                12 = Scattered
                13 = Dark scattered
                15 = Broken
                16 = Dark broken
                18 = Overcast
                19 = Dark overcast
                99 = Missing
        """
        okta_lookup = {
            0: 0.0,
            1: 0.1,
            2: 0.25,
            3: 0.4,
            4: 0.5,
            5: 0.6,
            6: 0.75,
            7: 0.9,
            8: 1.0,
        }
        df["tcc"] = np.nan
        # Prefer GA1, then GD1, then GF1
        if "GA1" in df:
            s = df["GA1"].astype(str)
            parts = s.str.split(",", expand=True)
            code = pd.to_numeric(parts[0], errors="coerce")
            df["tcc"] = code.map(
                lambda x: okta_lookup[x] if x in okta_lookup else np.nan
            )
        elif "GD1" in df:
            # Map GD categories to approximate fraction cover (0-1)
            gd_map = {0: 0.0, 1: 0.125, 2: 0.375, 3: 0.75, 4: 1.0}
            s = df["GD1"].astype(str)
            parts = s.str.split(",", expand=True)
            code = pd.to_numeric(parts[0], errors="coerce")
            df["tcc"] = code.map(lambda x: gd_map[x] if x in gd_map else np.nan)
        elif "GF1" in df:
            s = df["GF1"].astype(str)
            parts = s.str.split(",", expand=True)
            code = pd.to_numeric(parts[0], errors="coerce")
            df["tcc"] = code.map(
                lambda x: okta_lookup[x] if x in okta_lookup else np.nan
            )
        # Ensure output bounded [0,1]
        df["tcc"] = df["tcc"].where((df["tcc"] >= 0.0) & (df["tcc"] <= 1.0), np.nan)
        return df
