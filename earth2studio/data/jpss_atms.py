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
import os
import pathlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.jpss import (
    ATMS_NUM_CHANNELS,
    JPSSATMSLexicon,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import TimeTolerance, normalize_time_tolerance
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import eccodes
except ImportError:
    OptionalDependencyFailure("data")
    eccodes = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# NOAA-20 S3 bucket layout
# s3://noaa-nesdis-n20-pds/ATMS_BUFR/<YYYY>/<MM>/<DD>/*.bufr
# ---------------------------------------------------------------------------

# Satellite identifier mapping (WMO code 001007)
_SAT_ID_MAP: dict[int, str] = {
    224: "npp",  # Suomi NPP
    225: "n20",  # NOAA-20 / JPSS-1
    226: "n21",  # NOAA-21 / JPSS-2
}

# S3 bucket per satellite short-name
_SAT_BUCKET_MAP: dict[str, str] = {
    "n20": "noaa-nesdis-n20-pds",
    "n21": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}


@dataclass
class _ATMSAsyncTask:
    """Metadata for a single BUFR file download task."""

    s3_uri: str
    datetime_min: datetime
    datetime_max: datetime
    satellite: str
    variable: str
    bufr_key: str
    modifier: object  # Callable[[Any], Any]


@check_optional_dependencies()
class JPSS_ATMS:
    """JPSS ATMS (Advanced Technology Microwave Sounder) Level 1 BUFR
    brightness-temperature observations served from NOAA Open Data on AWS.

    Each BUFR file contains a single scan line with
    :data:`~earth2studio.lexicon.jpss.ATMS_NUM_FOVS` cross-track
    field-of-view (FOV) positions and
    :data:`~earth2studio.lexicon.jpss.ATMS_NUM_CHANNELS` microwave channels.
    The returned :class:`~pandas.DataFrame` has one row per FOV per channel,
    following the same convention as :class:`~earth2studio.data.UFSObsSat`.

    Parameters
    ----------
    satellites : list[str] | None, optional
        Satellite short-names to query.  Valid values are ``"n20"``
        (NOAA-20), ``"n21"`` (NOAA-21), and ``"npp"`` (Suomi NPP).
        By default ``None``, which queries all valid satellites.
    time_tolerance : TimeTolerance, optional
        Symmetric or asymmetric tolerance window around each requested time
        for selecting BUFR granules, by default ``np.timedelta64(30, "m")``.
    cache : bool, optional
        Cache downloaded BUFR files locally, by default True
    verbose : bool, optional
        Show download progress bars, by default True
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600
    max_workers : int, optional
        Maximum number of concurrent S3 fetch tasks, by default 24
    retries : int, optional
        Per-file retry count on transient I/O failures, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository:

    - https://registry.opendata.aws/noaa-nesdis-n20-pds/
    - https://www.star.nesdis.noaa.gov/jpss/ATMS.php
    - https://www.nesdis.noaa.gov/current-satellite-missions/currently-flying/joint-polar-satellite-system

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timedelta
        from earth2studio.data import JPSS_ATMS

        # Use all possible satellites
        ds = JPSS_ATMS()
        df = ds(datetime(2024, 6, 1, 12), ["atms"])

        # Use specific satellite
        ds = JPSS_ATMS(satellites=["n20"])
        df = ds(datetime(2024, 6, 1, 12), ["atms"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.JPSS_ATMS"
    VALID_SATELLITES = frozenset(["n20", "n21", "npp"])

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            pa.field(
                "scan_angle",
                pa.float32(),
                nullable=True,
                metadata={"bufr_name": "fieldOfViewNumber"},
            ),
            E2STUDIO_SCHEMA.field("channel_index"),
            E2STUDIO_SCHEMA.field("solza"),
            E2STUDIO_SCHEMA.field("solaza"),
            E2STUDIO_SCHEMA.field("satellite_za"),
            E2STUDIO_SCHEMA.field("satellite_aza"),
            E2STUDIO_SCHEMA.field("channel_quality_flag"),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        satellites: list[str] | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(30, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        max_workers: int = 24,
        retries: int = 3,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid options: {sorted(self.VALID_SATELLITES)}"
                )
        self._satellites = satellites
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs: s3fs.S3FileSystem | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------
    async def _async_init(self) -> None:
        """Initialise the async S3 filesystem."""
        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={},
            asynchronous=True,
            skip_instance_cache=True,
        )

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch ATMS brightness-temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return (e.g. ``["atms"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None (all).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with one row per FOV per channel.
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
                    self.fetch(time, variable, fields), timeout=self.async_timeout
                )
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async implementation of the data fetch.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialised! If calling this function "
                "directly, make sure the data source is initialised inside "
                "the async loop!"
            )

        session = await self.fs.set_session(refresh=True)

        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)
        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate variables
        for v in variable_list:
            try:
                JPSSATMSLexicon[v]  # type: ignore
            except KeyError:
                logger.error(f"Variable id {v} not found in JPSS ATMS lexicon")
                raise

        # Discover and download BUFR files within tolerance windows
        tasks = await self._create_tasks(time_list, variable_list)

        # Deduplicate by S3 URI
        uri_set = {t.s3_uri for t in tasks}
        fetch_jobs = [self._fetch_remote_file(uri) for uri in uri_set]
        await tqdm.gather(
            *fetch_jobs,
            desc="Fetching ATMS BUFR files",
            disable=(not self._verbose),
        )

        if session:
            await session.close()

        # Decode and compile
        df = self._compile_dataframe(tasks, schema)
        return df

    # ------------------------------------------------------------------
    # Task creation – discover BUFR granules in S3
    # ------------------------------------------------------------------
    async def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[_ATMSAsyncTask]:
        """Build download tasks by listing the S3 day-directory.

        For each requested time ± tolerance we list the relevant day
        directories on each satellite bucket and select files whose
        embedded start-timestamp falls within the tolerance window.
        """
        tasks: list[_ATMSAsyncTask] = []

        for v in variable_list:
            bufr_key, modifier = JPSSATMSLexicon[v]  # type: ignore

            for sat in self._satellites:
                bucket = _SAT_BUCKET_MAP[sat]

                for t in time_list:
                    tmin = t + self._tolerance_lower
                    tmax = t + self._tolerance_upper

                    # Iterate over calendar days covered by the window
                    day = tmin.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_day = tmax.replace(hour=0, minute=0, second=0, microsecond=0)

                    while day <= end_day:
                        prefix = (
                            f"{bucket}/ATMS_BUFR/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )
                        try:
                            listing = await self.fs._ls(prefix, detail=False)  # type: ignore[union-attr]
                        except FileNotFoundError:
                            logger.warning(f"No ATMS data at s3://{prefix}")
                            day += timedelta(days=1)
                            continue

                        for path in listing:
                            fname = path.rsplit("/", 1)[-1]
                            file_time = self._parse_filename_time(fname)
                            if file_time is None:
                                continue
                            if tmin <= file_time <= tmax:
                                tasks.append(
                                    _ATMSAsyncTask(
                                        s3_uri=f"s3://{path}",
                                        datetime_min=tmin,
                                        datetime_max=tmax,
                                        satellite=sat,
                                        variable=v,
                                        bufr_key=bufr_key,
                                        modifier=modifier,
                                    )
                                )

                        day += timedelta(days=1)

        return tasks

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, s3_uri: str) -> None:
        """Download a single BUFR file to local cache (with retry)."""
        local_path = self._cache_path(s3_uri)
        if pathlib.Path(local_path).is_file():
            return

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                data = await self.fs._cat_file(s3_uri.replace("s3://", "", 1))  # type: ignore[union-attr]
                with open(local_path, "wb") as fh:
                    fh.write(data)
                return
            except (OSError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        logger.warning(f"Failed to fetch {s3_uri} after {self._retries} retries")
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # BUFR decoding & DataFrame compilation
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        tasks: list[_ATMSAsyncTask],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode cached BUFR files and assemble the output DataFrame."""
        frames: list[pd.DataFrame] = []

        for task in tasks:
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {task.s3_uri}")
                continue

            try:
                df = self._decode_bufr(local_path, task)
            except Exception:
                logger.warning(f"Failed to decode {task.s3_uri}", exc_info=True)
                continue

            if df.empty:
                continue

            # Filter by time tolerance window
            mask = (df["time"] >= task.datetime_min) & (df["time"] <= task.datetime_max)
            df = df.loc[mask]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=schema.names)

        result = pd.concat(frames, ignore_index=True)
        result.attrs["source"] = self.SOURCE_ID
        return result[[name for name in schema.names if name in result.columns]]

    def _decode_bufr(
        self,
        path: str,
        task: _ATMSAsyncTask,
    ) -> pd.DataFrame:
        """Decode a single ATMS BUFR file into a DataFrame.

        Each BUFR message contains *N* subsets (FOVs).  For each subset the
        brightness temperature array has *C* channel values, yielding
        ``N * C`` rows in the output.
        """
        rows: list[dict] = []

        with open(path, "rb") as fh:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(fh)
                if msgid is None:
                    break
                try:
                    eccodes.codes_set(msgid, "unpack", 1)

                    n_subsets = eccodes.codes_get(msgid, "numberOfSubsets")

                    # Extract per-FOV arrays (length = n_subsets)
                    lat = eccodes.codes_get_array(msgid, "latitude")
                    lon = eccodes.codes_get_array(msgid, "longitude")
                    fov = eccodes.codes_get_array(msgid, "fieldOfViewNumber")
                    solza = eccodes.codes_get_array(msgid, "solarZenithAngle")
                    solaza = eccodes.codes_get_array(msgid, "solarAzimuth")
                    sat_za = eccodes.codes_get_array(msgid, "satelliteZenithAngle")
                    sat_aza = eccodes.codes_get_array(msgid, "bearingOrAzimuth")

                    # Brightness temperature array is channel-major:
                    # [ch1_fov0, ch1_fov1, ..., ch1_fovN, ch2_fov0, ...]
                    # i.e. shape (n_channels, n_fov) when reshaped, then
                    # transposed to (n_fov, n_channels) for row iteration.
                    bt_flat = eccodes.codes_get_array(msgid, task.bufr_key)
                    n_channels = ATMS_NUM_CHANNELS
                    n_fov = n_subsets

                    if bt_flat.size != n_fov * n_channels:
                        logger.warning(
                            f"Unexpected BT array size {bt_flat.size} in {path}, "
                            f"expected {n_fov}×{n_channels}. Skipping message."
                        )
                        continue

                    bt = bt_flat.reshape(n_channels, n_fov).T

                    # Per-channel quality flags (shape n_channels, one per channel)
                    try:
                        cqf = eccodes.codes_get_array(msgid, "channelDataQualityFlags")
                        # Normalise to exactly n_channels entries.  Some
                        # BUFR producers emit per-FOV-per-channel arrays
                        # (length n_fov * n_channels); take only the first
                        # n_channels in that case (values repeat per-FOV).
                        if cqf.size >= n_channels:
                            cqf = cqf[:n_channels].astype(np.uint16)
                        else:
                            cqf = np.zeros(n_channels, dtype=np.uint16)
                    except Exception:
                        cqf = np.zeros(n_channels, dtype=np.uint16)

                    # Time fields — may be scalars or per-subset arrays
                    # depending on the BUFR producer.  Use codes_get_array
                    # for all of them and broadcast scalars to length n_fov.
                    def _get_time_array(key: str) -> np.ndarray:
                        try:
                            arr = eccodes.codes_get_array(msgid, key)
                        except Exception:
                            arr = np.zeros(n_fov)
                        if arr.size == 1:
                            arr = np.full(n_fov, arr[0])
                        return arr.astype(int)

                    years = _get_time_array("year")
                    months = _get_time_array("month")
                    days = _get_time_array("day")
                    hours = _get_time_array("hour")
                    minutes = _get_time_array("minute")
                    seconds = _get_time_array("second")

                    # Satellite id
                    try:
                        sat_id = int(eccodes.codes_get(msgid, "satelliteIdentifier"))
                        sat_name = _SAT_ID_MAP.get(sat_id, task.satellite)
                    except Exception:
                        sat_name = task.satellite

                    # Build rows: one per (FOV, channel)
                    for i in range(n_fov):
                        try:
                            obs_time = datetime(
                                int(years[i]),
                                int(months[i]),
                                int(days[i]),
                                int(hours[i]),
                                int(minutes[i]),
                                int(seconds[i]),
                            )
                        except (ValueError, OverflowError):
                            continue  # skip FOVs with invalid timestamps

                        for ch in range(n_channels):
                            val = float(bt[i, ch])
                            # Skip missing / fill values
                            if val > 1e6 or val < 0:
                                continue

                            rows.append(
                                {
                                    "time": obs_time,
                                    "lat": float(lat[i]),
                                    "lon": float(lon[i]) % 360.0,
                                    "scan_angle": float(fov[i]),
                                    "channel_index": ch + 1,
                                    "solza": float(solza[i]),
                                    "solaza": float(solaza[i]),
                                    "satellite_za": float(sat_za[i]),
                                    "satellite_aza": float(sat_aza[i]),
                                    "channel_quality_flag": int(cqf[ch]),
                                    "satellite": sat_name,
                                    "observation": val,
                                    "variable": task.variable,
                                }
                            )
                finally:
                    eccodes.codes_release(msgid)

        if not rows:
            return pd.DataFrame(columns=self.SCHEMA.names)

        df = pd.DataFrame(rows)
        # Enforce schema dtypes
        df["time"] = pd.to_datetime(df["time"])
        df["lat"] = df["lat"].astype(np.float32)
        df["lon"] = df["lon"].astype(np.float32)
        df["scan_angle"] = df["scan_angle"].astype(np.float32)
        df["channel_index"] = df["channel_index"].astype(np.uint16)
        df["solza"] = df["solza"].astype(np.float32)
        df["solaza"] = df["solaza"].astype(np.float32)
        df["satellite_za"] = df["satellite_za"].astype(np.float32)
        df["satellite_aza"] = df["satellite_aza"].astype(np.float32)
        df["channel_quality_flag"] = df["channel_quality_flag"].astype(np.uint16)
        df["observation"] = df["observation"].astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # File-name timestamp parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_filename_time(filename: str) -> datetime | None:
        """Extract the scan start time from an ATMS BUFR filename.

        Expected pattern::

            ATMS_v1r0_j01_s<YYYYMMDDHHMMSSF>_e..._c....bufr

        Returns ``None`` if the filename does not match.
        """
        parts = filename.split("_")
        for part in parts:
            if part.startswith("s") and len(part) >= 15:
                try:
                    return datetime.strptime(part[1:15], "%Y%m%d%H%M%S")
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------------------
    # resolve_fields / cache / available
    # ------------------------------------------------------------------
    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert *fields* parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification.

        Returns
        -------
        pa.Schema
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not in SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(field.name).type
                if field.type != expected:
                    raise TypeError(
                        f"Field '{field.name}' type {field.type} != {expected}"
                    )
            return fields

        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in SCHEMA. Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "jpss_atms")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_jpss_atms_{self._tmp_cache_hash}"
            )
        return cache_location

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic local cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode())
        return os.path.join(self.cache, sha.hexdigest())

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are within the data range.

        Parameters
        ----------
        times : list[datetime]
            Date-times to validate.
        """
        start_date = datetime(2017, 11, 29)
        for t in times:
            if t < start_date:
                raise ValueError(
                    f"Requested date time {t} needs to be after "
                    f"{start_date} for JPSS ATMS"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether data is available for a given time.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date-time to check.

        Returns
        -------
        bool
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
