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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
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
from earth2studio.lexicon import GOESGLMLexicon
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    import netCDF4
except ImportError:
    OptionalDependencyFailure("data")
    netCDF4 = None  # type: ignore[assignment]


_GLM_PREFIX = "GLM-L2-LCFA"

_BUCKETS = {
    "G16": "noaa-goes16",
    "G17": "noaa-goes17",
    "G18": "noaa-goes18",
    "G19": "noaa-goes19",
}

# Each entry: (platform, start, end) - the platform is the active GOES
# satellite for the slot during ``[start, end)``. Boundaries are inclusive
# on the lower side and exclusive on the upper side. These dates reflect
# the public AWS GLM-L2-LCFA archive coverage; explicit-satellite requests
# outside a platform's window will simply return no events from S3.
_SLOT_HISTORY: dict[str, tuple[tuple[str, datetime, datetime], ...]] = {
    "east": (
        ("G16", datetime(2018, 2, 13), datetime(2025, 4, 4)),
        ("G19", datetime(2025, 4, 4), datetime.max),
    ),
    "west": (
        ("G17", datetime(2018, 12, 10), datetime(2023, 1, 4)),
        ("G18", datetime(2023, 1, 4), datetime.max),
    ),
}

_GLM_MIN_DATE = datetime(2018, 2, 13)

# Each LCFA file covers a ~20 s scan; widen the lower file-start bound by
# this much so events at the leading edge of a tight tolerance window are
# not silently dropped.
_GLM_FILE_DURATION = timedelta(seconds=20)


@dataclass(frozen=True)
class _GOESGLMFile:
    """One GLM L2 LCFA NetCDF file scheduled for download."""

    s3_uri: str
    satellite: str
    file_start: datetime


@check_optional_dependencies()
class GOESGLM:
    """NOAA GOES Geostationary Lightning Mapper (GLM) Level 2 Lightning
    Cluster-Filter Algorithm (LCFA) event data source.

    Returns per-event lightning observations from the GLM instrument on
    GOES-16/17/18/19, served as point observations in a pandas
    DataFrame. Each row corresponds to a single optical event detected
    by GLM with a sub-second timestamp, latitude/longitude, and the
    requested measurement (``flashe`` for optical energy in Joules,
    ``flashc`` for a constant 1.0 per detected event suitable for
    density aggregation).

    Files in the public NOAA AWS bucket are NetCDFs produced at roughly
    20 second cadence covering the GOES full-disk field of view. A
    spatial bounding box can be supplied to restrict events at parse
    time and reduce memory usage for large windows.

    Parameters
    ----------
    satellite : str, optional
        Source satellite selector. Pass ``"east"`` (default) or
        ``"west"`` to auto-select the active GOES-East / GOES-West
        platform for each requested timestamp; pass ``"G16"``,
        ``"G17"``, ``"G18"`` or ``"G19"`` to pin a single platform.
    lat_lon_bbox : tuple[float, float, float, float] | None, optional
        Bounding box ``(lat_min, lon_min, lat_max, lon_max)`` in
        degrees, applied at parse time. Accepts either ``[-180, 180)``
        or ``[0, 360)`` longitude convention (auto-detected when
        ``lon_max >= 180``). ``None`` (default) returns the full disk.
        For example, CONUS in the ``[-180, 180)`` convention is
        ``(24.5, -125.0, 49.5, -66.0)``.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for selecting events around each
        requested timestamp. Accepts a single value (symmetric ±
        window) or a tuple ``(lower, upper)`` for asymmetric windows,
        by default ``np.timedelta64(2, "m")``.
    cache : bool, optional
        Cache downloaded NetCDF files on local disk, by default True.
    verbose : bool, optional
        Show download progress bar, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation,
        by default 600.
    async_workers : int, optional
        Maximum number of concurrent S3 fetch tasks, by default 24.
    retries : int, optional
        Number of retry attempts per failed fetch task with
        exponential backoff, by default 3.

    Warning
    -------
    GLM produces hundreds of files per hour. Large time windows can
    download tens to hundreds of gigabytes of NetCDFs. Use
    ``lat_lon_bbox`` to discard out-of-region events on parse and
    keep ``time_tolerance`` bounded.

    Note
    ----
    Output longitudes are normalised to ``[0, 360)`` (Earth2Studio
    convention). Each event's timestamp is computed from the file's
    ``event_time_offset`` variable so per-event precision (~ms) is
    preserved.

    Note
    ----
    Additional information on the data repository:

    - https://www.goes-r.gov/products/baseline-LCFA.html
    - https://registry.opendata.aws/noaa-goes/
    - https://www.ncei.noaa.gov/products/satellite/goes-glm

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime
        import numpy as np
        from earth2studio.data import GOESGLM

        ds = GOESGLM(
            satellite="east",
            lat_lon_bbox=(24.5, -125.0, 49.5, -66.0),  # CONUS
            time_tolerance=np.timedelta64(5, "m"),
        )
        df = ds(datetime(2024, 6, 1, 18, 0), ["flashe", "flashc"])

    Badges
    ------
    region:na region:sa dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.goes_glm"
    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
            pa.field(
                "satellite",
                pa.string(),
                metadata={"description": "GOES satellite platform (G16/G17/G18/G19)"},
            ),
        ]
    )

    def __init__(
        self,
        satellite: str = "east",
        lat_lon_bbox: tuple[float, float, float, float] | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(2, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 24,
        retries: int = 3,
    ) -> None:
        sat = (
            satellite.lower()
            if satellite.lower() in ("east", "west")
            else satellite.upper()
        )
        if sat not in ("east", "west") and sat not in _BUCKETS:
            raise ValueError(
                f"Unknown satellite {satellite!r}; expected one of "
                f"'east', 'west', {sorted(_BUCKETS)}"
            )
        self._satellite = sat

        self._lat_lon_bbox = _normalize_lat_lon_bbox(lat_lon_bbox)

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        self.fs: s3fs.S3FileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialization of the anonymous S3 filesystem.

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

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch GLM lightning events for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return events for (UTC). Timezone-aware
            datetimes are converted to UTC automatically.
        variable : str | list[str] | VariableArray
            Variable ids defined in
            :py:class:`earth2studio.lexicon.GOESGLMLexicon`
            (``"flashe"`` and/or ``"flashc"``).
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` (default) returns all
            schema fields.

        Returns
        -------
        pd.DataFrame
            Event-level lightning observations with columns matching
            the resolved schema.
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
        """Async function to fetch GLM events.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return events for (UTC).
        variable : str | list[str] | VariableArray
            Variable ids defined in :py:class:`GOESGLMLexicon`.
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` (default) returns all
            schema fields.

        Returns
        -------
        pd.DataFrame
            Event-level lightning observations.
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        for v in variable_list:
            if v not in GOESGLMLexicon.VOCAB:
                raise KeyError(
                    f"Variable id {v!r} not found in GOESGLMLexicon. "
                    f"Available: {list(GOESGLMLexicon.VOCAB)}"
                )

        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        files = await self._discover_files(time_list)
        unique_uris = sorted({f.s3_uri for f in files})
        logger.info(
            f"[{self.SOURCE_ID}] discovered {len(unique_uris)} unique GLM "
            f"files across {len(time_list)} requested times"
        )

        async with managed_session(self.fs):
            coros = [
                async_retry(
                    self._fetch_remote_file,
                    uri,
                    retries=self._retries,
                    backoff=1.0,
                    task_timeout=120.0,
                    exceptions=(OSError, IOError, TimeoutError, ConnectionError),
                )
                for uri in unique_uris
            ]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching GLM L2 LCFA files",
                verbose=(not self._verbose),
            )

        return self._compile_dataframe(files, time_list, variable_list, schema)

    async def _discover_files(self, time_list: list[datetime]) -> list[_GOESGLMFile]:
        """List GLM L2 LCFA keys whose file-start times fall in any
        requested ``[t-tol, t+tol]`` window.

        Hourly S3 prefixes are listed once each (deduplicated) and the
        per-file start timestamps are parsed from the LCFA filename
        convention.
        """
        prefix_jobs: dict[tuple[str, str], None] = {}
        windows: list[tuple[datetime, datetime, str]] = []
        for t in time_list:
            tmin = t + self._tolerance_lower
            tmax = t + self._tolerance_upper
            sat = self._satellite_for_time(t)
            windows.append((tmin, tmax, sat))
            hr = (tmin - _GLM_FILE_DURATION).replace(minute=0, second=0, microsecond=0)
            while hr <= tmax:
                prefix_jobs[(sat, self._hour_prefix(sat, hr))] = None
                hr += timedelta(hours=1)

        async def _list_one(sat: str, prefix: str) -> list[tuple[str, str]]:
            try:
                keys = await self.fs._ls(  # type: ignore[union-attr]
                    f"{_BUCKETS[sat]}/{prefix}", detail=False
                )
            except FileNotFoundError:
                return []
            return [(sat, k) for k in keys if k.endswith(".nc")]

        listings = await gather_with_concurrency(
            [_list_one(sat, prefix) for (sat, prefix) in prefix_jobs],
            max_workers=self._async_workers,
            desc="Listing GLM L2 LCFA prefixes",
            verbose=(not self._verbose),
        )

        out: list[_GOESGLMFile] = []
        seen_uris: set[str] = set()
        for entries in listings:
            for sat, key in entries:
                file_start = self._parse_filename_start(key)
                if file_start is None:
                    continue
                for tmin, tmax, win_sat in windows:
                    if win_sat != sat:
                        continue
                    # GLM files cover ~20 s; accept any file whose scan
                    # starts within one file duration before ``tmin`` so
                    # events at the leading edge of the window are not
                    # silently dropped for tight tolerances.
                    if tmin - _GLM_FILE_DURATION <= file_start <= tmax:
                        uri = (
                            f"s3://{key}"
                            if key.startswith(_BUCKETS[sat])
                            else (f"s3://{_BUCKETS[sat]}/{key}")
                        )
                        if uri in seen_uris:
                            continue
                        seen_uris.add(uri)
                        out.append(
                            _GOESGLMFile(
                                s3_uri=uri,
                                satellite=sat,
                                file_start=file_start,
                            )
                        )
                        break
        return out

    async def _fetch_remote_file(self, uri: str) -> None:
        """Download one GLM NetCDF file into the cache directory."""
        if self.fs is None:
            raise ValueError("File system is not initialized")
        cache_path = self._cache_path(uri)
        if pathlib.Path(cache_path).is_file():
            return
        try:
            data = await self.fs._cat_file(uri)
        except FileNotFoundError:
            logger.warning(f"GLM file {uri} not found, skipping")
            return
        with open(cache_path, "wb") as fh:
            fh.write(data)

    def _compile_dataframe(
        self,
        files: list[_GOESGLMFile],
        time_list: list[datetime],
        variable_list: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Parse each cached file once, filter per requested time, and
        emit one long-format DataFrame across all variables and times.
        """
        events_by_file: dict[str, pd.DataFrame] = {}
        for f in files:
            local = self._cache_path(f.s3_uri)
            if not pathlib.Path(local).is_file():
                continue
            events = self._parse_glm_file(local, self._lat_lon_bbox)
            if events is None or events.empty:
                continue
            events["satellite"] = f.satellite
            events_by_file[f.s3_uri] = events

        if not events_by_file:
            return self._empty_result(schema)

        frames: list[pd.DataFrame] = []
        for t in time_list:
            tmin = pd.Timestamp(t + self._tolerance_lower)
            tmax = pd.Timestamp(t + self._tolerance_upper)
            sat = self._satellite_for_time(t)
            for events in events_by_file.values():
                if events["satellite"].iloc[0] != sat:
                    continue
                mask = (events["time"] >= tmin) & (events["time"] <= tmax)
                if not mask.any():
                    continue
                window = events.loc[mask]
                for v in variable_list:
                    sub = window[["time", "lat", "lon", "satellite"]].copy()
                    if v == "flashe":
                        sub["observation"] = window["event_energy"].astype(np.float32)
                    else:  # flashc
                        sub["observation"] = np.float32(1.0)
                    sub["variable"] = v
                    frames.append(sub)

        if not frames:
            return self._empty_result(schema)

        df = pd.concat(frames, ignore_index=True)
        # Normalise longitude from GLM's [-180, 180) to Earth2Studio's [0, 360)
        df["lon"] = (df["lon"].astype(np.float32) + 360.0) % 360.0
        df["lat"] = df["lat"].astype(np.float32)
        df.attrs["source"] = self.SOURCE_ID
        return df[[name for name in schema.names if name in df.columns]]

    def _empty_result(self, schema: pa.Schema) -> pd.DataFrame:
        empty = pd.DataFrame(
            {name: pd.Series(dtype="object") for name in self.SCHEMA.names}
        )
        empty.attrs["source"] = self.SOURCE_ID
        return empty[[name for name in schema.names if name in empty.columns]]

    @classmethod
    def _validate_time(cls, times: Iterable[datetime]) -> None:
        """Verify date times fall within the GLM archive window.

        Per-platform availability (G16/G17/G18/G19 cutovers) is enforced
        downstream in :py:meth:`_satellite_for_time`; this classmethod
        only screens out timestamps before the earliest archive date.
        """
        for t in times:
            if t < _GLM_MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than the GLM "
                    f"archive start ({_GLM_MIN_DATE.isoformat()})."
                )
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            if t > now_utc + timedelta(hours=1):
                raise ValueError(
                    f"Requested datetime {t} is in the future; "
                    f"GLM L2 LCFA archive has no entries beyond now."
                )

    def _satellite_for_time(self, t: datetime) -> str:
        """Resolve the configured satellite selector to a concrete
        platform code for a given timestamp."""
        if self._satellite in _BUCKETS:
            return self._satellite
        for sat, start, end in _SLOT_HISTORY[self._satellite]:
            if start <= t < end:
                return sat
        raise ValueError(
            f"No active GOES-{self._satellite} satellite known for "
            f"{t.isoformat()}; check the SLOT_HISTORY in goes_glm.py."
        )

    @staticmethod
    def _hour_prefix(sat: str, t: datetime) -> str:
        doy = t.timetuple().tm_yday
        return f"{_GLM_PREFIX}/{t.year:04d}/{doy:03d}/{t.hour:02d}/"

    def _cache_path(self, s3_uri: str) -> str:
        sha = hashlib.sha256(s3_uri.encode()).hexdigest()
        return os.path.join(self.cache, sha + ".nc")

    @property
    def cache(self) -> str:
        """Local cache directory for this data source."""
        cache_location = os.path.join(datasource_cache_root(), "goes_glm")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_goes_glm_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether data is available for a given time.

        Offline check against the GLM archive window; per-slot
        platform cutovers are enforced when the source is called.

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

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve a ``fields`` selector into a validated PyArrow schema."""
        if fields is None:
            return cls.SCHEMA
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            for f in fields:
                if f.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{f.name}' not in {cls.__name__} SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(f.name).type
                if f.type != expected:
                    raise TypeError(
                        f"Field '{f.name}' has type {f.type}, expected "
                        f"{expected} from class SCHEMA"
                    )
            return fields
        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in {cls.__name__} SCHEMA. "
                    f"Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @classmethod
    def _parse_filename_start(cls, key: str) -> datetime | None:
        """Extract the file-start datetime from a GLM L2 LCFA S3 key.

        Filenames follow
        ``OR_GLM-L2-LCFA_<G16|G17|G18|G19>_s<YYYYJJJHHMMSSF>_e..._c....nc``
        where ``F`` is a decisecond digit (rounded down here).
        """
        base = os.path.basename(key)
        idx = base.find("_s")
        if idx < 0:
            return None
        stamp = base[idx + 2 : idx + 2 + 13]
        if len(stamp) != 13 or not stamp.isdigit():
            return None
        try:
            return datetime.strptime(stamp, "%Y%j%H%M%S")
        except ValueError:
            return None

    @classmethod
    def _parse_glm_file(
        cls,
        path: str,
        lat_lon_bbox: tuple[float, float, float, float] | None,
    ) -> pd.DataFrame | None:
        """Parse a GLM L2 LCFA NetCDF file into a flat events DataFrame.

        Returns ``None`` if the file has no events or all events fall
        outside ``lat_lon_bbox``.
        """
        with netCDF4.Dataset(path) as ds:
            if "event_lat" not in ds.variables:
                return None
            if ds.dimensions["number_of_events"].size == 0:
                return None

            lat = np.asarray(ds.variables["event_lat"][:], dtype=np.float32)
            lon = np.asarray(ds.variables["event_lon"][:], dtype=np.float32)
            energy = np.asarray(ds.variables["event_energy"][:], dtype=np.float32)
            offset = np.asarray(ds.variables["event_time_offset"][:], dtype=np.float64)
            epoch = (
                pd.Timestamp(ds.time_coverage_start)
                .to_pydatetime()
                .replace(tzinfo=None)
            )

        times = np.asarray(
            [np.datetime64(epoch + timedelta(seconds=float(s)), "us") for s in offset],
            dtype="datetime64[us]",
        )

        if lat_lon_bbox is not None:
            lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
            mask = (
                (lat >= lat_min)
                & (lat <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )
            if not mask.any():
                return None
            lat = lat[mask]
            lon = lon[mask]
            energy = energy[mask]
            times = times[mask]

        return pd.DataFrame(
            {
                "time": pd.to_datetime(times),
                "lat": lat,
                "lon": lon,
                "event_energy": energy,
            }
        )


def _normalize_lat_lon_bbox(
    lat_lon_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Validate a ``(lat_min, lon_min, lat_max, lon_max)`` box.

    Accepts either ``[-180, 180)`` or ``[0, 360)`` longitude
    conventions and normalises to the native GLM ``[-180, 180)``
    convention used by the on-disk event coordinates.
    """
    if lat_lon_bbox is None:
        return None
    lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError(
            "lat_lon_bbox bounds must be "
            "(lat_min < lat_max, lon_min < lon_max); "
            f"got {lat_lon_bbox}"
        )
    if lon_max >= 180.0:
        lon_min = ((lon_min + 180.0) % 360.0) - 180.0
        lon_max = ((lon_max + 180.0) % 360.0) - 180.0
        if lon_min >= lon_max:
            raise ValueError(
                "lat_lon_bbox crosses the antimeridian after "
                "normalising from [0, 360) to [-180, 180); split it "
                f"into two boxes. Got {lat_lon_bbox}."
            )
    return (lat_min, lon_min, lat_max, lon_max)
