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
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.dwd_synop_reports import DWDSynopReportsLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    import pdbufr
except ImportError:
    OptionalDependencyFailure("data")
    pdbufr = None  # type: ignore[assignment]

# DWD Open Data rolling live SYNOP feeds, both decoded from WMO BUFR. The German
# feed is pure BUFR. The international feed is GTS-derived and additionally
# carries some bulletins in traditional alphanumeric (TAC) form; those are not
# decoded here (see the class docstring) -- only its BUFR payloads are read.
GERMANY_URL = "https://opendata.dwd.de/weather/weather_reports/synoptic/germany/"
INTERNATIONAL_URL = (
    "https://opendata.dwd.de/weather/weather_reports/synoptic/international/"
)
_FEED_URLS = {"germany": GERMANY_URL, "international": INTERNATIONAL_URL}
# Nominal file retention: the live feed is a rolling window and DWD prunes
# older bulletin files roughly after this age. It is approximate, not a hard
# cutoff, so time validation stays slightly permissive (see _validate_time).
_FEED_RETENTION = timedelta(days=2)
# DWD filenames use bulletin times, while reports contain observation times.
# Search 15 minutes before each requested observation window to allow small
# timestamp differences, and 90 minutes after it to allow publication delay.
# These margins affect only file discovery; decoded reports are filtered back
# to the exact observation window. More delayed reports may be missed.
_LEAD_PAD = timedelta(minutes=15)
_PUBLISH_LAG_PAD = timedelta(minutes=90)
# Filename embeds the bulletin time: Z__C_EDZW_<YYYYMMDDHHMMSS>_bda01,synop...
_FILENAME_TIME_RE = re.compile(r"_(\d{14})_")
# BUFR element short names kept as observation variables (WMO/pdbufr keys).
_BUFR_ELEMENTS = (
    "airTemperature",
    "dewpointTemperature",
    "windSpeed",
    "windDirection",
    "pressureReducedToMeanSeaLevel",
)
_META_COLUMNS = ("station", "time", "lat", "lon", "elev")
# Columns requested from pdbufr: identity/location/time plus the elements above.
_BUFR_COLUMNS = (
    "latitude",
    "longitude",
    "heightOfStationGroundAboveMeanSeaLevel",
    "blockNumber",
    "stationNumber",
    "stationOrSiteName",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    *_BUFR_ELEMENTS,
)
# Only position and date/hour are required to place a report. pdbufr defaults
# required_columns to all requested columns, which would silently drop any
# station that omits an optional key (station id, elevation, minute, or any
# element) -- common in SYNOP. Requiring only these keeps sparse reports.
_REQUIRED_BUFR_COLUMNS = (
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "hour",
)


def _num(series: pd.Series | None) -> pd.Series | None:
    """Coerce to numeric and null out ecCodes/pdbufr missing sentinels.

    pdbufr may pass through the raw BUFR missing values rather than NaN:
    ``CODES_MISSING_LONG`` (``2**31 - 1``) for integers and
    ``CODES_MISSING_DOUBLE`` (``~-1e100``) for floats.
    """
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s > -1e99) & (s < 2147483647), np.nan)


def _station_ids(raw: pd.DataFrame) -> tuple[list[str | None], list[bool]]:
    """Build station ids: WMO block+station number when present, else name.

    Returns the ids alongside a per-row flag that is True only when the id is a
    real WMO block+station number (not a fallback name). A fallback name may
    itself be five digits, so this flag -- not a ``\\d{5}`` check -- is what lets
    coalescing group true WMO ids without coordinates while keying fallback
    names on coordinates.
    """
    block = _num(raw.get("blockNumber"))
    number = _num(raw.get("stationNumber"))
    name = raw.get("stationOrSiteName")
    ids: list[str | None] = []
    is_wmo: list[bool] = []
    for i in range(len(raw)):
        b = block.iloc[i] if block is not None else None
        s = number.iloc[i] if number is not None else None
        if b is not None and s is not None and pd.notna(b) and pd.notna(s):
            ids.append(f"{int(b):02d}{int(s):03d}")
            is_wmo.append(True)
            continue
        nm = name.iloc[i] if name is not None else None
        ids.append(nm.strip() if isinstance(nm, str) and nm.strip() else None)
        is_wmo.append(False)
    return ids, is_wmo


def _decode_synop_bufr(path: str) -> pd.DataFrame | None:
    """Decode a DWD SYNOP BUFR file into a wide per-station DataFrame.

    Uses :mod:`pdbufr` (ECMWF's pandas BUFR reader) to flatten BUFR messages and
    subsets into one row per station report; :func:`_num` normalizes numeric
    values and missing-value sentinels. Columns are the metadata fields
    (``station, time, lat, lon, elev``) plus the BUFR element short names in
    :data:`_BUFR_ELEMENTS`, and an internal ``_is_wmo`` flag (True only for real
    WMO block+station ids) used by coalescing and dropped before output.

    Parameters
    ----------
    path : str
        Local path to the BUFR file.

    Returns
    -------
    pd.DataFrame | None
        Wide per-station DataFrame (possibly empty for a valid file with no
        reports), or ``None`` if the file could not be decoded at all — so the
        caller can distinguish a decode failure from an empty feed.
    """
    columns = list(_META_COLUMNS) + list(_BUFR_ELEMENTS) + ["_is_wmo"]
    try:
        raw = pdbufr.read_bufr(
            path,
            columns=_BUFR_COLUMNS,
            required_columns=_REQUIRED_BUFR_COLUMNS,
        )
    except Exception as e:
        logger.debug(f"Failed to decode BUFR {path}: {e}")
        return None
    if raw.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(index=raw.index)
    out["station"], out["_is_wmo"] = _station_ids(raw)
    if all(c in raw for c in ("year", "month", "day", "hour")):
        minute = _num(raw["minute"]) if "minute" in raw else None
        out["time"] = pd.to_datetime(
            {
                "year": _num(raw["year"]),
                "month": _num(raw["month"]),
                "day": _num(raw["day"]),
                "hour": _num(raw["hour"]),
                "minute": minute.fillna(0) if minute is not None else 0,
            },
            errors="coerce",
        )
    else:
        out["time"] = pd.NaT
    out["lat"] = _num(raw.get("latitude"))
    out["lon"] = _num(raw.get("longitude"))
    out["elev"] = _num(raw.get("heightOfStationGroundAboveMeanSeaLevel"))
    for k in _BUFR_ELEMENTS:
        out[k] = _num(raw[k]) if k in raw else np.nan
    # Return an owned frame (not a view) so the caller can add columns
    # (e.g. _bulletin) without a SettingWithCopyWarning.
    return out[columns].copy()


@dataclass
class _DWDAsyncTask:
    """A single bulletin file to download and decode."""

    url: str
    file_time: datetime


@check_optional_dependencies()
class DWDSynopReports:
    """DWD live SYNOP surface observations from the DWD Open Data server.

    Provides near-real-time surface observations decoded from WMO SYNOP BUFR. The
    ``germany`` feed is pure BUFR and primarily covers Germany. The
    ``international`` feed is GTS-derived and can include land stations worldwide;
    only its BUFR bulletins are decoded. That feed may contain reports in traditional
    alphanumeric (TAC) form instead of BUFR; those bulletins are not decoded here
    -- they are detected, counted, and skipped (with a warning), so those
    bulletins are omitted from the international results. Returned longitudes are
    normalized to ``[0, 360)`` to match the Earth2Studio convention.

    This endpoint is a rolling live feed, not a historical archive: files are
    posted every few minutes and the server retains only a short window
    (approximately 2 days). Reports are published in receipt batches that may
    lag the observation time; file discovery checks bulletins shortly before
    and up to ~90 minutes after each requested observation window, so more
    delayed reports may be omitted.

    Parameters
    ----------
    feed : Literal["germany", "international"], optional
        Which DWD BUFR feed to read, by default ``"germany"``.
        ``"international"`` broadens coverage to worldwide land BUFR stations (its
        non-BUFR/TAC bulletins are skipped).
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple ``(lower, upper)`` for asymmetric
        windows, by default ``np.timedelta64(30, "m")``.
    lat_lon_bbox : tuple[float, float, float, float] | None, optional
        Bounding box ``(lat_min, lon_min, lat_max, lon_max)`` restricting the
        returned stations. Longitudes may be given in ``[-180, 180]`` or
        ``[0, 360]``; a box with a negative western edge (``lon_min < 0``) is
        interpreted as ``[-180, 180]``. Endpoints are inclusive; boxes that cross
        the selected convention's seam (0/360, or -180/180) are not supported. By
        default None (all stations in the selected feed).
    cache : bool, optional
        Cache downloaded bulletin files locally, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default 600.
    async_workers : int, optional
        Max concurrent async fetch tasks, by default 16.
    retries : int, optional
        Number of retry attempts per retriable connection or I/O failure with
        exponential backoff, by default 3.

    Raises
    ------
    ValueError
        If ``feed`` is not one of the supported feeds, if ``lat_lon_bbox`` is
        malformed (wrong length, out-of-range or mixed longitude conventions,
        inverted or seam-crossing box), if ``time_tolerance`` is a tuple with
        the wrong length or ordering, or if ``async_workers`` < 1,
        ``retries`` < 0, or ``async_timeout`` <= 0.
    TypeError
        If ``time_tolerance`` has an unsupported type.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Users should review DWD's terms for their use case:

    - https://opendata.dwd.de/weather/weather_reports/synoptic/
    - https://www.dwd.de/EN/service/legal_notice/legal_notice_node.html

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc).replace(
            tzinfo=None, minute=0, second=0, microsecond=0
        )
        # All reporting stations in the German feed near a recent synoptic hour
        ds = DWDSynopReports(time_tolerance=timedelta(minutes=30))
        df = ds(now, ["t2m", "u10m", "v10m"])
        # Restrict to a lat/lon box (lat_min, lon_min, lat_max, lon_max)
        ds = DWDSynopReports(lat_lon_bbox=(47, 5.8, 55.1, 15.1))
        df = ds(now, ["t2m"])

    Badges
    ------
    region:eu region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.dwd_synop_reports"

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("elev"),
            E2STUDIO_SCHEMA.field("station"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
        ]
    )

    def __init__(
        self,
        feed: Literal["germany", "international"] = "germany",
        time_tolerance: TimeTolerance = np.timedelta64(30, "m"),
        lat_lon_bbox: tuple[float, float, float, float] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ) -> None:
        if feed not in _FEED_URLS:
            raise ValueError(f"feed must be one of {list(_FEED_URLS)}, got {feed!r}.")
        if async_workers < 1:
            raise ValueError(f"async_workers must be >= 1, got {async_workers}.")
        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}.")
        if async_timeout <= 0:
            raise ValueError(f"async_timeout must be > 0, got {async_timeout}.")
        if lat_lon_bbox is not None:
            if len(lat_lon_bbox) != 4:
                raise ValueError(
                    "lat_lon_bbox must be (lat_min, lon_min, lat_max, lon_max)."
                )
            lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
            lat_ok = -90 <= lat_min < lat_max <= 90
            lon_ok = (-180 <= lon_min < lon_max <= 180) or (
                0 <= lon_min < lon_max <= 360
            )
            if not (lat_ok and lon_ok):
                raise ValueError(
                    "lat_lon_bbox must be (lat_min, lon_min, lat_max, lon_max) "
                    "with latitudes in [-90, 90], longitudes in a single "
                    "[-180, 180] or [0, 360] convention, and min < max; inverted "
                    f"or seam-crossing boxes are unsupported. Got "
                    f"{lat_lon_bbox}."
                )
        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()
        self._feed = feed
        self._feed_url = _FEED_URLS[feed]
        self._lat_lon_bbox = lat_lon_bbox
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: Any = None

    async def _async_init(self) -> None:
        """Initialize the async HTTP filesystem.

        Note
        ----
        Async fsspec expects initialization inside the execution loop.
        """
        from fsspec.implementations.http import (  # type: ignore[import-untyped]
            HTTPFileSystem,
        )

        # skip_instance_cache ensures each instance owns its session, so one
        # instance's managed_session teardown cannot close a session another
        # instance is still using.
        self.fs = HTTPFileSystem(asynchronous=True, skip_instance_cache=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Retrieve DWD SYNOP observations for the given times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC). Observations within
            ``time_tolerance`` of any requested time are returned.
        variable : str | list[str] | VariableArray
            Variables to return. Must be in :class:`DWDSynopReportsLexicon`.
        fields : str | list[str] | pa.Schema | None, optional
            Schema fields to include in output. None returns all fields.

        Returns
        -------
        pd.DataFrame
            Long-format observation data frame following :attr:`SCHEMA`.

        Raises
        ------
        KeyError
            If a requested variable is not in the DWD lexicon, or a requested
            ``fields`` name is not in :attr:`SCHEMA`.
        TypeError
            If a provided ``fields`` schema has a type incompatible with
            :attr:`SCHEMA`.
        ValueError
            If a requested time is NaT, more than one hour in the future, or
            older than the feed retention window.
        RuntimeError
            If listing a feed directory fails, if every file download fails, if
            every fetched bulletin is unsupported (all TAC), or if no fetched
            bulletin could be decoded.
        TimeoutError
            If the fetch does not complete within ``async_timeout`` seconds.
        """
        try:
            df = _sync_async(
                self.fetch,
                time,
                variable,
                fields,
                timeout=self.async_timeout,
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
        """Async fetch of DWD SYNOP observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.
        fields : str | list[str] | pa.Schema | None, optional
            Schema fields to include in output, by default None.

        Returns
        -------
        pd.DataFrame
            Observation data.

        Raises
        ------
        KeyError
            If a requested variable is not in the DWD lexicon, or a requested
            ``fields`` name is not in :attr:`SCHEMA`.
        TypeError
            If a provided ``fields`` schema has a type incompatible with
            :attr:`SCHEMA`.
        ValueError
            If a requested time is NaT, more than one hour in the future, or
            older than the feed retention window.
        RuntimeError
            If listing a feed directory fails, if every file download fails, if
            every fetched bulletin is unsupported (all TAC), or if no fetched
            bulletin could be decoded.
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        output_fields = self.resolve_fields(fields)
        for v in variable_list:
            if v not in DWDSynopReportsLexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in DWDSynopReportsLexicon. "
                    f"Available: {list(DWDSynopReportsLexicon.VOCAB.keys())}"
                )

        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        if self._cache:
            self._prune_cache()

        # One window per requested time, kept separate so that far-apart times
        # do not pull every bulletin between them.
        windows = [
            (t + self._tolerance_lower, t + self._tolerance_upper) for t in time_list
        ]

        async with managed_session(self.fs):
            tasks = await self._create_tasks(windows)
            coros = [self._fetch_wrapper(task) for task in tasks]
            results = await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                desc="Fetching DWD SYNOP observations",
                verbose=(not self._verbose),
            )
            # (bulletin_time, cache_path) pairs for successfully fetched files.
            fetched = [(ft, p) for ft, p, _ in results if p is not None]
            n_download_failed = sum(1 for _, _, failed in results if failed)

        # Genuine download failures with nothing fetched indicate a service
        # problem, not "no observations". Files pruned from the live feed between
        # listing and download are expected misses and do not count as failures.
        if n_download_failed and not fetched:
            n_pruned = len(tasks) - n_download_failed
            raise RuntimeError(
                f"No DWD files were fetched: {n_download_failed} failed and "
                f"{n_pruned} were pruned from the live feed; the feed may be "
                f"unavailable."
            )
        if n_download_failed:
            logger.warning(
                f"DWD: {n_download_failed}/{len(tasks)} files failed to "
                f"download; returning partial results."
            )

        wide, n_failed, n_tac_skipped = self._decode_paths(fetched)
        # If every fetched file was unusable, surface why rather than return an
        # empty frame. An all-TAC window is a format limitation (this source
        # decodes BUFR only); anything else is a decoder/feed problem.
        if fetched and (n_failed + n_tac_skipped) == len(fetched):
            if n_tac_skipped == len(fetched):
                raise RuntimeError(
                    f"All {len(fetched)} international files for this window are "
                    f"traditional alphanumeric (TAC) bulletins, which this source "
                    f"does not decode (BUFR only); no BUFR observations available."
                )
            raise RuntimeError(
                f"All {len(fetched)} fetched DWD files were unusable "
                f"({n_failed} failed to decode, {n_tac_skipped} unsupported TAC)."
            )
        if n_tac_skipped:
            logger.warning(
                f"DWD: {n_tac_skipped}/{len(fetched)} international files are "
                f"TAC (unsupported, BUFR only); those stations are omitted."
            )
        if n_failed:
            logger.warning(
                f"DWD: {n_failed}/{len(fetched)} files failed to decode; "
                f"returning partial results."
            )

        df = self._compile_dataframe(wide, variable_list, windows, self._lat_lon_bbox)

        df = df[[f for f in output_fields.names if f in df.columns]]
        df.attrs["source"] = self.SOURCE_ID
        return df

    async def _create_tasks(
        self, windows: list[tuple[datetime, datetime]]
    ) -> list[_DWDAsyncTask]:
        """List the feed and select files matching any requested window.

        Parameters
        ----------
        windows : list[tuple[datetime, datetime]]
            Per-requested-time ``(min, max)`` observation windows.

        Returns
        -------
        list[_DWDAsyncTask]
            One task per bulletin file whose bulletin time falls in any window
            padded by ``_LEAD_PAD`` before and ``_PUBLISH_LAG_PAD`` after (to
            catch reports published after their observation time). Files
            between disjoint windows are excluded.
        """

        def in_any_window(ftime: datetime) -> bool:
            return any(
                lo - _LEAD_PAD <= ftime <= hi + _PUBLISH_LAG_PAD for lo, hi in windows
            )

        tasks: list[_DWDAsyncTask] = []
        dir_url = self._feed_url
        try:
            names = await async_retry(
                self.fs._ls,
                dir_url,
                detail=False,
                retries=self._retries,
                backoff=1.0,
                task_timeout=60.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
        except Exception as e:
            # Distinguish "DWD unavailable" from "no observations".
            raise RuntimeError(f"Failed to list DWD feed {dir_url}: {e}") from e
        for name in names:
            if not name.endswith(".bin") or "latest" in name:
                continue
            m = _FILENAME_TIME_RE.search(name.rsplit("/", 1)[-1])
            if not m:
                continue
            try:
                ftime = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
            except ValueError:
                continue
            if in_any_window(ftime):
                url = name if name.startswith("http") else dir_url + name
                tasks.append(_DWDAsyncTask(url=url, file_time=ftime))
        return tasks

    async def _fetch_wrapper(
        self, task: _DWDAsyncTask
    ) -> tuple[datetime, str | None, bool]:
        """Fetch one bulletin file, retrying connection and I/O failures.

        Returns
        -------
        tuple[datetime, str | None, bool]
            ``(bulletin_time, cache_path, failed)``. ``cache_path`` is ``None``
            for both a pruned file (an expected live-feed miss) and a genuine
            download failure; ``failed`` is True only for the latter.
        """
        try:
            # Retry only transient connection/I/O errors. An HTTP status error
            # (e.g. 5xx, raised by fsspec/aiohttp as ClientResponseError) is not
            # in this tuple, so it fails the file immediately; widen to retry 5xx.
            path = await async_retry(
                self._fetch_remote_file,
                task.url,
                retries=self._retries,
                backoff=1.0,
                task_timeout=120.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            return task.file_time, path, False
        except Exception as e:
            # A genuine fetch failure (retries exhausted); record it as failed so
            # fetch() reports the aggregate count, and log at debug to avoid
            # per-file noise. A file pruned between listing and download raises
            # FileNotFoundError, which _fetch_remote_file already treats as an
            # expected miss (returns None, failed=False), so it never reaches here.
            logger.debug(f"Failed to fetch {task.url}: {e}")
            return task.file_time, None, True

    async def _fetch_remote_file(self, url: str) -> str | None:
        """Download a bulletin file to the local cache and return its path.

        Returns ``None`` if the file has been pruned from the live feed between
        listing and download (an expected miss, not a failure).
        """
        if self.fs is None:
            raise ValueError("Filesystem not initialized")
        cache_path = self._cache_path(url)
        if os.path.exists(cache_path):
            return cache_path
        try:
            data = await self.fs._cat_file(url)
        except FileNotFoundError:
            return None
        # Write atomically (temp then rename) to prevent partially written
        # temporary files from being exposed at the final cache path; clean up
        # the temp file if the write or rename fails.
        tmp_path = pathlib.Path(f"{cache_path}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return cache_path

    def _decode_paths(
        self, fetched: list[tuple[datetime, str]]
    ) -> tuple[pd.DataFrame, int, int]:
        """Decode and concatenate fetched BUFR bulletin files.

        Files are decoded with :func:`_decode_synop_bufr`. The international feed
        may contain traditional alphanumeric (TAC ``AAXX``/``BBXX``) bulletins
        instead of BUFR; those are recognised, counted, and skipped
        rather than silently treated as empty. Any other non-BUFR payload
        (corrupt file, error page) is counted as a decode failure. Each file's
        bulletin time is recorded in the ``_bulletin`` column so later coalescing
        can prefer the newest bulletin.

        Returns
        -------
        tuple[pd.DataFrame, int, int]
            The concatenated wide DataFrame, the number of unusable files
            (unreadable files, unrecognised non-TAC payloads, or BUFR that would
            not parse), and the number of recognised TAC files skipped.
        """
        frames = []
        n_failed = 0
        n_tac_skipped = 0
        for file_time, p in fetched:
            if self._feed == "international":
                # The international feed may contain non-BUFR bulletins; this
                # source decodes BUFR only, so sniff each file.
                # (The germany feed is always BUFR -- no need to sniff.)
                try:
                    with open(p, "rb") as fh:
                        data = fh.read()
                except OSError:
                    # An unreadable/vanished cache file is one failed payload,
                    # not a reason to abort decoding the rest of the batch.
                    n_failed += 1
                    continue
                if b"BUFR" not in data:
                    # A traditional alphanumeric (TAC) bulletin is recognised and
                    # skipped; any other non-BUFR payload (corrupt file, error
                    # page, ...) is an unrecognised decode failure.
                    if b"AAXX" in data or b"BBXX" in data:
                        n_tac_skipped += 1
                    else:
                        n_failed += 1
                    continue
            frame = _decode_synop_bufr(p)
            if frame is None:
                # Retain undecodable files to avoid repeatedly downloading
                # unsupported bulletins; they are decoded again on later requests
                # and remain until cache pruning ages them out (or manual removal).
                # Count as failed and let the caller handle partial/total failure.
                n_failed += 1
                continue
            frame["_bulletin"] = pd.Timestamp(file_time)
            frames.append(frame)
        columns = list(_META_COLUMNS) + list(_BUFR_ELEMENTS) + ["_is_wmo", "_bulletin"]
        wide = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=columns)
        )
        return wide, n_failed, n_tac_skipped

    def _compile_dataframe(
        self,
        wide: pd.DataFrame,
        variables: list[str],
        windows: list[tuple[datetime, datetime]],
        lat_lon_bbox: tuple[float, float, float, float] | None,
    ) -> pd.DataFrame:
        """Filter, derive variables, and melt to the long observation schema.

        Parameters
        ----------
        wide : pd.DataFrame
            Decoded wide DataFrame (one row per station report).
        variables : list[str]
            Requested Earth2Studio variables.
        windows : list[tuple[datetime, datetime]]
            Per-requested-time ``(min, max)`` observation windows.
        lat_lon_bbox : tuple[float, float, float, float] | None
            Optional ``(lat_min, lon_min, lat_max, lon_max)`` filter.

        Returns
        -------
        pd.DataFrame
            Long-format frame following :attr:`SCHEMA`.
        """
        # Schema-typed empty frame so dtypes match SCHEMA even when no data.
        empty = self.SCHEMA.empty_table().to_pandas()
        if wide.empty:
            return empty

        df = wide.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "lat", "lon"])

        # Reject non-finite or out-of-range coordinates before normalization so a
        # malformed BUFR value (e.g. lon 999 -> 279 after % 360) cannot become a
        # plausible location. Raw BUFR longitudes are in [-180, 180].
        df = df[
            np.isfinite(df["lat"])
            & np.isfinite(df["lon"])
            & df["lat"].between(-90.0, 90.0)
            & df["lon"].between(-180.0, 180.0)
        ]
        if df.empty:
            return empty

        # Keep records inside any requested time window.
        mask = np.zeros(len(df), dtype=bool)
        tvals = df["time"]
        for lo, hi in windows:
            mask |= (tvals >= pd.Timestamp(lo)) & (tvals <= pd.Timestamp(hi))
        df = df[mask]
        if df.empty:
            return empty

        # Longitude to [0, 360) to match Earth2Studio convention.
        df["lon"] = df["lon"] % 360.0

        # Coalesce duplicate reports (the same report retransmitted across
        # overlapping bulletin files), keeping the first non-null value per field
        # so complementary fields are not discarded. Sort newest bulletin first so
        # a corrected value from a later bulletin wins any conflict. Identity:
        #  - true WMO block+station ids group by (station, time), so a
        #    coordinate correction across bulletins still merges;
        #  - fallback station names also key on coordinates, so distinct stations
        #    sharing a name (even a 5-digit one) are not merged;
        #  - null ids fall back to (lat, lon, time).
        if "_bulletin" in df.columns:
            df = df.sort_values("_bulletin", ascending=False, kind="stable")
        has_id = df[df["station"].notna()]
        no_id = df[df["station"].isna()]
        parts = []
        if not has_id.empty:
            # Prefer the decode-time WMO flag; a fallback name can itself be five
            # digits, so fall back to a \d{5} check only for frames lacking it.
            if "_is_wmo" in has_id.columns:
                is_wmo = has_id["_is_wmo"].fillna(False).astype(bool)
            else:
                is_wmo = has_id["station"].astype(str).str.fullmatch(r"\d{5}")
            wmo_id, name_id = has_id[is_wmo], has_id[~is_wmo]
            if not wmo_id.empty:
                parts.append(
                    wmo_id.groupby(
                        ["station", "time"], as_index=False, sort=False
                    ).first()
                )
            if not name_id.empty:
                parts.append(
                    name_id.groupby(
                        ["station", "lat", "lon", "time"], as_index=False, sort=False
                    ).first()
                )
        if not no_id.empty:
            parts.append(
                no_id.groupby(
                    ["lat", "lon", "time"], dropna=False, as_index=False, sort=False
                ).first()
            )
        df = pd.concat(parts, ignore_index=True) if parts else df
        df = df.drop(columns="_bulletin", errors="ignore")

        # Filter by bbox only after coalescing, so a newer bulletin's corrected
        # coordinates (newest wins) decide whether a station is inside the box
        # rather than a superseded earlier position.
        if lat_lon_bbox is not None:
            df = self._filter_bbox(df, lat_lon_bbox)
            if df.empty:
                return empty

        # Build requested Earth2Studio variable columns. u10m/v10m are derived
        # from windSpeed/windDirection; every other id maps to a BUFR column.
        df = self._derive_wind(df, variables)
        for v in variables:
            if v in ("u10m", "v10m"):
                continue
            key, modifier = DWDSynopReportsLexicon.get_item(v)
            if key in df.columns:
                df[v] = modifier(df[key])
            else:
                df[v] = np.nan

        id_vars = [f for f in _META_COLUMNS if f in df.columns]
        value_vars = [v for v in variables if v in df.columns]
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="variable",
            value_name="observation",
        )
        df_long = df_long.dropna(subset=["observation"])
        df_long = df_long[[name for name in self.SCHEMA.names if name in df_long]]
        for field in self.SCHEMA:
            if field.name in df_long.columns and pa.types.is_floating(field.type):
                df_long[field.name] = df_long[field.name].astype(
                    field.type.to_pandas_dtype()
                )
        return df_long.reset_index(drop=True)

    @staticmethod
    def _derive_wind(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
        """Derive u10m/v10m from windSpeed and windDirection (meteorological).

        Components follow the convention that direction is where the wind blows
        *from*: ``u = -speed * sin(dir)``, ``v = -speed * cos(dir)``.
        """
        if not ({"u10m", "v10m"} & set(variables)):
            return df
        speed = pd.to_numeric(df.get("windSpeed"), errors="coerce")
        direction = pd.to_numeric(df.get("windDirection"), errors="coerce")
        rad = np.radians(direction)
        # Zero speed means zero components regardless of direction, so a calm
        # report is valid even when direction is missing (SYNOP may omit
        # direction when calm). Any other missing value leaves it undefined.
        calm = speed == 0
        valid = (speed.notna() & direction.notna()) | calm
        if "u10m" in variables:
            df["u10m"] = (-np.sin(rad) * speed).where(~calm, 0.0).where(valid, np.nan)
        if "v10m" in variables:
            df["v10m"] = (-np.cos(rad) * speed).where(~calm, 0.0).where(valid, np.nan)
        return df

    @staticmethod
    def _filter_bbox(
        df: pd.DataFrame, lat_lon_bbox: tuple[float, float, float, float]
    ) -> pd.DataFrame:
        """Filter a frame (lon already in [0, 360)) by a lat/lon bounding box."""
        lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
        lon = df["lon"] if lon_min >= 0 else ((df["lon"] + 180) % 360) - 180
        lon_in_bbox = (lon >= lon_min) & (lon <= lon_max)
        # Endpoints are inclusive, but a station on the antimeridian (stored
        # 180.0) canonicalizes to -180.0 in the [-180, 180] convention, and the
        # 0/360 seam stores as 0.0. A box ending exactly on that seam must still
        # include the station sitting on it.
        if lon_max == 360:
            lon_in_bbox |= lon == 0.0
        elif lon_min < 0 and lon_max == 180:
            lon_in_bbox |= lon == -180.0
        return df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) & lon_in_bbox]

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert a ``fields`` argument into a validated pyarrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            None (full SCHEMA), a name, list of names, or a schema.

        Returns
        -------
        pa.Schema
            Schema of the requested fields.

        Raises
        ------
        KeyError
            If a requested field is not in the class SCHEMA.
        TypeError
            If a provided schema field type does not match the class SCHEMA.
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
                if field.type != cls.SCHEMA.field(field.name).type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, expected "
                        f"{cls.SCHEMA.field(field.name).type}"
                    )
            return fields
        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate requested times against the rolling retention window.

        Parameters
        ----------
        times : list[datetime]
            Times to validate.

        Raises
        ------
        ValueError
            If a time is missing (NaT/None), more than one hour in the future
            (a grace that admits a request for a nominal synoptic hour near the
            current time), or older than the retention window.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        # Retention applies to bulletin files, but an observation can be carried
        # by a bulletin published up to _PUBLISH_LAG_PAD later, so an obs slightly
        # older than the nominal retention may still be retrievable. Extend the
        # cutoff by that pad (matching file discovery) rather than reject early;
        # if the file is already pruned, discovery just returns no data.
        oldest = now - _FEED_RETENTION - _PUBLISH_LAG_PAD
        for t in times:
            # A missing time (e.g. np.datetime64("NaT") -> None) is not valid.
            if pd.isna(t):
                raise ValueError(f"Requested time must be a valid datetime, got {t!r}.")
            # Normalize timezone-aware datetimes to naive UTC for comparison.
            if t.tzinfo is not None:
                t = t.astimezone(timezone.utc).replace(tzinfo=None)
            if t > now + timedelta(hours=1):
                raise ValueError(
                    f"Requested time {t} is more than one hour in the future; DWD "
                    f"SYNOP is a live feed (the one-hour grace admits a request for "
                    f"a nominal synoptic hour near the current time)."
                )
            if t < oldest:
                raise ValueError(
                    f"Requested time {t} is older than the ~{_FEED_RETENTION.days}"
                    f"-day DWD live feed retention. The feed retains only a rolling "
                    f"window; use an archive source (e.g. GHCNHourly) for historical "
                    f"data."
                )

    def _cache_path(self, url: str) -> str:
        """Local cache path for a remote file URL."""
        sha = hashlib.sha256(url.encode()).hexdigest()[:24]
        return os.path.join(self.cache, f"dwd_synop_reports_{sha}.bin")

    def _prune_cache(self) -> None:
        """Delete stale cache files and orphaned temp artifacts to bound growth.

        Eviction is by file/dir local modification time (not the bulletin
        timestamp) older than ``_FEED_RETENTION`` plus a one-day grace period.
        Bulletin URLs are timestamped and become unavailable once the live window
        expires, so an entry past that window is permanently stale. The grace
        period keeps an undecodable entry usable while its bulletin could still
        be re-fetched; pruning bounds growth, it does not repair a bad entry
        within the useful window (the entry clears once it ages out). Also sweeps
        ``.tmp`` files and per-instance ``tmp_*`` dirs left behind by writes or
        ``cache=False`` runs that were hard-killed before their normal cleanup.
        """
        cutoff = (
            datetime.now(timezone.utc) - (_FEED_RETENTION + timedelta(days=1))
        ).timestamp()
        root = pathlib.Path(self.cache)
        try:
            # Stale cache files and any .tmp files orphaned by a write that was
            # hard-killed before its atomic rename (the finally-cleanup only runs
            # on normal exceptions).
            for f in list(root.glob("dwd_synop_reports_*.bin")) + list(
                root.glob("dwd_synop_reports_*.tmp")
            ):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                except OSError:
                    continue
            # Stale per-instance tmp dirs left by crashed cache=False runs whose
            # __call__ cleanup never executed (only reachable when this run uses a
            # persistent cache, i.e. self._cache is True).
            for d in root.glob("tmp_*"):
                try:
                    if d.is_dir() and d.stat().st_mtime < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    continue
        except OSError:
            pass

    @property
    def cache(self) -> str:
        """Local cache directory for downloaded data."""
        cache_location = os.path.join(datasource_cache_root(), "dwd_synop_reports")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(cache_location, f"tmp_{self._tmp_cache_hash}")
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if the given date time is within the live retention window.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            If the date time is fetchable.
        """
        if isinstance(time, np.datetime64):
            # Cast straight to microseconds (which .item() renders as a datetime);
            # routing through datetime64[ns] first would overflow and wrap dates
            # outside ~1678-2262 into bogus in-window values. Dates outside
            # Python's datetime range (year 1-9999) come back as int (or None for
            # NaT), not a datetime, and are simply not fetchable.
            time = time.astype("datetime64[us]").item()
            if not isinstance(time, datetime):
                return False
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
