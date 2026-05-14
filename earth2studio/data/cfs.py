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

import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pygrib
import s3fs
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    cancellable_to_thread,
    datasource_cache_root,
    gather_with_concurrency,
    prep_forecast_inputs,
)
from earth2studio.lexicon import CFSFluxLexicon, CFSLexicon
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


# Earliest cycle present in the AWS Open Data mirror of CFSv2 operational forecasts.
# See https://registry.opendata.aws/noaa-cfs/.
_AWS_HISTORY_START = datetime(year=2023, month=4, day=22, hour=0)

# Maximum forecast lead time exposed by this data source.  Member 1 of certain
# cycles integrates out to ~9 months, but members 2-4 stop well before that.
# Capping at 180 days keeps the validation conservative; out-of-range requests
# raise rather than wait for a 404 from the upstream store.
_MAX_LEAD_HOURS = 180 * 24


@dataclass
class CFSAsyncTask:
    """Async fetch request for a single (time, lead_time, variable) tuple.

    A vector grib record packs both `UGRD` and `VGRD` into the same byte range;
    in that case the two siblings share `cfs_byte_offset` / `cfs_byte_length`
    and disambiguate via `cfs_submsg_index` (1-based pygrib message index
    within the cached byte-range file).
    """

    data_array_indices: tuple[int, int, int]
    cfs_file_uri: str
    cfs_byte_offset: int
    cfs_byte_length: int
    cfs_submsg_index: int
    cfs_modifier: Callable


class CFS_FX:
    """NCEP Climate Forecast System v2 (CFSv2) pressure-level forecast source.

    CFSv2 is NCEP's operational coupled atmosphere-ocean-land-cryosphere
    forecast system. Each 6-hour cycle launches four ensemble members (1-4)
    that integrate forward in 6-hour steps; member 1 of selected cycles runs
    out to roughly 9 months. This data source exposes the `pgbf` product
    (pressure-level atmosphere, 1 degree regular lat-lon, 181 x 360).

    Parameters
    ----------
    member : int, optional
        CFS ensemble member, one of {1, 2, 3, 4}, by default 1.
    source : str, optional
        Backing store: ``"nomads"`` (default; NCEP's official real-time
        distribution, rolling ~7-day window) or ``"aws"`` (NOAA Big Data
        Program mirror at `s3://noaa-cfs-pds/`, anonymous, archive back to
        2023-04-22).
    cache : bool, optional
        Cache data source on local memory, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default 600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 16.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.

    Note
    ----
    The NOMADS rolling window keeps roughly the last seven days of cycles
    online; for older initial conditions use ``source="aws"``. Members 2-4
    integrate to substantially shorter lead times than member 1; the lead-time
    validator caps requests at 180 days regardless of member, but requests
    past a given member's actual horizon will raise `FileNotFoundError` at
    fetch time.

    Note
    ----
    Additional information on the data repository:

    - https://www.nco.ncep.noaa.gov/pmb/products/cfs/
    - https://registry.opendata.aws/noaa-cfs/
    - https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    # File prefix used for grib filenames inside the 6hrly_grib_NN/ directory.
    CFS_PRODUCT = "pgbf"

    # 1 degree regular lat-lon grid: 181 x 360 (90..-90 N->S, 0..359 E).
    CFS_LAT = np.linspace(90, -90, 181)
    CFS_LON = np.linspace(0, 359, 360)

    CFS_AWS_BUCKET = "noaa-cfs-pds"
    CFS_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod"

    CFS_MEMBERS = (1, 2, 3, 4)
    CFS_SOURCES = ("nomads", "aws")

    # Generous safety bound on a single grib message byte range; CFS pgbf
    # records (181 x 360 floats, simple-packed) are typically <2 MB.
    MAX_BYTE_SIZE = 5_000_000

    LEXICON: Any = CFSLexicon

    def __init__(
        self,
        member: int = 1,
        source: str = "nomads",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ):
        if member not in self.CFS_MEMBERS:
            raise ValueError(
                f"Invalid CFS member {member}; must be one of {self.CFS_MEMBERS}"
            )
        if source not in self.CFS_SOURCES:
            raise ValueError(
                f"Invalid CFS source {source!r}; must be one of {self.CFS_SOURCES}"
            )

        self._member = member
        self._source = source
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        if source == "nomads":
            self.uri_prefix = self.CFS_NOMADS_BASE

            def _history_range(time: datetime) -> None:
                # NOMADS keeps roughly the last seven cycles online; allow a
                # one-day pad so cycles published just under the wire still
                # validate.
                if time + timedelta(days=8) < datetime.now(timezone.utc).replace(
                    tzinfo=None
                ):
                    raise ValueError(
                        f"Requested date time {time} is outside the NOMADS "
                        "rolling window for CFS; use source='aws' for older "
                        "initial conditions."
                    )

        else:  # aws
            self.uri_prefix = self.CFS_AWS_BUCKET

            def _history_range(time: datetime) -> None:
                if time < _AWS_HISTORY_START:
                    raise ValueError(
                        f"Requested date time {time} is before CFS AWS "
                        f"archive start ({_AWS_HISTORY_START.isoformat()})."
                    )

        self._history_range = _history_range

        # Filesystem is lazily initialised inside the event loop.
        self.fs: s3fs.S3FileSystem | HTTPFileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialisation of the fsspec backend.

        Note
        ----
        Async fsspec expects initialisation inside the execution loop.
        """
        if self._source == "aws":
            self.fs = s3fs.S3FileSystem(
                anon=True,
                client_kwargs={},
                asynchronous=True,
                skip_instance_cache=True,
            )
        else:
            self.fs = HTTPFileSystem(asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve CFS forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initial-condition timestamps to return data for (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch (6-hour increments).
        variable : str | list[str] | VariableArray
            Variable identifier(s). Must be in the source's lexicon.

        Returns
        -------
        xr.DataArray
            CFS forecast data array with dimensions
            ``[time, lead_time, variable, lat, lon]``.
        """
        try:
            xr_array = _sync_async(
                self.fetch, time, lead_time, variable, timeout=self.async_timeout
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)
        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get CFS forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initial-condition timestamps to return data for (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch (6-hour increments).
        variable : str | list[str] | VariableArray
            Variable identifier(s). Must be in the source's lexicon.

        Returns
        -------
        xr.DataArray
            CFS forecast data array.
        """
        if self.fs is None:
            await self._async_init()

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        # s3fs needs an explicit aiohttp session for parallel fetches; the
        # HTTPFileSystem manages its own session lazily and does not accept
        # the s3fs ``refresh`` kwarg, so we only set up a session for s3.
        session = None
        if isinstance(self.fs, s3fs.S3FileSystem):
            session = await self.fs.set_session(refresh=True)

        try:
            # NaN-initialise so variables that are not resolved against the
            # grib index surface as detectable missing values instead of
            # arbitrary memory.
            xr_array = xr.DataArray(
                data=np.full(
                    (
                        len(time),
                        len(lead_time),
                        len(variable),
                        len(self.CFS_LAT),
                        len(self.CFS_LON),
                    ),
                    np.nan,
                ),
                dims=["time", "lead_time", "variable", "lat", "lon"],
                coords={
                    "time": time,
                    "lead_time": lead_time,
                    "variable": variable,
                    "lat": self.CFS_LAT,
                    "lon": self.CFS_LON,
                },
            )

            tasks = await self._create_tasks(time, lead_time, variable)
            coros = [self.fetch_wrapper(task, xr_array=xr_array) for task in tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=120.0,
                desc="Fetching CFS data",
                verbose=(not self._verbose),
            )
        finally:
            if session is not None:
                await session.close()

        return xr_array

    async def _create_tasks(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> list[CFSAsyncTask]:
        """Build the list of byte-range fetch tasks for the requested data.

        Parameters
        ----------
        time : list[datetime]
            Initial-condition timestamps.
        lead_time : list[timedelta]
            Forecast lead times.
        variable : list[str]
            Variable identifiers to fetch.

        Returns
        -------
        list[CFSAsyncTask]
            One task per (time, lead_time, variable) tuple resolved against
            the corresponding grib `.idx`.
        """
        tasks: list[CFSAsyncTask] = []

        # Fetch every required .idx in parallel up front.  Most CFS variable
        # records live in the same grib file (one file per IC x lead_time), so
        # we cache them in a dict keyed by URI.  Bound the index fetches by
        # the configured ``async_workers`` budget rather than launching all
        # of them simultaneously through a bare ``tqdm.gather``.
        idx_uris = [self._grib_index_uri(t, lt) for t in time for lt in lead_time]
        idx_results = await gather_with_concurrency(
            [self._fetch_index(uri) for uri in idx_uris],
            max_workers=self._async_workers,
            task_timeout=60.0,
            desc="Fetching CFS index files",
            verbose=True,
        )
        idx_by_uri = dict(zip(idx_uris, idx_results))

        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                index_table = idx_by_uri[self._grib_index_uri(t, lt)]
                for k, v in enumerate(variable):
                    try:
                        cfs_name_str, modifier = self.LEXICON[v]
                    except KeyError as e:
                        logger.error(
                            f"Variable id {v} not found in {self.LEXICON.__name__}"
                        )
                        raise e

                    # Entries are "<product>::<param>::<level>"; the product
                    # segment is informational (file prefix is set per-class)
                    # while ``<param>::<level>`` is the key used to look up the
                    # byte range from the grib index.
                    _, param_name, level = cfs_name_str.split("::", 2)
                    idx_key = f"{param_name}::{level}"

                    record: tuple[int, int, int] | None = None
                    for key, value in index_table.items():
                        # key is "<recno>::<param>::<level>"; substring match
                        # tolerates the recno prefix and matches both scalar
                        # records and decimal-suffixed vector siblings.
                        if idx_key in key:
                            record = value
                            break

                    if record is None:
                        logger.warning(
                            f"Variable {v} not found in index for {t} at {lt}; "
                            "values will be left unset."
                        )
                        continue

                    byte_offset, byte_length, submsg_index = record
                    tasks.append(
                        CFSAsyncTask(
                            data_array_indices=(i, j, k),
                            cfs_file_uri=self._grib_uri(t, lt),
                            cfs_byte_offset=byte_offset,
                            cfs_byte_length=byte_length,
                            cfs_submsg_index=submsg_index,
                            cfs_modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: CFSAsyncTask,
        xr_array: xr.DataArray,
    ) -> None:
        """Unpack a task, fetch with retry, and write into the output array."""
        out = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=60.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        i, j, k = task.data_array_indices
        xr_array[i, j, k] = out

    async def fetch_array(self, task: CFSAsyncTask) -> np.ndarray:
        """Fetch a single grib record from the remote CFS store.

        Parameters
        ----------
        task : CFSAsyncTask
            Task describing the URI, byte range, target grib short name, and
            value modifier.

        Returns
        -------
        np.ndarray
            Decoded 2-D grib field, with `task.cfs_modifier` applied.
        """
        logger.debug(
            f"Fetching CFS grib: {task.cfs_file_uri} "
            f"{task.cfs_byte_offset}-{task.cfs_byte_length}"
        )
        grib_file = await self._fetch_remote_file(
            task.cfs_file_uri,
            byte_offset=task.cfs_byte_offset,
            byte_length=task.cfs_byte_length,
        )

        # pygrib is sync-only.  Use cancellable_to_thread so a hung decode can
        # be abandoned without holding the event loop.
        values = await cancellable_to_thread(
            _decode_cfs_grib,
            grib_file,
            task.cfs_submsg_index,
            timeout=30.0,
        )
        return task.cfs_modifier(values)

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify date time validity for this CFS source.

        Checks both the universal CFS 6-hour cycle interval and the active
        backing store's history range.

        Parameters
        ----------
        times : list[datetime]
            Initial-condition date times to validate.

        Raises
        ------
        ValueError
            If a date time is not on a 6-hour CFS cycle or is outside the
            backing store's available range.
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be a 6-hour cycle for CFS"
                )
            self._history_range(time)

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify lead-time validity.

        Parameters
        ----------
        lead_times : list[timedelta]
            Forecast lead times to validate.

        Raises
        ------
        ValueError
            If a lead time is not a 6-hour multiple or exceeds the conservative
            180-day cap.
        """
        for delta in lead_times:
            if delta.total_seconds() < 0:
                raise ValueError(f"Lead time {delta} must be non-negative for CFS")
            if not delta.total_seconds() % 21600 == 0:
                raise ValueError(f"Lead time {delta} must be a 6-hour multiple for CFS")
            if delta.total_seconds() // 3600 > _MAX_LEAD_HOURS:
                raise ValueError(
                    f"Lead time {delta} exceeds the CFS data source cap of "
                    f"{_MAX_LEAD_HOURS // 24} days."
                )

    async def _fetch_index(self, index_uri: str) -> dict[str, tuple[int, int, int]]:
        """Fetch a CFS grib `.idx` file and parse it into a byte-range lookup.

        The CFS pgbf product packs `UGRD`/`VGRD` into a single vector grib
        message, which the `.idx` writes as decimal-suffixed sibling records
        (e.g. ``7.1`` for `UGRD` and ``7.2`` for `VGRD` at the same byte
        offset). The parser preserves those siblings as distinct entries that
        share a byte range; pygrib decodes the cached file as separate
        submessages and the submessage index disambiguates which component to
        return.

        Parameters
        ----------
        index_uri : str
            URI of the `.idx` file to fetch.

        Returns
        -------
        dict[str, tuple[int, int, int]]
            Mapping from `<recno>::<param>::<level>` to
            `(byte_offset, byte_length, submsg_index)`.
            `submsg_index` is the 1-based pygrib message index within the
            cached byte-range file; for scalar records this is always 1, for
            vector siblings it is parsed from the decimal suffix of `recno`.
        """
        try:
            local_path = await self._fetch_remote_file(index_uri)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The specified data index, {index_uri}, does not exist. "
                "Data seems to be missing."
            )
        with open(local_path) as fh:
            index_lines = [line.rstrip() for line in fh if line.strip()]

        # Walk the list once gathering ``(recno, offset, key)`` for each entry.
        parsed: list[tuple[str, int, str]] = []
        for line in index_lines:
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue
            recno = lsplit[0]
            offset = int(lsplit[1])
            key = f"{recno}::{lsplit[3]}::{lsplit[4]}"
            parsed.append((recno, offset, key))

        # Compute byte length using the next strictly-greater offset so that
        # vector-sibling records (same offset, decimal-suffixed recno) cover
        # the same byte range.
        index_table: dict[str, tuple[int, int, int]] = {}
        for idx, (recno, offset, key) in enumerate(parsed):
            next_offset: int | None = None
            for _next_recno, next_off, _next_key in parsed[idx + 1 :]:
                if next_off > offset:
                    next_offset = next_off
                    break
            if next_offset is None:
                # Last record: use a generous upper bound; fsspec/`_cat_file`
                # treats `end > size` as "rest of file".
                byte_length = self.MAX_BYTE_SIZE
            else:
                byte_length = next_offset - offset

            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length {byte_length} of variable {key} larger than "
                    f"safe threshold {self.MAX_BYTE_SIZE}"
                )

            # Decimal-suffixed recno (e.g. "7.2") indicates a vector sibling;
            # the digit after the dot is the 1-based pygrib submessage index
            # within the shared byte range.
            if "." in recno:
                try:
                    submsg_index = int(recno.split(".", 1)[1])
                except ValueError:
                    submsg_index = 1
            else:
                submsg_index = 1
            index_table[key] = (offset, byte_length, submsg_index)
        return index_table

    async def _fetch_remote_file(
        self, path: str, byte_offset: int = 0, byte_length: int | None = None
    ) -> str:
        """Fetch a remote file (or byte range) into the local cache.

        Parameters
        ----------
        path : str
            Remote URI (``s3://...`` style for AWS or HTTPS URL for NOMADS).
        byte_offset : int, optional
            Starting byte offset, by default 0.
        byte_length : int | None, optional
            Number of bytes to fetch, by default None (full file).

        Returns
        -------
        str
            Path to the cached file on local disk.
        """
        if self.fs is None:
            raise ValueError("File system is not initialised")

        sha = hashlib.sha256((path + str(byte_offset)).encode())
        cache_path = os.path.join(self.cache, sha.hexdigest())

        if pathlib.Path(cache_path).is_file():
            return cache_path

        end: int | None = None
        if byte_length:
            end = int(byte_offset + byte_length)

        try:
            data = await self.fs._cat_file(path, start=byte_offset, end=end)
        except FileNotFoundError as e:
            logger.error(f"Failed to download {path}: not found")
            raise e

        with open(cache_path, "wb") as fh:
            fh.write(data)
        return cache_path

    def _grib_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Build the grib file URI for a given IC and lead time."""
        valid = time + lead_time
        ic_stamp = f"{time:%Y%m%d%H}"
        valid_stamp = f"{valid:%Y%m%d%H}"
        member = f"{self._member:02d}"
        filename = f"{self.CFS_PRODUCT}{valid_stamp}.{member}.{ic_stamp}.grb2"
        cycle_dir = f"cfs.{time:%Y%m%d}/{time:%H}/6hrly_grib_{member}/{filename}"
        return self._join_uri(cycle_dir)

    def _grib_index_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Build the `.idx` URI for a given IC and lead time."""
        return self._grib_uri(time, lead_time) + ".idx"

    def _join_uri(self, suffix: str) -> str:
        """Join the source-specific URI prefix to a relative path."""
        if self._source == "nomads":
            return f"{self.uri_prefix.rstrip('/')}/{suffix}"
        # AWS: s3fs accepts ``bucket/key`` for ``_cat_file``.
        return f"{self.uri_prefix}/{suffix}"

    @property
    def cache(self) -> str:
        """Return the on-disk cache location for this instance."""
        cache_location = os.path.join(datasource_cache_root(), "cfs")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_cfs_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        member: int = 1,
    ) -> bool:
        """Check whether a given CFS initial condition is available on AWS.

        Uses the public AWS PDS bucket (the most complete public archive)
        rather than NOMADS, so this works regardless of the rolling-window
        cutoff on the NCEP server.

        Parameters
        ----------
        time : datetime | np.datetime64
            Initial-condition date time to check.
        member : int, optional
            CFS member to check, by default 1.

        Returns
        -------
        bool
            Whether the cycle directory exists in the AWS bucket.
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)
            time = time.replace(tzinfo=None)

        if member not in cls.CFS_MEMBERS:
            return False
        if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
            return False
        if time < _AWS_HISTORY_START:
            return False

        member_dir = f"{member:02d}"
        cycle_uri = (
            f"s3://{cls.CFS_AWS_BUCKET}/cfs.{time:%Y%m%d}/"
            f"{time:%H}/6hrly_grib_{member_dir}/"
        )
        fs = s3fs.S3FileSystem(anon=True)
        try:
            return fs.exists(cycle_uri)
        except OSError:
            return False


class CFS_FX_Flux(CFS_FX):
    """NCEP Climate Forecast System v2 surface-flux forecast source.

    Identical access pattern to :class:`CFS_FX` but exposes the ``flxf``
    product, which carries surface and near-surface diagnostic fields on the
    native T126 Gaussian grid (190 x 384). Use this source for surface fluxes
    (sensible/latent heat, radiation, precipitation rate), boundary-layer
    diagnostics, and cloud-cover summaries.

    Parameters
    ----------
    member : int, optional
        CFS ensemble member, one of {1, 2, 3, 4}, by default 1.
    source : str, optional
        Backing store: ``"nomads"`` (default) or ``"aws"``.
    cache : bool, optional
        Cache data source on local memory, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default 600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 16.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.

    Note
    ----
    The Gaussian latitude axis is non-uniform; values are the roots of the
    Legendre polynomial of degree 190 mapped through ``arcsin``. Downstream
    consumers expecting an equiangular grid will need to regrid first.

    Note
    ----
    Additional information on the data repository:

    - https://www.nco.ncep.noaa.gov/pmb/products/cfs/
    - https://registry.opendata.aws/noaa-cfs/

    Badges
    ------
    region:global dataclass:simulation product:precip product:land product:atmos
    """

    CFS_PRODUCT = "flxf"

    # T126 reduced Gaussian grid published as a regular Gaussian field with
    # 190 latitudes and 384 longitudes.  Latitudes are the (descending) roots
    # of the degree-190 Legendre polynomial; longitudes are equispaced.
    _NODES, _ = np.polynomial.legendre.leggauss(190)
    CFS_LAT = np.degrees(np.arcsin(_NODES))[::-1].copy()
    CFS_LON = np.linspace(0, 360 - 360 / 384, 384)

    LEXICON: Any = CFSFluxLexicon


def _decode_cfs_grib(grib_file: str, submsg_index: int) -> np.ndarray:
    """Decode a single CFS grib submessage from a cached byte-range file.

    Parameters
    ----------
    grib_file : str
        Path to the cached grib file on local disk. The file may contain a
        single message or, for vector wind packing, multiple submessages.
    submsg_index : int
        1-based pygrib message index to extract. For scalar records this is
        always 1; for vector records (e.g. `UGRD`/`VGRD` siblings) it selects
        the requested wind component.

    Returns
    -------
    np.ndarray
        Decoded 2-D field as a NumPy array.
    """
    try:
        grbs = pygrib.open(grib_file)
    except Exception as e:
        logger.error(f"Failed to open grib file {grib_file}")
        raise e
    try:
        # pygrib.open() uses 1-based message indexing.
        return np.asarray(grbs[submsg_index].values)
    except Exception as e:
        logger.error(
            f"Failed to read grib file {grib_file} at submessage {submsg_index}"
        )
        raise e
    finally:
        grbs.close()
