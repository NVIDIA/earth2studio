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
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pygrib
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.cfs import CFS_FX, CFS_FX_Flux
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


# Period of record exposed by the NCEI CFS reforecast 9-month-runs archive.
# Earliest cycle in the bucket is 1981-12-12; the series ends 2011-03-27.
_REFORECAST_HISTORY_START = datetime(year=1981, month=12, day=12, hour=0)
_REFORECAST_HISTORY_END = datetime(year=2011, month=3, day=27, hour=18)

# Forecast horizon: 9 months ~= 270 days at 6-h intervals.
_MAX_LEAD_HOURS = 270 * 24

# Map the NCEP wgrib2 short names embedded in the CFS lexicons to the
# corresponding ecCodes / pygrib ``parameterName`` strings.  Covers exactly
# the parameters referenced by CFSLexicon and CFSFluxLexicon as of writing;
# new lexicon entries must be added here as well.
_NCEP_TO_PARAM_NAME: dict[str, str] = {
    "DPT": "Dew point temperature",
    "HGT": "Geopotential height",
    "PRATE": "Precipitation rate",
    "PRES": "Pressure",
    "PRMSL": "Pressure reduced to MSL",
    "PWAT": "Precipitable water",
    "RH": "Relative humidity",
    "SNOD": "Snow depth",
    "SPFH": "Specific humidity",
    "TCDC": "Total cloud cover",
    "TMP": "Temperature",
    "UGRD": "u-component of wind",
    "VGRD": "v-component of wind",
}

# Static level-description -> (typeOfLevel, level) for non-pressure levels.
_LEVEL_DESC_TO_TYPE: dict[str, tuple[str, int]] = {
    "mean sea level": ("meanSea", 0),
    "surface": ("surface", 0),
    "entire atmosphere (considered as a single layer)": (
        "atmosphereSingleLayer",
        0,
    ),
    "high cloud layer": ("highCloudLayer", 0),
    "middle cloud layer": ("middleCloudLayer", 0),
    "low cloud layer": ("lowCloudLayer", 0),
}

# Pressure-level descriptions look like ``"500 mb"`` and height-above-ground
# levels look like ``"2 m above ground"``; both share a numeric prefix.
_PRESSURE_LEVEL_RE = re.compile(r"^(\d+)\s+mb$")
_HEIGHT_AGL_RE = re.compile(r"^(\d+)\s+m above ground$")


def _resolve_pygrib_filter(level_desc: str) -> tuple[str, int]:
    """Translate a lexicon level description into pygrib filter args.

    Parameters
    ----------
    level_desc : str
        Level description string from the lexicon entry (the third
        ``::``-separated field), e.g. ``"500 mb"``, ``"2 m above ground"``,
        or ``"surface"``.

    Returns
    -------
    tuple[str, int]
        ``(typeOfLevel, level)`` arguments suitable for ``pygrib.select``.

    Raises
    ------
    KeyError
        If ``level_desc`` is not a recognised level pattern.
    """
    static = _LEVEL_DESC_TO_TYPE.get(level_desc)
    if static is not None:
        return static
    m = _PRESSURE_LEVEL_RE.match(level_desc)
    if m is not None:
        return ("isobaricInhPa", int(m.group(1)))
    m = _HEIGHT_AGL_RE.match(level_desc)
    if m is not None:
        return ("heightAboveGround", int(m.group(1)))
    raise KeyError(f"Unrecognised CFS lexicon level description: {level_desc!r}")


@dataclass
class CFSReforecastAsyncTask:
    """Async fetch request for a single (time, lead_time) grib file.

    The reforecast archive has no `.idx` companions so each task downloads the
    whole grib file once and decodes every requested variable from it,
    avoiding redundant downloads when multiple variables share an (IC, lead)
    pair.
    """

    # (time_idx, lead_time_idx) pair indexing into the output xr.DataArray.
    data_array_indices: tuple[int, int]
    grib_uri: str
    # List of (variable_idx, param_name, type_of_level, level, modifier)
    # tuples giving every variable to extract from this grib file.
    variables: list[tuple[int, str, str, int, Callable]]


class CFS_Reforecast_FX:
    """NCEP CFSv2 6-hourly 9-month reforecast (pressure-level product).

    The CFS reforecast is a 30-year offline integration of CFSv2 produced by
    NCEP to support training and bias-correction of the operational system.
    Cycles are launched every 5 days from 1981-12-12 to 2011-03-27; each
    5-day cycle directory contains four 6-hourly initial conditions
    (00/06/12/18 UTC) integrated forward to roughly 9 months in 6-hour
    steps. Only ensemble member 01 of the operational 1-4 set is published
    in this 9-month series.

    This class exposes the pressure-level (``pgbf``) product on the same
    1 degree regular lat-lon grid (181 x 360) as the operational
    :class:`~earth2studio.data.CFS_FX`. Variable definitions are taken
    verbatim from :class:`~earth2studio.lexicon.CFSLexicon`, which the
    grib2 inventories match.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True. Cached files
        are large (~22 MB per (IC, lead) pgbf file) and are reused across
        variable requests for the same file.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default
        600.
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks, by default 16.
        Each task downloads one whole grib file.
    retries : int, optional
        Number of retry attempts per failed fetch task with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests. Unlike the operational
    archive, the reforecast bucket has no `.idx` companions; every variable
    request triggers a full grib-file download (~22 MB) if not already
    cached.

    Note
    ----
    The reforecast `9-month` series only publishes ensemble member 01; no
    ``member`` constructor argument is exposed.

    Note
    ----
    Additional information on the data repository:

    - https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system
    - https://www.ncei.noaa.gov/data/climate-forecast-system/access/reforecast/6-hourly-by-pressure-level-9-month-runs/
    - https://www.ncei.noaa.gov/data/climate-forecast-system/access/reforecast/6-hourly-flux-9-month-runs/

    Badges
    ------
    region:global dataclass:reanalysis product:wind product:temp product:atmos
    """

    # NCEI HTTPS base path; both pgbf and flxf live under the same parent
    # but differ in the product subdir.
    CFS_NCEI_BASE = (
        "https://www.ncei.noaa.gov/data/climate-forecast-system/access/reforecast"
    )

    # File prefix and product subdirectory for this product.  Overridden in
    # :class:`CFS_Reforecast_FX_Flux`.
    CFS_PRODUCT = "pgbf"
    CFS_NCEI_SUBDIR = "6-hourly-by-pressure-level-9-month-runs"

    # 1 degree regular lat-lon grid, identical to the operational pgbf.
    CFS_LAT = CFS_FX.CFS_LAT
    CFS_LON = CFS_FX.CFS_LON

    LEXICON: Any = CFSLexicon

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ):
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        # Filesystem is lazily initialised inside the event loop.
        self.fs: HTTPFileSystem | None = None

    async def _async_init(self) -> None:
        """Async initialisation of the HTTPS fsspec backend.

        Note
        ----
        Async fsspec expects initialisation inside the execution loop.
        """
        self.fs = HTTPFileSystem(asynchronous=True)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve CFS reforecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initial-condition timestamps to return data for (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch (6-hour increments, 0 to ~9 months).
        variable : str | list[str] | VariableArray
            Variable identifier(s). Must be in the source's lexicon.

        Returns
        -------
        xr.DataArray
            CFS reforecast data array with dimensions
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
        """Async fetch of CFS reforecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initial-condition timestamps (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Variable identifier(s). Must be in the source's lexicon.

        Returns
        -------
        xr.DataArray
            CFS reforecast data array.
        """
        if self.fs is None:
            await self._async_init()

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        xr_array = xr.DataArray(
            data=np.empty(
                (
                    len(time),
                    len(lead_time),
                    len(variable),
                    len(self.CFS_LAT),
                    len(self.CFS_LON),
                )
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
            task_timeout=180.0,
            desc="Fetching CFS reforecast data",
            verbose=(not self._verbose),
        )
        return xr_array

    async def _create_tasks(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> list[CFSReforecastAsyncTask]:
        """Build one async task per (time, lead_time) pair.

        Each task carries the list of requested variables so the grib file is
        downloaded and decoded exactly once per (IC, lead) pair regardless of
        how many variables are requested for it.

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
        list[CFSReforecastAsyncTask]
            One task per (time, lead_time) pair.
        """
        # Resolve every variable once up front; raises KeyError fast on a bad name.
        resolved: list[tuple[int, str, str, int, Callable]] = []
        for k, v in enumerate(variable):
            try:
                cfs_name_str, modifier = self.LEXICON[v]
            except KeyError as e:
                logger.error(
                    f"Variable id {v} not found in {self.LEXICON.__name__}"
                )
                raise e
            _, param_ncep, level_desc = cfs_name_str.split("::", 2)
            if param_ncep not in _NCEP_TO_PARAM_NAME:
                raise KeyError(
                    f"NCEP parameter {param_ncep!r} (variable {v!r}) has no "
                    "pygrib parameterName translation for the CFS reforecast"
                )
            type_of_level, level = _resolve_pygrib_filter(level_desc)
            resolved.append(
                (k, _NCEP_TO_PARAM_NAME[param_ncep], type_of_level, level, modifier)
            )

        tasks: list[CFSReforecastAsyncTask] = []
        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                tasks.append(
                    CFSReforecastAsyncTask(
                        data_array_indices=(i, j),
                        grib_uri=self._grib_uri(t, lt),
                        variables=resolved,
                    )
                )
        return tasks

    async def fetch_wrapper(
        self,
        task: CFSReforecastAsyncTask,
        xr_array: xr.DataArray,
    ) -> None:
        """Unpack a task, fetch with retry, and write all variables for it."""
        results = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=120.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        i, j = task.data_array_indices
        for k, values in results:
            xr_array[i, j, k] = values

    async def fetch_array(
        self, task: CFSReforecastAsyncTask
    ) -> list[tuple[int, np.ndarray]]:
        """Download a reforecast grib file and decode every requested variable.

        Parameters
        ----------
        task : CFSReforecastAsyncTask
            Task with URI and variable list.

        Returns
        -------
        list[tuple[int, np.ndarray]]
            List of ``(variable_idx, ndarray)`` pairs aligned with the
            requested variables on the task.
        """
        logger.debug(f"Fetching CFS reforecast grib: {task.grib_uri}")
        grib_file = await self._fetch_remote_file(task.grib_uri)

        # pygrib is sync-only; run in a thread with a generous timeout because
        # we are decoding many variables from one file.
        return await cancellable_to_thread(
            _decode_cfs_reforecast_grib,
            grib_file,
            task.variables,
            timeout=120.0,
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify date time validity for the CFS reforecast archive.

        Parameters
        ----------
        times : list[datetime]
            Initial-condition date times to validate.

        Raises
        ------
        ValueError
            If a date time is not on a 6-hour CFS cycle or is outside the
            archive's available range.
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be a 6-hour cycle for CFS"
                )
            if time < _REFORECAST_HISTORY_START or time > _REFORECAST_HISTORY_END:
                raise ValueError(
                    f"Requested date time {time} is outside the CFS reforecast "
                    f"9-month-runs archive range "
                    f"[{_REFORECAST_HISTORY_START.isoformat()}, "
                    f"{_REFORECAST_HISTORY_END.isoformat()}]."
                )

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
            If a lead time is not a 6-hour multiple, is negative, or
            exceeds the 9-month (270-day) forecast horizon.
        """
        for delta in lead_times:
            if delta.total_seconds() < 0:
                raise ValueError(f"Lead time {delta} must be non-negative for CFS")
            if not delta.total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Lead time {delta} must be a 6-hour multiple for CFS"
                )
            if delta.total_seconds() // 3600 > _MAX_LEAD_HOURS:
                raise ValueError(
                    f"Lead time {delta} exceeds the CFS reforecast 9-month cap"
                )

    async def _fetch_remote_file(self, uri: str) -> str:
        """Fetch a full grib file via HTTPS into the local cache.

        Parameters
        ----------
        uri : str
            Full HTTPS URL to the grib file.

        Returns
        -------
        str
            Path to the cached file on local disk.
        """
        if self.fs is None:
            raise ValueError("File system is not initialised")

        cache_path = os.path.join(
            self.cache, hashlib.sha256(uri.encode()).hexdigest()
        )
        if pathlib.Path(cache_path).is_file():
            return cache_path

        try:
            data = await self.fs._cat_file(uri)
        except FileNotFoundError as e:
            logger.error(
                f"CFS reforecast file not found at {uri}: cycle/lead may not "
                "fall on the 5-day reforecast schedule."
            )
            raise e

        with open(cache_path, "wb") as fh:
            fh.write(data)
        return cache_path

    def _grib_uri(self, time: datetime, lead_time: timedelta) -> str:
        """Build the HTTPS URI for a (time, lead_time) grib file."""
        valid = time + lead_time
        ic_stamp = f"{time:%Y%m%d%H}"
        valid_stamp = f"{valid:%Y%m%d%H}"
        # Cycle directory is the IC date (not lead-time date); the cycle calendar
        # is every 5 days within a year, with all 4 init hours co-located.
        cycle_date = f"{time:%Y%m%d}"
        filename = f"{self.CFS_PRODUCT}{valid_stamp}.01.{ic_stamp}.grb2"
        return (
            f"{self.CFS_NCEI_BASE}/{self.CFS_NCEI_SUBDIR}/"
            f"{time:%Y}/{time:%Y%m}/{cycle_date}/{filename}"
        )

    @property
    def cache(self) -> str:
        """Return the on-disk cache location for this instance."""
        cache_location = os.path.join(datasource_cache_root(), "cfs_reforecast")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_cfs_reforecast_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether a CFS reforecast initial condition is available.

        Performs offline validity checks (6-hour cycle, history range); the
        every-5-days cycle calendar is not enforced here because it shifts
        slightly across leap-year boundaries.  Off-schedule requests are
        surfaced as ``FileNotFoundError`` at fetch time.

        Parameters
        ----------
        time : datetime | np.datetime64
            Initial-condition date time to check.

        Returns
        -------
        bool
            Whether the offline checks pass.
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)
            time = time.replace(tzinfo=None)
        if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
            return False
        if time < _REFORECAST_HISTORY_START or time > _REFORECAST_HISTORY_END:
            return False
        return True


class CFS_Reforecast_FX_Flux(CFS_Reforecast_FX):
    """NCEP CFSv2 6-hourly 9-month reforecast (surface-flux product).

    Same archive and access pattern as :class:`CFS_Reforecast_FX` but
    exposes the ``flxf`` product on the native T126 Gaussian grid
    (190 x 384). Variable inventory mirrors
    :class:`~earth2studio.lexicon.CFSFluxLexicon` (which the reforecast
    grib2 inventories match).

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation, by default
        600.
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
    Additional information on the data repository:

    - https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system
    - https://www.ncei.noaa.gov/data/climate-forecast-system/access/reforecast/6-hourly-flux-9-month-runs/

    Badges
    ------
    region:global dataclass:reanalysis product:precip product:land product:atmos
    """

    CFS_PRODUCT = "flxf"
    CFS_NCEI_SUBDIR = "6-hourly-flux-9-month-runs"

    # T126 Gaussian grid, identical to the operational flxf.
    CFS_LAT = CFS_FX_Flux.CFS_LAT
    CFS_LON = CFS_FX_Flux.CFS_LON

    LEXICON: Any = CFSFluxLexicon


def _decode_cfs_reforecast_grib(
    grib_file: str,
    variables: list[tuple[int, str, str, int, Callable]],
) -> list[tuple[int, np.ndarray]]:
    """Decode every requested variable from a single CFS reforecast grib file.

    Parameters
    ----------
    grib_file : str
        Path to the cached grib file on local disk.
    variables : list of (var_idx, parameterName, typeOfLevel, level, modifier)
        Variables to extract from this grib file. Each tuple is processed
        independently against ``pygrib.select``; missing variables emit a
        warning but do not abort the others.

    Returns
    -------
    list[tuple[int, np.ndarray]]
        ``(variable_idx, ndarray)`` pairs in the order the variables appear
        in ``variables``. Variables that fail to match a grib record are
        omitted, leaving the corresponding slot of the output xr.DataArray
        in its default-initialised state.
    """
    try:
        grbs = pygrib.open(grib_file)
    except Exception as e:
        logger.error(f"Failed to open grib file {grib_file}")
        raise e
    try:
        out: list[tuple[int, np.ndarray]] = []
        for var_idx, param_name, type_of_level, level, modifier in variables:
            matches = grbs.select(
                parameterName=param_name,
                typeOfLevel=type_of_level,
                level=level,
            )
            if not matches:
                logger.warning(
                    f"CFS reforecast grib {grib_file} has no record matching "
                    f"parameterName={param_name!r} typeOfLevel={type_of_level!r} "
                    f"level={level}; variable will be left unset."
                )
                continue
            out.append((var_idx, modifier(np.asarray(matches[0].values))))
        return out
    finally:
        grbs.close()
