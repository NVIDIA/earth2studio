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
import io
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon.opera import OPERALexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import h5py
except ImportError:
    OptionalDependencyFailure("data")
    h5py = None  # type: ignore[assignment]


# CloudFerro S3 base URL for the OPERA composite archive (public, no auth)
_ARCHIVE_BASE = "https://s3.waw3-1.cloudferro.com/openradar-archive"

# First date in the OPERA archive
_MIN_DATE = datetime(2011, 10, 1)

# ODYSSEY-era composite interval: 15 minutes
_INTERVAL_SECONDS = 900

# CIRRUS/NIMBUS production began 2024-07: files renamed DBZH_QIND → DBZH etc.,
# resolution raised to 1 km for DBZH, and update cycle shortened to 5 minutes.
# The ODIM HDF5 structure also differs between eras (see _decode_odim_h5).
# Note: pixel resolution varies by product even within an era (e.g. CIRRUS era
# DBZH → 1 km / 3800×4400; RATE and ACRR → 2 km / 1900×2200).
_CIRRUS_START = datetime(2024, 7, 1)

# CIRRUS-era composite interval: 5 minutes
_CIRRUS_INTERVAL_SECONDS = 300

# PROJ string from ODIM HDF5 "where/projdef" attribute (WGS84 ellipsoid).
_PROJDEF = (
    "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +ellps=WGS84"
)

# Spatial extent of the OPERA LAEA grid, constant across all eras and products.
# Pixel scale = EXTENT / grid_size_in_that_dimension.
_GRID_EXTENT_X = 3_800_000.0  # metres east–west
_GRID_EXTENT_Y = 4_400_000.0  # metres north–south

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _compute_opera_latlon(xsize: int, ysize: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2-D lat/lon coordinate arrays for an OPERA LAEA grid.

    Pixel (row, col) centres are projected from LAEA to geographic coordinates
    using pyproj.  Row 0 is the northernmost row (largest y in LAEA space).
    Pixel scale is derived from the known total spatial extent and the grid
    dimensions, so this works for any OPERA era and product.

    Parameters
    ----------
    xsize : int
        Grid width (columns).
    ysize : int
        Grid height (rows).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lat, lon)`` arrays of shape ``(ysize, xsize)``, dtype float32.
    """
    import pyproj

    xscale = _GRID_EXTENT_X / xsize
    yscale = _GRID_EXTENT_Y / ysize
    transformer = pyproj.Transformer.from_crs(_PROJDEF, "EPSG:4326", always_xy=True)
    cols = np.arange(xsize, dtype=np.float64)
    rows = np.arange(ysize, dtype=np.float64)
    # Centre of each pixel in projected metres.
    # x increases left→right; y increases bottom→top, so row 0 has the
    # largest y value (north).
    # LL corner in this projected CRS is always at (0, -_GRID_EXTENT_Y).
    # x increases left→right from 0; y increases bottom→top from -_GRID_EXTENT_Y.
    x = cols * xscale + xscale / 2.0
    y = (ysize - 1 - rows) * yscale + yscale / 2.0 - _GRID_EXTENT_Y
    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)
    return lat.astype(np.float32), lon.astype(np.float32)


# ---------------------------------------------------------------------------
# ODIM HDF5 decoding helpers
# ---------------------------------------------------------------------------


def _odim_attrs(group: Any) -> dict[str, Any]:
    """Return a plain-Python dict of scalar HDF5 attributes."""
    result: dict[str, Any] = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.generic):
            v = v.item()
        result[k] = v
    return result


def _sort_odim_groups(parent: Any, prefix: str) -> list[str]:
    """Return numerically sorted names of ODIM sub-groups with given prefix."""

    def _key(name: str) -> tuple[int, str]:
        suffix = name.removeprefix(prefix)
        return (int(suffix) if suffix.isdigit() else 10**9, name)

    return sorted(
        [n for n, item in parent.items() if n.startswith(prefix) and hasattr(item, "items")],
        key=_key,
    )


def _apply_linear_scaling(raw: np.ndarray, what: dict[str, Any]) -> np.ndarray:
    """Apply ODIM gain/offset scaling and replace nodata/undetect with NaN."""
    gain = float(what.get("gain", 1.0))
    offset = float(what.get("offset", 0.0))
    nodata = what.get("nodata")
    undetect = what.get("undetect")
    result = raw.astype(np.float32) * gain + offset
    if nodata is not None:
        result[raw == nodata] = np.nan
    if undetect is not None:
        result[raw == undetect] = np.nan
    return result


def _decode_odim_h5(data_bytes: bytes, odim_quantity: str) -> np.ndarray:
    """Decode an in-memory ODIM HDF5 file and return a 2-D float32 array.

    Handles two ODIM HDF5 layout variants encountered in the OPERA archive:

    * **CIRRUS / NIMBUS era (ODIM 2.4+):** ``dataset*/data*/what/@quantity`` —
      the quantity tag sits inside each ``data*`` sub-group.
    * **ODYSSEY era (ODIM 2.0):** ``dataset*/what/@quantity`` — the quantity
      tag is at the *dataset* level; the data array lives in ``dataset*/data1``.

    Parameters
    ----------
    data_bytes : bytes
        Raw bytes of the ODIM HDF5 file.
    odim_quantity : str
        ODIM quantity code to extract (e.g. ``"DBZH"``).

    Returns
    -------
    np.ndarray
        Decoded 2-D float32 array whose shape matches the file's grid.

    Raises
    ------
    ValueError
        If the requested quantity is not found in the file.
    """
    with h5py.File(io.BytesIO(data_bytes), "r") as fh:
        for ds_name in _sort_odim_groups(fh, "dataset"):
            ds = fh[ds_name]

            # CIRRUS/NIMBUS era: quantity in data*/what (standard ODIM 2.4+)
            for da_name in _sort_odim_groups(ds, "data"):
                da = ds[da_name]
                if "what" not in da or "data" not in da:
                    continue
                what = _odim_attrs(da["what"])
                if str(what.get("quantity", "")).upper() != odim_quantity.upper():
                    continue
                return _apply_linear_scaling(da["data"][:], what)

            # ODYSSEY era: quantity in dataset*/what, data array in dataset*/data1
            if "what" in ds:
                what = _odim_attrs(ds["what"])
                if str(what.get("quantity", "")).upper() == odim_quantity.upper():
                    da_names = _sort_odim_groups(ds, "data")
                    if da_names and "data" in ds[da_names[0]]:
                        return _apply_linear_scaling(ds[da_names[0]]["data"][:], what)

    raise ValueError(f"ODIM quantity '{odim_quantity}' not found in HDF5 file")


# ---------------------------------------------------------------------------
# Async task
# ---------------------------------------------------------------------------


@dataclass
class _OPERAAsyncTask:
    """A single async fetch unit (one time × variable combination)."""

    time_idx: int
    var_idx: int
    url: str
    odim_quantity: str
    modifier: Callable


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


@check_optional_dependencies()
class OPERA:
    """EUMETNET OPERA European weather radar composite data source.

    Provides access to the pan-European OPERA composite radar products
    (reflectivity, rain rate, hourly accumulation) from the EUMETNET Open
    Radar Data (ORD) archive hosted on CloudFerro S3.  Data is stored in ODIM
    HDF5 format on a Lambert Equal-Area (LAEA) projection covering roughly
    70°N–32°N, 30°W–62°E.

    The pixel resolution and ODIM HDF5 structure differ between production eras:

    * **ODYSSEY era (before 2024-07-01):** 2 km per pixel, 1900 × 2200 grid,
      ODIM HDF5 version 2.0.  The quantity tag lives at the *dataset* level.
    * **CIRRUS / NIMBUS era (from 2024-07-01 onward):** ODIM version 2.4.
      Resolution varies by product: DBZH is 1 km (3800 × 4400); RATE and ACRR
      remain at 2 km (1900 × 2200).

    All grids share the same spatial extent (3800 km × 4400 km) and LAEA
    projection origin.  If a single call requests variables with different
    pixel resolutions, a :exc:`ValueError` is raised — request each resolution
    group separately.

    The output ``xr.DataArray`` uses dimensions ``(time, variable, y, x)``
    where ``y`` and ``x`` are pixel indices.  2-D geographic coordinates are
    attached as non-dimension coords ``_lat`` and ``_lon`` (shape
    ``(y, x)``), matching the pattern used by
    :class:`earth2studio.data.GOES` and
    :class:`earth2studio.data.HimawariAHI`.

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded HDF5 files locally, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch, by default 600.
    async_workers : int, optional
        Maximum concurrent async fetch tasks, by default 16.
    retries : int, optional
        Number of retry attempts per failed download with exponential
        backoff, by default 3.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.  Each ODIM HDF5 file is
    roughly 400 KB – 2 MB.

    Note
    ----
    EUMETNET Open Radar Data documentation:
    https://eumetnet.github.io/openradardata-documentation/

    CloudFerro S3 archive (public, no authentication required):
    https://s3.waw3-1.cloudferro.com/openradar-archive/

    Composite products available via ``OPERALexicon``:

    - ``refc`` – DBZH composite reflectivity (dBZ)
    - ``tprate`` – RATE instantaneous surface rain rate (mm h\ :sup:`-1`\)
    - ``tp01`` – ACRR 1-hour rainfall accumulation (m)

    Archive spans from October 2011 to present in 15-minute steps.

    Badges
    ------
    region:eu dataclass:observation product:radar product:precip
    """

    # Class-level lat/lon grid cache keyed by (ysize, xsize).
    _grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def _grid(cls, ysize: int, xsize: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the cached LAEA lat/lon grid, computing it on first call."""
        key = (ysize, xsize)
        if key not in cls._grid_cache:
            cls._grid_cache[key] = _compute_opera_latlon(xsize, ysize)
        return cls._grid_cache[key]

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ) -> None:
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None
        self.fs: Any = None

    # ------------------------------------------------------------------
    # Async filesystem init
    # ------------------------------------------------------------------

    async def _async_init(self) -> None:
        """Initialise the async HTTP filesystem (called lazily on first fetch)."""
        from fsspec.implementations.http import HTTPFileSystem

        self.fs = HTTPFileSystem(asynchronous=True)

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch OPERA composite data for the requested times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to fetch (UTC).  Must align to 15-minute boundaries
            for the ODYSSEY era (before 2024-07) or 5-minute boundaries for
            the CIRRUS era (2024-07 onward).  Must fall within the archive
            window (2011-10 onward).
        variable : str | list[str] | VariableArray
            Variable names from :class:`~earth2studio.lexicon.OPERALexicon`
            (``"refc"``, ``"tprate"``, ``"tp01"``).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``(time, variable, y, x)``.
            Non-dimension coordinates ``_lat`` and ``_lon`` hold the 2-D
            geographic position of each pixel.

        Raises
        ------
        KeyError
            If a variable name is not in :class:`~earth2studio.lexicon.OPERALexicon`.
        ValueError
            If a requested time is outside the archive range, not on a
            15-minute grid boundary, or if the requested variables map to
            different pixel resolutions (e.g. DBZH at 1 km mixed with RATE
            at 2 km in the post-2024-07 CIRRUS era).
        """
        try:
            xr_array = _sync_async(self.fetch, time, variable, timeout=self.async_timeout)
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)
        return xr_array

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async implementation of :meth:`__call__`.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to fetch.
        variable : str | list[str] | VariableArray
            Variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``(time, variable, y, x)``.
        """
        if self.fs is None:
            await self._async_init()

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        tasks = self._create_tasks(time_list, variable_list)

        # Accumulate results keyed by (time_idx, var_idx); shape is determined
        # from the actual fetched arrays rather than pre-allocated.
        results: dict[tuple[int, int], np.ndarray] = {}

        async def _accumulate(task: _OPERAAsyncTask) -> None:
            results[(task.time_idx, task.var_idx)] = await self.fetch_wrapper(task)

        async with managed_session(self.fs):
            coros = [_accumulate(task) for task in tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=120.0,
                desc="Fetching OPERA data",
                verbose=(not self._verbose),
            )

        # Validate that all variables have the same grid shape.
        shapes = {arr.shape for arr in results.values()}
        if len(shapes) > 1:
            shape_str = ", ".join(
                f"{variable_list[vi]}→{results[(0, vi)].shape}"
                for vi in range(len(variable_list))
                if (0, vi) in results
            )
            raise ValueError(
                f"Requested variables have different pixel grids in this era "
                f"({shape_str}). Fetch each resolution group in a separate call."
            )

        ysize, xsize = next(iter(shapes))
        lat, lon = self._grid(ysize, xsize)

        data = np.empty(
            (len(time_list), len(variable_list), ysize, xsize), dtype=np.float32
        )
        for (ti, vi), arr in results.items():
            data[ti, vi] = arr

        xr_array = xr.DataArray(
            data=data,
            dims=["time", "variable", "y", "x"],
            coords={
                "time": time_list,
                "variable": variable_list,
                "y": np.arange(ysize),
                "x": np.arange(xsize),
            },
        ).assign_coords({"_lat": (("y", "x"), lat), "_lon": (("y", "x"), lon)})
        return xr_array

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[_OPERAAsyncTask]:
        """Build one async task per (time, variable) combination.

        Parameters
        ----------
        time_list : list[datetime]
            Validated list of request times.
        variable_list : list[str]
            Validated list of variable names.

        Returns
        -------
        list[_OPERAAsyncTask]
            Tasks ready for concurrent execution.
        """
        tasks: list[_OPERAAsyncTask] = []
        for i, t in enumerate(time_list):
            for j, v in enumerate(variable_list):
                odim_quantity, modifier = OPERALexicon[v]
                url = self._build_url(t, odim_quantity)
                tasks.append(
                    _OPERAAsyncTask(
                        time_idx=i,
                        var_idx=j,
                        url=url,
                        odim_quantity=odim_quantity,
                        modifier=modifier,
                    )
                )
        return tasks

    # ------------------------------------------------------------------
    # Fetch wrapper (retry + apply modifier)
    # ------------------------------------------------------------------

    async def fetch_wrapper(self, task: _OPERAAsyncTask) -> np.ndarray:
        """Fetch a single task with retry and apply the variable modifier.

        Parameters
        ----------
        task : _OPERAAsyncTask
            Task describing what to fetch.

        Returns
        -------
        np.ndarray
            Decoded and modifier-applied 2-D float32 array.
        """
        out = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=120.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        return task.modifier(out)

    # ------------------------------------------------------------------
    # Fetch single array (pure async I/O)
    # ------------------------------------------------------------------

    async def fetch_array(self, task: _OPERAAsyncTask) -> np.ndarray:
        """Download one ODIM HDF5 file and decode the requested quantity.

        Parameters
        ----------
        task : _OPERAAsyncTask
            Describes the URL and ODIM quantity to extract.

        Returns
        -------
        np.ndarray
            Decoded 2-D float32 array.

        Raises
        ------
        ValueError
            If the filesystem is not initialised or the quantity is absent.
        """
        if self.fs is None:
            raise ValueError("Filesystem not initialized; call _async_init first")

        sha = hashlib.sha256(task.url.encode())
        cache_path = os.path.join(self.cache, sha.hexdigest())

        if pathlib.Path(cache_path).is_file():
            logger.debug(f"Cache hit {cache_path}")
            data_bytes = pathlib.Path(cache_path).read_bytes()
        else:
            logger.debug(f"Fetching {task.url}")
            data_bytes = await self.fs._cat_file(task.url)
            with open(cache_path, "wb") as fh:
                await asyncio.to_thread(fh.write, data_bytes)

        return _decode_odim_h5(data_bytes, task.odim_quantity)

    # ------------------------------------------------------------------
    # URL construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_url(cls, t: datetime, odim_quantity: str) -> str:
        """Construct the CloudFerro S3 HTTPS URL for an ODIM composite file.

        The filename component changed between the ODYSSEY era (pre-2024-07,
        e.g. ``DBZH_QIND.h5``) and the CIRRUS/NIMBUS era (2024-07+,
        e.g. ``DBZH.h5``).  The ODIM quantity code inside the HDF5 is
        identical in both eras.

        Parameters
        ----------
        t : datetime
            Composite valid time (UTC, aligned to 15-minute boundary).
        odim_quantity : str
            ODIM quantity code (``"DBZH"``, ``"RATE"``, or ``"ACRR"``).

        Returns
        -------
        str
            Full HTTPS URL to the ODIM HDF5 file on CloudFerro S3.
        """
        param = (
            OPERALexicon.LEGACY_FILENAME_PARAMS.get(odim_quantity, odim_quantity)
            if t < _CIRRUS_START
            else odim_quantity
        )
        date_path = t.strftime("%Y/%m/%d")
        ts = t.strftime("%Y%m%dT%H%M")
        return f"{_ARCHIVE_BASE}/{date_path}/OPERA/COMP/OPERA@{ts}@0@{param}.h5"

    # ------------------------------------------------------------------
    # Time validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that each time is within the archive range and on-grid.

        The allowed grid depends on the era:

        * **ODYSSEY era (before 2024-07-01):** 15-minute boundaries.
        * **CIRRUS era (from 2024-07-01 onward):** 5-minute boundaries.

        Parameters
        ----------
        times : list[datetime]
            Times to validate.

        Raises
        ------
        ValueError
            If any time precedes the archive start (October 2011) or does not
            align to the era-appropriate composite interval.
        """
        epoch = datetime(2000, 1, 1)
        for t in times:
            if t < _MIN_DATE:
                raise ValueError(
                    f"Requested time {t} is before the OPERA archive start {_MIN_DATE}"
                )
            interval = _CIRRUS_INTERVAL_SECONDS if t >= _CIRRUS_START else _INTERVAL_SECONDS
            if int((t - epoch).total_seconds()) % interval != 0:
                raise ValueError(
                    f"Requested time {t} does not align to "
                    f"{interval // 60}-minute OPERA composite intervals"
                )

    # ------------------------------------------------------------------
    # Cache property
    # ------------------------------------------------------------------

    @property
    def cache(self) -> str:
        """Local directory used to cache downloaded ODIM HDF5 files."""
        cache_location = os.path.join(datasource_cache_root(), "opera")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(cache_location, f"tmp_{self._tmp_cache_hash}")
        return cache_location

    # ------------------------------------------------------------------
    # Available classmethod
    # ------------------------------------------------------------------

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether a given time is fetchable from the OPERA archive.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date/time to check.

        Returns
        -------
        bool
            ``True`` if ``time`` falls within the archive window and aligns to
            the era-appropriate composite interval (15 minutes for ODYSSEY,
            5 minutes for CIRRUS).
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
