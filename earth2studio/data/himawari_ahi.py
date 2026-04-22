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
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import nest_asyncio
import numpy as np
import s3fs
import xarray as xr
from loguru import logger

from earth2studio.data.utils import (
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    managed_session,
    prep_data_inputs,
)
from earth2studio.lexicon import HimawariLexicon
from earth2studio.utils.type import TimeArray, VariableArray

# ISatSS L2 Full Disk tile layout:
# 5500 x 5500 pixel full disk (at 2 km resolution)
# 88 tiles of 550 x 550 pixels each, arranged in a circular pattern
# covering Earth's disk.  Tile positions are read from the NetCDF
# ``tile_row_offset`` and ``tile_column_offset`` attributes.
FULL_DISK_PIXELS = 5500
TILE_SIZE = 550

# Channel resolution mapping (ISatSS filename prefix)
# C01: 1km (010), C02: 1km (010), C03: 0.5km (005), C04: 1km (010)
# C05-C16: 2km (020)
CHANNEL_RESOLUTION: dict[str, str] = {
    "M1C01": "010",
    "M1C02": "010",
    "M1C03": "005",
    "M1C04": "010",
    "M1C05": "020",
    "M1C06": "020",
    "M1C07": "020",
    "M1C08": "020",
    "M1C09": "020",
    "M1C10": "020",
    "M1C11": "020",
    "M1C12": "020",
    "M1C13": "020",
    "M1C14": "020",
    "M1C15": "020",
    "M1C16": "020",
}

# Downsample factor from native resolution to 2km output grid.
# Higher-res channels need block averaging to produce 550x550 tiles.
# 0.5km → 2km = factor 4, 1km → 2km = factor 2, 2km → 2km = factor 1
CHANNEL_DOWNSAMPLE: dict[str, int] = {
    "M1C01": 2,
    "M1C02": 2,
    "M1C03": 4,
    "M1C04": 2,
    "M1C05": 1,
    "M1C06": 1,
    "M1C07": 1,
    "M1C08": 1,
    "M1C09": 1,
    "M1C10": 1,
    "M1C11": 1,
    "M1C12": 1,
    "M1C13": 1,
    "M1C14": 1,
    "M1C15": 1,
    "M1C16": 1,
}

# ISatSS tile offsets at 2km resolution (row_offset, col_offset in pixels).
# Tile numbers 1-88 map to positions in the 5500x5500 Full Disk grid.
# Tiles form a circular pattern matching Earth's visible disk.
TILE_OFFSETS_2KM: dict[int, tuple[int, int]] = {
    1: (0, 1100),
    2: (0, 1650),
    3: (0, 2200),
    4: (0, 2750),
    5: (0, 3300),
    6: (0, 3850),
    7: (550, 550),
    8: (550, 1100),
    9: (550, 1650),
    10: (550, 2200),
    11: (550, 2750),
    12: (550, 3300),
    13: (550, 3850),
    14: (550, 4400),
    15: (1100, 0),
    16: (1100, 550),
    17: (1100, 1100),
    18: (1100, 1650),
    19: (1100, 2200),
    20: (1100, 2750),
    21: (1100, 3300),
    22: (1100, 3850),
    23: (1100, 4400),
    24: (1100, 4950),
    25: (1650, 0),
    26: (1650, 550),
    27: (1650, 1100),
    28: (1650, 1650),
    29: (1650, 2200),
    30: (1650, 2750),
    31: (1650, 3300),
    32: (1650, 3850),
    33: (1650, 4400),
    34: (1650, 4950),
    35: (2200, 0),
    36: (2200, 550),
    37: (2200, 1100),
    38: (2200, 1650),
    39: (2200, 2200),
    40: (2200, 2750),
    41: (2200, 3300),
    42: (2200, 3850),
    43: (2200, 4400),
    44: (2200, 4950),
    45: (2750, 0),
    46: (2750, 550),
    47: (2750, 1100),
    48: (2750, 1650),
    49: (2750, 2200),
    50: (2750, 2750),
    51: (2750, 3300),
    52: (2750, 3850),
    53: (2750, 4400),
    54: (2750, 4950),
    55: (3300, 0),
    56: (3300, 550),
    57: (3300, 1100),
    58: (3300, 1650),
    59: (3300, 2200),
    60: (3300, 2750),
    61: (3300, 3300),
    62: (3300, 3850),
    63: (3300, 4400),
    64: (3300, 4950),
    65: (3850, 0),
    66: (3850, 550),
    67: (3850, 1100),
    68: (3850, 1650),
    69: (3850, 2200),
    70: (3850, 2750),
    71: (3850, 3300),
    72: (3850, 3850),
    73: (3850, 4400),
    74: (3850, 4950),
    75: (4400, 550),
    76: (4400, 1100),
    77: (4400, 1650),
    78: (4400, 2200),
    79: (4400, 2750),
    80: (4400, 3300),
    81: (4400, 3850),
    82: (4400, 4400),
    83: (4950, 1100),
    84: (4950, 1650),
    85: (4950, 2200),
    86: (4950, 2750),
    87: (4950, 3300),
    88: (4950, 3850),
}

# Bit depth per channel (ISatSS filename encoding)
CHANNEL_BITS: dict[str, str] = {
    "M1C01": "B11",
    "M1C02": "B11",
    "M1C03": "B11",
    "M1C04": "B11",
    "M1C05": "B14",
    "M1C06": "B14",
    "M1C07": "B14",
    "M1C08": "B14",
    "M1C09": "B14",
    "M1C10": "B14",
    "M1C11": "B14",
    "M1C12": "B14",
    "M1C13": "B14",
    "M1C14": "B14",
    "M1C15": "B14",
    "M1C16": "B14",
}


def _normalize_lon(lon_deg: float) -> float:
    """Wrap longitude to [-180, 180)."""
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _compute_pixel_roi(
    lat_lon_bbox: tuple[float, float, float, float],
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[int, int, int, int]:
    """Convert a lat/lon bounding box to pixel row/column bounds.

    Both ``[-180, 180]`` and ``[0, 360]`` longitude conventions are
    supported; longitudes are normalised to ``[-180, 180)`` before the
    grid lookup.

    Parameters
    ----------
    lat_lon_bbox : tuple[float, float, float, float]
        ``(lat_min, lon_min, lat_max, lon_max)`` in degrees.
    lat : np.ndarray
        2-D latitude array from the grid (NaN where off-Earth).
    lon : np.ndarray
        2-D longitude array from the grid (NaN where off-Earth).

    Returns
    -------
    tuple[int, int, int, int]
        ``(row_start, row_end, col_start, col_end)`` pixel bounds
        (end-exclusive) suitable for slicing.

    Raises
    ------
    ValueError
        If no grid points fall within the bounding box.
    """
    lat_min, lon_min, lat_max, lon_max = lat_lon_bbox
    lon_min = _normalize_lon(lon_min)
    lon_max = _normalize_lon(lon_max)

    # Normalise the grid longitudes too
    lon_norm = _normalize_lon(lon)

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    if lon_min > lon_max:
        # Bounding box wraps across the antimeridian (e.g. 160°E to -160°E)
        lon_mask = (lon_norm >= lon_min) | (lon_norm <= lon_max)
    else:
        lon_mask = (lon_norm >= lon_min) & (lon_norm <= lon_max)

    mask = lat_mask & lon_mask
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError(f"No grid points fall within lat_lon_bbox={lat_lon_bbox}")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


@dataclass
class HimawariAsyncTask:
    """Async task for fetching a single Himawari tile."""

    time_idx: int
    var_idx: int
    remote_uri: str
    modifier: Callable
    downsample_factor: int = 1


class HimawariAHI:
    """Himawari-8/9 AHI (Advanced Himawari Imager) data source.

    This data source provides access to Himawari-8 and Himawari-9 satellite
    imagery from the NOAA ISatSS L2 Full Disk product on AWS S3. The data
    uses the same geostationary fixed grid projection as GOES, with the
    satellite positioned at 140.7°E.

    Each AHI channel is stored as tiled NetCDF files (88 tiles per channel).
    Higher-resolution channels (C01/C02/C04 at 1 km, C03 at 0.5 km) are
    downsampled to the 2 km grid via block averaging, producing a uniform
    5500×5500 pixel Full Disk output for all 16 channels.

    Parameters
    ----------
    satellite : str, optional
        Which Himawari satellite to use ('himawari8' or 'himawari9'),
        by default 'himawari9'
    lat_lon_bbox : tuple[float, float, float, float] | None, optional
        Bounding box ``(lat_min, lon_min, lat_max, lon_max)`` in degrees to
        crop the full-disk image. Only tiles that overlap the requested region
        are downloaded.  When ``None`` the full disk is returned, by default None
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation,
        by default 600
    async_workers : int, optional
        Maximum number of concurrent async fetch tasks,
        by default 16
    retries : int, optional
        Number of retry attempts per failed fetch task with
        exponential backoff, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests. Each full disk image
    consists of 88 tile files per channel.

    Note
    ----
    Additional information on the data repository:

    - https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html
    - https://aws.amazon.com/marketplace/pp/prodview-eu33kalocbhiw
    - https://noaa-himawari9.s3.amazonaws.com/index.html

    Badges
    ------
    region:as region:au dataclass:observation product:sat
    """

    SCAN_TIME_FREQUENCY = 600  # 10-minute Full Disk cadence
    SCAN_DIMENSIONS = (FULL_DISK_PIXELS, FULL_DISK_PIXELS)  # (y, x) at 2km
    VALID_SATELLITES = ["himawari8", "himawari9"]
    HIMAWARI_HISTORY_RANGE: dict[str, tuple[datetime, datetime | None]] = {
        "himawari8": (
            datetime(2015, 7, 7),
            datetime(2022, 12, 13),
        ),
        "himawari9": (datetime(2022, 12, 13), None),
    }
    # Geostationary projection constants (ISatSS uses microradian coordinates)
    PERSPECTIVE_POINT_HEIGHT = 35785831.0  # meters (from ISatSS metadata)
    SEMI_MAJOR_AXIS = 6378137.0
    SEMI_MINOR_AXIS = 6356752.31414
    LATITUDE_OF_PROJECTION_ORIGIN = 0.0
    LONGITUDE_OF_PROJECTION_ORIGIN = 140.7  # degrees East

    # ISatSS Full Disk y/x coordinates in radians (converted from microradians)
    # From tile metadata: scale_factor=55.888 µrad, add_offset=-153719.917 µrad
    # Full disk spans -153719.917 to 153719.917 µrad = -0.153720 to 0.153720 rad
    # 5500 pixels at 2km resolution
    _Y_COORDS = np.linspace(0.15371991700, -0.15371991700, FULL_DISK_PIXELS)
    _X_COORDS = np.linspace(-0.15371991700, 0.15371991700, FULL_DISK_PIXELS)

    BASE_URL = "s3://noaa-{satellite}/AHI-L2-FLDK-ISatSS/{year:04d}/{month:02d}/{day:02d}/{hour:02d}{minute:02d}/"

    def __init__(
        self,
        satellite: str = "himawari9",
        lat_lon_bbox: tuple[float, float, float, float] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int = 16,
        retries: int = 3,
    ):
        self._satellite = satellite.lower()
        self._lat_lon_bbox = lat_lon_bbox
        self._cache = cache
        self._verbose = verbose
        self._async_workers = async_workers
        self._retries = retries
        self._async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        if self._satellite not in self.VALID_SATELLITES:
            raise ValueError(
                f"Invalid satellite '{self._satellite}'. "
                f"Must be one of {self.VALID_SATELLITES}"
            )

        # Pre-compute grid coordinates
        self._lat, self._lon = HimawariAHI.grid()

        # Pixel ROI bounds (lazily computed on first fetch when bbox is set)
        self._pixel_roi: tuple[int, int, int, int] | None = None

        # Attempt sync init of filesystem
        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                logger.warning(f"Himawari async init failed: {e}")
            self.fs = None

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem.

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
    ) -> xr.DataArray:
        """Function to get Himawari AHI data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC). Timezone-aware
            datetimes are converted to UTC automatically.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables
            to return. Must be in the Himawari lexicon (ahi01-ahi16).

        Returns
        -------
        xr.DataArray
            Data array with dimensions [time, variable, y, x]
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        try:
            xr_array = loop.run_until_complete(
                asyncio.wait_for(
                    self.fetch(time, variable), timeout=self._async_timeout
                )
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get Himawari AHI data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return using standardized names (ahi01-ahi16).

        Returns
        -------
        xr.DataArray
            Data array with dimensions [time, variable, y, x]
        """
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If you are calling this "
                "function directly make sure the data source is initialized "
                "inside the async loop!"
            )

        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Compute pixel ROI if bounding box is set (lazy — once)
        if self._lat_lon_bbox is not None and self._pixel_roi is None:
            self._pixel_roi = _compute_pixel_roi(
                self._lat_lon_bbox, self._lat, self._lon
            )

        # Determine output shape (cropped if bbox is set)
        if self._pixel_roi is not None:
            r0, r1, c0, c1 = self._pixel_roi
            ny = r1 - r0
            nx = c1 - c0
            lat_out = self._lat[r0:r1, c0:c1]
            lon_out = self._lon[r0:r1, c0:c1]
            y_coords = self._Y_COORDS[r0:r1]
            x_coords = self._X_COORDS[c0:c1]
        else:
            ny, nx = self.SCAN_DIMENSIONS
            lat_out = self._lat
            lon_out = self._lon
            y_coords = self._Y_COORDS
            x_coords = self._X_COORDS

        async with managed_session(self.fs) as session:  # noqa: F841
            # Pre-allocate output array
            xr_array = xr.DataArray(
                data=np.full(
                    (len(time), len(variable), ny, nx),
                    np.nan,
                    dtype=np.float32,
                ),
                dims=["time", "variable", "y", "x"],
                coords={
                    "time": time,
                    "variable": variable,
                    "y": y_coords,
                    "x": x_coords,
                },
            )

            # Build async tasks
            tasks = await self._create_tasks(time, variable)

            # Execute with bounded concurrency
            coros = [self.fetch_wrapper(task, xr_array=xr_array) for task in tasks]
            await gather_with_concurrency(
                coros,
                max_workers=self._async_workers,
                task_timeout=120.0,
                desc="Fetching Himawari data",
                verbose=(not self._verbose),
            )

        # Add curvilinear lat/lon coordinates
        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat_out), "_lon": (("y", "x"), lon_out)}
        )
        return xr_array

    async def _create_tasks(
        self,
        time: list[datetime],
        variable: list[str],
    ) -> list[HimawariAsyncTask]:
        """Build list of async download tasks for all time/variable/tile combinations.

        Parameters
        ----------
        time : list[datetime]
            Timestamps to download
        variable : list[str]
            Variables to download

        Returns
        -------
        list[HimawariAsyncTask]
            List of async task requests
        """
        tasks: list[HimawariAsyncTask] = []

        for i, t in enumerate(time):
            # Get base directory for this time step
            base_dir = self.BASE_URL.format(
                satellite=self._satellite,
                year=t.year,
                month=t.month,
                day=t.day,
                hour=t.hour,
                minute=t.minute,
            )

            # List files in the directory to discover tiles
            try:
                all_files = await self.fs._ls(base_dir, detail=False)  # type: ignore[attr-defined]
            except FileNotFoundError:
                logger.warning(f"No data found for {t} at {base_dir}, skipping")
                continue

            for j, v in enumerate(variable):
                try:
                    channel_id, modifier = HimawariLexicon[v]  # type: ignore[misc]
                except KeyError:
                    logger.warning(
                        f"Variable {v} not found in Himawari lexicon, skipping"
                    )
                    continue

                # Filter files for this channel
                channel_files = [f for f in all_files if f"-{channel_id}-" in f]

                if not channel_files:
                    logger.warning(f"No files found for channel {channel_id} at {t}")
                    continue

                for tile_file in channel_files:
                    # Parse tile number to validate it's a real tile
                    fname = tile_file.split("/")[-1]
                    tile_num = self._parse_tile_number(fname)
                    if tile_num is None:
                        continue

                    # Skip tiles that don't overlap the pixel ROI
                    if self._pixel_roi is not None:
                        if tile_num not in TILE_OFFSETS_2KM:
                            logger.debug(
                                f"Unknown tile number {tile_num} in {fname}, "
                                "cannot pre-filter — fetching anyway"
                            )
                        else:
                            tr, tc = TILE_OFFSETS_2KM[tile_num]
                            r0, r1, c0, c1 = self._pixel_roi
                            if (
                                tr + TILE_SIZE <= r0
                                or tr >= r1
                                or tc + TILE_SIZE <= c0
                                or tc >= c1
                            ):
                                continue

                    downsample = CHANNEL_DOWNSAMPLE.get(channel_id, 1)

                    tasks.append(
                        HimawariAsyncTask(
                            time_idx=i,
                            var_idx=j,
                            remote_uri=tile_file,
                            modifier=modifier,
                            downsample_factor=downsample,
                        )
                    )

        return tasks

    async def fetch_wrapper(
        self,
        task: HimawariAsyncTask,
        xr_array: xr.DataArray,
    ) -> None:
        """Unpack task, fetch with retry, and place result into output array.

        Parameters
        ----------
        task : HimawariAsyncTask
            Task to execute
        xr_array : xr.DataArray
            Output array to write into
        """
        result = await async_retry(
            self.fetch_array,
            task,
            retries=self._retries,
            backoff=1.0,
            task_timeout=60.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        if result is not None:
            data, row_offset, col_offset = result
            nr, nc = data.shape

            # When a pixel ROI is active, adjust offsets relative to the
            # cropped grid and clip the tile to the ROI boundaries.
            if self._pixel_roi is not None:
                r0, r1, c0, c1 = self._pixel_roi
                # Shift offsets into cropped coordinate space
                row_offset -= r0
                col_offset -= c0
                # Clip tile edges to output array boundaries
                src_r0 = max(0, -row_offset)
                src_c0 = max(0, -col_offset)
                dst_r0 = max(0, row_offset)
                dst_c0 = max(0, col_offset)
                dst_r1 = min(r1 - r0, row_offset + nr)
                dst_c1 = min(c1 - c0, col_offset + nc)
                src_r1 = src_r0 + (dst_r1 - dst_r0)
                src_c1 = src_c0 + (dst_c1 - dst_c0)
                if dst_r1 > dst_r0 and dst_c1 > dst_c0:
                    xr_array[
                        task.time_idx,
                        task.var_idx,
                        dst_r0:dst_r1,
                        dst_c0:dst_c1,
                    ] = data[src_r0:src_r1, src_c0:src_c1]
            else:
                xr_array[
                    task.time_idx,
                    task.var_idx,
                    row_offset : row_offset + nr,
                    col_offset : col_offset + nc,
                ] = data

    async def fetch_array(
        self, task: HimawariAsyncTask
    ) -> tuple[np.ndarray, int, int] | None:
        """Fetch a single tile from the remote store.

        Parameters
        ----------
        task : HimawariAsyncTask
            Task with all fetch metadata

        Returns
        -------
        tuple[np.ndarray, int, int] | None
            Tuple of (tile data at 2km resolution, row_offset, col_offset)
            in the output grid, or None if tile cannot be read.
            Offsets are read from the NetCDF ``tile_row_offset`` and
            ``tile_column_offset`` attributes and normalised to 2 km pixels.
        """
        # Download tile file to cache
        cache_path = await self._fetch_remote_file(task.remote_uri)

        # Open NetCDF and extract Sectorized_CMI + tile offsets
        ds = xr.open_dataset(cache_path)
        try:
            cmi = ds["Sectorized_CMI"].values
            # Read tile placement from NetCDF attributes (in native pixels)
            native_row_offset = int(ds.attrs.get("tile_row_offset", 0))
            native_col_offset = int(ds.attrs.get("tile_column_offset", 0))
        except KeyError:
            logger.warning(f"Sectorized_CMI not found in {task.remote_uri}")
            return None
        finally:
            ds.close()

        data = cmi.astype(np.float32)

        # Downsample higher-resolution channels to 2km grid via block
        # averaging.  C01/C02/C04 are 1km (factor 2), C03 is 0.5km
        # (factor 4), C05-C16 are already 2km (factor 1).
        f = task.downsample_factor
        if f > 1:
            nr, nc = data.shape
            out_r = nr // f
            out_c = nc // f
            data = data[: out_r * f, : out_c * f].reshape(out_r, f, out_c, f)
            data = np.nanmean(data, axis=(1, 3))

        # Convert offsets from native resolution to 2km output grid
        row_offset = native_row_offset // f
        col_offset = native_col_offset // f

        return task.modifier(data), row_offset, col_offset

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify that date times are valid for Himawari.

        Parameters
        ----------
        times : list[datetime]
            List of date times to validate

        Raises
        ------
        ValueError
            If a time is not on a 10-minute interval or outside valid range
        """
        for time in times:
            # Check 10-minute interval
            if (
                not (time - datetime(1900, 1, 1)).total_seconds()
                % self.SCAN_TIME_FREQUENCY
                == 0
            ):
                raise ValueError(
                    f"Requested date time {time} needs to be a 10-minute "
                    f"interval for Himawari Full Disk"
                )

            start_date, end_date = self.HIMAWARI_HISTORY_RANGE[self._satellite]
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} is before {self._satellite} "
                    f"became operational ({start_date})"
                )
            if end_date and time > end_date:
                raise ValueError(
                    f"Requested date time {time} is after {self._satellite} "
                    f"was retired ({end_date})"
                )

    async def _fetch_remote_file(self, path: str) -> str:
        """Fetch a remote file into the local cache.

        Parameters
        ----------
        path : str
            S3 path to fetch

        Returns
        -------
        str
            Local cache path
        """
        if self.fs is None:
            raise ValueError("File system is not initialized")

        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as f:
                f.write(data)

        return cache_path

    @staticmethod
    def _parse_tile_number(filename: str) -> int | None:
        """Parse tile number from ISatSS filename.

        Parameters
        ----------
        filename : str
            Filename like ``OR_HFD-020-B14-M1C07-T001_GH9_s...``

        Returns
        -------
        int | None
            Tile number (1-88), or None if parse fails
        """
        # Look for -T{NNN} pattern
        parts = filename.split("-")
        for part in parts:
            if part.startswith("T") and "_" in part:
                tile_str = part.split("_")[0]
                try:
                    return int(tile_str[1:])
                except ValueError:
                    pass
        return None

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "himawari")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_himawari_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        satellite: str = "himawari9",
    ) -> bool:
        """Check if given date time is available in the Himawari object store.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check
        satellite : str, optional
            Which satellite to check, by default 'himawari9'

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)
            # Strip timezone for comparison
            time = time.replace(tzinfo=None)

        satellite = satellite.lower()
        if satellite not in cls.VALID_SATELLITES:
            return False

        start_date, end_date = cls.HIMAWARI_HISTORY_RANGE[satellite]
        if time < start_date:
            return False
        if end_date and time > end_date:
            return False

        # Check 10-minute interval
        if (
            not (time - datetime(1900, 1, 1)).total_seconds() % cls.SCAN_TIME_FREQUENCY
            == 0
        ):
            return False

        return True

    @classmethod
    def grid(cls) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) in degrees for the native Himawari AHI grid.

        Uses the geostationary fixed grid projection from ISatSS metadata,
        which is the same projection formulation as GOES ABI.

        Note
        ----
        Projection reference:
        https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (lat, lon) arrays in degrees, shape (5500, 5500)
        """
        x_1d = cls._X_COORDS
        y_1d = cls._Y_COORDS

        lon_origin = cls.LONGITUDE_OF_PROJECTION_ORIGIN
        H = cls.PERSPECTIVE_POINT_HEIGHT + cls.SEMI_MAJOR_AXIS
        r_eq = cls.SEMI_MAJOR_AXIS
        r_pol = cls.SEMI_MINOR_AXIS

        x_2d, y_2d = np.meshgrid(x_1d, y_1d)

        with np.errstate(invalid="ignore"):
            lambda_0 = (lon_origin * np.pi) / 180.0

            a_var = np.power(np.sin(x_2d), 2.0) + (
                np.power(np.cos(x_2d), 2.0)
                * (
                    np.power(np.cos(y_2d), 2.0)
                    + (((r_eq * r_eq) / (r_pol * r_pol)) * np.power(np.sin(y_2d), 2.0))
                )
            )
            b_var = -2.0 * H * np.cos(x_2d) * np.cos(y_2d)
            c_var = (H**2.0) - (r_eq**2.0)
            r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (
                2.0 * a_var
            )

            s_x = r_s * np.cos(x_2d) * np.cos(y_2d)
            s_y = -r_s * np.sin(x_2d)
            s_z = r_s * np.cos(x_2d) * np.sin(y_2d)

            lat = (180.0 / np.pi) * (
                np.arctan(
                    ((r_eq * r_eq) / (r_pol * r_pol))
                    * (s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y)))
                )
            )
            lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

        return lat, lon
