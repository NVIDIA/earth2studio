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
import glob
import os
import pathlib
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta, timezone

import nest_asyncio
import numpy as np
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    async_retry,
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon.metop import MetOpMTGLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import eumdac
    import netCDF4
except ImportError:
    OptionalDependencyFailure("data")
    eumdac = None  # type: ignore[assignment]
    netCDF4 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Grid & projection constants for MTG-I FCI Full Disk
# ---------------------------------------------------------------------------
GRID_SIZE_2KM = (5568, 5568)
GRID_SIZE_1KM = (11136, 11136)

# Geostationary projection parameters (MTG-I1 at 0° longitude)
SUB_SATELLITE_LON = 0.0  # degrees
PERSPECTIVE_POINT_HEIGHT = 35786400.0  # metres
SEMI_MAJOR_AXIS = 6378137.0  # metres (WGS-84)
SEMI_MINOR_AXIS = 6356752.3142  # metres (WGS-84)

# Scale/offset for the 2 km grid angular coordinates (radians)
CFAC_2KM = 5.58879280e-05
COFF_2KM = 2784.5
LFAC_2KM = -5.58879280e-05
LOFF_2KM = 2784.5

# Scale/offset for the 1 km grid angular coordinates (radians)
CFAC_1KM = 2.79439640e-05
COFF_1KM = 5568.5
LFAC_1KM = -2.79439640e-05
LOFF_1KM = 5568.5

# Resolution → (CFAC, COFF, LFAC, LOFF, grid_size)
_GRID_PARAMS = {
    "2km": (CFAC_2KM, COFF_2KM, LFAC_2KM, LOFF_2KM, GRID_SIZE_2KM),
    "1km": (CFAC_1KM, COFF_1KM, LFAC_1KM, LOFF_1KM, GRID_SIZE_1KM),
}

# Channel name → native resolution string
_VARIABLE_RESOLUTION: dict[str, str] = {
    "vis_04": "1km",
    "vis_05": "1km",
    "vis_08": "1km",
    "vis_09": "1km",
    "nir_13": "1km",
    "nir_16": "2km",
    "wv_63": "2km",
    "wv_73": "2km",
    "ir_87": "2km",
    "ir_97": "2km",
    "ir_123": "2km",
    "ir_133": "2km",
}

# Operational start date for MTG-I1 FCI
_OPERATIONAL_START = datetime(2024, 1, 16, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: geostationary scan angle → lat/lon
# ---------------------------------------------------------------------------
def _mtg_fci_scan_to_latlon(
    cfac: float,
    coff: float,
    lfac: float,
    loff: float,
    nrows: int,
    ncols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MTG FCI scan angles to latitude/longitude arrays.

    Parameters
    ----------
    cfac : float
        Column scaling factor (radians per pixel)
    coff : float
        Column offset
    lfac : float
        Line scaling factor (radians per pixel)
    loff : float
        Line offset
    nrows : int
        Number of rows
    ncols : int
        Number of columns

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (lat, lon) arrays in degrees, NaN where off-Earth
    """
    cols = np.arange(1, ncols + 1, dtype=np.float64)
    rows = np.arange(1, nrows + 1, dtype=np.float64)

    x = (cols - coff) * cfac  # radians, E/W
    y = (rows - loff) * lfac  # radians, N/S

    xx, yy = np.meshgrid(x, y)

    cos_x = np.cos(xx)
    cos_y = np.cos(yy)
    sin_x = np.sin(xx)
    sin_y = np.sin(yy)

    r_eq = SEMI_MAJOR_AXIS
    r_pol = SEMI_MINOR_AXIS
    H = PERSPECTIVE_POINT_HEIGHT + r_eq

    with np.errstate(invalid="ignore"):
        a = sin_x**2 + cos_x**2 * (cos_y**2 + (r_eq / r_pol) ** 2 * sin_y**2)
        b = -2.0 * H * cos_x * cos_y
        c = H**2 - r_eq**2

        discriminant = b**2 - 4.0 * a * c
        r_s = (-b - np.sqrt(discriminant)) / (2.0 * a)

        s_x = r_s * cos_x * cos_y
        s_y = -r_s * sin_x
        s_z = r_s * cos_x * sin_y

        lat = np.degrees(
            np.arctan((r_eq / r_pol) ** 2 * s_z / np.sqrt((H - s_x) ** 2 + s_y**2))
        )
        lon_origin_rad = np.radians(SUB_SATELLITE_LON)
        lon = np.degrees(lon_origin_rad - np.arctan(s_y / (H - s_x)))

    return lat, lon


def _normalize_lon(lon_deg: float) -> float:
    """Normalise a longitude value into the ``[-180, 180)`` range.

    Accepts any real-valued longitude (e.g. 350 → -10, -200 → 160).

    Parameters
    ----------
    lon_deg : float
        Longitude in degrees.

    Returns
    -------
    float
        Longitude wrapped to ``[-180, 180)``.
    """
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
    """
    lat_min, lon_min, lat_max, lon_max = lat_lon_bbox

    # Normalise longitudes to [-180, 180) so both conventions work
    lon_min = _normalize_lon(lon_min)
    lon_max = _normalize_lon(lon_max)

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError(f"No grid points fall within lat_lon_bbox={lat_lon_bbox}")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _sort_body_key(name: str) -> int:
    """Extract numeric chunk index from a BODY segment filename for sorting.

    FCI segment filenames end with a chunk number before the extension,
    e.g. ``'…_BODY_003.nc'``.  This extracts the trailing integer.

    Parameters
    ----------
    name : str
        Segment filename (basename), e.g. ``'…_BODY_003.nc'``

    Returns
    -------
    int
        Numeric chunk index
    """
    stem = name.rsplit(".", 1)[0]  # remove .nc extension
    parts = stem.rsplit("_", 1)  # split on last underscore
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        # Fallback: find any trailing digits
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else 0


@check_optional_dependencies()
class MetOpMTG:
    """EUMETSAT MTG-I FCI Level-1C Full Disk calibrated radiance data source.

    This data source provides access to Meteosat Third Generation (MTG)
    Flexible Combined Imager (FCI) Level-1C Full Disk products via the
    EUMETSAT Data Store. The 12 spectral channels cover visible, near-IR,
    water-vapour and infrared bands at 1 km or 2 km nadir resolution.

    Parameters
    ----------
    resolution : str, optional
        Grid resolution, either ``'2km'`` or ``'1km'``, by default ``'2km'``
    lat_lon_bbox : tuple[float, float, float, float] | None, optional
        Bounding box ``(lat_min, lon_min, lat_max, lon_max)`` in degrees to
        crop the full-disk image. Only BODY segments that overlap the requested latitude
        range are read, saving disk and memory.  When ``None`` the full disk is
        returned, by default None
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Total timeout in seconds for the entire fetch operation,
        by default 1200
    retries : int, optional
        Number of retry attempts per failed fetch task with
        exponential backoff, by default 3

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.

    Note
    ----
    Requires EUMETSAT Data Store credentials. Set the following environment
    variables:

    - ``EUMETSAT_CONSUMER_KEY``: Your EUMETSAT API consumer key
    - ``EUMETSAT_CONSUMER_SECRET``: Your EUMETSAT API consumer secret

    Register at https://eoportal.eumetsat.int/ to obtain credentials.

    Note
    ----
    Additional information on the data repository:

    - https://data.eumetsat.int/product/EO:EUM:DAT:0662
    - https://www.eumetsat.int/mtg-fci-level-1c-full-disk

    Badges
    ------
    region:eu region:af dataclass:observation product:sat
    """

    COLLECTION_ID = "EO:EUM:DAT:0662"
    SCAN_FREQUENCY = 600

    def __init__(
        self,
        resolution: str = "2km",
        lat_lon_bbox: tuple[float, float, float, float] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 1200,
        retries: int = 3,
    ) -> None:
        self._resolution = self._resolve_resolution(resolution)
        self._lat_lon_bbox = lat_lon_bbox
        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout
        self._retries = retries
        self._tmp_cache_hash: str | None = None

        # Grid (lazily computed on first access)
        self._grid: tuple[np.ndarray, np.ndarray] | None = None
        # Pixel ROI bounds (lazily computed on first fetch when bbox is set)
        self._pixel_roi: tuple[int, int, int, int] | None = None

        # Credentials from environment variables
        self._consumer_key = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
        self._consumer_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")
        if not self._consumer_key or not self._consumer_secret:
            logger.warning(
                "EUMETSAT_CONSUMER_KEY and/or EUMETSAT_CONSUMER_SECRET not set. "
                "Data fetching will fail."
            )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve MTG FCI data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.  Must be in the ``MetOpMTGLexicon``.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, y, x]``.
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            xr_array = loop.run_until_complete(
                asyncio.wait_for(self.fetch(time, variable), timeout=self.async_timeout)
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
        """Async function to retrieve MTG FCI data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, y, x]``.
        """
        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)

        # Validate that all variables share the same resolution
        self._check_resolution_consistency(variable)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Ensure grid is computed
        lat, lon = self._ensure_grid()

        # Compute pixel ROI if bounding box is set
        if self._lat_lon_bbox is not None and self._pixel_roi is None:
            self._pixel_roi = _compute_pixel_roi(self._lat_lon_bbox, lat, lon)

        # Determine output shape (cropped if bbox is set)
        if self._pixel_roi is not None:
            r0, r1, c0, c1 = self._pixel_roi
            ny = r1 - r0
            nx = c1 - c0
            lat_out = lat[r0:r1, c0:c1]
            lon_out = lon[r0:r1, c0:c1]
        else:
            grid_size = _GRID_PARAMS[self._resolution][4]
            ny, nx = grid_size
            lat_out = lat
            lon_out = lon

        # Pre-allocate output
        y_coords = np.arange(ny, dtype=np.float64)
        x_coords = np.arange(nx, dtype=np.float64)
        xr_array = xr.DataArray(
            data=np.full((len(time), len(variable), ny, nx), np.nan, dtype=np.float32),
            dims=["time", "variable", "y", "x"],
            coords={
                "time": time,
                "variable": variable,
                "y": y_coords,
                "x": x_coords,
            },
        )

        # Phase 1: Download products in parallel (one per unique time)
        unique_times = list(dict.fromkeys(time))  # preserve order, deduplicate
        download_coros = [
            async_retry(
                asyncio.to_thread,
                self._fetch_product,
                t,
                retries=self._retries,
                backoff=2.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for t in unique_times
        ]
        product_dirs_list: list[str] = await tqdm.gather(
            *download_coros,
            desc="Downloading MTG products",
            disable=(not self._verbose),
        )
        product_dir_map = dict(zip(unique_times, product_dirs_list))

        # Phase 2: Read NetCDF segments sequentially (HDF5/netCDF4 is
        # not thread-safe — concurrent reads cause segfaults)
        for i, t in enumerate(time):
            product_dir = product_dir_map[t]
            for j, v in enumerate(variable):
                channel_name, modifier = MetOpMTGLexicon[v]
                data = self._read_channel(product_dir, channel_name, self._pixel_roi)
                xr_array[i, j] = modifier(data)

        # Attach grid coordinates
        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat_out), "_lon": (("y", "x"), lon_out)}
        )

        return xr_array

    def _fetch_product(self, time: datetime) -> str:
        """Download the MTG FCI product for the given time.

        Parameters
        ----------
        time : datetime
            UTC timestamp

        Returns
        -------
        str
            Path to the extracted product directory
        """
        # Build a cache-friendly directory name
        time_str = time.strftime("%Y%m%dT%H%M%S")
        product_cache = os.path.join(self.cache, f"mtg_{time_str}")

        # Only reuse cache if BODY segment files actually exist
        if os.path.isdir(product_cache):
            body_files = glob.glob(
                os.path.join(product_cache, "**", "*BODY*.nc"), recursive=True
            )
            if body_files:
                logger.debug("Using cached MTG product: {}", product_cache)
                return product_cache
            # Empty directory from a previous failed attempt — re-download
            logger.debug("Cached directory empty, re-downloading: {}", product_cache)

        pathlib.Path(product_cache).mkdir(parents=True, exist_ok=True)

        # Authenticate and search for products
        token = eumdac.AccessToken(
            credentials=(self._consumer_key, self._consumer_secret)
        )
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(self.COLLECTION_ID)

        dt_start = time - timedelta(seconds=self.SCAN_FREQUENCY // 2)
        dt_end = time + timedelta(seconds=self.SCAN_FREQUENCY // 2)

        products = collection.search(dtstart=dt_start, dtend=dt_end)
        product_list = list(products)

        if not product_list:
            raise FileNotFoundError(
                f"No MTG FCI products found for time {time} "
                f"(searched {dt_start} to {dt_end})"
            )

        product = product_list[0]
        if self._verbose:
            logger.info("Downloading MTG FCI product: {}", product)

        # Download and extract zip
        self._download_product(product, product_cache)

        return product_cache

    def _download_product(self, product: object, product_cache: str) -> None:
        """Download a product zip and extract BODY NetCDF segments.

        Parameters
        ----------
        product : eumdac.Product
            EUMETSAT product handle
        product_cache : str
            Directory to extract into
        """
        zip_path = os.path.join(product_cache, "product.zip")

        logger.debug("Downloading MTG product zip to {}", zip_path)
        with product.open() as stream:  # type: ignore[attr-defined]
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(stream, f)

        zip_size = os.path.getsize(zip_path)
        logger.debug("Downloaded zip size: {} bytes", zip_size)

        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise RuntimeError(
                f"Downloaded file is not a valid zip archive ({zip_size} bytes)"
            )

        # Extract only BODY NetCDF files
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            logger.debug("Zip contains {} entries", len(all_names))

            body_names = sorted(
                [
                    n
                    for n in all_names
                    if "BODY" in os.path.basename(n) and n.endswith(".nc")
                ],
                key=lambda n: _sort_body_key(os.path.basename(n)),
            )

            if not body_names:
                # Log all entry names for debugging
                logger.warning(
                    "No BODY .nc entries in zip. All entries: {}",
                    all_names[:20],
                )

            logger.debug("Extracting {} BODY segment files from zip", len(body_names))
            for name in body_names:
                zf.extract(name, product_cache)

        # Remove zip to save space
        os.remove(zip_path)

    def _read_channel(
        self,
        product_dir: str,
        channel_name: str,
        pixel_roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Read a single channel from extracted BODY segments using netCDF4.

        FCI Level-1C products store each channel's radiance data in an HDF5
        group structure:  ``/data/{channel}/measured/effective_radiance``.

        When *pixel_roi* is provided, only the BODY segments that overlap
        the requested row range are read and the result is cropped to the
        bounding box.

        Parameters
        ----------
        product_dir : str
            Path to extracted product directory
        channel_name : str
            FCI channel name (e.g., ``'vis_04'``)
        pixel_roi : tuple[int, int, int, int] | None, optional
            ``(row_start, row_end, col_start, col_end)`` pixel bounds
            (end-exclusive).  When ``None`` the full disk is returned.

        Returns
        -------
        np.ndarray
            2-D array (float32).  Full disk or cropped to *pixel_roi*.
        """
        # Find all BODY segment files
        body_files = glob.glob(
            os.path.join(product_dir, "**", "*BODY*.nc"), recursive=True
        )

        body_files.sort(key=lambda p: _sort_body_key(os.path.basename(p)))

        if not body_files:
            raise FileNotFoundError(f"No BODY segment files found in {product_dir}")

        # Read and vertically stack segments
        segments: list[np.ndarray] = []
        group_path = f"/data/{channel_name}/measured"

        # Track running row offset for selective segment loading
        row_offset = 0
        first_loaded_offset: int | None = None

        for bf in body_files:
            ds = netCDF4.Dataset(bf, "r")
            try:
                # Navigate to the channel group
                try:
                    grp = ds[group_path]
                except (IndexError, KeyError):
                    logger.debug(
                        "Group '{}' not found in {}; skipping",
                        group_path,
                        os.path.basename(bf),
                    )
                    continue

                if "effective_radiance" not in grp.variables:
                    logger.debug(
                        "effective_radiance not in group '{}' of {}; skipping",
                        group_path,
                        os.path.basename(bf),
                    )
                    continue

                var = grp.variables["effective_radiance"]
                shape = var.shape
                seg_rows = shape[-2] if len(shape) >= 2 else shape[0]
                seg_row_end = row_offset + seg_rows

                # Skip segments that do not overlap the ROI row range
                if pixel_roi is not None:
                    r0, r1, _c0, _c1 = pixel_roi
                    if seg_row_end <= r0 or row_offset >= r1:
                        row_offset = seg_row_end
                        continue

                if first_loaded_offset is None:
                    first_loaded_offset = row_offset

                data = var[:].filled(np.nan)

                # Apply scale_factor / add_offset if present but not
                # auto-applied (mask_and_scale is off by default in raw access)
                scale = getattr(var, "scale_factor", 1.0)
                offset = getattr(var, "add_offset", 0.0)
                if scale != 1.0 or offset != 0.0:
                    data = data.astype(np.float64) * scale + offset

                # Data may be 3-D (1, rows, cols) — squeeze
                if data.ndim == 3:
                    data = data[0]
                segments.append(data)
                row_offset = seg_row_end
            finally:
                ds.close()

        if not segments:
            raise ValueError(
                f"Channel '{channel_name}' not found in BODY segments "
                f"at {product_dir}"
            )

        full = np.concatenate(segments, axis=0)

        # Crop to bounding box if requested
        if pixel_roi is not None and first_loaded_offset is not None:
            r0, r1, c0, c1 = pixel_roi
            # Adjust row bounds relative to concatenated loaded data
            adj_r0 = max(0, r0 - first_loaded_offset)
            adj_r1 = min(full.shape[0], r1 - first_loaded_offset)
            c1 = min(full.shape[1], c1)
            full = full[adj_r0:adj_r1, c0:c1]

        return full.astype(np.float32)

    @staticmethod
    def _resolve_resolution(resolution: str) -> str:
        """Validate and normalise the resolution string.

        Parameters
        ----------
        resolution : str
            Resolution string (``'1km'`` or ``'2km'``)

        Returns
        -------
        str
            Normalised resolution string

        Raises
        ------
        ValueError
            If the resolution is not supported
        """
        res = resolution.lower().strip()
        if res not in _GRID_PARAMS:
            raise ValueError(
                f"Unsupported resolution '{resolution}'. "
                f"Must be one of {list(_GRID_PARAMS.keys())}"
            )
        return res

    def _check_resolution_consistency(self, variable: list[str]) -> None:
        """Verify all requested variables match the configured resolution.

        Parameters
        ----------
        variable : list[str]
            List of variable names

        Raises
        ------
        ValueError
            If any variable's native resolution differs from self._resolution
        """
        for v in variable:
            channel_name, _ = MetOpMTGLexicon[v]
            native_res = _VARIABLE_RESOLUTION.get(channel_name)
            if native_res is not None and native_res != self._resolution:
                raise ValueError(
                    f"Variable '{v}' (channel '{channel_name}') has native "
                    f"resolution {native_res}, but data source is configured "
                    f"for {self._resolution}. All requested variables must "
                    f"share the same resolution."
                )

    def _ensure_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute or return the cached lat/lon grid.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lat, lon) arrays
        """
        if self._grid is None:
            self._grid = MetOpMTG.grid(resolution=self._resolution)
        return self._grid

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify that requested times are valid.

        Parameters
        ----------
        times : list[datetime]
            UTC timestamps to validate

        Raises
        ------
        ValueError
            If any time is before operational start or not on the scan
            frequency interval
        """
        for time in times:
            # Make time aware if it isn't
            t = time if time.tzinfo is not None else time.replace(tzinfo=timezone.utc)
            if t < _OPERATIONAL_START:
                raise ValueError(
                    f"Requested date time {time} is before MTG-I1 FCI "
                    f"operational start ({_OPERATIONAL_START})"
                )
            if (
                int((t - datetime(2000, 1, 1, tzinfo=timezone.utc)).total_seconds())
                % cls.SCAN_FREQUENCY
                != 0
            ):
                raise ValueError(
                    f"Requested date time {time} needs to be on a "
                    f"{cls.SCAN_FREQUENCY}s interval for MetOpMTG"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "metop_mtg")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_mtg_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Check if a given date time is available.

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
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp((time - _unix) / _ds, timezone.utc)

        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True

    @staticmethod
    def grid(
        resolution: str = "2km",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) in degrees for the native MTG FCI grid.

        Parameters
        ----------
        resolution : str, optional
            Grid resolution (``'1km'`` or ``'2km'``), by default ``'2km'``

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (lat, lon) arrays in degrees. Off-Earth pixels are NaN.
        """
        res = MetOpMTG._resolve_resolution(resolution)
        cfac, coff, lfac, loff, (nrows, ncols) = _GRID_PARAMS[res]
        return _mtg_fci_scan_to_latlon(cfac, coff, lfac, loff, nrows, ncols)
