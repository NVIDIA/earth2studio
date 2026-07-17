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
from itertools import product
from typing import Literal

import numpy as np
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    async_retry,
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon.meteosat import MeteosatFCILexicon
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

# Geostationary projection parameters (MTG-I1 at 0° longitude)
SUB_SATELLITE_LON = 0.0  # degrees
PERSPECTIVE_POINT_HEIGHT = 35786400.0  # metres
SEMI_MAJOR_AXIS = 6378137.0  # metres (WGS-84)
SEMI_MINOR_AXIS = 6356752.3142  # metres (WGS-84)

# Grid parameters
_GRID_SCALE: dict[Literal["2km", "1km", "500m"], int] = {"2km": 1, "1km": 2, "500m": 4}
_GRID_SIZE: dict[Literal["2km", "1km", "500m"], tuple[int, int]] = {
    res: (5568 * scale, 5568 * scale) for (res, scale) in _GRID_SCALE.items()
}
_X_SCALE: dict[Literal["2km", "1km", "500m"], float] = {
    "2km": -5.58871526031607e-05,
    "1km": -2.79435763233999e-05,
    "500m": -1.39717881644274e-05,
}
_X_OFFSET: dict[Literal["2km", "1km", "500m"], float] = {
    "2km": 0.15561777642350116,
    "1km": 0.1556038047568524,
    "500m": 0.15559681889314542,
}
_Y_SCALE: dict[Literal["2km", "1km", "500m"], float] = {
    res: -scale for (res, scale) in _X_SCALE.items()
}
_Y_OFFSET: dict[Literal["2km", "1km", "500m"], float] = {
    res: -offset for (res, offset) in _X_OFFSET.items()
}


# Channel name → native resolution string
_VARIABLE_RESOLUTION: dict[
    Literal["FDHSI", "HRFI"], dict[str, Literal["2km", "1km", "500m"]]
] = {
    "FDHSI": {
        "vis_04": "1km",
        "vis_05": "1km",
        "vis_06": "1km",
        "vis_08": "1km",
        "vis_09": "1km",
        "nir_13": "1km",
        "nir_16": "1km",
        "nir_22": "1km",
        "ir_38": "2km",
        "wv_63": "2km",
        "wv_73": "2km",
        "ir_87": "2km",
        "ir_97": "2km",
        "ir_105": "2km",
        "ir_123": "2km",
        "ir_133": "2km",
    },
    "HRFI": {
        "vis_06": "500m",
        "nir_22": "500m",
        "ir_38": "1km",
        "ir_105": "1km",
    },
}

# Operational start date for MTG-I1 FCI
_OPERATIONAL_START = datetime(2024, 1, 16, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: geostationary scan angle → lat/lon
# ---------------------------------------------------------------------------
def _mtg_fci_scan_to_latlon(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MTG FCI scan angles to latitude/longitude arrays.

    Parameters
    ----------
    x : np.ndarray
        1-D x (column) scan-angle coordinates in radians, west-to-east
        (FCI convention: FCI_X[0] is the western-most column).
    y : np.ndarray
        1-D y (row) scan-angle coordinates in radians, south-to-north
        (FCI convention: FCI_Y[0] is the southern-most row).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lat, lon)`` arrays of shape ``(len(y), len(x))`` in degrees,
        NaN where off-Earth.
    """
    x = x[None, :]
    y = y[:, None]
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    sin_x = np.sin(-x)  # minus to account for FCI east-west convention
    sin_y = np.sin(y)

    r_eq = SEMI_MAJOR_AXIS
    r_pol = SEMI_MINOR_AXIS
    H = PERSPECTIVE_POINT_HEIGHT + r_eq

    with np.errstate(invalid="ignore"):
        a = sin_x**2 + cos_x**2 * (cos_y**2 + (r_eq / r_pol) ** 2 * sin_y**2)
        b = -2.0 * H * cos_x * cos_y
        c = H**2 - r_eq**2

        discriminant = b**2 - 4.0 * c * a
        r_s = (b + np.sqrt(discriminant)) / (-2.0 * a)

        d = r_s * cos_x
        s_x = d * cos_y
        s_y = -r_s * sin_x
        s_z = d * sin_y
        H_s_x = H - s_x

        lat = np.degrees(np.arctan((r_eq / r_pol) ** 2 * s_z / np.hypot(H_s_x, s_y)))
        lon_origin_rad = np.radians(SUB_SATELLITE_LON)
        lon = np.degrees(lon_origin_rad - np.arctan(s_y / H_s_x))

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
    lat_lon_bbox: tuple[tuple[float, float], tuple[float, float]],
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert a lat/lon bounding box to pixel row/column bounds.

    Both ``[-180, 180]`` and ``[0, 360]`` longitude conventions are
    supported; longitudes are normalised to ``[-180, 180)`` before the
    grid lookup.

    Parameters
    ----------
    lat_lon_bbox : tuple[tuple[float, float], tuple[float, float]]
        ``((lat_min, lat_max), (lon_min, lon_max))`` in degrees.
    lat : np.ndarray
        2-D latitude array from the grid (NaN where off-Earth).
    lon : np.ndarray
        2-D longitude array from the grid (NaN where off-Earth).

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ``((row_start, row_end), (col_start, col_end))`` pixel bounds
        (end-exclusive) suitable for slicing.
    """
    ((lat_min, lat_max), (lon_min, lon_max)) = lat_lon_bbox

    # Normalise longitudes to [-180, 180) so both conventions work
    lon_min = _normalize_lon(lon_min)
    lon_max = _normalize_lon(lon_max)

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    rows, cols = mask.nonzero()
    if rows.size == 0:
        raise ValueError(f"No grid points fall within lat_lon_bbox={lat_lon_bbox}")
    return (int(rows.min()), int(rows.max()) + 1), (
        int(cols.min()),
        int(cols.max()) + 1,
    )


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
class MeteosatFCI:
    """EUMETSAT MTG-I FCI Level-1C Full Disk calibrated radiance data source.

    This data source provides access to Meteosat Third Generation (MTG)
    Flexible Combined Imager (FCI) Level-1C Full Disk products via the
    EUMETSAT Data Store. The 16 spectral channels in the Full Disk High
    Spectral Imagery (FDHSI) product cover visible, near-IR, water-vapour and
    infrared bands at 1 km or 2 km nadir resolution. Four channels are also
    available in the High Resolution Fast Imagery (HRFI) product at 500 m or
    1 km. Variables are read from either collection depending on which resolution
    is set.

    Parameters
    ----------
    resolution : Literal["2km", "1km", "500m"], optional
        Grid resolution — ``'2km'``, ``'1km'``, or ``'500m'``. FDHSI channels
        are available at ``'2km'`` (IR/WV bands) or ``'1km'`` (VIS/NIR bands);
        HRFI channels are available at ``'1km'`` or ``'500m'``. By default
        ``'2km'``.
    lat_lon_bbox : tuple[tuple[float, float], tuple[float, float]] | None, optional
        Bounding box ``((lat_min, lat_max), (lon_min, lon_max))`` in degrees to
        crop the full-disk image. Only BODY segments that overlap the requested
        latitude range are read, saving disk and memory. When ``None`` the full
        disk is returned, by default None
    pixel_bbox : tuple[tuple[int, int], tuple[int, int]] | None, optional
        Bounding box ``((row_start, row_end), (col_start, col_end))`` in pixel
        coordinates (native FCI order: row 0 = south, end-exclusive) to crop
        the full-disk image. Mutually exclusive with ``lat_lon_bbox``. When
        ``None`` the full disk is returned, by default None
    flip_north_south : bool, optional
        When ``True`` the output array row 0 is north and latitude increases
        downward (conventional image orientation).  When ``False`` (the
        default) the native FCI convention is preserved: row 0 is south and
        latitude increases upward.
    cache : bool, optional
        Cache downloaded products on local disk, by default True
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

    COLLECTION_ID: dict[Literal["FDHSI", "HRFI"], str] = {
        "FDHSI": "EO:EUM:DAT:0662",
        "HRFI": "EO:EUM:DAT:0665",
    }
    SCAN_FREQUENCY: int = 600

    # FCI y scan-angle (radians) for the full disk, south-to-north.
    FCI_Y: dict[Literal["2km", "1km", "500m"], np.ndarray] = {
        res: np.arange(1, _GRID_SIZE[res][0] + 1) * _Y_SCALE[res] + _Y_OFFSET[res]
        for res in _GRID_SCALE
    }

    # FCI x scan-angle (radians) for the full disk, west-to-east.
    FCI_X: dict[Literal["2km", "1km", "500m"], np.ndarray] = {
        res: np.arange(1, _GRID_SIZE[res][1] + 1) * _X_SCALE[res] + _X_OFFSET[res]
        for res in _GRID_SCALE
    }

    def __init__(
        self,
        resolution: Literal["2km", "1km", "500m"] = "2km",
        lat_lon_bbox: tuple[tuple[float, float], tuple[float, float]] | None = None,
        pixel_bbox: tuple[tuple[int, int], tuple[int, int]] | None = None,
        flip_north_south: bool = False,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 1200,
        retries: int = 3,
    ) -> None:
        if resolution not in _GRID_SCALE:
            raise ValueError(f"resolution must be one of {tuple(_GRID_SCALE)}")
        self._resolution = resolution

        if (lat_lon_bbox is not None) and (pixel_bbox is not None):
            raise ValueError(
                "At most one of lat_lon_bbox or pixel_bbox can be specified."
            )
        self._lat_lon_bbox = lat_lon_bbox
        # Pixel ROI bounds (lazily computed on first fetch when lat-lon bbox is set)
        self._pixel_roi = pixel_bbox
        self.flip_north_south = flip_north_south
        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout
        self._retries = retries
        self._tmp_cache_hash: str | None = None

        # Grid (lazily computed on first access)
        self._grid: tuple[np.ndarray, np.ndarray] | None = None

        # Credentials from environment variables
        self._consumer_key = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
        self._consumer_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")
        # Attempt read from .eumdc/credentials file
        # https://gitlab.eumetsat.int/eumetlab/data-services/eumdac/-/blob/public/eumdac/config.py?ref_type=heads#L14
        # https://gitlab.eumetsat.int/eumetlab/data-services/eumdac/-/blob/public/eumdac/cli_helpers.py?ref_type=heads#L165
        if not self._consumer_key or not self._consumer_secret:
            eumdac_credentials_file = (
                pathlib.Path(
                    os.getenv("EUMDAC_CONFIG_DIR", (pathlib.Path.home() / ".eumdac"))
                )
                / "credentials"
            )
            try:
                with open(eumdac_credentials_file) as f:
                    credentials = f.read().strip()
                key, secret = credentials.split(",", 1)
                self._consumer_key = key.strip()
                self._consumer_secret = secret.strip()
            except (OSError, ValueError):
                logger.warning(
                    "EUMETSAT_CONSUMER_KEY and/or EUMETSAT_CONSUMER_SECRET not set. "
                    "Data fetching will fail."
                )

    def available_variables(self) -> set[str]:
        """Return variables available at the current resolution.

        Returns
        -------
        set
            Set of strings indicating the variables available at the resolution
            passed to the constructor.
        """
        return {
            k
            for (k, v) in MeteosatFCILexicon.VOCAB.items()
            if self._resolution
            in (
                _VARIABLE_RESOLUTION[collection].get(v[0])
                for collection in _VARIABLE_RESOLUTION
            )
        }

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
            Variables to return.  Must be in the ``MeteosatFCILexicon``.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, y, x]``.
        """
        try:
            xr_array = _sync_async(
                self.fetch, time, variable, timeout=self.async_timeout
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
        (nrows, ncols) = _GRID_SIZE[self._resolution]
        if self._pixel_roi is not None:
            ((i0, i1), (j0, j1)) = self._pixel_roi
            ny = i1 - i0
            nx = j1 - j0
            lat_out = lat[i0:i1, j0:j1]
            lon_out = lon[i0:i1, j0:j1]
        else:
            ((i0, i1), (j0, j1)) = ((0, nrows), (0, ncols))
            ny, nx = nrows, ncols
            lat_out = lat
            lon_out = lon

        # Pre-allocate output
        y_coords = MeteosatFCI.FCI_Y[self._resolution][i0:i1]
        if self.flip_north_south:
            y_coords = y_coords[::-1]
        x_coords = MeteosatFCI.FCI_X[self._resolution][j0:j1]
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

        # Phase 1: Download products in parallel (one per unique time and collection)
        unique_times = list(dict.fromkeys(time))  # preserve order, deduplicate
        collections = {  # the collection each variable is downloaded from
            v: (
                "FDHSI"
                if _VARIABLE_RESOLUTION["FDHSI"].get(MeteosatFCILexicon[v][0])  # type: ignore[misc]
                == self._resolution
                else "HRFI"
            )
            for v in variable
        }
        downloads = list(product(unique_times, set(collections.values())))
        download_coros = [
            async_retry(
                asyncio.to_thread,
                self._fetch_product,
                t,
                collection,
                retries=self._retries,
                backoff=2.0,
                exceptions=(OSError, IOError, TimeoutError, ConnectionError),
            )
            for (t, collection) in downloads
        ]
        product_dirs_list: list[str] = await tqdm.gather(
            *download_coros,
            desc="Downloading MTG products",
            disable=(not self._verbose),
        )
        product_dir_map = dict(zip(downloads, product_dirs_list))

        # Phase 2: Read NetCDF segments sequentially (HDF5/netCDF4 is
        # not thread-safe — concurrent reads cause segfaults)
        for i, t in enumerate(time):
            for j, v in enumerate(variable):
                product_dir = product_dir_map[t, collections[v]]
                channel_name, modifier = MeteosatFCILexicon[v]  # type: ignore[misc]
                if collections[v] == "HRFI":
                    channel_name += "_hr"
                data = self._read_channel(product_dir, channel_name, self._pixel_roi)
                if self.flip_north_south:
                    data = np.flipud(data)
                xr_array[i, j] = modifier(data)

        # Attach grid coordinates
        if self.flip_north_south:
            lat_out = np.flipud(lat_out)
            lon_out = np.flipud(lon_out)
        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat_out), "_lon": (("y", "x"), lon_out)}
        )

        return xr_array

    def _fetch_product(
        self, time: datetime, collection: Literal["FDHSI", "HRFI"]
    ) -> str:
        """Download the MTG FCI product for the given time and collection.

        Parameters
        ----------
        time : datetime
            UTC timestamp
        collection : Literal["FDHSI", "HRFI"]
            Collection to download from — ``'FDHSI'`` or ``'HRFI'``

        Returns
        -------
        str
            Path to the extracted product directory
        """
        # Build a cache-friendly directory name
        time_str = time.strftime("%Y%m%dT%H%M%S")
        product_cache = os.path.join(self.cache, collection, f"mtg_{time_str}")

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
        ds_collection = datastore.get_collection(self.COLLECTION_ID[collection])

        dt_start = time - timedelta(seconds=self.SCAN_FREQUENCY // 2)
        dt_end = time + timedelta(seconds=self.SCAN_FREQUENCY // 2)

        products = ds_collection.search(dtstart=dt_start, dtend=dt_end)
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
        pixel_roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
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
        pixel_roi : tuple[tuple[int, int], tuple[int, int]] | None, optional
            ``((row_start, row_end), (col_start, col_end))`` pixel bounds.
            When ``None`` the full disk is read.

        Returns
        -------
        np.ndarray
            2-D float32 array of shape ``(ny, nx)``.  Full disk or cropped
            to *pixel_roi*.  For the ``ir_38`` and ``ir_38_hr`` channels the
            dual-range HDR encoding is applied: raw values above 4095 are
            decoded with ``warm_scale_factor`` / ``warm_add_offset`` instead
            of the standard ``scale_factor`` / ``add_offset``.
        """
        # Find all BODY segment files
        body_files = glob.glob(
            os.path.join(product_dir, "**", "*BODY*.nc"), recursive=True
        )

        body_files.sort(key=lambda p: _sort_body_key(os.path.basename(p)))

        if not body_files:
            raise FileNotFoundError(f"No BODY segment files found in {product_dir}")

        (nrows, ncols) = _GRID_SIZE[self._resolution]

        if pixel_roi:
            ((i0, i1), (j0, j1)) = pixel_roi
        else:
            ((i0, i1), (j0, j1)) = ((0, nrows), (0, ncols))

        # Read and vertically stack segments
        segments: list[np.ndarray] = []
        group_path = f"/data/{channel_name}/measured"

        # Track running row offset for selective segment loading
        segment_i_offset = 0

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
                local_nrows = shape[-2]

                if (i0 >= segment_i_offset + local_nrows) or (i1 <= segment_i_offset):
                    segment_i_offset += local_nrows
                    continue

                local_i0 = max(0, i0 - segment_i_offset)
                local_i1 = min(local_nrows, i1 - segment_i_offset)

                # Disable auto scale/offset so we can handle fill values
                # and apply the transform ourselves exactly once.
                ds.set_auto_maskandscale(False)
                raw = var[..., local_i0:local_i1, j0:j1]
                if raw.ndim == 3:
                    raw = raw[0]

                fill = getattr(var, "_FillValue", None)
                scale = getattr(var, "scale_factor", 1.0)
                offset = getattr(var, "add_offset", 0.0)

                data = raw.astype(np.float64) * scale + offset
                if channel_name in ("ir_38", "ir_38_hr"):
                    warm_scale = getattr(var, "warm_scale_factor", 1.0)
                    warm_offset = getattr(var, "warm_add_offset", 0.0)
                    hdr_mask = raw > 4095
                    data[hdr_mask] = (
                        raw[hdr_mask].astype(np.float64) * warm_scale + warm_offset
                    )

                if fill is not None:
                    data[raw == fill] = np.nan
                segments.append(data)
                segment_i_offset += local_nrows
            finally:
                ds.close()

        if not segments:
            raise ValueError(
                f"Channel '{channel_name}' not found in BODY segments "
                f"at {product_dir}"
            )

        full = np.concatenate(segments, axis=0)
        return full.astype(np.float32)

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
            channel_name, _ = MeteosatFCILexicon[v]  # type: ignore[misc]
            available_res = [
                collection_res.get(channel_name)
                for collection_res in _VARIABLE_RESOLUTION.values()
            ]
            available_res = [res for res in available_res if res is not None]
            if self._resolution not in available_res:
                raise ValueError(
                    f"Variable '{v}' (channel '{channel_name}') has available "
                    f"resolutions {available_res}, but data source is configured "
                    f"for {self._resolution}. All requested variables must "
                    "share the same resolution."
                )

    def _ensure_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the full-disk lat/lon grid for the configured resolution.

        The result is computed on the first call and cached for subsequent
        calls.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lat, lon)`` arrays of shape ``(_GRID_SIZE[resolution])``,
            in degrees.  Off-Earth pixels are NaN.
        """
        if self._grid is None:
            self._grid = MeteosatFCI.grid(resolution=self._resolution)
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
                    f"{cls.SCAN_FREQUENCY}s interval for MeteosatFCI"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "meteosat_fci")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_meteosat_fci_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Check if a given datetime is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            Whether the requested date time is available.
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
        resolution: Literal["2km", "1km", "500m"] = "2km",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) in degrees for the native MTG FCI grid.

        Parameters
        ----------
        resolution : Literal["2km", "1km", "500m"], optional
            Grid resolution — ``'2km'``, ``'1km'``, or ``'500m'``, by default
            ``'2km'``

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lat, lon)`` arrays of shape ``(_GRID_SIZE[resolution])`` in
            degrees.  Off-Earth pixels are NaN.
        """
        return _mtg_fci_scan_to_latlon(
            MeteosatFCI.FCI_X[resolution], MeteosatFCI.FCI_Y[resolution]
        )

    @staticmethod
    def projection_extent(
        resolution: Literal["2km", "1km", "500m"] = "2km",
    ) -> tuple[float, float, float, float]:
        """Return the geostationary projection extent in metres for plotting.

        Computes the ``(x_min, x_max, y_min, y_max)`` extent of the FCI
        grid in native geostationary projection coordinates.  This is
        useful for plotting with ``imshow`` on a
        :class:`cartopy.crs.Geostationary` axis.

        Parameters
        ----------
        resolution : Literal["2km", "1km", "500m"], optional
            Grid resolution — ``'2km'``, ``'1km'``, or ``'500m'``, by default
            ``'2km'``

        Returns
        -------
        tuple[float, float, float, float]
            ``(x_min, x_max, y_min, y_max)`` in metres.
        """
        fci_x = MeteosatFCI.FCI_X[resolution]
        fci_y = MeteosatFCI.FCI_Y[resolution]

        h = PERSPECTIVE_POINT_HEIGHT
        # Pixel edges span 0.5 to N+0.5; extent uses outermost edges
        x_min = (fci_x[0] - 0.5 * (fci_x[1] - fci_x[0])) * h
        x_max = (fci_x[-1] + 0.5 * (fci_x[-1] - fci_x[-2])) * h
        y_min = (fci_y[0] - 0.5 * (fci_y[1] - fci_y[0])) * h
        y_max = (fci_y[-1] + 0.5 * (fci_y[-1] - fci_y[-2])) * h

        return (
            min(x_min, x_max),
            max(x_min, x_max),
            min(y_min, y_max),
            max(y_min, y_max),
        )
