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
import functools
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

# EUMETSAT collection identifier
_COLLECTION_ID = "EO:EUM:DAT:0662"

# Scan frequency in seconds (full disk every 10 minutes)
_SCAN_FREQUENCY = 600

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


def _sort_body_key(name: str) -> int:
    """Extract numeric chunk index from a BODY segment name for sorting.

    Parameters
    ----------
    name : str
        Segment name, e.g. ``'BODY_009'``

    Returns
    -------
    int
        Numeric chunk index
    """
    m = re.search(r"(\d+)", name)
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

    COLLECTION_ID = _COLLECTION_ID
    SCAN_FREQUENCY = _SCAN_FREQUENCY

    def __init__(
        self,
        resolution: str = "2km",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 1200,
        retries: int = 3,
    ) -> None:
        self._resolution = self._resolve_resolution(resolution)
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

        # Determine output shape
        grid_size = _GRID_PARAMS[self._resolution][4]
        ny, nx = grid_size

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

        # Build fetch tasks — one per (time, variable) pair
        async_tasks = []
        for i, t in enumerate(time):
            for j, v in enumerate(variable):
                async_tasks.append((i, j, t, v))

        func_map = map(
            functools.partial(self._fetch_wrapper, xr_array=xr_array), async_tasks
        )
        await tqdm.gather(
            *func_map, desc="Fetching MTG data", disable=(not self._verbose)
        )

        # Attach grid coordinates
        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat), "_lon": (("y", "x"), lon)}
        )

        return xr_array

    async def _fetch_wrapper(
        self,
        task: tuple[int, int, datetime, str],
        xr_array: xr.DataArray,
    ) -> None:
        """Unpack a task tuple and fetch a single (time, variable) slice.

        Parameters
        ----------
        task : tuple
            ``(time_idx, var_idx, time, variable)``
        xr_array : xr.DataArray
            Output array to fill in-place
        """
        i, j, t, v = task
        data = await async_retry(
            asyncio.to_thread,
            self._fetch_channel,
            t,
            v,
            retries=self._retries,
            backoff=2.0,
            exceptions=(OSError, IOError, TimeoutError, ConnectionError),
        )
        xr_array[i, j] = data

    def _fetch_channel(
        self,
        time: datetime,
        variable: str,
    ) -> np.ndarray:
        """Download and read a single channel for one time step.

        Parameters
        ----------
        time : datetime
            UTC timestamp
        variable : str
            Variable name in MetOpMTGLexicon

        Returns
        -------
        np.ndarray
            2-D array of shape ``(ny, nx)``
        """
        channel_name, modifier = MetOpMTGLexicon[variable]

        # Download product zip from EUMETSAT
        product_dir = self._fetch_product(time)

        # Read the channel from extracted NetCDF segments
        data = self._read_channel(product_dir, channel_name)

        return modifier(data)

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

        if os.path.isdir(product_cache):
            logger.debug("Using cached MTG product: {}", product_cache)
            return product_cache

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

        with product.open() as stream:  # type: ignore[attr-defined]
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(stream, f)

        # Extract only BODY NetCDF files
        with zipfile.ZipFile(zip_path, "r") as zf:
            body_names = sorted(
                [n for n in zf.namelist() if "BODY" in n and n.endswith(".nc")],
                key=_sort_body_key,
            )
            for name in body_names:
                zf.extract(name, product_cache)

        # Remove zip to save space
        os.remove(zip_path)

    def _read_channel(
        self,
        product_dir: str,
        channel_name: str,
    ) -> np.ndarray:
        """Read a single channel from extracted BODY segments using netCDF4.

        Parameters
        ----------
        product_dir : str
            Path to extracted product directory
        channel_name : str
            FCI channel name (e.g., ``'vis_04'``)

        Returns
        -------
        np.ndarray
            2-D full disk array
        """
        # Find all BODY segment files
        body_files = [
            os.path.join(root, f)
            for root, _dirs, files in os.walk(product_dir)
            for f in files
            if "BODY" in f and f.endswith(".nc")
        ]

        body_files.sort(key=lambda p: _sort_body_key(os.path.basename(p)))

        if not body_files:
            raise FileNotFoundError(f"No BODY segment files found in {product_dir}")

        # Read and vertically stack segments
        segments: list[np.ndarray] = []
        var_key = f"{channel_name}_effective_radiance"

        for bf in body_files:
            ds = netCDF4.Dataset(bf, "r")
            try:
                if var_key in ds.variables:
                    data = ds.variables[var_key][:].filled(np.nan)
                    # Data may be 3-D (1, rows, cols) — squeeze
                    if data.ndim == 3:
                        data = data[0]
                    segments.append(data)
            finally:
                ds.close()

        if not segments:
            raise ValueError(
                f"Channel '{channel_name}' not found in BODY segments "
                f"at {product_dir}"
            )

        full = np.concatenate(segments, axis=0)

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
                % _SCAN_FREQUENCY
                != 0
            ):
                raise ValueError(
                    f"Requested date time {time} needs to be on a "
                    f"{_SCAN_FREQUENCY}s interval for MetOpMTG"
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
