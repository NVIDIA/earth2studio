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

import glob
import hashlib
import os
import pathlib
import shutil
import signal
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MTGLexicon
from earth2studio.utils.type import TimeArray, VariableArray

# Native resolution (metres) for each non-HR MTG FCI channel.
# 1 km channels are downsampled to the uniform 2 km grid before return.
_VARIABLE_RESOLUTION: dict[str, int] = {
    "vis_04": 1000,
    "vis_05": 1000,
    "vis_08": 1000,
    "vis_09": 1000,
    "nir_13": 1000,
    "nir_16": 1000,
    "wv_63":  2000,
    "wv_73":  2000,
    "ir_87":  2000,
    "ir_97":  2000,
    "ir_123": 2000,
    "ir_133": 2000,
}

_DOWNLOAD_TIMEOUT = 300   # seconds before a product download is aborted
_COPY_BUFSIZE = 16 * 1024 * 1024   # 16 MB streaming buffer


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _mtg_fci_scan_to_latlon(
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    lon_origin_deg: float = 0.0,
    persp_point_height: float = 35786400.0,
    semi_major_axis: float = 6378137.0,
    semi_minor_axis: float = 6356752.314,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert FCI scan angles (radians) to geographic lat/lon (degrees).

    Pixels outside the Earth disk are NaN in both output arrays.
    """
    lon0 = np.deg2rad(lon_origin_deg)
    H = persp_point_height + semi_major_axis
    x2d, y2d = np.meshgrid(x_1d, y_1d)

    a = (
        np.sin(x2d) ** 2
        + np.cos(x2d) ** 2
        * (
            np.cos(y2d) ** 2
            + (semi_major_axis ** 2 / semi_minor_axis ** 2) * np.sin(y2d) ** 2
        )
    )
    b = -2.0 * H * np.cos(x2d) * np.cos(y2d)
    c = H ** 2 - semi_major_axis ** 2

    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
        s_x = r_s * np.cos(x2d) * np.cos(y2d)
        s_y = -r_s * np.sin(x2d)
        s_z = r_s * np.cos(x2d) * np.sin(y2d)
        lat = np.degrees(
            np.arctan(
                (semi_major_axis ** 2 / semi_minor_axis ** 2)
                * (s_z / np.sqrt((H - s_x) ** 2 + s_y ** 2))
            )
        )
        lon = np.degrees(lon0 - np.arctan(s_y / (H - s_x)))

    return lat, lon


# ---------------------------------------------------------------------------
# Downsampling functions (1 km → 2 km, shape (H, W) → (H//2, W//2))
# ---------------------------------------------------------------------------

def _ds_average(arr: np.ndarray) -> np.ndarray:
    """2×2 average pooling.  NaN in any pixel propagates to the output block."""
    return (arr[::2, ::2] + arr[::2, 1::2] + arr[1::2, ::2] + arr[1::2, 1::2]) / 4.0


def _ds_nearest(arr: np.ndarray) -> np.ndarray:
    """Nearest-neighbour (top-left pixel of each 2×2 block)."""
    return arr[::2, ::2]


def _ds_max(arr: np.ndarray) -> np.ndarray:
    """2×2 max pooling.  NaN in any pixel propagates to the output block."""
    return np.maximum(
        np.maximum(arr[::2, ::2], arr[::2, 1::2]),
        np.maximum(arr[1::2, ::2], arr[1::2, 1::2]),
    )


_DOWNSAMPLING_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "average": _ds_average,
    "nearest": _ds_nearest,
    "max":     _ds_max,
}


# ---------------------------------------------------------------------------
# Main data source
# ---------------------------------------------------------------------------

class MTG:
    """Meteosat Third Generation (MTG) FCI Full-Disk data source.

    Downloads MTG-I FCI Level-1C Full Disk data from the EUMETSAT Data Store on
    demand, then reads the extracted NetCDF segments and returns calibrated
    effective-radiance arrays on the 2 km grid.

    1 km channels (``vis_04``, ``vis_05``, ``vis_08``, ``vis_09``, ``nir_13``,
    ``nir_16``) are downsampled to 2 km before return. The method is controlled
    by the *downsampling* argument.

    An optional region-of-interest (*roi*) crops the output to a lat/lon
    bounding box. When *roi* is set, only the BODY segment files that overlap
    the requested row range are extracted from the downloaded zip archive,
    reducing disk writes and read time.

    Parameters
    ----------
    consumer_key : str
        EUMETSAT API consumer key.
    consumer_secret : str
        EUMETSAT API consumer secret.
    collection_id : str, optional
        EUMETSAT collection ID.  Defaults to ``"EO:EUM:DAT:0665"``
        (MTG-I1 FCI Level-1C Full Disk).
    roi : tuple[float, float, float, float] | None, optional
        Region of interest as ``(lat_min, lat_max, lon_min, lon_max)`` in
        degrees.  If *None* (default), the full 5568 × 5568 disk is returned.
    downsampling : str | Callable, optional
        Method used to downsample 1 km channels to the 2 km grid.
        Built-in options: ``"average"`` (default), ``"nearest"``, ``"max"``.
        A custom callable ``(arr: np.ndarray) -> np.ndarray`` mapping
        (H, W) → (H//2, W//2) is also accepted.
    cache : bool, optional
        Cache downloaded and extracted files locally, by default ``True``.
    verbose : bool, optional
        Show a per-timestep progress bar, by default ``True``.

    Note
    ----
    Requires the ``eumdac`` and ``netcdf4`` packages::

        pip install eumdac netcdf4

    EUMETSAT credentials: https://eoportal.eumetsat.int/

    Badges
    ------
    region:europe dataclass:observation product:sat
    """

    COLLECTION_ID = "EO:EUM:DAT:0662"
    SCAN_FREQUENCY = 600   # seconds between full-disk scans (10 minutes)
    GRID_SIZE = (5568, 5568)   # full disk at 2 km resolution

    # MTG FCI geostationary projection constants (sub-satellite point 0°E)
    LON_ORIGIN = 0.0
    PERSP_POINT_HEIGHT = 35786400.0
    SEMI_MAJOR_AXIS = 6378137.0
    SEMI_MINOR_AXIS = 6356752.314

    # Angular pixel-coordinate scaling for the 2 km full-disk grid
    # physical_angle_rad = pixel_index * SCALE + OFFSET
    X_OFFSET =  0.15561777642350116
    Y_OFFSET = -0.15561777642350116
    X_SCALE  = -5.58871526031607e-05
    Y_SCALE  =  5.58871526031607e-05

    _OPERATIONAL_START = datetime(2024, 1, 16, tzinfo=timezone.utc)

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        collection_id: str = COLLECTION_ID,
        roi: tuple[float, float, float, float] | None = None,
        downsampling: str | Callable[[np.ndarray], np.ndarray] = "average",
        cache: bool = True,
        verbose: bool = True,
    ):
        if isinstance(downsampling, str) and downsampling not in _DOWNSAMPLING_FNS:
            raise ValueError(
                f"downsampling must be one of {list(_DOWNSAMPLING_FNS)} or a callable, "
                f"got {downsampling!r}"
            )
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._collection_id = collection_id
        self._downsampling = downsampling
        self._cache = cache
        self._verbose = verbose

        # Build the full lat/lon grid once; used for ROI pixel-bound computation
        # and attached as non-dimension coordinates on every returned DataArray.
        self._lat, self._lon = MTG.grid()

        self._roi = roi
        if roi is not None:
            self._row_slice, self._col_slice = self._compute_pixel_roi(*roi)
        else:
            self._row_slice = slice(None)
            self._col_slice = slice(None)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch MTG FCI data from EUMETSAT.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Requested UTC datetimes.  Must be aligned to 10-minute intervals.
        variable : str | list[str] | VariableArray
            Earth2Studio variable IDs (e.g. ``"mtg_vis_04"``).

        Returns
        -------
        xr.DataArray
            Array with dimensions ``[time, variable, y, x]`` containing
            calibrated effective radiance (mW m⁻² sr⁻¹ cm).  When *roi* is
            set, y and x span only the requested region.
        """
        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        lat_crop = self._lat[self._row_slice, self._col_slice]
        lon_crop = self._lon[self._row_slice, self._col_slice]
        out_h, out_w = lat_crop.shape

        xr_array = xr.DataArray(
            data=np.empty((len(time), len(variable), out_h, out_w), dtype=np.float32),
            dims=["time", "variable", "y", "x"],
            coords={"time": time, "variable": variable},
        )

        datastore = self._authenticate()

        # Row range used for selective BODY segment extraction
        row_start = (
            self._row_slice.start if self._row_slice.start is not None else 0
        )
        row_end = (
            (self._row_slice.stop - 1)
            if self._row_slice.stop is not None
            else self.GRID_SIZE[0] - 1
        )

        for i, t in enumerate(
            tqdm(time, desc="Fetching MTG data", disable=not self._verbose)
        ):
            product_dir = self._fetch_product(datastore, t, row_start, row_end)
            for j, v in enumerate(variable):
                mtg_channel, modifier = MTGLexicon[v]
                xr_array[i, j] = modifier(self._read_channel(product_dir, mtg_channel))

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat_crop), "_lon": (("y", "x"), lon_crop)}
        )
        return xr_array

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _authenticate(self):
        """Authenticate with EUMETSAT and return a ``eumdac.DataStore``."""
        try:
            import eumdac
        except ImportError as exc:
            raise ImportError(
                "eumdac is required for the MTG data source.  "
                "Install with:  pip install eumdac"
            ) from exc
        token = eumdac.AccessToken((self._consumer_key, self._consumer_secret))
        return eumdac.DataStore(token)

    def _fetch_product(
        self, datastore, time: datetime, row_start: int, row_end: int
    ) -> str:
        """Return the path to a local directory containing extracted NC files.

        Searches ±10 min around *time*, downloads and selectively extracts the
        closest product (if not already cached), and returns the cache dir.
        The cache key includes the row range so different ROIs are stored
        separately.
        """
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        window = timedelta(minutes=10)
        collection = datastore.get_collection(self._collection_id)
        results = list(collection.search(dtstart=time - window, dtend=time + window))

        if not results:
            raise ValueError(
                f"No MTG products found for {time.isoformat()} "
                f"(searched ±10 min in collection '{self._collection_id}')"
            )

        def _sensing_start(p) -> datetime:
            t = p.sensing_start
            return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t

        product = min(
            results,
            key=lambda p: abs((_sensing_start(p) - time).total_seconds()),
        )

        # Include row range in the cache key so ROI caches don't collide
        cache_key = hashlib.sha256(
            f"{product}_{row_start}_{row_end}".encode()
        ).hexdigest()
        cache_dir = os.path.join(self.cache, cache_key)

        if os.path.isdir(cache_dir) and glob.glob(
            os.path.join(cache_dir, "**", "*BODY*.nc"), recursive=True
        ):
            logger.debug(f"MTG: using cached product {product!r}")
            return cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        self._download_product(product, cache_dir, row_start, row_end)
        return cache_dir

    def _download_product(
        self, product, dest_dir: str, row_start: int, row_end: int
    ) -> None:
        """Download the product zip and extract only the relevant BODY segments.

        BODY files are ordered by row position; segments whose row range does
        not overlap *[row_start, row_end]* are skipped.  All TRAIL files are
        always extracted (they carry the metadata needed for calibration).
        """
        logger.debug(f"MTG: downloading product to {dest_dir}")

        tmp_path = None
        old_handler = signal.signal(
            signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError("MTG download timed out")),
        )
        signal.alarm(_DOWNLOAD_TIMEOUT)
        try:
            # Stream to a temporary file so we can do random-access zip reads
            fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=dest_dir)
            with product.open() as src:
                with os.fdopen(fd, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=_COPY_BUFSIZE)

            if not zipfile.is_zipfile(tmp_path):
                raise RuntimeError(
                    f"Downloaded file is not a valid zip archive: {tmp_path}"
                )

            with zipfile.ZipFile(tmp_path, "r") as zf:
                all_names = zf.namelist()
                body_names = sorted(
                    n for n in all_names if "BODY" in os.path.basename(n)
                )
                trail_names = [
                    n for n in all_names if "TRAIL" in os.path.basename(n)
                ]

                n_body = len(body_names)
                if n_body > 0:
                    rows_per_seg = self.GRID_SIZE[0] / n_body
                    first_seg = max(0, int(row_start / rows_per_seg))
                    last_seg = min(n_body - 1, int(np.ceil(row_end / rows_per_seg)))
                    to_extract = body_names[first_seg : last_seg + 1] + trail_names
                    logger.debug(
                        f"MTG: extracting BODY segments {first_seg}–{last_seg} "
                        f"of {n_body} ({len(to_extract)} files total)"
                    )
                else:
                    to_extract = all_names

                for name in to_extract:
                    zf.extract(name, dest_dir)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _read_channel(self, folder: str, channel: str) -> np.ndarray:
        """Read a single FCI channel from the extracted product folder.

        Opens all ``*BODY*.nc`` segment files for the channel group
        ``/data/{channel}/measured``, applies scale_factor/add_offset to get
        calibrated radiance, masks fill pixels as NaN, downsamples 1 km
        channels to 2 km, then crops to the ROI.

        Returns
        -------
        np.ndarray
            float32 array of shape ``(out_h, out_w)``.
        """
        body_files = sorted(
            glob.glob(os.path.join(folder, "**", "*BODY*.nc"), recursive=True)
        )
        if not body_files:
            raise FileNotFoundError(f"No *BODY*.nc files found under {folder!r}")

        with xr.open_mfdataset(
            body_files,
            group=f"/data/{channel}/measured",
            mask_and_scale=False,
            engine="netcdf4",
            parallel=False,
        ) as ds:
            raw = ds.effective_radiance.load()

        scale_factor = float(raw.attrs.get("scale_factor", 1.0))
        add_offset   = float(raw.attrs.get("add_offset",   0.0))
        fill_value   = raw.attrs.get("_FillValue", None)

        radiance = raw.values.astype(np.float32) * scale_factor + add_offset
        if fill_value is not None:
            radiance[raw.values == int(fill_value)] = np.nan

        # Downsample 1 km channels to 2 km
        if _VARIABLE_RESOLUTION.get(channel, 2000) == 1000:
            radiance = self._apply_downsampling(radiance)

        # When only a subset of BODY segments was extracted, the array covers
        # rows [seg_row_start, seg_row_end] of the full disk.  Compute the
        # local slice that gives exactly the ROI rows within this partial array.
        if self._roi is not None:
            n_body_total = self.GRID_SIZE[0]   # rows in full 2 km disk
            # Approx row where the loaded data starts (first extracted segment)
            loaded_rows = radiance.shape[0]
            full_rows_covered = (
                self._row_slice.stop - self._row_slice.start
                if self._row_slice.stop is not None
                else n_body_total
            )
            row_offset = self._row_slice.start - max(
                0,
                self._row_slice.start
                - (radiance.shape[0] - full_rows_covered),
            )
            # Simpler: the loaded partial array may be wider than the ROI.
            # Compute local indices relative to the start of the loaded data.
            # We know: first_seg * rows_per_seg <= row_slice.start
            # and the loaded array has shape (loaded_rows, ...).
            # Use the known rows_per_seg to find the offset.
            n_body = max(
                1,
                round(self.GRID_SIZE[0] / radiance.shape[0])
                if self._row_slice.stop is not None
                else 1,
            )
            # Recompute first_seg from the actual loaded shape
            rows_per_seg_approx = self.GRID_SIZE[0] / max(
                1, round(self.GRID_SIZE[0] / max(radiance.shape[0], 1))
            )
            first_seg_row = int(
                (self._row_slice.start or 0)
                // rows_per_seg_approx
                * rows_per_seg_approx
            )
            local_row_start = (self._row_slice.start or 0) - first_seg_row
            local_row_end   = (
                (self._row_slice.stop or self.GRID_SIZE[0]) - first_seg_row
            )
            radiance = radiance[local_row_start:local_row_end, self._col_slice]
        
        return radiance

    def _apply_downsampling(self, arr: np.ndarray) -> np.ndarray:
        """Apply the configured downsampling method."""
        if callable(self._downsampling):
            return self._downsampling(arr)
        return _DOWNSAMPLING_FNS[self._downsampling](arr)

    def _compute_pixel_roi(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for the tight bounding box of the ROI."""
        mask = (
            (self._lat >= lat_min) & (self._lat <= lat_max) &
            (self._lon >= lon_min) & (self._lon <= lon_max)
        )
        rows, cols = np.where(mask)
        if len(rows) == 0:
            raise ValueError(
                f"ROI (lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]) "
                "contains no valid pixels in the MTG FCI full-disk grid."
            )
        return (
            slice(int(rows.min()), int(rows.max()) + 1),
            slice(int(cols.min()), int(cols.max()) + 1),
        )

    def _validate_time(self, times: list[datetime]) -> None:
        for t in times:
            t_secs = int(np.datetime64(t, "s").astype("int64"))
            if t_secs % self.SCAN_FREQUENCY != 0:
                raise ValueError(
                    f"Requested time {t} is not aligned to a 10-minute interval. "
                    f"MTG FCI full-disk scans occur every {self.SCAN_FREQUENCY} s."
                )

    # ------------------------------------------------------------------
    # Properties / class methods
    # ------------------------------------------------------------------

    @property
    def cache(self) -> str:
        """Local cache directory for downloaded MTG products."""
        cache_location = os.path.join(datasource_cache_root(), "mtg")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_mtg")
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether MTG data is likely available for a given time.

        Validates that *time* is within the operational window of MTG-I1
        (on-orbit since 2024-01-16).  Does **not** query the EUMETSAT catalogue.

        Parameters
        ----------
        time : datetime | np.datetime64

        Returns
        -------
        bool
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp(
                float((time - _unix) / _ds), tz=timezone.utc
            )
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        return time >= cls._OPERATIONAL_START

    @staticmethod
    def grid() -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) coordinate arrays for the full 2 km MTG FCI grid.

        Pixels outside the Earth disk are NaN.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lat, lon)`` in degrees, each shape ``(5568, 5568)``.
        """
        indices = np.arange(MTG.GRID_SIZE[0], dtype=np.float64)
        x_1d = indices * MTG.X_SCALE + MTG.X_OFFSET
        y_1d = indices * MTG.Y_SCALE + MTG.Y_OFFSET
        return _mtg_fci_scan_to_latlon(
            x_1d, y_1d,
            MTG.LON_ORIGIN,
            MTG.PERSP_POINT_HEIGHT,
            MTG.SEMI_MAJOR_AXIS,
            MTG.SEMI_MINOR_AXIS,
        )
