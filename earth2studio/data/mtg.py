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
import json
import os
import pathlib
import shutil
import signal
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MTGLexicon
from earth2studio.utils.type import TimeArray, VariableArray

# Native resolution (metres) for each non-HR MTG FCI channel
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

_DOWNLOAD_TIMEOUT = 1200      # seconds; covers 1 km (~800 MB) on a slow link
_MAX_DOWNLOAD_ATTEMPTS = 3    # retry transient network errors
_COPY_BUFSIZE = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Projection
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

    Pixels outside the Earth disk are NaN.
    """
    lon0 = np.deg2rad(lon_origin_deg)
    H = persp_point_height + semi_major_axis
    x2d, y2d = np.meshgrid(x_1d, y_1d)
    a = (
        np.sin(x2d) ** 2
        + np.cos(x2d) ** 2
        * (np.cos(y2d) ** 2 + (semi_major_axis ** 2 / semi_minor_axis ** 2) * np.sin(y2d) ** 2)
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


def _compute_pixel_roi(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    lat_grid: np.ndarray, lon_grid: np.ndarray,
) -> tuple[slice, slice]:
    """Return (row_slice, col_slice) bounding box of the ROI in pixel space."""
    mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
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


# ---------------------------------------------------------------------------
# Main data source
# ---------------------------------------------------------------------------

class MTG:
    """Meteosat Third Generation (MTG) FCI Full-Disk data source.

    Downloads MTG-I FCI Level-1C Full Disk data from the EUMETSAT Data Store on
    demand and returns calibrated effective-radiance arrays at the channel's
    native resolution — no spatial resampling is applied.

    **Native resolutions**

    * 2 km channels (5568 × 5568): ``wv_63``, ``wv_73``, ``ir_87``, ``ir_97``,
      ``ir_123``, ``ir_133``
    * 1 km channels (11136 × 11136): ``vis_04``, ``vis_05``, ``vis_08``,
      ``vis_09``, ``nir_13``, ``nir_16``

    All variables passed in a single :meth:`__call__` must share the same
    native resolution.  Make separate calls for 1 km and 2 km channels.

    An optional region-of-interest (*roi*) crops the spatial extent and limits
    which BODY segment files are extracted from the downloaded zip archive.

    Parameters
    ----------
    consumer_key : str
        EUMETSAT API consumer key.
    consumer_secret : str
        EUMETSAT API consumer secret.
    collection_id : str, optional
        EUMETSAT collection ID, by default ``"EO:EUM:DAT:0662"``
        (MTG-I1 FCI Level-1C Full Disk FDHSI).
    roi : tuple[float, float, float, float] | None, optional
        Region of interest as ``(lat_min, lat_max, lon_min, lon_max)`` in
        degrees.  When set, the output y × x extent is cropped and only the
        BODY segment files that overlap the requested row range are extracted
        from the product archive.
    cache : bool, optional
        Cache downloaded and extracted files locally, by default ``True``.
    verbose : bool, optional
        Show a per-timestep progress bar, by default ``True``.

    Note
    ----
    Requires ``eumdac`` and ``netcdf4``::

        pip install eumdac netcdf4

    Credentials: https://eoportal.eumetsat.int/

    Badges
    ------
    region:europe dataclass:observation product:sat
    """

    COLLECTION_ID = "EO:EUM:DAT:0662"
    SCAN_FREQUENCY = 600   # seconds (10 minutes)

    GRID_SIZE_2KM = (5568, 5568)
    GRID_SIZE_1KM = (11136, 11136)

    # MTG FCI geostationary projection constants (sub-satellite point 0°E)
    LON_ORIGIN        = 0.0
    PERSP_POINT_HEIGHT = 35786400.0
    SEMI_MAJOR_AXIS   = 6378137.0
    SEMI_MINOR_AXIS   = 6356752.314

    # Angular coordinate scaling (physical_angle_rad = pixel_index * SCALE + OFFSET)
    X_OFFSET   =  0.15561777642350116
    Y_OFFSET   = -0.15561777642350116
    X_SCALE_2KM = -5.58871526031607e-05
    Y_SCALE_2KM =  5.58871526031607e-05
    X_SCALE_1KM = X_SCALE_2KM / 2   # -2.794357630158035e-05
    Y_SCALE_1KM = Y_SCALE_2KM / 2   # +2.794357630158035e-05

    _OPERATIONAL_START = datetime(2024, 10, 1, tzinfo=timezone.utc)

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        collection_id: str = COLLECTION_ID,
        roi: tuple[float, float, float, float] | None = None,
        cache: bool = True,
        verbose: bool = True,
    ):
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._collection_id = collection_id
        self._roi = roi
        self._cache = cache
        self._verbose = verbose

        # Lazy per-resolution cache: populated on first __call__ for each resolution
        # Keys: "1km" | "2km"
        self._row_slice: dict[str, slice] = {}
        self._col_slice: dict[str, slice] = {}
        self._lat_crop: dict[str, np.ndarray] = {}
        self._lon_crop: dict[str, np.ndarray] = {}

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
            Requested UTC datetimes aligned to 10-minute intervals.
        variable : str | list[str] | VariableArray
            Earth2Studio variable IDs (e.g. ``"mtg_vis_04"``).  All variables
            must share the same native resolution (1 km or 2 km).

        Returns
        -------
        xr.DataArray
            Dimensions ``[time, variable, y, x]``.  ``_lat`` and ``_lon``
            non-dimension coordinates reflect the actual output resolution.
        """
        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)

        # Determine (and validate) a single resolution for all requested channels
        res_m = self._resolve_resolution(variable)
        res_str = "1km" if res_m == 1000 else "2km"

        # Lazily compute grid crop for this resolution
        self._ensure_grid(res_str)

        row_slice = self._row_slice[res_str]
        col_slice = self._col_slice[res_str]
        lat_crop  = self._lat_crop[res_str]
        lon_crop  = self._lon_crop[res_str]
        out_h, out_w = lat_crop.shape

        # 2km row bounds used for segment-level extraction
        row_start_2km, row_end_2km = self._roi_row_bounds_2km(res_str, row_slice)

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        xr_array = xr.DataArray(
            data=np.empty((len(time), len(variable), out_h, out_w), dtype=np.float32),
            dims=["time", "variable", "y", "x"],
            coords={"time": time, "variable": variable},
        )

        datastore = self._authenticate()

        for i, t in enumerate(
            tqdm(time, desc="Fetching MTG data", disable=not self._verbose)
        ):
            product_dir = self._fetch_product(datastore, t, row_start_2km, row_end_2km)
            for j, v in enumerate(variable):
                mtg_channel, modifier = MTGLexicon[v]
                xr_array[i, j] = modifier(
                    self._read_channel(product_dir, mtg_channel, res_str, row_slice, col_slice)
                )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), lat_crop), "_lon": (("y", "x"), lon_crop)}
        )
        return xr_array

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_resolution(self, variable: list[str]) -> int:
        """Return the single native resolution (1000 or 2000) for *variable*.

        Raises ``ValueError`` if variables have mixed resolutions.
        """
        resolutions = {_VARIABLE_RESOLUTION[MTGLexicon[v][0]] for v in variable}
        if len(resolutions) > 1:
            detail = {v: _VARIABLE_RESOLUTION[MTGLexicon[v][0]] for v in variable}
            raise ValueError(
                "All variables in a single call must share the same native resolution. "
                f"Received mixed resolutions: {detail}. "
                "Make separate calls for 1 km and 2 km channels."
            )
        return resolutions.pop()

    def _ensure_grid(self, res_str: str) -> None:
        """Lazily compute the cropped lat/lon grid for *res_str* if not cached."""
        if res_str in self._row_slice:
            return

        logger.debug(f"MTG: computing {res_str} lat/lon grid")
        lat_full, lon_full = MTG.grid(res_str)

        if self._roi is not None:
            rs, cs = _compute_pixel_roi(*self._roi, lat_full, lon_full)
        else:
            rs = cs = slice(None)

        self._row_slice[res_str] = rs
        self._col_slice[res_str] = cs
        self._lat_crop[res_str]  = lat_full[rs, cs].astype(np.float32)
        self._lon_crop[res_str]  = lon_full[rs, cs].astype(np.float32)

    def _roi_row_bounds_2km(
        self, res_str: str, row_slice: slice
    ) -> tuple[int, int]:
        """Return (row_start, row_end) in 2km pixel coordinates for segment selection."""
        if self._roi is None:
            return 0, self.GRID_SIZE_2KM[0] - 1
        row_start = row_slice.start or 0
        row_end   = (row_slice.stop - 1) if row_slice.stop else (
            self.GRID_SIZE_2KM[0] - 1 if res_str == "2km" else self.GRID_SIZE_1KM[0] - 1
        )
        if res_str == "1km":
            row_start //= 2
            row_end   //= 2
        return row_start, row_end

    def _authenticate(self):
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
        self, datastore, time: datetime, row_start_2km: int, row_end_2km: int
    ) -> str:
        """Download/extract the closest product and return its cache directory."""
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

        product = min(results, key=lambda p: abs((_sensing_start(p) - time).total_seconds()))

        cache_key = hashlib.sha256(
            f"{product}_{row_start_2km}_{row_end_2km}".encode()
        ).hexdigest()
        cache_dir = os.path.join(self.cache, cache_key)

        if os.path.isdir(cache_dir) and glob.glob(
            os.path.join(cache_dir, "**", "*BODY*.nc"), recursive=True
        ):
            logger.debug(f"MTG: using cached product {product!r}")
            return cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        self._download_product(product, cache_dir, row_start_2km, row_end_2km)
        return cache_dir

    def _download_product(
        self, product, dest_dir: str, row_start_2km: int, row_end_2km: int
    ) -> None:
        """Download zip and extract only BODY segments overlapping the 2km row range."""
        logger.debug(f"MTG: downloading product to {dest_dir}")
        tmp_path = self._download_zip(product, dest_dir)
        try:
            if not zipfile.is_zipfile(tmp_path):
                raise RuntimeError(f"Downloaded file is not a valid zip: {tmp_path}")

            with zipfile.ZipFile(tmp_path, "r") as zf:
                all_names = zf.namelist()
                body_names = sorted(n for n in all_names if "BODY" in os.path.basename(n))
                trail_names = [n for n in all_names if "TRAIL" in os.path.basename(n)]
                n_body = len(body_names)

                if n_body > 0:
                    rows_per_seg = self.GRID_SIZE_2KM[0] / n_body
                    # Add ±1 segment margin to absorb rounding errors
                    first_seg = max(0, int(row_start_2km / rows_per_seg) - 1)
                    last_seg  = min(n_body - 1, int(np.ceil(row_end_2km / rows_per_seg)) + 1)
                    to_extract = body_names[first_seg : last_seg + 1] + trail_names
                    logger.debug(
                        f"MTG: extracting BODY segments {first_seg}–{last_seg} "
                        f"of {n_body} ({len(to_extract)} files total)"
                    )
                else:
                    first_seg, last_seg = 0, 0
                    to_extract = all_names

                for name in to_extract:
                    zf.extract(name, dest_dir)

            # Persist segment metadata for use in _read_channel
            with open(os.path.join(dest_dir, "segment_info.json"), "w") as f:
                json.dump({"first_seg": first_seg, "n_body": n_body}, f)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _download_zip(self, product, dest_dir: str) -> str:
        """Download the product zip with retry and a per-attempt SIGALRM timeout.

        Returns the path to the downloaded zip file.  Caller is responsible for
        deleting it.  Retries up to ``_MAX_DOWNLOAD_ATTEMPTS`` times on transient
        network errors (timeouts, incomplete reads, connection resets).
        """
        import time as _time

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
            tmp_path: str | None = None
            # Install our timeout handler, saving whatever alarm was already armed
            # (e.g. pytest-timeout's SIGALRM) so we can restore it afterwards.
            old_handler = signal.signal(
                signal.SIGALRM,
                lambda *_: (_ for _ in ()).throw(TimeoutError("MTG download timed out")),
            )
            prev_remaining = signal.alarm(_DOWNLOAD_TIMEOUT)
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=dest_dir)
                with product.open() as src:
                    with os.fdopen(fd, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=_COPY_BUFSIZE)
                return tmp_path   # success
            except Exception as exc:
                last_exc = exc
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if attempt < _MAX_DOWNLOAD_ATTEMPTS:
                    wait = 2 ** attempt  # 2 s, 4 s
                    logger.warning(
                        f"MTG: download attempt {attempt}/{_MAX_DOWNLOAD_ATTEMPTS} failed "
                        f"({type(exc).__name__}: {exc}), retrying in {wait} s…"
                    )
                    _time.sleep(wait)
            finally:
                # Cancel our alarm and reinstate the previous one (if any).
                signal.alarm(0)
                if prev_remaining > 0:
                    signal.alarm(prev_remaining)
                signal.signal(signal.SIGALRM, old_handler)

        raise RuntimeError(
            f"MTG: all {_MAX_DOWNLOAD_ATTEMPTS} download attempts failed"
        ) from last_exc

    def _read_channel(
        self,
        folder: str,
        channel: str,
        res_str: str,
        row_slice: slice,
        col_slice: slice,
    ) -> np.ndarray:
        """Read one FCI channel from the extracted product folder.

        Reads ``*BODY*.nc`` segments, applies scale_factor/add_offset, masks
        fill pixels as NaN, then crops to the ROI.

        Returns
        -------
        np.ndarray
            float32, shape ``(out_h, out_w)``.
        """
        body_files = sorted(
            glob.glob(os.path.join(folder, "**", "*BODY*.nc"), recursive=True),
            key=lambda f: int(os.path.basename(f).rsplit("_", 1)[-1].replace(".nc", "")),
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

        # Crop to ROI when only a subset of segments was extracted
        if self._roi is not None:
            seg_info_path = os.path.join(folder, "segment_info.json")
            with open(seg_info_path) as f:
                seg_info = json.load(f)

            n_body    = seg_info["n_body"]
            first_seg = seg_info["first_seg"]

            native_size  = self.GRID_SIZE_1KM[0] if res_str == "1km" else self.GRID_SIZE_2KM[0]
            rows_per_seg = native_size / n_body
            first_seg_row = round(first_seg * rows_per_seg)

            local_start = (row_slice.start or 0) - first_seg_row
            local_end   = (row_slice.stop  or native_size) - first_seg_row
            radiance = radiance[local_start:local_end, col_slice]

        return radiance

    def _validate_time(self, times: list[datetime]) -> None:
        for t in times:
            if int(np.datetime64(t, "s").astype("int64")) % self.SCAN_FREQUENCY != 0:
                raise ValueError(
                    f"Requested time {t} is not aligned to a 10-minute interval "
                    f"(MTG FCI scans every {self.SCAN_FREQUENCY} s)."
                )

    # ------------------------------------------------------------------
    # Properties / class methods
    # ------------------------------------------------------------------

    @property
    def cache(self) -> str:
        cache_location = os.path.join(datasource_cache_root(), "mtg")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_mtg")
        return cache_location

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if *time* falls within the MTG-I1 operational window (≥ 2024-01-16).

        Does not query the EUMETSAT catalogue.
        """
        if isinstance(time, np.datetime64):
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.fromtimestamp(float((time - _unix) / _ds), tz=timezone.utc)
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        return time >= cls._OPERATIONAL_START

    @staticmethod
    def grid(resolution: str = "2km") -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) coordinate arrays for the MTG FCI full-disk grid.

        Parameters
        ----------
        resolution : str
            ``"2km"`` (5568 × 5568) or ``"1km"`` (11136 × 11136).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lat, lon)`` in degrees.  Off-disk pixels are NaN.
        """
        if resolution == "2km":
            size    = MTG.GRID_SIZE_2KM[0]
            x_scale = MTG.X_SCALE_2KM
            y_scale = MTG.Y_SCALE_2KM
        elif resolution == "1km":
            size    = MTG.GRID_SIZE_1KM[0]
            x_scale = MTG.X_SCALE_1KM
            y_scale = MTG.Y_SCALE_1KM
        else:
            raise ValueError(f"resolution must be '1km' or '2km', got {resolution!r}")
        indices = np.arange(size, dtype=np.float64)
        return _mtg_fci_scan_to_latlon(
            indices * x_scale + MTG.X_OFFSET,
            indices * y_scale + MTG.Y_OFFSET,
            MTG.LON_ORIGIN,
            MTG.PERSP_POINT_HEIGHT,
            MTG.SEMI_MAJOR_AXIS,
            MTG.SEMI_MINOR_AXIS,
        )
