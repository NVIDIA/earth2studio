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
import os
import pathlib
import shutil
import struct
import uuid
from datetime import datetime, timedelta

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MetOpAMSUALexicon
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    import eumdac
except ImportError:
    OptionalDependencyFailure("data")
    eumdac = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# AMSU-A EPS native binary format constants (MDR v4, 3464 bytes)
# ---------------------------------------------------------------------------
_GRH_SIZE = 20  # Generic Record Header
_MDR_RECORD_CLASS = 8
_MDR_RECORD_SUBCLASS = 2
_MPHR_RECORD_CLASS = 1

# MDR payload offsets (relative to start of MDR record)
_SCENE_RADIANCE_OFFSET = 22  # integer4, 15×30, SF=1e7
_ANGULAR_RELATION_OFFSET = 1842  # integer2, 4×30, SF=1e2
_EARTH_LOCATION_OFFSET = 2082  # integer4, 2×30, SF=1e4
_TERRAIN_ELEVATION_OFFSET = 2382  # integer2, 30

_NUM_CHANNELS = 15
_NUM_FOVS = 30
_MDR_SIZE = 3464

# Planck constants for radiance → brightness temperature conversion
_C1 = 1.191062e-05  # mW/m²/sr/cm⁻⁴
_C2 = 1.4387863  # cm·K

# Metop-B AMSU-A central wavenumbers (cm⁻¹) per channel 1–15
# From ATOVS L1B Product Guide, Appendix A
# NOTE: Inter-satellite wavenumber differences (Metop-A/B/C) are <0.0002 cm⁻¹,
# producing max BT bias <0.03 K — well below instrument NEdT (~0.2–0.5 K).
# Band correction coefficients (A, B) are identity (0, 1) for all NOAA-KLM
# platforms (per PGS §5.1.2.2.5). Using Metop-B values for all satellites.
_WAVENUMBERS = np.array(
    [
        0.793897,
        1.047421,
        1.677830,
        1.761235,
        1.787785,
        1.814590,
        1.832608,
        1.851295,
        1.911001,
        1.911001,
        1.911001,
        1.911001,
        1.911001,
        1.911001,
        2.968887,
    ],
    dtype=np.float64,
)

# Band correction A, B per channel (identity for all NOAA-KLM platforms)
_BAND_A = np.zeros(_NUM_CHANNELS, dtype=np.float64)
_BAND_B = np.ones(_NUM_CHANNELS, dtype=np.float64)

# Spacecraft ID mapping
_SPACECRAFT_MAP = {
    "1": "Metop-B",
    "2": "Metop-A",
    "3": "Metop-C",
    "4": "Metop-SGA1",
    "M01": "Metop-B",
    "M02": "Metop-A",
    "M03": "Metop-C",
}


def _radiance_to_bt(radiance: np.ndarray, channel_idx: int) -> np.ndarray:
    """Convert calibrated radiance to brightness temperature.

    Uses the inverse Planck function with band correction:
        T* = C2 * γ / ln(1 + C1 * γ³ / R)
        T  = (T* - A) / B

    Parameters
    ----------
    radiance : np.ndarray
        Calibrated radiance in mW/m²/sr/cm⁻¹
    channel_idx : int
        0-based channel index

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin
    """
    gamma = _WAVENUMBERS[channel_idx]
    a = _BAND_A[channel_idx]
    b = _BAND_B[channel_idx]

    # Guard against zero/negative radiance
    valid = radiance > 0
    bt = np.full_like(radiance, np.nan, dtype=np.float64)
    r = radiance[valid]
    t_star = _C2 * gamma / np.log(1.0 + _C1 * gamma**3 / r)
    bt[valid] = (t_star - a) / b
    return bt


def _parse_mphr(data: bytes) -> dict[str, str]:
    """Parse the Main Product Header Record (ASCII key=value pairs).

    Parameters
    ----------
    data : bytes
        Raw MPHR record bytes (including GRH)

    Returns
    -------
    dict[str, str]
        Key-value pairs from the header
    """
    text = data[_GRH_SIZE:].decode("ascii", errors="replace")
    result: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def _parse_grh(data: bytes, offset: int = 0) -> tuple[int, int, int, int]:
    """Parse a Generic Record Header at the given offset.

    Returns
    -------
    tuple
        (record_class, instrument_group, record_subclass, record_size)
    """
    record_class = data[offset]
    instrument_group = data[offset + 1]
    record_subclass = data[offset + 2]
    record_size = struct.unpack_from(">I", data, offset + 4)[0]
    return record_class, instrument_group, record_subclass, record_size


def _parse_sensing_time(mphr: dict[str, str]) -> tuple[datetime, datetime]:
    """Extract sensing start/end times from MPHR.

    Parameters
    ----------
    mphr : dict[str, str]
        Parsed MPHR key-value pairs

    Returns
    -------
    tuple[datetime, datetime]
        (sensing_start, sensing_end) as naive UTC datetimes
    """
    fmt = "%Y%m%d%H%M%S"
    start_str = mphr.get("SENSING_START", "")
    end_str = mphr.get("SENSING_END", "")

    # The time string may have a 'Z' suffix or extra chars
    start = datetime.strptime(start_str[:14], fmt)
    end = datetime.strptime(end_str[:14], fmt)
    return start, end


def _parse_native_amsua(data: bytes) -> pd.DataFrame:
    """Parse an AMSU-A Level 1B EPS native format file.

    Extracts MDR (Measurement Data Record) scan lines and converts
    calibrated radiances to brightness temperatures.

    Parameters
    ----------
    data : bytes
        Complete file contents of an AMSU-A .nat file

    Returns
    -------
    pd.DataFrame
        One row per (scan_line, FOV, channel) observation with columns:
        time, lat, lon, elev, observation, variable, satellite,
        scan_angle, channel_index, solza, solaza, satellite_za, satellite_aza
    """
    file_size = len(data)
    offset = 0

    # Step 1: Parse MPHR (first record)
    if file_size < _GRH_SIZE:
        return pd.DataFrame()

    rc, _, _, rec_size = _parse_grh(data, 0)
    if rc != _MPHR_RECORD_CLASS or rec_size > file_size:
        logger.warning("First record is not MPHR (class={})", rc)
        return pd.DataFrame()

    mphr = _parse_mphr(data[:rec_size])
    sensing_start, sensing_end = _parse_sensing_time(mphr)

    spacecraft_id = mphr.get("SPACECRAFT_ID", "")
    satellite = _SPACECRAFT_MAP.get(spacecraft_id, f"Metop-{spacecraft_id}")

    # Step 2: Collect MDR records
    offset = 0
    mdr_offsets: list[int] = []
    while offset + _GRH_SIZE <= file_size:
        rc, _, sc, rec_size = _parse_grh(data, offset)
        if rec_size < _GRH_SIZE or offset + rec_size > file_size:
            break
        if rc == _MDR_RECORD_CLASS and sc == _MDR_RECORD_SUBCLASS:
            mdr_offsets.append(offset)
        offset += rec_size

    n_scans = len(mdr_offsets)
    if n_scans == 0:
        logger.warning("No MDR records found in AMSU-A file")
        return pd.DataFrame()

    # Step 3: Pre-allocate arrays for all scans × 30 FOVs
    n_obs = n_scans * _NUM_FOVS
    lats = np.empty(n_obs, dtype=np.float32)
    lons = np.empty(n_obs, dtype=np.float32)
    elevs = np.empty(n_obs, dtype=np.float32)
    solar_za = np.empty(n_obs, dtype=np.float32)
    sat_za = np.empty(n_obs, dtype=np.float32)
    solar_azi = np.empty(n_obs, dtype=np.float32)
    sat_azi = np.empty(n_obs, dtype=np.float32)
    # Radiances: (n_scans*30, 15) → brightness temps per channel
    radiances = np.empty((n_obs, _NUM_CHANNELS), dtype=np.float64)
    scan_times = np.empty(n_obs, dtype="datetime64[ns]")

    # Compute per-scan time by linearly interpolating from sensing_start to sensing_end
    total_seconds = (sensing_end - sensing_start).total_seconds()
    if n_scans > 1:
        dt_per_scan = total_seconds / (n_scans - 1)
    else:
        dt_per_scan = 0.0

    for scan_idx, mdr_off in enumerate(mdr_offsets):
        base = scan_idx * _NUM_FOVS

        # Time for this scan line (linear interpolation)
        scan_time = sensing_start + timedelta(seconds=scan_idx * dt_per_scan)
        scan_time_ns = np.datetime64(scan_time, "ns")
        scan_times[base : base + _NUM_FOVS] = scan_time_ns

        # SCENE_RADIANCE: integer4, 15×30, SF=1e7 at offset 22
        # Interleaved: (ch1_fov1, ch2_fov1, ..., ch15_fov1, ch1_fov2, ...)
        rad_off = mdr_off + _SCENE_RADIANCE_OFFSET
        raw_rad = struct.unpack_from(f">{_NUM_CHANNELS * _NUM_FOVS}i", data, rad_off)
        # Reshape to (30, 15) — 30 FOVs, 15 channels per FOV
        rad_array = np.array(raw_rad, dtype=np.float64).reshape(
            _NUM_FOVS, _NUM_CHANNELS
        )
        radiances[base : base + _NUM_FOVS, :] = rad_array / 1e7

        # ANGULAR_RELATION: integer2, 4×30, SF=1e2 at offset 1842
        # Interleaved: (solza0,satza0,solazi0,satazi0, solza1,satza1,...)
        ang_off = mdr_off + _ANGULAR_RELATION_OFFSET
        raw_ang = struct.unpack_from(f">{4 * _NUM_FOVS}h", data, ang_off)
        ang = np.array(raw_ang, dtype=np.float32) / 100.0
        solar_za[base : base + _NUM_FOVS] = ang[0::4]
        sat_za[base : base + _NUM_FOVS] = ang[1::4]
        solar_azi[base : base + _NUM_FOVS] = ang[2::4]
        sat_azi[base : base + _NUM_FOVS] = ang[3::4]

        # EARTH_LOCATION: integer4, 2×30, SF=1e4 at offset 2082
        # Interleaved: (lat0, lon0, lat1, lon1, ..., lat29, lon29)
        loc_off = mdr_off + _EARTH_LOCATION_OFFSET
        raw_loc = struct.unpack_from(f">{2 * _NUM_FOVS}i", data, loc_off)
        loc = np.array(raw_loc, dtype=np.float64) / 1e4
        lats[base : base + _NUM_FOVS] = loc[0::2].astype(np.float32)
        # Convert longitude from [-180, 180] to [0, 360]
        lon_vals = loc[1::2]
        lon_vals = np.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
        lons[base : base + _NUM_FOVS] = lon_vals.astype(np.float32)

        # TERRAIN_ELEVATION: integer2, 30 at offset 2382
        elev_off = mdr_off + _TERRAIN_ELEVATION_OFFSET
        raw_elev = struct.unpack_from(f">{_NUM_FOVS}h", data, elev_off)
        elevs[base : base + _NUM_FOVS] = np.array(raw_elev, dtype=np.float32)

    # Step 4: Convert radiances to brightness temperatures per channel
    # Only include channels present in the lexicon
    channel_to_var = {v: k for k, v in MetOpAMSUALexicon.VOCAB.items()}
    valid_channels = sorted(channel_to_var.keys())  # 1-based channel indices
    n_valid_channels = len(valid_channels)

    bt_arrays: dict[int, np.ndarray] = {}
    for ch_idx in valid_channels:
        bt_arrays[ch_idx] = _radiance_to_bt(radiances[:, ch_idx - 1], ch_idx - 1)

    # Step 5: Build long-format DataFrame (one row per FOV × valid channel)
    rows_per_channel = n_obs
    total_rows = rows_per_channel * n_valid_channels

    all_times = np.tile(scan_times, n_valid_channels)
    all_lats = np.tile(lats, n_valid_channels)
    all_lons = np.tile(lons, n_valid_channels)
    all_elevs = np.tile(elevs, n_valid_channels)
    all_solza = np.tile(solar_za, n_valid_channels)
    all_solaza = np.tile(solar_azi, n_valid_channels)
    all_satza = np.tile(sat_za, n_valid_channels)
    all_sataza = np.tile(sat_azi, n_valid_channels)

    all_obs = np.empty(total_rows, dtype=np.float32)
    all_var = np.empty(total_rows, dtype=object)
    all_channel_idx = np.empty(total_rows, dtype=np.uint16)
    all_scan_angle = np.tile(sat_za, n_valid_channels)  # scan angle ≈ sat zenith

    for i, ch_idx in enumerate(valid_channels):
        start = i * rows_per_channel
        end = start + rows_per_channel
        all_obs[start:end] = bt_arrays[ch_idx].astype(np.float32)
        all_var[start:end] = channel_to_var[ch_idx]
        all_channel_idx[start:end] = ch_idx

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(all_times),
            "lat": all_lats,
            "lon": all_lons,
            "elev": all_elevs,
            "observation": all_obs,
            "variable": all_var,
            "satellite": satellite,
            "scan_angle": all_scan_angle,
            "channel_index": all_channel_idx,
            "solza": all_solza,
            "solaza": all_solaza,
            "satellite_za": all_satza,
            "satellite_aza": all_sataza,
        }
    )

    # Drop rows with invalid geolocation or NaN observations
    df = df.dropna(subset=["observation", "lat", "lon"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90)]

    return df


@check_optional_dependencies()
class MetOpAMSUA:
    """EUMETSAT MetOp AMSU-A Level 1B brightness temperature observations.

    Advanced Microwave Sounding Unit-A (AMSU-A) is a 15-channel cross-track
    scanning microwave radiometer aboard the MetOp series of polar-orbiting
    satellites. It measures calibrated scene radiances at frequencies from
    23.8 GHz to 89.0 GHz, providing atmospheric temperature profiles from
    the surface to the upper stratosphere (~48 km).

    This data source exposes **14 channels** (``amsua01`` through ``amsua14``).
    Channel 15 (89.0 GHz) is excluded because the L1B product marks ~97% of
    its measurements as missing due to quality filtering.

    This data source downloads Level 1B products from the EUMETSAT Data Store
    and parses the EPS native binary format to extract brightness temperatures,
    geolocation, and viewing geometry for each field of view (FOV).

    Each scan line contains 30 FOVs with ~47.6 km spatial resolution at nadir.
    A typical orbit pass contains ~767 scan lines (23,010 observations).

    Parameters
    ----------
    satellite : str, optional
        Satellite platform filter for product search. One of "Metop-B",
        "Metop-C", or None (all available). By default None.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(1, 'h')
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and info, by default True
    async_timeout : int, optional
        Time in seconds after which download will be cancelled if not finished,
        by default 600

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

    - https://data.eumetsat.int/data/map/EO:EUM:DAT:METOP:AMSUL1
    - https://user.eumetsat.int/s3/eup-strapi-media/pdf_atovsl1b_pg_8bbaa8ba48.pdf

    Badges
    ------
    region:global dataclass:observation product:temp product:sat
    """

    SOURCE_ID = "earth2studio.data.metop_amsua"
    COLLECTION_ID = "EO:EUM:DAT:METOP:AMSUL1"

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("elev"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
            E2STUDIO_SCHEMA.field("satellite"),
            E2STUDIO_SCHEMA.field("scan_angle"),
            E2STUDIO_SCHEMA.field("channel_index"),
            E2STUDIO_SCHEMA.field("solza"),
            E2STUDIO_SCHEMA.field("solaza"),
            E2STUDIO_SCHEMA.field("satellite_za"),
            E2STUDIO_SCHEMA.field("satellite_aza"),
        ]
    )

    def __init__(
        self,
        satellite: str | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(1, "h"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        self._satellite = satellite
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

        # Validate credentials early
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
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Function to get AMSU-A brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAMSUALexicon
            (e.g. "amsua01" through "amsua14").
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AMSU-A observation data in long format.
        """
        try:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        df = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable, fields), timeout=self.async_timeout
            )
        )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get AMSU-A data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAMSUALexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AMSU-A observation data in long format.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in MetOpAMSUALexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in MetOpAMSUALexicon. "
                    f"Available: {list(MetOpAMSUALexicon.VOCAB.keys())}"
                )

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Compute overall time window for product search
        all_times = [
            t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo else t
            for t in time_list
        ]
        dt_min = min(all_times) + self._tolerance_lower
        dt_max = max(all_times) + self._tolerance_upper

        # Download products from EUMETSAT Data Store
        product_files = await asyncio.to_thread(self._download_products, dt_min, dt_max)

        if not product_files:
            logger.warning(
                "No AMSU-A products found for time range {} to {}", dt_min, dt_max
            )
            return self._empty_dataframe(schema)

        # Parse each product file
        frames: list[pd.DataFrame] = []
        for fpath in product_files:
            with open(fpath, "rb") as f:
                raw = f.read()
            df = _parse_native_amsua(raw)
            if not df.empty:
                frames.append(df)

        if not frames:
            return self._empty_dataframe(schema)

        result = pd.concat(frames, ignore_index=True)

        # Filter by requested variables
        requested_channels = set(variable_list)
        result = result[result["variable"].isin(requested_channels)]

        # Filter by time window for each requested timestamp
        time_masks = []
        for t in all_times:
            t_min = t + self._tolerance_lower
            t_max = t + self._tolerance_upper
            mask = (result["time"] >= pd.Timestamp(t_min)) & (
                result["time"] <= pd.Timestamp(t_max)
            )
            time_masks.append(mask)

        if time_masks:
            combined_mask = time_masks[0]
            for mask in time_masks[1:]:
                combined_mask = combined_mask | mask
            result = result[combined_mask]

        # Select requested fields
        available_cols = [c for c in schema.names if c in result.columns]
        result = result[available_cols].reset_index(drop=True)

        return result

    def _download_products(self, dt_start: datetime, dt_end: datetime) -> list[str]:
        """Download AMSU-A products from EUMETSAT Data Store.

        Parameters
        ----------
        dt_start : datetime
            Search window start (UTC)
        dt_end : datetime
            Search window end (UTC)

        Returns
        -------
        list[str]
            Paths to downloaded native format files
        """
        token = eumdac.AccessToken(
            credentials=(self._consumer_key, self._consumer_secret)
        )
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(self.COLLECTION_ID)

        search_kwargs: dict = {
            "dtstart": dt_start,
            "dtend": dt_end,
        }
        if self._satellite:
            # eumdac search API expects friendly satellite names
            search_kwargs["sat"] = self._satellite

        products = collection.search(**search_kwargs)

        downloaded: list[str] = []
        for product in products:
            if self._verbose:
                logger.info(
                    "Downloading AMSU-A product: {} ({}–{})",
                    product,
                    getattr(product, "sensing_start", "?"),
                    getattr(product, "sensing_end", "?"),
                )

            # Find the .nat file entry
            nat_entry = None
            try:
                entries = product.entries
                for entry in entries:
                    if str(entry).endswith(".nat"):
                        nat_entry = entry
                        break
            except Exception:  # noqa: S110
                pass

            # Download to cache
            cache_name = f"{product}.nat"
            cache_path = os.path.join(self.cache, cache_name)
            if os.path.isfile(cache_path):
                downloaded.append(cache_path)
                continue

            try:
                with product.open(entry=nat_entry) as stream:
                    raw = stream.read()
                with open(cache_path, "wb") as f:
                    f.write(raw)
                downloaded.append(cache_path)
            except Exception as exc:
                logger.warning("Failed to download product {}: {}", product, exc)

        return downloaded

    def _empty_dataframe(self, schema: pa.Schema) -> pd.DataFrame:
        """Create an empty DataFrame matching the schema.

        Parameters
        ----------
        schema : pa.Schema
            Target schema

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with correct columns
        """
        return pd.DataFrame({name: pd.Series(dtype="object") for name in schema.names})

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification. None returns the full SCHEMA.

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields

        Raises
        ------
        KeyError
            If a requested field name is not in the SCHEMA
        TypeError
            If a field type doesn't match the SCHEMA
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
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from class SCHEMA"
                    )
            return fields

        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "metop_amsua")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_amsua_{self._tmp_cache_hash}"
            )
        return cache_location
