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
from earth2studio.lexicon import MetOpMHSLexicon
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
# MHS EPS native binary format constants (MDR v10, 4316 bytes)
# Ref: EUMETSAT MHS PFS (TEN/97-229) and MetopDatasets.jl MHS_XXX_1B_V10.csv
# ---------------------------------------------------------------------------
_GRH_SIZE = 20  # Generic Record Header
_MDR_RECORD_CLASS = 8
_MDR_RECORD_SUBCLASS = 2
_MPHR_RECORD_CLASS = 1
_GIADR_RECORD_CLASS = 5
_GIADR_RADIANCE_SUBCLASS = 2  # GIADR-MHS-RADIANCE subclass

# MDR field offsets (relative to start of MDR record, i.e. including 20B GRH)
# The PFS CSV lists record offsets; payload offset = record_offset - 20.
_SCENE_RADIANCE_OFFSET = 83  # integer4, [5,90], SF=1e7 (mW/m²/sr/cm⁻¹)
_ANGULAR_RELATION_OFFSET = 2598  # integer2, [4,90], SF=1e2 (deg)
_EARTH_LOCATION_OFFSET = 3318  # integer4, [2,90], SF=1e4 (deg)
_TERRAIN_ELEVATION_OFFSET = 4128  # integer2, 90 (m)
_QUALITY_OFFSET = 2352  # bitst(32), scan-level quality indicator

_NUM_CHANNELS = 5
_NUM_FOVS = 90
_MDR_SIZE = 4316

# MHS scan timing
# MHS shares the same antenna assembly as AMSU-A on MetOp:
# 3 antenna revolutions per 8 seconds → scan period = 8/3 s ≈ 2.667 s
# 90 FOVs per scan → dwell per FOV ≈ 2.667/90 ≈ 29.6 ms
#
# Source: NOAA KLM User's Guide, Section 3.9 "MHS Instrument Description"
#   https://www.star.nesdis.noaa.gov/mirs/documents/0.0_NOAA_KLM_Users_Guide.pdf
_SCAN_PERIOD_S = 8.0 / 3.0  # 2.667 s per scan revolution
_FOV_DWELL_S = _SCAN_PERIOD_S / _NUM_FOVS  # ~29.6 ms per FOV step

# Planck constants for radiance → brightness temperature conversion
_C1 = 1.191062e-05  # mW/m²/sr/cm⁻⁴
_C2 = 1.4387863  # cm·K

# MHS central wavenumbers (cm⁻¹) per channel 1–5
# Frequency → wavenumber: ν(cm⁻¹) = f(GHz) / c(cm/s) × 1e9
#   where c = 29.9792458 GHz·cm.  So ν = f / 29.9792458.
#
# These are fallback values. The parser reads the actual wavenumbers from the
# GIADR-MHS-RADIANCE record in each product file (which may differ slightly
# per platform), and replaces these defaults at parse time.
_WAVENUMBERS = np.array(
    [
        89.0 / 29.9792458,  # Ch1: 89.0 GHz → 2.9687 cm⁻¹
        157.0 / 29.9792458,  # Ch2: 157.0 GHz → 5.2369 cm⁻¹
        183.311 / 29.9792458,  # Ch3: 183.311±1 GHz → 6.1146 cm⁻¹
        183.311 / 29.9792458,  # Ch4: 183.311±3 GHz → 6.1146 cm⁻¹
        190.311 / 29.9792458,  # Ch5: 190.311 GHz → 6.3481 cm⁻¹
    ],
    dtype=np.float64,
)

# Band correction coefficients A, B per channel
# These are fallback values; the parser reads the actual intercept/slope
# from the GIADR-MHS-RADIANCE record (T_corrected = intercept + slope * T*).
# For MetOp platforms, slope ≈ 1 and intercept ≈ 0 (near-identity).
_BAND_A = np.zeros(_NUM_CHANNELS, dtype=np.float64)
_BAND_B = np.ones(_NUM_CHANNELS, dtype=np.float64)

# GIADR-MHS-RADIANCE (subclass=2) field offsets for band correction parameters
# Record offsets from MHS PFS: CENTRAL_WAVENUMBER_H1 at byte 418, then
# (wavenumber, intercept, slope) × 5 channels, 12 bytes per channel.
# All are integer4 with SF=10^6.
_GIADR_RAD_BAND_OFFSET = 418  # record offset of CENTRAL_WAVENUMBER_H1
_GIADR_RAD_BAND_STRIDE = 12  # bytes per channel (wavenum + intercept + slope)

# Spacecraft ID mapping (same as AMSU-A — same satellite bus)
_SPACECRAFT_MAP = {
    "1": "metop-b",
    "2": "metop-a",
    "3": "metop-c",
    "4": "metop-sga1",
    "M01": "metop-b",
    "M02": "metop-a",
    "M03": "metop-c",
}

# Reverse map: user-facing lowercase → eumdac API display name
_EUMDAC_SAT_NAME = {
    "metop-a": "Metop-A",
    "metop-b": "Metop-B",
    "metop-c": "Metop-C",
    "metop-sga1": "Metop-SGA1",
}


def _parse_giadr_radiance(
    data: bytes, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse GIADR-MHS-RADIANCE record for band correction parameters.

    Parameters
    ----------
    data : bytes
        Full file data
    offset : int
        Start of the GIADR record (including GRH)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (wavenumbers, intercepts, slopes) each of shape (5,) float64.
        wavenumbers in cm⁻¹, intercepts in K, slopes dimensionless.
    """
    wavenumbers = np.empty(_NUM_CHANNELS, dtype=np.float64)
    intercepts = np.empty(_NUM_CHANNELS, dtype=np.float64)
    slopes = np.empty(_NUM_CHANNELS, dtype=np.float64)

    for ch in range(_NUM_CHANNELS):
        base = offset + _GIADR_RAD_BAND_OFFSET + ch * _GIADR_RAD_BAND_STRIDE
        wn_raw = struct.unpack_from(">i", data, base)[0]
        intercept_raw = struct.unpack_from(">i", data, base + 4)[0]
        slope_raw = struct.unpack_from(">i", data, base + 8)[0]
        wavenumbers[ch] = wn_raw / 1e6  # SF=10^6
        intercepts[ch] = intercept_raw / 1e6
        slopes[ch] = slope_raw / 1e6

    return wavenumbers, intercepts, slopes


def _radiance_to_bt(
    radiance: np.ndarray,
    channel_idx: int,
    wavenumbers: np.ndarray | None = None,
    band_a: np.ndarray | None = None,
    band_b: np.ndarray | None = None,
) -> np.ndarray:
    """Convert calibrated radiance to brightness temperature.

    Uses the inverse Planck function with band correction:
        T* = C2 * γ / ln(1 + C1 * γ³ / R)
        T  = A + T* * B   (intercept + slope * T*)

    Parameters
    ----------
    radiance : np.ndarray
        Calibrated radiance in mW/m²/sr/cm⁻¹
    channel_idx : int
        0-based channel index
    wavenumbers : np.ndarray | None
        Central wavenumbers (cm⁻¹) per channel. Falls back to module default.
    band_a : np.ndarray | None
        Band correction intercept per channel (K). Falls back to module default.
    band_b : np.ndarray | None
        Band correction slope per channel. Falls back to module default.

    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin
    """
    wn = wavenumbers if wavenumbers is not None else _WAVENUMBERS
    a = (band_a if band_a is not None else _BAND_A)[channel_idx]
    b = (band_b if band_b is not None else _BAND_B)[channel_idx]
    gamma = wn[channel_idx]

    # Guard against zero/negative radiance
    valid = radiance > 0
    bt = np.full_like(radiance, np.nan, dtype=np.float64)
    r = radiance[valid]
    t_star = _C2 * gamma / np.log(1.0 + _C1 * gamma**3 / r)
    bt[valid] = a + t_star * b
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


def _parse_native_mhs(data: bytes) -> pd.DataFrame:
    """Parse an MHS Level 1B EPS native format file.

    Extracts MDR (Measurement Data Record) scan lines and converts
    calibrated radiances to brightness temperatures.  Band correction
    parameters (central wavenumber, intercept, slope) are read from the
    GIADR-MHS-RADIANCE record embedded in each product file.

    Parameters
    ----------
    data : bytes
        Complete file contents of an MHS .nat file

    Returns
    -------
    pd.DataFrame
        One row per (scan_line, FOV, channel) observation with columns:
        time, class, lat, lon, elev, scan_angle, channel_index, solza,
        solaza, satellite_za, satellite_aza, quality, satellite,
        observation, variable
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

    spacecraft_id = mphr.get("SPACECRAFT_ID", "")
    satellite = _SPACECRAFT_MAP.get(spacecraft_id, f"metop-{spacecraft_id.lower()}")

    # Step 2: Scan all records — collect MDR offsets and parse GIADR
    offset = 0
    mdr_offsets: list[int] = []
    giadr_wavenumbers: np.ndarray | None = None
    giadr_intercepts: np.ndarray | None = None
    giadr_slopes: np.ndarray | None = None

    while offset + _GRH_SIZE <= file_size:
        rc, _, sc, rec_size = _parse_grh(data, offset)
        if rec_size < _GRH_SIZE or offset + rec_size > file_size:
            break
        if rc == _MDR_RECORD_CLASS and sc == _MDR_RECORD_SUBCLASS:
            mdr_offsets.append(offset)
        elif rc == _GIADR_RECORD_CLASS and sc == _GIADR_RADIANCE_SUBCLASS:
            # Parse GIADR-MHS-RADIANCE for band correction parameters
            giadr_wavenumbers, giadr_intercepts, giadr_slopes = _parse_giadr_radiance(
                data, offset
            )
        offset += rec_size

    n_scans = len(mdr_offsets)
    if n_scans == 0:
        logger.warning("No MDR records found in MHS file")
        return pd.DataFrame()

    # Use GIADR values if found, otherwise fall back to defaults
    wn = giadr_wavenumbers if giadr_wavenumbers is not None else _WAVENUMBERS
    band_a = giadr_intercepts if giadr_intercepts is not None else _BAND_A
    band_b = giadr_slopes if giadr_slopes is not None else _BAND_B

    # Step 3: Pre-allocate arrays for all scans × 90 FOVs
    n_obs = n_scans * _NUM_FOVS
    lats = np.empty(n_obs, dtype=np.float32)
    lons = np.empty(n_obs, dtype=np.float32)
    elevs = np.empty(n_obs, dtype=np.float32)
    solar_za = np.empty(n_obs, dtype=np.float32)
    sat_za = np.empty(n_obs, dtype=np.float32)
    solar_azi = np.empty(n_obs, dtype=np.float32)
    sat_azi = np.empty(n_obs, dtype=np.float32)
    # Radiances: (n_scans*90, 5) → brightness temps per channel
    radiances = np.empty((n_obs, _NUM_CHANNELS), dtype=np.float64)
    scan_times = np.empty(n_obs, dtype="datetime64[ns]")
    quality = np.empty(n_obs, dtype=np.uint16)

    # CDS epoch for per-scanline UTC time
    _CDS_EPOCH = datetime(2000, 1, 1)

    # Pre-compute per-FOV time offsets within a scan
    fov_offsets_ns = np.array(
        [np.timedelta64(int(i * _FOV_DWELL_S * 1e9), "ns") for i in range(_NUM_FOVS)]
    )

    for scan_idx, mdr_off in enumerate(mdr_offsets):
        base = scan_idx * _NUM_FOVS

        # Per-scanline UTC time from CDS short (6 bytes at record offset 22)
        # UTC_SL_TIME_DAY (uint16) + UTC_SL_TIME_MS (uint32)
        cds_day = struct.unpack_from(">H", data, mdr_off + 22)[0]
        cds_ms = struct.unpack_from(">I", data, mdr_off + 24)[0]
        scan_time = _CDS_EPOCH + timedelta(days=cds_day, milliseconds=cds_ms)
        scan_time_ns = np.datetime64(scan_time, "ns")
        scan_times[base : base + _NUM_FOVS] = scan_time_ns + fov_offsets_ns

        # SCENE_RADIANCE: integer4, [5,90] FOV-major, SF=1e7
        # Memory layout: (ch1_fov1, ch2_fov1, ..., ch5_fov1, ch1_fov2, ...)
        rad_off = mdr_off + _SCENE_RADIANCE_OFFSET
        raw_rad = struct.unpack_from(f">{_NUM_CHANNELS * _NUM_FOVS}i", data, rad_off)
        # Reshape to (90, 5) — 90 FOVs, 5 channels per FOV
        rad_array = np.array(raw_rad, dtype=np.float64).reshape(
            _NUM_FOVS, _NUM_CHANNELS
        )
        radiances[base : base + _NUM_FOVS, :] = rad_array / 1e7

        # ANGULAR_RELATION: integer2, [4,90], SF=1e2
        # Memory: (solza0,satza0,solazi0,satazi0, solza1,satza1,...)
        ang_off = mdr_off + _ANGULAR_RELATION_OFFSET
        raw_ang = struct.unpack_from(f">{4 * _NUM_FOVS}h", data, ang_off)
        ang = np.array(raw_ang, dtype=np.float32) / 100.0
        solar_za[base : base + _NUM_FOVS] = ang[0::4]
        sat_za[base : base + _NUM_FOVS] = ang[1::4]
        solar_azi[base : base + _NUM_FOVS] = ang[2::4]
        sat_azi[base : base + _NUM_FOVS] = ang[3::4]

        # EARTH_LOCATION: integer4, [2,90], SF=1e4
        # Memory: (lat0, lon0, lat1, lon1, ..., lat89, lon89)
        loc_off = mdr_off + _EARTH_LOCATION_OFFSET
        raw_loc = struct.unpack_from(f">{2 * _NUM_FOVS}i", data, loc_off)
        loc = np.array(raw_loc, dtype=np.float64) / 1e4
        lats[base : base + _NUM_FOVS] = loc[0::2].astype(np.float32)
        # Convert longitude from [-180, 180] to [0, 360]
        lon_vals = loc[1::2]
        lon_vals = np.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
        lons[base : base + _NUM_FOVS] = lon_vals.astype(np.float32)

        # TERRAIN_ELEVATION: integer2, 90 at record offset 4128
        elev_off = mdr_off + _TERRAIN_ELEVATION_OFFSET
        raw_elev = struct.unpack_from(f">{_NUM_FOVS}h", data, elev_off)
        elevs[base : base + _NUM_FOVS] = np.array(raw_elev, dtype=np.float32)

        # QUALITY_INDICATOR: bitst(32) at record offset 2352, truncated to uint16
        mdr_qual = struct.unpack_from(">I", data, mdr_off + _QUALITY_OFFSET)[0]
        quality[base : base + _NUM_FOVS] = np.uint16(mdr_qual & 0xFFFF)

    # Step 4: Convert radiances to brightness temperatures per channel
    # All 5 MHS channels are used
    valid_channels = list(range(1, MetOpMHSLexicon.MHS_NUM_CHANNELS + 1))
    n_valid_channels = len(valid_channels)

    bt_arrays: dict[int, np.ndarray] = {}
    for ch_idx in valid_channels:
        bt_arrays[ch_idx] = _radiance_to_bt(
            radiances[:, ch_idx - 1], ch_idx - 1, wn, band_a, band_b
        )

    # Step 5: Build long-format DataFrame (one row per FOV × channel)
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
    all_channel_idx = np.empty(total_rows, dtype=np.uint16)
    all_scan_angle = np.tile(sat_za, n_valid_channels)  # scan angle ≈ sat zenith
    all_quality = np.tile(quality, n_valid_channels)

    for i, ch_idx in enumerate(valid_channels):
        start = i * rows_per_channel
        end = start + rows_per_channel
        all_obs[start:end] = bt_arrays[ch_idx].astype(np.float32)
        all_channel_idx[start:end] = ch_idx

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(all_times),
            "class": "rad",
            "lat": all_lats,
            "lon": all_lons,
            "elev": all_elevs,
            "scan_angle": all_scan_angle,
            "channel_index": all_channel_idx,
            "solza": all_solza,
            "solaza": all_solaza,
            "satellite_za": all_satza,
            "satellite_aza": all_sataza,
            "quality": all_quality,
            "satellite": satellite,
            "observation": all_obs,
            "variable": "mhs",
        }
    )

    # Drop rows with invalid geolocation or NaN observations
    df = df.dropna(subset=["observation", "lat", "lon"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90)]

    return df


@check_optional_dependencies()
class MetOpMHS:
    """EUMETSAT MetOp MHS Level 1B brightness temperature observations.

    Microwave Humidity Sounder (MHS) is a 5-channel cross-track scanning
    microwave radiometer aboard the MetOp series of polar-orbiting satellites.
    It measures calibrated scene radiances at frequencies from 89 GHz to
    190.3 GHz, providing humidity profiles from the surface to the upper
    troposphere.

    MHS is the successor to AMSU-B and operates on MetOp-A (decommissioned
    2021), MetOp-B, and MetOp-C.  It will be replaced by the Microwave
    Sounder (MWS) on the MetOp Second Generation (MetOp-SG) satellites.

    The returned :class:`~pandas.DataFrame` has one row per FOV per channel,
    following the same convention as :class:`~earth2studio.data.UFSObsSat`.
    The ``channel_index`` column (1--5) identifies each channel.

    This data source downloads Level 1B products from the EUMETSAT Data Store
    and parses the EPS native binary format to extract brightness temperatures,
    geolocation, and viewing geometry for each field of view (FOV).

    Each scan line contains 90 FOVs with ~16 km spatial resolution at nadir.
    A typical orbit pass contains ~2300 scan lines.

    Parameters
    ----------
    satellite : str, optional
        Satellite platform filter for product search. One of "metop-a",
        "metop-b", "metop-c", or None (all available). By default None.
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

    - https://data.eumetsat.int/data/map/EO:EUM:DAT:METOP:MHSL1
    - https://user.eumetsat.int/s3/eup-strapi-media/pdf_atovsl1b_pg_8bbaa8ba48.pdf

    Note
    ----
    Binary format constants (MDR offsets, record size) are from the EUMETSAT
    MHS Product Format Specification (TEN/97-229, version 10) and verified
    against the MetopDatasets.jl CSV format descriptions.

    Badges
    ------
    region:global dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.metop_mhs"
    COLLECTION_ID = "EO:EUM:DAT:METOP:MHSL1"
    VALID_SATELLITES = frozenset(["metop-a", "metop-b", "metop-c"])

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("class"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("elev"),
            E2STUDIO_SCHEMA.field("scan_angle"),
            E2STUDIO_SCHEMA.field("channel_index"),
            E2STUDIO_SCHEMA.field("solza"),
            E2STUDIO_SCHEMA.field("solaza"),
            E2STUDIO_SCHEMA.field("satellite_za"),
            E2STUDIO_SCHEMA.field("satellite_aza"),
            E2STUDIO_SCHEMA.field("quality"),
            pa.field("satellite", pa.string()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
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
        if satellite is not None and satellite not in self.VALID_SATELLITES:
            raise ValueError(
                f"Invalid satellite '{satellite}'. "
                f"Valid options: {sorted(self.VALID_SATELLITES)}"
            )
        self._satellite = satellite
        # eumdac API expects title-case names (e.g. "Metop-B")
        self._eumdac_satellite = _EUMDAC_SAT_NAME[satellite] if satellite else None
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
        """Function to get MHS brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpMHSLexicon
            (e.g. ``["mhs"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            MHS observation data in long format with one row per FOV
            per channel.
        """
        try:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            df = loop.run_until_complete(
                asyncio.wait_for(
                    self.fetch(time, variable, fields), timeout=self.async_timeout
                )
            )
        finally:
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get MHS data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpMHSLexicon
            (e.g. ``["mhs"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            MHS observation data in long format with one row per FOV
            per channel.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in MetOpMHSLexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in MetOpMHSLexicon. "
                    f"Available: {list(MetOpMHSLexicon.VOCAB.keys())}"
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
                "No MHS products found for time range {} to {}", dt_min, dt_max
            )
            return self._empty_dataframe(schema)

        # Parse each product file
        frames: list[pd.DataFrame] = []
        for fpath in product_files:
            with open(fpath, "rb") as f:
                raw = f.read()
            df = _parse_native_mhs(raw)
            if not df.empty:
                frames.append(df)

        if not frames:
            return self._empty_dataframe(schema)

        result = pd.concat(frames, ignore_index=True)

        # Deduplicate rows that may appear from overlapping tolerance windows
        result = result.drop_duplicates(
            subset=[
                "time",
                "lat",
                "lon",
                "channel_index",
                "satellite",
                "variable",
            ]
        )

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
        """Download MHS products from EUMETSAT Data Store.

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
        if self._eumdac_satellite:
            # eumdac search API expects title-case satellite names
            search_kwargs["sat"] = self._eumdac_satellite

        products = collection.search(**search_kwargs)

        downloaded: list[str] = []
        for product in products:
            if self._verbose:
                logger.info(
                    "Downloading MHS product: {} ({}–{})",
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
        cache_location = os.path.join(datasource_cache_root(), "metop_mhs")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_mhs_{self._tmp_cache_hash}"
            )
        return cache_location
