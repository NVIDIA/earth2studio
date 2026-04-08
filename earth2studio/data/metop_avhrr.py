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
import io
import os
import pathlib
import shutil
import struct
import uuid
import zipfile
from datetime import datetime, timedelta

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MetOpAVHRRLexicon
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
# AVHRR EPS native binary format constants
# ---------------------------------------------------------------------------
_GRH_SIZE = 20  # Generic Record Header

# Record class identifiers
_MPHR_RECORD_CLASS = 1
_SPHR_RECORD_CLASS = 2
_GIADR_RECORD_CLASS = 5
_MDR_RECORD_CLASS = 8

# Record subclass identifiers
_GIADR_RADIANCE_SUBCLASS = 1
_MDR_SUBCLASS = 2

# Number of channels and navigation points
_NUM_CHANNELS = 5  # Channels 1, 2, 3(a/b), 4, 5
_DEFAULT_EARTH_VIEWS = 2048  # Pixels per scan line
_NAV_NUM_POINTS = 103  # Tie-point navigation positions per scan line

# Radiance scale factors per channel (raw_value * scale = physical radiance)
# Derived from EPS PFS: Ch1,2,4,5 have SF=10^2, Ch3 has SF=10^4
# Formula: scale = 10 / float("10e" + N) where SF = "10^N"
_RADIANCE_SCALES = np.array([0.01, 0.01, 0.0001, 0.01, 0.01], dtype=np.float64)

# Geolocation and angular scale factors
_EARTH_LOCATION_SCALE = 0.0001  # 10^4 → degrees
_ANGULAR_RELATION_SCALE = 0.01  # 10^2 → degrees

# GIADR-radiance calibration field offsets (relative to payload start, after GRH)
# Solar irradiance for visible channels (integer2, scale 0.1 → W/m²)
_GIADR_CH1_SOLAR_IRRAD_OFFSET = 62  # 2 bytes
_GIADR_CH2_SOLAR_IRRAD_OFFSET = 66  # 2 bytes
_GIADR_CH3A_SOLAR_IRRAD_OFFSET = 70  # 2 bytes
# Thermal channel calibration (integer4)
_GIADR_CH3B_WAVENUMBER_OFFSET = 74  # scale 0.01 cm⁻¹
_GIADR_CH3B_CONSTANT1_OFFSET = 78  # scale 0.00001 K
_GIADR_CH3B_CONSTANT2_OFFSET = 82  # scale 0.000001 K/K
_GIADR_CH4_WAVENUMBER_OFFSET = 86  # scale 0.001 cm⁻¹
_GIADR_CH4_CONSTANT1_OFFSET = 90  # scale 0.00001 K
_GIADR_CH4_CONSTANT2_OFFSET = 94  # scale 0.000001 K/K
_GIADR_CH5_WAVENUMBER_OFFSET = 98  # scale 0.001 cm⁻¹
_GIADR_CH5_CONSTANT1_OFFSET = 102  # scale 0.00001 K
_GIADR_CH5_CONSTANT2_OFFSET = 106  # scale 0.000001 K/K

# MDR field offsets (relative to start of MDR record, including GRH)
# Computed from EPS PFS Annex: GRH(20) + DEGRADED(2) + EARTH_VIEWS(2)
_MDR_SCENE_RADIANCES_OFFSET = 24  # integer2, (5, 2048)
_MDR_SCENE_RADIANCES_SIZE = _NUM_CHANNELS * _DEFAULT_EARTH_VIEWS * 2  # 20480 bytes

# After SCENE_RADIANCES: TIME_ATTITUDE(4) + EULER(6) + NAV_STATUS(4) + ALT(4)
_MDR_ANG_REL_FIRST_OFFSET = (
    _MDR_SCENE_RADIANCES_OFFSET + _MDR_SCENE_RADIANCES_SIZE + 18
)  # integer2, (4,)
_MDR_ANG_REL_LAST_OFFSET = _MDR_ANG_REL_FIRST_OFFSET + 8  # integer2, (4,)
_MDR_EARTH_LOC_FIRST_OFFSET = _MDR_ANG_REL_LAST_OFFSET + 8  # integer4, (2,)
_MDR_EARTH_LOC_LAST_OFFSET = _MDR_EARTH_LOC_FIRST_OFFSET + 8  # integer4, (2,)
_MDR_NUM_NAV_POINTS_OFFSET = _MDR_EARTH_LOC_LAST_OFFSET + 8  # integer2
_MDR_ANG_REL_OFFSET = _MDR_NUM_NAV_POINTS_OFFSET + 2  # integer2, (103, 4)
_MDR_ANG_REL_SIZE = _NAV_NUM_POINTS * 4 * 2  # 824 bytes
_MDR_EARTH_LOC_OFFSET = _MDR_ANG_REL_OFFSET + _MDR_ANG_REL_SIZE  # integer4, (103, 2)
_MDR_EARTH_LOC_SIZE = _NAV_NUM_POINTS * 2 * 4  # 824 bytes

# FRAME_INDICATOR offset: after EARTH_LOCATIONS + quality + calibration + cloud + frame_sync
# This is deep in the record; we compute it relative to known fields
# Quality(4) + ScanLine(4) + CalQuality(6) + CountError(2) +
# vis calib (15*12=180) + ir calib (9*12=108) +
# CLOUD_INFORMATION(2048*2=4096) + FRAME_SYNC(12)
_MDR_QUALITY_OFFSET = _MDR_EARTH_LOC_OFFSET + _MDR_EARTH_LOC_SIZE  # u4
_MDR_CLOUD_INFO_OFFSET = _MDR_QUALITY_OFFSET + 4 + 4 + 6 + 2 + 180 + 108  # u2, (2048,)
_MDR_FRAME_SYNC_OFFSET = _MDR_CLOUD_INFO_OFFSET + 4096  # u2, (6,)
_MDR_FRAME_INDICATOR_OFFSET = _MDR_FRAME_SYNC_OFFSET + 12  # u4

# FRAME_INDICATOR bit 16: 0=channel 3b, 1=channel 3a
_FRAME_IND_3A_MASK = 1 << 16

# Planck constants for brightness temperature conversion
_C1 = 1.191062e-05  # mW/(m²·sr·cm⁻⁴)
_C2 = 1.4387863  # K·cm

# Spacecraft ID mapping (from MPHR)
_SPACECRAFT_MAP = {
    "1": "Metop-B",
    "2": "Metop-A",
    "3": "Metop-C",
    "4": "Metop-SGA1",
    "M01": "Metop-B",
    "M02": "Metop-A",
    "M03": "Metop-C",
}


# ---------------------------------------------------------------------------
# Calibration data class
# ---------------------------------------------------------------------------
class _AVHRRCalibration:
    """Calibration coefficients parsed from GIADR-radiance record."""

    __slots__ = (
        "ch1_solar_irrad",
        "ch2_solar_irrad",
        "ch3a_solar_irrad",
        "ch3b_wavenumber",
        "ch3b_a",
        "ch3b_b",
        "ch4_wavenumber",
        "ch4_a",
        "ch4_b",
        "ch5_wavenumber",
        "ch5_a",
        "ch5_b",
    )

    def __init__(self) -> None:
        # Visible channel solar irradiances (W/m²)
        self.ch1_solar_irrad: float = 0.0
        self.ch2_solar_irrad: float = 0.0
        self.ch3a_solar_irrad: float = 0.0
        # Thermal channel calibration: wavenumber (cm⁻¹), A (K), B (K/K)
        self.ch3b_wavenumber: float = 0.0
        self.ch3b_a: float = 0.0
        self.ch3b_b: float = 1.0
        self.ch4_wavenumber: float = 0.0
        self.ch4_a: float = 0.0
        self.ch4_b: float = 1.0
        self.ch5_wavenumber: float = 0.0
        self.ch5_a: float = 0.0
        self.ch5_b: float = 1.0


# ---------------------------------------------------------------------------
# Binary format parsing functions
# ---------------------------------------------------------------------------
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


def _parse_mphr(data: bytes, rec_size: int) -> dict[str, str]:
    """Parse the Main Product Header Record (ASCII key=value pairs).

    Parameters
    ----------
    data : bytes
        Raw MPHR record bytes (including GRH)
    rec_size : int
        Total record size in bytes

    Returns
    -------
    dict[str, str]
        Key-value pairs from the header
    """
    text = data[_GRH_SIZE:rec_size].decode("ascii", errors="replace")
    result: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


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
    start = datetime.strptime(start_str[:14], fmt)
    end = datetime.strptime(end_str[:14], fmt)
    return start, end


def _parse_giadr_radiance(data: bytes, offset: int, rec_size: int) -> _AVHRRCalibration:
    """Parse a GIADR-radiance record for calibration coefficients.

    Parameters
    ----------
    data : bytes
        Complete file bytes
    offset : int
        Start of GIADR record (including GRH)
    rec_size : int
        Total record size

    Returns
    -------
    _AVHRRCalibration
        Parsed calibration coefficients
    """
    cal = _AVHRRCalibration()
    payload = offset + _GRH_SIZE

    # Visible solar irradiances (integer2, scale 0.1 → W/m²)
    cal.ch1_solar_irrad = (
        struct.unpack_from(">h", data, payload + _GIADR_CH1_SOLAR_IRRAD_OFFSET)[0] * 0.1
    )
    cal.ch2_solar_irrad = (
        struct.unpack_from(">h", data, payload + _GIADR_CH2_SOLAR_IRRAD_OFFSET)[0] * 0.1
    )
    cal.ch3a_solar_irrad = (
        struct.unpack_from(">h", data, payload + _GIADR_CH3A_SOLAR_IRRAD_OFFSET)[0]
        * 0.1
    )

    # Ch3B: wavenumber (integer4, different scales)
    # CH3B_CENTRAL_WAVENUMBER: SF=10^2 → scale=0.01
    cal.ch3b_wavenumber = (
        struct.unpack_from(">i", data, payload + _GIADR_CH3B_WAVENUMBER_OFFSET)[0]
        * 0.01
    )
    cal.ch3b_a = (
        struct.unpack_from(">i", data, payload + _GIADR_CH3B_CONSTANT1_OFFSET)[0]
        * 0.00001
    )
    cal.ch3b_b = (
        struct.unpack_from(">i", data, payload + _GIADR_CH3B_CONSTANT2_OFFSET)[0]
        * 0.000001
    )

    # Ch4: wavenumber SF=10^3 → scale=0.001
    cal.ch4_wavenumber = (
        struct.unpack_from(">i", data, payload + _GIADR_CH4_WAVENUMBER_OFFSET)[0]
        * 0.001
    )
    cal.ch4_a = (
        struct.unpack_from(">i", data, payload + _GIADR_CH4_CONSTANT1_OFFSET)[0]
        * 0.00001
    )
    cal.ch4_b = (
        struct.unpack_from(">i", data, payload + _GIADR_CH4_CONSTANT2_OFFSET)[0]
        * 0.000001
    )

    # Ch5: wavenumber SF=10^3 → scale=0.001
    cal.ch5_wavenumber = (
        struct.unpack_from(">i", data, payload + _GIADR_CH5_WAVENUMBER_OFFSET)[0]
        * 0.001
    )
    cal.ch5_a = (
        struct.unpack_from(">i", data, payload + _GIADR_CH5_CONSTANT1_OFFSET)[0]
        * 0.00001
    )
    cal.ch5_b = (
        struct.unpack_from(">i", data, payload + _GIADR_CH5_CONSTANT2_OFFSET)[0]
        * 0.000001
    )

    return cal


def _radiance_to_bt(
    radiance: np.ndarray, wavenumber: float, a: float, b: float
) -> np.ndarray:
    """Convert spectral radiance to brightness temperature.

    Uses inverse Planck with band correction: BT = A + B * (C2*ν / ln(1 + C1*ν³/L))

    Parameters
    ----------
    radiance : np.ndarray
        Spectral radiance in mW/(m²·sr·cm⁻¹)
    wavenumber : float
        Central wavenumber (cm⁻¹)
    a : float
        Band correction intercept (K)
    b : float
        Band correction slope (K/K)

    Returns
    -------
    np.ndarray
        Brightness temperature (K)
    """
    valid = radiance > 0
    bt = np.full_like(radiance, np.nan, dtype=np.float64)
    r = radiance[valid]
    t_planck = _C2 * wavenumber / np.log(1.0 + _C1 * wavenumber**3 / r)
    bt[valid] = a + b * t_planck
    return bt


def _radiance_to_refl(radiance: np.ndarray, solar_irrad: float) -> np.ndarray:
    """Convert radiance to reflectance percentage.

    Formula: reflectance(%) = radiance * π * 100 / solar_irradiance

    Parameters
    ----------
    radiance : np.ndarray
        Radiance in W/(m²·sr)
    solar_irrad : float
        Solar filtered irradiance (W/m²)

    Returns
    -------
    np.ndarray
        Reflectance (%)
    """
    valid = (radiance > 0) & (solar_irrad > 0)
    refl = np.full_like(radiance, np.nan, dtype=np.float64)
    refl[valid] = radiance[valid] * np.pi * 100.0 / solar_irrad
    return refl


def _parse_native_avhrr(
    data: bytes,
    subsample: int = 1,
) -> pd.DataFrame:
    """Parse an AVHRR Level 1B EPS native binary file.

    Reads the binary file record by record, extracts calibration
    coefficients from GIADR, and measurement data from MDR records.
    Visible channels are calibrated to reflectance (%); thermal channels
    are calibrated to brightness temperature (K).

    Parameters
    ----------
    data : bytes
        Complete file contents of an AVHRR .nat file
    subsample : int, optional
        Scan-line subsampling factor, by default 1.
        Across-track uses 103 EPS navigation tie points.

    Returns
    -------
    pd.DataFrame
        One row per (pixel, channel) observation with ``variable="avhrr"``,
        ``class="refl"`` for visible channels and ``class="rad"`` for thermal.
    """
    file_size = len(data)
    if file_size < _GRH_SIZE:
        return pd.DataFrame()

    # Step 1: Parse MPHR (first record)
    rc, _, _, rec_size = _parse_grh(data, 0)
    if rc != _MPHR_RECORD_CLASS or rec_size > file_size:
        logger.warning("First record is not MPHR (class={})", rc)
        return pd.DataFrame()

    mphr = _parse_mphr(data, rec_size)
    sensing_start, sensing_end = _parse_sensing_time(mphr)

    spacecraft_id = mphr.get("SPACECRAFT_ID", "")
    satellite = _SPACECRAFT_MAP.get(spacecraft_id, f"Metop-{spacecraft_id}")

    # Step 2: Scan all records — find GIADR-radiance and MDR offsets
    calibration: _AVHRRCalibration | None = None
    mdr_offsets: list[int] = []
    offset = 0

    while offset + _GRH_SIZE <= file_size:
        rc, _, sc, rec_size = _parse_grh(data, offset)
        if rec_size < _GRH_SIZE or offset + rec_size > file_size:
            break

        if rc == _GIADR_RECORD_CLASS and sc == _GIADR_RADIANCE_SUBCLASS:
            calibration = _parse_giadr_radiance(data, offset, rec_size)

        elif rc == _MDR_RECORD_CLASS and sc == _MDR_SUBCLASS:
            mdr_offsets.append(offset)

        offset += rec_size

    if calibration is None:
        logger.warning("No GIADR-radiance record found in AVHRR file")
        return pd.DataFrame()

    n_total_scans = len(mdr_offsets)
    if n_total_scans == 0:
        logger.warning("No MDR records found in AVHRR file")
        return pd.DataFrame()

    # Step 3: Subsample scan lines
    selected_scans = list(range(0, n_total_scans, subsample))
    n_scans = len(selected_scans)
    n_pixels = n_scans * _NAV_NUM_POINTS

    # Step 4: Channel map — always extract all channels.
    # Channel index in SCENE_RADIANCES: 0=ch1, 1=ch2, 2=ch3a/3b, 3=ch4, 4=ch5
    # channel_index values match MetOpAVHRRLexicon.AVHRR_CHANNEL_INDEX
    channel_map: dict[str, dict] = {
        "1": {"rad_idx": 0, "type": "vis", "ch_num": 1},
        "2": {"rad_idx": 1, "type": "vis", "ch_num": 2},
        "3a": {"rad_idx": 2, "type": "vis", "ch_num": 3},
        "3b": {"rad_idx": 2, "type": "ir", "ch_num": 4},
        "4": {"rad_idx": 3, "type": "ir", "ch_num": 5},
        "5": {"rad_idx": 4, "type": "ir", "ch_num": 6},
    }

    all_channels = channel_map

    # Step 5: Pre-allocate arrays for geolocation (shared across channels)
    lats = np.empty(n_pixels, dtype=np.float32)
    lons = np.empty(n_pixels, dtype=np.float32)
    solza = np.empty(n_pixels, dtype=np.float32)
    satza = np.empty(n_pixels, dtype=np.float32)
    solazi = np.empty(n_pixels, dtype=np.float32)
    satazi = np.empty(n_pixels, dtype=np.float32)
    times = np.empty(n_pixels, dtype="datetime64[ns]")

    # Pre-allocate raw radiance for all 5 channels at tie points
    raw_radiances = np.empty((n_pixels, _NUM_CHANNELS), dtype=np.float64)
    # Frame indicator per scan line (for 3a/3b switching)
    frame_indicators = np.empty(n_scans, dtype=np.uint32)
    # Scan-level quality indicator (uint32 bitmask, truncated to uint16)
    quality_per_pixel = np.zeros(n_pixels, dtype=np.uint16)

    # Compute per-scan time interpolation
    total_seconds = (sensing_end - sensing_start).total_seconds()
    if n_total_scans > 1:
        dt_per_scan = total_seconds / (n_total_scans - 1)
    else:
        dt_per_scan = 0.0

    # Pixel indices of the 103 navigation tie points across track
    nav_cols = np.arange(4, 4 + _NAV_NUM_POINTS * 20, 20)[:_NAV_NUM_POINTS]

    # Step 6: Read MDR records
    for i, scan_global_idx in enumerate(selected_scans):
        mdr_off = mdr_offsets[scan_global_idx]
        base = i * _NAV_NUM_POINTS

        # Time for this scan line
        scan_time = sensing_start + timedelta(seconds=scan_global_idx * dt_per_scan)
        times[base : base + _NAV_NUM_POINTS] = np.datetime64(scan_time, "ns")

        # SCENE_RADIANCES: integer2, (5, 2048) at MDR offset 24
        # We only need values at the 103 tie-point pixel columns
        rad_off = mdr_off + _MDR_SCENE_RADIANCES_OFFSET
        for ch in range(_NUM_CHANNELS):
            for tp in range(_NAV_NUM_POINTS):
                pixel_col = nav_cols[tp]
                byte_off = rad_off + (ch * _DEFAULT_EARTH_VIEWS + pixel_col) * 2
                raw_val = struct.unpack_from(">h", data, byte_off)[0]
                raw_radiances[base + tp, ch] = raw_val * _RADIANCE_SCALES[ch]

        # ANGULAR_RELATIONS: integer2, (103, 4) — interleaved per point
        ang_off = mdr_off + _MDR_ANG_REL_OFFSET
        raw_ang = struct.unpack_from(f">{_NAV_NUM_POINTS * 4}h", data, ang_off)
        ang = np.array(raw_ang, dtype=np.float32) * _ANGULAR_RELATION_SCALE
        # Shape: (103*4,) interleaved as (solza0,satza0,solazi0,satazi0,solza1,...)
        solza[base : base + _NAV_NUM_POINTS] = ang[0::4]
        satza[base : base + _NAV_NUM_POINTS] = ang[1::4]
        solazi[base : base + _NAV_NUM_POINTS] = ang[2::4]
        satazi[base : base + _NAV_NUM_POINTS] = ang[3::4]

        # EARTH_LOCATIONS: integer4, (103, 2) — interleaved (lat0,lon0,lat1,lon1,...)
        loc_off = mdr_off + _MDR_EARTH_LOC_OFFSET
        raw_loc = struct.unpack_from(f">{_NAV_NUM_POINTS * 2}i", data, loc_off)
        loc = np.array(raw_loc, dtype=np.float64) * _EARTH_LOCATION_SCALE
        lat_vals = loc[0::2].astype(np.float32)
        lon_vals = loc[1::2]
        lon_vals = np.where(lon_vals < 0, lon_vals + 360.0, lon_vals).astype(np.float32)
        lats[base : base + _NAV_NUM_POINTS] = lat_vals
        lons[base : base + _NAV_NUM_POINTS] = lon_vals

        # FRAME_INDICATOR: u4
        frame_indicators[i] = struct.unpack_from(
            ">I", data, mdr_off + _MDR_FRAME_INDICATOR_OFFSET
        )[0]

        # MDR_QUALITY: u4 scan-level quality bitmask, truncated to uint16
        mdr_qual = struct.unpack_from(">I", data, mdr_off + _MDR_QUALITY_OFFSET)[0]
        quality_per_pixel[base : base + _NAV_NUM_POINTS] = np.uint16(
            mdr_qual & 0xFFFF
        )

    # Step 7: Calibrate and build per-channel DataFrames
    frames: list[pd.DataFrame] = []

    for ch_key, ch_info in all_channels.items():
        rad_idx = ch_info["rad_idx"]
        ch_type = ch_info["type"]
        ch_num = ch_info["ch_num"]
        obs_class = "refl" if ch_type == "vis" else "rad"

        raw_rad = raw_radiances[:, rad_idx].copy()

        if ch_type == "vis":
            # Visible channel → reflectance (%)
            solar_irrad_map = {
                0: calibration.ch1_solar_irrad,
                1: calibration.ch2_solar_irrad,
                2: calibration.ch3a_solar_irrad,
            }
            solar_irrad = solar_irrad_map[rad_idx]
            obs = _radiance_to_refl(raw_rad, solar_irrad).astype(np.float32)

            # For ch3a: mask out scan lines where 3b is active (bit16=0)
            if ch_key == "3a":
                for i_scan in range(n_scans):
                    if (frame_indicators[i_scan] & _FRAME_IND_3A_MASK) == 0:
                        base = i_scan * _NAV_NUM_POINTS
                        obs[base : base + _NAV_NUM_POINTS] = np.nan
        else:
            # Thermal channel → brightness temperature (K)
            ir_cal_map = {
                "3b": (
                    calibration.ch3b_wavenumber,
                    calibration.ch3b_a,
                    calibration.ch3b_b,
                ),
                "4": (
                    calibration.ch4_wavenumber,
                    calibration.ch4_a,
                    calibration.ch4_b,
                ),
                "5": (
                    calibration.ch5_wavenumber,
                    calibration.ch5_a,
                    calibration.ch5_b,
                ),
            }
            wn, a, b = ir_cal_map[ch_key]
            obs = _radiance_to_bt(raw_rad, wn, a, b).astype(np.float32)

            # For ch3b: mask out scan lines where 3a is active (bit16=1)
            if ch_key == "3b":
                for i_scan in range(n_scans):
                    if (frame_indicators[i_scan] & _FRAME_IND_3A_MASK) != 0:
                        base_idx = i_scan * _NAV_NUM_POINTS
                        obs[base_idx : base_idx + _NAV_NUM_POINTS] = np.nan

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(times),
                "class": obs_class,
                "lat": lats,
                "lon": lons,
                "scan_angle": satza,
                "channel_index": np.full(n_pixels, ch_num, dtype=np.uint16),
                "solza": solza,
                "solaza": solazi,
                "satellite_za": satza,
                "satellite_aza": satazi,
                "quality": quality_per_pixel,
                "satellite": satellite,
                "observation": obs,
                "variable": "avhrr",
            }
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Drop invalid observations (NaN obs, NaN coords, out-of-range lat)
    result = result.dropna(subset=["observation", "lat", "lon"])
    result = result[(result["lat"] >= -90) & (result["lat"] <= 90)]

    return result


@check_optional_dependencies()
class MetOpAVHRR:
    """EUMETSAT MetOp AVHRR Level 1B radiance and brightness temperature observations.

    The Advanced Very High Resolution Radiometer (AVHRR) is a 6-channel
    cross-track scanning radiometer aboard the MetOp series of polar-orbiting
    satellites. It measures calibrated radiances in visible (0.58-1.6 µm)
    and infrared (3.7-12.5 µm) bands at 1 km spatial resolution.

    Channels 1-2 and 3A provide reflectances (visible/NIR), while channels
    3B, 4, and 5 provide brightness temperatures (thermal IR). Channels 3A
    and 3B cannot operate simultaneously.

    The returned :class:`~pandas.DataFrame` has one row per pixel per channel,
    following the same convention as :class:`~earth2studio.data.UFSObsSat`.
    The ``channel_index`` column (1--6) identifies each channel.  The ``class``
    column differentiates observation types: ``"refl"`` for visible/NIR
    channels (1, 2, 3A) and ``"rad"`` for thermal IR channels (3B, 4, 5).

    This data source downloads Level 1B products from the EUMETSAT Data Store
    and parses the EPS native binary format directly. Calibration coefficients
    are read from the GIADR-radiance record; visible channels are converted
    to reflectance (%), thermal channels to brightness temperature (K).

    Geolocation uses the 103 EPS navigation tie points per scan line
    (~20 km spacing), avoiding expensive full-resolution interpolation.

    Parameters
    ----------
    satellite : str, optional
        Satellite platform filter. One of "Metop-B", "Metop-C", or None
        (all available). By default None.
    subsample : int, optional
        Scan-line subsampling factor. AVHRR produces ~1000+ scan lines per
        orbit; subsampling reduces data volume. By default 1.
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
    of data to your local machine for large requests. Each AVHRR orbit file
    is approximately 1 GB.

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

    - https://data.eumetsat.int/data/map/EO:EUM:DAT:METOP:AVHRRL1
    - https://user.eumetsat.int/s3/eup-strapi-media/pdf_ten_97231_eps_avhrr_l1_pgs_d2b7482b08.pdf

    Badges
    ------
    region:global dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.metop_avhrr"
    COLLECTION_ID = "EO:EUM:DAT:METOP:AVHRRL1"

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("class"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
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
        subsample: int = 1,
        time_tolerance: TimeTolerance = np.timedelta64(1, "h"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        self._satellite = satellite
        self._subsample = subsample
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

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
        """Function to get AVHRR radiance/brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAVHRRLexicon
            (e.g. ``["avhrr"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AVHRR observation data in long format with one row per pixel
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
        """Async function to get AVHRR data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAVHRRLexicon
            (e.g. ``["avhrr"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AVHRR observation data in long format with one row per pixel
            per channel.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in MetOpAVHRRLexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in MetOpAVHRRLexicon. "
                    f"Available: {list(MetOpAVHRRLexicon.VOCAB.keys())}"
                )

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Compute overall time window
        all_times = [
            t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo else t
            for t in time_list
        ]
        dt_min = min(all_times) + self._tolerance_lower
        dt_max = max(all_times) + self._tolerance_upper

        # Download products
        nat_files = await asyncio.to_thread(self._download_products, dt_min, dt_max)

        if not nat_files:
            logger.warning(
                "No AVHRR products found for time range {} to {}", dt_min, dt_max
            )
            return self._empty_dataframe(schema)

        # Parse each product file
        frames: list[pd.DataFrame] = []
        for fpath in nat_files:
            with open(fpath, "rb") as f:
                raw = f.read()
            df = _parse_native_avhrr(raw, self._subsample)
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

        # Filter by time windows
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
        """Download AVHRR products from EUMETSAT Data Store.

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
            search_kwargs["sat"] = self._satellite

        products = collection.search(**search_kwargs)

        downloaded: list[str] = []
        for product in products:
            if self._verbose:
                logger.info(
                    "Downloading AVHRR product: {} ({}–{})",
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

            cache_name = f"{product}.nat"
            cache_path = os.path.join(self.cache, cache_name)
            if os.path.isfile(cache_path):
                downloaded.append(cache_path)
                continue

            try:
                with product.open(entry=nat_entry) as stream:
                    raw = stream.read()

                # Handle ZIP-wrapped products
                if raw[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        nat_names = [n for n in zf.namelist() if n.endswith(".nat")]
                        if nat_names:
                            raw = zf.read(nat_names[0])

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
        cache_location = os.path.join(datasource_cache_root(), "metop_avhrr")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_avhrr_{self._tmp_cache_hash}"
            )
        return cache_location
