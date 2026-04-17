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

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    radiance_to_bt,
)
from earth2studio.lexicon import MetOpIASILexicon
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
# IASI L1C EPS native binary format constants
# ---------------------------------------------------------------------------
_GRH_SIZE = 20  # Generic Record Header
_MDR_RECORD_CLASS = 8
_MDR_INSTRUMENT_GROUP = 8  # IASI instrument group
_MDR_RECORD_SUBCLASS = 2  # Earth-view MDR
_MPHR_RECORD_CLASS = 1
_GIADR_RECORD_CLASS = 5
_GIADR_SF_SUBCLASS = 1  # Scale Factors subclass

_NUM_EFOVS = 30  # EFOVs per scan line
_NUM_IFOVS = 4  # IFOVs per EFOV (2×2)
_NUM_CHANNELS_ALLOC = 8700  # Allocated spectral samples per IFOV
_NUM_CHANNELS = 8461  # Valid spectral channels (645–2760 cm⁻¹)

# EPS epoch: 2000-01-01 00:00:00 UTC
_EPS_EPOCH = datetime(2000, 1, 1)

# Spacecraft ID mapping (same as AMSU-A/MHS)
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


# ---------------------------------------------------------------------------
# MDR field layout — computed offsets using known field sizes
# ---------------------------------------------------------------------------
# The IASI MDR has variable-size header fields at the start, making
# hardcoded absolute offsets unreliable across format versions. Instead,
# we compute offsets relative to the MDR payload start by scanning known
# field blocks sequentially.
#
# Strategy: We use the field sizes from the EPS PFS specification to build
# a cumulative offset table. Fields are listed in their MDR order.


def _decode_eps_datetime(data: bytes, offset: int) -> datetime:
    """Decode a 6-byte EPS datetime to Python datetime.

    EPS datetime format: 2 bytes uint16 BE (days since 2000-01-01)
    + 4 bytes uint32 BE (milliseconds of day).

    Parameters
    ----------
    data : bytes
        Raw data buffer
    offset : int
        Byte offset to start of the 6-byte field

    Returns
    -------
    datetime
        Decoded datetime (naive UTC)
    """
    day = struct.unpack_from(">H", data, offset)[0]
    ms = struct.unpack_from(">I", data, offset + 2)[0]
    return _EPS_EPOCH + timedelta(days=day, milliseconds=ms)


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
    start = datetime.strptime(start_str[:14], fmt)
    end = datetime.strptime(end_str[:14], fmt)
    return start, end


def _parse_giadr_scale_factors(
    data: bytes, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse GIADR Scale Factors record for spectral radiance conversion.

    Extracts per-band scale factor information needed to convert int16
    spectral radiance values to physical units. The band_first/band_last
    values are absolute spectral sample numbers (not 1-based IASI channel
    numbers). The scale factor values are positive integers representing
    the negative power of 10: physical = raw × 10^(-sf).

    Parameters
    ----------
    data : bytes
        Raw file data
    offset : int
        Byte offset to start of GIADR record (including GRH)

    Returns
    -------
    tuple
        (band_first_ch, band_last_ch, band_sf, nb_bands) where:
        - band_first_ch: Absolute first sample number per band (shape nb_bands)
        - band_last_ch: Absolute last sample number per band (shape nb_bands)
        - band_sf: Positive scale factor per band (use as negative exponent)
        - nb_bands: Number of scale factor bands
    """
    # After GRH (20 bytes):
    # IDefScaleSondNbScale: int16 (1)
    # IDefScaleSondNsfirst: int16 (10)
    # IDefScaleSondNslast: int16 (10)
    # IDefScaleSondScaleFactor: int16 (10)
    payload = offset + _GRH_SIZE
    nb_bands = struct.unpack_from(">h", data, payload)[0]

    first_off = payload + 2
    first_ch = np.array(struct.unpack_from(f">{10}h", data, first_off), dtype=np.int32)

    last_off = first_off + 20
    last_ch = np.array(struct.unpack_from(f">{10}h", data, last_off), dtype=np.int32)

    sf_off = last_off + 20
    sf = np.array(struct.unpack_from(f">{10}h", data, sf_off), dtype=np.int32)

    return first_ch[:nb_bands], last_ch[:nb_bands], sf[:nb_bands], nb_bands


def _build_channel_scale_array(
    band_first: np.ndarray,
    band_last: np.ndarray,
    band_sf: np.ndarray,
    nsfirst: int = 2581,
) -> np.ndarray:
    """Build per-channel scale factor array from per-band GIADR data.

    The GIADR band_first/band_last values are absolute sample numbers
    (starting from nsfirst=2581 for standard IASI). These are converted
    to 0-based IASI channel indices by subtracting nsfirst.

    The scale factor values are positive integers representing the
    negative power of 10: physical = raw × 10^(-sf).

    Parameters
    ----------
    band_first : np.ndarray
        Absolute first sample number per band (from GIADR)
    band_last : np.ndarray
        Absolute last sample number per band (from GIADR)
    band_sf : np.ndarray
        Scale factor per band (positive integer; used as negative exponent)
    nsfirst : int, optional
        First sample number from IDefNsfirst1b, by default 2581

    Returns
    -------
    np.ndarray
        Scale multiplier for each of the 8461 channels. Shape (8461,).
    """
    scales = np.ones(_NUM_CHANNELS, dtype=np.float64)
    for i in range(len(band_first)):
        # Convert absolute sample numbers to 0-based channel indices
        ch_start = int(band_first[i]) - nsfirst
        ch_end = int(band_last[i]) - nsfirst + 1  # Inclusive → exclusive
        ch_start = max(ch_start, 0)
        ch_end = min(ch_end, _NUM_CHANNELS)
        if ch_start < ch_end:
            # Scale factor is positive int used as NEGATIVE exponent
            scales[ch_start:ch_end] = 10.0 ** (-float(band_sf[i]))
    return scales


def _compute_mdr_field_offsets(data: bytes, mdr_offset: int) -> dict[str, int]:
    """Compute byte offsets for key fields within an IASI MDR.

    Because the IASI MDR has variable-size header fields, we walk the
    known field structure sequentially to compute absolute byte offsets
    for the fields we need.

    References
    ----------

    - https://user.eumetsat.int/s3/eup-strapi-media/pdf_iasi_level_1_pfs_2105bc9ccf.pdf
    - https://github.com/stcorp/codadef-eps/blob/master/types/IASI_MDR_L1C_v5.xml

    Parameters
    ----------
    data : bytes
        Raw file data
    mdr_offset : int
        Byte offset to start of this MDR record (including GRH)

    Returns
    -------
    dict[str, int]
        Mapping of field name to absolute byte offset in data
    """
    # Parse record size from GRH to validate
    _, _, _, rec_size = _parse_grh(data, mdr_offset)

    # Start after GRH
    pos = mdr_offset + _GRH_SIZE

    # DEGRADED_INST_MDR: uint8 (1 byte)
    pos += 1
    # DEGRADED_PROC_MDR: uint8 (1 byte)
    pos += 1

    # GEPSIasiMode: 4 bytes (uint8 + 3-byte bitfield)
    pos += 4
    # GEPSOPSProcessingMode: 4 bytes
    pos += 4
    # GEPSIdConf: 32 bytes (instrument configuration)
    pos += 32

    # GEPSLocIasiAvhrr_IASI: VSFInt[30][4][2] = 30*4*2*5 = 1200 bytes
    pos += 1200
    # GEPSLocIasiAvhrr_IIS: VSFInt[30][25][2] = 30*25*2*5 = 7500 bytes
    pos += 7500

    # OBT: raw48[30] = 30*6 = 180 bytes
    pos += 180
    # OnboardUTC: EPSdatetime[30] = 30*6 = 180 bytes
    pos += 180

    # GEPSDatIasi: EPSdatetime[30] = 30*6 = 180 bytes
    offsets: dict[str, int] = {}
    offsets["GEPSDatIasi"] = pos
    pos += 180

    # GIsfLinOrigin: int32[2] = 8 bytes
    pos += 8
    # GIsfColOrigin: int32[2] = 8 bytes
    pos += 8
    # GIsfPds1: int32[2] = 8 bytes
    pos += 8
    # GIsfPds2: int32[2] = 8 bytes
    pos += 8
    # GIsfPds3: int32[2] = 8 bytes
    pos += 8
    # GIsfPds4: int32[2] = 8 bytes
    pos += 8

    # GEPS_CCD: uint8[30] = 30 bytes
    pos += 30
    # GEPS_SP: int32[30] = 120 bytes
    pos += 120

    # GIrcImage: uint16[30][64][64] = 30*64*64*2 = 245760 bytes
    pos += 245760

    # Quality flags section
    # GQisFlagQual: uint8[30][4][3] = 360 bytes
    pos += 360
    # GQisFlagQualDetailed: uint16[30][4] = 240 bytes
    # 16-bit bitmask per IFOV (PFS §3.5.1.5 / §5.4.1.3):
    #   bit 0: hardware failure
    #   bit 1: Band 1 spike contamination
    #   bit 2: Band 2 spike contamination
    #   bit 3: Band 3 spike contamination
    #   bit 4: NZPD / complex calibration error
    #   bit 5: on-board general quality (BBofFlagSpectNonQual)
    #   bit 6: overflow / underflow
    #   bit 7: spectral calibration error
    #   bit 8: radiometric post-calibration error
    #   bit 9: GQisFlagQual summary flag (all bands)
    #   bit 10: missing sounder data
    #   bit 11: missing IIS data
    #   bit 12: missing AVHRR data
    #   bits 13-15: spare
    # 0 = clean observation; non-zero = bitwise OR of active flags.
    offsets["GQisFlagQualDetailed"] = pos
    pos += 240

    # GQisQualIndex: VSFInt = 5 bytes
    pos += 5
    # GQisQualIndexIIS: VSFInt = 5 bytes
    pos += 5
    # GQisQualIndexLoc: VSFInt = 5 bytes
    pos += 5
    # GQisQualIndexRad: VSFInt = 5 bytes
    pos += 5
    # GQisQualIndexSpect: VSFInt = 5 bytes
    pos += 5
    # GQisSysTecIISQual: uint32 = 4 bytes
    pos += 4
    # GQisSysTecSondQual: uint32 = 4 bytes
    pos += 4

    # Geolocation section
    # GGeoSondLoc: int32[30][4][2] = 960 bytes
    offsets["GGeoSondLoc"] = pos
    pos += 960
    # GGeoSondAnglesMETOP: int32[30][4][2] = 960 bytes
    offsets["GGeoSondAnglesMETOP"] = pos
    pos += 960
    # GGeoIISAnglesMETOP: int32[30][25][2] = 6000 bytes
    pos += 6000
    # GGeoSondAnglesSUN: int32[30][4][2] = 960 bytes
    offsets["GGeoSondAnglesSUN"] = pos
    pos += 960
    # GGeoIISAnglesSUN: int32[30][25][2] = 6000 bytes
    pos += 6000
    # GGeoIISLoc: int32[30][25][2] = 6000 bytes
    pos += 6000
    # EARTH_SATELLITE_DISTANCE: uint32 = 4 bytes
    pos += 4

    # Spectral calibration fields
    # IDefSpectDWn1b: VSFInt = 5 bytes
    offsets["IDefSpectDWn1b"] = pos
    pos += 5
    # IDefNsfirst1b: int32 = 4 bytes
    offsets["IDefNsfirst1b"] = pos
    pos += 4
    # IDefNslast1b: int32 = 4 bytes
    pos += 4

    # GS1cSpect: int16[30][4][8700] = 2,088,000 bytes
    offsets["GS1cSpect"] = pos

    return offsets


def _parse_native_iasi(
    data: bytes,
    channel_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Parse an IASI Level 1C EPS native format file.

    Extracts MDR (Measurement Data Record) scan lines and converts
    calibrated spectral radiances to brightness temperatures.

    Parameters
    ----------
    data : bytes
        Complete file contents of an IASI L1C .nat file
    channel_indices : np.ndarray | None, optional
        0-based channel indices to extract (subset of 0..8460).
        If None, all 8461 channels are extracted.

    Returns
    -------
    pd.DataFrame
        One row per (scan_line, EFOV, IFOV, channel) observation with
        columns matching MetOpIASI.SCHEMA.
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

    # Step 2: Scan all records — collect GIADR scale factors and MDR offsets
    offset = 0
    mdr_offsets: list[int] = []
    giadr_sf_offset: int | None = None

    while offset + _GRH_SIZE <= file_size:
        rc, ig, sc, rec_size = _parse_grh(data, offset)
        if rec_size < _GRH_SIZE or offset + rec_size > file_size:
            break
        if (
            rc == _MDR_RECORD_CLASS
            and ig == _MDR_INSTRUMENT_GROUP
            and sc == _MDR_RECORD_SUBCLASS
        ):
            mdr_offsets.append(offset)
        elif rc == _GIADR_RECORD_CLASS and sc == _GIADR_SF_SUBCLASS:
            giadr_sf_offset = offset
        offset += rec_size

    n_scans = len(mdr_offsets)
    if n_scans == 0:
        logger.warning("No IASI MDR records found in file")
        return pd.DataFrame()

    if giadr_sf_offset is None:
        logger.warning("No GIADR Scale Factors record found — cannot decode radiances")
        return pd.DataFrame()

    # Step 3: Parse GIADR scale factors
    band_first, band_last, band_sf, _ = _parse_giadr_scale_factors(
        data, giadr_sf_offset
    )

    # Read nsfirst from the first MDR to map GIADR absolute sample numbers
    # to 0-based IASI channel indices
    first_mdr_offsets = _compute_mdr_field_offsets(data, mdr_offsets[0])
    nsfirst = struct.unpack_from(">i", data, first_mdr_offsets["IDefNsfirst1b"])[0]

    channel_scales = _build_channel_scale_array(
        band_first, band_last, band_sf, nsfirst=nsfirst
    )

    # Determine channels to extract
    if channel_indices is None:
        ch_idx = np.arange(_NUM_CHANNELS, dtype=np.int32)
    else:
        ch_idx = np.asarray(channel_indices, dtype=np.int32)
    n_ch = len(ch_idx)

    # Step 4: Pre-allocate arrays for all scans × EFOVs × IFOVs
    n_obs_per_scan = _NUM_EFOVS * _NUM_IFOVS  # 30 × 4 = 120
    n_obs = n_scans * n_obs_per_scan

    lats = np.empty(n_obs, dtype=np.float32)
    lons = np.empty(n_obs, dtype=np.float32)
    solar_za = np.empty(n_obs, dtype=np.float32)
    solar_azi = np.empty(n_obs, dtype=np.float32)
    sat_za = np.empty(n_obs, dtype=np.float32)
    sat_azi = np.empty(n_obs, dtype=np.float32)
    quality = np.empty(n_obs, dtype=np.uint16)
    scan_times = np.empty(n_obs, dtype="datetime64[ns]")

    # Radiances for selected channels: (n_obs, n_ch)
    radiances = np.empty((n_obs, n_ch), dtype=np.float64)

    # We also need wavenumber info — extract from first MDR
    wavenumber_cm: np.ndarray | None = None

    # Compute relative field offsets once from the first MDR and reuse
    # (MDR layout is identical across all records in a conforming IASI L1C file)
    rel_offsets = {k: v - mdr_offsets[0] for k, v in first_mdr_offsets.items()}

    for scan_idx, mdr_off in enumerate(mdr_offsets):
        base = scan_idx * n_obs_per_scan

        # Reuse relative offsets — avoids recomputing per MDR
        field_offsets = {k: mdr_off + rel for k, rel in rel_offsets.items()}

        # Extract wavenumber calibration from first MDR
        if wavenumber_cm is None:
            dwn_off = field_offsets["IDefSpectDWn1b"]
            # VSFInt: 1 byte int8 scale, 4 byte int32 value
            # Physical value = value × 10^(-scale)
            dwn_scale = struct.unpack_from(">b", data, dwn_off)[0]
            dwn_value = struct.unpack_from(">i", data, dwn_off + 1)[0]
            dwn_m = dwn_value * (10.0 ** (-dwn_scale))  # in m⁻¹ (typically 25)

            nsfirst_off = field_offsets["IDefNsfirst1b"]
            nsfirst = struct.unpack_from(">i", data, nsfirst_off)[0]

            # Wavenumber of channel n (0-based) = (nsfirst + n) * dwn in m⁻¹
            all_wn_m = (nsfirst + np.arange(_NUM_CHANNELS, dtype=np.float64)) * dwn_m
            all_wn_cm = all_wn_m / 100.0  # Convert m⁻¹ to cm⁻¹
            wavenumber_cm = all_wn_cm[ch_idx]

        # GEPSDatIasi: EPSdatetime[30] — corrected UTC per EFOV
        time_off = field_offsets["GEPSDatIasi"]
        for efov in range(_NUM_EFOVS):
            dt = _decode_eps_datetime(data, time_off + efov * 6)
            dt_ns = np.datetime64(dt, "ns")
            obs_base = base + efov * _NUM_IFOVS
            scan_times[obs_base : obs_base + _NUM_IFOVS] = dt_ns

        # GGeoSondLoc: int32[30][4][2], SF=1e6 — [efov][ifov][lon=0,lat=1]
        geo_off = field_offsets["GGeoSondLoc"]
        raw_geo = np.frombuffer(
            data, dtype=">i4", count=_NUM_EFOVS * _NUM_IFOVS * 2, offset=geo_off
        ).reshape(_NUM_EFOVS, _NUM_IFOVS, 2)
        raw_geo_f = raw_geo.astype(np.float64) / 1e6

        # Flatten [30][4] → [120] and assign in one shot
        flat_lats = raw_geo_f[:, :, 1].ravel().astype(np.float32)
        flat_lons = raw_geo_f[:, :, 0].ravel()
        # Convert longitude from [-180, 180] to [0, 360]
        flat_lons = np.where(flat_lons < 0, flat_lons + 360.0, flat_lons).astype(
            np.float32
        )
        lats[base : base + n_obs_per_scan] = flat_lats
        lons[base : base + n_obs_per_scan] = flat_lons

        # GGeoSondAnglesMETOP: int32[30][4][2], SF=1e6 — [efov][ifov][zen=0,azi=1]
        sat_ang_off = field_offsets["GGeoSondAnglesMETOP"]
        raw_sat_ang = np.frombuffer(
            data,
            dtype=">i4",
            count=_NUM_EFOVS * _NUM_IFOVS * 2,
            offset=sat_ang_off,
        ).reshape(_NUM_EFOVS, _NUM_IFOVS, 2)
        raw_sat_ang_f = raw_sat_ang.astype(np.float64) / 1e6

        sat_za[base : base + n_obs_per_scan] = (
            raw_sat_ang_f[:, :, 0].ravel().astype(np.float32)
        )
        sat_azi[base : base + n_obs_per_scan] = (
            raw_sat_ang_f[:, :, 1].ravel().astype(np.float32)
        )

        # GGeoSondAnglesSUN: int32[30][4][2], SF=1e6 — [efov][ifov][zen=0,azi=1]
        sun_ang_off = field_offsets["GGeoSondAnglesSUN"]
        raw_sun_ang = np.frombuffer(
            data,
            dtype=">i4",
            count=_NUM_EFOVS * _NUM_IFOVS * 2,
            offset=sun_ang_off,
        ).reshape(_NUM_EFOVS, _NUM_IFOVS, 2)
        raw_sun_ang_f = raw_sun_ang.astype(np.float64) / 1e6

        solar_za[base : base + n_obs_per_scan] = (
            raw_sun_ang_f[:, :, 0].ravel().astype(np.float32)
        )
        solar_azi[base : base + n_obs_per_scan] = (
            raw_sun_ang_f[:, :, 1].ravel().astype(np.float32)
        )

        # GQisFlagQualDetailed: uint16[30][4] — per-IFOV quality bitmask.
        # See bit definitions in _compute_mdr_field_offsets().
        qual_off = field_offsets["GQisFlagQualDetailed"]
        raw_qual = np.frombuffer(
            data,
            dtype=">u2",
            count=_NUM_EFOVS * _NUM_IFOVS,
            offset=qual_off,
        ).reshape(_NUM_EFOVS, _NUM_IFOVS)

        quality[base : base + n_obs_per_scan] = raw_qual.ravel()

        # GS1cSpect: int16[30][4][8700] — L1C calibrated spectra
        spect_off = field_offsets["GS1cSpect"]

        # Vectorized extraction: read full spectral block and index channels
        raw_spect = np.frombuffer(
            data,
            dtype=">i2",
            count=_NUM_EFOVS * _NUM_IFOVS * _NUM_CHANNELS_ALLOC,
            offset=spect_off,
        ).reshape(_NUM_EFOVS, _NUM_IFOVS, _NUM_CHANNELS_ALLOC)

        # Select only requested channels and apply per-channel scale factors
        selected = raw_spect[:, :, ch_idx].astype(np.float64)  # (30, 4, n_ch)
        selected *= channel_scales[ch_idx]
        radiances[base : base + n_obs_per_scan, :] = selected.reshape(
            n_obs_per_scan, n_ch
        )

    # Step 5: Convert radiance to mW/(m²·sr·cm⁻¹) for Planck function
    # The GIADR-scaled values are in W/(m²·sr·m⁻¹). Convert:
    # W/(m²·sr·m⁻¹) → mW/(m²·sr·cm⁻¹):
    #   × 1000 (W → mW)  ×  100 (m⁻¹ → cm⁻¹ spectral density)  = × 1e5
    # Note: L[per cm⁻¹] = L[per m⁻¹] × 100 because 1 cm⁻¹ = 100 m⁻¹
    radiances *= 1e5

    # Step 6: Convert radiances to brightness temperatures
    if wavenumber_cm is None:
        logger.warning("Could not determine wavenumber calibration")
        return pd.DataFrame()

    bt = radiance_to_bt(radiances, wavenumber_cm)

    # Step 7: Build long-format DataFrame (one row per IFOV × channel)
    n_ch_total = n_ch
    total_rows = n_obs * n_ch_total

    # Channel indices are 1-based in the output (1..8461)
    ch_1based = ch_idx + 1

    all_times = np.tile(scan_times, n_ch_total)
    all_lats = np.tile(lats, n_ch_total)
    all_lons = np.tile(lons, n_ch_total)
    all_solza = np.tile(solar_za, n_ch_total)
    all_solaza = np.tile(solar_azi, n_ch_total)
    all_satza = np.tile(sat_za, n_ch_total)
    all_sataza = np.tile(sat_azi, n_ch_total)
    all_quality = np.tile(quality, n_ch_total)
    all_scan_angle = np.tile(sat_za, n_ch_total)  # scan angle ≈ sat zenith

    all_obs = np.empty(total_rows, dtype=np.float32)
    all_channel_idx = np.empty(total_rows, dtype=np.uint16)

    for i in range(n_ch_total):
        start = i * n_obs
        end = start + n_obs
        all_obs[start:end] = bt[:, i].astype(np.float32)
        all_channel_idx[start:end] = ch_1based[i]

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(all_times),
            "class": "rad",
            "lat": all_lats,
            "lon": all_lons,
            "elev": np.float32(0.0),  # Satellite — no terrain elevation
            "scan_angle": all_scan_angle,
            "channel_index": all_channel_idx,
            "solza": all_solza,
            "solaza": all_solaza,
            "satellite_za": all_satza,
            "satellite_aza": all_sataza,
            "quality": all_quality,
            "satellite": satellite,
            "observation": all_obs,
            "variable": "iasi",
        }
    )

    # Drop rows with invalid geolocation or NaN observations
    df = df.dropna(subset=["observation", "lat", "lon"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90)]

    return df


@check_optional_dependencies()
class MetOpIASI:
    """EUMETSAT MetOp IASI Level 1C brightness temperature observations.

    The Infrared Atmospheric Sounding Interferometer (IASI) is a Fourier
    transform infrared spectrometer aboard the MetOp series of polar-orbiting
    satellites. It measures calibrated spectral radiances across 8461 channels
    in the thermal infrared (645–2760 cm⁻¹, 3.6–15.5 µm), providing
    atmospheric temperature and humidity profiles, trace gas columns, and
    surface temperature at ~12 km spatial resolution.

    Each scan line contains 30 Extended Fields Of View (EFOVs), each
    consisting of a 2×2 array of 4 Instantaneous Fields Of View (IFOVs),
    yielding 120 spectra per scan line. A typical orbit pass contains
    ~1400 scan lines (~168,000 IFOV spectra).

    To manage memory and processing time, a ``channel_indices`` parameter
    allows selecting a subset of channels. By default only a representative
    set of 100 channels spanning the three spectral bands is extracted.

    The returned :class:`~pandas.DataFrame` has one row per IFOV per channel,
    following the same convention as :class:`~earth2studio.data.MetOpAMSUA`.
    The ``channel_index`` column (1–8461) identifies each spectral channel.

    This data source downloads Level 1C products from the EUMETSAT Data Store
    and parses the EPS native binary format to extract brightness temperatures,
    geolocation, and viewing geometry.

    Parameters
    ----------
    satellite : str, optional
        Satellite platform filter for product search. One of "metop-a",
        "metop-b", "metop-c", or None (all available). By default None.
    channel_indices : list[int] | np.ndarray | None, optional
        0-based channel indices to extract (subset of 0..8460). If None,
        a default set of 100 representative channels is used. Pass
        ``np.arange(8461)`` to extract all channels (warning: very large
        output).
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
    of data to your local machine for large requests. IASI L1C files are
    typically 100–200 MB each, with ~42–45 products per day across 3
    satellites.

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

    - https://data.eumetsat.int/product/EO:EUM:DAT:METOP:IASIL1C-ALL
    - https://user.eumetsat.int/resources/user-guides/metop-iasi-level-1-data-guide
    - Clerbaux et al. (2009), Atmos. Chem. Phys., 9, 6041–6054

    Badges
    ------
    region:global dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.metop_iasi"
    COLLECTION_ID = "EO:EUM:DAT:METOP:IASIL1C-ALL"
    VALID_SATELLITES = frozenset(["metop-a", "metop-b", "metop-c"])

    # Default representative channels: 100 channels spanning all 3 bands
    # Band 1 (645–1144 cm⁻¹): channels 1–1997, 40 channels
    # Band 2 (1210–1990 cm⁻¹): channels 1998–5116, 35 channels
    # Band 3 (2000–2760 cm⁻¹): channels 5117–8461, 25 channels
    DEFAULT_CHANNELS = np.concatenate(
        [
            np.linspace(0, 1996, 40, dtype=np.int32),
            np.linspace(1997, 5115, 35, dtype=np.int32),
            np.linspace(5116, 8460, 25, dtype=np.int32),
        ]
    )

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
        channel_indices: list[int] | np.ndarray | None = None,
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
        self._eumdac_satellite = _EUMDAC_SAT_NAME[satellite] if satellite else None

        # Channel selection
        if channel_indices is not None:
            self._channel_indices = np.asarray(channel_indices, dtype=np.int32)
        else:
            self._channel_indices = self.DEFAULT_CHANNELS.copy()

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
        """Function to get IASI brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpIASILexicon
            (e.g. ``["iasi"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            IASI observation data in long format with one row per IFOV
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
        """Async function to get IASI data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpIASILexicon
            (e.g. ``["iasi"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            IASI observation data in long format with one row per IFOV
            per channel.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in MetOpIASILexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in MetOpIASILexicon. "
                    f"Available: {list(MetOpIASILexicon.VOCAB.keys())}"
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
                "No IASI products found for time range {} to {}", dt_min, dt_max
            )
            return self._empty_dataframe(schema)

        # Parse each product file
        frames: list[pd.DataFrame] = []
        for fpath in product_files:
            with open(fpath, "rb") as f:
                raw = f.read()
            df = _parse_native_iasi(raw, channel_indices=self._channel_indices)
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
        """Download IASI L1C products from EUMETSAT Data Store.

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
            search_kwargs["sat"] = self._eumdac_satellite

        products = collection.search(**search_kwargs)

        downloaded: list[str] = []
        for product in products:
            if self._verbose:
                logger.info(
                    "Downloading IASI product: {} ({}–{})",
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
            except Exception as exc:  # noqa: S110
                logger.debug(
                    "Could not list entries for product {}: {} — using default entry",
                    product,
                    exc,
                )

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
        cache_location = os.path.join(datasource_cache_root(), "metop_iasi")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_iasi_{self._tmp_cache_hash}"
            )
        return cache_location
