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
import hashlib
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
    gather_with_concurrency,
    prep_data_inputs,
    radiance_to_bt,
)
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.lexicon.jpss import JPSSCrISLexicon
from earth2studio.utils.time import TimeTolerance, normalize_time_tolerance
from earth2studio.utils.type import TimeArray, VariableArray

# S3 bucket per satellite short-name
_SAT_BUCKET_MAP: dict[str, str] = {
    "n20": "noaa-nesdis-n20-pds",
    "n21": "noaa-nesdis-n21-pds",
    "npp": "noaa-nesdis-snpp-pds",
}

# Platform identifier in filenames
_SAT_PLATFORM_MAP: dict[str, str] = {
    "n20": "j01",
    "n21": "j02",
    "npp": "npp",
}

# Reverse mapping: platform code -> satellite short-name
_PLATFORM_SAT_MAP: dict[str, str] = {v: k for k, v in _SAT_PLATFORM_MAP.items()}

# Earliest date with CrIS FSR data on S3 per satellite
_SAT_START_DATE: dict[str, datetime] = {
    "npp": datetime(2023, 9, 6),
    "n20": datetime(2023, 9, 6),
    "n21": datetime(2023, 9, 6),
}

# ---------------------------------------------------------------------------
# CrIS instrument constants
# ---------------------------------------------------------------------------
_CRIS_NUM_CHANNELS_LW: int = 717
_CRIS_NUM_CHANNELS_MW: int = 869
_CRIS_NUM_CHANNELS_SW: int = 637
_CRIS_NUM_CHANNELS: int = (
    _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW + _CRIS_NUM_CHANNELS_SW
)  # 2223

_CRIS_NUM_FOR: int = 30  # Fields of Regard per scan line
_CRIS_NUM_FOV: int = 9  # Fields of View per FOR (3x3 detector array)

# Nominal CrIS look-angle geometry. The FOR center advances across the scan,
# while each detector FOV is offset by the rotating 3x3 focal-plane geometry.
# NOAA CrIS SDR ATBD (successor to JPSS document 474-00032):
# https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-002_JPSS_ATBD_CRIS-SDR_fsr_20180614.pdf
# NCEP GSI implementation and geometry constants:
# https://github.com/NOAA-EMC/GSI/blob/3c1f5fe2fbafd201d5125cfac38056bd7fbc4333/src/gsi/read_cris.f90#L208-L223
_CRIS_SCAN_START_DEG: float = -48.330
_CRIS_SCAN_STEP_DEG: float = 3.3331
_CRIS_FOV_DISTANCE_RAD: np.ndarray = np.asarray(
    [
        2.71510e-2,
        1.91986e-2,
        2.71510e-2,
        1.91986e-2,
        0.0,
        1.91986e-2,
        2.71510e-2,
        1.91986e-2,
        2.71510e-2,
    ],
    dtype=np.float64,
)
_CRIS_FOV_DIRECTION_RAD: np.ndarray = np.asarray(
    [4.77057, 3.98517, 3.19977, 5.55597, 0.0, 2.41437, 0.05818, 0.84358, 1.62897],
    dtype=np.float64,
)


def _nominal_cris_scan_angle(
    field_of_regard: np.ndarray, field_of_view: np.ndarray
) -> np.ndarray:
    """Return the nominal CrIS look angle in degrees for each footprint."""
    field_of_regard = np.asarray(field_of_regard, dtype=np.int64)
    field_of_view = np.asarray(field_of_view, dtype=np.int64)
    if np.any((field_of_regard < 1) | (field_of_regard > _CRIS_NUM_FOR)):
        raise ValueError("CrIS field_of_regard must be in the range 1--30")
    if np.any((field_of_view < 1) | (field_of_view > _CRIS_NUM_FOV)):
        raise ValueError("CrIS field_of_view must be in the range 1--9")

    for_offset: np.ndarray = field_of_regard - 1
    fov_offset: np.ndarray = field_of_view - 1
    for_rotation = np.deg2rad(for_offset * _CRIS_SCAN_STEP_DEG)
    angle = np.deg2rad(_CRIS_SCAN_START_DEG) + for_rotation
    angle += _CRIS_FOV_DISTANCE_RAD[fov_offset] * np.sin(
        _CRIS_FOV_DIRECTION_RAD[fov_offset] - for_rotation
    )
    return np.rad2deg(angle).astype(np.float32)


# GSI / CRTM sensor_chan numbering for CrIS FSR.
# https://www.star.nesdis.noaa.gov/jpss/documents/UserGuides/CrIS_SDR_Users_Guide1p1_20180405.pdf
# (Table 4, LWIR, MWIR FSR, SWIR FSR)
#
# The physical instrument has 2223 channels (717 + 869 + 637) but the CRTM
# SpcCoeff defines only 2211 *science* channels (713 + 865 + 633).  Each band
# has 2 guard channels at the low-wavenumber end and 2 at the high-wavenumber
# end (4 total per band, 12 overall) that are excluded from CRTM and never
# appear in GSI diagnostic files.  GSI uses contiguous 1-based numbering
# across the three science bands:
#
#   Band   | JPSS 0-based |  sensor_chan  | Wavenumber (cm^-1)
#   -------+--------------+--------------+--------------------
#   (guard)| 0 .. 1       | 0 (sentinel) | 648.75 -- 649.375
#   LWIR   | 2 .. 714     | 1 .. 713     | 650.0 -- 1095.0
#   (guard)| 715 .. 716   | 0 (sentinel) | 1095.625 -- 1096.25
#   (guard)| 717 .. 718   | 0 (sentinel) | 1208.75 -- 1209.375
#   MWIR   | 719 .. 1583  | 714 .. 1578  | 1210.0 -- 1750.0
#   (guard)| 1584 .. 1585 | 0 (sentinel) | 1750.625 -- 1751.25
#   (guard)| 1586 .. 1587 | 0 (sentinel) | 2153.75 -- 2154.375
#   SWIR   | 1588 .. 2220 | 1579 .. 2211 | 2155.0 -- 2550.0
#   (guard)| 2221 .. 2222 | 0 (sentinel) | 2550.625 -- 2551.25
#

# Guard channels are assigned sensor_chan 0 (sentinel for "not in GSI/CRTM").
_CRIS_NUM_SCIENCE_LW: int = 713  # CRTM science channels per band
_CRIS_NUM_SCIENCE_MW: int = 865
_CRIS_NUM_SCIENCE_SW: int = 633
_CRIS_NUM_GUARD_LO: int = 2  # guard channels at low-wavenumber end of each band
_CRIS_NUM_GUARD_HI: int = 2  # guard channels at high-wavenumber end of each band

# Full (unapodized) sensor_chan mapping — 2223 elements including guard channels.
_CRIS_GSI_SENSOR_CHAN: np.ndarray = np.zeros(
    _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW + _CRIS_NUM_CHANNELS_SW,
    dtype=np.uint16,
)
# LWIR: skip 2 low guards, 713 science channels → 1..713, 2 high guards → 0
_CRIS_GSI_SENSOR_CHAN[
    _CRIS_NUM_GUARD_LO : _CRIS_NUM_GUARD_LO + _CRIS_NUM_SCIENCE_LW
] = np.arange(1, _CRIS_NUM_SCIENCE_LW + 1, dtype=np.uint16)
# MWIR: skip 2 low guards, 865 science channels → 714..1578, 2 high guards → 0
_mw_start = _CRIS_NUM_CHANNELS_LW  # physical index where MWIR begins
_CRIS_GSI_SENSOR_CHAN[
    _mw_start + _CRIS_NUM_GUARD_LO : _mw_start
    + _CRIS_NUM_GUARD_LO
    + _CRIS_NUM_SCIENCE_MW
] = np.arange(
    _CRIS_NUM_SCIENCE_LW + 1,
    _CRIS_NUM_SCIENCE_LW + _CRIS_NUM_SCIENCE_MW + 1,
    dtype=np.uint16,
)
# SWIR: skip 2 low guards, 633 science channels → 1579..2211, 2 high guards → 0
_sw_start = _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW
_CRIS_GSI_SENSOR_CHAN[
    _sw_start + _CRIS_NUM_GUARD_LO : _sw_start
    + _CRIS_NUM_GUARD_LO
    + _CRIS_NUM_SCIENCE_SW
] = np.arange(
    _CRIS_NUM_SCIENCE_LW + _CRIS_NUM_SCIENCE_MW + 1,
    _CRIS_NUM_SCIENCE_LW + _CRIS_NUM_SCIENCE_MW + _CRIS_NUM_SCIENCE_SW + 1,
    dtype=np.uint16,
)

# Apodized sensor_chan mapping — 2211 science channels only (guard trimmed).
_CRIS_GSI_SENSOR_CHAN_APOD: np.ndarray = np.arange(
    1,
    _CRIS_NUM_SCIENCE_LW + _CRIS_NUM_SCIENCE_MW + _CRIS_NUM_SCIENCE_SW + 1,
    dtype=np.uint16,
)

# CrIS wavenumber grid for each channel (cm^-1).
# All three bands use a uniform 0.625 cm^-1 spacing.
# The SDR stores 2 guard channels at the low-wavenumber end of each band,
# so the physical grid starts 2 * 0.625 = 1.25 cm^-1 below the science start.
_CRIS_WAVENUMBER: np.ndarray = np.concatenate(
    [
        648.75 + 0.625 * np.arange(_CRIS_NUM_CHANNELS_LW),  # LWIR
        1208.75 + 0.625 * np.arange(_CRIS_NUM_CHANNELS_MW),  # MWIR
        2153.75 + 0.625 * np.arange(_CRIS_NUM_CHANNELS_SW),  # SWIR
    ]
).astype(np.float64)

# Apodized (science-only) wavenumber grid — guard channels removed.
_CRIS_WAVENUMBER_APOD: np.ndarray = np.concatenate(
    [
        650.0 + 0.625 * np.arange(_CRIS_NUM_SCIENCE_LW),  # LWIR science
        1210.0 + 0.625 * np.arange(_CRIS_NUM_SCIENCE_MW),  # MWIR science
        2155.0 + 0.625 * np.arange(_CRIS_NUM_SCIENCE_SW),  # SWIR science
    ]
).astype(np.float64)

# ---------------------------------------------------------------------------
# Hamming apodization constants
# ---------------------------------------------------------------------------
# CrIS is a Fourier Transform Spectrometer. The native SDR files store
# unapodized (sinc ILS) spectral radiance; this operator produces the Hamming-
# apodized science spectrum defined by the NOAA CrIS SDR ATBD.
_HAMMING_A0: float = 0.54
_HAMMING_A1: float = 0.23  # symmetric: a_{-1} = a_{+1}

# Band offsets into the concatenated 2223-channel array (for per-band apodization).
_BAND_SLICES: list[tuple[int, int, int]] = [
    (0, _CRIS_NUM_CHANNELS_LW, _CRIS_NUM_SCIENCE_LW),  # LWIR
    (
        _CRIS_NUM_CHANNELS_LW,
        _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW,
        _CRIS_NUM_SCIENCE_MW,
    ),  # MWIR
    (
        _CRIS_NUM_CHANNELS_LW + _CRIS_NUM_CHANNELS_MW,
        _CRIS_NUM_CHANNELS,
        _CRIS_NUM_SCIENCE_SW,
    ),  # SWIR
]


def _hamming_apodize(radiance: np.ndarray) -> np.ndarray:
    """Apply Hamming apodization to unapodized CrIS spectral radiance.

    The 3-tap symmetric Hamming convolution kernel ``[0.23, 0.54, 0.23]``
    is the exact spectral-domain equivalent of multiplying the CrIS
    interferogram by ``w(x) = 0.54 + 0.46 cos(pi x / L)``.  It is applied
    independently to each of the three bands (LWIR, MWIR, SWIR) in radiance
    space.  The 2 guard channels at each end of each band (4 per band, 12
    total) are trimmed after convolution, reducing the total from 2223 to
    2211 science channels.

    Reflect padding preserves the temporary band length. It affects only the
    outer guard-channel outputs that are discarded; every retained science
    channel is convolved with its real neighboring source channels.

    References
    ----------

    - https://www.star.nesdis.noaa.gov/jpss/documents/UserGuides/CrIS_SDR_Users_Guide1p1_20180405.pdf
    - https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-002_JPSS_ATBD_CRIS-SDR_fsr_20180614.pdf
    - https://www-cdn.eumetsat.int/files/2022-11/12%20-%20Tobin%20-%20CrIS_spectral_20221019.pdf

    """
    squeeze = radiance.ndim == 1
    if squeeze:
        radiance = radiance[np.newaxis, :]

    out_parts: list[np.ndarray] = []
    for band_start, band_end, n_science in _BAND_SLICES:
        band = radiance[:, band_start:band_end]  # (n_fov, n_band)
        # Pad one sample on each side using reflect (symmetric boundary)
        padded = np.pad(band, ((0, 0), (1, 1)), mode="reflect")
        # 3-tap Hamming convolution
        apod = (
            _HAMMING_A1 * padded[:, :-2]
            + _HAMMING_A0 * padded[:, 1:-1]
            + _HAMMING_A1 * padded[:, 2:]
        )
        # Trim guard channels: 2 at low end + 2 at high end → keep n_science
        out_parts.append(apod[:, _CRIS_NUM_GUARD_LO : _CRIS_NUM_GUARD_LO + n_science])

    result = np.concatenate(out_parts, axis=1)
    if squeeze:
        return result[0]
    return result


# HDF5 dataset paths in SDR files (CrIS-FS-SDR)
_SDR_GROUP = "All_Data/CrIS-FS-SDR_All"
_SDR_RADIANCE_KEYS = {
    "LW": f"{_SDR_GROUP}/ES_RealLW",  # shape (n_scan, 30, 9, 717)
    "MW": f"{_SDR_GROUP}/ES_RealMW",  # shape (n_scan, 30, 9, 869)
    "SW": f"{_SDR_GROUP}/ES_RealSW",  # shape (n_scan, 30, 9, 637)
}

# Quality flag datasets
_SDR_QF_KEYS = {
    "QF1": f"{_SDR_GROUP}/QF1_SCAN_CRISSDR",  # shape (n_scan,)
    "QF2": f"{_SDR_GROUP}/QF2_CRISSDR",  # shape (n_scan, 9, 3)
    "QF3": f"{_SDR_GROUP}/QF3_CRISSDR",  # shape (n_scan, 30, 9, 3)
}

# HDF5 dataset paths in GEO files (CrIS-SDR-GEO)
_GEO_GROUP = "All_Data/CrIS-SDR-GEO_All"
_GEO_KEYS = {
    "lat": f"{_GEO_GROUP}/Latitude",  # shape (n_scan, 30, 9)
    "lon": f"{_GEO_GROUP}/Longitude",  # shape (n_scan, 30, 9)
    "height": f"{_GEO_GROUP}/Height",  # shape (n_scan, 30, 9)
    "sat_za": f"{_GEO_GROUP}/SatelliteZenithAngle",  # shape (n_scan, 30, 9)
    "sat_aza": f"{_GEO_GROUP}/SatelliteAzimuthAngle",  # shape (n_scan, 30, 9)
    "sol_za": f"{_GEO_GROUP}/SolarZenithAngle",  # shape (n_scan, 30, 9)
    "sol_aza": f"{_GEO_GROUP}/SolarAzimuthAngle",  # shape (n_scan, 30, 9)
    "for_time": f"{_GEO_GROUP}/FORTime",  # shape (n_scan, 30) IET µs
}

_GEO_GRANULE_GROUP = "Data_Products/CrIS-SDR-GEO/CrIS-SDR-GEO_Gran_0"

# IDPS Epoch Time (IET) is a count of TAI-length microseconds from the
# 1958-01-01 epoch, not elapsed UTC microseconds. NOAA JPSS CDFCB-X Vol. I,
# Revision F, section 3.3.1:
# https://www.nesdis.noaa.gov/s3/2024-01/474-00001-01_JPSS-CDFCB-X-Vol-I_F.pdf


def _hdf_text(value: Any, attribute: str) -> str:
    values = np.asarray(value)
    if values.size != 1:
        raise ValueError(f"CrIS GEO attribute {attribute!r} must contain one value")
    scalar = values.item()
    if isinstance(scalar, (bytes, np.bytes_)):
        return bytes(scalar).decode("ascii").rstrip("\x00")
    if isinstance(scalar, str):
        return scalar.rstrip("\x00")
    raise ValueError(f"CrIS GEO attribute {attribute!r} must contain text")


def _read_hdf_dataset(file: h5py.File, path: str, selection: Any = ()) -> np.ndarray:
    """Read an HDF5 dataset after narrowing the path to a dataset object."""
    dataset = file[path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"CrIS HDF5 path {path!r} must contain a dataset")
    return np.asarray(dataset[selection])


def _read_cris_time_anchor(geo: h5py.File) -> tuple[np.datetime64, int]:
    """Read the paired UTC/IET beginning anchor carried by a GEO granule."""
    granule = geo[_GEO_GRANULE_GROUP]
    if not isinstance(granule, (h5py.Dataset, h5py.Group)):
        raise ValueError(
            f"CrIS HDF5 path {_GEO_GRANULE_GROUP!r} must contain an HDF5 object"
        )
    attributes = granule.attrs
    date = _hdf_text(attributes["Beginning_Date"], "Beginning_Date")
    time = _hdf_text(attributes["Beginning_Time"], "Beginning_Time")
    try:
        utc = np.datetime64(
            datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S.%fZ"), "us"
        )
    except ValueError as exc:
        raise ValueError(f"Invalid CrIS GEO UTC anchor: {date} {time}") from exc

    iet_values = np.asarray(attributes["N_Beginning_Time_IET"])
    if iet_values.size != 1:
        raise ValueError(
            "CrIS GEO attribute 'N_Beginning_Time_IET' must contain one value"
        )
    return utc, int(iet_values.item())


def _iet_to_utc(
    iet_microseconds: np.ndarray, anchor_utc: np.datetime64, anchor_iet: int
) -> np.ndarray:
    """Convert IET using the UTC/IET anchor published with the granule."""
    iet = np.asarray(iet_microseconds, dtype=np.int64)
    return anchor_utc + (iet - anchor_iet).astype("timedelta64[us]")


@dataclass
class _CrISAsyncTask:
    """Metadata for a single CrIS granule download task (SDR + GEO pair)."""

    sdr_uri: str
    geo_uri: str
    datetime_min: datetime
    datetime_max: datetime
    satellite: str
    variable: str
    modifier: Callable[[Any], Any]


@dataclass
class _CrISDecodedGranule:
    """Compact decoded data from a single CrIS granule."""

    lat: np.ndarray  # (n_valid,) float32
    lon: np.ndarray  # (n_valid,) float32
    scan_line: np.ndarray  # (n_valid,) uint32, one-based within granule
    field_of_regard: np.ndarray  # (n_valid,) uint16, one-based
    field_of_view: np.ndarray  # (n_valid,) uint16, one-based
    sat_za: np.ndarray  # (n_valid,) float32
    sat_aza: np.ndarray  # (n_valid,) float32
    sol_za: np.ndarray  # (n_valid,) float32
    sol_aza: np.ndarray  # (n_valid,) float32
    quality: np.ndarray  # (n_valid,) uint16
    times: np.ndarray  # (n_valid,) datetime64[us]
    brightness_temperature: np.ndarray  # (n_valid, n_channels) float32, K
    source_uri: str
    satellite: str
    variable: str


class JPSS_CRIS:
    """JPSS CrIS (Cross-track Infrared Sounder) Full Spectral Resolution (FSR)
    Level 1 brightness temperature observations served from NOAA Open Data on
    AWS.

    Raw spectral radiance from the HDF5 SDR files is converted to Planck
    brightness temperature (K) at each channel center wavenumber. Reproducing
    a coefficient-based downstream product exactly also requires that product's
    release-pinned Planck constants.

    By default, Hamming apodization is applied to the unapodized (sinc ILS)
    radiance before the Planck inversion. This follows the NOAA three-point
    radiance-space Hamming definition. The 2 guard channels at each end of each
    band (4 per band, 12 total) are trimmed during apodization. With no
    ``sensor_indices`` projection, this yields 2211 science channels; setting
    ``apodize=False`` instead retains all 2223 unapodized channels, including
    12 guard channels with ``sensor_index=0``.

    Each HDF5 granule contains a small number of scan lines, each with 30
    Fields of Regard (FOR) and 9 Fields of View (FOV) per FOR (3x3 detector
    array).  In FSR mode the instrument produces 2223 spectral channels:

    - **LWIR** (9.14--15.38 µm, 650--1095 cm^-1): 717 channels at 0.625 cm^-1
    - **MWIR** (5.71--8.26 µm, 1210--1750 cm^-1): 869 channels at 0.625 cm^-1
    - **SWIR** (3.92--4.64 µm, 2155--2550 cm^-1): 637 channels at 0.625 cm^-1

    With ``sensor_indices=None`` and ``apodize=True`` (default), guard channels
    are trimmed and the output has 2211 channels with contiguous
    ``sensor_index`` 1--2211.

    ``scan_line``, ``field_of_regard``, and ``field_of_view`` preserve the
    one-based source-array position of each footprint.  ``scan_line`` is local
    to its source granule. ``scan_angle`` is the nominal CrIS look angle derived
    from FOR/FOV geometry, while ``satellite_za`` preserves the independently
    measured ``SatelliteZenithAngle`` from the GEO product.

    With ``sensor_indices=None`` and ``apodize=False``, the returned
    :class:`~pandas.DataFrame` has one row per FOV per channel including guard
    channels. The ``sensor_index`` column uses the GSI ``sensor_chan`` numbering
    convention:

    - **LWIR** channels 0--1 (0-based) → sensor_chan 0 (guard; not in GSI)
    - **LWIR** channels 2--714 (0-based) → sensor_chan 1--713
    - **LWIR** channels 715--716 (0-based) → sensor_chan 0 (guard; not in GSI)
    - **MWIR** channels 717--718 (0-based) → sensor_chan 0 (guard; not in GSI)
    - **MWIR** channels 719--1583 (0-based) → sensor_chan 714--1578
    - **MWIR** channels 1584--1585 (0-based) → sensor_chan 0 (guard; not in GSI)
    - **SWIR** channels 1586--1587 (0-based) → sensor_chan 0 (guard; not in GSI)
    - **SWIR** channels 1588--2220 (0-based) → sensor_chan 1579--2211
    - **SWIR** channels 2221--2222 (0-based) → sensor_chan 0 (guard; not in GSI)

    Data is stored as paired HDF5 files on S3:

    - **SDR** (``SCRIF_*.h5``): spectral radiance arrays
    - **GEO** (``GCRSO_*.h5``): geolocation (lat, lon, angles, time)

    Parameters
    ----------
    satellites : list[str] | None, optional
        Satellite short-names to query.  Valid values are ``"n20"``
        (NOAA-20), ``"n21"`` (NOAA-21), and ``"npp"`` (Suomi NPP).
        By default ``None``, which queries all valid satellites.
    subsample : int, optional
        Temporal sub-sampling factor applied at the granule level.  Each
        CrIS granule covers roughly 32 seconds of observations; setting
        ``subsample=N`` selects every *N*-th granule from the time-ordered
        list, reducing data volume by approximately that factor.  By
        default 1 (no sub-sampling).
    apodize : bool, optional
        Apply Hamming apodization to the unapodized SDR radiance before
        converting to brightness temperature.  When ``True`` (default),
        the 3-tap Hamming kernel ``[0.23, 0.54, 0.23]`` is convolved
        per-band in radiance space and the 2 guard channels at each
        end of each band are trimmed, yielding the 2211-channel Hamming-
        apodized science grid before any ``sensor_indices`` projection. Set to
        ``False`` to retain the unapodized source grid; with no projection this
        contains all 2223 channels, including 12 guards with
        ``sensor_index=0``.

        .. note::

           NOAA's CrIS SDR ATBD defines Hamming apodization on the Nyquist
           grid as this exact three-point radiance-space operator. Exact
           brightness-temperature reproduction additionally requires the same
           Planck constants as the comparison product.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single value
        (symmetric +/- window) or a tuple (lower, upper) for asymmetric windows,
        by default, np.timedelta64(10, 'm').
    cache : bool, optional
        Cache downloaded HDF5 files locally, by default True
    verbose : bool, optional
        Show download progress bars, by default True
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600
    max_workers : int, optional
        Maximum number of concurrent S3 fetch tasks, by default 24
    retries : int, optional
        Per-file retry count on transient I/O failures, by default 3
    sensor_indices : list[int] | np.ndarray | None, optional
        Science-channel ``sensor_index`` values to return (1--2211).  Values
        are normalized to source order and must be unique.  Projection occurs
        after full-band Hamming apodization (when enabled), but before Planck
        inversion and long-format expansion.  By default None, all channels
        are returned; for ``apodize=False`` this includes the 12 guard channels
        whose ``sensor_index`` is 0.

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests.  Each CrIS granule pair
    (SDR + GEO) is approximately 16 MB.

    Note
    ----
    Additional information on the data repository:

    - https://registry.opendata.aws/noaa-jpss/
    - https://www.star.nesdis.noaa.gov/jpss/CrIS.php
    - https://www.star.nesdis.noaa.gov/jpss/documents/UserGuides/CrIS_SDR_Users_Guide1p1_20180405.pdf

    Badges
    ------
    region:global dataclass:observation product:sat
    """

    SOURCE_ID = "earth2studio.data.JPSS_CRIS"
    VALID_SATELLITES = frozenset(["n20", "n21", "npp"])

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("class"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            pa.field(
                "scan_angle",
                pa.float32(),
                nullable=True,
                metadata={
                    "description": (
                        "Nominal CrIS look angle derived from FOR/FOV geometry (deg)"
                    )
                },
            ),
            pa.field(
                "scan_line",
                pa.uint32(),
                nullable=True,
                metadata={
                    "description": (
                        "One-based scan position within the source SDR/GEO granule"
                    )
                },
            ),
            pa.field(
                "field_of_regard",
                pa.uint16(),
                nullable=True,
                metadata={
                    "description": "One-based CrIS Earth-scene FOR index (1--30)"
                },
            ),
            pa.field(
                "field_of_view",
                pa.uint16(),
                nullable=True,
                metadata={"description": "One-based CrIS detector FOV index (1--9)"},
            ),
            E2STUDIO_SCHEMA.field("sensor_index"),
            E2STUDIO_SCHEMA.field("wavenumber"),
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
        satellites: list[str] | None = None,
        subsample: int = 1,
        apodize: bool = True,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        max_workers: int = 24,
        retries: int = 3,
        sensor_indices: list[int] | np.ndarray | None = None,
    ) -> None:
        if satellites is None:
            satellites = list(self.VALID_SATELLITES)
        else:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {invalid}. "
                    f"Valid options: {sorted(self.VALID_SATELLITES)}"
                )
        self._satellites = satellites
        self._subsample = max(1, int(subsample))
        self._apodize = apodize
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._retries = retries
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        if sensor_indices is None:
            self._sensor_indices: np.ndarray | None = None
        else:
            requested = tuple(sensor_indices)
            if not requested:
                raise ValueError("sensor_indices must contain at least one channel")
            if any(
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or index < 1
                or index
                > _CRIS_NUM_SCIENCE_LW + _CRIS_NUM_SCIENCE_MW + _CRIS_NUM_SCIENCE_SW
                for index in requested
            ):
                raise ValueError(
                    "sensor_indices must be integer science-channel indices "
                    "from 1 to 2211"
                )
            normalized = np.asarray(
                sorted(int(index) for index in requested), dtype=np.uint16
            )
            if len(np.unique(normalized)) != len(normalized):
                raise ValueError("sensor_indices must be unique")
            self._sensor_indices = normalized

        self.fs: s3fs.S3FileSystem | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    def _channel_projection(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source positions, sensor indices, and wavenumbers to publish."""
        sensor_indices = (
            _CRIS_GSI_SENSOR_CHAN_APOD if self._apodize else _CRIS_GSI_SENSOR_CHAN
        )
        wavenumbers = _CRIS_WAVENUMBER_APOD if self._apodize else _CRIS_WAVENUMBER

        if self._sensor_indices is None:
            positions: np.ndarray = np.arange(len(sensor_indices), dtype=np.intp)
        else:
            positions = np.flatnonzero(np.isin(sensor_indices, self._sensor_indices))

        return positions, sensor_indices[positions], wavenumbers[positions]

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------
    async def _async_init(self) -> None:
        """Initialise the async S3 filesystem."""
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
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch CrIS FSR brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return (e.g. ``["crisfsr"]``).
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None (all).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with one row per FOV per channel.
        """
        try:
            df = _sync_async(
                self.fetch, time, variable, fields, timeout=self.async_timeout
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
        """Async implementation of the data fetch.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable names to return.
        fields : str | list[str] | pa.Schema | None, optional
            Subset of schema fields to include, by default None.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """
        if self.fs is None:
            await self._async_init()

        session = await self.fs.set_session(refresh=True)  # type: ignore[union-attr]

        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)
        self._validate_time(time_list)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Validate variables
        for v in variable_list:
            try:
                JPSSCrISLexicon[v]  # type: ignore
            except KeyError:
                logger.error(f"Variable id {v} not found in JPSS CrIS lexicon")
                raise

        # Discover and download HDF5 file pairs within tolerance windows
        tasks = await self._create_tasks(time_list, variable_list)

        # Deduplicate by S3 URI (both SDR and GEO)
        uri_set: set[str] = set()
        for t in tasks:
            uri_set.add(t.sdr_uri)
            uri_set.add(t.geo_uri)

        fetch_jobs = [self._fetch_remote_file(uri) for uri in uri_set]
        await gather_with_concurrency(
            fetch_jobs,
            max_workers=self._max_workers,
            desc="Fetching CrIS HDF5 files",
            verbose=(not self._verbose),
        )

        if session:
            await session.close()

        # Decode and compile
        df = self._compile_dataframe(tasks, schema)
        return df

    async def _create_tasks(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> list[_CrISAsyncTask]:
        """Build download tasks by listing the S3 day-directory.

        For each requested time +/- tolerance we list the relevant day
        directories on each satellite bucket and select SDR files whose
        embedded start-timestamp falls within the tolerance window.  The
        corresponding GEO file is paired by matching the common filename
        fields (platform, date, start time, end time, orbit).

        SDR and GEO directory listings are issued concurrently.
        """
        tasks: list[_CrISAsyncTask] = []

        for v in variable_list:
            _, modifier = JPSSCrISLexicon[v]  # type: ignore

            for sat in self._satellites:
                bucket = _SAT_BUCKET_MAP[sat]

                for t in time_list:
                    tmin = t + self._tolerance_lower
                    tmax = t + self._tolerance_upper

                    # Iterate over calendar days covered by the window
                    day = tmin.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_day = tmax.replace(hour=0, minute=0, second=0, microsecond=0)

                    while day <= end_day:
                        sdr_prefix = (
                            f"{bucket}/CrIS-FS-SDR/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )
                        geo_prefix = (
                            f"{bucket}/CrIS-SDR-GEO/"
                            f"{day.year:04d}/{day.month:02d}/{day.day:02d}/"
                        )

                        # Issue both listings concurrently
                        sdr_coro = self.fs._ls(sdr_prefix, detail=False)  # type: ignore[union-attr]
                        geo_coro = self.fs._ls(geo_prefix, detail=False)  # type: ignore[union-attr]

                        sdr_listing: list[str] = []
                        geo_listing: list[str] = []
                        try:
                            sdr_listing, geo_listing = await asyncio.gather(
                                sdr_coro, geo_coro
                            )
                        except FileNotFoundError:
                            # One or both directories missing — try individually
                            try:
                                sdr_listing = await self.fs._ls(  # type: ignore[union-attr]
                                    sdr_prefix, detail=False
                                )
                            except FileNotFoundError:
                                logger.warning(f"No CrIS data at s3://{sdr_prefix}")
                            try:
                                geo_listing = await self.fs._ls(  # type: ignore[union-attr]
                                    geo_prefix, detail=False
                                )
                            except FileNotFoundError:
                                logger.warning(f"No CrIS GEO data at s3://{geo_prefix}")

                        if not sdr_listing:
                            day += timedelta(days=1)
                            continue

                        # Build GEO lookup keyed by granule key
                        geo_lookup: dict[str, str] = {}
                        for gpath in geo_listing:
                            gname = gpath.rsplit("/", 1)[-1]
                            if gname.startswith("GCRSO_"):
                                geo_lookup[self._granule_key(gname)] = gpath

                        for path in sdr_listing:
                            fname = path.rsplit("/", 1)[-1]
                            if not fname.startswith("SCRIF_"):
                                continue

                            file_time = self._parse_filename_time(fname)
                            if file_time is None:
                                continue
                            if tmin <= file_time <= tmax:
                                sdr_key = self._granule_key(fname)
                                if sdr_key not in geo_lookup:
                                    logger.warning(f"No matching GEO file for {fname}")
                                    continue
                                tasks.append(
                                    _CrISAsyncTask(
                                        sdr_uri=f"s3://{path}",
                                        geo_uri=f"s3://{geo_lookup[sdr_key]}",
                                        datetime_min=tmin,
                                        datetime_max=tmax,
                                        satellite=sat,
                                        variable=v,
                                        modifier=modifier,
                                    )
                                )

                        day += timedelta(days=1)

        return tasks

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, s3_uri: str) -> None:
        """Download a single HDF5 file to local cache (with retry)."""
        local_path = self._cache_path(s3_uri)
        if pathlib.Path(local_path).is_file():
            return

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                data = await self.fs._cat_file(s3_uri.replace("s3://", "", 1))  # type: ignore[union-attr]
                with open(local_path, "wb") as fh:
                    fh.write(data)
                return
            except (OSError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        logger.warning(f"Failed to fetch {s3_uri} after {self._retries} retries")
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # HDF5 decoding & DataFrame compilation
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        tasks: list[_CrISAsyncTask],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode cached HDF5 files and assemble the output DataFrame.

        HDF5 decoding is parallelised across threads.  Both h5py I/O and
        numpy compute release the GIL, so threads give effective speedup
        without the serialisation overhead of multiprocessing.

        Each granule is decoded into a compact :class:`_CrISDecodedGranule`
        (spatial arrays + 2-D brightness-temperature matrix).  The expensive
        expansion to long-format (one row per channel per FOV) is done once at
        the end over all granules combined, avoiding intermediate DataFrame
        allocation per granule.
        """

        def _decode_one(task: _CrISAsyncTask) -> _CrISDecodedGranule | None:
            sdr_path = self._cache_path(task.sdr_uri)
            geo_path = self._cache_path(task.geo_uri)

            if not pathlib.Path(sdr_path).is_file():
                logger.warning(f"Cached SDR file missing for {task.sdr_uri}")
                return None
            if not pathlib.Path(geo_path).is_file():
                logger.warning(f"Cached GEO file missing for {task.geo_uri}")
                return None

            try:
                return self._decode_hdf5(sdr_path, geo_path, task)
            except Exception:
                logger.warning(f"Failed to decode {task.sdr_uri}", exc_info=True)
                return None

        # Sub-sample granules (temporal sub-sampling) before decoding
        if self._subsample > 1:
            tasks = tasks[:: self._subsample]

        # Decode all granules in parallel using threads
        n_workers = min(len(tasks), self._max_workers) if tasks else 1
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_decode_one, tasks))

        granules = [g for g in results if g is not None]

        if not granules:
            return pd.DataFrame(columns=schema.names)

        # --- Batch: concatenate compact spatial arrays across granules ---
        all_lat = np.concatenate([g.lat for g in granules])
        all_lon = np.concatenate([g.lon for g in granules])
        all_scan_line = np.concatenate([g.scan_line for g in granules])
        all_field_of_regard = np.concatenate([g.field_of_regard for g in granules])
        all_field_of_view = np.concatenate([g.field_of_view for g in granules])
        all_sat_za = np.concatenate([g.sat_za for g in granules])
        all_sat_aza = np.concatenate([g.sat_aza for g in granules])
        all_sol_za = np.concatenate([g.sol_za for g in granules])
        all_sol_aza = np.concatenate([g.sol_aza for g in granules])
        all_quality = np.concatenate([g.quality for g in granules])
        all_times = np.concatenate([g.times for g in granules])
        all_brightness_temperature = np.concatenate(
            [g.brightness_temperature for g in granules]
        )  # (N, n_channels)

        n_total = len(all_lat)
        n_channels = all_brightness_temperature.shape[1]

        # Build satellite/variable arrays for dedup support
        # (each granule may have different satellite)
        sat_pieces = [np.broadcast_to(g.satellite, g.lat.shape[0]) for g in granules]
        var_pieces = [np.broadcast_to(g.variable, g.lat.shape[0]) for g in granules]
        source_codes: dict[str, int] = {}
        source_pieces: list[np.ndarray] = []
        for granule in granules:
            source_code = source_codes.setdefault(granule.source_uri, len(source_codes))
            source_pieces.append(
                np.full(granule.lat.shape[0], source_code, dtype=np.int32)
            )
        all_sat = np.concatenate(sat_pieces)
        all_var = np.concatenate(var_pieces)
        source_i = np.concatenate(source_pieces)

        # Free granule list — data is now in the concatenated arrays
        del granules

        # Deduplicate repeated tasks by exact source footprint identity. Do not
        # use rounded time/location: distinct detector FOVs can be nearly
        # collocated. Scan line is granule-local, so source URI is also keyed.
        _, sat_codes = np.unique(all_sat, return_inverse=True)
        _, var_codes = np.unique(all_var, return_inverse=True)
        sat_i = sat_codes.astype(np.int32)
        var_i = var_codes.astype(np.int32)

        # np.lexsort uses the last key as primary; this order mirrors the
        # complete source-identity equality check below.
        order = np.lexsort(
            (
                all_field_of_view,
                all_field_of_regard,
                all_scan_line,
                var_i,
                sat_i,
                source_i,
            )
        )
        sorted_sat = sat_i[order]
        sorted_var = var_i[order]
        sorted_source = source_i[order]
        sorted_scan_line = all_scan_line[order]
        sorted_field_of_regard = all_field_of_regard[order]
        sorted_field_of_view = all_field_of_view[order]
        diffs = (
            (sorted_source[1:] != sorted_source[:-1])
            | (sorted_sat[1:] != sorted_sat[:-1])
            | (sorted_var[1:] != sorted_var[:-1])
            | (sorted_scan_line[1:] != sorted_scan_line[:-1])
            | (sorted_field_of_regard[1:] != sorted_field_of_regard[:-1])
            | (sorted_field_of_view[1:] != sorted_field_of_view[:-1])
        )
        unique_mask: np.ndarray = np.empty(n_total, dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = diffs
        keep_idx = order[unique_mask]
        keep_idx.sort()  # preserve original order

        if len(keep_idx) < n_total:
            all_lat = all_lat[keep_idx]
            all_lon = all_lon[keep_idx]
            all_scan_line = all_scan_line[keep_idx]
            all_field_of_regard = all_field_of_regard[keep_idx]
            all_field_of_view = all_field_of_view[keep_idx]
            all_sat_za = all_sat_za[keep_idx]
            all_sat_aza = all_sat_aza[keep_idx]
            all_sol_za = all_sol_za[keep_idx]
            all_sol_aza = all_sol_aza[keep_idx]
            all_quality = all_quality[keep_idx]
            all_times = all_times[keep_idx]
            all_brightness_temperature = all_brightness_temperature[keep_idx]
            all_sat = all_sat[keep_idx]
            all_var = all_var[keep_idx]
            n_total = len(keep_idx)

        # --- Expand to long-format using PyArrow for efficiency ---
        n_rows = n_total * n_channels
        all_scan_angle = _nominal_cris_scan_angle(
            all_field_of_regard, all_field_of_view
        )

        # The compact decoder has already applied this source-ordered projection.
        _, sensor_chan, wavenumber = self._channel_projection()
        if len(sensor_chan) != n_channels:
            raise RuntimeError("Decoded CrIS channels do not match the projection")

        arrs: dict[str, pa.Array] = {
            "time": pa.array(np.repeat(all_times, n_channels), type=pa.timestamp("ns")),
            "class": pa.DictionaryArray.from_arrays(
                np.zeros(n_rows, dtype=np.int8), ["rad"]
            ),
            "lat": pa.array(np.repeat(all_lat, n_channels), type=pa.float32()),
            "lon": pa.array(np.repeat(all_lon, n_channels), type=pa.float32()),
            "scan_angle": pa.array(
                np.repeat(all_scan_angle, n_channels), type=pa.float32()
            ),
            "scan_line": pa.array(
                np.repeat(all_scan_line, n_channels), type=pa.uint32()
            ),
            "field_of_regard": pa.array(
                np.repeat(all_field_of_regard, n_channels), type=pa.uint16()
            ),
            "field_of_view": pa.array(
                np.repeat(all_field_of_view, n_channels), type=pa.uint16()
            ),
            "sensor_index": pa.array(
                np.tile(sensor_chan, n_total),
                type=pa.uint16(),
            ),
            "wavenumber": pa.array(
                np.tile(wavenumber, n_total),
                type=pa.float64(),
            ),
            "solza": pa.array(np.repeat(all_sol_za, n_channels), type=pa.float32()),
            "solaza": pa.array(np.repeat(all_sol_aza, n_channels), type=pa.float32()),
            "satellite_za": pa.array(
                np.repeat(all_sat_za, n_channels), type=pa.float32()
            ),
            "satellite_aza": pa.array(
                np.repeat(all_sat_aza, n_channels), type=pa.float32()
            ),
            "quality": pa.array(np.repeat(all_quality, n_channels), type=pa.uint16()),
            "observation": pa.array(
                all_brightness_temperature.ravel(), type=pa.float32()
            ),
        }

        # Build satellite and variable using DictionaryArray for memory
        unique_sats = list(dict.fromkeys(all_sat))
        sat_codes = {s: i for i, s in enumerate(unique_sats)}
        sat_indices: np.ndarray = np.repeat(
            np.array([sat_codes[s] for s in all_sat], dtype=np.int8), n_channels
        )
        arrs["satellite"] = pa.DictionaryArray.from_arrays(sat_indices, unique_sats)

        unique_vars = list(dict.fromkeys(all_var))
        var_codes = {v: i for i, v in enumerate(unique_vars)}
        var_indices: np.ndarray = np.repeat(
            np.array([var_codes[v] for v in all_var], dtype=np.int8), n_channels
        )
        arrs["variable"] = pa.DictionaryArray.from_arrays(var_indices, unique_vars)

        # Select only schema-requested columns
        schema_names = [n for n in schema.names if n in arrs]
        tbl = pa.table({n: arrs[n] for n in schema_names})
        result = tbl.to_pandas()

        result.attrs["source"] = self.SOURCE_ID
        return result

    def _decode_hdf5(
        self,
        sdr_path: str,
        geo_path: str,
        task: _CrISAsyncTask,
    ) -> _CrISDecodedGranule | None:
        """Decode a CrIS SDR + GEO HDF5 file pair into compact arrays.

        Returns a :class:`_CrISDecodedGranule` containing spatial arrays
        (one element per valid FOV) and a 2-D brightness-temperature matrix
        ``(n_valid, n_channels)``.  The expensive channel-expansion into
        long-format rows is deferred to
        :py:meth:`_compile_dataframe`.

        Scan lines whose FORTime falls entirely outside the tolerance
        window are skipped before reading the (much larger) radiance
        arrays.

        Parameters
        ----------
        sdr_path : str
            Local path to the CrIS SDR HDF5 file.
        geo_path : str
            Local path to the CrIS GEO HDF5 file.
        task : _CrISAsyncTask
            Task metadata (tolerance bounds, satellite, variable, modifier).
        """
        tmin_dt64 = np.datetime64(task.datetime_min, "us")
        tmax_dt64 = np.datetime64(task.datetime_max, "us")

        # --- Phase 1: Read GEO time first (tiny) to filter scan lines ---
        with h5py.File(geo_path, "r") as geo:
            anchor_utc, anchor_iet = _read_cris_time_anchor(geo)
            # Shape: (n_scan, 30), containing IET microseconds.
            for_time = _read_hdf_dataset(geo, _GEO_KEYS["for_time"])

        # Convert IET relative to the source-provided UTC/IET granule anchor.
        time_dt64: np.ndarray = _iet_to_utc(for_time, anchor_utc, anchor_iet)
        # A scan line is relevant if ANY FOR in that scan is within window
        scan_has_valid_time = np.any(
            (time_dt64 >= tmin_dt64) & (time_dt64 <= tmax_dt64) & (for_time > 0),
            axis=1,
        )  # (n_scan,)

        if not scan_has_valid_time.any():
            return None

        valid_scans = np.where(scan_has_valid_time)[0]

        # --- Phase 2: Read only the scan lines we need ---
        with h5py.File(sdr_path, "r") as sdr, h5py.File(geo_path, "r") as geo:
            # HDF5 fancy indexing with sorted indices (efficient for contiguous)
            rad_lw = _read_hdf_dataset(sdr, _SDR_RADIANCE_KEYS["LW"], valid_scans)
            rad_mw = _read_hdf_dataset(sdr, _SDR_RADIANCE_KEYS["MW"], valid_scans)
            rad_sw = _read_hdf_dataset(sdr, _SDR_RADIANCE_KEYS["SW"], valid_scans)

            try:
                qf3 = _read_hdf_dataset(sdr, _SDR_QF_KEYS["QF3"], valid_scans)
            except KeyError:
                qf3 = np.zeros(
                    (len(valid_scans), _CRIS_NUM_FOR, _CRIS_NUM_FOV, 3),
                    dtype=np.uint8,
                )

            lat = _read_hdf_dataset(geo, _GEO_KEYS["lat"], valid_scans)
            lon = _read_hdf_dataset(geo, _GEO_KEYS["lon"], valid_scans)
            sat_za = _read_hdf_dataset(geo, _GEO_KEYS["sat_za"], valid_scans)
            sat_aza = _read_hdf_dataset(geo, _GEO_KEYS["sat_aza"], valid_scans)
            sol_za = _read_hdf_dataset(geo, _GEO_KEYS["sol_za"], valid_scans)
            sol_aza = _read_hdf_dataset(geo, _GEO_KEYS["sol_aza"], valid_scans)

        # Use the pre-read time but only for valid scans
        for_time = for_time[valid_scans]

        n_scan = len(valid_scans)

        # Concatenate radiance: (n_scan, n_for, n_fov, n_channels)
        radiance = np.concatenate([rad_lw, rad_mw, rad_sw], axis=-1)
        del rad_lw, rad_mw, rad_sw  # free memory early
        n_for = radiance.shape[1]
        n_fov = radiance.shape[2]
        n_channels = radiance.shape[3]
        n_spatial = n_scan * n_for * n_fov

        # Combine QF3 band flags into single uint16 for the quality column.
        # These flags are exposed as metadata; filtering is left to the caller.
        qf_combined = (
            qf3[:, :, :, 0].astype(np.uint16)
            | (qf3[:, :, :, 1].astype(np.uint16) << 4)
            | (qf3[:, :, :, 2].astype(np.uint16) << 8)
        )
        del qf3

        # Flatten spatial dims
        lat_flat: np.ndarray = lat.reshape(-1).astype(np.float32)
        lon_flat: np.ndarray = lon.reshape(-1).astype(np.float32)
        scan_line_flat: np.ndarray = np.repeat(
            valid_scans.astype(np.uint32) + 1, n_for * n_fov
        )
        field_of_regard_flat = np.tile(
            np.repeat(np.arange(1, n_for + 1, dtype=np.uint16), n_fov), n_scan
        )
        field_of_view_flat = np.tile(
            np.arange(1, n_fov + 1, dtype=np.uint16), n_scan * n_for
        )

        # Expand for_time to (n_scan, n_for, n_fov)
        for_time_3d = np.broadcast_to(
            for_time[:, :, np.newaxis], (n_scan, n_for, n_fov)
        )
        time_flat = for_time_3d.reshape(-1)

        # Convert times for tolerance filtering
        times_dt64: np.ndarray = _iet_to_utc(time_flat, anchor_utc, anchor_iet)

        # Valid spatial mask: good lat/lon, positive time, AND within tolerance
        valid_spatial = (
            (lat_flat >= -90.0)
            & (lat_flat <= 90.0)
            & (lon_flat >= -180.1)
            & (lon_flat <= 360.1)
            & (time_flat > 0)
            & (times_dt64 >= tmin_dt64)
            & (times_dt64 <= tmax_dt64)
        )

        if not valid_spatial.any():
            return None

        # Apply spatial mask
        lat_valid = lat_flat[valid_spatial]
        lon_valid = lon_flat[valid_spatial] % 360.0
        times_valid = times_dt64[valid_spatial]
        scan_line_valid = scan_line_flat[valid_spatial]
        field_of_regard_valid = field_of_regard_flat[valid_spatial]
        field_of_view_valid = field_of_view_flat[valid_spatial]

        sat_za_valid = sat_za.reshape(-1)[valid_spatial].astype(np.float32)
        sat_aza_valid = sat_aza.reshape(-1)[valid_spatial].astype(np.float32)
        sol_za_valid = sol_za.reshape(-1)[valid_spatial].astype(np.float32)
        sol_aza_valid = sol_aza.reshape(-1)[valid_spatial].astype(np.float32)
        qf_valid = qf_combined.reshape(-1)[valid_spatial].astype(np.uint16)

        # Radiance: (n_valid, n_channels)
        radiance_valid = radiance.reshape(n_spatial, n_channels)[valid_spatial]
        del radiance  # free the large array

        # Apply modifier
        radiance_valid = task.modifier(radiance_valid).astype(np.float32)

        # Mask out fill values: set them to NaN (kept for now, could filter)
        bad = (radiance_valid <= -1e6) | (radiance_valid >= 1e6)
        if bad.any():
            radiance_valid = radiance_valid.copy()
            radiance_valid[bad] = np.float32("nan")

        # Optional Hamming apodization: apply the NOAA three-point operator to
        # the unapodized (sinc ILS) radiance, then trim the 2 guard channels at
        # each end of each band.
        if self._apodize:
            radiance_valid = _hamming_apodize(radiance_valid)

        # Apodization needs neighboring source channels. Project only after
        # that transform, while the data is still compact radiance spectra.
        channel_positions, _, wn = self._channel_projection()
        radiance_valid = radiance_valid[:, channel_positions]

        # Convert spectral radiance at each channel center wavenumber. Products
        # built with a coefficient package may use release-pinned Planck
        # constants and therefore differ slightly at numerical precision.
        brightness_temperature = radiance_to_bt(radiance_valid, wn).astype(np.float32)

        return _CrISDecodedGranule(
            lat=lat_valid,
            lon=lon_valid,
            scan_line=scan_line_valid,
            field_of_regard=field_of_regard_valid,
            field_of_view=field_of_view_valid,
            sat_za=sat_za_valid,
            sat_aza=sat_aza_valid,
            sol_za=sol_za_valid,
            sol_aza=sol_aza_valid,
            quality=qf_valid,
            times=times_valid,
            brightness_temperature=brightness_temperature,
            source_uri=task.sdr_uri,
            satellite=task.satellite,
            variable=task.variable,
        )

    @staticmethod
    def _parse_filename_time(filename: str) -> datetime | None:
        """Extract the granule start time from a CrIS SDR filename.

        Expected pattern::

            SCRIF_{platform}_d{YYYYMMDD}_t{HHMMSSF}_e{HHMMSSF}_b{orbit}_c{creation}_oebc_ops.h5

        The ``t`` field encodes the start time as HHMMSS followed by a
        tenths-of-second digit.

        Returns ``None`` if the filename does not match.
        """
        parts = filename.split("_")
        # Find the date part (dYYYYMMDD) and time part (tHHMMSSF)
        date_str: str | None = None
        time_str: str | None = None

        for part in parts:
            if part.startswith("d") and len(part) == 9:
                date_str = part[1:]  # YYYYMMDD
            elif part.startswith("t") and len(part) == 8:
                time_str = part[1:7]  # HHMMSS (ignore tenths digit)

        if date_str is None or time_str is None:
            return None

        try:
            return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _granule_key(filename: str) -> str:
        """Extract the matching key from an SDR or GEO filename.

        Both SDR (SCRIF) and GEO (GCRSO) files share a common key
        formed by the platform, date, start-time, end-time, and orbit
        fields.  The creation timestamp (``_c`` field) differs between
        the two products, so it is excluded from the key.

        Parameters
        ----------
        filename : str
            Filename (basename only) of an SDR or GEO HDF5 file.

        Returns
        -------
        str
            Key string ``"{platform}_{date}_{start}_{end}_{orbit}"``.
        """
        parts = filename.split("_")
        # parts: [prefix, platform, d<date>, t<start>, e<end>, b<orbit>, ...]
        if len(parts) >= 6:
            return "_".join(parts[1:6])  # platform_d*_t*_e*_b*
        return filename  # fallback – should not happen for valid files

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert *fields* parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification.

        Returns
        -------
        pa.Schema
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not in SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(field.name).type
                if field.type != expected:
                    raise TypeError(
                        f"Field '{field.name}' type {field.type} != {expected}"
                    )
            return fields

        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in SCHEMA. Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "jpss_cris")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_jpss_cris_{self._tmp_cache_hash}"
            )
        return cache_location

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic local cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode())
        return os.path.join(self.cache, sha.hexdigest())

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that requested times are within the data range.

        Parameters
        ----------
        times : list[datetime]
            Date-times to validate.
        """
        start_date = min(_SAT_START_DATE.values())
        for t in times:
            if t < start_date:
                raise ValueError(
                    f"Requested date time {t} needs to be after "
                    f"{start_date} for JPSS CrIS"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check whether data is available for a given time.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date-time to check.

        Returns
        -------
        bool
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
