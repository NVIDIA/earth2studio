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
"""Spectral axis and brightness temperature conversion for hyperspectral IR sounders.

IASI and CrIS are archived as radiance and must be converted to brightness temperature
before use. AIRS is already in kelvin in the NNJA aggregate BUFR files.

Planck constants follow CODATA 2018 in the units CRTM and GSI use. The same pair is
referred to as PLANCK_C1 / PLANCK_C2 in some CRTM references.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# CODATA 2018 Planck constants in mW / m^2 / sr / cm^4 and cm·K respectively.
# Source: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
C1 = 1.191042972e-5  # mW m^-2 sr^-1 cm^4
C2 = 1.438776877  # cm K

# IASI: uniform spectral grid, EUMETSAT L1C. Verified against GSI diagnostics.
IASI_FIRST_WAVENUMBER_CM = 645.0
IASI_SPACING_CM = 0.25
IASI_CHANNELS = 8461

# CrIS FSR: three bands numbered contiguously in CRTM order. Band edges verified against
# the archive's band_calibration_quality column boundaries.
CRIS_SPACING_CM = 0.625
CRIS_BANDS = (
    (650.0, 1, 713),  # long wave
    (1210.0, 714, 1578),  # mid wave
    (2155.0, 1579, 2211),  # short wave
)


def wavenumber_cm_inverse(sensor: str, channels: Sequence[int] | Any) -> Any:
    """Centre wavenumber in cm⁻¹ for each channel number.

    AIRS has no formulaic spectral grid — its per-channel wavenumbers are
    read from the ``LOGRCW`` field encoded in the aggregate BUFR instead.

    Parameters
    ----------
    sensor : str
        One of ``"iasi"``, ``"cris"``.
    channels : sequence of int
        Instrument channel numbers (1-based).
    """
    channels = np.asarray(channels, dtype=np.int64)
    if sensor == "iasi":
        _check_range(sensor, channels, 1, IASI_CHANNELS)
        return IASI_FIRST_WAVENUMBER_CM + IASI_SPACING_CM * (channels - 1)
    if sensor == "cris":
        _check_range(sensor, channels, CRIS_BANDS[0][1], CRIS_BANDS[-1][2])
        result = np.zeros(channels.shape, dtype=np.float64)
        for start, first, last in CRIS_BANDS:
            inside = (channels >= first) & (channels <= last)
            result[inside] = start + CRIS_SPACING_CM * (channels[inside] - first)
        return result
    raise ValueError(f"no spectral axis defined for sensor {sensor!r}")


def brightness_temperature(
    radiance: Any,
    wavenumber: Any,
    *,
    dtype: Any = np.float64,
) -> Any:
    """Monochromatic Planck inversion: radiance → brightness temperature.

    Parameters
    ----------
    radiance : array-like
        Spectral radiance in mW m⁻² sr⁻¹ (cm⁻¹)⁻¹.
    wavenumber : array-like
        Channel centre wavenumber in cm⁻¹, same shape as ``radiance``.

    Returns
    -------
    np.ndarray
        Brightness temperature in K. NaN where radiance ≤ 0.
    """
    radiance = np.asarray(radiance, dtype=dtype)
    wavenumber = np.asarray(wavenumber, dtype=np.float64)
    numerator = np.asarray(C2 * wavenumber, dtype=dtype)
    argument = np.asarray(C1 * wavenumber**3, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature = numerator / np.log(1.0 + argument / radiance)
    return np.where(radiance > 0, temperature, np.nan)


def iasi_radiance_mw(scra: Any, chsf: Any) -> Any:
    """Convert IASI SCRA integer codes to mW m⁻² sr⁻¹ (cm⁻¹)⁻¹.

    The NNJA IASI BUFR stores per-footprint SCRA integer values and per-band
    CHSF scale exponents. The physical radiance in W m⁻² sr⁻¹ (m⁻¹)⁻¹
    (per unit wavenumber) is ``SCRA × 10^(-CHSF)``. Converting to
    mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ adds a factor of 10⁵ (×1000 for mW, ×100 for
    per-cm⁻¹ vs per-m⁻¹).

    Parameters
    ----------
    scra : array-like
        Raw SCRA integer values from the BUFR (pybufrkit decoded, not yet scaled).
    chsf : array-like
        Per-channel CHSF exponent, same shape as ``scra``.
    """
    scra = np.asarray(scra, dtype=np.float64)
    chsf = np.asarray(chsf, dtype=np.float64)
    return scra * np.power(10.0, -chsf) * 1e5


def cris_radiance_mw(srad: Any) -> Any:
    """Convert CrIS SRAD (W m⁻² sr⁻¹ (cm⁻¹)⁻¹) to mW m⁻² sr⁻¹ (cm⁻¹)⁻¹."""
    return np.asarray(srad, dtype=np.float64) * 1000.0


def _check_range(sensor: str, channels: Any, low: int, high: int) -> None:
    if channels.size and (channels.min() < low or channels.max() > high):
        raise ValueError(
            f"{sensor} channels must be in [{low}, {high}]; "
            f"got [{channels.min()}, {channels.max()}]"
        )
