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
"""Spectral axis and radiance unit conversions for hyperspectral IR sounders.

IASI and CrIS are archived as radiance and must be converted to brightness temperature
before use (via :func:`earth2studio.data.utils.radiance_to_bt`; pure Planck inversion
with no band correction, matching the convention used by the MetOp IASI and JPSS CrIS
granule sources). AIRS is already in kelvin in the NNJA aggregate BUFR files.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# IASI: uniform spectral grid, EUMETSAT L1C. Verified against GSI diagnostics.
IASI_FIRST_WAVENUMBER_CM = 645.0
IASI_SPACING_CM = 0.25
IASI_CHANNELS = 8461

# CrIS FSR: three bands (start wavenumber cm⁻¹, first channel, last channel)
# numbered contiguously on the CRTM/GSI sensor_chan grid; the same grid as
# jpss_cris.py builds from its per-band science channel counts. Band ranges
# match the NFQF/NCQF per-band quality struct in the crisf4 BUFR (0-33-077 /
# 0-33-076).
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
