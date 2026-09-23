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

"""Calculate relative humidity and wet-bulb temperature.

The calculations use air temperature in kelvin, specific humidity in kg/kg, and pressure in
pascals. They use only NumPy and do not depend on a particular model, grid, or DSX.

For best accuracy, all three inputs should represent the same atmospheric level. Combining a
2-metre temperature with humidity or pressure from a higher level can reduce accuracy; these
functions do not correct that mismatch.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

# Inputs are converted to NumPy float64 for stable numerical precision. Results may be a NumPy
# scalar or an ndarray, depending on the input shapes and broadcasting.
_FloatOrArray: TypeAlias = np.float64 | NDArray[np.float64]

# Source: ECMWF IFS Documentation CY48R1, Part IV:
# https://www.ecmwf.int/en/elibrary/81370-ifs-documentation-cy48r1-part-iv-physical-processes
# Ratio of the gas constants for dry air and water vapour (Rd/Rv).
_EPSILON = 0.621981

# Constants used in the wet-bulb energy calculation.
_CP_AIR = 1005.0  # specific heat of air at constant pressure, J/(kg K)
_LATENT_HEAT_VAP = 2.501e6  # latent heat of vaporisation near 0 degC, J/kg

# Allow small input or numerical deviations above saturation without logging a warning.
_SUPERSATURATION_WARNING_PERCENT = 102.0


def _saturation_vapor_pressure(t_K: ArrayLike) -> _FloatOrArray:
    """Mixed-phase saturation vapour pressure (Pa) per ECMWF IFS (CY48R1)."""
    T = np.asarray(t_K, dtype=float)
    es_w = 611.21 * np.exp(17.502 * (T - 273.16) / (T - 32.19))
    es_i = 611.21 * np.exp(22.587 * (T - 273.16) / (T + 0.7))
    # Blend quadratically from ice at or below -23°C to liquid water at or above 0°C.
    alpha = np.clip((T - 250.16) / (273.16 - 250.16), 0.0, 1.0) ** 2
    return alpha * es_w + (1.0 - alpha) * es_i


def rh_from_specific_humidity(
    t_K: ArrayLike, q: ArrayLike, p_Pa: ArrayLike
) -> _FloatOrArray:
    """Relative humidity from temperature, specific humidity, and pressure.

    Uses the ECMWF IFS mixed-phase saturation calculation: liquid water at or above 0°C, ice at or
    below -23°C, and a quadratic blend between those temperatures.

    Parameters
    ----------
    t_K : ArrayLike
        Temperature (Kelvin).
    q : ArrayLike
        Specific humidity (kg/kg).
    p_Pa : ArrayLike
        Pressure (Pa).

    Returns
    -------
    np.float64 | np.ndarray
        Relative humidity in percent. Finite values are clipped to [0, 100]. Locations with any
        non-finite input return ``NaN``. Raw values above 102% are logged before clipping.
    """
    T = np.asarray(t_K, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p = np.asarray(p_Pa, dtype=float)
    T, q_arr, p = np.broadcast_arrays(T, q_arr, p)
    valid = np.isfinite(T) & np.isfinite(q_arr) & np.isfinite(p)

    # Use valid placeholders for non-finite inputs during the calculation; restore those outputs
    # to NaN.
    T = np.where(valid, T, 273.15)
    q_arr = np.where(valid, q_arr, 0.0)
    p = np.where(valid, p, 101325.0)

    # Vapour pressure from specific humidity: e = p*q / (epsilon + q*(1 - epsilon)).
    e = p * q_arr / (_EPSILON + q_arr * (1.0 - _EPSILON))
    es = _saturation_vapor_pressure(T)
    raw = 100.0 * e / es
    # Log substantial supersaturation while avoiding warnings for minor input or numerical noise.
    if np.any(np.asarray(raw) > _SUPERSATURATION_WARNING_PERCENT):
        logger.warning(
            "supersaturation: raw RH up to {:.1f}% clipped to 100%",
            float(np.nanmax(raw)),
        )
    result = np.clip(raw, 0.0, 100.0)
    return np.where(valid, result, np.nan)


def wet_bulb_K(t_K: ArrayLike, q: ArrayLike, p_Pa: ArrayLike) -> _FloatOrArray:
    """Calculate wet-bulb temperature while accounting for air pressure.

    The function solves the wet-bulb energy balance with a vectorized bisection search. It uses the
    same mixed-phase saturation calculation as :func:`rh_from_specific_humidity`. Supersaturated
    input is limited to saturation, for which wet-bulb temperature equals air temperature.

    Parameters
    ----------
    t_K : ArrayLike
        Air temperature in kelvin.
    q : ArrayLike
        Specific humidity in kg/kg.
    p_Pa : ArrayLike
        Air pressure in pascals.

    Returns
    -------
    np.float64 | np.ndarray
        Wet-bulb temperature in kelvin. Locations with any non-finite input return ``NaN``.

    Notes
    -----
    The latent heat of vaporization is held constant at 2.501e6 J/kg.
    """
    T = np.asarray(t_K, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p = np.asarray(p_Pa, dtype=float)
    T, q_arr, p = np.broadcast_arrays(T, q_arr, p)
    valid = np.isfinite(T) & np.isfinite(q_arr) & np.isfinite(p)

    # Use valid placeholders for non-finite inputs during the calculation; restore those outputs
    # to NaN.
    T = np.where(valid, T, 273.15)
    q_arr = np.where(valid, q_arr, 0.0)
    p = np.where(valid, p, 101325.0)

    # Actual vapour pressure from specific humidity (same relation as the RH derivation).
    e = p * q_arr / (_EPSILON + q_arr * (1.0 - _EPSILON))
    # Wet-bulb temperature cannot exceed air temperature. For supersaturated input, limit vapour
    # pressure to saturation so the result becomes Tw = T and remains inside the search range.
    e = np.minimum(e, _saturation_vapor_pressure(T))
    gamma = _CP_AIR * p / (_EPSILON * _LATENT_HEAT_VAP)
    hi = np.array(T, dtype=float)  # writable copy; Tw <= T (dry-bulb)
    lo = hi - 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        # A positive result means the midpoint is too warm, so move the upper bound down.
        f = _saturation_vapor_pressure(mid) - gamma * (T - mid) - e
        below = f > 0.0
        hi = np.where(below, mid, hi)
        lo = np.where(below, lo, mid)
    result = 0.5 * (lo + hi)
    return np.where(valid, result, np.nan)
