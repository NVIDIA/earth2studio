# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HealDA's vertical-coordinate convention for GPS radio-occultation levels.

A bending-angle level carries an impact parameter, not a pressure. HealDA
mixes GPS-RO with conventional observations and needs a finite pressure and
height per level; it derives both from the occultation's own refractivity
levels (``HEIT``/``ARFR``), never from a retrieval product:

- ``refraction_corrected_height``: the height where the refractive index
  profile satisfies ``n(h) * (R_c + h) = a`` for impact parameter ``a``.
- ``dry_pressure_hpa``: hydrostatic integration of the dry-air density implied
  by refractivity, ``rho = 100 N / (k1 R_d)``.
- ``height_to_pressure_hpa``: the 1976 US Standard Atmosphere at a height.
- ``blended_pressure_hpa``: ``0.8 * standard + 0.2 * dry`` below 5 km and pure
  dry above.

``gpsro_level_coordinates`` combines them with the fallback order used in
HealDA training (blended -> dry -> standard -> standard at geometric impact
height) so a level always gets a finite pressure when its impact height is
finite. ``assign_gpsro_coordinates`` applies it per occultation to a
conventional observation DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Smith-Weintraub dry-air refractivity constant, K hPa^-1.
K1_REFRACTIVITY = 77.6
R_DRY = 287.05
G0 = 9.80665
BLEND_TOP_M = 5000.0
BLEND_STANDARD_WEIGHT = 0.8
# Log-linear fit depth for the top-of-profile boundary pressure.
SCALE_HEIGHT_FIT_DEPTH_M = 10_000.0
# Mean Earth radius; the message ELRC differs by <~20 km, which moves the
# refraction-corrected height by only metres.
NOMINAL_RADIUS_CURVATURE_M = 6_371_000.0
GPS_VARIABLE = "gps"
GPS_REFRACTIVITY_VARIABLE = "gps_refractivity"


def height_to_pressure_hpa(height_m: np.ndarray) -> np.ndarray:
    """Pressure (hPa) at a geometric height from the 1976 US Standard Atmosphere.

    Parameters
    ----------
    height_m : np.ndarray
        Heights in meters; clipped to ``[0, 60000]``.

    Returns
    -------
    np.ndarray
        Pressure in hPa (float32).
    """
    h = np.clip(np.asarray(height_m, dtype=np.float64), 0.0, 60_000.0)
    p = np.empty(h.shape, dtype=np.float64)

    trop = h <= 11_000.0
    p[trop] = 1013.25 * np.power(np.maximum(1.0 - 2.25577e-5 * h[trop], 1e-6), 5.25588)

    strat1 = (h > 11_000.0) & (h <= 20_000.0)
    p[strat1] = 226.321 * np.exp(-1.57686e-4 * (h[strat1] - 11_000.0))

    strat2 = (h > 20_000.0) & (h <= 32_000.0)
    t2 = 216.65 + 0.001 * (h[strat2] - 20_000.0)
    p[strat2] = 54.7489 * np.power(t2 / 216.65, -34.1632)

    strat3 = (h > 32_000.0) & (h <= 47_000.0)
    t3 = 228.65 + 0.0028 * (h[strat3] - 32_000.0)
    p[strat3] = 8.68019 * np.power(t3 / 228.65, -12.2011)

    strat4 = (h > 47_000.0) & (h <= 51_000.0)
    p[strat4] = 1.10906 * np.exp(-1.26227e-4 * (h[strat4] - 47_000.0))

    meso = h > 51_000.0
    t5 = 270.65 - 0.0028 * (h[meso] - 51_000.0)
    p[meso] = 0.669387 * np.power(np.maximum(t5 / 270.65, 1e-6), 12.2011)

    return p.astype(np.float32)


def _clean_profile(
    height_m: np.ndarray, refractivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Finite, positive refractivity levels sorted by ascending height."""
    h = np.asarray(height_m, dtype=np.float64)
    n = np.asarray(refractivity, dtype=np.float64)
    keep = np.isfinite(h) & np.isfinite(n) & (n > 0.0)
    h, n = h[keep], n[keep]
    if h.size == 0:
        return h, n
    order = np.argsort(h, kind="stable")
    h, n = h[order], n[order]
    # Duplicate heights make interpolation ill-defined; keep the first.
    unique = np.concatenate([[True], np.diff(h) > 0.0])
    return h[unique], n[unique]


def refraction_corrected_height(
    impact_parameter_m: np.ndarray,
    radius_curvature_m: float,
    profile_height_m: np.ndarray,
    profile_refractivity: np.ndarray,
    iterations: int = 24,
) -> np.ndarray:
    """Height where ``n(h) * (R_c + h)`` equals the impact parameter.

    Bouguer's rule ties the impact parameter ``a`` of a ray to the radius
    ``r = R_c + h`` of its tangent point through the refractive index
    ``n = 1 + 1e-6 N``. The geometric ``a - R_c`` overstates the tangent height
    by ``~1e-6 N r`` (about 2 km near the surface); this iterates the fixed
    point ``h = a / n(h) - R_c`` with ``N`` interpolated linearly in height and
    held at its end values outside the profile. The map contracts by
    ``~1e-6 r |dN/dh|`` (about 0.3 near the surface) per step, so 24 steps
    take a 2 km initial error below a millimetre.

    Parameters
    ----------
    impact_parameter_m : np.ndarray
        Impact parameter per bending-angle level (m).
    radius_curvature_m : float
        Local Earth radius of curvature (m).
    profile_height_m : np.ndarray
        Heights of the refractivity levels (m).
    profile_refractivity : np.ndarray
        Refractivity at those levels (N-units).
    iterations : int, optional
        Fixed-point iterations, by default 24.

    Returns
    -------
    np.ndarray
        Refraction-corrected tangent height (m); NaN where the profile has
        fewer than two usable levels or the impact parameter is not finite.
    """
    a = np.asarray(impact_parameter_m, dtype=np.float64)
    h_profile, n_profile = _clean_profile(profile_height_m, profile_refractivity)
    if h_profile.size < 2:
        return np.full(a.shape, np.nan, dtype=np.float32)
    height = a - radius_curvature_m
    for _ in range(iterations):
        refractivity = np.interp(height, h_profile, n_profile)
        height = a / (1.0 + 1e-6 * refractivity) - radius_curvature_m
    height = np.where(np.isfinite(a), height, np.nan)
    return height.astype(np.float32)


def dry_pressure_profile_hpa(
    profile_height_m: np.ndarray, profile_refractivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hydrostatic dry pressure on the refractivity levels.

    Dry-air refractivity ``N = k1 P / T`` and the ideal gas law give the density
    ``rho = 100 N / (k1 R_d)`` without knowing the temperature. Pressure is the
    weight of the air above: the profile is integrated downward from the top
    with the trapezoid rule, and the boundary above the top level is closed
    analytically as ``rho_top g H`` where ``H`` is the e-folding scale height
    fitted to ``ln N`` over the top :data:`SCALE_HEIGHT_FIT_DEPTH_M`.

    Parameters
    ----------
    profile_height_m : np.ndarray
        Heights of the refractivity levels (m).
    profile_refractivity : np.ndarray
        Refractivity at those levels (N-units).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ascending level heights (m) and dry pressure (hPa) on them; both empty
        when fewer than two usable levels exist.
    """
    h, n = _clean_profile(profile_height_m, profile_refractivity)
    if h.size < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    density = 100.0 * n / (K1_REFRACTIVITY * R_DRY)

    top_layer = h >= h[-1] - SCALE_HEIGHT_FIT_DEPTH_M
    if top_layer.sum() < 2:
        top_layer[-2:] = True
    slope = np.polyfit(h[top_layer], np.log(n[top_layer]), 1)[0]
    # A non-decaying top layer has no finite column above it; fall back to the
    # standard atmosphere's ~7 km scale height rather than an infinite pressure.
    scale_height = -1.0 / slope if slope < 0.0 else 7_000.0
    p_top = density[-1] * G0 * scale_height

    layer_mass = 0.5 * (density[1:] + density[:-1]) * np.diff(h) * G0
    pressure_pa = p_top + np.concatenate([np.cumsum(layer_mass[::-1])[::-1], [0.0]])
    return h, pressure_pa / 100.0


def dry_pressure_hpa(
    height_m: np.ndarray,
    profile_height_m: np.ndarray,
    profile_refractivity: np.ndarray,
) -> np.ndarray:
    """Dry pressure (hPa) at arbitrary heights, log-linear in the profile.

    Parameters
    ----------
    height_m : np.ndarray
        Heights to evaluate (m).
    profile_height_m : np.ndarray
        Heights of the refractivity levels (m).
    profile_refractivity : np.ndarray
        Refractivity at those levels (N-units).

    Returns
    -------
    np.ndarray
        Dry pressure (hPa, float32); NaN outside the profile's height span or
        when the profile is unusable.
    """
    height = np.asarray(height_m, dtype=np.float64)
    h, p = dry_pressure_profile_hpa(profile_height_m, profile_refractivity)
    if h.size < 2:
        return np.full(height.shape, np.nan, dtype=np.float32)
    inside = np.isfinite(height) & (height >= h[0]) & (height <= h[-1])
    out = np.full(height.shape, np.nan, dtype=np.float64)
    out[inside] = np.exp(np.interp(height[inside], h, np.log(p)))
    return out.astype(np.float32)


def blended_pressure_hpa(
    height_m: np.ndarray, dry_hpa: np.ndarray, standard_hpa: np.ndarray
) -> np.ndarray:
    """``0.8 * standard + 0.2 * dry`` below 5 km, dry above; NaN if either is."""
    height = np.asarray(height_m, dtype=np.float64)
    dry = np.asarray(dry_hpa, dtype=np.float64)
    standard = np.asarray(standard_hpa, dtype=np.float64)
    low = BLEND_STANDARD_WEIGHT * standard + (1.0 - BLEND_STANDARD_WEIGHT) * dry
    blended = np.where(height < BLEND_TOP_M, low, dry)
    blended = np.where(np.isfinite(dry) & np.isfinite(standard), blended, np.nan)
    return blended.astype(np.float32)


def gpsro_level_coordinates(
    impact_parameter_m: np.ndarray,
    radius_curvature_m: float,
    profile_height_m: np.ndarray,
    profile_refractivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pressure and height coordinates for bending-angle levels.

    Parameters
    ----------
    impact_parameter_m : np.ndarray
        Impact parameter per bending-angle level (m).
    radius_curvature_m : float
        Local Earth radius of curvature (m).
    profile_height_m : np.ndarray
        Heights of the message's refractivity levels (m); may be empty.
    profile_refractivity : np.ndarray
        Refractivity at those levels (N-units); may be empty.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(pressure_hpa, height_m)`` per level. Height is the
        refraction-corrected tangent height when a refractivity profile is
        available, else the geometric impact height. Pressure is the first
        finite, positive candidate of blended, dry, standard-at-corrected-height
        and standard-at-geometric-height.
    """
    a = np.asarray(impact_parameter_m, dtype=np.float64)
    geometric = (a - radius_curvature_m).astype(np.float32)
    corrected = refraction_corrected_height(
        a, radius_curvature_m, profile_height_m, profile_refractivity
    )
    height = np.where(np.isfinite(corrected) & (corrected > 0.0), corrected, geometric)

    dry = dry_pressure_hpa(height, profile_height_m, profile_refractivity)
    standard = height_to_pressure_hpa(height)
    blended = blended_pressure_hpa(height, dry, standard)

    pressure: np.ndarray = height_to_pressure_hpa(geometric).astype(np.float64)
    for candidate in (standard, dry, blended):
        c = np.asarray(candidate, dtype=np.float64)
        pressure = np.where(np.isfinite(c) & (c > 0.0), c, pressure)
    return pressure.astype(np.float32), height.astype(np.float32)


def qfro_bit_set(qfro: np.ndarray | int | None, bit: int) -> np.ndarray:
    """True where WMO 0-33-039 ``QFRO`` bit ``bit`` is set (bit 1 is the MSB
    of the 16-bit flag table)."""
    values = np.asarray(qfro, dtype=np.float64)
    values = np.where(np.isnan(values), 0.0, values).astype(np.int64)
    return ((values >> (16 - int(bit))) & 1) == 1


def assign_gpsro_coordinates(
    frame: pd.DataFrame, radius_curvature_m: float = NOMINAL_RADIUS_CURVATURE_M
) -> pd.DataFrame:
    """Derive ``pres``/``elev`` for ``gps`` rows from ``gps_refractivity`` rows.

    Rows are grouped into occultations by ``time`` and ``type`` (plus
    ``station`` when present). Within each occultation the ``gps`` rows'
    geometric impact height (``elev``) and the refractivity profile
    (``gps_refractivity`` observation vs ``elev``) feed
    :func:`gpsro_level_coordinates`; ``pres`` becomes the derived pressure in
    Pa and ``elev`` the refraction-corrected height. All ``gps_refractivity``
    rows are dropped from the result.

    Parameters
    ----------
    frame : pd.DataFrame
        Conventional observations in the NCEP public schema.
    radius_curvature_m : float, optional
        Earth radius of curvature used to recover the impact parameter, by
        default :data:`NOMINAL_RADIUS_CURVATURE_M` (metre-level height error
        versus the message's ``ELRC``).

    Returns
    -------
    pd.DataFrame
        Copy of ``frame`` without refractivity rows.
    """
    out = frame.reset_index(drop=True)
    variable = out["variable"].astype(str)
    is_gps = (variable == GPS_VARIABLE).to_numpy()
    is_profile = (variable == GPS_REFRACTIVITY_VARIABLE).to_numpy()
    if not is_gps.any():
        return out.loc[~is_profile].copy()

    keys = ["time", "type"] + (["station"] if "station" in out.columns else [])
    pres = out["pres"].to_numpy(dtype=np.float64, na_value=np.nan).copy()
    elev = out["elev"].to_numpy(dtype=np.float64, na_value=np.nan).copy()
    gps_family = out.loc[is_gps | is_profile]
    for _, group in gps_family.groupby(keys, sort=False, dropna=False):
        group_idx = group.index.to_numpy()
        gps_idx = group_idx[is_gps[group_idx]]
        if gps_idx.size == 0:
            continue
        profile_idx = group_idx[is_profile[group_idx]]
        pressure_hpa, height = gpsro_level_coordinates(
            elev[gps_idx] + radius_curvature_m,
            radius_curvature_m,
            elev[profile_idx],
            out["observation"].to_numpy(dtype=np.float64, na_value=np.nan)[profile_idx],
        )
        pres[gps_idx] = pressure_hpa.astype(np.float64) * 100.0
        elev[gps_idx] = height
    out["pres"] = pres.astype(np.float32)
    out["elev"] = elev.astype(np.float32)
    return out.loc[~is_profile].copy()
