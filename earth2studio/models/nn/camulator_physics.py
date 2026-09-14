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

# Source: https://github.com/WillyChap/miles-credit (branch camulator_huggingface)
# credit/postblock/_postblock.py, credit/physics_core.py, climate/WindPP.py
# Apache-2.0, Copyright NSF NCAR Machine Integration and Learning for Earth
# Systems (MILES). Batched re-implementation of the CAMulator inference-time
# conservation fixers and wind-artifact filter.

from collections.abc import Sequence

import torch
import torch.nn.functional as F

# Physical constants (credit/physics_constants.py)
RAD_EARTH = 6371000.0  # m
GRAVITY = 9.80665  # m s-2
RHO_WATER = 1000.0  # kg m-3
LH_WATER = 2.501e6  # J kg-1
CP_DRY = 1004.64  # J kg-1 K-1
CP_VAPOR = 1810.0  # J kg-1 K-1


def grid_area(lat2d: torch.Tensor, lon2d: torch.Tensor) -> torch.Tensor:
    """Grid-cell area (m^2) as ``R^2 * d(sin lat) * d(lon)`` with second-order
    gradients, matching CREDIT's ``physics_hybrid_sigma_level``.

    Parameters
    ----------
    lat2d : torch.Tensor
        Latitude in degrees, shape (lat, lon).
    lon2d : torch.Tensor
        Longitude in degrees, shape (lat, lon).

    Returns
    -------
    torch.Tensor
        Area in m^2, shape (lat, lon).
    """
    lat_rad = torch.deg2rad(lat2d)
    lon_rad = torch.deg2rad(lon2d)
    d_phi = torch.gradient(torch.sin(lat_rad), dim=0, edge_order=2)[0]
    d_lambda = torch.gradient(lon_rad, dim=1, edge_order=2)[0]
    d_lambda = (d_lambda + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.abs(RAD_EARTH**2 * d_phi * d_lambda)


def weighted_sum(q: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Area-weighted global sum over the last two dims, accumulated in float64.

    Parameters
    ----------
    q : torch.Tensor
        Field with shape (..., lat, lon).
    area : torch.Tensor
        Cell area, shape (lat, lon).

    Returns
    -------
    torch.Tensor
        Sum with shape (...), in the dtype of ``q``.
    """
    return (q.double() * area.double()).sum(dim=(-2, -1)).to(q.dtype)


def column_integral(
    q: torch.Tensor, sp: torch.Tensor, hyai: torch.Tensor, hybi: torch.Tensor
) -> torch.Tensor:
    """Vertical pressure integral of a layer-midpoint quantity on hybrid
    sigma-pressure levels, ``sum_k q_k * (p_{k+1} - p_k)``.

    Parameters
    ----------
    q : torch.Tensor
        Layer values, shape (batch, level, lat, lon).
    sp : torch.Tensor
        Surface pressure (Pa), shape (batch, lat, lon).
    hyai : torch.Tensor
        Interface hybrid ``a`` coefficients (Pa), shape (level + 1,).
    hybi : torch.Tensor
        Interface hybrid ``b`` coefficients, shape (level + 1,).

    Returns
    -------
    torch.Tensor
        Integral, shape (batch, lat, lon).
    """
    pressure = hyai.view(1, -1, 1, 1) + hybi.view(1, -1, 1, 1) * sp.unsqueeze(1)
    delta_p = pressure.diff(dim=1)
    return torch.sum(q * delta_p, dim=1)


def tracer_clip(
    y: torch.Tensor,
    channels: Sequence[int],
    center: torch.Tensor,
    scale: torch.Tensor,
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> torch.Tensor:
    """Clip tracer channels of a normalized tensor to physical bounds
    (denormalize, clamp, renormalize per channel).

    Parameters
    ----------
    y : torch.Tensor
        Normalized prediction, shape (batch, channel, lat, lon). Modified in place.
    channels : Sequence[int]
        Channel indices to clip.
    center : torch.Tensor
        Per-tracer normalization mean, shape (len(channels),).
    scale : torch.Tensor
        Per-tracer normalization std, shape (len(channels),).
    minimum : Sequence[float]
        Physical lower bound per tracer.
    maximum : Sequence[float]
        Physical upper bound per tracer (``inf`` for none).

    Returns
    -------
    torch.Tensor
        The clipped tensor ``y``.
    """
    for i, c in enumerate(channels):
        m = center[i]
        s = scale[i]
        chan = y[:, c] * s + m
        chan = torch.clamp(chan, min=float(minimum[i]), max=float(maximum[i]))
        y[:, c] = (chan - m) / s
    return y


def global_mass_fix(
    sp_in: torch.Tensor,
    q_in: torch.Tensor,
    sp_pred: torch.Tensor,
    q_pred: torch.Tensor,
    hyai: torch.Tensor,
    hybi: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    """Rescale predicted surface pressure so that the global dry-air mass equals
    that of the input state (CREDIT ``GlobalMassFixer`` on hybrid sigma levels;
    specific total water is left untouched).

    Parameters
    ----------
    sp_in : torch.Tensor
        Input surface pressure (Pa), shape (batch, lat, lon).
    q_in : torch.Tensor
        Input specific total water (kg kg-1), shape (batch, level, lat, lon).
    sp_pred : torch.Tensor
        Predicted surface pressure (Pa), shape (batch, lat, lon).
    q_pred : torch.Tensor
        Predicted specific total water (kg kg-1), shape (batch, level, lat, lon).
    hyai : torch.Tensor
        Interface hybrid ``a`` coefficients (Pa), shape (level + 1,).
    hybi : torch.Tensor
        Interface hybrid ``b`` coefficients, shape (level + 1,).
    area : torch.Tensor
        Cell area (m^2), shape (lat, lon).

    Returns
    -------
    torch.Tensor
        Corrected surface pressure, shape (batch, lat, lon).
    """
    mass_dry_t0 = weighted_sum(
        column_integral(1 - q_in, sp_in, hyai, hybi) / GRAVITY, area
    )

    delta_a = hyai.diff().view(1, -1, 1, 1)
    delta_b = hybi.diff().view(1, -1, 1, 1)
    p_dry_a = (delta_a * (1 - q_pred)).sum(1)
    p_dry_b = (delta_b * (1 - q_pred)).sum(1)
    # CREDIT accumulates these two terms in the working precision (float32).
    mass_dry_a = (p_dry_a * area).sum((-2, -1)) / GRAVITY
    mass_dry_b = (p_dry_b * sp_pred * area).sum((-2, -1)) / GRAVITY

    ratio = (mass_dry_t0 - mass_dry_a) / mass_dry_b
    return sp_pred * ratio.view(-1, 1, 1)


def global_water_fix(
    sp_in: torch.Tensor,
    q_in: torch.Tensor,
    sp_pred: torch.Tensor,
    q_pred: torch.Tensor,
    precip: torch.Tensor,
    evapor: torch.Tensor,
    hyai: torch.Tensor,
    hybi: torch.Tensor,
    area: torch.Tensor,
    n_seconds: float,
) -> torch.Tensor:
    """Rescale predicted precipitation so the global water budget closes
    (CREDIT ``GlobalWaterFixer``).

    Parameters
    ----------
    sp_in, q_in : torch.Tensor
        Input surface pressure (Pa) and specific total water (kg kg-1).
    sp_pred, q_pred : torch.Tensor
        Predicted surface pressure (Pa) and specific total water (kg kg-1).
    precip : torch.Tensor
        Predicted precipitation over the step (m), shape (batch, lat, lon).
    evapor : torch.Tensor
        Predicted surface water flux over the step (m, positive into the
        surface, i.e. negative for evaporation), shape (batch, lat, lon).
    hyai, hybi : torch.Tensor
        Interface hybrid coefficients, shape (level + 1,).
    area : torch.Tensor
        Cell area (m^2), shape (lat, lon).
    n_seconds : float
        Length of the model step in seconds.

    Returns
    -------
    torch.Tensor
        Corrected precipitation (m), shape (batch, lat, lon).
    """
    precip_flux = precip * RHO_WATER / n_seconds
    evapor_flux = evapor * RHO_WATER / n_seconds

    twc_in = column_integral(q_in, sp_in, hyai, hybi) / GRAVITY
    twc_pred = column_integral(q_pred, sp_pred, hyai, hybi) / GRAVITY
    dtwc_dt = (twc_pred - twc_in) / n_seconds

    twc_sum = weighted_sum(dtwc_dt, area)
    e_sum = weighted_sum(evapor_flux, area)
    p_sum = weighted_sum(precip_flux, area)

    residual = -twc_sum - e_sum - p_sum
    ratio = (p_sum + residual) / p_sum
    return precip * ratio.view(-1, 1, 1)


def global_energy_fix(
    sp_in: torch.Tensor,
    t_in: torch.Tensor,
    q_in: torch.Tensor,
    u_in: torch.Tensor,
    v_in: torch.Tensor,
    sp_pred: torch.Tensor,
    t_pred: torch.Tensor,
    q_pred: torch.Tensor,
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    toa_down_sw: torch.Tensor,
    toa_up_sw: torch.Tensor,
    toa_up_lw: torch.Tensor,
    surf_down_sw: torch.Tensor,
    surf_up_sw: torch.Tensor,
    surf_down_lw: torch.Tensor,
    surf_up_lw: torch.Tensor,
    surf_sh: torch.Tensor,
    surf_lh: torch.Tensor,
    phis: torch.Tensor,
    hyai: torch.Tensor,
    hybi: torch.Tensor,
    area: torch.Tensor,
    n_seconds: float,
) -> torch.Tensor:
    """Rescale predicted temperature so the global column total-energy tendency
    matches the net TOA minus surface energy fluxes (CREDIT
    ``GlobalEnergyFixerUpDown``, camulator_huggingface branch).

    Parameters
    ----------
    sp_in, t_in, q_in, u_in, v_in : torch.Tensor
        Input surface pressure (Pa), temperature (K), specific total water
        (kg kg-1), zonal and meridional wind (m s-1). ``sp`` has shape
        (batch, lat, lon); the others (batch, level, lat, lon).
    sp_pred, t_pred, q_pred, u_pred, v_pred : torch.Tensor
        Predicted counterparts.
    toa_down_sw : torch.Tensor
        TOA downwelling shortwave over the step (J m-2), shape (batch, lat, lon).
    toa_up_sw, toa_up_lw : torch.Tensor
        TOA upwelling shortwave and longwave over the step (J m-2).
    surf_down_sw, surf_up_sw, surf_down_lw, surf_up_lw : torch.Tensor
        Surface radiative fluxes over the step (J m-2).
    surf_sh, surf_lh : torch.Tensor
        Surface sensible and latent heat flux over the step (J m-2, positive
        into the surface).
    phis : torch.Tensor
        Surface geopotential (m^2 s-2), shape (lat, lon).
    hyai, hybi : torch.Tensor
        Interface hybrid coefficients, shape (level + 1,).
    area : torch.Tensor
        Cell area (m^2), shape (lat, lon).
    n_seconds : float
        Length of the model step in seconds.

    Returns
    -------
    torch.Tensor
        Corrected temperature (K), shape (batch, level, lat, lon).
    """
    cp_t0 = (1 - q_in) * CP_DRY + q_in * CP_VAPOR
    cp_t1 = (1 - q_pred) * CP_DRY + q_pred * CP_VAPOR

    ken_t0 = 0.5 * (u_in**2 + v_in**2)
    ken_t1 = 0.5 * (u_pred**2 + v_pred**2)

    e_qgk_t0 = LH_WATER * q_in + phis + ken_t0
    e_qgk_t1 = LH_WATER * q_pred + phis + ken_t1

    r_t = (toa_down_sw - toa_up_sw - toa_up_lw) / n_seconds
    r_t_sum = weighted_sum(r_t, area)

    f_s = (
        surf_down_sw - surf_up_sw + surf_down_lw - surf_up_lw + surf_sh + surf_lh
    ) / n_seconds
    f_s_sum = weighted_sum(f_s, area)

    e_level_t0 = cp_t0 * t_in + e_qgk_t0
    e_level_t1 = cp_t1 * t_pred + e_qgk_t1

    te_t0 = column_integral(e_level_t0, sp_in, hyai, hybi) / GRAVITY
    te_t1 = column_integral(e_level_t1, sp_pred, hyai, hybi) / GRAVITY

    global_te_t0 = weighted_sum(te_t0, area)
    global_te_t1 = weighted_sum(te_t1, area)

    ratio = (global_te_t0 + n_seconds * (r_t_sum - f_s_sum)) / global_te_t1
    e_t1_correct = e_level_t1 * ratio.view(-1, 1, 1, 1)
    return (e_t1_correct - e_qgk_t1) / cp_t1


def _gaussian_1d(
    sigma: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    ks = int(2 * sigma * 3 + 1)
    if ks % 2 == 0:
        ks += 1
    xx = torch.arange(ks, dtype=dtype, device=device) - ks // 2
    g = torch.exp(-0.5 * (xx / sigma) ** 2)
    return g / g.sum()


def wind_artifact_filter(
    y: torch.Tensor,
    levels: int,
    var_offsets: Sequence[int],
    mask_var_offsets: tuple[int, int],
    mask_level: int = 14,
    target_levels: Sequence[int] = tuple(range(9, 21)),
    speed_threshold: float = 2.8,
    smooth_sigma_zonal: float = 2.0,
    smooth_sigma_meridional: float = 0.5,
    dilation_zonal: int = 15,
    dilation_meridional: int = 5,
    falloff_sigma: float = 4.0,
    preserve_amplitude: bool = True,
) -> torch.Tensor:
    """Suppress the grid-scale jet artifact: build a mask where the wind speed at
    ``mask_level`` exceeds ``speed_threshold``, dilate it anisotropically, blend a
    Gaussian-smoothed field into the masked region for the target variables and
    levels (CREDIT ``WindPP.py``, camulator_huggingface branch). Operates on
    whatever units ``y`` is in (CAMulator applies it to normalized values).

    Parameters
    ----------
    y : torch.Tensor
        Prediction, shape (batch, channel, lat, lon). Modified in place.
    levels : int
        Number of vertical levels per 3D variable.
    var_offsets : Sequence[int]
        First channel index of each 3D variable to filter.
    mask_var_offsets : tuple[int, int]
        First channel index of the zonal and meridional wind variables.
    mask_level : int
        Level used to compute the wind-speed mask.
    target_levels : Sequence[int]
        Levels to filter.
    speed_threshold : float
        Wind-speed threshold for the mask (same units as ``y``).
    smooth_sigma_zonal : float
        Gaussian smoothing sigma along longitude (grid points).
    smooth_sigma_meridional : float
        Gaussian smoothing sigma along latitude (grid points).
    dilation_zonal : int
        Mask dilation width along longitude (grid points).
    dilation_meridional : int
        Mask dilation width along latitude (grid points).
    falloff_sigma : float
        Sigma of the mask falloff (latitude); longitude uses twice this value.
    preserve_amplitude : bool
        Rescale the smoothed field to preserve the mask-weighted RMS amplitude.

    Returns
    -------
    torch.Tensor
        The filtered tensor ``y``.
    """
    dtype, device = y.dtype, y.device
    u = y[:, mask_var_offsets[0] + mask_level].unsqueeze(1)
    v = y[:, mask_var_offsets[1] + mask_level].unsqueeze(1)

    speed = torch.sqrt(u**2 + v**2)
    mask = (speed > speed_threshold).to(dtype)

    dil_kernel = torch.ones(
        1, 1, dilation_meridional, dilation_zonal, device=device, dtype=dtype
    )
    mask = F.conv2d(
        mask, dil_kernel, padding=(dilation_meridional // 2, dilation_zonal // 2)
    )
    mask = torch.clamp(mask, 0, 1)

    k_lat = int(2 * falloff_sigma * 2 + 1)
    k_lon = int(2 * falloff_sigma * 4 + 1)
    k_lat += k_lat % 2 == 0
    k_lon += k_lon % 2 == 0
    x_lat = torch.arange(k_lat, dtype=dtype, device=device) - k_lat // 2
    g_lat = torch.exp(-0.5 * (x_lat / falloff_sigma) ** 2)
    g_lat = g_lat / g_lat.sum()
    x_lon = torch.arange(k_lon, dtype=dtype, device=device) - k_lon // 2
    g_lon = torch.exp(-0.5 * (x_lon / (falloff_sigma * 2)) ** 2)
    g_lon = g_lon / g_lon.sum()
    falloff = (g_lat.unsqueeze(1) * g_lon.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    blend = F.conv2d(mask, falloff, padding=(k_lat // 2, k_lon // 2))

    g_lat = _gaussian_1d(smooth_sigma_meridional, dtype, device)
    g_lon = _gaussian_1d(smooth_sigma_zonal, dtype, device)
    smooth = (g_lat.unsqueeze(1) * g_lon.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    pad = (g_lat.shape[0] // 2, g_lon.shape[0] // 2)

    for offset in var_offsets:
        for level in target_levels:
            field = y[:, offset + level].unsqueeze(1)
            field_smooth = F.conv2d(field, smooth, padding=pad)
            if preserve_amplitude:
                num = (blend * field**2).sum(dim=(-2, -1), keepdim=True)
                den = (blend * field_smooth**2).sum(dim=(-2, -1), keepdim=True)
                alpha = torch.clamp(torch.sqrt(num / (den + 1e-12)), max=4.0)
                field_smooth = alpha * field_smooth
            y[:, offset + level] = (blend * field_smooth + (1 - blend) * field).squeeze(
                1
            )
    return y
