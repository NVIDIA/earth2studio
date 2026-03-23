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

# %%
"""
StormCast Ensemble 100 m Wind Interpolation
===========================================

Run a short StormCast ensemble forecast, then interpolate hybrid-level winds
to 100 m AGL using:

  H_agl = Zhl / 9.81

where ``Zhl`` is the hybrid-level geopotential-like field in m^2 s^-2.
"""
# /// script
# dependencies = [
#   "earth2studio[data,stormcast] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "xarray",
# ]
# ///

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

import earth2studio.run as run
from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

os.makedirs("outputs", exist_ok=True)
load_dotenv()

G0 = 9.81
TARGET_AGL_M = 100.0
N_STEPS = 2
ENSEMBLE_SIZE = 4
BATCH_SIZE = 2


def _parse_level(name: str, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)hl", name)
    if match is None:
        return None
    return int(match.group(1))


def _collect_hl_names(ds: xr.Dataset, prefix: str) -> tuple[list[str], list[int]]:
    pairs: list[tuple[int, str]] = []
    for name in ds.data_vars:
        level = _parse_level(str(name), prefix)
        if level is not None:
            pairs.append((level, str(name)))
    pairs.sort(key=lambda p: p[0])
    levels = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    return names, levels


def _interp_uv_to_height(
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    target: float,
    lev_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate u/v from hybrid levels to target height.

    Inputs are shaped [..., nlev, hrrr_y, hrrr_x]. This supports deterministic
    and ensemble outputs.
    """
    if not (u.shape == v.shape == h.shape):
        raise ValueError(f"u/v/h shapes must match; got {u.shape}, {v.shape}, {h.shape}")

    nlev = h.shape[lev_axis]
    if nlev < 2:
        raise ValueError("Need at least two hybrid levels for interpolation.")

    # Bracketing level index per pixel/member/time: k <= target < k+1.
    k = np.sum(h <= target, axis=lev_axis) - 1
    k = np.clip(k, 0, nlev - 2).astype(np.int64)
    k_exp = np.expand_dims(k, axis=lev_axis)

    h0 = np.take_along_axis(h, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    h1 = np.take_along_axis(h, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)
    u0 = np.take_along_axis(u, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    u1 = np.take_along_axis(u, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)
    v0 = np.take_along_axis(v, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    v1 = np.take_along_axis(v, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)

    den = np.where(np.abs(h1 - h0) > 1e-12, h1 - h0, 1e-12)
    w = np.clip((target - h0) / den, 0.0, 1.0)

    u_t = u0 + w * (u1 - u0)
    v_t = v0 + w * (v1 - v0)
    return u_t, v_t


# %%
# Run a short StormCast ensemble forecast and save to Zarr.
package = StormCast.load_default_package()
model = StormCast.load_model(package)
data = HRRR()
perturb = Zero()
io = ZarrBackend(file_name="outputs/stormcast_raw_ensemble.zarr", backend_kwargs={"overwrite": True})

today = datetime.today() - timedelta(days=1)
date = today.isoformat().split("T")[0]
io = run.ensemble(
    [date],
    N_STEPS,
    ENSEMBLE_SIZE,
    model,
    data,
    io,
    perturb,
    batch_size=BATCH_SIZE,
)

print("Raw StormCast output saved:", "outputs/stormcast_raw_ensemble.zarr")
print(f"Ensemble settings: ensemble_size={ENSEMBLE_SIZE}, batch_size={BATCH_SIZE}")

# %%
# Load output and interpolate hybrid-level winds to 100 m AGL.
ds = xr.open_zarr("outputs/stormcast_raw_ensemble.zarr")

z_names, z_levels = _collect_hl_names(ds, "Z")
u_names, u_levels = _collect_hl_names(ds, "u")
v_names, v_levels = _collect_hl_names(ds, "v")

if not z_names or not u_names or not v_names:
    raise RuntimeError("Could not find Z*hl, u*hl, v*hl variables in StormCast output.")
if not (z_levels == u_levels == v_levels):
    raise RuntimeError(
        f"Level mismatch across Z/u/v hybrid fields: {z_levels}, {u_levels}, {v_levels}"
    )

z_da = xr.concat([ds[name] for name in z_names], dim=xr.DataArray(z_levels, dims="level")) / G0
u_da = xr.concat([ds[name] for name in u_names], dim=xr.DataArray(u_levels, dims="level"))
v_da = xr.concat([ds[name] for name in v_names], dim=xr.DataArray(v_levels, dims="level"))

spatial_dims = ("hrrr_y", "hrrr_x")
for dim in spatial_dims:
    if dim not in z_da.dims:
        raise RuntimeError(f"Expected '{dim}' in StormCast output dims={z_da.dims}")

lead_dims = [d for d in z_da.dims if d not in ("level", *spatial_dims)]
ordered_dims = tuple(lead_dims + ["level", *spatial_dims])
z_da = z_da.transpose(*ordered_dims)
u_da = u_da.transpose(*ordered_dims)
v_da = v_da.transpose(*ordered_dims)

lev_axis = len(lead_dims)
u100, v100 = _interp_uv_to_height(
    u_da.values,
    v_da.values,
    z_da.values,
    TARGET_AGL_M,
    lev_axis=lev_axis,
)
ws100 = np.hypot(u100, v100)

out_dims = tuple(lead_dims + list(spatial_dims))
coords = {d: z_da.coords[d].values for d in out_dims}

out = xr.Dataset(
    data_vars={
        "u100m": xr.DataArray(u100, dims=out_dims, coords=coords),
        "v100m": xr.DataArray(v100, dims=out_dims, coords=coords),
        "ws100m": xr.DataArray(ws100, dims=out_dims, coords=coords),
    },
    attrs={
        "description": "100 m AGL wind interpolated from StormCast hybrid levels",
        "height_assumption": "H_agl = Zhl / 9.81",
        "target_height_m": TARGET_AGL_M,
        "ensemble_size": ENSEMBLE_SIZE,
        "batch_size": BATCH_SIZE,
        "levels_used": ",".join(str(l) for l in z_levels),
    },
)

out.to_zarr("outputs/stormcast_100m_wind_ensemble.zarr", mode="w")
print("100 m wind output saved:", "outputs/stormcast_100m_wind_ensemble.zarr")
print("Interpolated variables:", list(out.data_vars))
