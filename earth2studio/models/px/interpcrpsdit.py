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

import math
import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime
from typing import Literal, cast, get_args

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_dim
from earth2studio.utils.coords import CoordSystem, map_coords
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from physicsnemo import Module as PhysicsNemoModule
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("interp-crps-dit")
    PhysicsNemoModule = None
    cos_zenith_angle = None

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]
# Physical surface invariants appended to the static stack after the base statics
PHYS_EXTRA = ["log_z0", "lai", "rs_min", "theta_sat", "field_capacity", "wilting_point"]
# Length scales (grid cells) of the multi-scale latent noise, one channel each
NOISE_SCALES = (2, 4, 8, 16, 32, 64)
# Coarse gap range (hours) seen in training; other gaps warn but still run
_GAP_MIN_H, _GAP_MAX_H = 3, 10
# Channels the model was trained to tolerate missing
OptionalVariable = Literal["sp", "u100m", "v100m", "tcwv"]
OPTIONAL_VARIABLES = list(get_args(OptionalVariable))
# Training grid height, 721-lat sources are cropped to this (drops the -90 row)
_TRAIN_LAT = 720


def _gaussian_radius(sigma: float) -> int:
    # Kernel half-width: 3-sigma truncation, at least 1
    return max(1, int(round(3.0 * sigma)))


def _gaussian_blur_valid(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # Valid (no-pad) separable Gaussian blur, x must be oversized by the kernel radius per side.
    # Dividing by sum(k**2) restores unit variance of blurred white noise.
    r = _gaussian_radius(sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    kh = k.view(1, 1, -1, 1).expand(c, 1, 2 * r + 1, 1)
    kw = k.view(1, 1, 1, -1).expand(c, 1, 1, 2 * r + 1)
    x = F.conv2d(x, kh, groups=c)
    x = F.conv2d(x, kw, groups=c)
    return x / (k * k).sum()


def _znorm(a: np.ndarray) -> np.ndarray:
    # Global z-score of an invariant field
    a = a.astype("float32")
    return ((a - a.mean()) / (a.std() + 1e-6)).astype("float32")


@check_optional_dependencies()
class InterpCRPSDiT(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-shot, endpoint-pinned CRPS temporal interpolation with a DiT backbone (73 variables at
    0.25 degree resolution, 720x1440).

    Upsamples the coarse trajectory of a base prognostic model to ``num_interp_steps`` sub-steps per
    coarse interval. For interior fraction ``tau`` and per-member latent ``z`` each frame is produced
    in a single forward pass as ``out(tau) = (1 - tau) x0 + tau xT + sin(pi tau) f(x0, xT, cond, z)``,
    a linear blend plus a DiT correction ``f`` that vanishes at both endpoints. Endpoints are copied
    from the base model, only interior frames are modelled. Ensemble spread comes entirely from the
    per-member latent ``z`` (trained with a CRPS objective).

    The coarse gap is taken from ``px_model`` (trained on 3-10 h) and the output resolution is
    ``gap / num_interp_steps``, e.g. a 6 h gap with ``num_interp_steps=36`` gives 10 min.
    ``num_interp_steps`` must divide the gap in minutes. A regional sub-domain is available via
    :meth:`set_domain`.

    Note
    ----
    Trained and validated only on ERA5. The interior is a learned statistical interpolation with no
    physical conservation guarantee, so quality may degrade with other base data (e.g. GFS). A gap
    outside 3-10 h only warns. The DiT runs in fp32 by default, see ``amp_dtype``.

    Parameters
    ----------
    dit : torch.nn.Module
        DiT network predicting the correction ``f`` (a physicsnemo Module)
    center, scale : torch.Tensor
        Per-variable normalization, shape (1, 73, 1, 1)
    static_inv : torch.Tensor
        Time-invariant conditioning maps (12, H, W): sin/cos lat, sin/cos lon, lsm, orog and 6
        physical fields
    lat2d, lon2d : torch.Tensor
        (H, W) latitude/longitude grids in degrees, defines the output grid
    px_model : PrognosticModel, optional
        Base model producing the coarse trajectory, must be set before running
    num_interp_steps : int, optional
        Sub-steps per coarse interval, must divide the coarse gap in minutes, by default 6
    lon_pad : int, optional
        Circular longitude pad (columns) to reduce the date-line seam, 0 disables, by default 128
    seed : int | None, optional
        If set, noise is drawn from a per-device generator seeded with this value. Results depend on
        ``batch_size`` and the number of init times, by default None (global RNG)
    drop_variables : list[OptionalVariable] | None, optional
        Variables the base model does not supply, a subset of ``sp, u100m, v100m, tcwv``. Dropped
        channels are flagged absent to the DiT and omitted from the output, by default None
    amp_dtype : torch.dtype | None, optional
        ``None`` runs the DiT in fp32, ``torch.bfloat16`` / ``torch.float16`` run it under autocast.
        Endpoint pinning stays in fp32, by default None

    Badges
    ------
    region:global class:mrf product:wind product:temp product:atmos year:2026 gpu:80gb
    """

    def __init__(
        self,
        dit: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        static_inv: torch.Tensor,
        lat2d: torch.Tensor,
        lon2d: torch.Tensor,
        px_model: PrognosticModel | None = None,
        num_interp_steps: int = 6,
        lon_pad: int = 128,
        seed: int | None = None,
        drop_variables: list[OptionalVariable] | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_interp_steps < 1:
            raise ValueError(f"num_interp_steps must be >= 1, got {num_interp_steps}")
        if amp_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValueError(
                f"amp_dtype must be None (fp32), torch.bfloat16, or torch.float16, got {amp_dtype}"
            )
        self.dit = dit
        self.px_model = px_model
        self.num_interp_steps = num_interp_steps
        self.lon_pad = lon_pad
        self.amp_dtype = amp_dtype
        self.seed = seed
        self._gen: torch.Generator | None = None  # lazily created when seed is set
        self.noise_scales = NOISE_SCALES
        self.variables = np.array(VARIABLES)
        drop = drop_variables or []
        bad = [v for v in drop if v not in OPTIONAL_VARIABLES]
        if bad:
            raise ValueError(
                f"drop_variables must be a subset of the trained-optional channels {OPTIONAL_VARIABLES}; got {bad}"
            )
        self.drop_idx = np.array([VARIABLES.index(v) for v in drop], dtype=int)
        _dropped = set(self.drop_idx.tolist())
        self.present_idx = np.array(
            [i for i in range(len(VARIABLES)) if i not in _dropped], dtype=int
        )
        self.present_variables = np.array([VARIABLES[i] for i in self.present_idx])
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("static_inv", static_inv)
        self.register_buffer("lat2d", lat2d)
        self.register_buffer("lon2d", lon2d)
        # Cached CPU grids for cos-zenith and output_coords
        self._lat_np = lat2d.detach().cpu().numpy()
        self._lon_np = lon2d.detach().cpu().numpy()
        self._lat1d = self._lat_np[:, 0]
        self._lon1d = self._lon_np[0, :]
        # Sub-domain halo (top, bottom, left, right) trimmed from emitted frames, and output grid
        self._halo = (0, 0, 0, 0)
        self._out_lat1d = self._lat1d
        self._out_lon1d = self._lon1d
        # Minimum run-grid size per side, derived from the DiT architecture in load_model
        self._min_domain_cells = 64

    def __str__(self) -> str:
        return "InterpCRPSDiT"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system, taken from the wrapped ``px_model``.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary (``batch, time, lead_time, variable, lat, lon``).
        """
        if self.px_model is None:
            raise ValueError("Base model, px_model, must be set")
        input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": np.empty(0),
                "lat": np.empty(0),
                "lon": np.empty(0),
            }
        )
        for key, value in self.px_model.input_coords().items():
            if key in input_coords:
                input_coords[key] = value
        return input_coords

    def _gap_minutes(self) -> int:
        # Coarse gap in minutes from the px_model step
        if self.px_model is None:
            raise ValueError("px_model must be set before computing the coarse gap")
        ic = self.px_model.input_coords()
        oc = self.px_model.output_coords(ic)
        gap_m = float(
            (oc["lead_time"][-1] - ic["lead_time"][-1]) / np.timedelta64(1, "m")
        )
        gap_min = round(gap_m)
        if gap_min <= 0 or abs(gap_m - gap_min) > 1e-6:
            raise ValueError(
                f"px_model coarse step must be a positive whole number of minutes, got {gap_m} min"
            )
        return gap_min

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system at the sub-step cadence (coarse gap / ``num_interp_steps``),
        one sub-step past the input lead time.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.

        Raises
        ------
        ValueError
            If the coarse gap (in minutes) is not divisible by ``num_interp_steps``.

        Warns
        -----
        UserWarning
            If the coarse gap is outside the trained 3-10 h range.
        """
        gap_min = self._gap_minutes()
        gap_h = gap_min / 60.0
        if not (_GAP_MIN_H <= gap_h <= _GAP_MAX_H):
            warnings.warn(
                f"coarse gap {gap_h:g} h is outside the trained "
                f"[{_GAP_MIN_H}, {_GAP_MAX_H}] h range; the gap conditioning channel is "
                "extrapolated, so results are off-distribution and unvalidated.",
                stacklevel=2,
            )
        if gap_min % self.num_interp_steps != 0:
            raise ValueError(
                f"num_interp_steps={self.num_interp_steps} must divide the {gap_min}-min coarse gap"
            )
        step_min = gap_min // self.num_interp_steps
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(step_min, "m")]),
                "variable": np.array(self.present_variables),
                "lat": np.asarray(self._out_lat1d, dtype=np.float64),
                "lon": np.asarray(self._out_lon1d, dtype=np.float64),
            }
        )
        # Grid values and variables are reconciled by map_coords in _default_generator
        handshake_dim(input_coords, "variable", -3)
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "lon", -1)
        output_coords["batch"] = input_coords.get("batch", np.empty(0))
        output_coords["time"] = input_coords.get("time", np.empty(0))
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"][-1]
        )
        return output_coords

    def run_coords(self) -> CoordSystem:
        """Lat/lon grid a wrapped ``px_model`` must cover: the full model grid or, after
        :meth:`set_domain`, the sub-domain run grid (bounding box plus halo, aligned to the patch
        size). A larger grid containing it is accepted and cropped.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary with the run-grid ``lat``/``lon``.
        """
        return OrderedDict(
            {
                "lat": np.asarray(self._lat1d, dtype=np.float64),
                "lon": np.asarray(self._lon1d, dtype=np.float64),
            }
        )

    @staticmethod
    def _snap_to_patch(lo: int, hi: int, n: int, p: int) -> tuple[int, int]:
        # Snap window [lo, hi) outward to multiples of the patch size p, flooring at the grid edge n.
        # The floor can shrink the window below the request, set_domain detects and raises.
        lo = (lo // p) * p
        hi = ((hi + p - 1) // p) * p
        if hi > n:
            hi = (n // p) * p
        return lo, hi

    def set_domain(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        halo: int = 0,
        min_cells: int | None = None,
    ) -> "InterpCRPSDiT":
        """Restrict the model to a regional sub-domain for a lat/lon bounding box.

        Returns a new :class:`InterpCRPSDiT` sharing the loaded ``dit`` with the static maps and
        output grid sliced to the smallest block covering the box. The wrapped ``px_model`` is
        carried over and its frames are cropped to the run grid with ``map_coords``, so it may supply
        the run grid directly (see :meth:`run_coords`) or a larger grid containing it. Longitude wrap
        is disabled (``lon_pad=0``).

        Warning
        -------
        The ``dit`` is shared and mutated in place on each forward, so a parent and its sub-domains
        can run sequentially but not concurrently. The sub-domain is placed on the parent's current
        device and does not follow later ``.to(...)`` calls on the parent. Skill outside the global
        ERA5 training distribution is unvalidated.

        Parameters
        ----------
        lat_min, lat_max, lon_min, lon_max : float
            Bounding box in degrees, inside the current grid. Does not wrap the date line.
        halo : int, optional
            Extra cells added per side to run on, then trimmed from the output to push boundary
            artifacts off the returned box (not validated), by default 0
        min_cells : int | None, optional
            Minimum run-grid size per side so the NATTEN kernel fits the latent grid, by default
            derived from the architecture at load

        Returns
        -------
        InterpCRPSDiT
            Sub-domain model with its own ``run_coords`` and ``output_coords`` (bounding box)

        Raises
        ------
        ValueError
            If the bounding box is degenerate, outside the grid or selects no cells, ``halo`` is
            negative, ``min_cells`` is below 1, patch snapping shrinks the window below the box, or
            the run window is below ``min_cells``.
        """
        lat1d, lon1d = np.asarray(self._lat1d), np.asarray(self._lon1d)
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError(
                "require lat_min < lat_max and lon_min < lon_max (no date-line wrap)"
            )
        if halo < 0:
            raise ValueError(f"halo must be non-negative, got {halo}")
        if min_cells is not None and min_cells < 1:
            raise ValueError(f"min_cells must be >= 1, got {min_cells}")
        if (
            lat_min < lat1d.min()
            or lat_max > lat1d.max()
            or lon_min < lon1d.min()
            or lon_max > lon1d.max()
        ):
            raise ValueError(
                f"bounding box lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] is outside the current grid "
                f"lat[{lat1d.min()},{lat1d.max()}] lon[{lon1d.min()},{lon1d.max()}]"
            )
        inside_lat = (lat1d >= lat_min) & (lat1d <= lat_max)
        inside_lon = (lon1d >= lon_min) & (lon1d <= lon_max)
        if not inside_lat.any() or not inside_lon.any():
            raise ValueError(
                "bounding box selects no grid cells (smaller than one grid cell?); widen it"
            )
        rows, cols = np.where(inside_lat)[0], np.where(inside_lon)[0]
        i0, i1 = int(rows[0]), int(rows[-1]) + 1
        j0, j1 = int(cols[0]), int(cols[-1]) + 1
        hn, wn = lat1d.shape[0], lon1d.shape[0]
        ph, pw = self.dit.patch_size
        # Expand by halo, clamp to the grid, then snap to the patch size
        i0e, i1e = max(0, i0 - halo), min(hn, i1 + halo)
        j0e, j1e = max(0, j0 - halo), min(wn, j1 + halo)
        if halo and (
            i0 - i0e < halo or i1e - i1 < halo or j0 - j0e < halo or j1e - j1 < halo
        ):
            warnings.warn(
                "halo clamped at the grid edge; a residual edge artifact may remain there",
                stacklevel=2,
            )
        i0e, i1e = self._snap_to_patch(i0e, i1e, hn, ph)
        j0e, j1e = self._snap_to_patch(j0e, j1e, wn, pw)
        halo_trim = (i0 - i0e, i1e - i1, j0 - j0e, j1e - j1)
        if any(t < 0 for t in halo_trim):
            raise ValueError(
                f"patch_size={ph, pw} snapping shrank the run window below the bounding box "
                "(grid dim not a multiple of patch_size); widen the bounding box"
            )
        min_cells = self._min_domain_cells if min_cells is None else min_cells
        if min(i1e - i0e, j1e - j0e) < min_cells:
            raise ValueError(
                f"run grid {i1e - i0e}x{j1e - j0e} is below the {min_cells}-cell per-side minimum "
                "(the DiT's NATTEN kernel must fit the latent grid); widen the bounding box or lower min_cells"
            )
        sub = InterpCRPSDiT(
            dit=self.dit,  # shared, sequential use only
            center=self.center,
            scale=self.scale,
            static_inv=self.static_inv[:, i0e:i1e, j0e:j1e].clone(),
            lat2d=self.lat2d[i0e:i1e, j0e:j1e].clone(),
            lon2d=self.lon2d[i0e:i1e, j0e:j1e].clone(),
            px_model=self.px_model,
            num_interp_steps=self.num_interp_steps,
            lon_pad=0,
            seed=self.seed,
            drop_variables=cast(
                "list[OptionalVariable]", [VARIABLES[int(i)] for i in self.drop_idx]
            ),
            amp_dtype=self.amp_dtype,
        )
        sub._halo = halo_trim
        sub._out_lat1d = lat1d[i0:i1]  # bounding box without halo
        sub._out_lon1d = lon1d[j0:j1]
        sub._min_domain_cells = self._min_domain_cells
        return sub.to(self.center.device)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default model package.

        Returns
        -------
        Package
            Default model package
        """
        return Package(
            "hf://nvidia/interp-dit-crps-era5@6f0cac913635773dad78684199a977933a4fb401",
            cache_options={
                "cache_storage": Package.default_cache("interp_crps_dit"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        px_model: PrognosticModel | None = None,
        num_interp_steps: int = 6,
        drop_variables: list[OptionalVariable] | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> PrognosticModel:
        """Load from a package with the DiT checkpoint, normalization stats and static maps.

        Parameters
        ----------
        package : Package
            Package holding ``CRPSModel.mdlus``, ``set_phys.nc``, ``global_means.npy``,
            ``global_stds.npy``.
        px_model : PrognosticModel, optional
            Base model producing the coarse trajectory (may be set later instead).
        num_interp_steps : int, optional
            Sub-steps per coarse interval, must divide the coarse gap in minutes, by default 6
        drop_variables : list[OptionalVariable] | None, optional
            Variables the base does not supply, see the class docstring, by default None
        amp_dtype : torch.dtype | None, optional
            DiT inference precision, see the class docstring, by default None

        Returns
        -------
        PrognosticModel
            The loaded ``InterpCRPSDiT``.

        Raises
        ------
        ValueError
            If ``set_phys.nc`` latitude is not descending or has non-finite values, or the
            normalization arrays are not ``len(VARIABLES)`` finite channels with positive scales.
        """
        try:
            package.resolve("config.json")  # HF download statistics
        except FileNotFoundError:
            pass

        dit = PhysicsNemoModule.from_checkpoint(package.resolve("CRPSModel.mdlus"))
        dit.eval()

        nvar = len(VARIABLES)
        mean = np.load(package.resolve("global_means.npy")).reshape(-1)
        std = np.load(package.resolve("global_stds.npy")).reshape(-1)
        for name, arr in (("global_means.npy", mean), ("global_stds.npy", std)):
            if arr.shape[0] != nvar:
                raise ValueError(
                    f"{name} must have {nvar} channels, got {arr.shape[0]}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values")
        if not (std > 0).all():
            raise ValueError(
                "global_stds.npy must be strictly positive (nonzero scale)"
            )
        center = torch.as_tensor(mean.reshape(1, nvar, 1, 1), dtype=torch.float32)
        scale = torch.as_tensor(std.reshape(1, nvar, 1, 1), dtype=torch.float32)

        # 12-channel static stack from set_phys.nc, cos_zenith is appended per frame
        with xr.open_dataset(package.resolve("set_phys.nc")) as ds:
            lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            if lat[0] <= lat[-1]:
                raise ValueError(
                    "set_phys.nc latitude must be descending (90 -> -90) to match training"
                )

            def _get(v: str) -> np.ndarray:
                a = ds[v]
                a = a.isel(valid_time=0) if "valid_time" in a.dims else a
                return np.asarray(a.values, dtype="float32")

            h, w = len(lat), len(lon)
            latr, lonr = np.deg2rad(lat), np.deg2rad(lon)
            sin_lat = np.broadcast_to(np.sin(latr)[:, None], (h, w))
            cos_lat = np.broadcast_to(np.cos(latr)[:, None], (h, w))
            sin_lon = np.broadcast_to(np.sin(lonr)[None, :], (h, w))
            cos_lon = np.broadcast_to(np.cos(lonr)[None, :], (h, w))
            chans = [
                sin_lat,
                cos_lat,
                sin_lon,
                cos_lon,
                _znorm(_get("lsm")),
                _znorm(_get("z")),
            ]
            chans += [_znorm(_get(v)) for v in PHYS_EXTRA]
            static = np.stack([np.asarray(c, dtype="float32") for c in chans])
            lat2d: np.ndarray
            lon2d: np.ndarray
            lon2d, lat2d = np.meshgrid(lon, lat)  # (H, W)

        # Crop 721 -> 720 after _znorm, training normalized on the native grid
        static = static[:, :_TRAIN_LAT, :]
        lat2d, lon2d = lat2d[:_TRAIN_LAT, :], lon2d[:_TRAIN_LAT, :]
        if not np.isfinite(static).all():
            raise ValueError("non-finite invariant channel(s) in set_phys.nc")

        model = cls(
            dit,
            center=center,
            scale=scale,
            static_inv=torch.as_tensor(static, dtype=torch.float32),
            lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
            lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
            px_model=px_model,
            num_interp_steps=num_interp_steps,
            drop_variables=drop_variables,
            amp_dtype=amp_dtype,
        )
        # Minimum sub-domain size: largest attention kernel x patch size
        kernels = [
            int(v)
            for m in dit.modules()
            if (k := getattr(m, "attn_kernel", None)) is not None
            for v in (k if isinstance(k, (tuple, list)) else (k,))
        ]
        if kernels:
            model._min_domain_cells = max(kernels) * max(dit.patch_size)
        return model

    def _cpad(self, x: torch.Tensor) -> torch.Tensor:
        # Circular longitude pad by lon_pad columns, modulo indexing handles pad > width
        p = self.lon_pad
        if p == 0:
            return x
        idx = torch.arange(-p, x.shape[-1] + p, device=x.device) % x.shape[-1]
        return x[..., idx]

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        # Trim the sub-domain halo off a run-grid frame
        top, bot, left, right = self._halo
        if not (top or bot or left or right):
            return x
        h, w = x.shape[-2], x.shape[-1]
        return x[..., top : h - bot, left : w - right]

    def _dit_forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Rebind the RoPE/NATTEN latent grid for this (H, W), the only per-resolution state. Mutates
        # self.dit in place so instances sharing one DiT must run sequentially.
        ph, pw = self.dit.patch_size
        if x.shape[-2] % ph != 0 or x.shape[-1] % pw != 0:
            raise ValueError(
                f"grid {tuple(x.shape[-2:])} not divisible by patch_size {(ph, pw)}"
            )
        lh, lw = x.shape[-2] // ph, x.shape[-1] // pw
        det = getattr(self.dit.detokenizer, "proj", self.dit.detokenizer)
        det.h_patches, det.w_patches = lh, lw
        # fp32 by default (disables ambient autocast), otherwise autocast at amp_dtype
        ctx = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if self.amp_dtype is None
            else torch.autocast(device_type=x.device.type, dtype=self.amp_dtype)
        )
        with ctx:
            return self.dit(
                x.float(),
                tau.float(),
                condition=None,
                attn_kwargs={"latent_hw": (lh, lw)},
            )

    def _draw_noise(
        self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Multi-scale latent (b, n_noise, H, W): white noise per scale, oversized by the kernel radius
        # then valid-blurred so variance is stationary. Draws come from the seeded per-device
        # generator if set, else the global RNG. Periodic in longitude when lon_pad > 0.
        periodic = self.lon_pad > 0
        gen = None
        if self.seed is not None:
            if self._gen is None or self._gen.device != torch.device(device):
                self._gen = torch.Generator(device=device)
                self._gen.manual_seed(self.seed)
            gen = self._gen
        chans = []
        for scale in self.noise_scales:
            sigma = scale / 2.0
            r = _gaussian_radius(sigma)
            if periodic:
                noise = torch.randn(
                    b,
                    1,
                    h + 2 * r,
                    w,
                    device=device,
                    dtype=torch.float32,
                    generator=gen,
                )
                idx = torch.arange(-r, w + r, device=device) % w
                noise = noise[..., idx]
            else:
                noise = torch.randn(
                    b,
                    1,
                    h + 2 * r,
                    w + 2 * r,
                    device=device,
                    dtype=torch.float32,
                    generator=gen,
                )
            chans.append(_gaussian_blur_valid(noise, sigma))
        return torch.cat(chans, dim=1).to(dtype)

    def _build_cond(
        self,
        when: datetime,
        gap_val: float,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # DiT conditioning for one frame, (batch, 87, H, W): 13 invariants (static maps + live
        # cos_zenith) + 1 gap channel + 73 presence mask (0 for dropped variables)
        h, w = self.static_inv.shape[-2], self.static_inv.shape[-1]
        cz = cos_zenith_angle(when, self._lon_np, self._lat_np)
        cz = torch.as_tensor(
            np.asarray(cz, dtype="float32"), device=device, dtype=dtype
        )[None, None]
        inv = torch.cat(
            [
                self.static_inv.to(dtype)[None].expand(batch, -1, -1, -1),
                cz.expand(batch, -1, -1, -1),
            ],
            dim=1,
        )
        gap = torch.full((batch, 1, h, w), float(gap_val), device=device, dtype=dtype)
        mask = torch.ones(batch, len(self.variables), h, w, device=device, dtype=dtype)
        if len(self.drop_idx):
            mask[:, self.drop_idx] = 0.0
        return torch.cat([inv, gap, mask], dim=1)

    @torch.inference_mode()
    def _sample_tau(
        self,
        x0n: torch.Tensor,
        xTn: torch.Tensor,
        cond: torch.Tensor,
        tau: float,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # One normalized (B,73,H,W) interpolation at fraction tau with latent z (B,n_noise,H,W)
        p = self.lon_pad
        x0p, xTp = self._cpad(x0n), self._cpad(xTn)
        inp = torch.cat([x0p, xTp, self._cpad(cond), self._cpad(z)], dim=1)
        taut = torch.full(
            (x0n.shape[0],), float(tau), device=x0n.device, dtype=x0n.dtype
        )
        f = self._dit_forward(inp, taut)
        t = taut.reshape(-1, 1, 1, 1)
        out = (1.0 - t) * x0p + t * xTp + torch.sin(math.pi * t) * f
        return out[..., p : p + x0n.shape[-1]] if p > 0 else out

    def _pad_present_to_full(self, x_present: torch.Tensor) -> torch.Tensor:
        # Expand present variables to the full set, dropped channels are filled with their center
        if not len(self.drop_idx):
            return x_present
        shape = list(x_present.shape)
        shape[-3] = len(self.variables)
        full = x_present.new_empty(shape)
        full[..., self.present_idx, :, :] = x_present
        full[..., self.drop_idx, :, :] = (
            self.center[0, self.drop_idx]
            .reshape(len(self.drop_idx), 1, 1)
            .to(x_present.dtype)
        )
        return full

    @torch.inference_mode()
    def _interpolate(
        self, x0: torch.Tensor, x1: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        # Yield interior sub-steps between coarse states x0 and x1 (physical units)
        x0n = (x0 - self.center) / self.scale
        x1n = (x1 - self.center) / self.scale
        gap_min = self._gap_minutes()
        gap_val = (gap_min / 60.0 - 6.0) / 3.0
        step_min = gap_min // self.num_interp_steps
        base_lead = coords["lead_time"][-1]
        nt = x0n.shape[1]  # number of init times
        b, hh, ww = len(coords["batch"]), x0n.shape[-2], x0n.shape[-1]
        # One latent per (member, init time) reused across sub-steps for temporal coherence
        z = self._draw_noise(b * nt, hh, ww, x0n.device, x0n.dtype).reshape(
            b, nt, -1, hh, ww
        )
        for interp_step in range(1, self.num_interp_steps):
            tau = interp_step / self.num_interp_steps
            sub = OrderedDict(coords)
            sub["lead_time"] = np.array(
                [base_lead + np.timedelta64(step_min * interp_step, "m")]
            )
            sub["variable"] = np.array(self.present_variables)
            sub["lat"] = np.asarray(self._out_lat1d, dtype=np.float64)
            sub["lon"] = np.asarray(self._out_lon1d, dtype=np.float64)
            out = torch.zeros_like(x0n)
            for ti, t in enumerate(sub["time"]):
                for lti, lt in enumerate(sub["lead_time"]):
                    when = datetime.fromisoformat(str(t + lt)[:19])
                    cond = self._build_cond(when, gap_val, b, x0n.device, x0n.dtype)
                    out[:, ti, lti] = self._sample_tau(
                        x0n[:, ti, lti], x1n[:, ti, lti], cond, tau, z[:, ti]
                    )
            emitted = self._trim(out * self.scale + self.center)
            if len(self.drop_idx):
                emitted = emitted[..., self.present_idx, :, :]
            yield (emitted, sub)

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run one step, returning the initial condition.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Initial-condition frame and its coordinate system
        """
        return next(self._default_generator(x, coords))

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        if self.px_model is None:
            raise ValueError(
                "Base model, px_model, must be set before executing the model."
            )
        x0 = None
        coords0: CoordSystem | None = None
        # Base frames are mapped onto the run grid with map_coords (e.g. 721 -> 720 lat crop or global
        # -> sub-domain box)
        run_target = OrderedDict(
            {
                "variable": np.array(self.present_variables),
                "lat": np.asarray(self._lat1d, dtype=np.float64),
                "lon": np.asarray(self._lon1d, dtype=np.float64),
            }
        )
        for fc_step, (x, coords) in enumerate(self.px_model.create_iterator(x, coords)):
            x, coords = self.front_hook(x, coords)
            if fc_step == 0 and not (
                np.all(np.isin(self._lat1d, coords["lat"]))
                and np.all(np.isin(self._lon1d, coords["lon"]))
            ):
                warnings.warn(
                    "px_model grid does not contain the model run grid; the base is being nearest-regridded "
                    "onto it (expected an exact 0.25 deg grid). Interpolation proceeds but may be degraded.",
                    stacklevel=2,
                )
            # Map present variables only, then expand to the full set with dropped channels at center
            x_present, coords_present = map_coords(x, coords, run_target)
            x_run = self._pad_present_to_full(x_present)
            coords_run = OrderedDict(coords_present)
            coords_run["variable"] = np.array(self.variables)
            # Crop to the output grid for the emitted endpoint. output_coords also validates the gap,
            # its lead_time is ignored by map_coords.
            x_emit, emit_coords = map_coords(
                x_run, coords_run, self.output_coords(coords_run)
            )
            # rear_hook only touches emitted frames so the interpolation anchor stays unhooked
            if fc_step == 0:
                x0, coords0 = x_run, coords_run
                yield self.rear_hook(x_emit, emit_coords)
            else:
                for xo, co in self._interpolate(x0, x_run, coords0):
                    yield self.rear_hook(xo, co)
                yield self.rear_hook(x_emit, emit_coords)
                x0, coords0 = x_run, coords_run

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Yield the initial condition, then interpolated sub-steps and coarse endpoints.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Each emitted frame and its coordinate system
        """
        yield from self._default_generator(x, coords)
