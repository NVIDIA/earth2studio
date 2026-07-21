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
# Physical surface invariants appended to the invariant stack (between the base statics and cos_zenith).
PHYS_EXTRA = ["log_z0", "lai", "rs_min", "theta_sat", "field_capacity", "wilting_point"]
# Spatial length scales (grid cells) of the multi-scale latent noise; each becomes one channel of the
# per-member latent z (drawn as white noise, blurred with sigma = scale/2 in _draw_noise) driving ensemble spread.
NOISE_SCALES = (2, 4, 8, 16, 32, 64)
# Min/max coarse bracket width (h between consecutive source frames) the checkpoint was trained on;
# used only to warn on off-distribution gaps in output_coords (out-of-range still runs). The gap is fed to the DiT as a constant
# conditioning channel, affine-normalized to ~unit scale as (gap_h - 6) / 3 (training-time mean 6, scale 3;
# see gap_val in _interpolate) -- the 6 and 3 are the standardization constants, unrelated to this 3-10 range.
_GAP_MIN_H, _GAP_MAX_H = 3, 10
# The only channels the model was trained to tolerate missing. Dropping any other (core) channel is
# off-distribution and is rejected.
OptionalVariable = Literal["sp", "u100m", "v100m", "tcwv"]
OPTIONAL_VARIABLES = list(get_args(OptionalVariable))
# Training grid height: 721-lat sources are cropped to this (dropping the -90 pole row) to match training.
_TRAIN_LAT = 720


def _gaussian_radius(sigma: float) -> int:
    # Kernel half-width for a Gaussian blur: the standard 3-sigma truncation, floored at 1 so a usable
    # (>=3-tap) kernel is returned even for small sigma. Sets the oversize pad _gaussian_blur_valid expects.
    return max(1, int(round(3.0 * sigma)))


def _gaussian_blur_valid(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # Valid (no-pad) separable Gaussian blur; ``x`` must be oversized by the kernel radius on each side.
    # Divides by the analytic ``sum(k*k)`` so unit-variance white noise stays unit-variance (stationary
    # variance, no reflect-pad edge artifact). Reproduces the noise generator used at training.
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
    # Global z-score (field mean/std, eps 1e-6), matching the training-time invariant normalization.
    a = a.astype("float32")
    return ((a - a.mean()) / (a.std() + 1e-6)).astype("float32")


@check_optional_dependencies()
class InterpCRPSDiT(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-shot, endpoint-pinned CRPS temporal interpolation with a DiT backbone (73 variables at
    0.25 degree resolution, 720x1440).

    The DiT backbone always operates on the full 73-channel state internally; the public output is those
    73 variables minus any ``drop_variables`` (channels the base does not supply), so a base with fewer
    variables can drive it and the output carries only the variables actually available.

    Interpolates a base model's widely-spaced trajectory to ``num_interp_steps`` sub-steps per bracket.
    For each interior fraction ``tau`` and per-member latent ``z`` the field is produced in a single
    forward as ``out(tau) = (1 - tau) x0 + tau xT + sin(pi tau) f(x0, xT, cond, z)`` -- a linear base
    plus a DiT correction ``f`` whose ``sin(pi tau)`` envelope drives it to zero at both endpoints. The
    bracket endpoints are not modelled: they are copied from the base model's coarse frames (after those
    frames are mapped onto the output grid); only the interior ``tau`` are modelled. Ensemble spread comes entirely from the
    per-member latent ``z`` (the model is trained with a CRPS objective, which rewards a spread that
    matches forecast uncertainty).

    A "bracket" is one coarse interval between two consecutive source frames; its width (gap) is taken
    from the wrapped ``px_model`` and may be any of the trained 3-10 h widths (set automatically). The
    output resolution is ``gap / num_interp_steps`` (e.g. a 6 h gap with ``num_interp_steps=36`` -> 10
    min); ``num_interp_steps`` must divide the coarse gap in minutes. A regional sub-domain is available
    via :meth:`set_domain`, which slices the conditioning maps + output grid to a bounding box.

    Warning
    -------
    The model weights are not yet hosted, so :meth:`load_default_package` returns a placeholder URL that
    fails to download; pass your own local ``Package`` to :meth:`load_model` until they are published.
    A base forecast/reanalysis model must be set (``px_model`` / :meth:`load_model`) before running. Its
    coarse step must be divisible by ``num_interp_steps``; a step outside the trained 3-10 h range only
    warns (off-distribution, but still run). The DiT runs in fp32 by default; ``amp_dtype`` runs its
    forward under bf16/fp16 autocast (weights stay fp32; autocast downcasts eligible operations).

    Parameters
    ----------
    dit : torch.nn.Module
        The DiT network that predicts the correction ``f`` (a physicsnemo Module) with ``natten2d_rope`` attention.
    center, scale : torch.Tensor
        Per-variable normalization, shape (1, 73, 1, 1).
    static_inv : torch.Tensor
        Time-invariant conditioning maps (12, H, W): sin/cos lat, sin/cos lon, lsm, orog + 6 physical.
    lat2d, lon2d : torch.Tensor
        (H, W) latitude/longitude grids (degrees); define the output grid + the live cos-zenith channel.
    px_model : PrognosticModel, optional
        Base model producing the coarse trajectory (set before running).
    num_interp_steps : int, optional
        Sub-steps per bracket; must divide the coarse gap (min). Output resolution = gap/num_interp_steps
        (6 h gap: 6 -> 1 h, 36 -> 10 min). Default 6.
    lon_pad : int, optional
        Circular longitude pad (cols) that reduces the date-line seam (the DiT attention has no built-in
        longitude wraparound); 0 disables, by default 128.
    seed : int | None, optional
        If set, ensemble-member noise is drawn from a per-device generator seeded by this value; members
        are distinct and reproducible for a fixed run layout. The per-member draw depends on how the
        ensemble is chunked (``batch_size``) and on the number of init times per call, so seeded results
        are not bit-identical when that layout changes. ``None`` (default) uses the global RNG
        (nondeterministic run-to-run).
    drop_variables : list[OptionalVariable] | None, optional
        Variables the base model does not supply (or that should be ignored). The base need only provide
        the remaining variables; each dropped channel is expanded to its climatological center internally
        (normalized zero, flagged absent to the DiT via the presence mask) and is **not** included in the
        output -- so the model emits ``VARIABLES`` minus these. This lets a base that lacks them drive the
        interpolator (e.g. a 69-variable model missing ``u100m, v100m, sp, tcwv``). Must be a subset of the
        trained-optional channels ``sp, u100m, v100m, tcwv``; dropping any other (core) channel raises.
        Default None.
    amp_dtype : torch.dtype | None, optional
        DiT inference precision. ``None`` (default) runs fp32 (autocast disabled). ``torch.bfloat16`` /
        ``torch.float16`` runs the forward under autocast (weights stay fp32; autocast downcasts eligible
        operations). Only the
        DiT correction ``f`` is computed at reduced precision: the linear base + envelope pin run in fp32,
        so the copied endpoints at ``tau=0/1`` are unaffected.

    Note
    ----
    Each interior frame is produced by a single network forward pass (a full bracket runs one forward
    per interior sub-step); there is no iterative SDE/ODE solver.
    Trained and validated only on ERA5: the bracket endpoints are the verbatim base-model states
    (correct for any base), but the interior is learned (a statistical interpolation, with no
    physical/conservation guarantee), so with a non-ERA5 source (e.g. GFS) the inputs differ from the
    ERA5 training data and interior quality may degrade.

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
        # DiT inference precision (see the amp_dtype docstring); None -> fp32.
        self.amp_dtype = amp_dtype
        self.seed = (
            seed  # None -> global RNG (nondeterministic); int -> reproducible members
        )
        self._gen: torch.Generator | None = (
            None  # lazily created per-device when seed is set
        )
        self.noise_scales = NOISE_SCALES
        self.variables = np.array(VARIABLES)
        # Dropped input variables: zeroed in x0/xT (normalized space) and flagged 0 in the presence mask.
        drop = drop_variables or []
        bad = [v for v in drop if v not in OPTIONAL_VARIABLES]
        if bad:
            raise ValueError(
                f"drop_variables must be a subset of the trained-optional channels {OPTIONAL_VARIABLES}; got {bad}"
            )
        self.drop_idx = np.array([VARIABLES.index(v) for v in drop], dtype=int)
        # Present (emitted) channels: the base need only supply these. Dropped channels are declared
        # unavailable -- expanded to center internally (normalized zero for the DiT) and never emitted.
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
        # cached CPU numpy grids for the (numpy-only) cos-zenith call + output_coords -- buffers never change.
        self._lat_np = lat2d.detach().cpu().numpy()
        self._lon_np = lon2d.detach().cpu().numpy()
        self._lat1d = self._lat_np[
            :, 0
        ]  # regular grid -> 1D lat/lon: the run grid the DiT sees
        self._lon1d = self._lon_np[0, :]
        # Sub-domain halo (top, bottom, left, right) px trimmed from every emitted frame, and the
        # trimmed output grid. Defaults (no halo, out == run) are set by set_domain when it slices.
        self._halo = (0, 0, 0, 0)
        self._out_lat1d = self._lat1d
        self._out_lon1d = self._lon1d
        # Per-side floor so the DiT's NATTEN kernel fits the latent grid. This 64 is only a pre-load
        # default; load_model overwrites it with the architecture's attn_kernel x patch for the real DiT.
        self._min_domain_cells = 64

    def __str__(self) -> str:
        return "InterpCRPSDiT"

    # ------------------------------------------------------------------ coords
    def input_coords(self) -> CoordSystem:
        """Input coordinate system, derived from the wrapped ``px_model`` (its grid). The base frame is
        fed to ``px_model`` on its own grid and only then reconciled onto the model's run grid with
        ``map_coords``, so the input contract is the ``px_model`` grid.

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
        # Coarse bracket width (gap) in minutes, from the wrapped px_model's step.
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
        """Output coordinate system: the model grid at the sub-step cadence (coarse gap /
        num_interp_steps), one sub-step past the input lead time. Raises if the coarse gap is not
        divisible by ``num_interp_steps`` and warns if it is outside the trained 3-10 h range, and
        checks the input exposes variable/lat/lon dims in the expected positions (grid values + variable
        set are reconciled later by ``map_coords``).

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
            If the coarse gap is outside the trained 3-10 h range. The run is still allowed (the gap
            conditioning channel simply extrapolates), but the interior is off-distribution and
            unvalidated.
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
        # Reported (output) grid = the trimmed bounding box interior; may differ from the run grid under a halo.
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
        # Report the model's own output grid regardless of the input grid: the base
        # frame is fed to px_model on its grid, then map_coords-reconciled onto the run grid (721->720, or
        # a global base -> sub-domain bounding box) in _default_generator. We only require the spatial dims to be
        # present in the expected positions; grid values + variable set are reconciled by map_coords
        # (which raises if a required variable is missing).
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
        """The lat/lon grid a wrapped ``px_model`` must cover -- the full model grid, or, after
        :meth:`set_domain`, the sub-domain run grid (bounding box + halo, aligned to the patch size). A
        larger grid that contains it is accepted and cropped.

        Unlike :meth:`input_coords` (which derives lat/lon from ``px_model`` and so is unavailable until
        a base model is attached), this reads the fixed run grid directly, so it is the grid to build a
        regional ``px_model`` against right after ``set_domain``.

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

    # ------------------------------------------------------------------ sub-domain
    @staticmethod
    def _snap_to_patch(lo: int, hi: int, n: int, p: int) -> tuple[int, int]:
        # Snap a run window [lo, hi) to multiples of the DiT patch size ``p`` (the detokenizer needs the
        # pixel grid divisible by patch). Grows the low edge down and the high edge up; if the high edge
        # would exceed ``n`` it floors to the largest in-bounds multiple (so the extent stays divisible).
        # At the grid boundary that floor can drop below the requested bounding box and yield a negative halo
        # trim -- ``set_domain`` detects that case and raises.
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
        """Restrict the (global) model to a regional sub-domain for a lat/lon bounding box.

        Returns a new :class:`InterpCRPSDiT` that shares the loaded ``dit`` but whose invariant stack /
        output grid are sliced to the smallest block of the current grid covering ``[lat_min, lat_max] x
        [lon_min, lon_max]``. No learnable weight depends on the grid size (RoPE/NATTEN state rebinds per
        forward), so the DiT runs on the cropped grid directly, subject to the per-side minimum size and
        patch-divisibility constraints this method enforces. Longitude wrap is
        disabled (``lon_pad=0``) because a regional block is not periodic.

        The wrapped ``px_model`` is carried over. Its coarse trajectory is reconciled onto the run grid
        (the bounding box expanded by ``halo``, aligned to the patch size) with ``map_coords``, so it may supply that grid
        directly (e.g. a regional :class:`~earth2studio.models.px.datareplay.DataReplay`, available as
        :meth:`run_coords`) or a larger grid that contains it (e.g. a global base is cropped to the bounding box).

        ``halo`` (px, default 0 = off): run on a block expanded by ``halo`` real cells per side and trim
        it off every emitted frame -- intended to push the DiT's boundary artifact off the returned bounding box
        interior along the (non-periodic) sub-domain edge (a plausible mitigation, not validated on this
        path); clamps + warns at the global grid edge.

        Warning
        -------
        The shared ``dit`` rebinds its latent-grid state in place per forward, so a parent and its
        sub-domains (or several sub-domains) are safe to run sequentially but not concurrently against
        the same ``dit`` -- run concurrent domains in separate processes. Interior skill is unvalidated
        outside the trained global ERA5 distribution. The returned sub-domain is placed on the parent's
        current device; if you later move the parent with ``.to(...)`` it will not propagate to an
        already-returned sub (the sub holds its own cloned invariant/grid buffers) -- move the sub
        explicitly, or call ``set_domain`` after the parent is on its final device.

        Parameters
        ----------
        lat_min, lat_max, lon_min, lon_max : float
            Bounding box (degrees); must lie inside the current grid. Longitude does not wrap the
            date line (require ``lon_min < lon_max`` within the grid's longitude span).
        halo : int, optional
            Real cells added per side then trimmed off the output (boundary-artifact guard), by default 0.
        min_cells : int | None, optional
            Per-side floor on the run grid (NATTEN kernel must fit the latent). Defaults to the model's
            ``_min_domain_cells`` (derived from the architecture at load, ``attn_kernel x patch``; 64 before loading).

        Returns
        -------
        InterpCRPSDiT
            A fixed sub-domain model with its own ``input_coords`` (run grid) / ``output_coords`` (bounding box).

        Raises
        ------
        ValueError
            If the bounding box is degenerate, ``halo`` is negative, ``min_cells`` is less than 1, the
            box lies outside the current grid or selects no cells, patch-size snapping at the grid
            boundary shrinks the run window below the requested bounding box, or the run window is smaller than
            the ``min_cells`` per-side floor (the NATTEN kernel must fit the latent grid).
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
        # Expand by the halo (real surrounding cells), clamp to the grid edge, then patch-snap the window.
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
            dit=self.dit,  # shared by reference -- sequential use only (see Warning)
            center=self.center,
            scale=self.scale,
            static_inv=self.static_inv[:, i0e:i1e, j0e:j1e].clone(),
            lat2d=self.lat2d[i0e:i1e, j0e:j1e].clone(),
            lon2d=self.lon2d[i0e:i1e, j0e:j1e].clone(),
            px_model=self.px_model,
            num_interp_steps=self.num_interp_steps,
            lon_pad=0,  # regional block is not longitude-periodic
            seed=self.seed,
            drop_variables=cast(
                "list[OptionalVariable]", [VARIABLES[int(i)] for i in self.drop_idx]
            ),
            amp_dtype=self.amp_dtype,
        )
        sub._halo = halo_trim
        sub._out_lat1d = lat1d[
            i0:i1
        ]  # bounding box interior (run rows minus halo) -> matches _trim
        sub._out_lon1d = lon1d[j0:j1]
        sub._min_domain_cells = (
            self._min_domain_cells
        )  # inherit the model floor (a per-call min_cells
        # override is one-shot: it gates this crop but is not persisted onto the returned sub)
        # Place the sub on the parent's current device: its own (sliced) buffers are otherwise built on
        # the slice's device and would desync from the shared dit if the parent had been moved with .to().
        return sub.to(self.center.device)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default model package.

        .. warning::
            The model weights are **not yet hosted**: the returned package points at a placeholder
            Hugging Face URL, so downloading it (e.g. via :meth:`load_model`) will fail until the weights
            are published. Until then, pass your own local ``Package`` to :meth:`load_model` instead.

        Returns
        -------
        Package
            The default model package (placeholder URL; see the warning above).
        """
        # TODO: PLACEHOLDER Hugging Face URL -- replace with the real repo once the package is uploaded.
        warnings.warn(
            "InterpCRPSDiT weights are not yet hosted: load_default_package() returns a placeholder "
            "Hugging Face URL that will fail to download. Pass a local Package(...) to load_model() "
            "until the weights are published.",
            stacklevel=2,
        )
        return Package(
            "hf://nvidia/earth2studio-interp-crps-dit",
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
        """Load from a package: the DiT checkpoint, normalization stats, and the invariant stack.

        Parameters
        ----------
        package : Package
            Package holding ``CRPSModel.mdlus``, ``set_phys.nc``, ``global_means.npy``,
            ``global_stds.npy``.
        px_model : PrognosticModel, optional
            Base model producing the coarse trajectory (may be set later instead).
        num_interp_steps : int, optional
            Sub-steps per bracket (must divide the coarse gap in minutes), by default 6.
        drop_variables : list[OptionalVariable] | None, optional
            Optional variables the base does not supply (subset of ``sp, u100m, v100m, tcwv``); each is
            expanded to center internally and omitted from the output. Default None. See the class
            docstring for the full contract.
        amp_dtype : torch.dtype | None, optional
            DiT inference precision (``None`` = fp32; ``torch.bfloat16``/``torch.float16`` = autocast),
            by default None.

        Returns
        -------
        PrognosticModel
            The loaded ``InterpCRPSDiT``.

        Raises
        ------
        ValueError
            If ``set_phys.nc`` latitude is not descending (90 -> -90) or an invariant channel is
            non-finite, or if the normalization arrays are not exactly ``len(VARIABLES)`` finite
            channels with strictly positive scales.
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
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

        # Build the 12-channel static invariant stack from set_phys.nc, reproducing the training-time
        # invariant ordering + normalization exactly. cos_zenith is appended live.
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
            static = np.stack(
                [np.asarray(c, dtype="float32") for c in chans]
            )  # (12, H, W)
            lat2d: np.ndarray
            lon2d: np.ndarray
            lon2d, lat2d = np.meshgrid(lon, lat)  # (H, W)

        # 721 -> 720 crop (drop the -90 row) to match the training grid, if needed. order is
        # order matters here: _znorm above ran on the full nc grid, then we crop -- matching training (which
        # z-scores on the native grid). Do not crop before _znorm (would shift the invariant stats).
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
        # Derive the sub-domain per-side floor from the arch (largest attention kernel x patch, so the
        # neighborhood-attention kernel fits the latent grid). Best-effort: keep the default if the
        # DiT does not expose ``attn_kernel``.
        kernels = [
            int(v)
            for m in dit.modules()
            if (k := getattr(m, "attn_kernel", None)) is not None
            for v in (k if isinstance(k, (tuple, list)) else (k,))
        ]
        if kernels:
            model._min_domain_cells = max(kernels) * max(dit.patch_size)
        return model

    # ------------------------------------------------------------------ internals
    def _cpad(self, x: torch.Tensor) -> torch.Tensor:
        # Circular longitude pad by ``lon_pad`` columns (identity when 0). Genuinely periodic for any pad
        # width via modulo indexing -- a slice-based pad would clamp (and later crop to zero) when lon_pad
        # exceeds the grid width. Output width is ``W + 2 * lon_pad``; matches the slice form for pad <= W.
        p = self.lon_pad
        if p == 0:
            return x
        idx = torch.arange(-p, x.shape[-1] + p, device=x.device) % x.shape[-1]
        return x[..., idx]

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        # Trim the sub-domain halo border off a run-grid frame (last two dims); identity with no halo.
        top, bot, left, right = self._halo
        if not (top or bot or left or right):
            return x
        h, w = x.shape[-2], x.shape[-1]
        return x[..., top : h - bot, left : w - right]

    def _dit_forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Resolution-rebind + precision-controlled DiT forward: bind the latent grid to this (H, W), then
        # run the correction ``f`` under the configured autocast (fp32 unless ``amp_dtype`` is set).
        # Rebind the RoPE/NATTEN latent grid + detokenizer patch counts for this (H, W) -- the only
        # per-resolution state (no learnable weight depends on the grid). Mutates self.dit in place:
        # instances sharing one DiT (e.g. sub-domains) must run sequentially, not concurrently.
        ph, pw = self.dit.patch_size
        # Fail fast on a non-patch-divisible grid: integer division would silently truncate the latent
        # patch counts and misalign the detokenizer (load_model's crop and set_domain's snapping keep it aligned).
        if x.shape[-2] % ph != 0 or x.shape[-1] % pw != 0:
            raise ValueError(
                f"grid {tuple(x.shape[-2:])} not divisible by patch_size {(ph, pw)}"
            )
        lh, lw = x.shape[-2] // ph, x.shape[-1] // pw
        det = getattr(self.dit.detokenizer, "proj", self.dit.detokenizer)
        det.h_patches, det.w_patches = lh, lw
        # Precision: fp32 by default (disable any ambient autocast). If amp_dtype is set, run the forward
        # under autocast at that dtype: weights stay fp32 and autocast downcasts eligible ops, so no checkpoint
        # recast is needed (the linear base and envelope pin in _sample_tau run in fp32).
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
        # Batched multi-scale latent (b, n_noise, H, W): draw-larger-then-valid-blur per scale so the
        # variance is spatially stationary. Members are drawn independently from an advancing generator --
        # the per-device generator seeded by ``self.seed`` if set (reproducible), else the global RNG
        # (which E2S ensemble pipelines seed upstream); either way distinct members, no cross-chunk
        # collision. Longitude-periodic when lon_pad>0 so the circular pad wraps near-continuously.
        periodic = self.lon_pad > 0
        gen = None
        if (
            self.seed is not None
        ):  # per-device generator that advances across draws (see docstring)
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
        # Build the DiT conditioning maps for one output frame -- the ``cond`` block that ``_sample_tau``
        # concatenates with the endpoints and latent (``[x0, xT, cond, z]``) so the learned correction
        # ``f`` is conditioned on geography, time, bracket width, and input presence (not just the two
        # endpoints). Called once per emitted timestamp in ``_interpolate`` (cos_zenith is time-varying,
        # so it must be rebuilt per frame rather than cached).
        # Layout (batch, 87, H, W) = invariants(13: static 12 + live cos_zenith) + gap(1) + mask(73):
        #   - invariants: static geography (sin/cos lat-lon, lsm, orog, physical fields), cos_zenith last
        #   - gap: the coarse bracket width, normalized to (gap_h - 6)/3 (see gap_val in _interpolate)
        #   - mask: ones, with dropped variables set to 0 so the DiT sees them as intentionally-absent
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
        )  # (batch, 13, H, W) with cos_zenith last
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
        # One interpolation at fraction ``tau`` for a normalized batch (B,73,H,W) with a given latent
        # ``z`` (B,n_noise,H,W): circular-pad -> DiT -> envelope pin -> crop. Returns normalized.
        p = self.lon_pad
        x0p, xTp = self._cpad(x0n), self._cpad(xTn)
        inp = torch.cat([x0p, xTp, self._cpad(cond), self._cpad(z)], dim=1)
        taut = torch.full(
            (x0n.shape[0],), float(tau), device=x0n.device, dtype=x0n.dtype
        )
        f = self._dit_forward(inp, taut)
        t = taut.reshape(-1, 1, 1, 1)
        out = (
            (1.0 - t) * x0p + t * xTp + torch.sin(math.pi * t) * f
        )  # envelope pin on padded grid
        return out[..., p : p + x0n.shape[-1]] if p > 0 else out

    def _pad_present_to_full(self, x_present: torch.Tensor) -> torch.Tensor:
        # Expand a base frame carrying only the present variables back to the full variable set,
        # filling dropped channels with their center so they normalize to zero for the DiT. Identity
        # when nothing is dropped. Variable axis is dim -3 (batch, time, lead, var, lat, lon).
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
        # Yield the interior sub-steps between two coarse (physical-unit) states x0, x1. One latent per
        # (ensemble member, init time) is drawn per bracket and reused across sub-steps (coherent).
        # Dropped channels arrive center-filled from _pad_present_to_full, so they normalize to zero here
        # (training-consistent) with no explicit masking needed.
        x0n = (x0 - self.center) / self.scale
        x1n = (x1 - self.center) / self.scale
        gap_min = self._gap_minutes()
        gap_val = (gap_min / 60.0 - 6.0) / 3.0
        step_min = gap_min // self.num_interp_steps
        base_lead = coords["lead_time"][-1]
        nt = x0n.shape[1]  # number of init times
        b, hh, ww = len(coords["batch"]), x0n.shape[-2], x0n.shape[-1]
        # One latent per (member, init time), drawn once per bracket and reused across sub-steps so a
        # member's interior stays temporally coherent. Drawing (b*nt) members keeps distinct init times'
        # noise independent; z[:, ti] then selects time ti's per-member latent. (nt==1 is the common case
        # and draws identically to a plain (b) draw.)
        z = self._draw_noise(b * nt, hh, ww, x0n.device, x0n.dtype).reshape(
            b, nt, -1, hh, ww
        )
        for interp_step in range(1, self.num_interp_steps):
            tau = interp_step / self.num_interp_steps
            # Sub-step coords: advance lead from the bracket start; report the trimmed output grid. Built
            # directly (not via output_coords) so the interior runs on the run grid (incl. the halo border)
            # while the emitted frame is the trimmed bounding box -- no per-step re-handshake against the run grid.
            sub = OrderedDict(coords)
            sub["lead_time"] = np.array(
                [base_lead + np.timedelta64(step_min * interp_step, "m")]
            )
            sub["variable"] = np.array(self.present_variables)
            sub["lat"] = np.asarray(self._out_lat1d, dtype=np.float64)
            sub["lon"] = np.asarray(self._out_lon1d, dtype=np.float64)
            out = torch.zeros_like(x0n)
            # x0n is (batch, time, lead, var, H, W). The base px_model emits one frame per step, so the
            # lead dim is always 1 and the inner `lti` loop runs once (not a general multi-lead sweep).
            # `batch` is the ensemble dim; `z[:, ti]` gives each init time `ti` its own per-member latent,
            # so multiple init times get independent noise while a member stays coherent across the bracket.
            for ti, t in enumerate(sub["time"]):
                for lti, lt in enumerate(sub["lead_time"]):
                    when = datetime.fromisoformat(str(t + lt)[:19])
                    cond = self._build_cond(when, gap_val, b, x0n.device, x0n.dtype)
                    out[:, ti, lti] = self._sample_tau(
                        x0n[:, ti, lti], x1n[:, ti, lti], cond, tau, z[:, ti]
                    )
            emitted = self._trim(out * self.scale + self.center)
            if len(self.drop_idx):  # emit only the present (non-dropped) variables
                emitted = emitted[..., self.present_idx, :, :]
            yield (emitted, sub)

    # ------------------------------------------------------------------ public
    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run one step: return the initial condition (step 0).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            The initial-condition frame and its coordinate system.
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
        # Reconcile each base frame onto the run grid: map_coords crops the common
        # 721-lat base grid to the model's 720 (a contiguous slice), or a global base to a sub-domain
        # bounding box+halo, so a standard 0.25 deg prognostic can drive the interpolator. x_run keeps
        # the run grid the interior interpolation needs (incl. the halo border); a second map_coords then
        # slices it to the trimmed output grid for emission (keeping the base lead_time verbatim).
        run_target = OrderedDict(
            {
                "variable": np.array(self.present_variables),
                "lat": np.asarray(self._lat1d, dtype=np.float64),
                "lon": np.asarray(self._lon1d, dtype=np.float64),
            }
        )
        for fc_step, (x, coords) in enumerate(self.px_model.create_iterator(x, coords)):
            # front_hook transforms each incoming base frame before it drives the bracket. It feeds both
            # the emitted endpoint and the anchor (x0) carried into the next bracket, so a hooked input
            # stays consistent across the rollout.
            x, coords = self.front_hook(x, coords)
            if fc_step == 0 and not (
                np.all(np.isin(self._lat1d, coords["lat"]))
                and np.all(np.isin(self._lon1d, coords["lon"]))
            ):
                # The run grid is not a subset of the base grid, so map_coords will nearest-regrid rather
                # than crop -- a signal the base is not the expected 0.25 deg grid (still runs, ERA5-trained).
                warnings.warn(
                    "px_model grid does not contain the model run grid; the base is being nearest-regridded "
                    "onto it (expected an exact 0.25 deg grid). Interpolation proceeds but may be degraded.",
                    stacklevel=2,
                )
            # Map only the present (non-dropped) variables from the base, then expand back to the full
            # variable set with center at the dropped positions (normalized zero for the DiT). The base
            # therefore need only supply VARIABLES minus drop_variables (e.g. a 69-variable model).
            x_present, coords_present = map_coords(x, coords, run_target)
            x_run = self._pad_present_to_full(x_present)
            coords_run = OrderedDict(coords_present)
            coords_run["variable"] = np.array(self.variables)
            # Crop the run-grid frame down to the trimmed output grid to emit the coarse endpoint.
            # The output_coords(...) target does two things:
            #   1. gives map_coords the output grid to crop to (variable + _out_lat1d/lon), and
            #   2. as a side effect, validates the gap (raises if not divisible by num_interp_steps,
            #      warns if outside the trained 3-10 h range).
            # output_coords also advances lead_time, but that does not matter: map_coords ignores
            # lead_time, so emit_coords keeps the frame's original (coarse-endpoint) lead_time.
            x_emit, emit_coords = map_coords(
                x_run, coords_run, self.output_coords(coords_run)
            )
            # rear_hook: transform each emitted frame just before yield. It acts only on the emitted
            # tensor (x_emit / interior xo), never on x_run, so the interpolation anchor (x0) stays
            # unhooked and the rollout state cannot be corrupted by an output-modifying hook.
            if fc_step == 0:
                x0, coords0 = x_run, coords_run
                yield self.rear_hook(x_emit, emit_coords)
            else:
                for xo, co in self._interpolate(x0, x_run, coords0):
                    yield self.rear_hook(xo, co)
                # the coarse endpoint, trimmed to the output grid
                yield self.rear_hook(x_emit, emit_coords)
                x0, coords0 = x_run, coords_run

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Time-integrating iterator: yield the initial condition, then the interpolated sub-steps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Each emitted frame (verbatim endpoints + interpolated interior) and its coordinate system.
        """
        yield from self._default_generator(x, coords)
