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

"""COSMO-REA downscaling diagnostic models (ERA5 -> COSMO-REA6 / REA2).

Earth2Studio :class:`DiagnosticModel` wrapper for the ``corrdiff-cosmo-era5``
package. One class: ``mode`` selects mean (deterministic) or diffusion (sampled
with the EDM/Karras scheme); ``resolution`` selects the rea6/rea2 checkpoint. Both
modes use a DiT-RoPE network (a diffusion transformer with rotary position
embeddings) at a fixed output resolution; the network is crop-size agnostic, so a
sub-region of any size runs in a single forward.

Domain handling:

* The model consumes an ERA5 crop and downscales onto a **rotated-pole** target
  grid. ``time`` is a leading coordinate dimension (not folded into batch)
  because the solar-zenith conditioning channel depends on the validity time.
* The package ships the **native** trained grid + static invariants verbatim, plus
  an **extended** grid + invariants covering a margin around it. Full-domain
  ``__call__`` takes input regridded onto ``input_coords()`` (the native footprint);
  sub-regions -- including into the extended margin -- are reached via :meth:`set_domain`
  (next bullet).
* A **sub-region** is obtained with :meth:`set_domain`, which slices the grid +
  exact invariants to a bounding box (no aggregation). The bbox may lie inside the
  native trained footprint, or reach into the extended invariant margin -- the
  latter proceeds with a one-time out-of-distribution warning; beyond the extended
  extent raises. Both the mean and diffusion models are DiT-RoPE (crop-size
  agnostic at the fixed resolution), so a sub-domain runs at any size in a single
  forward.
"""

import json
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.lexicon import CosmoLexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim, interp
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    import natten  # noqa: F401  # the DiT needs NATTEN (neighborhood attention)
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.models.dit import DiT
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("cosmo")
    cos_zenith_angle = None


# Fixed background channel layout.
# The order must match the network's trained channel layout -- do not reorder:
#   [ ERA5 (z-scored) | sin_lat cos_lat sin_lon cos_lon | elevation_norm
#     | land_fraction | cos_zenith | *extra invariants (z0 surface roughness
#       length, slopes, ...) ]
# Normalization: ERA5 + elevation/z0/slopes/dist-coast = z-score (the *_norm
# channels); land_fraction + continentality = identity; sin/cos + cos_zenith = raw.
POS_VARIABLES = ("sin_lat", "cos_lat", "sin_lon", "cos_lon")
COS_ZENITH_VARIABLE = "cos_zenith"
# Downwelling shortwave fluxes: every one of these present in the output must be
# covered by a solar gate (a dawn/dusk ramp) -- _parse_constraints validates
# this so a shortwave channel can never be left without night handling.
SHORTWAVE_VARIABLES = ("ASWDIR_S", "ASWDIFD_S")

# COSMO output names -> Earth2Studio names live in ``CosmoLexicon``
# (earth2studio/lexicon/cosmo.py). It is consumed ONLY at output_coords (internal
# indexing keeps the interior COSMO names) via ``CosmoLexicon.to_e2studio``, which
# returns the Earth2Studio name plus a unit scale (e.g. CLCT % -> tcc 0-1 fraction,
# applied at the end of postprocess). Names with no canonical equivalent fall back
# to their lowercased COSMO name.

# Supported COSMO-REA variants — "rea6" (~6 km) and "rea2" (~2.2 km). Each is a
# distinct product with its own native grid, model, invariants, and transforms.
# This tuple only lists the variant names to validate the ``resolution`` argument
# (also used as the package-subfolder key); the actual grid resolution isn't stored
# here because the native grid ships verbatim in the package.
SUPPORTED_VARIANTS = ("rea6", "rea2")

# Hosted URI for the combined (rea6/ + rea2/) downscaling package, used by
# ``load_default_package`` / ``from_pretrained``. Left ``None`` until weight
# hosting + license clearance land; flipping this on is the only change needed.
# Planned host: Hugging Face (commit-pinned), e.g.:
#   "hf://<org>/corrdiff-cosmo-era5@<commit-sha>"
# (NGC also works: "ngc://models/nvidia/earth-2/corrdiff_cosmo_era5@v0.1")
# The package nests rea6/ and rea2/ subfolders; ``load_model(..., resolution=)``
# selects the subfolder, so one URI serves all four models.
DEFAULT_PACKAGE_URI: str | None = None


def _points_in_grid_footprint(
    plat: np.ndarray, plon: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray
) -> np.ndarray:
    """Whether geographic points fall inside the (curvilinear) grid footprint.

    The native rotated grid is a curved quad in geographic space, so a lat/lon
    bounding box can poke outside it even while inside its lat/lon extent. This
    runs a ray-cast point-in-polygon test against the grid's outer boundary ring
    (the edge-cell centers, so it is conservative by ~half a cell). Assumes the
    footprint does not cross the antimeridian (true for the European domains).

    Parameters
    ----------
    plat, plon : np.ndarray
        Query point latitudes / longitudes (any shape, broadcast together).
    lat2d, lon2d : np.ndarray
        The target grid's 2D [lat, lon] coordinate arrays.

    Returns
    -------
    np.ndarray
        Boolean mask (shape of ``plat``), True where the point is inside.
    """
    ring_lat = np.concatenate(
        [lat2d[0, :], lat2d[:, -1], lat2d[-1, ::-1], lat2d[::-1, 0]]
    ).astype(np.float64)
    ring_lon = np.concatenate(
        [lon2d[0, :], lon2d[:, -1], lon2d[-1, ::-1], lon2d[::-1, 0]]
    ).astype(np.float64)
    plat = np.atleast_1d(plat).astype(np.float64)
    plon = np.atleast_1d(plon).astype(np.float64)
    inside = np.zeros(plat.shape, dtype=bool)
    n = ring_lat.size
    j = n - 1
    # Even-odd (crossing-number) ray cast: toggle `inside` each time the ring
    # edge i-j straddles the point's latitude and crosses to its left. The
    # 1e-30 avoids div-by-zero on horizontal edges.
    for i in range(n):
        yi, yj = ring_lat[i], ring_lat[j]
        xi, xj = ring_lon[i], ring_lon[j]
        straddle = (yi > plat) != (yj > plat)
        x_int = (xj - xi) * (plat - yi) / (yj - yi + 1e-30) + xi
        inside ^= straddle & (plon < x_int)
        j = i
    return inside


# Optional derived output: hub-height wind components ``u{H}m``/``v{H}m``, from
# vertical interpolation of the model-level winds to a requested height (opt-in via
# the ``hub_heights``/``hub_interp``/``wind_levels`` constructor args; computed in
# ``postprocess_output`` and appended to ``output_coords``). Wind speed ``ws{H}m``
# is not produced here: compose the stock ``DerivedWS(levels=["{H}m"])`` on the
# output.


def _interp_levels_to_height(
    values: torch.Tensor,
    level_heights: torch.Tensor,
    target: float,
    method: Literal["linear", "log"] = "linear",
) -> torch.Tensor:
    """Per-pixel vertical interpolation of a profile to one target height.

    Parameters
    ----------
    values : torch.Tensor
        ``[..., K, H, W]`` profile (e.g. a wind component on ``K`` levels),
        ordered ascending in geometric height along the ``K`` axis.
    level_heights : torch.Tensor
        ``[K, H, W]`` geometric height (metres above ground) of each level, ascending in
        ``K``. Per-pixel (terrain-following), broadcast over the leading dims.
    target : float
        Target height (metres above ground).
    method : {"linear", "log"}
        Interpolation in geometric height (``"linear"``) or in ``ln(height)``
        (``"log"``, a log-law-style profile; heights are floored at 1 m). Both
        clamp to the nearest level outside the profile range (no extrapolation).

    Returns
    -------
    torch.Tensor
        ``[..., H, W]`` interpolated value at ``target``.
    """
    k = level_heights.shape[0]
    if method == "log":
        h = torch.log(level_heights.clamp_min(1.0))
        t = float(np.log(max(float(target), 1.0)))
    elif method == "linear":
        h = level_heights
        t = float(target)
    else:
        raise ValueError(f"method must be 'linear' or 'log', got {method!r}")
    out = values[..., 0, :, :].clone()  # clamp below the lowest level
    for j in range(k - 1):
        lo, hi = h[j], h[j + 1]  # [H, W]
        w = ((t - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        interp_val = (
            values[..., j, :, :] + (values[..., j + 1, :, :] - values[..., j, :, :]) * w
        )
        out = torch.where((t >= lo) & (t < hi), interp_val, out)
    return torch.where(t >= h[k - 1], values[..., k - 1, :, :], out)


@check_optional_dependencies()
class CorrDiffCosmoEra5(torch.nn.Module, AutoModelMixin):
    """COSMO-REA downscaling model: ERA5 -> high-resolution COSMO-REA.

    Diagnostic model that downscales a global ERA5 state to high-resolution
    COSMO-REA regional reanalysis over Europe -- COSMO-REA6 (~6 km) or COSMO-REA2
    (~2.2 km), selected with ``resolution``. The input is an ERA5 state, so it can
    downscale an ERA5 analysis directly or run behind a global forecast model
    (e.g. SFNO -> CorrDiffCosmoEra5).

    ``mode`` selects one of two networks trained per resolution:

    * ``"mean"`` -- a deterministic regression predicting the conditional mean
      ``E[y | x]``: a single smooth field in one forward pass. Fast; a good first
      high-resolution look or a deterministic conditional-mean baseline.
    * ``"diffusion"`` -- a generative model sampling the conditional distribution
      ``p(y | x)``: an ensemble of ``number_of_samples`` realizations (seeded for
      reproducibility) that also captures the spread the mean cannot represent.

    Both networks are PhysicsNeMo DiTs at a fixed grid resolution but crop-size
    agnostic, so :meth:`set_domain` returns an instance restricted to any lat/lon
    sub-region without retraining (bounded to the trained footprint). Each
    resolution ships an extended grid beyond its native footprint; for COSMO-REA2
    -- whose native footprint is central-European -- the extended grid reaches a
    broad European domain, enabling 2.2 km downscaling across it. Outputs are
    surface and model-level (3D) fields -- winds,
    temperature, humidity, precipitation, cloud cover, fluxes, TKE, PBL height;
    variables with a canonical Earth2Studio name are relabelled via
    :class:`~earth2studio.lexicon.CosmoLexicon` and COSMO-specific fields keep a
    descriptive name. Optionally emits derived hub-height wind components (see
    ``hub_heights``).

    Parameters
    ----------
    era5_variables : Sequence[str]
        ERA5 input variable names (canonical Earth2Studio lexicon), in network order.
    output_variables : Sequence[str]
        Output (COSMO-REA) variable names, in network output-channel order.
    regression_model : torch.nn.Module | None
        Network for ``mode="mean"`` (an upstream physicsnemo RoPE-NATTEN ``DiT``).
    diffusion_model : torch.nn.Module | None
        Network for ``mode="diffusion"`` (an upstream physicsnemo
        ``EDMPreconditioner(ConcatConditionWrapper(DiT))``).
    resolution : {"rea6", "rea2"}
        Which COSMO-REA resolution this package targets.
    mode : {"mean", "diffusion"}
        Which model to run.
    lat_input_grid, lon_input_grid : torch.Tensor
        1D increasing regular ERA5 input grid (the native default footprint).
    lat_output_grid, lon_output_grid : torch.Tensor
        2D curvilinear native (rotated-pole) target grid [out_lat, out_lon].
    era5_center, era5_scale : torch.Tensor
        ERA5 input normalization (mean/std), size [n_era5].
    out_center, out_scale : torch.Tensor
        Output normalization (mean/std) in transformed space, size [n_out].
    static_invariants : OrderedDict[str, torch.Tensor]
        Native, already-normalized static invariant channels (non-position),
        each [out_lat, out_lon] — e.g. ``elevation_norm``, ``land_fraction``,
        ``z0_lu``, ``slope_east``, ``slope_north``. Position channels (sin/cos
        lat/lon) and ``cos_zenith`` are computed, not stored here.
    pre_invariant_variables, post_invariant_variables : Sequence[str]
        Names of the invariant channels before / after ``cos_zenith`` in the
        background channel order.
    channel_transforms : dict | None
        Per-output-channel nonlinear transform spec (from the package metadata),
        used to invert after de-normalizing.
    constraints : dict | None
        Physical-constraint spec (``metadata["constraints"]``): per-channel
        ``bounds`` (min/max clamps) and a shortwave ``sza_gate`` (the dawn/dusk
        solar gate), applied in physical space in postprocess for both modes.
    number_of_samples : int
        Number of samples (diffusion); the ``sample`` dim is kept (size 1) for
        ``mode="mean"`` so both modes share an output contract. The constructor
        value is the default; it is settable at runtime between calls
        (``model.number_of_samples = N``).
    physical_clamp : bool
        Apply physical-bounds + the shortwave dawn/dusk solar gate to outputs.
    number_of_steps : int
        Diffusion sampler step count (EDM Heun). Defaults to 18.
    sigma_min, sigma_max : float
        Diffusion noise-schedule bounds. Default to 0.002 / 800.0.
    rho : float
        Karras noise-schedule exponent. Defaults to 7.0.
    solver : {"heun", "euler"}
        Diffusion ODE solver: ``"heun"`` (2nd-order, default) or ``"euler"``
        (1st-order).
    seed : int | None
        Base RNG seed for diffusion sampling (a per-sample offset is added).
        ``None`` (default) leaves sampling unseeded.
    amp : bool
        Run the network forward passes (regression patches and the diffusion
        denoiser) under ``torch.autocast`` bf16. Roughly halves inference time
        on Ampere+ GPUs at a negligible accuracy cost; the sampler bookkeeping
        stays in fp64. Defaults to ``False`` (full precision).
    hub_heights : Sequence[float], optional
        Optional list of above-ground heights (m) at which to emit interpolated
        wind components ``u{H}m``/``v{H}m`` (e.g. ``[100]`` -> ``u100m``,
        ``v100m``), vertically interpolated from the model's 3D wind levels. Wind
        speed is obtained by composing the stock ``DerivedWS(levels=["{H}m"])``
        diagnostic on the output. ``None`` (default) -> no derived outputs and
        behavior/``output_variables`` are unchanged. Requires ``wind_levels``.
        Names follow the lexicon ``m``-suffix convention (m above surface, as
        ``u100m``); arbitrary ``H`` need not be pre-registered in the lexicon
        (outputs are produced, not fetched). A height outside the levels' range
        is clamped (not extrapolated) and warns.
    hub_interp : {"linear", "log"}, optional
        Interpolation method for the hub-height wind components: ``"linear"`` (in
        geometric height) or ``"log"`` (in ``ln(height)``). Both are
        roughness-free between levels. Defaults to ``"linear"``.
    wind_levels : dict, optional
        Package metadata describing the 3D wind levels for hub-height derivation:
        each level's ``{u, v}`` output channels + its ``a, b`` height coefficients
        (``height = a + b*elevation_invariant``) and the elevation invariant name.
        Supplied by the package; required when ``hub_heights`` is set.

    Notes
    -----
    The diffusion sampler controls (``number_of_steps``, ``sigma_min``,
    ``sigma_max``, ``rho``, ``solver``) default to the package metadata but are
    exposed for inference tuning rather than fixed: as with any diffusion sampler,
    these settings can shape the sampled output distribution (e.g. ensemble spread
    and the representation of extremes).

    The models are trained on ERA5 (global input) paired with COSMO-REA regional
    reanalysis (high-resolution target) over Europe:

    * COSMO-REA6 (~6 km), DWD: https://reanalysis.meteo.uni-bonn.de/?COSMO-REA6
    * COSMO-REA2 (~2.2 km), DWD: https://reanalysis.meteo.uni-bonn.de/?COSMO-REA2
    * ERA5, ECMWF: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5

    Badges
    ------
    region:eu class:ds product:wind product:precip product:temp product:atmos year:2026 gpu:80gb
    """

    def __init__(
        self,
        era5_variables: Sequence[str],
        output_variables: Sequence[str],
        regression_model: torch.nn.Module | None,
        diffusion_model: torch.nn.Module | None,
        resolution: Literal["rea6", "rea2"],
        mode: Literal["mean", "diffusion"],
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        era5_center: torch.Tensor,
        era5_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        static_invariants: "OrderedDict[str, torch.Tensor]",
        pre_invariant_variables: Sequence[str],
        post_invariant_variables: Sequence[str],
        channel_transforms: dict | None = None,
        constraints: dict | None = None,
        number_of_samples: int = 1,
        physical_clamp: bool = True,
        number_of_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800.0,
        rho: float = 7.0,
        solver: Literal["heun", "euler"] = "heun",
        seed: int | None = None,
        amp: bool = False,
        hub_heights: Sequence[float] | None = None,
        hub_interp: Literal["linear", "log"] = "linear",
        wind_levels: dict | None = None,
    ):
        super().__init__()

        if mode not in ("mean", "diffusion"):
            raise ValueError(
                f"mode must be 'mean' or 'diffusion' (got {mode!r}). The two are "
                "independent standalone networks: unlike a residual/CorrDiff-style "
                "model, the diffusion conditions on ERA5 directly and is NOT a "
                "residual added to the mean -- so there is no combined 'both' mode."
            )
        if mode == "mean" and regression_model is None:
            raise ValueError("mode='mean' requires regression_model")
        if mode == "diffusion" and diffusion_model is None:
            raise ValueError("mode='diffusion' requires diffusion_model")
        if resolution not in SUPPORTED_VARIANTS:
            raise ValueError(f"resolution must be one of {list(SUPPORTED_VARIANTS)}")
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("number_of_samples must be a positive integer")

        self.mode = mode
        self.resolution = resolution
        self.number_of_samples = number_of_samples
        self.physical_clamp = physical_clamp
        if number_of_steps < 2:
            raise ValueError(
                f"number_of_steps must be >= 2 for the EDM schedule (got "
                f"{number_of_steps}); the schedule's step/(n-1) ramp divides by "
                "zero at n=1, silently producing NaN samples."
            )
        if solver not in ("heun", "euler"):
            raise ValueError(f"solver must be 'heun' or 'euler', got {solver!r}.")
        self.number_of_steps = number_of_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.seed = seed
        self.amp = amp

        self.era5_variables = list(era5_variables)
        self.output_variables = list(output_variables)
        self.pre_invariant_variables = list(pre_invariant_variables)
        self.post_invariant_variables = list(post_invariant_variables)
        # Full ordered background channel list (order must match the trained
        # layout -- see the channel layout up top).
        self.background_variables = (
            self.era5_variables
            + self.pre_invariant_variables
            + [COS_ZENITH_VARIABLE]
            + self.post_invariant_variables
        )

        self.regression_model = regression_model
        self.diffusion_model = diffusion_model

        # Grids
        self._validate_input_grid(lat_input_grid, lon_input_grid)
        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)
        self.lat_input_numpy = lat_input_grid.cpu().numpy()
        self.lon_input_numpy = lon_input_grid.cpu().numpy()
        self.lat_output_numpy = lat_output_grid.cpu().numpy()
        self.lon_output_numpy = lon_output_grid.cpu().numpy()

        # Normalization stats kept as buffers and applied later, not here:
        #   - era5_center / era5_scale: z-score the ERA5 inputs (preprocess_input).
        #   - out_center / out_scale: de-normalize the network output
        #     (postprocess_output). This lands in the model's "transformed space";
        #     the channel transforms are then inverted to reach physical units
        #     (for the mean those transforms are identity -> already physical).
        n_era5 = len(self.era5_variables)
        n_out = len(self.output_variables)
        self.register_buffer("era5_center", era5_center.view(1, n_era5, 1, 1))
        self.register_buffer("era5_scale", era5_scale.view(1, n_era5, 1, 1))
        self.register_buffer("out_center", out_center.view(1, n_out, 1, 1))
        self.register_buffer("out_scale", out_scale.view(1, n_out, 1, 1))

        # Native static invariants loaded from the package (elevation,
        # land_fraction, z0, slopes, ...), already normalized by load_model
        # (z-score for the _norm channels, identity for the rest), stacked into one
        # buffer; _static_names maps each name to its row, so the stacking order
        # does not matter (lookup is by name). The position channels (sin/cos
        # lat/lon) and cos_zenith
        # are NOT stored -- they are computed per call (from the grid / valid
        # time). At use time rows are pulled by name and split around cos_zenith
        # into the trained pre/post groups (see the channel layout up top).
        self._static_names = list(static_invariants.keys())
        self.register_buffer(
            "static_invariants",
            (
                torch.stack(list(static_invariants.values()), dim=0)
                if static_invariants
                else torch.empty(0)
            ),
        )

        self._channel_transforms = channel_transforms or {}
        self._parse_channel_transforms(self._channel_transforms)
        self._constraints = constraints or {}
        self._parse_constraints(self._constraints)
        # Output value rescales to match a canonical lexicon name's units (e.g.
        # CLCT % -> tcc fraction). From CosmoLexicon; applied at end of postprocess.
        self._output_unit_scale = [
            (i, scale)
            for i, v in enumerate(self.output_variables)
            for scale in (CosmoLexicon.to_e2studio(v)[1],)
            if scale != 1.0
        ]
        # Optional hub-height wind (derived; see the design note above). Off by
        # default -> behavior + output_variables unchanged. When requested, the
        # package must supply ``wind_levels``: per model level, the u/v output
        # channels (the wind values) plus a,b coeffs giving that level's height
        # (= a + b*elevation). Interpolation needs both -- the values and the
        # heights to interpolate them between (full schema in _setup_hub_wind).
        self._wind_levels = wind_levels
        self._hub_heights = [float(h) for h in (hub_heights or [])]
        self._hub_interp = hub_interp
        if self._hub_heights:
            if not all(h > 0 for h in self._hub_heights):
                raise ValueError(
                    f"hub_heights must be positive (m above ground), got "
                    f"{self._hub_heights}"
                )
            if wind_levels is None:
                raise ValueError(
                    "hub_heights requested but this package has no 'wind_levels' "
                    "metadata (3D wind levels + per-level height coefficients). "
                    "Hub-height wind is only available for packages that ship it."
                )
            if hub_interp not in ("linear", "log"):
                raise ValueError(
                    f"hub_interp must be 'linear' or 'log', got {hub_interp!r}"
                )
            self._setup_hub_wind(wind_levels)
        # Emit the interpolated wind COMPONENTS (u{H}m, v{H}m) -- the grid-tied
        # part only the model can do. Wind SPEED is a pointwise sqrt(u^2+v^2):
        # compose the stock ``DerivedWS(levels=["{H}m"])`` diagnostic on the
        # output rather than duplicating it here (keeps direction too).
        self._hub_labels = [
            f"{int(h) if float(h).is_integer() else h}m" for h in self._hub_heights
        ]
        self._derived_variables = [
            f"{c}{lbl}" for lbl in self._hub_labels for c in ("u", "v")
        ]
        # Output coord labels: COSMO names -> Earth2Studio names via CosmoLexicon
        # (internal indexing uses interior names); derived u/v{H}m appended last.
        self._output_coord_variables = np.array(
            [CosmoLexicon.to_e2studio(v)[0] for v in self.output_variables]
            + self._derived_variables
        )
        # check_inputs: user-flippable runtime toggle (an attribute, not a
        # constructor arg, so it can be set on an existing instance, e.g.
        # ``dx.check_inputs = False``). Warns if the network input has NaN/Inf.
        self.check_inputs = True
        # _halo: internal state managed by set_domain(halo=...), NOT a free toggle.
        # (top, bottom, left, right) px that set_domain() runs then trims off the
        # output; (0,0,0,0) = no halo (the default, i.e. set_domain not called or
        # called with halo=0).
        self._halo = (0, 0, 0, 0)
        # Extended grid + invariants for set_domain() into the out-of-distribution
        # (OOD) margin band.
        # Set by load_model (the loaded model); None on sub-domains/synthetic
        # instances -> set_domain then restricts to the native footprint only.
        self._ext_lat_numpy: np.ndarray | None = None
        self._ext_lon_numpy: np.ndarray | None = None
        self._ext_static_numpy: np.ndarray | None = None
        # DiT grid constraints, used by set_domain to snap the run grid: the
        # detokenizer reconstructs (H//patch)*patch rows, so an odd extent would
        # mismatch the allocated output; and NATTEN needs the latent >= kernel, so
        # the run domain must be >= attn_kernel*patch per side. load_model sets
        # these from the arch; the (1, 0) defaults make them no-ops for synthetic
        # instances that have no real DiT.
        self._patch_size = 1
        self._min_domain_cells = 0

    # ── validation ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_input_grid(lat: torch.Tensor, lon: torch.Tensor) -> None:
        """Validate the ERA5 input grid: 1D, strictly increasing, and regularly
        spaced. These are the assumptions ``latlon_interpolation_regular`` relies
        on -- it derives a single spacing from the first interval and uses
        ``searchsorted``, both of which silently produce wrong values otherwise.
        """
        for name, g in (("latitude", lat), ("longitude", lon)):
            if g.ndim != 1:
                raise ValueError(f"Input {name} grid must be 1D (regular grid).")
            if g.numel() < 2:
                raise ValueError(f"Input {name} grid must have at least 2 points.")
            d = torch.diff(g)
            if not torch.all(d > 0):
                suffix = " (south to north)" if name == "latitude" else ""
                raise ValueError(f"Input {name} must be strictly increasing{suffix}.")
            # The interpolator uses g[1]-g[0] as THE spacing for the whole axis.
            if not torch.allclose(d, d[0], rtol=1e-3, atol=0.0):
                raise ValueError(
                    f"Input {name} must be regularly spaced (uniform grid)."
                )

    def _check_bounds(self, lat_out: np.ndarray, lon_out: np.ndarray) -> None:
        """Target grid must lie within the ERA5 input grid (no extrapolation).

        ``interp.latlon_interpolation_regular`` has no border clamp, so an
        out-of-bounds target would index past the array. Also note the input
        grid must not cross the antimeridian (no lon wrap is applied).
        """
        lat0, lat1 = float(self.lat_input_numpy[0]), float(self.lat_input_numpy[-1])
        lon0, lon1 = float(self.lon_input_numpy[0]), float(self.lon_input_numpy[-1])
        # Strict containment: a target exactly on the first input point would
        # index the interpolator at -1 (wrap to the far edge -> garbage), so the
        # input grid must STRICTLY bracket the target (the shipped grids carry a
        # >=1 deg margin, so this only rejects a degenerate tight crop).
        if lat_out.min() <= lat0 or lat_out.max() >= lat1:
            raise ValueError(
                f"Target latitude [{lat_out.min():.2f}, {lat_out.max():.2f}] is not "
                f"strictly inside the ERA5 input grid [{lat0:.2f}, {lat1:.2f}]. "
                "Provide an ERA5 crop that covers the target domain with a margin."
            )
        if lon_out.min() <= lon0 or lon_out.max() >= lon1:
            raise ValueError(
                f"Target longitude [{lon_out.min():.2f}, {lon_out.max():.2f}] is not "
                f"strictly inside the ERA5 input grid [{lon0:.2f}, {lon1:.2f}]. "
                "Provide an ERA5 crop that covers the target domain with a margin "
                "(and does not cross the antimeridian)."
            )

    # ── channel transforms ──────────────────────────────────────────────────

    def _parse_channel_transforms(self, channel_transforms: dict) -> None:
        """Pre-compute inverse-transform / clamp index lists.

        Forward transforms (preprocessing): ``log_eps`` is ``x -> log(1 + x/eps)``,
        ``asinh`` is ``x -> arcsinh(x/eps)`` (inverse ``eps*sinh(y)``), and
        ``logit_eps[_percent]`` is ``x -> logit(eps + (1-2*eps)*(x/scale))``.
        """

        def _spec(ch: str) -> tuple[str, float, float]:
            """Normalize a channel's transform spec to (name, eps, scale); reject a
            bare string (one with no explicit eps/scale)."""
            t = channel_transforms.get(ch, "")
            if isinstance(t, dict):
                return (
                    str(t.get("transform", "")).lower(),
                    float(t.get("eps", 1e-5)),
                    float(t.get("scale", 1.0)),
                )
            if t:  # a non-empty bare string carries no eps/scale -> refuse loudly
                raise ValueError(
                    f"channel_transforms[{ch!r}] must be a dict with explicit "
                    f"'eps'/'scale'; got bare string {t!r}."
                )
            return "", 1e-5, 1.0

        self._log_eps_idx: list[tuple[int, float]] = []  # inverse eps*(exp(y)-1)
        self._asinh_idx: list[tuple[int, float]] = []  # inverse eps*sinh(y)
        self._logit_idx: list[tuple[int, float, float]] = []  # (idx, eps, scale)
        for i, ch in enumerate(self.output_variables):
            name, eps, scale = _spec(ch)
            if not name:
                continue  # identity (no transform)
            elif "log_eps" in name:
                self._log_eps_idx.append((i, eps))
            elif "asinh" in name:
                self._asinh_idx.append((i, eps))
            elif "logit_eps" in name:
                # _percent uses a 0-100 scale (default 100); plain uses scale as
                # given. Single branch -> no "percent" vs plain substring ordering trap.
                pct = "percent" in name
                self._logit_idx.append(
                    (i, eps, (scale if scale != 1.0 else 100.0) if pct else scale)
                )
            else:
                raise ValueError(
                    f"channel_transforms[{ch!r}] has unsupported transform {name!r}; "
                    "expected one of: log_eps, asinh, logit_eps[_percent] (or none)."
                )

    def _parse_constraints(self, constraints: dict | None) -> None:
        """Pre-compute inline physical-constraint indices (bounds + solar gate).

        Applied in postprocess in PHYSICAL space (after de-norm + inverse
        transform). For clamp/ReLU bounds the result is exactly equivalent to a
        training-time constraint wrapper applying them in normalized space: the
        de-norm and inverse transforms are monotonic (order-preserving), so
        clamping to the physical bounds gives the same result as clamping in
        normalized space. The solar gate (a [0,1] multiply) is applied physically
        too and is mode-independent: there is a SINGLE gate list, used identically
        by both the mean and diffusion products.
        Spec is ``metadata["constraints"]``: ``bounds: {ch: {min, max}}`` and
        ``sza_gate: {half_width, channels: {ch: {threshold, half_width}}}``.
        Channels not in ``output_variables`` are ignored. A channel's ``half_width``
        overrides the gate-wide default (it must be > 0).
        """
        constraints = constraints or {}
        idx = {ch: i for i, ch in enumerate(self.output_variables)}
        self._bound_lo: list[tuple[int, float]] = []
        self._bound_up: list[tuple[int, float]] = []
        for ch, b in (constraints.get("bounds") or {}).items():
            if ch not in idx or not isinstance(b, dict):
                continue
            if str(b.get("mode", "")).lower() == "sigmoid":
                # Two-sided sigmoid is a training-time op (lo+(up-lo)*sigmoid(raw));
                # applied after de-norm it gives a different result, so it cannot be
                # moved to inference post-processing.
                raise NotImplementedError(
                    f"constraints.bounds[{ch!r}].mode='sigmoid' is unsupported as "
                    "inference post-processing; use 'clamp'."
                )
            if b.get("min") is not None:
                self._bound_lo.append((idx[ch], float(b["min"])))
            if b.get("max") is not None:
                self._bound_up.append((idx[ch], float(b["max"])))
        self._sza_gate: list[tuple[int, float, float]] = []
        sz = constraints.get("sza_gate") or {}
        _dhw = sz.get("half_width")  # None (missing) -> default
        default_hw = 0.05 if _dhw is None else float(_dhw)
        for ch, c in (sz.get("channels") or {}).items():
            if ch not in idx:
                continue
            if isinstance(c, dict):
                th = float(c.get("threshold", 0.01))
                _hw = c.get("half_width")
                hw = default_hw if _hw is None else float(_hw)
            else:
                th, hw = float(c), default_hw
            if hw <= 0:  # the dawn/dusk gate ramp width must be strictly positive
                raise ValueError(f"sza_gate half_width must be > 0 ({ch}: {hw})")
            self._sza_gate.append((idx[ch], th, hw))
        # Single source of truth: once a solar gate is configured at all, every
        # shortwave output must be covered by the ramp gate -- a partially
        # gated set is a loud error, not a silent gap. (A package with no sza_gate
        # at all is left ungated by design; there is no hardcoded night-zero.)
        gated = {self.output_variables[i] for i, _, _ in self._sza_gate}
        ungated_sw = [
            ch
            for ch in self.output_variables
            if ch in SHORTWAVE_VARIABLES and ch not in gated
        ]
        if self._sza_gate and ungated_sw:
            raise ValueError(
                f"sza_gate must cover all shortwave outputs; missing {ungated_sw}"
            )
        self._has_constraints = bool(self._bound_lo or self._bound_up or self._sza_gate)

    def _apply_constraints(
        self,
        x: torch.Tensor,
        valid_time: datetime,
        lat2d: np.ndarray,
        lon2d: np.ndarray,
    ) -> None:
        """In-place physical-space constraints (applied AFTER de-norm + inverse
        transforms, so ``x`` is physical here), identically for both modes.

        Order: (1) metadata min/max bounds -- a physical-space clamp is always valid;
        (2) the solar gate on shortwave: a smooth dawn/dusk ramp, the [0,1] multiply
        ``clamp((cos_z - th + hw)/(2*hw), 0, 1)`` (cos_z = max over the hourly
        window; ``hw`` is the ramp half-width). It is exactly 0 at deep night
        (``cos_z <= th - hw``). Both products use the same gate, so their twilight
        behaviour (and the diffusion ensemble mean vs the regression mean) stay
        consistent.
        """
        for i, lo in self._bound_lo:
            x[:, i].clamp_(min=lo)
        for i, up in self._bound_up:
            x[:, i].clamp_(max=up)
        if self._sza_gate:
            cos_z = torch.as_tensor(
                self._cos_zenith(valid_time, lat2d, lon2d),
                device=x.device,
                dtype=x.dtype,
            )[
                None
            ]  # [1, H, W] broadcasts over the sample dim
            for i, th, hw in self._sza_gate:
                gate = ((cos_z - th + hw) / (2.0 * hw)).clamp(0.0, 1.0)
                x[:, i] = (x[:, i] * gate).clamp(min=0.0)

    # ── coordinate systems (time is a leading coordinate dimension, not batched) ──

    def input_coords(self) -> CoordSystem:
        """Input coordinate system. ``time`` is a dynamic leading dim; lat/lon
        are the native ERA5 footprint (regrid the ERA5 input onto this grid)."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "variable": np.array(self.era5_variables),
                "lat": self.lat_input_numpy,
                "lon": self.lon_input_numpy,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system on the rotated-pole target grid.

        The input must be on the native ERA5 grid (:meth:`input_coords`); for a
        sub-region use :meth:`set_domain` (which gives a new instance with its own
        native grid). Arbitrary/flexible domains are not supported.
        """
        target = self.input_coords()
        handshake_dim(input_coords, "time", 1)
        handshake_dim(input_coords, "variable", -3)
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "lon", -1)
        handshake_coords(input_coords, target, "variable")
        if not self._is_native_input(
            np.asarray(input_coords["lat"]), np.asarray(input_coords["lon"])
        ):
            raise ValueError(
                "CorrDiffCosmoEra5 requires the native input grid from "
                "input_coords() (regrid your ERA5 onto it). For a sub-region use "
                "set_domain(); arbitrary/flexible domains are not supported."
            )
        lat_out, lon_out = self.lat_output_numpy, self.lon_output_numpy
        # Halo crop runs on an expanded grid but reports/returns the trimmed bbox.
        top, bot, left, right = self._halo
        if top or bot or left or right:
            H, W = lat_out.shape
            lat_out = lat_out[top : H - bot, left : W - right]
            lon_out = lon_out[top : H - bot, left : W - right]

        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "time": input_coords["time"],
                "variable": self._output_coord_variables,
                "lat": lat_out,
                "lon": lon_out,
            }
        )
        return output_coords

    def _is_native_input(self, lat: np.ndarray, lon: np.ndarray) -> bool:
        """True iff lat/lon match the native ERA5 input grid (within tolerance)."""
        return (
            lat.shape == self.lat_input_numpy.shape
            and lon.shape == self.lon_input_numpy.shape
            and np.allclose(lat, self.lat_input_numpy)
            and np.allclose(lon, self.lon_input_numpy)
        )

    # ── invariants ──────────────────────────────────────────────────────────

    def _position_channels(
        self, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> torch.Tensor:
        """sin/cos(lat), sin/cos(lon) for the target grid, [4, H, W]."""
        return torch.stack(
            [
                torch.sin(torch.deg2rad(lat2d)),
                torch.cos(torch.deg2rad(lat2d)),
                torch.sin(torch.deg2rad(lon2d)),
                torch.cos(torch.deg2rad(lon2d)),
            ],
            dim=0,
        )

    def _static_background(
        self, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pre_zenith, post_zenith) static invariant stacks for the grid.

        ``pre_zenith`` = [sin_lat, cos_lat, sin_lon, cos_lon, <pre non-position>];
        ``post_zenith`` = [<post non-position>]. cos_zenith is inserted between
        them per call (it is time-dependent). The invariants are the native
        (verbatim/sliced) buffers — the grid is always native (or a crop of it).
        """
        pos = self._position_channels(lat2d, lon2d)  # [4, H, W]
        non_pos = {
            name: self.static_invariants[i] for i, name in enumerate(self._static_names)
        }

        def _stack(names: Sequence[str]) -> torch.Tensor:
            chans = []
            for n in names:
                if n in POS_VARIABLES:
                    chans.append(pos[POS_VARIABLES.index(n)])
                else:
                    chans.append(non_pos[n])
            return torch.stack(chans, dim=0) if chans else lat2d.new_empty(0)

        return _stack(self.pre_invariant_variables), _stack(
            self.post_invariant_variables
        )

    def _cos_zenith(
        self, valid_time: datetime, lat2d: np.ndarray, lon2d: np.ndarray
    ) -> np.ndarray:
        """max(cos_zenith(t-1h), cos_zenith(t)) for an hourly-mean target."""
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)
        cz0 = cos_zenith_angle(valid_time - timedelta(hours=1), lon2d, lat2d)
        cz1 = cos_zenith_angle(valid_time, lon2d, lat2d)
        return np.maximum(cz0, cz1).astype(np.float32)

    # ── preprocessing / forward / postprocessing ────────────────────────────

    def _interpolate(
        self, x: torch.Tensor, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear-regrid [C, H_in, W_in] from the regular ERA5 grid onto the
        2D rotated target grid -> [C, H_out, W_out]."""
        return interp.latlon_interpolation_regular(
            x, self.lat_input_grid, self.lon_input_grid, lat2d, lon2d
        )

    def preprocess_input(
        self,
        era5: torch.Tensor,
        valid_time: datetime,
        lat2d: torch.Tensor,
        lon2d: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the normalized network background from a raw ERA5 frame.

        ``era5`` is [n_era5, H_in, W_in]. Returns [1, C_bg, H_out, W_out].
        """
        era5_r = self._interpolate(era5, lat2d, lon2d).unsqueeze(0)
        era5_r = (era5_r - self.era5_center) / self.era5_scale

        pre, post = self._static_background(lat2d, lon2d)
        H, W = lat2d.shape
        cz = torch.as_tensor(
            self._cos_zenith(valid_time, lat2d.cpu().numpy(), lon2d.cpu().numpy()),
            device=era5_r.device,
            dtype=torch.float32,
        ).view(1, 1, H, W)

        parts = [era5_r, pre.unsqueeze(0), cz]
        if post.numel():
            parts.append(post.unsqueeze(0))
        background = torch.concat(parts, dim=1).to(torch.float32)
        # Input health check: NaN/Inf (e.g. a data-source gap, fill value, or bad
        # interpolation at the grid edge) silently produces a garbage downscaled
        # field — warn loudly rather than fail silently. We deliberately do NOT also
        # flag "suspiciously large" values: the model is meant to run on inputs far
        # from its European training climate (a hot desert, a high mountain), which
        # produce values many std-devs from the training mean yet are perfectly
        # valid. So a large normalized magnitude means "far from the training
        # distribution" (the intended use), not corrupt data — only NaN/Inf
        # reliably indicates broken input.
        if self.check_inputs and not torch.isfinite(background).all():
            warnings.warn(
                "CorrDiffCosmoEra5: non-finite (NaN/Inf) values in the network "
                "background; the downscaled output will be invalid. Check the ERA5 "
                "input for missing data / fill values / off-grid interpolation.",
                stacklevel=2,
            )
        return background

    def _setup_hub_wind(self, wind_levels: dict) -> None:
        """Build the channel-index maps + per-level height coefficients for the
        hub-height wind derivation, from the package ``wind_levels`` block::

            {
              "elevation_invariant": "elevation_norm",
              "levels": [{"u": "U_L40", "v": "V_L40", "a": .., "b": ..}, ...],
            }

        Per-level above-ground height (m) is ``a + b * elevation_invariant`` — fit
        against the NORMALIZED elevation, so the wrapper reuses
        the elevation invariant it already holds and slices (valid on the
        extended grid too).
        Levels are stored ascending by nominal height.

        Only the model wind levels are used as interpolation nodes — there is no
        separate 10 m anchor. The lowest model level (~10 m above ground) already
        covers
        the near-surface, and every node shares one terrain-following geometry
        fit, so per-pixel heights stay monotonic by construction (the guard below
        cannot trip on a well-formed block). A fixed-height 10 m anchor (``b=0``)
        would instead cross a terrain-following lowest level (``b<0``) at low
        elevation, which is why it is deliberately omitted. NB: "no anchor" is
        only about the interpolation node set, not the model outputs. The model's
        own 10 m wind (``u10m``/``v10m``) is still emitted as a normal output
        channel; it is simply not reused as a 10 m node when interpolating to hub
        height (reusing it would be exactly the fixed-height anchor ruled out).
        """
        wl = wind_levels
        entries = [
            (lev["u"], lev["v"], float(lev["a"]), float(lev["b"]))
            for lev in wl.get("levels", [])
        ]
        if len(entries) < 2:
            raise ValueError(
                "wind_levels must define at least 2 'levels' to interpolate; got "
                f"{len(entries)}. A single level yields a constant (non-interpolated) "
                "field at every hub height."
            )
        entries.sort(key=lambda e: e[2])  # ascending nominal height
        ovi = {v: i for i, v in enumerate(self.output_variables)}
        missing = sorted({c for e in entries for c in (e[0], e[1]) if c not in ovi})
        if missing:
            raise ValueError(
                f"wind_levels references channels absent from output_variables: {missing}"
            )
        elev = wl.get("elevation_invariant", "elevation_norm")
        if elev not in self._static_names:
            raise ValueError(
                f"wind_levels.elevation_invariant {elev!r} is not a static invariant "
                f"({self._static_names})"
            )
        self._hub_elev_idx = self._static_names.index(elev)
        # Build the hub buffers on the same device as the other static buffers
        # (load_model builds static_invariants on `device`; the wrapper itself is
        # not separately .to(device)'d, so default-CPU tensors here would mismatch
        # static_invariants at the monotonicity check and at derive time).
        dev = self.static_invariants.device
        self.register_buffer(
            "_hub_u_idx",
            torch.tensor([ovi[e[0]] for e in entries], dtype=torch.long, device=dev),
        )
        self.register_buffer(
            "_hub_v_idx",
            torch.tensor([ovi[e[1]] for e in entries], dtype=torch.long, device=dev),
        )
        self.register_buffer(
            "_hub_a", torch.tensor([e[2] for e in entries], device=dev)
        )
        self.register_buffer(
            "_hub_b", torch.tensor([e[3] for e in entries], device=dev)
        )
        # ``_interp_levels_to_height`` assumes the PER-PIXEL heights ``a+b*elev``
        # are ascending in the level axis. Levels are sorted by nominal ``a``, but
        # a large ``b`` spread could in principle invert the order at extreme
        # terrain — validate over the actual elevation range so a bad metadata
        # block fails loudly here rather than producing silently-wrong winds.
        # (A well-formed levels-only block passes by construction; see docstring.)
        if self.static_invariants.numel():
            # _hub_a/_hub_b were built on static_invariants.device above, so they
            # are co-located with the elevation field here (no device juggling).
            elev_field = self.static_invariants[self._hub_elev_idx]
            h = self._hub_a[:, None, None] + self._hub_b[:, None, None] * elev_field
            if not bool((h[1:] - h[:-1] >= -1e-4).all()):
                raise ValueError(
                    "wind_levels height coefficients (a + b*elevation) are not "
                    "monotonic in height across this grid's elevation range; "
                    "interpolation requires ascending level heights."
                )
            # Heights are per-pixel, so the band where EVERY pixel interpolates is
            # [max(lowest level), min(highest level)]. A requested hub height
            # outside it is CLAMPED (held constant) at some pixels rather than
            # interpolated -- not extrapolated, but silently flat, which is
            # misleading. Warn (don't fail: a top-level proxy may be wanted).
            band_lo = float(h[0].max())  # highest "lowest-level" height on the grid
            band_hi = float(h[-1].min())  # lowest "highest-level" height on the grid
            outside = [hh for hh in self._hub_heights if hh < band_lo or hh > band_hi]
            if outside:
                warnings.warn(
                    f"hub_heights {outside} m fall outside the grid's fully "
                    f"resolvable band [{band_lo:.0f}, {band_hi:.0f}] m (model levels "
                    f"~{self._hub_a.min():.0f}-{self._hub_a.max():.0f} m above "
                    f"ground); winds "
                    "there are clamped to the nearest level (held constant), not "
                    "interpolated. Choose a height within the band to avoid this.",
                    UserWarning,
                    stacklevel=2,
                )

    def _derive_hub_wind(self, x: torch.Tensor) -> torch.Tensor:
        """Append interpolated wind COMPONENTS ``u{H}m, v{H}m`` (in that order per
        height) to physical output ``x`` ``[S, n_out, H, W]`` by per-pixel vertical
        interpolation of the model level winds to each hub height. Per-level height
        above ground =
        ``a + b * elevation_invariant``. Wind speed is left to a composed
        ``DerivedWS`` (sqrt(u^2+v^2)) so direction is preserved here."""
        elev = self.static_invariants[self._hub_elev_idx]  # [H, W] (normalized)
        heights = (
            self._hub_a[:, None, None] + self._hub_b[:, None, None] * elev
        )  # [K,H,W]
        u = x[:, self._hub_u_idx]  # [S, K, H, W]
        v = x[:, self._hub_v_idx]
        comps = []
        for h in self._hub_heights:
            # u{H}m then v{H}m -- matches DerivedWS's expected [u, v] pair order.
            comps.append(
                _interp_levels_to_height(u, heights, h, self._hub_interp).unsqueeze(1)
            )
            comps.append(
                _interp_levels_to_height(v, heights, h, self._hub_interp).unsqueeze(1)
            )
        return torch.cat([x, *comps], dim=1)

    def postprocess_output(
        self,
        x: torch.Tensor,
        valid_time: datetime,
        lat2d: np.ndarray,
        lon2d: np.ndarray,
    ) -> torch.Tensor:
        """De-normalize -> invert per-channel transforms -> physical clamp/bounds +
        solar gate -> unit rescale (e.g. CLCT % -> tcc fraction) -> append derived
        hub-height wind. ``lat2d``/``lon2d`` are numpy (for the solar gate)."""
        x = x * self.out_scale + self.out_center
        # Invert per-channel transforms (exact inverses from channel_transforms):
        #   log_eps:  log(1 + x/eps)        -> eps*(exp(y) - 1)
        #   asinh:    arcsinh(x/eps)        -> eps*sinh(y)              (signed)
        #   logit_eps[_percent]: logit(eps + (1-2eps)*(x/scale))
        #                                   -> ((sigmoid(y)-eps)/(1-2eps))*scale
        for idx, eps in self._log_eps_idx:
            x[:, idx] = eps * torch.expm1(x[:, idx])
        for idx, eps in self._asinh_idx:
            x[:, idx] = eps * torch.sinh(x[:, idx])
        for idx, eps, scale in self._logit_idx:
            x[:, idx] = ((torch.sigmoid(x[:, idx]) - eps) / (1.0 - 2.0 * eps)) * scale

        if self.physical_clamp:
            # Transform round-off guard (no-op for the physical regression, whose
            # transform lists are empty; keeps diffusion inverse-transforms in range).
            for idx, _eps in self._log_eps_idx:
                x[:, idx].clamp_(min=0.0)
            for idx, _eps, scale in self._logit_idx:
                x[:, idx].clamp_(min=0.0, max=scale)
            # asinh channels (heat fluxes) are signed — no nonnegativity clamp.
            # Physical-space metadata bounds + the solar gate (a dawn/dusk ramp on
            # shortwave), applied identically in both modes. No-op if unset.
            self._apply_constraints(x, valid_time, lat2d, lon2d)
        # Unit rescale to canonical lexicon units (after constraints): CLCT % -> tcc fraction.
        for idx, scale in self._output_unit_scale:
            x[:, idx] = x[:, idx] * scale
        # Derived hub-height wind components (u{H}m, v{H}m), appended after the
        # trained channels.
        if self._hub_heights:
            x = self._derive_hub_wind(x)
        return x

    def _snap_to_patch(self, lo: int, hi: int, n: int) -> tuple[int, int]:
        """Snap a run-window extent ``[lo, hi)`` (one axis) to a multiple of
        ``_patch_size``. Called by :meth:`set_domain` per axis: the DiT patch
        detokenizer requires the run grid's extent to be a multiple of
        ``patch_size`` (an odd extent would floor to extent-1 and mismatch the
        output grid).

        Grows ``[lo, hi)`` OUTWARD into the surrounding grid to the next multiple
        of ``p`` -- real neighbouring cells, bounded by ``[0, n)`` (``n`` = grid
        dim); ``set_domain`` folds this growth into the halo trim so the reported
        bbox is unchanged. ``hi`` grows first, then ``lo``. It only shrinks if the
        grid dim ``n`` itself isn't a multiple of ``p`` and the window already
        spans it (no room to grow) -- which would yield a window smaller than the
        requested bbox, so the caller raises on the resulting negative trim
        (unreachable as shipped: all grid dims are even and ``p == 2``). No-op when
        ``_patch_size == 1`` (synthetic test instances)."""
        p = self._patch_size
        need = (-(hi - lo)) % p
        if need:
            add = min(need, n - hi)
            hi += add
            need -= add
            add = min(need, lo)
            lo -= add
            need -= add
            if need:  # grid dim not a multiple of p and no room -> shrink
                hi -= (hi - lo) % p
        return lo, hi

    @staticmethod
    def _rebind_latent(dit: torch.nn.Module, H: int, W: int) -> None:
        """Rebind a RoPE/NATTEN DiT to an (H, W) output domain (its construction
        grid was the training patch). Two pieces of per-grid metadata change -- the
        attention's latent grid and the detokenizer's patch counts -- and nothing
        else: no learnable weight depends on the grid, so this is all that differs
        per resolution. Shared by the regression forward and the diffusion sampler.
        Mutates ``dit`` in place, so callers sharing one network (e.g. sub-domains
        from ``set_domain``) must not run concurrently -- see ``set_domain``.
        """
        ph, pw = dit.tokenizer.patch_size
        latent_hw = (H // ph, W // pw)  # pixel grid -> latent (post-patchify) grid
        # (1) attention: NATTEN neighbour windows + the RoPE tables are built per
        # latent grid, so the attention layers need the new latent_hw.
        dit.attn_kwargs_forward["latent_hw"] = latent_hw
        # (2) detokenizer: the token->pixel reshape uses these patch counts. They
        # live on the detokenizer, or on its ``.proj`` for the ConvDetokenizer
        # wrapper -- rebind whichever holds them.
        detok = dit.detokenizer
        target = detok.proj if hasattr(detok, "proj") else detok
        target.h_patches, target.w_patches = latent_hw

    def _inference_context(self) -> AbstractContextManager:
        """Context manager wrapping the network forward passes.

        Returns :func:`contextlib.nullcontext` (full precision) by default, or a
        bf16 ``torch.autocast`` context when ``amp`` is enabled. The sampler
        bookkeeping in :meth:`_denoise` stays in fp64 regardless — autocast only
        affects the network's own float32 ops.
        """
        if self.amp:
            return torch.autocast(
                device_type=self.lat_output_grid.device.type, dtype=torch.bfloat16
            )
        return nullcontext()

    def _regression_forward(self, background: torch.Tensor) -> torch.Tensor:
        """Regression forward -> [1, n_out, H, W] (normalized space; de-normalized
        in postprocess_output, as for the diffusion path).

        The mean is a DiT-RoPE net: a single full-domain forward. LayerNorm is
        per-token (no spatial reduction) and RoPE/NATTEN are local, so it runs at
        any crop size (the resolution is fixed); only the latent reshape metadata
        is rebound per grid (constant t=0 is applied inside the net).
        """
        H, W = background.shape[-2:]
        self._rebind_latent(self.regression_model, H, W)
        bg = background.to(torch.float32)
        with self._inference_context():
            # Regression net is the bare DiT (not EDM-wrapped). Its forward is
            # net(x, t, condition): x = bg (the conditioning enters as the input
            # channels), t = a dummy 0 (no diffusion noise level in a regression),
            # condition = None (no separate vector conditioning).
            return self.regression_model(
                bg, bg.new_zeros(bg.shape[0]), condition=None
            ).float()

    def _denoise(self, background: torch.Tensor, seed: int | None) -> torch.Tensor:
        """One diffusion sample: EDM deterministic sampler over the DiT-RoPE
        conditioned on ``background`` -> [1, n_out, H, W] (normalized space;
        de-normalized + inverse-transformed in postprocess_output, as for the
        regression path).

        Full-domain single-forward — the DiT's NATTEN neighborhood attention +
        axial RoPE run at any crop size (the resolution is fixed). This is the
        single overridable seam for per-step cross-domain blending.
        """
        net = self.diffusion_model
        dev = background.device
        H, W = background.shape[-2:]
        cond = background.to(torch.float32)  # ConcatConditionWrapper cond_concat

        # Rebind the DiT latent grid to this domain (the construction grid was the
        # training patch size); the RoPE cos/sin tables rebuild for the new
        # latent_hw inside attention.
        self._rebind_latent(net.model.model, H, W)
        gen = (
            torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        )
        latents = torch.randn(
            (1, len(self.output_variables), H, W),
            device=dev,
            dtype=torch.float32,
            generator=gen,
        )

        # Deterministic EDM/Karras sampling via physicsnemo's standard sampler.
        # The preconditioner ``net(x, sigma, condition)`` returns the denoised (x0)
        # estimate; with the EDM schedule sigma(t)=t it is the x0-predictor.
        # ``EDMNoiseScheduler`` builds the Karras timesteps (sigma_min/sigma_max/rho,
        # appending the terminal t=0 step) and ``sample`` runs ``number_of_steps`` of
        # the chosen ODE solver (heun 2nd-order / euler 1st-order). Initial latents
        # are scaled to sigma_max; bookkeeping is fp64 (net forwards stay fp32).
        scheduler = EDMNoiseScheduler(
            sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
        )

        def x0_predictor(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return net(
                x.float(), t.to(torch.float32).reshape(-1), condition=cond
            ).double()

        denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
        with self._inference_context():
            out = sample(
                denoiser,
                latents.double() * self.sigma_max,
                scheduler,
                num_steps=self.number_of_steps,
                solver=self.solver,
            )
        return out.float()

    @torch.inference_mode()
    def _forward(
        self,
        era5: torch.Tensor,
        valid_time: datetime,
        lat2d: torch.Tensor,
        lon2d: torch.Tensor,
    ) -> torch.Tensor:
        """Forward for one ERA5 frame -> [number_of_samples, C_out, H, W] physical
        (C_out = the trained outputs, plus the derived hub-height wind components
        when ``hub_heights`` is set).

        With a halo crop the model runs on its own (expanded) grid and the result
        is trimmed back to the requested bbox — so the kept interior never sees the
        boundary halo. ``lat2d``/``lon2d`` (the trimmed output grid) are ignored in
        that case in favor of the stored expanded grid.
        """
        top, bot, left, right = self._halo
        if top or bot or left or right:
            lat2d, lon2d = self.lat_output_grid, self.lon_output_grid
        background = self.preprocess_input(era5, valid_time, lat2d, lon2d)
        lat_np, lon_np = lat2d.cpu().numpy(), lon2d.cpu().numpy()

        if self.mode == "mean":
            # channels_last suits the regression net's conv detokenizer
            # (proj_reshape_2d_conv); the diffusion net uses the plain reshape
            # detokenizer, so it is left in the default layout.
            background = background.to(memory_format=torch.channels_last)
            out = self.postprocess_output(
                self._regression_forward(background), valid_time, lat_np, lon_np
            )
            # Deterministic: broadcast over the (size 1..N) sample dim.
            if self.number_of_samples > 1:
                out = out.expand(self.number_of_samples, -1, -1, -1)
        else:
            # diffusion: draw N independent samples (different noise per sample)
            out = torch.cat(
                [
                    self.postprocess_output(
                        self._denoise(
                            background, None if self.seed is None else self.seed + i
                        ),
                        valid_time,
                        lat_np,
                        lon_np,
                    )
                    for i in range(self.number_of_samples)
                ],
                dim=0,
            )
        if top or bot or left or right:  # trim the halo border, leaving the bbox
            out = out[..., top : out.shape[-2] - bot, left : out.shape[-1] - right]
        return out

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run the model. ``x`` is [batch, time, variable, lat, lon]; ``coords``
        carries a ``time`` axis (validity times) driving the solar-zenith channel."""
        output_coords = self.output_coords(coords)

        lat2d_np = np.asarray(output_coords["lat"])
        lon2d_np = np.asarray(output_coords["lon"])
        self._check_bounds(lat2d_np, lon2d_np)
        lat2d = torch.as_tensor(lat2d_np, device=x.device, dtype=torch.float32)
        lon2d = torch.as_tensor(lon2d_np, device=x.device, dtype=torch.float32)

        valid_times = timearray_to_datetime(np.asarray(output_coords["time"]))

        # out shape: [batch, sample, time, variable, H, W] (H,W explicit — NOT
        # len() over the 2D lat/lon arrays).
        H, W = lat2d.shape
        out = torch.zeros(
            (
                output_coords["batch"].shape[0],
                self.number_of_samples,
                len(valid_times),
                len(self._output_coord_variables),  # trained outputs + u/v{H}m
                H,
                W,
            ),
            device=x.device,
            dtype=torch.float32,
        )
        for b in range(out.shape[0]):
            for t in range(out.shape[2]):
                out[b, :, t] = self._forward(x[b, t], valid_times[t], lat2d, lon2d)
        return out, output_coords

    def to(self, device: torch.device) -> "CorrDiffCosmoEra5":
        """Move the model to a device (the active regression/diffusion sub-network
        is a registered submodule, so ``super().to`` moves it too).

        Parameters
        ----------
        device : torch.device
            Device to move the model to
        """
        return super().to(device)

    # ── package loading ──────────────────────────────────────────────────────

    def set_domain(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        margin_deg: float = 1.0,
        halo: int = 0,
    ) -> "CorrDiffCosmoEra5":
        """Restrict the model to a sub-domain for a lat/lon bounding box.

        Returns a NEW :class:`CorrDiffCosmoEra5` that shares the loaded network(s)
        but whose target grid + static invariants are sliced to the smallest
        block of the (rotated) grid covering ``[lat_min, lat_max] x [lon_min,
        lon_max]``. The returned instance is a fixed-domain model with its own
        ``input_coords``/``output_coords``, so it composes with the standard run
        pipelines. The ERA5 input grid is recomputed to cover it (+ ``margin_deg``).

        The network is shared by reference and each forward rebinds its latent-grid
        state on it in place, so the parent and its sub-domains are safe to run
        sequentially but NOT concurrently against the same network (two threads or
        async tasks would overwrite each other's grid binding -> wrong-grid output);
        run concurrent domains in separate processes.

        Three-tier coverage (when the package ships extended invariants):

        * bbox inside the **native** trained footprint -> proceed (validated);
        * bbox reaching into the **extended margin** (invariants exist there but
          it is outside the trained footprint) -> proceed + a one-time OOD warning;
        * bbox beyond the **extended extent** -> ``ValueError`` (no invariants).

        Without extended invariants only the native footprint is supported. A
        returned sub-domain keeps only its own (cropped) grid, not the full or
        extended one, so call ``set_domain`` once on the full loaded model: a
        second ``set_domain`` on a returned sub-domain cannot reach back to the
        full or extended footprint (a bbox outside the sub-domain's grid raises).
        For several sub-regions, call ``set_domain`` multiple times on the full
        model instead.

        ``halo`` (px, default 0 = off): run on a block expanded by ``halo`` real
        cells per side and trim it off the output, keeping the returned bbox
        interior clear of the DiT's boundary artifact (~32 px); clamps + warns at
        the grid edge. Both models are DiT-RoPE (crop-size agnostic at the fixed
        resolution), so any size runs in a single forward — but the run grid is
        snapped to a multiple of
        the DiT ``patch_size`` (an odd extent would floor to extent-1) and must be
        at least ``attn_kernel*patch`` cells per side (NATTEN must fit the latent),
        so very small bboxes raise.
        """
        # Slice from the extended grid when available (it contains the native
        # footprint as a sub-block); otherwise the native grid only.
        ext = self._ext_lat_numpy is not None
        latg = self._ext_lat_numpy if ext else self.lat_output_numpy
        long_ = self._ext_lon_numpy if ext else self.lon_output_numpy

        # All four corners must lie inside the available footprint (a partially
        # outside bbox would silently clip to a wrong region — refuse instead).
        corner_lat = np.array([lat_min, lat_min, lat_max, lat_max])
        corner_lon = np.array([lon_min, lon_max, lon_min, lon_max])
        covered = _points_in_grid_footprint(corner_lat, corner_lon, latg, long_)
        if not covered.all():
            raise ValueError(
                f"domain lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] is not "
                f"fully inside the {'extended' if ext else 'native'} footprint "
                f"(corners outside: {int((~covered).sum())}/4) — no invariants there."
            )
        # OOD warning when the bbox reaches past the validated native footprint.
        if (
            ext
            and not _points_in_grid_footprint(
                corner_lat, corner_lon, self.lat_output_numpy, self.lon_output_numpy
            ).all()
        ):
            warnings.warn(
                "domain extends beyond the validated COSMO-REA footprint into the "
                "extended margin: invariants exist there but the model is "
                "out-of-distribution (skill unvalidated).",
                stacklevel=2,
            )

        inside = (
            (latg >= lat_min)
            & (latg <= lat_max)
            & (long_ >= lon_min)
            & (long_ <= lon_max)
        )
        if not inside.any():
            raise ValueError(
                f"bbox lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] selects no "
                "grid cells (smaller than one grid cell?); widen the bbox."
            )
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        i0, i1 = int(rows[0]), int(rows[-1]) + 1
        j0, j1 = int(cols[0]), int(cols[-1]) + 1

        # Halo: expand the block by `halo` px, clamped to the grid edge (real
        # surrounding data); the border is run then trimmed in _forward.
        Hn, Wn = latg.shape
        i0e, i1e = max(0, i0 - halo), min(Hn, i1 + halo)
        j0e, j1e = max(0, j0 - halo), min(Wn, j1 + halo)
        if halo and (
            i0 - i0e < halo or i1e - i1 < halo or j0 - j0e < halo or j1e - j1 < halo
        ):
            warnings.warn(
                f"halo={halo} clamped at the grid edge; a residual edge artifact "
                "may remain there.",
                stacklevel=2,
            )
        # Snap the RUN grid to a multiple of patch_size (the DiT detokenizer needs
        # it; an odd extent would floor to extent-1 and mismatch the output). The
        # extra cells are real data, trimmed back via the halo so the reported
        # bbox stays exact.
        i0e, i1e = self._snap_to_patch(i0e, i1e, Hn)
        j0e, j1e = self._snap_to_patch(j0e, j1e, Wn)
        if min(i1e - i0e, j1e - j0e) < self._min_domain_cells:
            raise ValueError(
                f"domain {i1e - i0e}x{j1e - j0e} is below the "
                f"{self._min_domain_cells}-cell per-side minimum (the DiT's NATTEN "
                "kernel must fit the latent grid); widen the bbox."
            )
        halo_trim = (i0 - i0e, i1e - i1, j0 - j0e, j1e - j1)  # top, bottom, left, right
        # top/left can't go negative (snap only grows or holds the low edge); a
        # negative bottom/right means the patch-snap shrank the run window below the
        # requested bbox (grid dim not a multiple of patch_size) -> fail loudly
        # rather than silently return a smaller-than-requested domain.
        if halo_trim[1] < 0 or halo_trim[3] < 0:
            raise ValueError(
                f"grid dimension not divisible by patch_size={self._patch_size}: "
                "snapping the run window shrank it below the requested bbox. Provide "
                "a grid whose dimensions are multiples of patch_size."
            )
        i0, i1, j0, j1 = i0e, i1e, j0e, j1e

        dev = self.lat_output_grid.device
        if ext:
            lat_out = torch.as_tensor(
                latg[i0:i1, j0:j1], device=dev, dtype=torch.float32
            )
            lon_out = torch.as_tensor(
                long_[i0:i1, j0:j1], device=dev, dtype=torch.float32
            )
            static = OrderedDict(
                (
                    name,
                    torch.as_tensor(
                        self._ext_static_numpy[k, i0:i1, j0:j1],
                        device=dev,
                        dtype=torch.float32,
                    ),
                )
                for k, name in enumerate(self._static_names)
            )
        else:
            lat_out = self.lat_output_grid[i0:i1, j0:j1].clone()
            lon_out = self.lon_output_grid[i0:i1, j0:j1].clone()
            static = OrderedDict(
                (name, self.static_invariants[k, i0:i1, j0:j1].clone())
                for k, name in enumerate(self._static_names)
            )
        lo, ln = lat_out.cpu().numpy(), lon_out.cpu().numpy()
        # Rebuild the ERA5 input axes at the package's own (validated-uniform) input
        # spacing rather than a hardcoded value, so a non-0.25-deg package works too.
        dlat = float(self.lat_input_numpy[1] - self.lat_input_numpy[0])
        dlon = float(self.lon_input_numpy[1] - self.lon_input_numpy[0])

        def _reg(a0: float, a1: float, step: float) -> torch.Tensor:
            """Regular axis at the input-grid spacing, spanning [a0, a1] + margin_deg."""
            return torch.arange(
                float(np.floor(a0) - margin_deg),
                float(np.ceil(a1) + margin_deg + step),
                step,
                dtype=torch.float32,
                device=dev,
            )

        sub = CorrDiffCosmoEra5(
            era5_variables=self.era5_variables,
            output_variables=self.output_variables,
            regression_model=self.regression_model,
            diffusion_model=self.diffusion_model,
            resolution=self.resolution,
            mode=self.mode,
            lat_input_grid=_reg(lo.min(), lo.max(), dlat),
            lon_input_grid=_reg(ln.min(), ln.max(), dlon),
            lat_output_grid=lat_out,
            lon_output_grid=lon_out,
            era5_center=self.era5_center.flatten(),
            era5_scale=self.era5_scale.flatten(),
            out_center=self.out_center.flatten(),
            out_scale=self.out_scale.flatten(),
            static_invariants=static,
            pre_invariant_variables=self.pre_invariant_variables,
            post_invariant_variables=self.post_invariant_variables,
            channel_transforms=self._channel_transforms,
            constraints=self._constraints,
            number_of_samples=self.number_of_samples,
            physical_clamp=self.physical_clamp,
            number_of_steps=self.number_of_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            solver=self.solver,
            seed=self.seed,
            amp=self.amp,
            hub_heights=(self._hub_heights or None),
            hub_interp=self._hub_interp,
            wind_levels=self._wind_levels,
        )
        sub._halo = halo_trim
        sub.check_inputs = self.check_inputs
        sub._patch_size = self._patch_size
        sub._min_domain_cells = self._min_domain_cells
        return sub

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained COSMO-REA downscaling package (combined rea6/rea2).

        Returns the hosted package once :data:`DEFAULT_PACKAGE_URI` is set; pick
        the model within it via ``load_model(..., mode=, resolution=)`` (or rely
        on the ``mean``/``rea6`` defaults through ``from_pretrained``). Until
        weight hosting + license clearance land, ``DEFAULT_PACKAGE_URI`` is
        ``None`` and this raises ``NotImplementedError`` — build a package
        locally and pass its path to ``load_model``/``from_pretrained`` instead.
        """
        if DEFAULT_PACKAGE_URI is None:
            raise NotImplementedError(
                "No default COSMO-REA package is hosted yet (weight hosting + "
                "license clearance pending). Build a package and pass its path "
                "to load_model() or from_pretrained('<path>'). When hosting "
                "lands, set DEFAULT_PACKAGE_URI in "
                "earth2studio.models.dx.corrdiff_cosmo_era5.py."
            )
        return Package(
            DEFAULT_PACKAGE_URI,
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_cosmo_era5"),
                "same_names": True,
            },
        )

    @staticmethod
    def _load_json(package: Package, filename: str) -> dict:
        """Resolve and parse a JSON file from the package."""
        with open(package.resolve(filename), encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"{filename} is empty")
        return json.loads(content)

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        device: str | None = None,
        mode: Literal["mean", "diffusion"] = "mean",
        resolution: Literal["rea6", "rea2"] = "rea6",
        hub_heights: Sequence[float] | None = None,
        hub_interp: Literal["linear", "log"] = "linear",
    ) -> DiagnosticModel:
        """Load a COSMO-REA downscaling model from a package.

        ``mode`` ({"mean", "diffusion"}) and ``resolution`` ({"rea6", "rea2"})
        select the checkpoint within one package. All other
        configuration (sampler, normalization, transforms) comes from the package
        metadata.

        ``hub_heights``/``hub_interp`` opt into derived hub-height wind components
        (``u{H}m``/``v{H}m``; compose ``DerivedWS`` for speed); they require the
        package to carry a ``wind_levels`` metadata block (else requesting
        ``hub_heights`` raises). See the constructor.
        """
        # Validate the selectors up front so a bad value fails with a clear message
        # rather than a cryptic missing-file when resolving "<resolution>/...".
        if resolution not in SUPPORTED_VARIANTS:
            raise ValueError(f"resolution must be one of {list(SUPPORTED_VARIANTS)}")
        if mode not in ("mean", "diffusion"):
            raise ValueError(f"mode must be 'mean' or 'diffusion' (got {mode!r}).")

        # Resolve the package-root config.json so HuggingFace records the download
        # (its content is not used here); tolerate its absence.
        try:
            package.resolve("config.json")
        except (FileNotFoundError, ValueError):
            pass

        # Resolution-subfolder layout: the package nests each resolution under
        # its own subfolder (``rea6/``, ``rea2/``) so all four models share one
        # package path and ``resolution`` selects the subfolder. Every artifact
        # path below is resolved through ``prefix``.
        prefix = f"{resolution}/"
        metadata = cls._load_json(package, prefix + "metadata.json")
        era5_variables = metadata["era5_variables"]
        output_variables = metadata["output_variables"]
        pre_invariant_variables = metadata["pre_invariant_variables"]
        post_invariant_variables = metadata.get("post_invariant_variables", [])
        # Per-mode output representation (self-describing; falls back to the older
        # single-representation behaviour for packages without a "modes" block).
        modes = metadata.get("modes", {})
        if modes and mode not in modes:
            raise ValueError(
                f"package metadata has a 'modes' block but no entry for mode={mode!r} "
                f"(available: {list(modes)})"
            )
        mode_meta = modes.get(mode, {})
        out_stats_key = mode_meta.get("stats", "output")
        channel_transforms = (
            metadata.get("channel_transforms", {})
            if mode_meta.get("channel_transforms", True)
            else {}
        )
        # Constraints are applied in postprocess in PHYSICAL space (after de-norm
        # AND any inverse transforms), where they are mode-independent: the de-norm
        # and inverse transforms are order-preserving, so a min/max CLAMP on the
        # physical field matches the training-time clamp; the solar gate is a [0,1]
        # multiply on the physical field.
        # So the FULL constraint set (bounds + solar gate) is applied identically to
        # BOTH products -- this both bounds the z-scored non-negative channels (ALWU_S/
        # ATHD_S/QV_2M/Q_L*) and keeps the diffusion ensemble mean consistent with
        # the regression mean through twilight (the gate is linear, so E[g*y]=g*E[y]).
        constraints = metadata.get("constraints") or {}
        number_of_samples = metadata.get("number_of_samples", 1)
        sampler = metadata.get("sampler", {})

        ckpt_key = f"{resolution}_{mode}"
        try:
            ckpt = metadata["checkpoints"][ckpt_key]
        except KeyError as e:
            raise ValueError(
                f"package metadata has no checkpoint entry {ckpt_key!r} "
                f"(available: {list(metadata.get('checkpoints', {}))})."
            ) from e
        regression_model = None
        diffusion_model = None
        # Load the network with physicsnemo's standard ``from_checkpoint``. The
        # package ships a bare ``DiT`` for the regression and an
        # ``EDMPreconditioner(ConcatConditionWrapper(DiT))`` for the diffusion, each
        # saved with its architecture (incl. the axial-2D-RoPE NATTEN backend), so
        # the module is reconstructed faithfully -- no manual rebuild needed.
        ckpt_path = package.resolve(prefix + ckpt)
        if mode == "mean":
            loader, label = DiT.from_checkpoint, "regression"
        else:
            loader, label = EDMPreconditioner.from_checkpoint, "diffusion"
        try:
            model = loader(ckpt_path)
        except Exception as e:
            raise ValueError(
                f"could not load the {label} network from {ckpt!r} (resolved "
                f"{ckpt_path}) via from_checkpoint: {e}. Expected a physicsnemo "
                f".mdlus saved by the package builder."
            ) from e
        model = model.eval()
        model.requires_grad_(False)
        if device is not None:
            model = model.to(device)
        if mode == "mean":
            regression_model = model
        else:
            diffusion_model = model

        stats = cls._load_json(package, prefix + "stats.json")
        miss_e = [v for v in era5_variables if v not in stats.get("era5", {})]
        miss_o = [v for v in output_variables if v not in stats.get(out_stats_key, {})]
        if miss_e or miss_o:
            raise ValueError(
                f"package stats.json (resolution/mode '{resolution}/{mode}') is "
                f"missing normalization stats for ERA5 vars {miss_e} and/or output "
                f"vars {miss_o}; the metadata variable lists disagree with stats.json."
            )
        era5_center = torch.tensor(
            [stats["era5"][v]["mean"] for v in era5_variables], device=device
        )
        era5_scale = torch.tensor(
            [stats["era5"][v]["std"] for v in era5_variables], device=device
        )
        out_center = torch.tensor(
            [stats[out_stats_key][v]["mean"] for v in output_variables], device=device
        )
        out_scale = torch.tensor(
            [stats[out_stats_key][v]["std"] for v in output_variables], device=device
        )
        # One-shot sanity check on the loaded normalization stats (runs once at
        # load, not per inference): a misaligned/garbage stats file silently
        # produces wrong-scaled inputs and garbage output. Scales must be finite
        # and strictly positive; centers finite.
        for nm, c, s in (
            ("era5", era5_center, era5_scale),
            ("output", out_center, out_scale),
        ):
            if not (
                torch.isfinite(c).all() and torch.isfinite(s).all() and (s > 0).all()
            ):
                raise ValueError(
                    f"{nm} normalization stats are invalid (non-finite, or std<=0) — "
                    f"check the package stats.json (resolution/mode '{resolution}/{mode}')."
                )

        # Native default grids: lat/lon_input are the 1D regular ERA5 source grid;
        # lat/lon_output are the 2D curvilinear (rotated-pole) COSMO-REA target.
        # (set_domain slices sub-grids from these, or from the extended grid below.)
        with xr.open_dataset(package.resolve(prefix + "grids.nc")) as ds:
            lat_input_grid = torch.as_tensor(
                np.asarray(ds["lat_input"]), device=device, dtype=torch.float32
            )
            lon_input_grid = torch.as_tensor(
                np.asarray(ds["lon_input"]), device=device, dtype=torch.float32
            )
            lat_output_grid = torch.as_tensor(
                np.asarray(ds["lat_output"]), device=device, dtype=torch.float32
            )
            lon_output_grid = torch.as_tensor(
                np.asarray(ds["lon_output"]), device=device, dtype=torch.float32
            )

        # Native static invariants, position channels excluded (recomputed from
        # the grid so they cannot drift). Preferred format: the extended PHYSICAL
        # file (`metadata["invariants"]`) — sample the native sub-block by index
        # (native_offset/native_shape) and z-score with the training norm-stats.
        # Falls back to the older pre-sliced/pre-normalized `invariants.nc`.
        inv_names = [
            n
            for n in (*pre_invariant_variables, *post_invariant_variables)
            if n not in POS_VARIABLES
        ]
        static_invariants: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Keep the full extended grid + invariants (CPU numpy) for set_domain() to
        # reach the OOD margin band beyond the native footprint. The native
        # invariants for the default output are sliced from the extended ones below
        # (native_offset/native_shape); the native output grid itself comes from
        # grids.nc above.
        ext_lat_np = ext_lon_np = ext_static_np = None
        inv_meta = metadata.get("invariants")
        if inv_meta and inv_meta.get("file"):
            i0, j0 = (int(v) for v in inv_meta["native_offset"])
            ny, nx = (int(v) for v in inv_meta["native_shape"])
            norm = cls._load_json(package, prefix + inv_meta["norm_stats_file"])[
                "channels"
            ]
            with xr.open_dataset(package.resolve(prefix + inv_meta["file"])) as ds:
                ext_lat_np = np.asarray(ds["lat"], dtype=np.float32)
                ext_lon_np = np.asarray(ds["lon"], dtype=np.float32)
                ext_stack = []
                for name in inv_names:
                    # A z-scored channel's name ends in "_norm" (e.g.
                    # "elevation_norm"), but the physical .nc file stores it under
                    # the bare name ("elevation"). Strip "_norm" to get the file's
                    # variable name.
                    phys = name[:-5] if name.endswith("_norm") else name
                    arr = np.asarray(ds[phys], dtype=np.float32)  # full extended grid
                    spec = norm.get(phys, {"method": "identity"})
                    if spec.get("method") == "zscore":
                        mean, std = float(spec["mean"]), float(spec["std"])
                        # Fail loud on a bad std rather than silently flooring it
                        # (which would inflate the channel) -- mirrors the stats.json
                        # std<=0 check.
                        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
                            raise ValueError(
                                f"invariant {phys!r} has invalid z-score stats "
                                f"(mean={mean}, std={std}) in "
                                f"{inv_meta['norm_stats_file']!r}."
                            )
                        arr = (arr - mean) / std
                    ext_stack.append(arr)
                    static_invariants[name] = torch.as_tensor(
                        arr[i0 : i0 + ny, j0 : j0 + nx],
                        device=device,
                        dtype=torch.float32,
                    )
                ext_static_np = np.stack(ext_stack) if ext_stack else None
        else:
            with xr.open_dataset(package.resolve(prefix + "invariants.nc")) as ds:
                for name in inv_names:
                    static_invariants[name] = torch.as_tensor(
                        np.asarray(ds[name]), device=device, dtype=torch.float32
                    )

        logger.info(
            f"Loaded CorrDiffCosmoEra5 resolution={resolution} mode={mode} "
            f"({len(output_variables)} output channels)"
        )
        model = cls(
            era5_variables=era5_variables,
            output_variables=output_variables,
            regression_model=regression_model,
            diffusion_model=diffusion_model,
            resolution=resolution,
            mode=mode,
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            era5_center=era5_center,
            era5_scale=era5_scale,
            out_center=out_center,
            out_scale=out_scale,
            static_invariants=static_invariants,
            pre_invariant_variables=pre_invariant_variables,
            post_invariant_variables=post_invariant_variables,
            channel_transforms=channel_transforms,
            constraints=constraints,
            number_of_samples=number_of_samples,
            number_of_steps=sampler.get("num_steps", 18),
            sigma_min=sampler.get("sigma_min", 0.002),
            sigma_max=sampler.get("sigma_max", 800.0),
            rho=sampler.get("rho", 7.0),
            solver=sampler.get("solver", "heun"),
            hub_heights=hub_heights,
            hub_interp=hub_interp,
            wind_levels=metadata.get("wind_levels"),
        )
        # Stash the extended grid + invariants on the model (see above).
        model._ext_lat_numpy = ext_lat_np
        model._ext_lon_numpy = ext_lon_np
        model._ext_static_numpy = ext_static_np
        arch = metadata["regression"] if mode == "mean" else metadata["diffusion"]
        model._patch_size = int(arch["patch_size"])
        model._min_domain_cells = int(arch["attn_kernel_size"]) * model._patch_size
        # Durable device placement: the wrapper's buffers are built piecemeal with
        # device=device above; a final super().to(device) over the whole module
        # (every registered buffer + the active sub-network, a registered submodule)
        # is the safety net against any future default-CPU buffer mismatching.
        if device is not None:
            model = model.to(device)
        return model
