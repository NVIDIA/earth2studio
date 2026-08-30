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

"""Autoregressive COSMO-REA prognostic model conditioned on ERA5.

Each step samples the COSMO-REA state at ``t + 1 h`` using the state at ``t`` and
ERA5 data at ``t + 1 h``. The conditioning input is the concatenated ERA5
background and previous state; there is no separate scalar condition.

The COSMO data processing follows
:class:`~earth2studio.models.dx.corrdiff_cosmo_era5.CorrDiffCosmoEra5`. The
one-step and rollout interfaces follow
:class:`~earth2studio.models.px.stormcastconus.StormCastCONUS`. Unlike the
downscaling model, this prognostic model also converts its COSMO-REA input from
physical space to model space.
"""

import json
import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterator, Sequence
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.lexicon import CosmoLexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim, handshake_size, interp
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, TimeArray

try:
    import natten  # noqa: F401  # the DiT needs NATTEN (neighborhood attention)
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("stormcast-europe")
    cos_zenith_angle = None


SUPPORTED_VARIANTS = ("rea6", "rea2")

# Position and solar-zenith channels are computed for each grid and valid time,
# rather than stored as static invariants.
POS_VARIABLES = ("sin_lat", "cos_lat", "sin_lon", "cos_lon")
COS_ZENITH_VARIABLE = "cos_zenith"

# Shortwave-radiation channels reduced to zero at night by the solar gate.
SHORTWAVE_VARIABLES = ("ASWDIR_S", "ASWDIFD_S")


def _points_in_grid_footprint(
    plat: np.ndarray, plon: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray
) -> np.ndarray:
    """Check whether geographic points lie inside a curvilinear grid footprint.

    Uses the grid's outer cell centers as a polygon boundary and checks whether each
    point lies inside it. The grid must not wrap from 180° to -180° longitude.
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
    for i in range(n):
        yi, yj = ring_lat[i], ring_lat[j]
        xi, xj = ring_lon[i], ring_lon[j]
        straddle = (yi > plat) != (yj > plat)
        x_int = (xj - xi) * (plat - yi) / (yj - yi + 1e-30) + xi
        inside ^= straddle & (plon < x_int)
        j = i
    return inside


def _interp_levels_to_height(
    values: torch.Tensor,
    level_heights: torch.Tensor,
    target: float,
    method: Literal["linear", "log"] = "linear",
) -> torch.Tensor:
    """Per-pixel vertical interpolation of a profile to one target height.

    ``values`` is ``[..., K, H, W]`` (ascending in geometric height along ``K``);
    ``level_heights`` is ``[K, H, W]`` (m above ground, ascending, terrain-following).
    ``method="linear"`` interpolates by height. ``method="log"`` interpolates using
    the logarithm of each height, treating heights below 1 m as 1 m. Both use the
    nearest level outside the profile instead of extrapolating.
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
class StormCastEurope(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Autoregressive COSMO-REA prognostic model conditioned on ERA5.

    Each step samples the COSMO-REA state at ``t + 1 h`` using the state at ``t`` and
    ERA5 data at ``t + 1 h``. ERA5 data are fetched from
    ``conditioning_data_source``.

    Parameters
    ----------
    state_variables : Sequence[str]
        COSMO-REA state variable names in network channel order. The model receives
        these variables at ``t`` and predicts the same variables at ``t + 1 h``.
    era5_variables : Sequence[str]
        ERA5 background variable names, in network order.
    diffusion_model : torch.nn.Module
        PhysicsNeMo ``EDMPreconditioner(ConcatConditionWrapper(DiT))`` diffusion
        model used to sample the state at ``t + 1 h``. It is conditioned on the ERA5
        background at ``t + 1 h`` and the COSMO-REA state at ``t``.
    resolution : {"rea6", "rea2"}
        Spatial resolution of the predicted COSMO-REA state: ``"rea6"`` (~6 km) or
        ``"rea2"`` (~2.2 km).
    lat_input_grid, lon_input_grid : torch.Tensor
        1D increasing regular ERA5 background grid (the ERA5 crop footprint).
    lat_output_grid, lon_output_grid : torch.Tensor
        2D curvilinear (rotated-pole) COSMO-REA state grid [rea_y, rea_x].
    era5_center, era5_scale : torch.Tensor
        ERA5 background normalization (mean/std), size [n_era5].
    state_center, state_scale : torch.Tensor
        Mean and standard deviation of each state channel after its channel
        transform, with size [n_state]. These values normalize the transformed state
        before it enters the model.
    static_invariants : OrderedDict[str, torch.Tensor]
        Static fields on the COSMO-REA output grid, already normalized for the model.
        Each has shape [rea_y, rea_x]. Examples include ``elevation_norm``,
        ``land_fraction``, and ``z0_lu``. Latitude and longitude position channels
        are computed separately.
    pre_invariant_variables, post_invariant_variables : Sequence[str]
        Names and order of the invariant channels placed before and after
        ``cos_zenith`` when constructing the model's background input. The order
        must match the network's training configuration.
    channel_transforms : dict | None
        Per-state-channel nonlinear transform spec (from package metadata). The
        forward transform maps physical -> model space; the inverse recovers
        physical after de-normalizing.
    conditioning_data_source : DataSource or ForecastSource or None
        Data source for the ERA5 background. Required for inference.
    number_of_steps : int
        Diffusion sampler step count (EDM). Defaults to 18.
    sigma_min, sigma_max, rho : float
        EDM noise-schedule bounds / exponent.
    solver : {"heun", "euler"}
        Diffusion ODE solver.
    physical_clamp : bool
        When enabled, clamp model outputs after converting them back to physical
        values. Channels using ``log_eps`` are limited to values >= 0, and channels
        using ``logit`` are limited to ``[0, scale]``. Other channels are unchanged.
    amp : bool
        Run the network forward under bf16 autocast.
    constraints : dict | None
        Physical bounds and shortwave solar gate. The solar gate smoothly reduces
        shortwave-radiation outputs to zero when the Sun is below the horizon.
        ``None`` permits unconstrained direct construction. :meth:`load_model`
        requires package constraints and applies them to every predicted state.
    hub_heights : Sequence[float], optional
        Heights (m above ground) at which to derive hub-height wind components
        ``u{H}m``/``v{H}m`` by vertically interpolating the wind levels. Off by
        default. The derived components are appended to ``output_coords`` and are
        not fed back into the rollout. Requires ``wind_levels``.
    hub_interp : {"linear", "log"}
        Method used to interpolate wind between model levels: ``"linear"``
        interpolates by height, while ``"log"`` interpolates by the natural
        logarithm of height.
    wind_levels : dict, optional
        Metadata describing the wind levels used for interpolation. Each level
        identifies its ``u`` and ``v`` state channels and the ``a`` and ``b``
        coefficients used to calculate its height above ground:
        ``height = a + b * elevation``. The metadata also names the static elevation
        channel, which defaults to ``elevation_norm``. Supplied by the package and
        required when ``hub_heights`` is set. See :meth:`_setup_hub_wind`.

    Notes
    -----
    The prognostic state passes through four representations:

    * public physical state, with canonical Earth2Studio names and units;
    * internal physical state, with checkpoint COSMO names and units;
    * channel-transformed state, using each channel's configured nonlinear transform
      or the identity transform;
    * model state, where the transformed values are z-score normalized.

    Inputs follow this order and outputs follow it in reverse. Physical constraints
    are applied in internal physical space before conversion to public units.

    :meth:`set_domain` restricts the model to a latitude-longitude sub-domain. It
    crops the grid and invariants from the extended grid when available and returns
    a model that rolls out on the selected domain.

    The ERA5 background is regridded onto the COSMO-REA state grid with
    ``latlon_interpolation_regular`` (as in
    :class:`~earth2studio.models.dx.corrdiff_cosmo_era5.CorrDiffCosmoEra5`), and
    :meth:`_check_bounds` requires the COSMO grid strictly inside the packaged ERA5
    grid, so the background is always interpolated from surrounding cells (never
    extrapolated).

    The model advances COSMO-REA regional reanalysis over Europe, forced by the
    ERA5 background at the target time:

    * COSMO-REA6 (~6 km), DWD: https://reanalysis.meteo.uni-bonn.de/?COSMO-REA6
    * COSMO-REA2 (~2.2 km), DWD: https://reanalysis.meteo.uni-bonn.de/?COSMO-REA2
    * ERA5, ECMWF: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5

    Badges
    ------
    region:eu class:nwc product:wind product:temp product:precip product:atmos year:2026 gpu:80gb
    """

    def __init__(
        self,
        state_variables: Sequence[str],
        era5_variables: Sequence[str],
        diffusion_model: torch.nn.Module,
        resolution: Literal["rea6", "rea2"],
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        era5_center: torch.Tensor,
        era5_scale: torch.Tensor,
        state_center: torch.Tensor,
        state_scale: torch.Tensor,
        static_invariants: "OrderedDict[str, torch.Tensor]",
        pre_invariant_variables: Sequence[str],
        post_invariant_variables: Sequence[str],
        channel_transforms: dict | None = None,
        conditioning_data_source: DataSource | ForecastSource | None = None,
        number_of_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800.0,
        rho: float = 7.0,
        solver: Literal["heun", "euler"] = "heun",
        physical_clamp: bool = True,
        amp: bool = False,
        constraints: dict | None = None,
        hub_heights: Sequence[float] | None = None,
        hub_interp: Literal["linear", "log"] = "linear",
        wind_levels: dict | None = None,
    ) -> None:
        super().__init__()

        if resolution not in SUPPORTED_VARIANTS:
            raise ValueError(f"resolution must be one of {list(SUPPORTED_VARIANTS)}")
        if number_of_steps < 2:
            raise ValueError(
                f"number_of_steps must be >= 2 for the EDM schedule (got "
                f"{number_of_steps}); the step/(n-1) ramp divides by zero at n=1."
            )
        if solver not in ("heun", "euler"):
            raise ValueError(f"solver must be 'heun' or 'euler', got {solver!r}.")
        if not (
            np.isfinite(sigma_min)
            and np.isfinite(sigma_max)
            and 0 < sigma_min < sigma_max
        ):
            raise ValueError(
                f"require finite 0 < sigma_min < sigma_max for the EDM schedule, got "
                f"sigma_min={sigma_min}, sigma_max={sigma_max}."
            )
        if not (np.isfinite(rho) and rho > 0):
            raise ValueError(f"rho must be finite and > 0, got {rho}.")

        self.resolution = resolution
        self.diffusion_model = diffusion_model

        self.state_variables = list(state_variables)
        self.era5_variables = list(era5_variables)
        self.pre_invariant_variables = list(pre_invariant_variables)
        self.post_invariant_variables = list(post_invariant_variables)
        if not self.pre_invariant_variables:
            raise ValueError(
                "pre_invariant_variables must be non-empty (it holds at least the "
                "sin/cos lat/lon position channels)."
            )
        # Each invariant name must identify one channel across both groups.
        all_inv = [*self.pre_invariant_variables, *self.post_invariant_variables]
        if len(set(all_inv)) != len(all_inv):
            dups = sorted({n for n in all_inv if all_inv.count(n) > 1})
            raise ValueError(
                f"duplicate invariant names {dups} in pre/post_invariant_variables; "
                "each invariant must appear exactly once."
            )
        # Position channels are generated from the output latitude and longitude for
        # each call. Their order must match the background-channel order used during
        # training.
        missing_pos = [
            p for p in POS_VARIABLES if p not in self.pre_invariant_variables
        ]
        if missing_pos:
            raise ValueError(
                f"pre_invariant_variables is missing position channels {missing_pos}; "
                f"all of {list(POS_VARIABLES)} must be present."
            )
        present_pos = [p for p in self.pre_invariant_variables if p in POS_VARIABLES]
        if present_pos != list(POS_VARIABLES):
            raise ValueError(
                f"position channels appear as {present_pos} but must be in the trained "
                f"order {list(POS_VARIABLES)} within pre_invariant_variables."
            )
        # Full ordered background channel list (must match the trained layout):
        #   [ ERA5 | pre-invariants (sin/cos lat/lon, elevation_norm, ...) |
        #     cos_zenith | post-invariants ]
        self.background_variables = (
            self.era5_variables
            + self.pre_invariant_variables
            + [COS_ZENITH_VARIABLE]
            + self.post_invariant_variables
        )

        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source was provided to StormCastEurope; set the "
                "conditioning_data_source attribute before running inference."
            )

        # Sampler config
        self.number_of_steps = number_of_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.physical_clamp = physical_clamp
        self.amp = amp

        # Grids
        self._validate_input_grid(lat_input_grid, lon_input_grid)
        # Output latitude and longitude must define one finite 2D curvilinear grid.
        if (
            lat_output_grid.ndim != 2
            or lon_output_grid.ndim != 2
            or lat_output_grid.numel() == 0
        ):
            raise ValueError(
                "lat_output_grid / lon_output_grid must be non-empty 2D curvilinear "
                f"grids, got shapes {tuple(lat_output_grid.shape)} / "
                f"{tuple(lon_output_grid.shape)}."
            )
        if lat_output_grid.shape != lon_output_grid.shape:
            raise ValueError(
                f"lat_output_grid {tuple(lat_output_grid.shape)} and lon_output_grid "
                f"{tuple(lon_output_grid.shape)} must have the same shape."
            )
        if not (
            torch.isfinite(lat_output_grid).all()
            and torch.isfinite(lon_output_grid).all()
        ):
            raise ValueError(
                "lat_output_grid / lon_output_grid contain non-finite values; the "
                "footprint check and ERA5 regridding require a finite output grid."
            )
        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)
        self.lat_input_numpy = lat_input_grid.cpu().numpy()
        self.lon_input_numpy = lon_input_grid.cpu().numpy()
        self.lat_output_numpy = lat_output_grid.cpu().numpy()
        self.lon_output_numpy = lon_output_grid.cpu().numpy()
        # Use integer rea_y and rea_x indices for the grid dimensions; geographic
        # latitude and longitude are stored as 2D arrays.
        self.rea_y = np.arange(lat_output_grid.shape[0])
        self.rea_x = np.arange(lat_output_grid.shape[1])

        # Normalization buffers
        n_era5 = len(self.era5_variables)
        n_state = len(self.state_variables)
        self.register_buffer("era5_center", era5_center.view(1, n_era5, 1, 1))
        self.register_buffer("era5_scale", era5_scale.view(1, n_era5, 1, 1))
        self.register_buffer("state_center", state_center.view(1, n_state, 1, 1))
        self.register_buffer("state_scale", state_scale.view(1, n_state, 1, 1))

        # Static invariants (already normalized by load_model); lookup by name.
        self._static_names = list(static_invariants.keys())
        # Require every named non-position invariant at construction time.
        required_static = [
            n
            for n in (*self.pre_invariant_variables, *self.post_invariant_variables)
            if n not in POS_VARIABLES
        ]
        missing_static = [n for n in required_static if n not in self._static_names]
        if missing_static:
            raise ValueError(
                f"static_invariants is missing non-position invariant channels "
                f"{missing_static}; they are listed in pre/post_invariant_variables "
                f"but absent from static_invariants (have: {self._static_names})."
            )
        # All static invariants must match the output-grid shape before they can be
        # stacked with the position channels.
        out_hw = tuple(lat_output_grid.shape)
        for nm, t in static_invariants.items():
            if tuple(t.shape) != out_hw:
                raise ValueError(
                    f"static invariant {nm!r} has shape {tuple(t.shape)} != the COSMO "
                    f"output grid {out_hw}."
                )
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

        # Physical bounds and the shortwave solar gate are applied when configured.
        self._constraints = constraints or {}
        self._parse_constraints(self._constraints)

        # Each COSMO output point must lie strictly inside the ERA5 input grid so
        # that it has surrounding ERA5 cells for interpolation. Otherwise, the
        # regridding routine would extrapolate.
        self._check_bounds()

        self.time_step = np.timedelta64(1, "h")

        # Keep checkpoint variable names and channel order internally to match the
        # network. At the public API boundary, convert names to canonical
        # Earth2Studio names and rescale CLCT and TOT_PRECIP without changing channel
        # order.
        lexicon_mappings = [CosmoLexicon.to_e2studio(v) for v in self.state_variables]
        self._public_variables = np.array([name for name, _ in lexicon_mappings])
        self._unit_scale = [
            (i, scale) for i, (_, scale) in enumerate(lexicon_mappings) if scale != 1.0
        ]
        if len(set(self._public_variables.tolist())) != len(self._public_variables):
            raise ValueError(
                "State variables map to duplicate Earth2Studio names "
                f"({list(self._public_variables)}). Public input variable names must "
                "be unique; check state_variables and CosmoLexicon."
            )
        self._n_state = len(self.state_variables)

        # Optional hub-height wind outputs. Package metadata defines the wind channels
        # and height coefficients used for interpolation. These derived values are
        # output only and are not used as input to the next forecast step.
        self._wind_levels = wind_levels
        self._hub_heights = [float(h) for h in (hub_heights or [])]
        self._hub_interp = hub_interp
        if self._hub_heights:
            if not all(np.isfinite(h) and h > 0 for h in self._hub_heights):
                raise ValueError(
                    f"hub_heights must be finite and positive (m above ground), got "
                    f"{self._hub_heights}"
                )
            if wind_levels is None:
                raise ValueError(
                    "hub_heights requested but this package has no 'wind_levels' "
                    "metadata; hub-height wind is only available for packages that "
                    "ship it."
                )
            if hub_interp not in ("linear", "log"):
                raise ValueError(
                    f"hub_interp must be 'linear' or 'log', got {hub_interp!r}"
                )
            self._setup_hub_wind(wind_levels)
        self._hub_labels = [
            f"{int(h) if float(h).is_integer() else h}m" for h in self._hub_heights
        ]
        # Distinct hub labels: duplicate/degenerate heights (e.g. [100, 100.0]) would
        # emit colliding output variables.
        if len(set(self._hub_labels)) != len(self._hub_labels):
            raise ValueError(
                f"hub_heights produce duplicate output labels {self._hub_labels}; "
                "each hub height must map to a distinct u{H}m/v{H}m name."
            )
        # Emit interpolated wind components (u{H}m, v{H}m). Wind speed can be
        # derived from these components.
        self._derived_variables = [
            f"{c}{lbl}" for lbl in self._hub_labels for c in ("u", "v")
        ]
        # A derived hub name must not collide with an existing public (state) output.
        collisions = sorted(set(self._derived_variables) & set(self._public_variables))
        if collisions:
            raise ValueError(
                f"derived hub-height wind outputs {collisions} collide with existing "
                "state variables; choose hub heights whose u{H}m/v{H}m names are "
                "unused."
            )
        # Output variables include the state and derived hub-height wind components.
        # Input variables contain only the state.
        self._output_variables = np.array(
            list(self._public_variables) + self._derived_variables
        )

        # Extended grids and invariants allow set_domain to use the package margin.
        # Patch and minimum sizes constrain the selected grid. The rollout uses the
        # exact sub-domain because an autoregressive model cannot trim a halo before
        # feeding the state into the next step.
        self._ext_lat_numpy: np.ndarray | None = None
        self._ext_lon_numpy: np.ndarray | None = None
        self._ext_static_numpy: np.ndarray | None = None
        self._patch_size = 1
        self._min_domain_cells = 0

    # ── validation / transforms ──────────────────────────────────────────────

    @staticmethod
    def _validate_input_grid(lat: torch.Tensor, lon: torch.Tensor) -> None:
        """Validate the regular ERA5 grid used by the interpolation routine."""
        for name, g in (("latitude", lat), ("longitude", lon)):
            if g.ndim != 1:
                raise ValueError(f"Input {name} grid must be 1D (regular grid).")
            if g.numel() < 2:
                raise ValueError(f"Input {name} grid must have at least 2 points.")
            if not torch.isfinite(g).all():
                raise ValueError(f"Input {name} grid must contain only finite values.")
            d = torch.diff(g)
            if not torch.all(d > 0):
                suffix = " (south to north)" if name == "latitude" else ""
                raise ValueError(f"Input {name} must be strictly increasing{suffix}.")
            if not torch.allclose(d, d[0], rtol=1e-3, atol=0.0):
                raise ValueError(f"Input {name} must be regularly spaced (uniform).")

    def _check_bounds(self) -> None:
        """Ensure each COSMO output point lies strictly inside the ERA5 input grid.

        This provides surrounding ERA5 cells for interpolation and prevents
        ``latlon_interpolation_regular`` from extrapolating.
        """
        lat0, lat1 = float(self.lat_input_numpy[0]), float(self.lat_input_numpy[-1])
        lon0, lon1 = float(self.lon_input_numpy[0]), float(self.lon_input_numpy[-1])
        lat_out, lon_out = self.lat_output_numpy, self.lon_output_numpy
        if lat_out.min() <= lat0 or lat_out.max() >= lat1:
            raise ValueError(
                f"COSMO output latitude [{lat_out.min():.2f}, {lat_out.max():.2f}] is "
                f"not strictly inside the ERA5 input grid [{lat0:.2f}, {lat1:.2f}]."
            )
        if lon_out.min() <= lon0 or lon_out.max() >= lon1:
            raise ValueError(
                f"COSMO output longitude [{lon_out.min():.2f}, {lon_out.max():.2f}] is "
                f"not strictly inside the ERA5 input grid [{lon0:.2f}, {lon1:.2f}]."
            )

    def _parse_channel_transforms(self, channel_transforms: dict) -> None:
        """Pre-compute per-channel transform index lists (forward + inverse).

        Forward transforms (physical -> model space):
          ``log_eps``:            x -> log(1 + x/eps)      inverse eps*(exp(y)-1)
          ``asinh``:              x -> arcsinh(x/eps)      inverse eps*sinh(y)
          ``logit_eps[_percent]``: x -> logit(eps + (1-2eps)*(x/scale))
                                   inverse ((sigmoid(y)-eps)/(1-2eps))*scale
        """

        def _spec(ch: str) -> tuple[str, float, float]:
            t = channel_transforms.get(ch, "")
            if isinstance(t, dict):
                return (
                    str(t.get("transform", "")).lower(),
                    float(t.get("eps", 1e-5)),
                    float(t.get("scale", 1.0)),
                )
            if t:
                raise ValueError(
                    f"channel_transforms[{ch!r}] must be a dictionary of transform "
                    f"settings; got bare string {t!r}."
                )
            return "", 1e-5, 1.0

        self._log_eps_idx: list[tuple[int, float]] = []
        self._asinh_idx: list[tuple[int, float]] = []
        self._logit_idx: list[tuple[int, float, float]] = []
        for i, ch in enumerate(self.state_variables):
            name, eps, scale = _spec(ch)
            if not name:
                continue
            elif name == "log_eps":
                if not (np.isfinite(eps) and eps > 0):
                    raise ValueError(
                        f"channel_transforms[{ch!r}].eps must be finite and > 0 "
                        f"for log_eps (got {eps})."
                    )
                self._log_eps_idx.append((i, eps))
            elif name == "asinh":
                if not (np.isfinite(eps) and eps > 0):
                    raise ValueError(
                        f"channel_transforms[{ch!r}].eps must be finite and > 0 "
                        f"for asinh (got {eps})."
                    )
                self._asinh_idx.append((i, eps))
            elif name in ("logit_eps", "logit_eps_percent"):
                if not (np.isfinite(eps) and 0 < eps < 0.5):
                    raise ValueError(
                        f"channel_transforms[{ch!r}].eps must be finite and satisfy "
                        f"0 < eps < 0.5 for {name} (got {eps})."
                    )
                effective_scale = (
                    (scale if scale != 1.0 else 100.0)
                    if name == "logit_eps_percent"
                    else scale
                )
                if not (np.isfinite(effective_scale) and effective_scale > 0):
                    raise ValueError(
                        f"channel_transforms[{ch!r}].scale must be finite and > 0 "
                        f"for {name} (got {effective_scale})."
                    )
                self._logit_idx.append((i, eps, effective_scale))
            else:
                raise ValueError(
                    f"channel_transforms[{ch!r}] has unsupported transform {name!r}; "
                    "expected one of: log_eps, asinh, logit_eps[_percent] (or none)."
                )

    def _state_to_model(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a physical COSMO state to its normalized model representation.

        Applies each configured channel transform, followed by z-score normalization.
        ``x`` has shape [batch, n_state, H, W].
        """
        x = x.clone()
        for idx, eps in self._log_eps_idx:
            x[:, idx] = torch.log1p(x[:, idx].clamp_min(0.0) / eps)
        for idx, eps in self._asinh_idx:
            x[:, idx] = torch.asinh(x[:, idx] / eps)
        for idx, eps, scale in self._logit_idx:
            u = (x[:, idx] / scale).clamp(0.0, 1.0)
            p = eps + (1.0 - 2.0 * eps) * u
            x[:, idx] = torch.logit(p)
        return (x - self.state_center) / self.state_scale

    def _state_from_model(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a normalized model state to physical COSMO values.

        Reverses z-score normalization and channel transforms, then optionally clamps
        bounded channels to their physical ranges.
        """
        x = x * self.state_scale + self.state_center
        for idx, eps in self._log_eps_idx:
            x[:, idx] = eps * torch.expm1(x[:, idx])
        for idx, eps in self._asinh_idx:
            x[:, idx] = eps * torch.sinh(x[:, idx])
        for idx, eps, scale in self._logit_idx:
            x[:, idx] = ((torch.sigmoid(x[:, idx]) - eps) / (1.0 - 2.0 * eps)) * scale
        if self.physical_clamp:
            for idx, _eps in self._log_eps_idx:
                x[:, idx].clamp_(min=0.0)
            for idx, _eps, scale in self._logit_idx:
                x[:, idx].clamp_(min=0.0, max=scale)
            # asinh channels may be signed and are not clamped.
        return x

    # ── physical constraints ─────────────────────────────────────────────────

    def _parse_constraints(self, constraints: dict) -> None:
        """Validate constraints and map their channel names to state indices.

        ``bounds`` defines optional physical minimum and maximum values per channel.
        ``sza_gate`` defines the solar threshold and transition width for each
        shortwave-radiation channel. Entries for channels absent from
        ``state_variables`` are ignored.
        """
        idx = {ch: i for i, ch in enumerate(self.state_variables)}
        self._bound_lo: list[tuple[int, float]] = []
        self._bound_up: list[tuple[int, float]] = []
        for ch, b in (constraints.get("bounds") or {}).items():
            if ch not in idx:
                continue  # entries for channels absent from the state are ignored
            if not isinstance(b, dict):
                raise ValueError(
                    f"constraints.bounds[{ch!r}] must be a dict of bound settings, "
                    f"got {b!r}."
                )
            mode = str(b.get("mode", "")).lower()
            if mode == "sigmoid":
                raise NotImplementedError(
                    f"constraints.bounds[{ch!r}].mode='sigmoid' is unsupported as "
                    "inference post-processing; use 'clamp'."
                )
            if mode not in ("", "clamp", "relu"):
                raise ValueError(
                    f"constraints.bounds[{ch!r}].mode={mode!r} is unsupported; use "
                    "'clamp' or 'relu' (both apply a hard clamp), or omit it."
                )
            # Reject non-finite or reversed bounds before rollout.
            lo = float(b["min"]) if b.get("min") is not None else None
            up = float(b["max"]) if b.get("max") is not None else None
            if lo is not None and not np.isfinite(lo):
                raise ValueError(
                    f"constraints.bounds[{ch!r}].min is not finite ({lo})."
                )
            if up is not None and not np.isfinite(up):
                raise ValueError(
                    f"constraints.bounds[{ch!r}].max is not finite ({up})."
                )
            if lo is not None and up is not None and lo > up:
                raise ValueError(
                    f"constraints.bounds[{ch!r}] has min ({lo}) > max ({up}); "
                    "the valid interval is empty."
                )
            if lo is not None:
                self._bound_lo.append((idx[ch], lo))
            if up is not None:
                self._bound_up.append((idx[ch], up))
        self._sza_gate: list[tuple[int, float, float]] = []
        sz = constraints.get("sza_gate") or {}
        _dhw = sz.get("half_width")
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
            # Non-finite values would produce an invalid gate.
            if not np.isfinite(th):
                raise ValueError(f"sza_gate threshold must be finite ({ch}: {th})")
            if not (np.isfinite(hw) and hw > 0):
                raise ValueError(
                    f"sza_gate half_width must be finite and > 0 ({ch}: {hw})"
                )
            self._sza_gate.append((idx[ch], th, hw))
        gated = {self.state_variables[i] for i, _, _ in self._sza_gate}
        ungated_sw = [
            ch
            for ch in self.state_variables
            if ch in SHORTWAVE_VARIABLES and ch not in gated
        ]
        # Any non-empty constraint configuration must include a solar gate for every
        # shortwave state channel. Direct construction with constraints=None or an
        # empty dictionary remains unconstrained.
        if constraints and ungated_sw:
            raise ValueError(
                f"constraints.sza_gate must cover all shortwave state outputs "
                f"{list(SHORTWAVE_VARIABLES)}; missing a gate for {ungated_sw} "
                "(radiation would stay nonzero at night)."
            )
        self._has_constraints = bool(self._bound_lo or self._bound_up or self._sza_gate)

    def _apply_constraints(
        self,
        x: torch.Tensor,
        valid_time: datetime,
        lat2d: np.ndarray,
        lon2d: np.ndarray,
    ) -> None:
        """Apply physical bounds and the shortwave solar gate in place.

        ``x`` has shape ``[1, n_state, H, W]`` and is in physical space. The
        constrained state is fed into the next rollout step.
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
            ).unsqueeze(
                0
            )  # [1, H, W], matching each constrained channel
            for i, th, hw in self._sza_gate:
                gate = ((cos_z - th + hw) / (2.0 * hw)).clamp(0.0, 1.0)
                x[:, i] = (x[:, i] * gate).clamp(min=0.0)

    # ── derived hub-height wind ───────────────────────────────────────────────

    def _setup_hub_wind(self, wind_levels: dict) -> None:
        """Prepare the channel indices and level heights used for hub-height wind.

        ``wind_levels`` names the static elevation field and lists the model wind
        levels. Each level identifies its ``u`` and ``v`` state channels and
        coefficients ``a`` and ``b``::

            {"elevation_invariant": "elevation_norm",
             "levels": [{"u": "U_L40", "v": "V_L40", "a": .., "b": ..}, ...]}

        Its height above ground at each grid cell is calculated as
        ``a + b * elevation``.
        """
        entries = [
            (lev["u"], lev["v"], float(lev["a"]), float(lev["b"]))
            for lev in wind_levels.get("levels", [])
        ]
        if len(entries) < 2:
            raise ValueError(
                "wind_levels must define at least 2 'levels' to interpolate; got "
                f"{len(entries)}. A single level yields a constant (non-interpolated) "
                "field at every hub height."
            )
        entries.sort(key=lambda e: e[2])  # ascending nominal height
        svi = {v: i for i, v in enumerate(self.state_variables)}
        missing = sorted({c for e in entries for c in (e[0], e[1]) if c not in svi})
        if missing:
            raise ValueError(
                f"wind_levels references channels absent from state_variables: {missing}"
            )
        elev = wind_levels.get("elevation_invariant", "elevation_norm")
        if elev not in self._static_names:
            raise ValueError(
                f"wind_levels.elevation_invariant {elev!r} is not a static invariant "
                f"({self._static_names})"
            )
        self._hub_elev_idx = self._static_names.index(elev)
        dev = self.static_invariants.device
        self.register_buffer(
            "_hub_u_idx",
            torch.tensor([svi[e[0]] for e in entries], dtype=torch.long, device=dev),
        )
        self.register_buffer(
            "_hub_v_idx",
            torch.tensor([svi[e[1]] for e in entries], dtype=torch.long, device=dev),
        )
        self.register_buffer(
            "_hub_a", torch.tensor([e[2] for e in entries], device=dev)
        )
        self.register_buffer(
            "_hub_b", torch.tensor([e[3] for e in entries], device=dev)
        )
        # Per-pixel heights must increase along the level axis.
        if self.static_invariants.numel():
            elev_field = self.static_invariants[self._hub_elev_idx]
            h = self._hub_a[:, None, None] + self._hub_b[:, None, None] * elev_field
            if not bool((h[1:] - h[:-1] >= -1e-4).all()):
                raise ValueError(
                    "wind_levels height coefficients (a + b*elevation) are not "
                    "monotonic in height across this grid's elevation range; "
                    "interpolation requires ascending level heights."
                )
            # Every grid point supports interpolation within this height range:
            # [max(lowest level), min(highest level)].
            band_lo = float(h[0].max())
            band_hi = float(h[-1].min())
            outside = [hh for hh in self._hub_heights if hh < band_lo or hh > band_hi]
            if outside:
                warnings.warn(
                    f"Requested hub heights {outside} m are outside the grid-wide "
                    f"interpolation range [{band_lo:.0f}, {band_hi:.0f}] m. At grid "
                    "cells where a requested height lies outside the available model "
                    "levels, wind is taken from the nearest level instead of "
                    "interpolated. Choose a height within this range to interpolate "
                    "at every grid cell.",
                    UserWarning,
                    stacklevel=2,
                )

    def _derive_hub_wind(self, x: torch.Tensor) -> torch.Tensor:
        """Append hub-height wind components to a state in physical units.

        ``x`` has shape ``[..., n_state, H, W]``. Wind is interpolated vertically at
        each grid point. During rollout, only the original state channels, not the
        derived ``u{H}m`` and ``v{H}m`` channels, are used as input to the next step.
        """
        elev = self.static_invariants[self._hub_elev_idx]  # [H, W]
        heights = self._hub_a[:, None, None] + self._hub_b[:, None, None] * elev
        u = x[..., self._hub_u_idx, :, :]  # [..., K, H, W]
        v = x[..., self._hub_v_idx, :, :]
        comps = []
        for h in self._hub_heights:
            comps.append(
                _interp_levels_to_height(u, heights, h, self._hub_interp).unsqueeze(-3)
            )
            comps.append(
                _interp_levels_to_height(v, heights, h, self._hub_interp).unsqueeze(-3)
            )
        return torch.cat([x, *comps], dim=-3)

    def _decode_units(self, x: torch.Tensor) -> torch.Tensor:
        """Convert public Earth2Studio units to internal COSMO units in place.

        Conversion is applied along the variable axis of ``x``, which has shape
        ``[batch, time, lead_time, variable, y, x]``. For this model, total cloud
        cover changes from a fraction (``tcc``) to percent (``CLCT``), and total
        precipitation changes from metres (``tp``) to millimetres
        (``TOT_PRECIP``). Other channels are unchanged.
        """
        for idx, scale in self._unit_scale:
            x[:, :, :, idx] = x[:, :, :, idx] / scale
        return x

    def _encode_units(self, x: torch.Tensor) -> torch.Tensor:
        """Convert internal COSMO units to public Earth2Studio units in place.

        This reverses :meth:`_decode_units`: ``CLCT`` percent becomes ``tcc`` as a
        fraction, and ``TOT_PRECIP`` millimetres become ``tp`` in metres.
        """
        for idx, scale in self._unit_scale:
            x[:, :, :, idx] = x[:, :, :, idx] * scale
        return x

    # ── coordinate systems ───────────────────────────────────────────────────

    def input_coords(self) -> CoordSystem:
        """Return the expected coordinates for an input COSMO-REA state.

        Batch and initialization time are unrestricted, input lead time starts at
        zero, and variable labels use canonical Earth2Studio names. The model maps
        these labels to checkpoint channels internally.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": self._public_variables,
                "rea_y": self.rea_y,
                "rea_x": self.rea_x,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system, one time-step (1 h) ahead.

        Variables are the canonical state names plus any derived hub-height wind
        components (``u{H}m``/``v{H}m``); the input stays the state names.
        """
        self._validate_coords(input_coords)
        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "time": input_coords["time"],
                "lead_time": input_coords["lead_time"] + self.time_step,
                "variable": self._output_variables,
                "rea_y": self.rea_y,
                "rea_x": self.rea_x,
            }
        )
        return output_coords

    def _validate_coords(self, input_coords: CoordSystem) -> None:
        """Validate coordinates for output_coords, __call__, and the iterator."""
        # Derived hub-height wind is output-only. If a caller manually chains
        # ``model(*model(x, coords))`` and feeds the full previous output back, the
        # incoming variables include ``u{H}m``/``v{H}m``; surface an actionable error
        # instead of a generic variable-handshake failure. The standard rollout uses
        # create_iterator(), which strips these before feeding the state forward.
        if self._hub_heights and "variable" in input_coords:
            fed_back = sorted(
                set(map(str, np.asarray(input_coords["variable"])))
                & set(self._derived_variables)
            )
            if fed_back:
                raise ValueError(
                    f"input coordinates include derived hub-height wind variables "
                    f"{fed_back}, which are output-only. Feed only the state channels "
                    "back, or use create_iterator() for rollout (it strips the derived "
                    "channels before the next step)."
                )
        target = self.input_coords()
        handshake_dim(input_coords, "batch", 0)
        handshake_dim(input_coords, "time", 1)
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "rea_y", 4)
        handshake_dim(input_coords, "rea_x", 5)
        handshake_size(input_coords, "rea_y", self.rea_y.shape[0])
        handshake_size(input_coords, "rea_x", self.rea_x.shape[0])
        handshake_coords(input_coords, target, "variable")
        handshake_coords(input_coords, target, "rea_y")
        handshake_coords(input_coords, target, "rea_x")

    # ── invariants / background ──────────────────────────────────────────────

    def _position_channels(
        self, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> torch.Tensor:
        """Create position channels from target-grid coordinates in degrees.

        Returns ``[sin(lat), cos(lat), sin(lon), cos(lon)]`` with shape
        ``[4, H, W]``.
        """
        lat_rad = torch.deg2rad(lat2d)
        lon_rad = torch.deg2rad(lon2d)
        return torch.stack(
            [
                torch.sin(lat_rad),
                torch.cos(lat_rad),
                torch.sin(lon_rad),
                torch.cos(lon_rad),
            ],
            dim=0,
        )

    def _static_background(
        self, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build invariant stacks in training order around ``cos_zenith``.

        Position channels are calculated from the grid. Other static channels come
        from ``static_invariants``.
        """
        pos = self._position_channels(lat2d, lon2d)
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
        """Return the cosine of the solar zenith angle for the target hour.

        Uses the larger value at the start and end of the hour
        (``valid_time - 1 h`` and ``valid_time``). A timezone-naive ``valid_time`` is
        interpreted as UTC.
        """
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)
        cz0 = cos_zenith_angle(valid_time - timedelta(hours=1), lon2d, lat2d)
        cz1 = cos_zenith_angle(valid_time, lon2d, lat2d)
        return np.maximum(cz0, cz1).astype(np.float32)

    def _interpolate(
        self, x: torch.Tensor, lat2d: torch.Tensor, lon2d: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear-regrid [C, H_in, W_in] ERA5 -> [C, H_out, W_out] target grid."""
        return interp.latlon_interpolation_regular(
            x, self.lat_input_grid, self.lon_input_grid, lat2d, lon2d
        )

    def _background(
        self,
        era5: torch.Tensor,
        valid_time: datetime,
        lat2d: torch.Tensor,
        lon2d: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the model background for one target time.

        Regrids and normalizes ``era5``, then concatenates the ERA5 channels,
        pre-zenith invariants, ``cos_zenith``, and post-zenith invariants in training
        order. ``era5`` has shape ``[n_era5, H_in, W_in]``; the result has shape
        ``[1, C_bg, H_out, W_out]``.
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
        if not torch.isfinite(background).all():
            raise RuntimeError(
                f"StormCastEurope: non-finite value in the assembled background "
                f"at valid time {valid_time}. Check ERA5 conditioning, static "
                "invariants, and grid coordinates for missing or invalid values."
            )
        return background

    # ── diffusion sampling ───────────────────────────────────────────────────

    def _rebind_latent(self, H: int, W: int) -> tuple[int, int]:
        """Update DiT attention and detokenizer sizes for an ``(H, W)`` output grid."""
        dit = self.diffusion_model.model.model
        ph, pw = dit.tokenizer.patch_size
        if H % ph or W % pw:
            raise ValueError(
                f"COSMO grid {H}x{W} is not divisible by the DiT patch size "
                f"{(ph, pw)}. The detokenizer requires height and width to be "
                "multiples of their corresponding patch dimensions."
            )
        latent_hw = (H // ph, W // pw)
        dit.attn_kwargs_forward["latent_hw"] = latent_hw
        detok = dit.detokenizer
        target = detok.proj if hasattr(detok, "proj") else detok
        target.h_patches, target.w_patches = latent_hw
        return latent_hw

    def _denoise(
        self,
        cond_concat: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one next-state sample in normalized model space.

        ``cond_concat`` contains ``[background, previous_state]`` with shape
        ``[1, n_condition, H, W]``. The result has shape
        ``[1, n_state, H, W]``.
        """
        net = self.diffusion_model
        dev = cond_concat.device
        H, W = cond_concat.shape[-2:]
        latent_hw = self._rebind_latent(H, W)
        condition = cond_concat.float()

        # Draw fresh noise from the global PyTorch generator on each call. Callers
        # can seed torch for reproducibility.
        latents = torch.randn(
            (1, len(self.state_variables), H, W),
            device=dev,
            dtype=torch.float32,
        )
        scheduler = EDMNoiseScheduler(
            sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
        )

        def x0_predictor(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return net(
                x.float(),
                t.to(torch.float32).reshape(-1),
                condition=condition,
                attn_kwargs={"latent_hw": latent_hw},
            ).double()

        denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
        ctx = (
            torch.autocast(dev.type, dtype=torch.bfloat16)
            if self.amp
            else nullcontext()
        )
        with ctx:
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
        x: torch.Tensor,
        era5: torch.Tensor,
        valid_time: datetime,
        lat2d: torch.Tensor,
        lon2d: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one state by one hour in internal physical COSMO units.

        This method is used by both one-step and iterator inference, so its
        non-finite checks apply to both.

        Parameters
        ----------
        x : torch.Tensor
            Current state in internal physical COSMO units, with shape
            ``[1, n_state, H, W]``.
        era5 : torch.Tensor
            Unnormalized ERA5 background at the target time, with shape
            ``[n_era5, H_in, W_in]``.
        valid_time : datetime
            Target valid time used for the ``cos_zenith`` background channel.
        lat2d, lon2d : torch.Tensor
            2D target grid (rotated-pole COSMO-REA).
        """
        # Validate the input before applying channel transforms.
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "StormCastEurope: non-finite value in the input state at valid "
                f"time {valid_time}."
            )
        prev_state = self._state_to_model(x)  # physical -> model space
        # _background validates the assembled conditioning tensor.
        background = self._background(era5, valid_time, lat2d, lon2d)
        # The trained channel order is [background, prev_state].
        cond_concat = torch.cat([background, prev_state], dim=1)

        out = self._denoise(cond_concat)
        if not torch.isfinite(out).all():
            raise RuntimeError(
                "StormCastEurope: non-finite value in the normalized model output "
                f"at valid time {valid_time}."
            )
        physical = self._state_from_model(out)
        # Inverse transforms such as expm1 and sinh can overflow, so validate again.
        if not torch.isfinite(physical).all():
            raise RuntimeError(
                "StormCastEurope: non-finite value after inverse channel "
                f"transforms at valid time {valid_time}. This may indicate overflow."
            )
        # Apply constraints in internal physical space before returning the state.
        if self._has_constraints:
            self._apply_constraints(
                physical, valid_time, lat2d.cpu().numpy(), lon2d.cpu().numpy()
            )
        return physical

    # ── prognostic interface ─────────────────────────────────────────────────

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Predict the COSMO-REA state at ``t + 1 h`` from the state at ``t``.

        Here, ``t`` is the input valid time: initialization time plus input lead time.

        Parameters
        ----------
        x : torch.Tensor
            Input state in public Earth2Studio units, with shape
            ``[batch, time, lead_time, variable, rea_y, rea_x]``.
        coords : CoordSystem
            Coordinates describing the input state.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and its coordinates at ``t + 1 h``.
        """
        # Validate the input coordinates and prepare coordinates for t + 1 h.
        output_coords = self.output_coords(coords)

        # Verify that every tensor dimension matches its coordinate length.
        expected = tuple(
            len(coords[k])
            for k in ("batch", "time", "lead_time", "variable", "rea_y", "rea_x")
        )
        if tuple(x.shape) != expected:
            raise ValueError(
                f"input tensor shape {tuple(x.shape)} does not match coords "
                f"{expected} (batch, time, lead_time, variable, rea_y, rea_x)."
            )
        if x.device != self.state_center.device:
            raise ValueError(
                f"input is on {x.device} but the model is on "
                f"{self.state_center.device}; move the input to the model's device."
            )

        lat2d = self.lat_output_grid.to(x.device)
        lon2d = self.lon_output_grid.to(x.device)

        # Fetch ERA5 at t + 1 h for each input valid time t. The returned time and
        # lead_time axes align with the loops below.
        era5 = self._get_conditioning(coords, x.device)

        # Convert public Earth2Studio units to internal COSMO units before inference.
        x = self._decode_units(x.clone())
        for j, time in enumerate(coords["time"]):
            for k, lead_time in enumerate(coords["lead_time"]):
                target_time = time + lead_time + self.time_step
                valid_time = _to_datetime(target_time)
                for b in range(len(coords["batch"])):
                    era5_bt = era5[b, j, k]
                    x[b, j, k] = self._forward(
                        x[b, j, k].unsqueeze(0),
                        era5_bt,
                        valid_time,
                        lat2d,
                        lon2d,
                    ).squeeze(0)
        # Output shape: [batch, time, lead_time, variable, rea_y, rea_x].
        out = self._encode_units(x)
        # Append requested hub-height wind components as output-only channels.
        if self._hub_heights:
            out = self._derive_hub_wind(out)
        return out, output_coords

    def _get_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> torch.Tensor:
        """Fetch ERA5 at ``t + 1 h`` on the packaged regular input grid.

        Here, ``t`` is the input valid time. Returns a tensor with shape
        ``[batch, time, lead_time, n_era5, H_in, W_in]``; :meth:`_background` then
        regrids each frame onto the curvilinear COSMO grid.
        """
        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCastEurope has been called without initializing the model's "
                "conditioning_data_source."
            )
        # Request ERA5 at t + 1 h by advancing each input lead time by one hour.
        lead_time = coords["lead_time"] + self.time_step
        # Map the source onto the regular lat_input_grid / lon_input_grid.
        era5, era5_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=np.array(self.era5_variables),
            lead_time=lead_time,
            device=device,
            interp_to=OrderedDict(
                {"_lat": self.lat_input_numpy, "_lon": self.lon_input_numpy}
            ),
            interp_method="linear",
        )
        target_axes = ["time", "lead_time", "variable", "_lat", "_lon"]
        # Recover the tensor axis order while ignoring non-dimension coordinate keys.
        # ForecastSource data may place variable before lead_time.
        src_axes = [k for k in era5_coords if k in target_axes]
        if era5.ndim != len(src_axes) or set(src_axes) != set(target_axes):
            raise RuntimeError(
                f"ERA5 conditioning has axes {list(era5_coords)} (tensor rank "
                f"{era5.ndim}); expected the dims {target_axes}. Check the data source."
            )
        if src_axes != target_axes:
            era5 = era5.permute(*(src_axes.index(k) for k in target_axes)).contiguous()
        # Verify the values associated with each axis. Matching shapes are not enough:
        # reordered variables, times, or grid coordinates would assign data to the
        # wrong model inputs.
        returned_vars = list(np.asarray(era5_coords["variable"]))
        if returned_vars != list(self.era5_variables):
            raise RuntimeError(
                f"ERA5 conditioning returned variables {returned_vars}, expected "
                f"{list(self.era5_variables)} in that order; the background channels "
                "would be mis-mapped. Check the conditioning data source."
            )
        if not np.array_equal(
            np.asarray(era5_coords["time"]), np.asarray(coords["time"])
        ):
            raise RuntimeError(
                "ERA5 conditioning returned a different time axis than requested; the "
                "background would be mis-timed relative to the state."
            )
        if not np.array_equal(
            np.asarray(era5_coords["lead_time"]), np.asarray(lead_time)
        ):
            raise RuntimeError(
                "ERA5 conditioning returned a different lead_time axis than the "
                "requested t + 1 h lead-time axis; the background would be mis-timed."
            )
        for key, label, expected_axis in (
            ("_lat", "latitude", self.lat_input_numpy),
            ("_lon", "longitude", self.lon_input_numpy),
        ):
            returned_axis = np.asarray(era5_coords[key])
            if returned_axis.shape != expected_axis.shape or not np.allclose(
                returned_axis, expected_axis, rtol=1e-6, atol=1e-6
            ):
                raise RuntimeError(
                    f"ERA5 conditioning returned {label} coordinates that do not "
                    "match the packaged ERA5 input grid."
                )
        # ERA5 forcing is identical across batch members and read-only downstream, so
        # broadcast it as a view rather than copying (repeat) a potentially large frame.
        era5 = era5.unsqueeze(0).expand(len(coords["batch"]), -1, -1, -1, -1, -1)
        expected = (
            len(coords["batch"]),
            len(coords["time"]),
            len(lead_time),
            len(self.era5_variables),
            self.lat_input_numpy.shape[0],
            self.lon_input_numpy.shape[0],
        )
        if tuple(era5.shape) != expected:
            raise RuntimeError(
                f"ERA5 conditioning has shape {tuple(era5.shape)}, expected "
                f"{expected}; check the data source and the interp_to input grid."
            )
        return era5

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Yield the initial condition, then successive rollout steps."""
        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCastEurope has been called without initializing the model's "
                "conditioning_data_source."
            )
        coords = coords.copy()
        self._validate_coords(coords)

        # Include hub-height wind in the initial condition so all outputs have the
        # same variables.
        yield self._augment_output(x, coords)

        while True:
            # __call__ fetches the ERA5 background for each target time.
            x, coords = self.front_hook(x, coords)
            out, out_coords = self(x, coords)  # (state + hub); hub is output-only
            # Apply the rear hook to the state, then derive hub-height wind from the
            # updated state. Without requested hub heights, augmentation is a no-op.
            x, coords = self._state_from_output(out, out_coords)
            x, coords = self.rear_hook(x, coords)
            out, out_coords = self._augment_output(x, coords)
            yield out, out_coords
            # x, coords already hold the (post-hook) state fed to the next step.

    def _augment_output(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Append hub-height wind components and coordinates when requested."""
        if not self._hub_heights:
            return x, coords
        out_coords = coords.copy()
        out_coords["variable"] = self._output_variables
        return self._derive_hub_wind(x), out_coords

    def _state_from_output(
        self, out: torch.Tensor, out_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Remove output-only hub-height wind channels before the next rollout step."""
        if not self._hub_heights:
            return out, out_coords
        coords = out_coords.copy()
        coords["variable"] = self._public_variables
        return out[:, :, :, : self._n_state].contiguous(), coords

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Yield the initial state, then forecasts at successive one-hour lead times."""
        yield from self._default_generator(x, coords)

    # ── sub-domains ──────────────────────────────────────────────────────────

    def _snap_to_patch(self, lo: int, hi: int, n: int) -> tuple[int, int]:
        """Adjust a selected grid interval to contain complete DiT patches.

        ``[lo, hi)`` is one axis of the sub-domain selected by :meth:`set_domain`,
        within an axis containing ``n`` cells. Its length must be divisible by
        ``_patch_size`` because the DiT tokenizer and detokenizer operate on complete
        patches. For example, a 5-cell interval grows to 6 cells when the patch size
        is 2.

        The interval grows at the high edge first and then at the low edge, preserving
        the requested sub-domain whenever possible. It shrinks only when the full
        grid leaves no room to grow. A patch size of 1 requires no adjustment.
        """
        patch_size = self._patch_size
        cells_needed = (-(hi - lo)) % patch_size
        if cells_needed:
            growth = min(cells_needed, n - hi)
            hi += growth
            cells_needed -= growth
            growth = min(cells_needed, lo)
            lo -= growth
            cells_needed -= growth
            if cells_needed:
                # Expansion is impossible; use the largest fitting multiple.
                hi -= (hi - lo) % patch_size
        return lo, hi

    def set_domain(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        margin_deg: float = 1.0,
    ) -> "StormCastEurope":
        """Restrict the model to a sub-domain for a lat/lon bounding box.

        The returned model shares the network and conditioning source with the
        parent. Its COSMO grid is the smallest rectangular grid block containing the
        requested bounding box, expanded when necessary to contain complete DiT
        patches.

        ``margin_deg`` adds coverage only to the regular ERA5 conditioning grid. It
        does not add COSMO cells or create a rollout halo.

        The package may contain an extended COSMO grid beyond the validated native
        footprint. If the final patch-aligned grid enters this region, a warning is
        issued because model skill there has not been validated. A bounding box
        outside the extended grid raises an error.

        The returned model shares its network and grid settings with the parent. Use
        the parent and its sub-domains sequentially, on the same device and in the
        same train/eval mode. Call ``set_domain`` on the full model because returned
        sub-domain models do not retain unselected grid data.

        No extra COSMO halo is run and removed. Boundary values are reused as input at
        the next step, so boundary errors can move inward during long rollouts. To
        protect a region of interest, request a larger bounding box; ``margin_deg``
        does not provide this protection.
        """
        if not all(
            np.isfinite(v) for v in (lat_min, lat_max, lon_min, lon_max, margin_deg)
        ):
            raise ValueError(
                "set_domain bounds and margin_deg must be finite; got "
                f"lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] margin={margin_deg}."
            )
        if lat_min > lat_max or lon_min > lon_max:
            raise ValueError(
                "set_domain requires lat_min <= lat_max and lon_min <= lon_max; got "
                f"lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]."
            )
        if margin_deg <= 0:
            raise ValueError(f"set_domain margin_deg must be > 0 (got {margin_deg}).")
        ext_lat, ext_lon = self._ext_lat_numpy, self._ext_lon_numpy
        ext = ext_lat is not None
        if ext:
            # Guard both extended coordinate arrays.
            if ext_lat is None or ext_lon is None:
                raise RuntimeError("extended lat/lon arrays are not set")
            latg, long_ = ext_lat, ext_lon
        else:
            latg, long_ = self.lat_output_numpy, self.lon_output_numpy

        corner_lat = np.array([lat_min, lat_min, lat_max, lat_max])
        corner_lon = np.array([lon_min, lon_max, lon_min, lon_max])
        if not _points_in_grid_footprint(corner_lat, corner_lon, latg, long_).all():
            raise ValueError(
                f"domain lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] is not "
                f"fully inside the {'extended' if ext else 'native'} footprint; "
                "no invariants there."
            )
        # The out-of-distribution margin warning is issued below on the FINAL,
        # patch-aligned grid, because _snap_to_patch may grow the run window into the
        # extended margin even when the requested box is inside the native footprint.

        inside = (
            (latg >= lat_min)
            & (latg <= lat_max)
            & (long_ >= lon_min)
            & (long_ <= lon_max)
        )
        if not inside.any():
            raise ValueError(
                f"bounding box lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}] "
                "selects no grid cells (smaller than one grid cell?); widen it."
            )
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        i0, i1 = int(rows[0]), int(rows[-1]) + 1
        j0, j1 = int(cols[0]), int(cols[-1]) + 1
        Hn, Wn = latg.shape
        i0s, i1s = self._snap_to_patch(i0, i1, Hn)
        j0s, j1s = self._snap_to_patch(j0, j1, Wn)
        # Raise if patch alignment removes cells from the requested bounding box.
        if i1s < i1 or j1s < j1:
            raise ValueError(
                f"grid dimension not divisible by patch_size={self._patch_size}: "
                "snapping the run window shrank it below the requested bounding box. "
                "Widen it or provide a grid whose dimensions are multiples of "
                "patch_size."
            )
        i0, i1, j0, j1 = i0s, i1s, j0s, j1s
        if min(i1 - i0, j1 - j0) < self._min_domain_cells:
            raise ValueError(
                f"sub-domain {i1 - i0}x{j1 - j0} is below the "
                f"{self._min_domain_cells}-cell per-side minimum (the DiT NATTEN "
                "kernel must fit the latent grid); widen the bounding box."
            )
        # Warn if the FINAL patch-aligned grid reaches beyond the validated native
        # footprint (either the request did, or _snap_to_patch grew it there).
        if ext:
            gc_lat = np.array(
                [latg[i0, j0], latg[i0, j1 - 1], latg[i1 - 1, j0], latg[i1 - 1, j1 - 1]]
            )
            gc_lon = np.array(
                [
                    long_[i0, j0],
                    long_[i0, j1 - 1],
                    long_[i1 - 1, j0],
                    long_[i1 - 1, j1 - 1],
                ]
            )
            if not _points_in_grid_footprint(
                gc_lat, gc_lon, self.lat_output_numpy, self.lon_output_numpy
            ).all():
                warnings.warn(
                    "sub-domain reaches beyond the validated COSMO-REA footprint into "
                    "the extended margin: invariants exist but the model is "
                    "out-of-distribution (skill unvalidated).",
                    UserWarning,
                    stacklevel=2,
                )

        dev = self.lat_output_grid.device
        if ext:
            if self._ext_static_numpy is None:
                raise RuntimeError("extended static array is not set")
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
        lat_values, lon_values = lat_out.cpu().numpy(), lon_out.cpu().numpy()
        dlat = float(self.lat_input_numpy[1] - self.lat_input_numpy[0])
        dlon = float(self.lon_input_numpy[1] - self.lon_input_numpy[0])

        def _era5_axis(a0: float, a1: float, step: float) -> torch.Tensor:
            # Regular ERA5 input axis covering [a0, a1] plus margin_deg on each side,
            # so the cropped COSMO grid stays strictly inside it.
            return torch.arange(
                float(np.floor(a0) - margin_deg),
                float(np.ceil(a1) + margin_deg + step),
                step,
                dtype=torch.float32,
                device=dev,
            )

        sub = StormCastEurope(
            state_variables=self.state_variables,
            era5_variables=self.era5_variables,
            diffusion_model=self.diffusion_model,
            resolution=self.resolution,
            lat_input_grid=_era5_axis(lat_values.min(), lat_values.max(), dlat),
            lon_input_grid=_era5_axis(lon_values.min(), lon_values.max(), dlon),
            lat_output_grid=lat_out,
            lon_output_grid=lon_out,
            era5_center=self.era5_center.flatten(),
            era5_scale=self.era5_scale.flatten(),
            state_center=self.state_center.flatten(),
            state_scale=self.state_scale.flatten(),
            static_invariants=static,
            pre_invariant_variables=self.pre_invariant_variables,
            post_invariant_variables=self.post_invariant_variables,
            channel_transforms=self._channel_transforms,
            conditioning_data_source=self.conditioning_data_source,
            number_of_steps=self.number_of_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            solver=self.solver,
            physical_clamp=self.physical_clamp,
            amp=self.amp,
            constraints=self._constraints,
            hub_heights=(self._hub_heights or None),
            hub_interp=self._hub_interp,
            wind_levels=self._wind_levels,
        )
        sub._patch_size = self._patch_size
        sub._min_domain_cells = self._min_domain_cells
        return sub

    # ── package loading ──────────────────────────────────────────────────────

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        # TODO: pin to a commit hash (``@<sha>``) once the weights are uploaded.
        package = Package(
            "hf://nvidia/stormcast-cosmo-era5",
            cache_options={
                "cache_storage": Package.default_cache("stormcast-cosmo-era5"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        device: str | None = None,
        resolution: Literal["rea6", "rea2"] = "rea6",
        conditioning_data_source: DataSource | ForecastSource | None = None,
        hub_heights: Sequence[float] | None = None,
        hub_interp: Literal["linear", "log"] = "linear",
    ) -> "StormCastEurope":
        """Load an autoregressive COSMO-REA model from a package.

        ``resolution`` selects the ``"rea6"`` or ``"rea2"`` checkpoint within one
        package. Variables, normalization, transforms, constraints, and checkpoint
        paths come from the package metadata. Optional sampler settings use the
        constructor defaults when omitted. The ``physical_clamp`` and ``amp`` flags
        are read only from the nested ``sampler`` block.

        The packaged physical ``constraints`` (per-variable bounds + shortwave
        solar gate) are always applied; there is no option to disable them.

        ``hub_heights`` (m above ground) enables the derived hub-height wind
        components ``u{H}m``/``v{H}m`` (requires a ``wind_levels`` package block);
        ``hub_interp`` is the vertical interpolation (``"linear"``/``"log"``).
        """
        if resolution not in SUPPORTED_VARIANTS:
            raise ValueError(f"resolution must be one of {list(SUPPORTED_VARIANTS)}")

        try:
            package.resolve("config.json")
        except (FileNotFoundError, ValueError):
            pass

        prefix = f"{resolution}/"

        def _load_json(filename: str) -> dict:
            with open(package.resolve(prefix + filename), encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                raise ValueError(f"{filename} is empty")
            return json.loads(content)

        metadata = _load_json("metadata.json")
        era5_variables = metadata["era5_variables"]
        state_variables = metadata["state_variables"]
        pre_invariant_variables = metadata["pre_invariant_variables"]
        post_invariant_variables = metadata.get("post_invariant_variables", [])
        channel_transforms = metadata.get("channel_transforms", {})
        sampler = metadata.get("sampler", {})
        # Constraints are mandatory and always applied: require the block rather
        # than silently loading an unconstrained model.
        if not metadata.get("constraints"):
            raise ValueError(
                f"package metadata (resolution {resolution}) has no 'constraints' "
                "block; StormCastEurope requires it (the physical bounds + solar "
                "gate are always applied). Rebuild the package with the block."
            )

        ckpt_key = resolution
        try:
            ckpt = metadata["checkpoints"][ckpt_key]
        except KeyError as e:
            raise ValueError(
                f"package metadata has no checkpoint entry {ckpt_key!r} "
                f"(available: {list(metadata.get('checkpoints', {}))})."
            ) from e

        ckpt_path = package.resolve(prefix + ckpt)
        # The packaged checkpoint must be self-describing so from_checkpoint can
        # reconstruct the network.
        try:
            model = EDMPreconditioner.from_checkpoint(ckpt_path).eval()
        except Exception as e:
            raise ValueError(
                f"could not load the AR diffusion network from {ckpt!r} (resolved "
                f"{ckpt_path}) via from_checkpoint: {e}. Expected a physicsnemo "
                ".mdlus checkpoint."
            ) from e
        model.requires_grad_(False)
        if device is not None:
            model = model.to(device)

        stats = _load_json("stats.json")

        def _stat(group: str, var: str, moment: str) -> float:
            try:
                return stats[group][var][moment]
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"stats.json is missing {group!r} {moment!r} for variable "
                    f"{var!r} (resolution {resolution}); every era5/state variable "
                    "needs a mean and std entry."
                ) from e

        era5_center = torch.tensor(
            [_stat("era5", v, "mean") for v in era5_variables], device=device
        )
        era5_scale = torch.tensor(
            [_stat("era5", v, "std") for v in era5_variables], device=device
        )
        state_center = torch.tensor(
            [_stat("state", v, "mean") for v in state_variables], device=device
        )
        state_scale = torch.tensor(
            [_stat("state", v, "std") for v in state_variables], device=device
        )
        for nm, c, s in (
            ("era5", era5_center, era5_scale),
            ("state", state_center, state_scale),
        ):
            if not (
                torch.isfinite(c).all() and torch.isfinite(s).all() and (s > 0).all()
            ):
                raise ValueError(
                    f"{nm} normalization stats are invalid (non-finite or std<=0); "
                    f"check the package stats.json (resolution {resolution})."
                )

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

        inv_names = [
            n
            for n in (*pre_invariant_variables, *post_invariant_variables)
            if n not in POS_VARIABLES
        ]
        static_invariants: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Preferred packages store unnormalized invariants on an extended grid. Select
        # the native block using native_offset and native_shape, then normalize it
        # with the training statistics. Metadata names for normalized channels may
        # end in "_norm", while the file stores the corresponding base name. Legacy
        # packages may instead provide invariants.nc, already cropped and normalized.
        inv_meta = metadata.get("invariants")
        out_shape = tuple(int(s) for s in lat_output_grid.shape)
        # Keep the full extended coordinates and normalized invariants for set_domain;
        # leave them as None when no extended file is available.
        ext_lat_np = ext_lon_np = ext_static_np = None
        if inv_meta and inv_meta.get("file"):
            i0, j0 = (int(v) for v in inv_meta["native_offset"])
            ny, nx = (int(v) for v in inv_meta["native_shape"])
            if i0 < 0 or j0 < 0 or ny <= 0 or nx <= 0:
                raise ValueError(
                    f"invalid invariants native_offset {(i0, j0)} / native_shape "
                    f"{(ny, nx)} (resolution {resolution}); must be non-negative "
                    "offsets and positive shape."
                )
            if (ny, nx) != out_shape:
                raise ValueError(
                    f"invariants native_shape {(ny, nx)} != COSMO output grid "
                    f"{out_shape} (resolution {resolution}); the cropped invariants "
                    "would not align with the state grid."
                )
            declared = inv_meta.get("channels")
            norm_json = _load_json(inv_meta["norm_stats_file"])
            norm = norm_json.get("channels")
            if norm is None:
                raise ValueError(
                    f"{inv_meta['norm_stats_file']!r} has no 'channels' block "
                    f"(resolution {resolution})."
                )
            ext_stack: list = []
            with xr.open_dataset(package.resolve(prefix + inv_meta["file"])) as ds:
                if "lat" in ds and "lon" in ds:
                    ext_lat_np = np.asarray(ds["lat"], dtype=np.float32)
                    ext_lon_np = np.asarray(ds["lon"], dtype=np.float32)
                    # Confirm that the native crop covers the same grid cells as the
                    # output grid. Matching shapes alone cannot detect an incorrect
                    # native_offset that shifts the invariants relative to the state.
                    lat_out_np = (
                        lat_output_grid.detach().cpu().numpy().astype(np.float32)
                    )
                    lon_out_np = (
                        lon_output_grid.detach().cpu().numpy().astype(np.float32)
                    )
                    for nm, ext_a, out_a in (
                        ("lat", ext_lat_np, lat_out_np),
                        ("lon", ext_lon_np, lon_out_np),
                    ):
                        if i0 + ny > ext_a.shape[0] or j0 + nx > ext_a.shape[1]:
                            raise ValueError(
                                f"native crop [{i0}:{i0 + ny}, {j0}:{j0 + nx}] is out "
                                f"of bounds for extended {nm} of shape {ext_a.shape} "
                                f"(resolution {resolution})."
                            )
                        crop = ext_a[i0 : i0 + ny, j0 : j0 + nx]
                        if not np.allclose(crop, out_a, atol=1e-4):
                            raise ValueError(
                                f"extended {nm} native crop at offset {(i0, j0)} does "
                                f"not match the grids.nc output grid (resolution "
                                f"{resolution}); the invariants would be mis-registered "
                                "against the state grid. Check native_offset."
                            )
                for name in inv_names:
                    phys = name[:-5] if name.endswith("_norm") else name
                    if declared is not None and phys not in declared:
                        raise ValueError(
                            f"invariant {name!r} (physical {phys!r}) is not in "
                            f"metadata invariants channels {declared} (resolution "
                            f"{resolution})."
                        )
                    if phys not in ds:
                        raise ValueError(
                            f"invariants file {inv_meta['file']!r} has no variable "
                            f"{phys!r} (for invariant {name!r}, resolution "
                            f"{resolution}); available: {list(ds.data_vars)}."
                        )
                    arr = np.asarray(ds[phys], dtype=np.float32)
                    ah, aw = arr.shape
                    if i0 + ny > ah or j0 + nx > aw:
                        raise ValueError(
                            f"native crop [{i0}:{i0 + ny}, {j0}:{j0 + nx}] is out of "
                            f"bounds for invariant {phys!r} of shape {arr.shape} "
                            f"(resolution {resolution})."
                        )
                    spec = norm.get(phys)
                    method = (spec or {}).get("method", "identity")
                    # A "_norm" channel is z-scored by contract; require its stats.
                    if name.endswith("_norm") and method != "zscore":
                        raise ValueError(
                            f"invariant {name!r} is marked normalized (_norm) but "
                            f"{inv_meta['norm_stats_file']!r} has no zscore stats for "
                            f"{phys!r} (got method {method!r})."
                        )
                    if method == "zscore":
                        if "mean" not in spec or "std" not in spec:
                            raise ValueError(
                                f"z-score stats for {phys!r} are missing 'mean'/'std' "
                                f"in {inv_meta['norm_stats_file']!r}."
                            )
                        mean, std = float(spec["mean"]), float(spec["std"])
                        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
                            raise ValueError(
                                f"invariant {phys!r} has invalid z-score stats "
                                f"(mean={mean}, std={std}) in "
                                f"{inv_meta['norm_stats_file']!r}."
                            )
                        arr = (arr - mean) / std
                    elif method != "identity":
                        raise ValueError(
                            f"invariant {phys!r} has unknown normalization method "
                            f"{method!r} in {inv_meta['norm_stats_file']!r} (expected "
                            "'zscore' or 'identity')."
                        )
                    ext_stack.append(arr)  # full normalized extended field
                    static_invariants[name] = torch.as_tensor(
                        arr[i0 : i0 + ny, j0 : j0 + nx],
                        device=device,
                        dtype=torch.float32,
                    )
            if ext_stack and ext_lat_np is not None:
                ext_static_np = np.stack(ext_stack)
                # set_domain may select cells outside the native crop, so validate
                # the complete extended arrays.
                for nm, arr in (
                    ("lat", ext_lat_np),
                    ("lon", ext_lon_np),
                    ("invariants", ext_static_np),
                ):
                    if not np.isfinite(arr).all():
                        raise ValueError(
                            f"extended {nm} array (resolution {resolution}) contains "
                            "non-finite values and cannot be used by set_domain."
                        )
        else:
            with xr.open_dataset(package.resolve(prefix + "invariants.nc")) as ds:
                for name in inv_names:
                    if name not in ds:
                        raise ValueError(
                            f"invariants.nc has no variable {name!r} (resolution "
                            f"{resolution}); it is listed as a non-position invariant "
                            f"in the metadata but absent from the packaged invariants "
                            f"(available: {list(ds.data_vars)})."
                        )
                    static_invariants[name] = torch.as_tensor(
                        np.asarray(ds[name]), device=device, dtype=torch.float32
                    )

        model_obj = cls(
            state_variables=state_variables,
            era5_variables=era5_variables,
            diffusion_model=model,
            resolution=resolution,
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            era5_center=era5_center,
            era5_scale=era5_scale,
            state_center=state_center,
            state_scale=state_scale,
            static_invariants=static_invariants,
            pre_invariant_variables=pre_invariant_variables,
            post_invariant_variables=post_invariant_variables,
            channel_transforms=channel_transforms,
            conditioning_data_source=conditioning_data_source,
            number_of_steps=sampler.get("num_steps", 18),
            sigma_min=sampler.get("sigma_min", 0.002),
            sigma_max=sampler.get("sigma_max", 800.0),
            rho=sampler.get("rho", 7.0),
            solver=sampler.get("solver", "heun"),
            physical_clamp=sampler.get("physical_clamp", True),
            amp=sampler.get("amp", False),
            constraints=metadata["constraints"],
            hub_heights=hub_heights,
            hub_interp=hub_interp,
            wind_levels=metadata.get("wind_levels"),
        )
        # Require at least one constraint that applies to a state variable.
        if not model_obj._has_constraints:
            raise ValueError(
                f"package metadata (resolution {resolution}) ships a 'constraints' "
                "block with no bounds or solar gate that applies to state_variables. "
                "Check the constraint channel names."
            )
        # Extended grid/invariants + DiT grid constraints for set_domain().
        model_obj._ext_lat_numpy = ext_lat_np
        model_obj._ext_lon_numpy = ext_lon_np
        model_obj._ext_static_numpy = ext_static_np
        dit = model.model.model
        patch = tuple(int(p) for p in dit.tokenizer.patch_size)
        if patch[0] != patch[1]:
            raise ValueError(
                f"non-square DiT patch {patch} is unsupported (sub-domain sizing "
                "assumes a single square patch); rebuild with a square patch."
            )
        ph = patch[0]
        model_obj._patch_size = ph
        # Each side of the latent grid must fit the NATTEN kernel. Convert this minimum
        # to output-grid cells using the patch size, and verify any metadata value
        # against the loaded attention blocks.
        block_kernels = {
            int(k)
            for block in dit.blocks
            if (k := getattr(getattr(block, "attention", None), "attn_kernel", None))
            is not None
        }
        if len(block_kernels) > 1:
            raise ValueError(
                f"DiT attention blocks report inconsistent NATTEN kernels "
                f"{sorted(block_kernels)}; set_domain sub-domain sizing assumes a "
                "single kernel across the network."
            )
        kernel = block_kernels.pop() if block_kernels else None
        meta_arch = metadata.get("diffusion") or metadata.get("architecture") or {}
        meta_kernel = meta_arch.get("attn_kernel_size")
        if meta_kernel is not None:
            if kernel is not None and int(meta_kernel) != kernel:
                raise ValueError(
                    f"package metadata attn_kernel_size={meta_kernel} disagrees with "
                    f"the loaded DiT NATTEN kernel {kernel}; the architecture metadata "
                    "is inconsistent with the shipped checkpoint."
                )
            kernel = int(meta_kernel)
        if kernel is None:
            raise ValueError(
                "could not determine the DiT NATTEN attn_kernel: no attention block "
                "exposes it and the metadata carries no diffusion.attn_kernel_size. "
                "Sub-domain sizing (set_domain) would be unsafe; ship "
                "diffusion.attn_kernel_size in the package metadata."
            )
        model_obj._min_domain_cells = kernel * ph
        # Validate the full grid before inference or sub-domain selection.
        oh, ow = out_shape
        if oh % ph or ow % ph:
            raise ValueError(
                f"COSMO output grid {out_shape} is not divisible by the DiT patch "
                f"size {ph}; the tokenizer/detokenizer would mismatch."
            )
        if min(oh, ow) < model_obj._min_domain_cells:
            raise ValueError(
                f"COSMO output grid {out_shape} is below the "
                f"{model_obj._min_domain_cells}-cell NATTEN kernel minimum per side; "
                "the network cannot run on this grid."
            )
        if device is not None:
            model_obj = model_obj.to(device)
        return model_obj


def _to_datetime(t: "np.datetime64 | TimeArray") -> datetime:
    """Convert a NumPy datetime64 scalar to a timezone-naive UTC datetime."""
    return datetime.fromisoformat(str(np.datetime_as_string(np.asarray(t), unit="s")))
