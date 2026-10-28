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

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.models.da.utils import (
    dfseries_to_torch,
    filter_time_range,
    validate_observation_fields,
)
from earth2studio.models.dx.corrdiff_cosmo_era5 import CorrDiffCosmoEra5
from earth2studio.utils import coord_array, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordinateSystem, FrameSchema, TimeTolerance

if TYPE_CHECKING:
    from earth2studio.data.base import DataFrameSource, DataSource

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    OptionalDependencyFailure("cosmo")
    cKDTree = None

try:
    from physicsnemo.diffusion.guidance import (
        DataConsistencyDPSGuidance,
        DPSScorePredictor,
    )
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
except ImportError:
    OptionalDependencyFailure("cosmo")
    DataConsistencyDPSGuidance = None
    DPSScorePredictor = None
    EDMNoiseScheduler = None


@check_optional_dependencies()
class CorrDiffCosmoEra5SDA(torch.nn.Module, AutoModelMixin):
    """CorrDiff-COSMO with score-based data assimilation (SDA) via diffusion
    posterior sampling (DPS). Wraps a diffusion-mode :class:`CorrDiffCosmoEra5`
    downscaler and, at inference, nudges each denoising step toward sparse point
    observations -- producing a high-resolution COSMO-REA analysis over Europe that
    is conditioned on the ERA5 driving state and guided toward the observations.

    Takes as input:

    - An ERA5 driving state (the same input the downscaler conditions on).
    - Sparse point observations (a DataFrame of lat/lon/variable/observation).

    CorrDiff-COSMO directly models ``p(y | ERA5)``, so observations are mapped into
    its normalized output space without residual mean subtraction or addition. It is
    a single-shot diagnostic downscaler with no propagated state:
    :meth:`__call__` produces an independent analysis
    for each requested time, conditioned on the corresponding ERA5 state.

    Each observation is mapped to its nearest output-grid cell and normalized as
    ``(obs - center) / scale``. This direct mapping supports only identity-transform,
    unit-scale channels, such as ``u10m`` / ``v10m`` and ``u3d_l47`` / ``v3d_l47``.
    Other channels require a custom observation operator and are rejected.

    Note that the ``u3d_l*`` channels are terrain-following model levels: the
    above-ground height of ``u3d_l47`` is ``a + b * elevation_norm`` (in normalized
    elevation; nominally ~119.5 m at zero elevation anomaly). The exact height varies
    with terrain, so pick the level whose height suits the observations.

    Parameters
    ----------
    model : CorrDiffCosmoEra5
        A diffusion-mode downscaler (``mode="diffusion"``), already cropped to the
        target region with :meth:`CorrDiffCosmoEra5.set_domain` if desired.
    assimilate_variables : tuple[str, ...]
        Output variables to assimilate (required); must be identity-transform,
        unit-scale channels, e.g. ``("u10m", "v10m")`` (surface wind, available in
        either resolution) or ``("u3d_l47", "v3d_l47")`` (REA2 terrain-following
        model-level wind; above-ground height varies with elevation, see the note
        above). Choose the channel whose physical height matches your observations --
        there is no default, since the choice depends on the observations being
        assimilated.
    time_tolerance : TimeTolerance, optional
        Observations within this window of the analysis time are used. A single
        value is symmetric; the default is ±10 minutes.
    number_of_samples : int | None, optional
        Posterior ensemble size (independent DPS draws, seeds ``seed + i``);
        defaults to the wrapped model's ``number_of_samples``.
    sampler_steps : int | None, optional
        Diffusion sampler steps; defaults to the wrapped model's
        ``number_of_steps``.
    sda_std_obs : float | Mapping[str, float], optional
        Observation-noise standard deviation for DPS guidance, by default 0.75.
        A scalar is broadcast to every assimilated variable (in that variable's
        physical units); a mapping sets it per variable and must contain exactly the
        assimilated variables (unknown keys are rejected) -- e.g.
        ``{"u10m": 0.5, "v10m": 0.5}`` in m/s. When temperature is also
        assimilated, its entry could be ``"t2m": 1.0`` in K. Every value must be
        finite and > 0. It is the effective uncertainty per occupied grid cell
        (multiple observations in one cell are averaged, not reduced by sqrt(n)).
    sda_gamma : float, optional
        SDA covariance scaling in the DPS likelihood, by default 7.5e-5. Positive
        values account for denoiser-estimate uncertainty across diffusion noise
        levels. Larger values weaken observation guidance, especially early in
        denoising, so the analysis stays closer to the unguided downscaler and may
        fit observations less closely. Set to 0 for classical DPS without this
        correction and the strongest guidance for a fixed ``sda_std_obs``. Must be
        >= 0.
    amp : bool, optional
        Run the guided diffusion sampler under ``torch.autocast`` bf16, by default
        False (full precision, matching the wrapped downscaler). Setting True can
        reduce runtime and peak GPU memory; it takes effect on CUDA only (CPU always
        runs full precision).

    solver : str | None, optional
        Select the numerical method for integrating the sampling ODE in reverse
        time. ``None`` preserves the wrapped model's solver, by default None
    solver_options : dict[str, Any] | None, optional
        Options passed to the selected solver to customize its behavior or
        override defaults. Do not pass ``num_steps`` here; set the sampling step
        count with ``sampler_steps`` instead. By default None

    Badges
    ------
    region:global class:data-assimilation class:downscaling product:wind product:temp
    product:precip product:atmos product:insitu year:2026 gpu:80gb
    provider:nvidia backend:pytorch
    """

    stochastic = True
    _rng: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the independent posterior sample stream.

        Parameters
        ----------
        seed : int
            Seed for a local CPU generator; global RNG state is unchanged.
        reset : bool, optional
            Restart the stream when True; otherwise initialize only if absent.
            Defaults to True.
        """
        if reset or self._rng is None:
            self._rng = torch.Generator().manual_seed(seed)

    def __init__(
        self,
        model: CorrDiffCosmoEra5,
        assimilate_variables: tuple[str, ...],
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        number_of_samples: int | None = None,
        sampler_steps: int | None = None,
        sda_std_obs: float | Mapping[str, float] = 0.75,
        sda_gamma: float = 7.5e-5,
        amp: bool = False,
        *,
        solver: str | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if model.mode != "diffusion":
            raise ValueError(
                "CorrDiffCosmoEra5SDA requires a diffusion-mode model "
                f'(got mode={model.mode!r}); load with mode="diffusion".'
            )
        if not np.isfinite(sda_gamma) or sda_gamma < 0:
            raise ValueError(
                f"sda_gamma must be a finite value >= 0 (got {sda_gamma})."
            )
        if not assimilate_variables:
            raise ValueError("assimilate_variables must be non-empty.")
        self.model = model
        self.solver = solver
        if solver_options is not None and "num_steps" in solver_options:
            raise ValueError(
                "Do not pass 'num_steps' in solver_options; use sampler_steps instead."
            )
        self.solver_options = dict(solver_options or {})
        self._tolerance = normalize_time_tolerance(time_tolerance)
        self.number_of_samples = (
            model.number_of_samples if number_of_samples is None else number_of_samples
        )
        if self.number_of_samples < 1:
            raise ValueError(
                f"number_of_samples must be >= 1 (got {self.number_of_samples})."
            )
        self.sampler_steps = (
            model.number_of_steps if sampler_steps is None else sampler_steps
        )
        if self.sampler_steps < 2:
            raise ValueError(
                "sampler_steps must be >= 2 for the EDM schedule "
                f"(got {self.sampler_steps})."
            )
        # Per-variable observation-noise std (in each variable's physical units). A
        # scalar broadcasts to every assimilated variable; a mapping must contain
        # exactly the assimilated variables. Copy the values into a dict keyed by
        # variable so mapping order does not matter and later changes to the caller's
        # mapping do not affect the model.
        if isinstance(sda_std_obs, Mapping):
            configured = set(assimilate_variables)
            missing = [v for v in assimilate_variables if v not in sda_std_obs]
            if missing:
                raise ValueError(
                    f"sda_std_obs mapping is missing entries for {missing}."
                )
            extra = [k for k in sda_std_obs if k not in configured]
            if extra:
                raise ValueError(
                    f"sda_std_obs mapping has entries for non-assimilated "
                    f"variables {extra}."
                )
            obs_std = {v: float(sda_std_obs[v]) for v in assimilate_variables}
        else:
            obs_std = {v: float(sda_std_obs) for v in assimilate_variables}
        for v, std in obs_std.items():
            if not np.isfinite(std) or std <= 0:
                raise ValueError(
                    f"sda_std_obs for {v!r} must be > 0 and finite (got {std})."
                )
        self._obs_std = obs_std
        self.sda_dps_norm = 2
        self.sda_gamma = sda_gamma
        self.amp = amp

        # Place observations on the model's current denoise grid -- the grid the
        # denoiser runs on, including any halo/patch-padding cells (a cropped sub-domain
        # if set_domain was used, else the full footprint). DPS is applied here; the
        # returned output is halo-trimmed (when applicable) to model.output_coords. Use
        # 3D points on the unit sphere for nearest-cell lookup so longitude wrap-around
        # is handled.
        self._lat_np = model.lat_output_grid.detach().cpu().numpy()
        self._lon_np = model.lon_output_grid.detach().cpu().numpy()
        gxyz = self._latlon_to_xyz(self._lat_np.ravel(), self._lon_np.ravel())
        self._grid_tree = cKDTree(gxyz)
        # Reject obs that fall outside the domain: anything farther than ~1.5 cells
        # (chord length) from every grid point (nearest-cell snapping alone would
        # pin distant obs to the border, as StormCast avoids via point-in-polygon).
        g2d = gxyz.reshape(*self._lat_np.shape, 3)
        # Estimate typical spacing between neighboring grid points in both directions.
        # The median is insensitive to unusual spacing near edges or isolated outliers.
        row_spacing = np.median(np.linalg.norm(np.diff(g2d, axis=0), axis=-1))
        column_spacing = np.median(np.linalg.norm(np.diff(g2d, axis=1), axis=-1))
        self._obs_max_dist = 1.5 * float(np.hypot(row_spacing, column_spacing))

        # Assimilable channels: the dx model's identity-transform, unit-scale
        # outputs. A nonlinear transform or a lexicon unit scale != 1 cannot be
        # mapped linearly into normalized output space. The dx model owns this
        # classification (``_identity_output_indices``, deny-by-default so a future
        # transform type is excluded automatically); keys are the Earth2Studio
        # lexicon names taken from the public output_coords vocabulary.
        names = model.output_coords(model.input_coords())["variable"].values.tolist()
        self._out_idx = {names[i]: i for i in model._identity_output_indices}
        for v in assimilate_variables:
            if v not in self._out_idx:
                raise ValueError(
                    f"assimilate_variables contains {v!r}, which is not an "
                    "assimilable (identity-transform, unit-scale) output channel "
                    f"(assimilable channels: {sorted(self._out_idx)})."
                )
        self.assimilate_variables = list(assimilate_variables)

    @property
    def device(self) -> torch.device:
        """Device the assimilation model lives on.

        Derived from the wrapped downscaler's registered grid buffer -- the single
        authoritative device -- so it tracks the model even if the model is moved
        directly (no duplicated wrapper state to diverge).

        Returns
        -------
        torch.device
            The device of the assimilation model.
        """
        return self.model.lat_output_grid.device

    @staticmethod
    def _latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Convert latitude/longitude to Cartesian points on a unit sphere.

        Euclidean distance between these points is the chord length: the straight-line
        distance through 3D space. It handles longitude wrap-around and preserves
        geographic nearest-neighbour ordering.
        """
        latitude_radians = np.deg2rad(np.asarray(lat, dtype=np.float64))
        longitude_radians = np.deg2rad(np.asarray(lon, dtype=np.float64))
        cos_latitude = np.cos(latitude_radians)
        return np.stack(
            [
                cos_latitude * np.cos(longitude_radians),
                cos_latitude * np.sin(longitude_radians),
                np.sin(latitude_radians),
            ],
            axis=-1,
        )

    # ── coordinate systems ───────────────────────────────────────────────────

    def input_coords(self) -> tuple[CoordinateSystem, FrameSchema]:
        """Declare ERA5 background fields and optional surface observations.

        Returns
        -------
        tuple[CoordinateSystem, FrameSchema]
            ERA5 signature followed by the observation schema. Observations are
            filtered around each background time with the configured tolerance.
            Pass None for the observation argument to run unguided downscaling.
        """
        signature = self.model.input_coords()
        background = coord_array(
            ("time", "variable", "lat", "lon"),
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "variable": signature["variable"].values,
                "lat": signature.lat.values,
                "lon": signature.lon.values,
            },
            dynamic=("time",),
        )
        observations = FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(self.assimilate_variables, dtype=str),
            }
        )
        return background, observations

    def default_sources(self) -> tuple[DataSource, DataFrameSource]:
        """Recommend ARCO ERA5 initialization and GHCNh observation data sources.

        Returns
        -------
        tuple[DataSource, DataFrameSource]
            ``ARCO_ERA5`` for the ERA5 driving state followed by ``GHCNHourly`` for
            the surface station observation slot, restricted to the stations inside
            the model's output domain and using the model's observation time
            tolerance.
        """
        from earth2studio.data import ARCO_ERA5, GHCNHourly

        lat = self.model.lat_output_numpy
        lon = self.model.lon_output_numpy
        bbox = (float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max()))
        stations = GHCNHourly.get_stations_bbox(bbox)
        return ARCO_ERA5(), GHCNHourly(
            stations=stations, time_tolerance=self._tolerance, verbose=False
        )

    def output_coords(
        self, input_coords: CoordinateSystem, obs: FrameSchema | None = None
    ) -> CoordinateSystem:
        """Output coordinate system of the assimilation model.

        Matches the dimensions returned by :meth:`__call__` on the COSMO-REA
        analysis grid, given the ERA5 initialization coordinates.

        Parameters
        ----------
        input_coords : CoordinateSystem
            The ERA5 driving-state signature (from :meth:`input_coords`),
            validated against the wrapped model's native input grid.
        obs : FrameSchema | None, optional
            Observation schema; output geometry follows the background signature.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with dims
            ``(time, sample, variable, y, x)`` with 2D ``lat``/``lon`` on the
            analysis grid.
        """
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "lon", -1)
        oc = self.model.output_coords(input_coords)
        lat2d, lon2d = np.asarray(oc["lat"]), np.asarray(oc["lon"])
        h, w = lat2d.shape
        return coord_array(
            ("time", "sample", "variable", "y", "x"),
            {
                "time": input_coords["time"].values,
                "sample": np.arange(self.number_of_samples),
                "variable": np.asarray(oc["variable"]),
                "y": np.arange(h),
                "x": np.arange(w),
                "lat": (("y", "x"), lat2d),
                "lon": (("y", "x"), lon2d),
            },
            dynamic=input_coords.attrs.get("earth2studio_dynamic_dims", ()),
        )

    # ── observations ─────────────────────────────────────────────────────────

    def _build_obs_tensors(
        self,
        obs: pd.DataFrame | None,
        request_time: np.datetime64,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse obs DataFrame -> (y, mask, std_y) on the denoise grid, in the
        model's normalized output space.

        DIRECT model: y = (observation - out_center) / out_scale (no residual). Each
        observation is snapped to the nearest denoise-grid cell; out-of-domain and
        non-finite obs are dropped, multiple obs in a cell are averaged. ``std_y`` is
        1 everywhere except observed cells, which carry the (normalized) obs noise.
        """
        n_out = len(self.model.output_variables)
        hm, wm = self._lat_np.shape
        y = torch.zeros(1, n_out, hm, wm, device=device)
        mask = torch.zeros_like(y)
        std_y = torch.ones_like(y)
        if obs is None or len(obs) == 0:
            return y, mask, std_y

        validate_observation_fields(obs, required_fields=list(self.input_coords()[1]))

        tf = filter_time_range(obs, request_time, self._tolerance, time_column="time")
        if len(tf) == 0:
            return y, mask, std_y

        # lat/lon/variable must be host arrays for the (CPU) KDTree lookup and the
        # string comparison below; the observation values go straight to the device
        # (zero-copy from cudf via dfseries_to_torch).
        obs_lat = tf["lat"].to_numpy().astype(np.float64)
        obs_lon = tf["lon"].to_numpy().astype(np.float64)
        obs_var = tf["variable"].to_numpy().astype(str)
        # fillna keeps the cudf dlpack path null-safe (nulls -> NaN, then dropped by
        # the on-device isfinite filter below); a no-op for float pandas values.
        obs_val_t = dfseries_to_torch(
            tf["observation"].fillna(float("nan")), dtype=torch.float32, device=device
        )

        # Nearest denoise-grid cell (3D chord distance) + in-domain/finite masks.
        finite_coordinates = np.isfinite(obs_lat) & np.isfinite(obs_lon)
        oxyz = self._latlon_to_xyz(
            np.where(finite_coordinates, obs_lat, 0.0),
            np.where(finite_coordinates, obs_lon, 0.0),
        )
        dist, flat = self._grid_tree.query(oxyz)
        ci, cj = np.unravel_index(flat, (hm, wm))
        keep = finite_coordinates & (dist <= self._obs_max_dist)

        cen = self.model.out_center
        sca = self.model.out_scale
        for v in self.assimilate_variables:
            sel = keep & (obs_var == v)
            if not sel.any():
                continue
            k = self._out_idx[v]
            c = cen[0, k, 0, 0].item()
            s = sca[0, k, 0, 0].item()
            if not np.isfinite(s) or s == 0:
                continue
            sel_t = torch.as_tensor(sel, device=device)
            yv = (obs_val_t[sel_t] - c) / s
            yi = torch.as_tensor(ci[sel], device=device, dtype=torch.long)
            xi = torch.as_tensor(cj[sel], device=device, dtype=torch.long)
            flat_idx = yi * wm + xi
            # Drop non-finite observation values on-device (no host round-trip);
            # cells left with no finite obs stay unobserved via cnt == 0 below.
            finite = torch.isfinite(yv)
            yv = yv[finite]
            flat_idx = flat_idx[finite]
            acc = torch.zeros(hm * wm, device=device)
            cnt = torch.zeros_like(acc)
            acc.scatter_add_(0, flat_idx, yv)
            cnt.scatter_add_(0, flat_idx, torch.ones_like(yv))
            occ = cnt > 0
            # Empty cells have cnt == 0; clamp to 1 so we never divide by zero.
            y[0, k] = torch.where(occ, acc / cnt.clamp(min=1), acc).view(hm, wm)
            mask[0, k] = occ.float().view(hm, wm)
            # Store the normalized observation uncertainty at observed cells.
            # Elsewhere, unit uncertainty keeps the guidance denominator finite
            # (zero could cause division by zero when gamma=0). The mask marks those
            # cells as unobserved, so y=0 is not treated as a target there.
            std_y[0, k] = torch.where(
                occ, torch.full_like(acc, self._obs_std[v] / s), torch.ones_like(acc)
            ).view(hm, wm)
        return y, mask, std_y

    # ── DPS-guided sampling ──────────────────────────────────────────────────

    def _run_diffusion(
        self,
        background: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        std_y: torch.Tensor,
        seed: int | None,
    ) -> torch.Tensor:
        """One observation-constrained diffusion sample (normalized output space).

        Reuses the wrapped downscaler's :meth:`CorrDiffCosmoEra5._denoise` sampling
        loop via its score-predictor seam, injecting a DPS-guided
        ``DPSScorePredictor`` (same physicsnemo primitives as StormCastSDA) in place
        of the plain denoiser. The caller (:meth:`_forward`) runs under
        ``torch.no_grad``; DPS re-enables gradients locally for its score correction,
        so the sampler's own arithmetic stays graph-free (memory) while the guidance
        term is still differentiated. Direct model: the conditioning is the
        normalized ERA5 background, not a residual mean.
        """

        def make_score_predictor(
            x0_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            scheduler: EDMNoiseScheduler,
        ) -> DPSScorePredictor:
            guidance = DataConsistencyDPSGuidance(
                mask,
                y,
                std_y,
                norm=self.sda_dps_norm,
                gamma=self.sda_gamma,
                sigma_fn=scheduler.sigma,
                alpha_fn=scheduler.alpha,
            )
            return DPSScorePredictor(
                x0_predictor=x0_predictor,
                x0_to_score_fn=scheduler.x0_to_score,
                guidances=guidance,
            )

        # bf16 autocast (amp) around the guided sampler: _denoise uses nullcontext for
        # the guided path, so this outer autocast is what enables mixed precision. DPS
        # re-enables grad inside; bf16 autograd through the denoiser is stable here.
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.amp and background.device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            return self.model._denoise(
                background,
                seed,
                score_predictor_factory=make_score_predictor,
                num_steps=self.sampler_steps,
                solver=self.solver,
                solver_options=self.solver_options,
            )

    @torch.no_grad()
    def _forward(
        self,
        era5: torch.Tensor,
        valid_time: datetime,
        y: torch.Tensor,
        mask: torch.Tensor,
        std_y: torch.Tensor,
    ) -> torch.Tensor:
        """One ERA5 frame + obs -> physical posterior ensemble [n_samples, C_out,
        H', W'] (halo/patch trimmed to the reported output grid). ``@torch.no_grad``
        keeps the sampler graph-free; DPS re-enables grad internally for guidance."""
        m = self.model
        top, bot, left, right = m._halo
        background = m.preprocess_input(
            era5, valid_time, m.lat_output_grid, m.lon_output_grid
        )
        members = []
        for _ in range(self.number_of_samples):
            seed = int(torch.randint(2**63 - 1, (), generator=self._rng).item())
            normalized = self._run_diffusion(background, y, mask, std_y, seed)
            out = m.postprocess_output(
                normalized, valid_time, self._lat_np, self._lon_np
            )
            if top or bot or left or right:
                out = out[..., top : out.shape[-2] - bot, left : out.shape[-1] - right]
            members.append(out)
        return torch.cat(members, dim=0)

    # ── public API ───────────────────────────────────────────────────────────

    @staticmethod
    def _as_datetime(t: np.datetime64) -> datetime:
        dt = pd.Timestamp(t).to_pydatetime()
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    def _require_cupy_for_cuda(self) -> None:
        """Guard the same-device output contract: on CUDA the analysis must be
        returned as a CuPy array, so CuPy is required there (rather than silently
        falling back to CPU NumPy). Checked at execution entry so it fails before any
        sampling, not after."""
        if self.device.type == "cuda" and cp is None:
            raise RuntimeError(
                "CUDA execution requires CuPy to return the analysis on-device "
                "(AssimilationModel contract). Install the 'da-cosmo' extra, or "
                "run the model on CPU."
            )

    def _to_output_dataarray(
        self, out: torch.Tensor, times: np.ndarray, oc: CoordinateSystem
    ) -> xr.DataArray:
        """[n_time, n_sample, C_out, H', W'] -> DataArray with 2D lat/lon coords."""
        # Same-device analysis per the AssimilationModel contract: CuPy on CUDA,
        # NumPy on CPU. (Backstop -- the primary check runs at execution entry.)
        self._require_cupy_for_cuda()
        device = self.device
        if device.type == "cuda" and cp is not None:
            with cp.cuda.Device(device.index or 0):
                data = cp.asarray(out.detach())
        else:
            data = out.detach().cpu().numpy()
        return xr.DataArray(
            data=data,
            dims=["time", "sample", "variable", "y", "x"],
            coords={
                "time": times,
                "sample": np.arange(out.shape[1]),
                "variable": np.asarray(oc["variable"]),
                "y": np.arange(out.shape[-2]),
                "x": np.arange(out.shape[-1]),
                "lat": (["y", "x"], np.asarray(oc["lat"])),
                "lon": (["y", "x"], np.asarray(oc["lon"])),
            },
        )

    @staticmethod
    def _era5_frames(x: xr.DataArray) -> xr.DataArray:
        """Normalize the ERA5 driving DataArray to dims (time, variable, lat, lon)."""
        if "lead_time" in x.dims:
            if x.sizes["lead_time"] != 1:
                raise ValueError(
                    "x must have a size-1 lead_time (got "
                    f"{x.sizes['lead_time']}); pass one lead time at a time."
                )
            x = x.isel(lead_time=0, drop=True)
        return x.transpose("time", "variable", "lat", "lon")

    # NOTE: no @torch.inference_mode() -- DPS guidance differentiates through the
    # denoiser, so gradients must stay enabled. Memory is
    # bounded by @torch.no_grad() on _forward.
    def __call__(
        self,
        x: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Assimilate ``obs`` into the downscaled analysis for each time in ``x``.

        Parameters
        ----------
        x : xr.DataArray
            ERA5 driving state on the native input grid (dims include ``time``,
            ``variable``, ``lat``, ``lon``; a size-1 ``lead_time`` is squeezed).
        obs : pd.DataFrame | None, optional
            Sparse observations (columns ``time``/``lat``/``lon``/``variable``/
            ``observation``), or ``None`` for a free (unconstrained) downscaling.

        Returns
        -------
        xr.DataArray
            Posterior COSMO-REA analysis, dims ``(time, sample, variable, y, x)``
            with 2D ``lat``/``lon`` coordinates.

        Raises
        ------
        ValueError
            If ``x`` carries a ``lead_time`` dimension of size != 1, or if ``obs``
            is missing a required column.
        RuntimeError
            If the model is on CUDA but CuPy is not installed (the analysis cannot
            be returned on-device; install the ``da-cosmo`` extra or run on CPU).
        """
        self._require_cupy_for_cuda()  # fail before sampling, not after
        device = self.device
        x = self._era5_frames(x)
        oc = self.output_coords(x)

        x_tensor, _ = x.e2s.to_torch()
        x_tensor = x_tensor.to(device=device, dtype=torch.float32)
        times = np.atleast_1d(x.coords["time"].values)
        outs = []
        for j, t in enumerate(times):
            y, mask, std_y = self._build_obs_tensors(obs, t, device)
            outs.append(
                self._forward(x_tensor[j], self._as_datetime(t), y, mask, std_y)
            )
        out = torch.stack(outs, dim=0)  # [n_time, n_sample, C_out, H', W']
        return self._to_output_dataarray(out, times, oc)

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the CorrDiff-COSMO model package (shared with the downscaler).

        Returns
        -------
        Package
            The default CorrDiff-COSMO model package.
        """
        return CorrDiffCosmoEra5.load_default_package()

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        assimilate_variables: tuple[str, ...],
        resolution: str = "rea2",
        domain: dict | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        number_of_samples: int | None = None,
        sampler_steps: int | None = None,
        sda_std_obs: float | Mapping[str, float] = 0.75,
        sda_gamma: float = 7.5e-5,
        amp: bool = False,
        *,
        solver: str | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> AssimilationModel:
        """Load the assimilation model from a CorrDiff-COSMO package.

        Loads a diffusion-mode :class:`CorrDiffCosmoEra5` internally and wraps it.

        ``domain`` (optional) crops the downscaler to a sub-region BEFORE wrapping,
        so the observation grid stays in sync; it is forwarded verbatim to
        :meth:`CorrDiffCosmoEra5.set_domain` (e.g.
        ``domain=dict(lat_min=52.5, lat_max=55.5, lon_min=5.5, lon_max=9.5)``).
        Cropping the wrapped ``.model`` afterwards would leave a stale obs grid.

        Parameters
        ----------
        package : Package
            CorrDiff-COSMO model package (see :meth:`load_default_package`).
        assimilate_variables : tuple[str, ...]
            Output channels to assimilate (required); must be identity-transform,
            unit-scale channels (e.g. ``("u10m", "v10m")`` surface wind, available in
            either resolution, or ``("u3d_l47", "v3d_l47")`` REA2 terrain-following
            model-level wind whose above-ground height varies with elevation). There
            is no default, since the choice depends on the observations being
            assimilated.
        resolution : str, optional
            COSMO-REA resolution, ``"rea2"`` or ``"rea6"``, by default ``"rea2"``.
        domain : dict | None, optional
            Sub-region crop forwarded to :meth:`CorrDiffCosmoEra5.set_domain`
            (keys ``lat_min``/``lat_max``/``lon_min``/``lon_max``), by default
            ``None`` (full native footprint).
        time_tolerance : TimeTolerance, optional
            Window for matching observation times to the analysis time, by default
            10 minutes.
        number_of_samples : int | None, optional
            Posterior ensemble size; defaults to the wrapped model's value.
        sampler_steps : int | None, optional
            Number of diffusion sampler steps; defaults to the wrapped model's
            ``number_of_steps``.
        sda_std_obs : float | Mapping[str, float], optional
            Assumed observation-noise std (lower trusts obs more), by default 0.75.
            A scalar broadcasts to every variable; a mapping sets it per variable and
            must contain exactly the assimilated variables (e.g.
            ``{"u10m": 0.5, "v10m": 0.5}``).
        sda_gamma : float, optional
            SDA covariance scaling in the DPS likelihood, by default 7.5e-5. Positive
            values account for denoiser-estimate uncertainty across diffusion noise
            levels. Larger values weaken observation guidance, especially early in
            denoising, so the analysis stays closer to the unguided downscaler and may
            fit observations less closely. Set to 0 for classical DPS without this
            correction and the strongest guidance for a fixed ``sda_std_obs``. Must
            be >= 0.
        amp : bool, optional
            Run the guided sampler under ``torch.autocast`` bf16 (CUDA only), by
            default False. Setting True can reduce runtime and GPU memory.

        solver : str | None, optional
            Select the numerical method for integrating the sampling ODE in
            reverse time. ``None`` preserves the wrapped model's solver, by
            default None
        solver_options : dict[str, Any] | None, optional
            Options passed to the selected solver to customize its behavior or
            override defaults. Do not pass ``num_steps`` here; set the sampling step
            count with ``sampler_steps`` instead. By default None

        Returns
        -------
        AssimilationModel
            The wrapped :class:`CorrDiffCosmoEra5SDA` assimilation model.
        """
        model = CorrDiffCosmoEra5.load_model(
            package, mode="diffusion", resolution=resolution
        )
        if domain is not None:
            model = model.set_domain(**domain)
        return cls(
            model,
            time_tolerance=time_tolerance,
            assimilate_variables=assimilate_variables,
            number_of_samples=number_of_samples,
            sampler_steps=sampler_steps,
            solver=solver,
            solver_options=solver_options,
            sda_std_obs=sda_std_obs,
            sda_gamma=sda_gamma,
            amp=amp,
        )

    def to(self, device: str | torch.device) -> CorrDiffCosmoEra5SDA:
        """Move the model (and wrapped downscaler) to ``device``.

        Parameters
        ----------
        device : str | torch.device
            Target device to move the model to.

        Returns
        -------
        CorrDiffCosmoEra5SDA
            This model, moved in place (returned for chaining).
        """
        # self.model is a registered submodule and is moved recursively by super().to.
        super().to(device)
        return self
