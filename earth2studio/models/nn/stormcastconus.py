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

import warnings
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from datetime import datetime
from itertools import product
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

from earth2studio.data import GFS_FX, HRRR, DataSource, ForecastSource, fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.coords import map_coords
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.obs import ObsGridMapping
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.guidance import (
        DataConsistencyDPSGuidance,
        DPSScorePredictor,
    )
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
    from tensordict import TensorDict
except ImportError:
    OptionalDependencyFailure("stormcast-conus")
    OmegaConf = None
    TensorDict = None


# Variables used in StormCastCONUS
VARIABLES = (
    ["u10m", "v10m", "t2m", "msl"]
    + [
        var + str(level)
        for var, level in product(
            ["u", "v", "t", "q", "Z", "p"],
            map(
                lambda x: str(x) + "hl",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30],
            ),
        )
        if not ((var == "p") and (int(level.replace("hl", "")) > 20))
    ]
    + [
        "refc",
    ]
)

CONDITIONING_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "sp", "msl"] + [
    var + str(level)
    for var, level in product(["u", "v", "z", "t", "q"], [1000, 850, 500, 250])
]

FULL_MODEL_HRRR_BBOX = ((17, 1041), (3, 1795))


@check_optional_dependencies()
class StormCastCONUSBase(torch.nn.Module, AutoModelMixin):
    """StormCast-CONUS generative convection-allowing model for the full CONUS domain.

    - High-resolution (3km) HRRR state over the Continental United States (99 vars)
    - High-resolution land-sea mask and orography invariants
    - Coarse resolution (25km) global state (26 vars)

    The high-resolution grid is the HRRR Lambert conformal projection
    Coarse-resolution inputs are regridded to the HRRR grid internally.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2408.10958
    - https://huggingface.co/nvidia/stormcast-v1-era5-hrrr

    Parameters
    ----------
    diffusion_model : torch.nn.Module
        Configured diffusion model (e.g. a :class:`_SplitModelWrapper` instance
        created by :meth:`load_model`). Must be a :class:`_SplitModelWrapper` instance
        to set ``hrrr_lat_lim`` or ``hrrr_lon_lim`` to non-default values.
    means : torch.Tensor
        Per-channel mean for normalising the high-resolution state.
    stds : torch.Tensor
        Per-channel standard deviation for normalising the high-resolution state.
    invariants : torch.Tensor
        Static invariant fields (e.g. land-sea mask, orography).
    conditioning_means : torch.Tensor
        Per-channel mean for normalising the low-resolution conditioning.
    conditioning_stds : torch.Tensor
        Per-channel standard deviation for normalising the low-resolution conditioning.
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, defaults to be the StormCastCONUS region in Continental
        United States, by default (17, 1041)
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, defaults to be the StormCastCONUS region in Continental
        United States, by default (3, 1795)
    variables : np.ndarray, optional
        High-resolution variable names, by default ``np.array(VARIABLES)``.
    conditioning_variables : np.ndarray, optional
        Low-resolution conditioning variable names, by default
        ``np.array(CONDITIONING_VARIABLES)``.
    conditioning_data_source : DataSource or ForecastSource or None, optional
        Data source for global conditioning. Required for inference, by default None.
    sampler_args : dict, optional
        Overrides for the EDM sampler/scheduler. Recognised keys:
        ``sigma_min``, ``sigma_max``, ``rho`` (scheduler), and
        ``S_churn``, ``S_min``, ``S_max``, ``S_noise`` (solver).
        Unspecified keys use sensible defaults.
    num_diffusion_steps : int, optional
        Number of diffusion sampling steps for the EDM (no-obs) path, by default 18.
    num_sda_diffusion_steps : int, optional
        Number of diffusion sampling steps for the SDA (obs-guided) path, by default 36.
    batch_size : int, optional
        Maximum batch size processed in one forward pass, by default 1.
    time_tolerance : np.timedelta64, tuple[np.timedelta64, np.timedelta64], or None, optional
        Time window for filtering observations around each target time.
        A single ``np.timedelta64`` creates a symmetric window; a 2-tuple
        ``(lower, upper)`` is passed directly to ``ObsGridMapping.obs_to_grid``.
        ``None`` disables time filtering, by default None.
    sda_std_obs : float or dict[str, float], optional
        Observation noise standard deviation (in physical units) used by the
        SDA denoiser when observations are provided. A scalar applies uniformly;
        a dict maps variable names to per-variable values (default 0.1 for
        unlisted variables). Dict values are converted to normalised units
        internally, by default 0.1.
    sda_dps_norm : float, optional
        Gradient normalisation factor for DPS guidance, by default 2.
    sda_gamma : float, optional
        DPS guidance step size / scale, by default 0.001.
    use_amp : bool, optional
        Whether to run the diffusion forward pass under ``torch.autocast`` with
        bfloat16, by default True.
    clamp_values : bool, optional
        Whether to apply reflectivity clipping in ``_forward``. When the model is
        loaded via :meth:`load_model`, this flag is also forwarded to
        :class:`_SplitModelWrapper` to enable per-variable physical-minimum clamping,
        by default True.
    """

    def __init__(
        self,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        conditioning_means: torch.Tensor,
        conditioning_stds: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (17, 1041),
        hrrr_lon_lim: tuple[int, int] = (3, 1795),
        variables: np.ndarray = np.array(VARIABLES),
        conditioning_variables: np.ndarray = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | ForecastSource | None = None,
        sampler_args: dict[str, float | int] | None = None,
        num_diffusion_steps: int = 18,
        num_sda_diffusion_steps: int = 36,
        batch_size: int = 1,
        time_tolerance: (
            np.timedelta64 | tuple[np.timedelta64, np.timedelta64] | None
        ) = None,
        sda_std_obs: float | dict[str, float] = 0.1,
        sda_dps_norm: float = 2,
        sda_gamma: float = 0.001,
        use_amp: bool = True,
        clamp_values: bool = True,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.sampler_args = {
            "sigma_min": 0.01,
            "sigma_max": 200,
            "sigma_data": 1.0,
            "rho": 7,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1,
        }
        if sampler_args is not None:
            self.sampler_args.update(sampler_args)

        if (hrrr_lat_lim, hrrr_lon_lim) != FULL_MODEL_HRRR_BBOX:
            if not isinstance(self.diffusion_model, _SplitModelWrapper):
                raise ValueError("To crop the model to a subdomain, diffusion_model must be _SplitModelWrapper.")

            p = self.diffusion_model.model_high.model.model.patch_size
            if (hrrr_lat_lim[0] - FULL_MODEL_HRRR_BBOX[0][0]) % p[0]:
                raise ValueError(
                    f"hrrr_lat_lim[0] - {FULL_MODEL_HRRR_BBOX[0]} must be divisible by {p[0]}"
                )
            if (hrrr_lat_lim[1] - hrrr_lat_lim[0]) % p[0]:
                raise ValueError(
                    f"hrrr_lat_lim[1] - hrrr_lat_lim[0] must be divisible by {p[0]}"
                )
            if (hrrr_lon_lim[0] - FULL_MODEL_HRRR_BBOX[1][0]) % p[1]:
                raise ValueError(
                    f"hrrr_lon_lim[0] - {FULL_MODEL_HRRR_BBOX[1]} must be divisible by {p[1]}"
                )
            if (hrrr_lon_lim[1] - hrrr_lon_lim[0]) % p[1]:
                raise ValueError(
                    f"hrrr_lon_lim[1] - hrrr_lon_lim[0] must be divisible by {p[1]}"
                )

            self.diffusion_model.crop_model((hrrr_lat_lim, hrrr_lon_lim))

        hrrr_lat, hrrr_lon = HRRR.grid()
        self.lat = hrrr_lat[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.lon = hrrr_lon[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.hrrr_x = HRRR.HRRR_X[hrrr_lon_lim[0] : hrrr_lon_lim[1]]
        self.hrrr_y = HRRR.HRRR_Y[hrrr_lat_lim[0] : hrrr_lat_lim[1]]

        self.variables = variables
        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source was provided to StormCastCONUS, "
                "set the conditioning_data_source attribute of the model "
                "before running inference."
            )
        self.register_buffer("conditioning_means", conditioning_means)
        self.register_buffer("conditioning_stds", conditioning_stds)

        # Cache grid limits and precomputed normalisation factors
        self.hrrr_lat_lim = hrrr_lat_lim
        self.hrrr_lon_lim = hrrr_lon_lim
        self.register_buffer("lat_tensor", torch.as_tensor(self.lat))
        self.register_buffer("lon_tensor", torch.as_tensor(self.lon))
        self.register_buffer("mean_inv_std", -self.means / self.stds)
        self.register_buffer("inv_std", 1.0 / self.stds)
        self.register_buffer(
            "cond_mean_inv_std", -self.conditioning_means / self.conditioning_stds
        )
        self.register_buffer("cond_inv_std", 1.0 / self.conditioning_stds)
        _scheduler_keys = {"sigma_min", "sigma_max", "sigma_data", "rho"}
        self.scheduler = EDMNoiseScheduler(
            **{k: v for k, v in self.sampler_args.items() if k in _scheduler_keys}
        )

        self.num_diffusion_steps = num_diffusion_steps
        self.num_sda_diffusion_steps = num_sda_diffusion_steps
        self.batch_size = batch_size
        self.time_tolerance = time_tolerance
        self.obs_grid_mapping: ObsGridMapping | None = None
        if isinstance(sda_std_obs, dict):
            std_obs = torch.as_tensor(
                [sda_std_obs.get(v, 0.1) for v in self.variables], dtype=torch.float32
            )
            std_obs = std_obs[None, :, None, None] * self.inv_std
            self.register_buffer("sda_std_obs", std_obs)
        else:
            self.sda_std_obs = sda_std_obs
        self.sda_dps_norm = sda_dps_norm
        self.sda_gamma = sda_gamma
        self.use_amp = use_amp
        self.clamp_values = clamp_values
        self.refc_channel = list(variables).index("refc")

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output coordinates.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

        target_input_coords = self.input_coords()

        handshake_dim(input_coords, "hrrr_x", 5)
        handshake_dim(input_coords, "hrrr_y", 4)
        handshake_dim(input_coords, "variable", 3)
        # Index coords are arbitrary as long its on the HRRR grid, so just check size
        handshake_size(input_coords, "hrrr_y", self.lat.shape[0])
        handshake_size(input_coords, "hrrr_x", self.lat.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    def get_obs_mapping(self, device: torch.device | None) -> ObsGridMapping:
        """Return the obs grid mapping, (re-)creating it if the device has changed.

        Building the ``LinearNDInterpolator`` over the full HRRR grid is
        expensive, so the mapping is cached and only rebuilt on a device change.
        """
        if self.obs_grid_mapping is None or self.obs_grid_mapping.device != device:
            self.obs_grid_mapping = ObsGridMapping(
                grid_variables=self.variables,
                grid_lat=self.lat,
                grid_lon=self.lon,
                device=device,
            )
        return self.obs_grid_mapping

    def compile_model(self) -> None:
        """Compile the diffusion model forward pass with ``torch.compile``."""
        self.diffusion_model.forward = torch.compile(
            self.diffusion_model.forward, dynamic=False
        )

    def normalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Normalise ``x`` to zero mean, unit variance."""
        return torch.addcmul(
            self.mean_inv_std, x, self.inv_std, out=x if in_place else None
        )

    def denormalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Denormalise ``x`` back to physical units."""
        return torch.addcmul(self.means, x, self.stds, out=x if in_place else None)

    def _normalize_condition(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Normalise low-resolution conditioning to zero mean, unit variance."""
        return torch.addcmul(self.cond_mean_inv_std, conditioning, self.cond_inv_std)

    @torch.no_grad()
    def _forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        time: np.datetime64,
        y_obs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one diffusion forward step and return a denormalised state.

        Parameters
        ----------
        x : torch.Tensor
            High-resolution state, shape ``(batch, C, H, W)``.
        conditioning : torch.Tensor
            Low-resolution conditioning, shape ``(batch, C_cond, H, W)``.
        time : np.datetime64
            Target valid time, used for cosine-zenith angle and annual phase.
        y_obs : torch.Tensor, optional
            Gridded observations for SDA guidance, shape ``(1, C, H, W)``.
        mask : torch.Tensor, optional
            Binary observation mask matching ``y_obs``; required when ``y_obs`` is given.

        Returns
        -------
        torch.Tensor
            Denormalised predicted state, same shape as ``x``.
        """
        year_phase = (
            2
            * np.pi
            * (
                (time - time.astype("datetime64[Y]")).astype("timedelta64[h]")
                / np.timedelta64(8766, "h")
            )
        )
        scalar_conds = (
            torch.as_tensor([np.sin(year_phase), np.cos(year_phase)])
            .to(device=x.device, dtype=x.dtype)
            .repeat(x.shape[0], 1)
        )
        time_dt = datetime.fromisoformat(str(time))
        cos_zen = cos_zenith_angle(time_dt, self.lon_tensor, self.lat_tensor).expand(
            x.shape[0], 1, -1, -1
        )

        conditioning = self._normalize_condition(conditioning)
        x = self.normalize(x)
        invariant_tensor = self.invariants[
            :,
            :,
            self.hrrr_lat_lim[0] : self.hrrr_lat_lim[1],
            self.hrrr_lon_lim[0] : self.hrrr_lon_lim[1],
        ].repeat(x.shape[0], 1, 1, 1)

        # Concat for diffusion conditioning
        cond = torch.cat((x, conditioning, cos_zen, invariant_tensor), dim=1)
        condition = TensorDict(
            {"cond_concat": cond, "cond_vec": scalar_conds}, device=x.device
        )

        denoiser = self._edm_denoiser(condition)
        if y_obs is None:
            x = self._sample(x, denoiser)
        else:
            y_obs = self.normalize(y_obs)
            denoiser = self._sda_denoiser(condition, y_obs, mask)
            x = self._sample(x, denoiser, num_steps=self.num_sda_diffusion_steps)

        self.denormalize(x, in_place=True)
        if self.clamp_values:
            # set refc to realistic values (values between -10 and 0 dBZ do not exist in data)
            refc = x[..., self.refc_channel, :, :]
            refc[refc < 0.0] = -10.0

        return x

    def _conditioned_x0_predictor(
        self, condition: TensorDict
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return an x0-predictor closure conditioned on ``condition``."""

        def x0_predictor(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with torch.autocast(
                x_noisy.device.type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                x = self.diffusion_model(x_noisy, t, condition=condition)
                return x

        return x0_predictor

    def _sample(
        self, state: torch.Tensor, denoiser: Any, num_steps: int | None = None
    ) -> torch.Tensor:
        """Draw a diffusion sample.

        Parameters
        ----------
        state : torch.Tensor
            Reference tensor; its shape, device, and dtype are used for the initial latents.
        denoiser : Any
            Configured denoiser (EDM or SDA) returned by ``scheduler.get_denoiser``.
        num_steps : int, optional
            Number of diffusion steps; defaults to ``self.num_diffusion_steps``.

        Returns
        -------
        torch.Tensor
            Normalised diffusion output, same shape as ``state``.
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        latents = self.sampler_args["sigma_max"] * torch.randn_like(state)

        return sample(
            denoiser,
            latents,
            noise_scheduler=self.scheduler,
            num_steps=num_steps,
            solver="edm_stochastic_heun",
            solver_options={
                "S_churn": self.sampler_args["S_churn"],
                "S_min": self.sampler_args["S_min"],
                "S_max": self.sampler_args["S_max"],
                "S_noise": self.sampler_args["S_noise"],
            },
        )

    def _edm_denoiser(self, condition: TensorDict) -> Any:
        """Build an unconditional EDM denoiser for the given conditioning."""
        return self.scheduler.get_denoiser(
            x0_predictor=self._conditioned_x0_predictor(condition)
        )

    def _sda_denoiser(
        self, condition: TensorDict, y_obs: torch.Tensor, mask: torch.Tensor
    ) -> Any:
        """Build a DPS-guided SDA denoiser for the given conditioning and observations."""
        guidance = DataConsistencyDPSGuidance(
            mask=mask,
            y=y_obs,
            std_y=self.sda_std_obs,
            norm=self.sda_dps_norm,
            gamma=self.sda_gamma,
            sigma_fn=self.scheduler.sigma,
            alpha_fn=self.scheduler.alpha,
        )
        score_predictor = DPSScorePredictor(
            x0_predictor=self._conditioned_x0_predictor(condition),
            x0_to_score_fn=self.scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = self.scheduler.get_denoiser(score_predictor=score_predictor)

        return denoiser

    @torch.no_grad()  # safe - PhysicsNeMo SDA code uses torch.enable_grad()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        obs: pd.DataFrame | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        obs : pd.DataFrame, tuple[torch.Tensor, torch.Tensor], or None, optional
            Observations for SDA guidance. Either a dataframe with columns
            ``variable``, ``lat``, ``lon``, ``observation``, ``time``, or a
            pre-gridded ``(y_obs, mask)`` tuple. ``None`` runs unconditional
            diffusion.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """

        # StormCast-CONUS wants the low-res conditioning at t + 1 h so we do output_coords first
        output_coords = self.output_coords(coords)
        conditioning = self._get_conditioning(output_coords, x.shape[0], x.device)
        x = x.clone()  # prevent editing of argument

        for j, time in enumerate(coords["time"]):
            for k, lead_time in enumerate(coords["lead_time"]):
                t = time + lead_time
                if obs is not None:
                    if isinstance(obs, tuple):
                        (y_obs, mask) = obs
                    else:
                        y_obs, mask = self.get_obs_mapping(x.device).obs_to_grid(
                            obs, self.variables, t, self.time_tolerance
                        )
                else:
                    (y_obs, mask) = (None, None)

                for i0 in range(0, len(coords["batch"]), self.batch_size):
                    i1 = i0 + self.batch_size
                    x[i0:i1, j, k] = self._forward(
                        x[i0:i1, j, k],
                        conditioning[i0:i1, j, k],
                        t,
                        y_obs=y_obs,
                        mask=mask,
                    )

        return x, output_coords

    @batch_func()
    def create_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], pd.DataFrame | None, None]:
        """Create a generator for autoregressive rollout.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system after each time step

        Receives
        --------
        pd.DataFrame or None
            Observation dataframe with columns 'variable', 'lat', 'lon',
            'observation', 'time', or None for no observations

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """
        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCastCONUS has been called without initializing the model's "
                "conditioning_data_source"
            )

        obs = yield x, coords

        try:
            while True:
                x, coords = self(x, coords, obs=obs)
                obs = yield x, coords
        except GeneratorExit:
            pass

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Iterator wrapper around :meth:`create_generator` without observation input."""
        yield from self.create_generator(x, coords)

    def _get_conditioning(
        self,
        coords: CoordSystem,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Fetch and interpolate low-resolution conditioning to the HRRR grid.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system containing ``time`` and ``lead_time``.
        batch_size : int
            Number of ensemble members; conditioning is replicated accordingly.
        device : torch.device
            Target device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Conditioning tensor of shape ``(batch, time, lead_time, C_cond, H, W)``.

        Raises
        ------
        RuntimeError
            If ``conditioning_data_source`` is not set.
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCastCONUS has been called without initializing the model's conditioning_data_source"
            )

        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=device,
            interp_to=coords | {"_lat": self.lat, "_lon": self.lon},
            interp_method="linear",
        )
        # ensure data dimensions in the expected order
        conditioning_coords_ordered = OrderedDict(
            {
                k: conditioning_coords[k]
                for k in ["time", "lead_time", "variable", "lat", "lon"]
            }
        )
        conditioning, conditioning_coords = map_coords(
            conditioning, conditioning_coords, conditioning_coords_ordered
        )

        # Add a batch dim
        conditioning = conditioning.repeat(batch_size, 1, 1, 1, 1, 1)
        conditioning_coords.update({"batch": np.empty(0)})
        conditioning_coords.move_to_end("batch", last=False)

        # Handshake conditioning coords
        handshake_coords(conditioning_coords, coords, "lead_time")
        handshake_coords(conditioning_coords, coords, "time")

        return conditioning

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        # package = Package(
        #     "hf://nvidia/stormcast-conus@...",
        #     cache_options={
        #         "cache_storage": Package.default_cache("stormcast-conus"),
        #         "same_names": True,
        #     },
        # )
        # return package
        raise NotImplementedError("No default package.")

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(),
        **model_kwargs: Any,
    ) -> Protocol:
        """Load a :class:`StormCastCONUS` model from a package.

        Parameters
        ----------
        package : Package
            Package to load model from
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source to use for global conditioning, by default GFS_FX
        **model_kwargs
            Additional keyword arguments forwarded to the model constructor
            (e.g. ``hrrr_lat_lim``, ``num_diffusion_steps``, ``use_amp``,
            ``clamp_values``). ``clamp_values`` is additionally applied to
            the internal :class:`_SplitModelWrapper`.

        Returns
        -------
        Protocol
            StormCast-CONUS model
        """
        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        model_high = EDMPreconditioner.from_checkpoint(
            package.resolve("StormCastCONUS-high.mdlus")
        ).eval()
        model_pz_low = EDMPreconditioner.from_checkpoint(
            package.resolve("StormCastCONUS-pz-low.mdlus")
        ).eval()
        model_tq_low = EDMPreconditioner.from_checkpoint(
            package.resolve("StormCastCONUS-tq-low.mdlus")
        ).eval()
        model_uv_low = EDMPreconditioner.from_checkpoint(
            package.resolve("StormCastCONUS-uv-low.mdlus")
        ).eval()

        # Load metadata: means, stds, grid
        store = zarr.storage.ZipStore(package.resolve("metadata.zarr.zip"), mode="r")
        metadata = xr.open_zarr(store, zarr_format=2)

        variables = np.array(metadata["variable_state"], copy=True)
        if "mslp" in variables:
            mslp_ind = list(variables).index("mslp")
            variables[mslp_ind] = "msl"
        conditioning_variables = metadata["variable_background"].values

        # Expand dims and tensorify normalization buffers
        means = torch.from_numpy(metadata["means"].values[None, :, None, None])
        stds = torch.from_numpy(metadata["stds"].values[None, :, None, None])
        conditioning_means = torch.from_numpy(
            metadata["conditioning_means"].values[None, :, None, None]
        )
        conditioning_stds = torch.from_numpy(
            metadata["conditioning_stds"].values[None, :, None, None]
        )

        # Load invariants
        invariants = metadata["invariants"].values
        invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)

        # EDM sampler arguments
        sampler_args = config.sampler_args

        # Optional per-checkpoint variable split override
        variables_split = (
            OmegaConf.to_object(config.variables_split)
            if "variables_split" in config
            else None
        )

        diffusion_model = _SplitModelWrapper(
            model_high=model_high,
            model_pz_low=model_pz_low,
            model_tq_low=model_tq_low,
            model_uv_low=model_uv_low,
            mean=means,
            std=stds,
            sigma_threshold=config["sigma_threshold"],
            clamp_values=model_kwargs.get("clamp_values", True),
            variables=variables,
            variables_split=variables_split,
        )

        return cls(
            diffusion_model,
            means,
            stds,
            invariants,
            variables=variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            sampler_args=sampler_args,
            **model_kwargs,
        )


class _SplitModelWrapper(torch.nn.Module):
    """Route diffusion denoising across sigma-dependent sub-models.

    At high noise levels (``sigma >= sigma_threshold``), a single
    high-resolution model predicts all state variables. At lower noise levels,
    three specialised sub-models predict disjoint variable groups (pressure /
    geopotential, temperature / moisture, winds), and their outputs are merged
    channel-wise.

    Optionally crops positional embeddings and tokeniser geometry when inference
    is run on a subdomain of the full CONUS HRRR grid.

    Parameters
    ----------
    model_high : torch.nn.Module
        EDM-preconditioned model for high-noise denoising.
    model_pz_low : torch.nn.Module
        Low-noise model for mean sea level pressure and pressure-level fields.
    model_tq_low : torch.nn.Module
        Low-noise model for temperature, moisture, and reflectivity fields.
    model_uv_low : torch.nn.Module
        Low-noise model for wind fields.
    mean : torch.Tensor
        Per-channel mean used to compute normalised value clamps.
    std : torch.Tensor
        Per-channel standard deviation used to compute normalised value clamps.
    sigma_threshold : float, optional
        Noise level above which ``model_high`` is used exclusively, by default 3.0.
    clamp_values : bool, optional
        Whether to clamp denoised outputs to variable-specific minima, by default True.
    full_grid_shape : tuple[int, int], optional
        Full HRRR grid shape ``(H, W)`` before cropping, by default ``(1024, 1792)``.
    variables : np.ndarray, optional
        Ordered variable names for the model state. Used to build per-split channel
        masks and physical-minimum clamps. Defaults to the module-level ``VARIABLES``.
    variables_split : dict[str, list[str]], optional
        Mapping of split name to variable names for each low-noise sub-model.
        Defaults to ``VARIABLES_SPLIT``. Loaded from the model package
        config when available.
    """

    VARIABLES_SPLIT: dict[str, list[str]] = {
        "pz": ["msl"] + [v for v in VARIABLES if v[0] in ("p", "Z")],
        "tq": [v for v in VARIABLES if v[0] in ("t", "q")] + ["refc"],
        "uv": [v for v in VARIABLES if v[0] in ("u", "v")],
    }

    MIN_VALUES: dict[str, float] = {
        v: 0.0 for v in VARIABLES if any(v.startswith(c) for c in ["p", "Z", "t", "q"])
    }
    MIN_VALUES["msl"] = 0.0
    MIN_VALUES["refc"] = -10.0

    def __init__(
        self,
        model_high: torch.nn.Module,
        model_pz_low: torch.nn.Module,
        model_tq_low: torch.nn.Module,
        model_uv_low: torch.nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        sigma_threshold: float = 3.0,
        clamp_values: bool = True,
        full_grid_shape: tuple[int, int] = (1024, 1792),
        variables: np.ndarray | None = None,
        variables_split: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.model_high = model_high
        self.model_pz_low = model_pz_low
        self.model_tq_low = model_tq_low
        self.model_uv_low = model_uv_low
        self.models = {
            "high": self.model_high,
            "pz_low": self.model_pz_low,
            "tq_low": self.model_tq_low,
            "uv_low": self.model_uv_low,
        }

        self.sigma_threshold = sigma_threshold
        self.clamp_values = clamp_values

        _variables = list(VARIABLES) if variables is None else list(variables)
        _variables_split = (
            self.VARIABLES_SPLIT if variables_split is None else variables_split
        )

        for split, split_vars in _variables_split.items():
            self.register_buffer(
                f"split_mask_{split}",
                torch.as_tensor([(v in split_vars) for v in _variables]),
            )

        min_values = [
            _SplitModelWrapper.MIN_VALUES.get(var, -np.inf) for var in _variables
        ]
        min_values_t = (torch.as_tensor(min_values)[None, :, None, None] - mean) / std
        self.register_buffer("min_values", min_values_t)

        self.full_grid_shape = full_grid_shape
        self.grid_shape = full_grid_shape
        self.pos_embed_full = {
            k: model.model.model.tokenizer.pos_embed
            for (k, model) in self.models.items()
        }

    def crop_model(self, bbox: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """Crop all sub-models to an HRRR index bounding box.

        Parameters
        ----------
        bbox : tuple[tuple[int, int], tuple[int, int]]
            ``((lat_start, lat_end), (lon_start, lon_end))`` indices on the full
            HRRR grid. Bounds must align with the patch size of ``model_high``.
        """
        self.grid_shape = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0])
        p = self.model_high.model.model.patch_size
        (H_full, W_full) = (
            self.full_grid_shape[0] // p[0],
            self.full_grid_shape[1] // p[1],
        )
        bb_crop = (
            (bbox[0][0] // p[0], bbox[0][1] // p[0]),
            (bbox[1][0] // p[1], bbox[1][1] // p[1]),
        )

        def _crop_single_posemb(pos_embed_full: torch.Tensor) -> torch.Tensor:
            pos_embed_full_2D = pos_embed_full.view(
                H_full, W_full, pos_embed_full.shape[-1]
            )
            pos_embed_crop_2D = pos_embed_full_2D[
                bb_crop[0][0] : bb_crop[0][1], bb_crop[1][0] : bb_crop[1][1], :
            ]
            (H_crop, W_crop, C) = pos_embed_crop_2D.shape
            return pos_embed_crop_2D.reshape(H_crop * W_crop, C)

        for key, model in self.models.items():
            dit = model.model.model
            dit.input_size = self.grid_shape
            dit.tokenizer.input_size = self.grid_shape
            dit.tokenizer.h_patches = self.grid_shape[0] // p[0]
            dit.tokenizer.w_patches = self.grid_shape[1] // p[1]
            pos_embed_crop = _crop_single_posemb(self.pos_embed_full[key])
            dit.tokenizer.register_parameter(
                "pos_embed", torch.nn.Parameter(pos_embed_crop)
            )
            dit.detokenizer.input_size = self.grid_shape
            dit.detokenizer.h_patches = self.grid_shape[0] // p[0]
            dit.detokenizer.w_patches = self.grid_shape[1] // p[1]

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: TensorDict,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        """Route the forward pass to the high- or low-sigma sub-model.

        Parameters
        ----------
        x : torch.Tensor
            Noisy normalised state, shape ``(batch, C, H, W)``.
        sigma : torch.Tensor
            Current noise level(s).
        condition : TensorDict
            Conditioning dictionary passed to each sub-model.
        **model_kwargs
            Additional keyword arguments forwarded to sub-models.

        Returns
        -------
        torch.Tensor
            Predicted clean state, same shape as ``x``.

        Note
        ----
        ``@torch.no_grad()`` is intentionally absent. When called via the SDA
        path, ``DPSScorePredictor`` enables gradients and needs the computational
        graph to propagate through this module for the DPS score correction.
        """
        sigma = sigma.flatten()

        attn_kwargs = dict(model_kwargs.get("attn_kwargs", {}))
        p = self.model_high.model.model.patch_size
        latent_hw = (self.grid_shape[0] // p[0], self.grid_shape[1] // p[1])
        attn_kwargs.update({"latent_hw": latent_hw})
        model_kwargs["attn_kwargs"] = attn_kwargs

        if sigma[0] >= self.sigma_threshold:
            y = self.model_high(x, sigma, condition, **model_kwargs)
        else:
            y = self.models["pz_low"](x, sigma, condition, **model_kwargs)
            for split in ["tq", "uv"]:
                y_split = self.models[f"{split}_low"](
                    x, sigma, condition, **model_kwargs
                )
                split_mask = getattr(self, f"split_mask_{split}")
                y[:, split_mask] = y_split[:, split_mask]

        if self.clamp_values:
            y = torch.clamp(y, min=self.min_values, out=None if y.requires_grad else y)

        return y
