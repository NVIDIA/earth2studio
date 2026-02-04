# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import json
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import ArrayLike

from earth2studio.data import GFS_FX, HRRR
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.data.utils import fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import NearestNeighborInterpolator
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo.models import Module
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("stormscope")
    Module = None  # type: ignore[assignment]
    cos_zenith_angle = None  # type: ignore[assignment]

from earth2studio.models.nn.stormscope_util import (
    DropInDiT,
    EDMPrecond,
)


def model_wrap(model: Module) -> nn.Module:
    """Wrap a physicsnemo Module so it is compatible with the preconditioning
    and sampler used by StormScope.
    TODO: Remove once core EDMPrecond architecture is fully upstreamed
    """
    return EDMPrecond(
        model=DropInDiT(
            pnm=model,
        ),
    )


@check_optional_dependencies()
class StormScopeBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """StormScope diffusion prognostic base model with staged denoising.

    Variants should subclass to define dataset/resolution specifics (e.g., grids,
    variables, and data loading).

    This base class supports an "ensemble of experts" approach where the diffusion
    sampler is invoked in multiple stages, each using a different set of model
    weights applicable to a specific sigma range. Stages are defined by `model_spec`.
    Single-stage (standard diffusion sampling with one denoiser) is also supported.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications. Each entry must contain:

        - 'model': physicsnemo.Module
        - 'sigma_min': float
        - 'sigma_max': float

        The sigma interval determines which expert is applied during denoising.
    means : torch.Tensor
        Per-variable mean used to normalize inputs. Expected shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std used to normalize inputs. Expected shape [1, C, 1, 1].
    variables : np.ndarray
        Names of prognostic variables corresponding to the high-resolution state.
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    conditioning_variables : np.ndarray | None, optional
        Names of external conditioning variables. Default is None.
    conditioning_data_source : Any | None, optional
        Data source for external conditioning. Subclasses should define how this is
        used; base class does not fetch conditioning by default. Default is None.
    sampler_args : dict[str, Any] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 m] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [60 m] (i.e., 1 hour from the current time).
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    input_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of input data.
        Points beyond this distance are masked as invalid. Default is 12.0.
    conditioning_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of conditioning data.
        Points beyond this distance are masked as invalid. Default is 26.0.
    """

    # Constants used to normalize lat/lon input features
    _CENTRAL_LAT_CONSTANT = 38.5
    _LAT_SCALE = 15
    _CENTRAL_LON_CONSTANT = 262.5
    _LON_SCALE = 36

    # Whether to place the state first in the conditioning input
    _STATE_FIRST = True

    # Dtype to use in the EDM sampler
    _SAMPLER_DTYPE = torch.float64

    # Constant to fill invalid gridpoints in the input after normalization
    _INPUT_INVALID_FILL_CONSTANT = 0.0

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        variables: np.ndarray,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.ndarray | None = None,
        conditioning_data_source: Any | None = None,
        sampler_args: dict[str, Any] | None = {"num_steps": 100, "S_churn": 10},
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        input_interp_max_dist_km: float = 12.0,
        conditioning_interp_max_dist_km: float = 26.0,
    ):
        super().__init__()
        # Validate and store staged models
        if not isinstance(model_spec, list) or len(model_spec) == 0:
            raise ValueError("model_spec must be a non-empty list of stage dicts.")

        if conditioning_data_source is None:
            logger.warning(
                "No conditioning data source was provided to StormScope; set the conditioning_data_source attribute "
                "of the model before running inference with iterator mode, or use the call_with_conditioning method."
            )

        self.input_times = input_times
        self.output_times = output_times
        self.sliding_window = len(input_times) > len(output_times)

        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self._lat_cpu_copy = self.latitudes.cpu().numpy()
        self._lon_cpu_copy = self.longitudes.cpu().numpy()
        self.register_buffer("valid_mask", torch.ones_like(latitudes, dtype=torch.bool))
        self.register_buffer(
            "conditioning_valid_mask", torch.ones_like(latitudes, dtype=torch.bool)
        )
        self._input_interp_max_dist_km = input_interp_max_dist_km
        self._conditioning_interp_max_dist_km = conditioning_interp_max_dist_km

        if y_coords is not None and x_coords is not None:
            self.y = y_coords
            self.x = x_coords
        else:
            self.y = np.arange(latitudes.shape[0])
            self.x = np.arange(longitudes.shape[1])

        for i, spec in enumerate(model_spec):
            if not isinstance(spec, dict):
                raise TypeError(f"model_spec[{i}] must be a dict.")
            for key in ("model", "sigma_min", "sigma_max"):
                if key not in spec:
                    raise KeyError(f"model_spec[{i}] missing required key '{key}'.")

        # Sort stages by descending sigma_max to ensure large->small sampling schedule
        self.model_spec = sorted(
            model_spec, key=lambda s: float(s["sigma_max"]), reverse=True
        )
        # Store in ModuleList so `.to(device)` works
        self.stage_models = nn.ModuleList([spec["model"] for spec in self.model_spec])
        self.start_sigma = self.model_spec[-1]["sigma_min"]
        self.end_sigma = self.model_spec[0]["sigma_max"]

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.variables = variables

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)
        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        self.sampler_args = sampler_args or {}
        self.variables = variables
        self.input_interp = None
        self.conditioning_interp = None

    @classmethod
    def load_default_package(cls) -> Package:
        """Load a default local package for StormScope models."""
        package = Package(
            "hf://nvidia/stormscope-goes-mrms@6ee31e07afe3decb012740f3be17531207c3db5e",
            cache_options={
                "cache_storage": Package.default_cache("stormscope"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    def load_model(
        cls,
        package: Package,
        *args: Any,
        **kwargs: Any,
    ) -> PrognosticModel:
        """Load model from package. Must be implemented by subclasses."""
        raise NotImplementedError(
            "StormScopeBase.load_model must be implemented by a subclass."
        )

    def build_input_interpolator(
        self,
        input_lats: torch.Tensor | ArrayLike,
        input_lons: torch.Tensor | ArrayLike,
        max_dist_km: float | None = None,
    ) -> nn.Module:
        """Build a module to handle interpolating data on an arbitrary input lat/lon grid
        to the internal lat/lon grid, done via nearest neighbor interpolation.

        Parameters
        ----------
        input_lats : torch.Tensor | ArrayLike
            Input latitudes.
        input_lons : torch.Tensor | ArrayLike
            Input longitudes.
        max_dist_km : float, optional
            Maximum great-circle distance (km) to accept a nearest neighbor match.
            Target points farther than this threshold are marked invalid and will
            receive NaNs in the output. By default 6.0.
        """
        if max_dist_km is None:
            max_dist_km = self._input_interp_max_dist_km
        self.input_interp = NearestNeighborInterpolator(
            source_lats=input_lats,
            source_lons=input_lons,
            target_lats=self.latitudes,
            target_lons=self.longitudes,
            max_dist_km=max_dist_km,
        ).to(self.latitudes.device)

        interp = cast(NearestNeighborInterpolator, self.input_interp)
        if torch.any(~interp.valid_mask):
            logger.warning(
                "Some input gridpoints are invalid after interpolation. "
                "This may be expected if the input data source is not available at "
                "all gridpoints, but consider double-checking coordinates and/or the "
                "max_dist_km parameter. Invalid points will be filled with the model's "
                f"_INPUT_INVALID_FILL_CONSTANT ({self._INPUT_INVALID_FILL_CONSTANT})."
            )

        # Store the mask tracking valid gridpoints, shape [H, W]
        self.register_buffer(
            "valid_mask", interp.valid_mask.reshape(len(self.y), len(self.x))
        )
        self.valid_mask: torch.Tensor = self.valid_mask.to(device=self.latitudes.device)

    def build_conditioning_interpolator(
        self,
        conditioning_lats: torch.Tensor | ArrayLike,
        conditioning_lons: torch.Tensor | ArrayLike,
        max_dist_km: float | None = None,
    ) -> nn.Module:
        """Build a module to handle interpolating data on an arbitrary input lat/lon grid
        to the internal lat/lon grid, done via nearest neighbor interpolation.

        Parameters
        ----------
        conditioning_lats : torch.Tensor | ArrayLike
            Conditioning latitudes.
        conditioning_lons : torch.Tensor | ArrayLike
            Conditioning longitudes.
        max_dist_km : float, optional
            Maximum great-circle distance (km) to accept a nearest neighbor match.
            Target points farther than this threshold are marked invalid and will
            receive NaNs in the output. By default 6.0.
        """
        if max_dist_km is None:
            max_dist_km = self._conditioning_interp_max_dist_km
        self.conditioning_interp = NearestNeighborInterpolator(
            source_lats=conditioning_lats,
            source_lons=conditioning_lons,
            target_lats=self.latitudes,
            target_lons=self.longitudes,
            max_dist_km=max_dist_km,
        ).to(self.latitudes.device)

        cinterp = cast(NearestNeighborInterpolator, self.conditioning_interp)
        if torch.any(~cinterp.valid_mask):
            logger.warning(
                "Some conditioning gridpoints are invalid after interpolation. "
                "This may be expected if the conditioning data source is not available at "
                "all gridpoints, but consider double-checking coordinates and/or the "
                "max_dist_km parameter. Invalid points will be filled with the model's "
                f"_INPUT_INVALID_FILL_CONSTANT ({self._INPUT_INVALID_FILL_CONSTANT})."
            )

        # Store the mask tracking valid gridpoints, shape [H, W]
        self.register_buffer(
            "conditioning_valid_mask",
            cinterp.valid_mask.reshape(len(self.y), len(self.x)),
        )
        self.conditioning_valid_mask: torch.Tensor = self.conditioning_valid_mask.to(
            device=self.latitudes.device
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system. Subclasses should override for specific variants."""
        raise NotImplementedError(
            "StormScopeBase.input_coords must be implemented by a subclass."
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output coordinates.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary for the model output.
        """
        raise NotImplementedError(
            "StormScopeBase.output_coords must be implemented by a subclass."
        )

    def fetch_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch external conditioning data. Subclasses should override.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system.
        device : torch.device
            Device on which the conditioning tensor should reside.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Conditioning tensor aligned with `coords`.
        """
        raise NotImplementedError(
            "StormScopeBase.fetch_conditioning must be implemented by a subclass."
        )

    def normalize_conditioning(
        self, conditioning: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Normalize external conditioning with stored stats if available."""
        if conditioning is None:
            return None
        x = conditioning
        if "conditioning_means" in self._buffers:
            x = x - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            x = x / self.conditioning_stds
        return x

    def _stack_lead_times(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Stack lead times along the channel dimension. Reshapes tensors from (..., n_lt, n_vars, y, x)
        to (..., 1, n_lt * n_vars, y, x) and updates the coordinate system to reflect the new shape.
        The last lead time from the original coordinate system is used as the lead time for the stacked tensor.
        """
        lt_dim = list(coords.keys()).index("lead_time")
        var_dim = list(coords.keys()).index("variable")
        if var_dim != lt_dim + 1:
            raise ValueError(
                f"The coordinate order must be [..., lead_time, variable, ...], got {list(coords.keys())}"
            )
        n_lt = len(coords["lead_time"])
        n_vars = len(coords["variable"])
        stacked = x.reshape(
            *x.shape[:lt_dim], n_lt * n_vars, *x.shape[lt_dim + 2 :]
        ).unsqueeze(lt_dim)
        stacked_coords = coords.copy()
        stacked_coords["lead_time"] = stacked_coords["lead_time"][-1:]
        stacked_coords["variable"] = np.array(
            [
                f"{var}(t+{str(lt)})"
                for lt in stacked_coords["lead_time"]
                for var in stacked_coords["variable"]
            ]
        )
        return stacked, stacked_coords

    def build_condition(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
        conditioning_coords: CoordSystem | None = None,
    ) -> torch.Tensor:
        """Construct the full conditioning input to the diffusion model.
        Subclasses can override to customize the channel layout.

        Parameters
        ----------
        x_norm : torch.Tensor
            Normalized input high-resolution state.
        coords : CoordSystem
            Input coordinate system to provide time and lead time for cos zenith angle calculation
        conditioning : torch.Tensor | None, optional
            Optional normalized external conditioning, by default None.
        conditioning_coords : CoordSystem | None, optional
            Optional coordinate system for the conditioning, by default None.

        Returns
        -------
        torch.Tensor
            Conditioning tensor to pass to the diffusion sampler.
        """

        if self.sliding_window:
            # Reshape input/conditioning to (..., 1, n_lt * n_vars, y, x)
            x, coords = self._stack_lead_times(x, coords)
            if conditioning is not None:
                if conditioning_coords is None:
                    raise ValueError(
                        "Expected conditioning_coords when conditioning is provided"
                    )
                conditioning, conditioning_coords = self._stack_lead_times(
                    conditioning, conditioning_coords
                )

        # Fold batch/time/lead_time dimensions
        b, t, lt, _, _, _ = x.shape
        if lt != 1:
            raise ValueError(f"Expected 1 lead time in prepared input data, got {lt}")
        x = x.reshape(b * t * lt, *x.shape[3:])
        if conditioning is not None:
            conditioning = conditioning.reshape(b * t * lt, *conditioning.shape[3:])

        parts = [x]
        if conditioning is not None:
            if self._STATE_FIRST:
                parts.append(conditioning)
            else:
                parts.insert(0, conditioning)
        if self.latitudes is not None and self.longitudes is not None:
            normed_lat = (self.latitudes - self._CENTRAL_LAT_CONSTANT) / self._LAT_SCALE
            normed_lon = (
                self.longitudes + 180.0
            ) % 360.0 - 180.0  # trained models expect [-180, 180] longitudes
            normed_lon = (normed_lon - self._CENTRAL_LON_CONSTANT) / self._LON_SCALE
            latlon_input = torch.cat(
                (normed_lat[None, None, :, :], normed_lon[None, None, :, :]), dim=1
            )
            parts.append(latlon_input.repeat(b * t, 1, 1, 1))

        # Build cos zenith features from input and target times
        times = np.array(coords["time"]).astype(np.datetime64)
        lead_times = np.array(coords["lead_time"]).astype(np.timedelta64)
        input_cz_times = np.concatenate(
            [times + lead_times[-1]] * b, axis=0
        )  # batch and time dims are folded together
        target_cz_times = input_cz_times + self.output_times[-1]
        input_cz_times = np.array(
            [
                datetime.fromtimestamp(
                    t.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for t in input_cz_times
            ]
        )
        target_cz_times = np.array(
            [
                datetime.fromtimestamp(
                    t.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for t in target_cz_times
            ]
        )
        cz_0 = np.stack(
            [
                cos_zenith_angle(t, self._lon_cpu_copy, self._lat_cpu_copy)
                for t in input_cz_times
            ],
            axis=0,
        )  # shape [B*T, H, W]
        cz_1 = np.stack(
            [
                cos_zenith_angle(t, self._lon_cpu_copy, self._lat_cpu_copy)
                for t in target_cz_times
            ],
            axis=0,
        )  # shape [B*T, H, W]

        parts.extend(
            [
                torch.from_numpy(cz_0).to(device=x.device, dtype=x.dtype)[
                    :, None, :, :
                ],
                torch.from_numpy(cz_1).to(device=x.device, dtype=x.dtype)[
                    :, None, :, :
                ],
            ]
        )

        return torch.cat(parts, dim=1)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
        conditioning_coords: CoordSystem | None = None,
    ) -> torch.Tensor:
        """Run a single prognostic step using staged diffusion denoising.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for a single step with shape [B, T, L, C, H, W].
        coords : CoordSystem
            Coordinates describing `x`.
        conditioning : torch.Tensor | None, optional
            External conditioning aligned to `x` if used, by default None.
        conditioning_coords : CoordSystem | None, optional
            Coordinate system for the conditioning, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor for the same step with denoised prognostic variables.
        """

        if x.dim() != 6 or (conditioning is not None and conditioning.dim() != 6):
            cond_shape = conditioning.shape if conditioning is not None else None
            raise ValueError(
                f"Input tensors must have 6 dimensions [B, T, L, C, H, W], got {x.shape} and {cond_shape} for input and conditioning respectively"
            )

        b, t, lt, _, _, _ = x.shape

        # Scale input and fill invalid gridpoints
        x_norm = (x - self.means) / self.stds
        x_norm = torch.where(self.valid_mask, x_norm, self._INPUT_INVALID_FILL_CONSTANT)
        output_dtype = x_norm.dtype

        # Scale conditioning and zero-fill invalid gridpoints
        if conditioning is not None:
            conditioning_norm = self.normalize_conditioning(conditioning)
            conditioning_norm = torch.where(
                self.conditioning_valid_mask, conditioning_norm, 0.0
            )
        else:
            conditioning_norm = None

        # Combined conditioning to diffusion model: state, conditioning variables, extras (lat/lon, cos zenith angle)
        condition = self.build_condition(
            x=x_norm,
            coords=coords,
            conditioning=conditioning_norm,
            conditioning_coords=conditioning_coords,
        )
        latents = torch.randn(
            b * t, *x.shape[3:], device=x.device, dtype=x.dtype
        )  # shape [B*T, C, H, W]

        # Run diffusion sampler
        out = self._edm_sampler(
            latents=latents,
            condition=condition,
            sigma_min=self.start_sigma,
            sigma_max=self.end_sigma,
            **self.sampler_args,
        ).to(output_dtype)

        out = out.reshape(b, t, len(self.output_times), *out.shape[1:])

        out = torch.where(self.valid_mask, out, torch.nan)
        out = out * self.stds + self.means
        return out

    def _edm_sampler(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        randn_like: Callable[[torch.Tensor], torch.Tensor] = torch.randn_like,
        num_steps: int = 18,
        sigma_max: float = 500,
        sigma_min: float = 0.004,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        progress_bar: Any | None = None,
    ) -> torch.Tensor:
        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0
        # Main sampling loop.
        x_next = latents.to(self._SAMPLER_DTYPE) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # select active expert based on t_cur
            active_net = self._select_expert(t_cur)

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = active_net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = active_net(
                x_hat, t_hat, class_labels=class_labels, condition=condition
            ).to(self._SAMPLER_DTYPE)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                # Select the active network for the next step
                active_net_prime = self._select_expert(t_next)

                denoised = active_net_prime(
                    x_next, t_next, class_labels=class_labels, condition=condition
                ).to(self._SAMPLER_DTYPE)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if progress_bar:
                progress_bar.update()

        return x_next

    def _select_expert(self, t_cur: torch.Tensor) -> nn.Module:
        """Select the active denoising expert based on the current time step."""

        eps = 1e-7  # small epsilon to avoid floating point issues
        for i, stage in enumerate(self.model_spec):
            if t_cur >= stage["sigma_min"] - eps and t_cur < stage["sigma_max"] + eps:
                return self.stage_models[i]
        raise ValueError(
            f"No denoising expert found for time step {t_cur.cpu().item()}, {stage['sigma_min']}"
        )

    def prep_input(
        self, x: torch.Tensor, coords: CoordSystem, conditioning: bool = False
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepares the input tensor for the prognostic model."""
        if not conditioning:
            type_label = "input"
            interpolator = self.input_interp
            valid_mask = self.valid_mask
        else:
            type_label = "conditioning"
            interpolator = self.conditioning_interp
            valid_mask = self.conditioning_valid_mask

        # Expect data on the model's native grid by default; only accept other grids if we can interpolate
        if (
            "y" not in coords
            or "x" not in coords
            or coords["y"].shape != self.y.shape
            or coords["x"].shape != self.x.shape
            or (coords["y"] != self.y).any()
            or (coords["x"] != self.x).any()
        ):

            if interpolator is None:
                raise ValueError(
                    f"Using {type_label} data on a non-native grid requires interpolation, call build_{type_label}_interpolator first"
                )

            x = interpolator(x)
            x_coords = coords.copy()

            # Remove the previous spatial dims from x_coords (last two) and replace
            for _ in range(2):
                x_coords.popitem()
            x_coords["y"] = self.y
            x_coords["x"] = self.x
            x_coords.move_to_end("y")
            x_coords.move_to_end("x")
        else:
            x_coords = coords.copy()

        # Handle invalid gridpoints: initially zero-fill any NaNs so they don't cause problems in concat/prep ops
        # Data will be zero-filled again after normalization in the forward method.
        # if not (conditioning and torch.any(self.conditioning_valid_mask)):
        x = torch.where(~valid_mask, 0.0, x)

        return x, x_coords

    @batch_func()
    def next_input(
        self,
        pred: torch.Tensor,
        pred_coords: CoordSystem,
        x: torch.Tensor,
        x_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Gets the inputs for the next step of the prognostic model given a pair
        of inputs and predictions that were just run. If the model uses a sliding
        window, the oldest lead time in the input is removed and the latest
        prediction is added for the next step. If the model does not
        use a sliding window, the prediction is returned as is.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor.
        pred_coords : CoordSystem
            Prediction coordinate system.
        x : torch.Tensor
            Input tensor.
        x_coords : CoordSystem
            Input coordinate system.
        """
        lt_dim = list(x_coords.keys()).index("lead_time")
        lt_dim_pred = list(pred_coords.keys()).index("lead_time")
        if lt_dim != lt_dim_pred or lt_dim != 2:
            raise ValueError(
                f"The lead time dimension must be in the 3rd position for the input and prediction, got {lt_dim} and {lt_dim_pred} respectively"
            )

        if self.sliding_window:
            x, x_coords = self.prep_input(x, x_coords)  # Sanitize/regrid
            n_in, n_out = len(self.input_times), len(self.output_times)
            next_input = torch.zeros_like(x)
            next_input_coords = x_coords.copy()
            next_input[:, :, : n_in - n_out, ...] = x[:, :, n_out:, ...]
            next_input[:, :, n_in - n_out :, ...] = pred[:, :, :, ...]
            next_input_coords["lead_time"] = (
                self.input_times + pred_coords["lead_time"][-1]
            )
        else:
            next_input = pred
            next_input_coords = pred_coords.copy()

        return next_input, next_input_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs the prognostic model one step. Assumes the last two dimensions of the input tensor are the spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system.
        """

        x, x_coords = self.prep_input(x, coords)

        # Fetch and prep conditioning data if needed
        if (
            self.conditioning_variables is not None
            and len(self.conditioning_variables) > 0
        ):
            conditioning, conditioning_coords = self.fetch_conditioning(
                coords, device=x.device
            )
            conditioning, conditioning_coords = self.prep_input(
                conditioning, conditioning_coords, conditioning=True
            )

            # Broadcast to batch dimension if needed. Expect [B, T, L, C, H, W].
            if conditioning.dim() == x.dim() - 1:
                conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)
        else:
            conditioning = None
            conditioning_coords = None

        output_coords = self.output_coords(x_coords)

        x = self._forward(
            x,
            x_coords,
            conditioning=conditioning,
            conditioning_coords=conditioning_coords,
        )

        return x, output_coords

    @torch.inference_mode()
    def call_with_conditioning(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor,
        conditioning_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Calls the prognostic model with explicitly provided conditioning. Useful when
        combining multiple cross-conditioned models during rollout (does not require
        model to define a conditioning data source).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.
        conditioning : torch.Tensor
            Conditioning tensor.
        conditioning_coords : CoordSystem
            Conditioning coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system.
        """

        if (
            "time" not in coords
            or "batch" not in coords
            or len(coords["time"]) == 0
            or len(coords["batch"]) == 0
        ):
            raise ValueError(
                "Invalid coordinates for call_with_conditioning: must contain 'time' and 'batch' dimensions with nonzero length"
            )

        x, x_coords = self.prep_input(x, coords)
        conditioning, conditioning_coords = self.prep_input(
            conditioning, conditioning_coords, conditioning=True
        )
        output_coords = self.output_coords(x_coords)

        x = self._forward(
            x,
            x_coords,
            conditioning=conditioning,
            conditioning_coords=conditioning_coords,
        )

        return x, output_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        # Yield the initial condition, but use prep_input to regrid if needed
        # Return the result with any invalid points set to nan
        coords = coords.copy()
        x, coords = self.prep_input(x, coords)
        ic_coords = coords.copy()
        ic_coords["lead_time"] = ic_coords["lead_time"][-1:]
        yield torch.where(~self.valid_mask, torch.nan, x), ic_coords

        while True:
            x, coords = self.front_hook(x, coords)
            x_pred, coords_pred = self.__call__(x, coords)
            x_pred, coords_pred = self.rear_hook(x_pred, coords_pred)
            yield x_pred, coords_pred.copy()

            x, coords = self.next_input(x_pred, coords_pred, x, coords)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates an iterator to perform time-integration of the prognostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model containing the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)


class StormScopeGOES(StormScopeBase):
    """StormScope model forecasting GOES data on the HRRR grid.

    This model supports multiple variants at different spatiotemporal resolutions:

      - 6km resolution, 60 minute timestep
      - 6km resolution, 10 minute timestep
      - 3km resolution, 10 minute timestep

    Selection between these can be made by passing the ``model_name argument`` to this
    class's ``load_model`` method.

    The 6km/10min model uses a sliding window of 6 input timesteps and predicts one
    output timestep; other models use a single input timestep and predict one output
    timestep.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications; see `StormScopeBase`.
    means : torch.Tensor
        Per-variable mean for normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for normalization, shape [1, C, 1, 1].
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    variables : np.ndarray, optional
        GOES input variables. Default is
        ["abi01c", "abi02c", "abi03c", "abi07c", "abi08c", "abi09c", "abi10c", "abi13c"].
    conditioning_variables : np.ndarray, optional
        Auxiliary conditioning variables. Default is ["z500"].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    conditioning_data_source : Any | None, optional
        Data source for external conditioning. Default is None.
    sampler_args : dict[str, Any] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 m] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [60 m] (i.e., 1 hour from the current time).
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    input_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of input data.
        Points beyond this distance are masked as invalid. Default is 12.0.
    conditioning_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of conditioning data.
        Points beyond this distance are masked as invalid. Default is 26.0.

    Note
    ----
    To have a unified coordinate system over CONUS for convenience, the model uses the HRRR grid.
    As a result, there are portions of the domain which go beyond the extent of the GOES-East data,
    so these portions are masked as invalid (set to NaN).
    """

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        variables: np.ndarray = np.array(
            [
                "abi01c",
                "abi02c",
                "abi03c",
                "abi07c",
                "abi08c",
                "abi09c",
                "abi10c",
                "abi13c",
            ]
        ),
        conditioning_variables: np.ndarray = np.array(["z500"]),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_data_source: Any | None = None,
        sampler_args: dict[str, float | int] | None = {"num_steps": 100, "S_churn": 10},
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        input_interp_max_dist_km: float = 12.0,
        conditioning_interp_max_dist_km: float = 26.0,
    ):

        super().__init__(
            model_spec=model_spec,
            means=means,
            stds=stds,
            variables=variables,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_variables=conditioning_variables,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            sampler_args=sampler_args,
            input_times=input_times,
            output_times=output_times,
            y_coords=y_coords,
            x_coords=x_coords,
            input_interp_max_dist_km=input_interp_max_dist_km,
            conditioning_interp_max_dist_km=conditioning_interp_max_dist_km,
        )
        self.means: torch.Tensor = self.means[:, -len(self.variables) :, :, :]
        self.stds: torch.Tensor = self.stds[:, -len(self.variables) :, :, :]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.input_times,
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
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
            Coordinate system dictionary.
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.output_times,
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
            }
        )
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "x", 5)
        handshake_dim(input_coords, "y", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_size(input_coords, "y", self.latitudes.shape[0])
        handshake_size(input_coords, "x", self.latitudes.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"][-1]
        )
        return output_coords

    def fetch_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch external conditioning data.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system.
        device : torch.device
            Device on which the conditioning tensor should reside.
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormScopeGOES has been called without initializing the model's conditioning_data_source"
            )

        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=device,
        )

        return conditioning, conditioning_coords

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: str = "6km_60min_natten_cos_zenith_input_eoe_v2",
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(),
    ) -> PrognosticModel:
        """Load model from package.

        Parameters
        ----------
        package : Package
            Package to load model from
        model_name : str, optional
            Model name to load; allows for selection between different variants of the model:
            - "6km_60min_natten_cos_zenith_input_eoe_v2": 6km resolution, 60 minute timestep
            - "6km_10min_natten_pure_obs_zenith_6steps": 6km resolution, 10 minute timestep, sliding window of 6 input timesteps
            - "6km_10min_natten_pure_obs_zenith_eoe": 6km resolution, 10 minute timestep, single input timestep
            - "3km_10min_natten_pure_obs_cos_zenith_input_eoe": 3km resolution, 10 minute timestep
        conditioning_data_source : DataSource | ForecastSource | None, optional
            Data source to use for conditioning, by default None.

        Returns
        -------
        PrognosticModel
            Instantiated StormScopeGOES model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        with open(package.resolve("registry.json")) as f:
            registry = json.load(f)
            pkg = registry[model_name]

        model_spec = []
        for m in pkg["checkpoints"]:
            model = Module.from_checkpoint(package.resolve(m["path"]))
            model_spec.append(
                {
                    "model": model_wrap(model),
                    "sigma_min": float(m["sigma_min"]),
                    "sigma_max": float(m["sigma_max"]),
                }
            )

        # Grid coordinates: crop a subregion from the HRRR grid
        image_size = pkg["image_size"]
        spatial_downsample = pkg["spatial_downsample"]
        latitudes = torch.from_numpy(np.load(package.resolve("lat.npy")))
        longitudes = (
            torch.from_numpy(np.load(package.resolve("lon.npy"))) + 360.0
        ) % 360.0
        hrrr_y, hrrr_x = HRRR.HRRR_Y, HRRR.HRRR_X
        full_y, full_x = latitudes.shape[0], longitudes.shape[1]
        anchor_y = int((full_y - image_size[0]) / 2)
        anchor_x = int((full_x - image_size[1]) / 2)
        latitudes = latitudes[
            anchor_y : anchor_y + image_size[0], anchor_x : anchor_x + image_size[1]
        ]
        longitudes = longitudes[
            anchor_y : anchor_y + image_size[0], anchor_x : anchor_x + image_size[1]
        ]
        y = hrrr_y[anchor_y : anchor_y + image_size[0]]
        x = hrrr_x[anchor_x : anchor_x + image_size[1]]

        # Spatial downsample
        y = y[::spatial_downsample]
        x = x[::spatial_downsample]
        latitudes = latitudes[::spatial_downsample, ::spatial_downsample]
        longitudes = longitudes[::spatial_downsample, ::spatial_downsample]

        # Input/output timesteps configuration
        if pkg["sliding_window"]:
            # N input timesteps, 1 output timestep, with resolution step_interval
            n_steps, step_interval = pkg["n_steps"], pkg["step_interval"]
            input_times = np.arange(-n_steps + 1, 1) * np.timedelta64(
                step_interval, "m"
            )
            output_times = np.array([np.timedelta64(step_interval, "m")])
        else:
            # 1 input, 1 output, with resolution step_interval
            input_times = np.array([np.timedelta64(0, "m")])
            output_times = np.array([np.timedelta64(pkg["step_interval"], "m")])

        # Conditioning variables
        conditioning_variables = np.array(pkg["conditioning_vars"])

        # Normalization constants
        means = torch.from_numpy(np.load(package.resolve("goes_means.npy")))[
            None, :, None, None
        ]
        stds = torch.from_numpy(np.load(package.resolve("goes_stds.npy")))[
            None, :, None, None
        ]
        if len(conditioning_variables) > 0:
            conditioning_means = torch.from_numpy(
                np.expand_dims(np.load(package.resolve("era5_means.npy")), 0)
            )[None, :, None, None]
            conditioning_stds = torch.from_numpy(
                np.expand_dims(np.load(package.resolve("era5_stds.npy")), 0)
            )[None, :, None, None]
        else:
            conditioning_means = torch.empty(0)
            conditioning_stds = torch.empty(0)

        return cls(
            model_spec=model_spec,
            means=means.to(dtype=torch.float32),
            stds=stds.to(dtype=torch.float32),
            latitudes=latitudes.to(dtype=torch.float32),
            longitudes=longitudes.to(dtype=torch.float32),
            conditioning_means=conditioning_means.to(dtype=torch.float32),
            conditioning_stds=conditioning_stds.to(dtype=torch.float32),
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            input_times=input_times,
            output_times=output_times,
            y_coords=y,
            x_coords=x,
            input_interp_max_dist_km=6.0 * spatial_downsample,
        )


class StormScopeMRMS(StormScopeBase):
    """StormScope model forecasting MRMS data on the HRRR grid.

    This model supports multiple variants at different temporal resolutions:
      - 6km resolution, 60 minute timestep
      - 6km resolution, 10 minute timestep
    Selection between these can be made by passing the ``model_name argument`` to this
    class's ``load_model`` method.

    The 6km/10min model uses a sliding window of 6 input timesteps and predicts one
    output timestep; other models use a single input timestep and predict one output
    timestep. All StormScopeMRMS models by default expect GOES-East data as
    conditioning; typically in a forecasting run this can be provided by passing the
    predictions from a StormScopeGOES model to this model's ``call_with_conditioning``
    method. Otherwise, the user must provide a conditioning data source for the model
    to use during inference.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications; see `StormScopeBase`.
    means : torch.Tensor
        Per-variable mean for normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for normalization, shape [1, C, 1, 1].
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    variables : np.ndarray, optional
        MRMS input variables. Default is ["refc"].
    conditioning_variables : np.ndarray, optional
        Auxiliary conditioning variables (typically GOES channels). Default is
        ["abi01c", "abi02c", "abi03c", "abi07c", "abi08c", "abi09c", "abi10c", "abi13c"].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    conditioning_data_source : Any | None, optional
        Data source for external conditioning. Default is None.
    sampler_args : dict[str, float | int] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 h] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [1 h] (i.e., 1 hour from the current time).
    input_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of input data.
        Points beyond this distance are masked as invalid. Default is 12.0.
    conditioning_interp_max_dist_km : float, optional
        Maximum distance in kilometers for nearest neighbor interpolation of conditioning data.
        Points beyond this distance are masked as invalid. Default is 26.0.

    Note
    ----
    To have a unified coordinate system over CONUS for convenience, the model uses the HRRR grid.
    As a result, there are portions of the domain which go beyond the extent of the MRMS data,
    so these portions are masked as invalid (set to NaN).
    """

    _STATE_FIRST = False
    _INPUT_INVALID_FILL_CONSTANT = (
        -0.25285158
    )  # reflectivity of -10 is normalized to this

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        variables: np.ndarray = np.array(["refc"]),
        conditioning_variables: np.ndarray = np.array(
            [
                "abi01c",
                "abi02c",
                "abi03c",
                "abi07c",
                "abi08c",
                "abi09c",
                "abi10c",
                "abi13c",
            ]
        ),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_data_source: Any | None = None,
        sampler_args: dict[str, float | int] | None = {"num_steps": 100, "S_churn": 10},
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        input_interp_max_dist_km: float = 12.0,
        conditioning_interp_max_dist_km: float = 12.0,
    ):

        super().__init__(
            model_spec=model_spec,
            means=means,
            stds=stds,
            variables=variables,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_variables=conditioning_variables,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            sampler_args=sampler_args,
            y_coords=y_coords,
            x_coords=x_coords,
            input_times=input_times,
            output_times=output_times,
            input_interp_max_dist_km=input_interp_max_dist_km,
            conditioning_interp_max_dist_km=conditioning_interp_max_dist_km,
        )
        self.means: torch.Tensor = self.means[:, -len(self.variables) :, :, :]
        self.stds: torch.Tensor = self.stds[:, -len(self.variables) :, :, :]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.input_times,
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
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
            Coordinate system dictionary.
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.output_times,
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
            }
        )
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "x", 5)
        handshake_dim(input_coords, "y", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_size(input_coords, "y", self.latitudes.shape[0])
        handshake_size(input_coords, "x", self.latitudes.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"][-1]
        )
        return output_coords

    def fetch_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch external conditioning data.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system.
        device : torch.device
            Device on which the conditioning tensor should reside.
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormScopeMRMS has been called without initializing the model's conditioning_data_source"
            )

        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=device,
        )
        return conditioning, conditioning_coords

    def prep_input(
        self, x: torch.Tensor, coords: CoordSystem, conditioning: bool = False
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepares the input tensor for the MRMS prognostic model. Same behavior as the base
        class, but with additional value imputation/standardization for low reflectivity values.
        """

        x, x_coords = super().prep_input(x, coords, conditioning=conditioning)

        if not conditioning:
            x = torch.where(
                x <= -20.0, -10, x
            )  # Impute -10 for low reflectivity values

        return x, x_coords

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: str = "6km_60min_natten_cos_zenith_input_mrms_eoe",
        conditioning_data_source: DataSource | ForecastSource | None = None,
    ) -> PrognosticModel:
        """Load model from package.

        Parameters
        ----------
        package : Package
            Package to load model from
        model_name : str, optional
            Model name to load; allows for selection between different variants of the model:
              - "6km_60min_natten_cos_zenith_input_mrms_eoe": 6km resolution, 60 minute timestep
              - "6km_10min_natten_pure_obs_mrms_obs_6steps": 6km resolution, 10 minute timestep
        conditioning_data_source : DataSource | ForecastSource | None, optional
            Data source to use for conditioning, by default None.

        Returns
        -------
        PrognosticModel
            Instantiated StormScopeMRMS model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        with open(package.resolve("registry.json")) as f:
            registry = json.load(f)
            pkg = registry[model_name]

        model_spec = []
        for m in pkg["checkpoints"]:
            model = Module.from_checkpoint(package.resolve(m["path"]))
            model_spec.append(
                {
                    "model": model_wrap(model),
                    "sigma_min": float(m["sigma_min"]),
                    "sigma_max": float(m["sigma_max"]),
                }
            )

        # Grid coordinates: crop a subregion from the HRRR grid
        image_size = pkg["image_size"]
        spatial_downsample = pkg["spatial_downsample"]
        latitudes = torch.from_numpy(np.load(package.resolve("lat.npy")))
        longitudes = (
            torch.from_numpy(np.load(package.resolve("lon.npy"))) + 360.0
        ) % 360.0
        hrrr_y, hrrr_x = HRRR.HRRR_Y, HRRR.HRRR_X
        full_y, full_x = latitudes.shape[0], longitudes.shape[1]
        anchor_y = int((full_y - image_size[0]) / 2)
        anchor_x = int((full_x - image_size[1]) / 2)
        latitudes = latitudes[
            anchor_y : anchor_y + image_size[0], anchor_x : anchor_x + image_size[1]
        ]
        longitudes = longitudes[
            anchor_y : anchor_y + image_size[0], anchor_x : anchor_x + image_size[1]
        ]
        y = hrrr_y[anchor_y : anchor_y + image_size[0]]
        x = hrrr_x[anchor_x : anchor_x + image_size[1]]

        # Spatial downsample
        y = y[::spatial_downsample]
        x = x[::spatial_downsample]
        latitudes = latitudes[::spatial_downsample, ::spatial_downsample]
        longitudes = longitudes[::spatial_downsample, ::spatial_downsample]

        # Input/output timesteps configuration
        if pkg["sliding_window"]:
            # N input timesteps, 1 output timestep, with resolution step_interval
            n_steps, step_interval = pkg["n_steps"], pkg["step_interval"]
            input_times = np.arange(-n_steps + 1, 1) * np.timedelta64(
                step_interval, "m"
            )
            output_times = np.array([np.timedelta64(step_interval, "m")])
        else:
            # 1 input, 1 output, with resolution step_interval
            input_times = np.array([np.timedelta64(0, "m")])
            output_times = np.array([np.timedelta64(pkg["step_interval"], "m")])

        # Conditioning variables
        conditioning_variables = np.array(pkg["conditioning_vars"])

        # Normalization constants
        means = torch.from_numpy(np.load(package.resolve("mrms_means.npy")))[
            None, :, None, None
        ]
        stds = torch.from_numpy(np.load(package.resolve("mrms_stds.npy")))[
            None, :, None, None
        ]
        if len(conditioning_variables) > 0:
            conditioning_means = torch.from_numpy(
                np.load(package.resolve("goes_means.npy"))
            )[None, :, None, None]
            conditioning_stds = torch.from_numpy(
                np.load(package.resolve("goes_stds.npy"))
            )[None, :, None, None]
        else:
            conditioning_means = torch.empty(0)
            conditioning_stds = torch.empty(0)

        return cls(
            model_spec=model_spec,
            means=means.to(dtype=torch.float32),
            stds=stds.to(dtype=torch.float32),
            latitudes=latitudes.to(dtype=torch.float32),
            longitudes=longitudes.to(dtype=torch.float32),
            conditioning_means=conditioning_means.to(dtype=torch.float32),
            conditioning_stds=conditioning_stds.to(dtype=torch.float32),
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            y_coords=y,
            x_coords=x,
            input_times=input_times,
            output_times=output_times,
            input_interp_max_dist_km=6.0 * spatial_downsample,
            conditioning_interp_max_dist_km=6.0 * spatial_downsample,
        )
