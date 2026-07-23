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

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import ArrayLike

from earth2studio.data import HRRR
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import NearestNeighborInterpolator
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo import Module  # type: ignore[import-untyped]
    from physicsnemo.utils.insolation import (  # type: ignore[import-untyped]
        insolation as pnm_insolation,
    )
except ImportError:
    OptionalDependencyFailure("stormscope")
    Module = None  # type: ignore[assignment]
    pnm_insolation = None  # type: ignore[assignment]


class _MaskedModel(nn.Module):
    def __init__(self, model: nn.Module, mask: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mask", mask)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs) * self.mask


@check_optional_dependencies()
class StormScopeDxNSRDB(torch.nn.Module, AutoModelMixin):
    """Estimate Global Horizontal Irradiance from GOES imagery.

    This diagnostic model is designed to be used with the
    :py:class:`earth2studio.models.px.StormScopeGOES` prognostic model, consuming
    its GOES channel outputs to produce solar irradiance estimates.

    A regression network produces a first estimate that is refined with a
    warm-started diffusion sampler. Each requested sample is generated
    independently at the input observation time.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/StormScope-NSRDB

    Parameters
    ----------
    diffusion_model : nn.Module
        Diffusion denoiser.
    regression_model : nn.Module
        Deterministic first-guess model.
    sigma_min : float
        Minimum diffusion noise level.
    sigma_max : float
        Warm-start diffusion noise level.
    conditioning_means : torch.Tensor
        Per-channel GOES means.
    conditioning_stds : torch.Tensor
        Per-channel GOES standard deviations.
    conditioning_variables : np.ndarray
        GOES input variables.
    output_variables : np.ndarray
        Output variables, typically ``["ghi"]``.
    latitudes : torch.Tensor
        Native-grid latitudes with shape ``[H, W]``.
    longitudes : torch.Tensor
        Native-grid longitudes with shape ``[H, W]``.
    invariants : torch.Tensor | None, optional
        Static conditioning channels, by default None.
    valid_mask : torch.Tensor | None, optional
        Native-grid output mask, by default None.
    y_coords : np.ndarray | None, optional
        Native-grid y coordinates, by default None.
    x_coords : np.ndarray | None, optional
        Native-grid x coordinates, by default None.
    input_interp_max_dist_km : float, optional
        Maximum input interpolation distance, by default 12.0.
    number_of_samples : int, optional
        Number of GHI samples per input, by default 1.
    seed : int | None, optional
        Base random seed, by default None.
    num_steps : int, optional
        Number of diffusion steps, by default 12.
    amp : bool, optional
        Enable automatic mixed precision, by default True.

    Example
    -------
    Using the diagnostic with forecasted GOES imagery from
    :py:class:`earth2studio.models.px.StormScopeGOES`. A
    :py:class:`earth2studio.data.GOES` data source can also be used directly
    in place of the prognostic model output:

    >>> # Load StormScopeGOES prognostic and NSRDB diagnostic models
    >>> package = StormScopeBase.load_default_package()
    >>> goes_model = StormScopeGOES.load_model(package, conditioning_data_source=None)
    >>> goes_model = goes_model.to("cuda").eval()
    >>> nsrdb_model = StormScopeDxNSRDB.load_model(StormScopeDxNSRDB.load_default_package())
    >>> nsrdb_model = nsrdb_model.to("cuda")
    >>>
    >>> # Build interpolators from GOES grid to model grid
    >>> goes_lat, goes_lon = GOES.grid(satellite="goes16", scan_mode="C")
    >>> goes_model.build_input_interpolator(goes_lat, goes_lon)
    >>> nsrdb_model.build_input_interpolator(goes_lat, goes_lon)
    >>>
    >>> # Run one GOES forecast step then estimate GHI
    >>> y_goes, y_coords = goes_model(x, coords)
    >>> ghi_coords = y_coords.copy()
    >>> del ghi_coords["lead_time"]
    >>> ghi, ghi_coords = nsrdb_model(y_goes.squeeze(2), ghi_coords)

    Badges
    ------
    region:na class:nwc product:solar year:2026 gpu:60gb
    """

    _SAMPLER_DTYPE = torch.float64
    _MODEL_DTYPE = torch.float32
    _CLEARNESS_INDEX_EPS = 10.0
    _SOLAR_CONSTANT = 1361.0

    def __init__(
        self,
        diffusion_model: nn.Module,
        regression_model: nn.Module,
        sigma_min: float,
        sigma_max: float,
        conditioning_means: torch.Tensor,
        conditioning_stds: torch.Tensor,
        conditioning_variables: np.ndarray,
        output_variables: np.ndarray,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        invariants: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        input_interp_max_dist_km: float = 12.0,
        number_of_samples: int = 1,
        seed: int | None = None,
        num_steps: int = 12,
        amp: bool = True,
    ) -> None:
        super().__init__()
        if number_of_samples < 1:
            raise ValueError("number_of_samples must be positive")
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self.register_buffer("conditioning_means", conditioning_means)
        self.register_buffer("conditioning_stds", conditioning_stds)
        self.register_buffer(
            "valid_mask",
            (
                torch.ones_like(latitudes, dtype=torch.bool)
                if valid_mask is None
                else valid_mask.to(dtype=torch.bool)
            ),
        )
        self.register_buffer(
            "input_valid_mask", torch.ones_like(latitudes, dtype=torch.bool)
        )
        if invariants is not None:
            self.register_buffer("invariants", invariants)
        else:
            self.invariants = None

        mask = self.valid_mask.reshape(1, 1, *self.valid_mask.shape).to(
            dtype=torch.float32
        )
        self.diffusion_model = _MaskedModel(diffusion_model, mask)
        self.regression_model = regression_model
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.conditioning_variables = np.asarray(conditioning_variables)
        self.output_variables = np.asarray(output_variables)
        self.y = (
            np.asarray(y_coords)
            if y_coords is not None
            else np.arange(latitudes.shape[0])
        )
        self.x = (
            np.asarray(x_coords)
            if x_coords is not None
            else np.arange(longitudes.shape[1])
        )
        self._lat_cpu_copy = self.latitudes.detach().cpu().numpy()
        self._lon_cpu_copy = self.longitudes.detach().cpu().numpy()
        self._input_interp_max_dist_km = input_interp_max_dist_km
        self.input_interp: nn.Module | None = None
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.num_steps = num_steps
        self.amp = amp

    def input_coords(self) -> CoordSystem:
        """Input coordinate system.

        Returns
        -------
        CoordSystem
            GOES input coordinates.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "variable": self.conditioning_variables.copy(),
                "y": self.y.copy(),
                "x": self.x.copy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system.

        Parameters
        ----------
        input_coords : CoordSystem
            Native-grid GOES input coordinates.

        Returns
        -------
        CoordSystem
            Sampled GHI output coordinates.
        """
        target = self.input_coords()
        handshake_dim(input_coords, "variable", 2)
        handshake_dim(input_coords, "y", 3)
        handshake_dim(input_coords, "x", 4)
        handshake_coords(input_coords, target, "variable")
        handshake_coords(input_coords, target, "y")
        handshake_coords(input_coords, target, "x")
        return OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "time": input_coords["time"],
                "variable": self.output_variables.copy(),
                "y": self.y.copy(),
                "x": self.x.copy(),
            }
        )

    def __str__(self) -> str:
        return "StormScopeDxNSRDB"

    @classmethod
    def load_default_package(cls) -> Package:
        """Default StormScope NSRDB package.

        Returns
        -------
        Package
            Model package.
        """
        return Package(
            "hf://nvidia/StormScope-NSRDB",
            cache_options={
                "cache_storage": Package.default_cache("stormscope_solar_nsrdb")
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        model_name: str = "stormscope_solar_goes_nsrdb",
        number_of_samples: int = 1,
        seed: int | None = None,
    ) -> DiagnosticModel:
        """Load the StormScope NSRDB diagnostic.

        Parameters
        ----------
        package : Package
            Package containing model assets.
        number_of_samples : int, optional
            Number of GHI samples, by default 1.
        seed : int | None, optional
            Base random seed, by default None.

        Returns
        -------
        DiagnosticModel
            Loaded diagnostic model.
        """
        try:
            package.resolve("config.json")
        except FileNotFoundError:
            pass

        registry = cls._load_registry(package)
        package_spec = registry["stormscope_solar_goes_nsrdb"]
        checkpoints = package_spec["checkpoints"]
        if len(checkpoints) != 1:
            raise ValueError("StormScopeDxNSRDB requires one diffusion checkpoint")
        checkpoint = checkpoints[0]
        diffusion_model = Module.from_checkpoint(package.resolve(checkpoint["path"]))
        regression_model = Module.from_checkpoint(
            package.resolve(package_spec["regression_checkpoint"]["path"])
        )
        diffusion_model.eval().requires_grad_(False)
        regression_model.eval().requires_grad_(False)

        latitudes, longitudes, y, x, spatial_downsample = cls._build_grid(
            package, package_spec
        )
        conditioning_variables = np.array(package_spec["conditioning_vars"])
        conditioning_means, conditioning_stds = cls._build_normalization(
            package, registry, conditioning_variables
        )

        lat_rad = np.deg2rad(latitudes.numpy())
        lon_rad = np.deg2rad(longitudes.numpy())
        invariants = np.stack(
            [
                np.sin(lat_rad),
                np.cos(lat_rad),
                np.sin(lon_rad),
                np.cos(lon_rad),
                cls._load_invariant(package, "altitude.npy", package_spec).numpy(),
                cls._load_invariant(package, "elev_std.npy", package_spec).numpy(),
            ]
        ).astype(np.float32)
        valid_mask = cls._load_invariant(package, "nsrdb_mask.npy", package_spec) > 0

        return cls(
            diffusion_model=diffusion_model,
            regression_model=regression_model,
            sigma_min=float(checkpoint["sigma_min"]),
            sigma_max=float(checkpoint["sigma_max"]),
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_variables=conditioning_variables,
            output_variables=np.array(package_spec["variables"]),
            latitudes=latitudes.to(dtype=torch.float32),
            longitudes=longitudes.to(dtype=torch.float32),
            invariants=torch.from_numpy(np.nan_to_num(invariants)),
            valid_mask=valid_mask,
            y_coords=y,
            x_coords=x,
            input_interp_max_dist_km=6.0 * spatial_downsample,
            number_of_samples=number_of_samples,
            seed=seed,
        )

    def build_input_interpolator(
        self,
        input_lats: torch.Tensor | ArrayLike,
        input_lons: torch.Tensor | ArrayLike,
        max_dist_km: float | None = None,
    ) -> nn.Module:
        """Build an interpolator from an input grid to the model grid.

        Parameters
        ----------
        input_lats : torch.Tensor | ArrayLike
            Input latitudes.
        input_lons : torch.Tensor | ArrayLike
            Input longitudes.
        max_dist_km : float | None, optional
            Maximum nearest-neighbor distance, by default None.

        Returns
        -------
        nn.Module
            Input interpolation module.
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
        interpolator = cast(NearestNeighborInterpolator, self.input_interp)
        if torch.any(~interpolator.valid_mask):
            logger.warning(
                "Some input grid points are invalid after StormScope interpolation"
            )
        self.input_valid_mask = interpolator.valid_mask.reshape(
            len(self.y), len(self.x)
        ).to(self.latitudes.device)
        return self.input_interp

    @staticmethod
    def _load_registry(package: Package) -> dict[str, Any]:
        with open(package.resolve("registry.json")) as registry_file:
            return json.load(registry_file)

    @classmethod
    def _build_normalization(
        cls, package: Package, registry: dict[str, Any], names: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalization = registry["normalization"]
        name_to_location: dict[str, tuple[str, int]] = {}
        for group_name, group in normalization.items():
            for index, name in enumerate(group["order"]):
                name_to_location[name] = (group_name, index)

        means: np.ndarray = np.zeros(len(names), dtype=np.float32)
        stds: np.ndarray = np.ones(len(names), dtype=np.float32)
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for output_index, name in enumerate(names):
            if name not in name_to_location:
                raise KeyError(f"No normalization is defined for '{name}'")
            group_name, input_index = name_to_location[name]
            group = normalization[group_name]
            if group.get("file_prefix") is None:
                continue
            if group_name not in arrays:
                prefix = group["file_prefix"]
                arrays[group_name] = (
                    np.atleast_1d(np.load(package.resolve(f"{prefix}_means.npy"))),
                    np.atleast_1d(np.load(package.resolve(f"{prefix}_stds.npy"))),
                )
            group_means, group_stds = arrays[group_name]
            means[output_index] = group_means[input_index]
            stds[output_index] = group_stds[input_index]
        return (
            torch.from_numpy(means)[None, :, None, None],
            torch.from_numpy(stds)[None, :, None, None],
        )

    @staticmethod
    def _crop_invariant(
        array: torch.Tensor, image_size: list[int], spatial_downsample: int
    ) -> torch.Tensor:
        anchor_y = int((array.shape[0] - image_size[0]) / 2)
        anchor_x = int((array.shape[1] - image_size[1]) / 2)
        array = array[
            anchor_y : anchor_y + image_size[0],
            anchor_x : anchor_x + image_size[1],
        ]
        return array[::spatial_downsample, ::spatial_downsample]

    @classmethod
    def _load_invariant(
        cls, package: Package, filename: str, package_spec: dict[str, Any]
    ) -> torch.Tensor:
        array = torch.from_numpy(np.load(package.resolve(filename))).to(
            dtype=torch.float32
        )
        return cls._crop_invariant(
            array,
            package_spec["image_size"],
            package_spec["spatial_downsample"],
        )

    @classmethod
    def _build_grid(
        cls, package: Package, package_spec: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        image_size = package_spec["image_size"]
        spatial_downsample = package_spec["spatial_downsample"]
        latitudes = torch.from_numpy(np.load(package.resolve("lat.npy")))
        longitudes = (
            torch.from_numpy(np.load(package.resolve("lon.npy"))) + 360.0
        ) % 360.0
        anchor_y = int((latitudes.shape[0] - image_size[0]) / 2)
        anchor_x = int((longitudes.shape[1] - image_size[1]) / 2)
        latitudes = cls._crop_invariant(latitudes, image_size, spatial_downsample)
        longitudes = cls._crop_invariant(longitudes, image_size, spatial_downsample)
        y = HRRR.HRRR_Y[anchor_y : anchor_y + image_size[0]][::spatial_downsample]
        x = HRRR.HRRR_X[anchor_x : anchor_x + image_size[1]][::spatial_downsample]
        return latitudes, longitudes, y, x, spatial_downsample

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_input(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        native_grid = (
            "y" in coords
            and "x" in coords
            and np.array_equal(coords["y"], self.y)
            and np.array_equal(coords["x"], self.x)
        )
        if native_grid:
            output_coords = coords.copy()
        else:
            if self.input_interp is None:
                raise ValueError(
                    "Using GOES data on a non-native grid requires "
                    "build_input_interpolator"
                )
            x = self.input_interp(x)
            output_coords = coords.copy()
            output_coords.popitem()
            output_coords.popitem()
            output_coords["y"] = self.y
            output_coords["x"] = self.x
        return torch.where(self.input_valid_mask, x, 0.0), output_coords

    def _target_datetimes(self, coords: CoordSystem) -> np.ndarray:
        times = np.asarray(coords["time"]).astype(np.datetime64)
        return np.array(
            [
                datetime.fromtimestamp(
                    time.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for time in times
            ]
        )

    def _insolation(
        self, coords: CoordSystem, batch_size: int, scale: float
    ) -> torch.Tensor:
        import pandas as pd  # type: ignore[import-untyped]

        target = self._target_datetimes(coords)
        dates = np.array([pd.Timestamp(time) for time in np.tile(target, batch_size)])
        insolation = pnm_insolation(
            dates,
            self._lat_cpu_copy.astype(np.float32),
            self._lon_cpu_copy.astype(np.float32),
            scale=scale,
            daily=False,
            clip_zero=True,
        )
        return torch.from_numpy(insolation)[:, None]

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.conditioning_means) / self.conditioning_stds

    def _build_condition(self, x: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
        batch_size, time_size = x.shape[:2]
        parts = [
            self._sanitize(x).reshape(batch_size * time_size, *x.shape[2:]),
            self._sanitize(self._insolation(coords, batch_size, scale=1.0)).to(
                device=x.device, dtype=x.dtype
            ),
        ]
        if self.invariants is not None:
            invariants = self._sanitize(self.invariants).to(
                device=x.device, dtype=x.dtype
            )
            parts.append(invariants.repeat(batch_size * time_size, 1, 1, 1))
        return torch.cat(parts, dim=1)

    def _edm_sampler(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
        rho: float = 7,
    ) -> torch.Tensor:
        step_indices = torch.arange(
            self.num_steps, dtype=torch.float64, device=latents.device
        )
        steps = (
            self.sigma_max ** (1 / rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        steps = torch.cat([steps, torch.zeros_like(steps[:1])])
        next_state = latents.to(self._SAMPLER_DTYPE) * steps[0]

        for index, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
            current_state = next_state
            current_state_hat = current_state
            batch_size = current_state_hat.shape[0]
            current_step_batch = (
                current_step.reshape(1).expand(batch_size).to(self._MODEL_DTYPE)
            )
            denoised = self.diffusion_model(
                current_state_hat.to(self._MODEL_DTYPE),
                current_step_batch,
                condition=condition,
            ).to(self._SAMPLER_DTYPE)
            current_derivative = (current_state_hat - denoised) / current_step
            next_state = (
                current_state_hat + (next_step - current_step) * current_derivative
            )

            if index < self.num_steps - 1:
                next_step_batch = (
                    next_step.reshape(1).expand(batch_size).to(self._MODEL_DTYPE)
                )
                denoised = self.diffusion_model(
                    next_state.to(self._MODEL_DTYPE),
                    next_step_batch,
                    condition=condition,
                ).to(self._SAMPLER_DTYPE)
                next_derivative = (next_state - denoised) / next_step
                next_state = current_state_hat + (next_step - current_step) * (
                    0.5 * current_derivative + 0.5 * next_derivative
                )
        return next_state

    def _forward_sample(self, x: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("StormScopeDxNSRDB requires [batch, time, variable, y, x]")
        batch_size, time_size = x.shape[:2]
        height, width = self.latitudes.shape
        device, dtype = x.device, x.dtype
        insolation = self._insolation(
            coords, batch_size, scale=self._SOLAR_CONSTANT
        ).to(device=device, dtype=dtype)
        insolation = insolation.reshape(batch_size, time_size, 1, height, width)

        normalized = self._sanitize(self._normalize_input(x))
        normalized = torch.where(self.input_valid_mask, normalized, 0.0)
        condition = self._build_condition(normalized, coords)
        regression = self._sanitize(self.regression_model(condition))
        regression = regression * self.valid_mask.reshape(1, 1, height, width).to(
            dtype=regression.dtype
        )
        noise = torch.randn(
            (batch_size * time_size, len(self.output_variables), height, width),
            device=device,
            dtype=self._SAMPLER_DTYPE,
        )
        initial_state = regression / self.sigma_max + noise
        with torch.autocast(device_type=device.type, enabled=self.amp):
            output = self._edm_sampler(initial_state, condition).to(dtype)
        output = output.reshape(
            batch_size, time_size, len(self.output_variables), height, width
        )
        output = output * (insolation + self._CLEARNESS_INDEX_EPS)
        output = torch.clamp(self._sanitize(output), min=0)
        return torch.where(self.valid_mask, output, torch.nan)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Generate GHI samples from GOES imagery.

        Parameters
        ----------
        x : torch.Tensor
            GOES tensor with shape ``[batch, time, variable, y, x]``.
        coords : CoordSystem
            GOES coordinates.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            GHI samples and output coordinates.
        """
        x, coords = self._prepare_input(x, coords)
        output_coords = self.output_coords(coords)
        samples = []
        for sample_index in range(self.number_of_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + sample_index)
            samples.append(self._forward_sample(x, coords))
        return torch.stack(samples, dim=1), output_coords
