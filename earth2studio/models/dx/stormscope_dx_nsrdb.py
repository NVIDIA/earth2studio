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
from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from earth2studio.data import HRRR
from earth2studio.grids import CurvilinearGrid, NearestRegridder
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    coord_array,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource

try:
    from physicsnemo import Module  # type: ignore[import-untyped]
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.utils.insolation import (  # type: ignore[import-untyped]
        insolation as pnm_insolation,
    )

    from earth2studio.models._diffusion import SOLVERS, seeded_rng
    from earth2studio.models._diffusion import solver_options as _solver_options
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
    max_dist_km : float, optional
        Maximum distance (km) from a model gridpoint to the nearest GOES pixel
        regridded onto it by :meth:`default_sources`, by default 12.0.
    number_of_samples : int, optional
        Number of GHI samples per input, by default 1.
    num_steps : int, optional
        Number of diffusion steps, by default 12.
    amp : bool, optional
        Enable automatic mixed precision, by default True.

    solver : str, optional
        Select the numerical method for integrating the sampling ODE in reverse
        time. By default "heun"
    solver_options : dict[str, Any] | None, optional
        Options passed to the selected solver to customize its behavior or
        override defaults. Do not pass ``num_steps`` here; set the sampling step
        count with ``num_steps`` instead. By default None

    Example
    -------
    Estimating GHI from GOES observations, fetched onto the model grid through
    :meth:`default_sources`. Forecasts of the three-kilometer
    :py:class:`earth2studio.models.px.StormScopeGOES` variants share this grid and
    can be passed in place of the observations:

    >>> model = StormScopeDxNSRDB.load_model(StormScopeDxNSRDB.load_default_package())
    >>> model = model.to("cuda")
    >>> x = fetch_data(
    ...     model.default_sources(),
    ...     time=np.array([np.datetime64("2024-07-15T18:00")]),
    ...     variable=model.input_coords()["variable"].values,
    ... ).isel(lead_time=0).expand_dims(batch=[0])
    >>> ghi = model(x)

    Badges
    ------
    region:na class:nowcasting product:solar year:2026 gpu:48gb
    provider:nvidia backend:pytorch
    """

    _SAMPLER_DTYPE = torch.float64
    _MODEL_DTYPE = torch.float32
    _CLEARNESS_INDEX_EPS = 10.0
    _SOLAR_CONSTANT = 1361.0
    _INPUT_GRID = "goes-east-conus"

    stochastic = True

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
        max_dist_km: float = 12.0,
        number_of_samples: int = 1,
        num_steps: int = 12,
        amp: bool = True,
        *,
        solver: str = "heun",
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if solver not in SOLVERS:
            raise ValueError(
                f"Unknown solver {solver!r}; choose from {tuple(SOLVERS)}."
            )
        self.solver = solver
        if solver_options is not None and "num_steps" in solver_options:
            raise ValueError(
                "Do not pass 'num_steps' in solver_options; use num_steps instead."
            )
        self.solver_options = dict(solver_options or {})
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
        self.max_dist_km = max_dist_km
        self.number_of_samples = number_of_samples
        self.num_steps = num_steps
        self.amp = amp

    def _native_grid(self) -> CurvilinearGrid:
        return CurvilinearGrid(
            self._lat_cpu_copy, self._lon_cpu_copy, y=self.y, x=self.x
        )

    def _input_regridder(self) -> NearestRegridder:
        """Nearest-neighbor regridder of GOES-East CONUS onto the model grid."""
        from earth2studio.models.px.stormscope import _default_regridder

        return _default_regridder(
            NearestRegridder,
            self._INPUT_GRID,
            self._native_grid(),
            max_dist_km=self.max_dist_km,
        )

    def _set_default_coverage(self) -> None:
        """Set the input mask to the model gridpoints the default source covers."""
        coverage = torch.from_numpy(np.array(self._input_regridder().coverage))
        self.register_buffer("input_valid_mask", coverage.to(self.latitudes.device))

    def default_sources(self) -> "DataSource":
        """Recommend GOES-East CONUS imagery on the model grid.

        Returns
        -------
        DataSource
            GOES input source, regridded to the model grid by nearest neighbor
            within ``max_dist_km``; uncovered gridpoints are NaN.
        """
        from earth2studio.data import GOES, RegriddedSource

        return RegriddedSource(GOES(scan_mode="C"), self._input_regridder())

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature for GOES observations.
        """
        return coord_array(
            ("batch", "time", "variable", "y", "x"),
            {"variable": self.conditioning_variables.copy()},
            dynamic=("batch", "time"),
            grid=self._native_grid(),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the diagnostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform,
            on the model grid.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature for sampled GHI.
        """
        handshake_dataarray(input_coords, self.input_coords())
        leading = input_coords.dims[:-3]
        native = self._native_grid()
        coords = {
            k: v.variable.copy(deep=True)
            for k, v in input_coords.coords.items()
            if "variable" not in v.dims
        }
        coords.update(
            variable=self.output_variables.copy(),
            sample=np.arange(self.number_of_samples),
        )
        # Samples follow the leading prefix so dynamic time remains a wildcard.
        result = coord_array(
            (*leading, "sample", "variable", *native.dims),
            coords,
            sizes={d: input_coords.sizes[d] for d in leading},
            dynamic=input_coords.attrs.get("earth2studio_dynamic_dims", ()),
            grid=native,
            dtype=input_coords.dtype,
            name=input_coords.name,
            attrs=deepcopy(input_coords.attrs),
        )
        result.encoding = deepcopy(input_coords.encoding)
        return result

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
        number_of_samples: int = 1,
        *,
        solver: str = "heun",
        solver_options: dict[str, Any] | None = None,
    ) -> DiagnosticModel:
        """Load the StormScope NSRDB diagnostic.

        Parameters
        ----------
        package : Package
            Package containing model assets.
        number_of_samples : int, optional
            Number of GHI samples, by default 1.

        solver : str, optional
            Select the numerical method for integrating the sampling ODE in
            reverse time. By default "heun"
        solver_options : dict[str, Any] | None, optional
            Options passed to the selected solver to customize its behavior or
            override defaults. Do not pass ``num_steps`` here; set the sampling step
            count with ``num_steps`` instead. By default None

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

        model = cls(
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
            max_dist_km=6.0 * spatial_downsample,
            solver=solver,
            solver_options=solver_options,
            number_of_samples=number_of_samples,
        )
        model._set_default_coverage()
        return model

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

    def _target_datetimes(self, times: np.ndarray) -> np.ndarray:
        return np.array(
            [
                datetime.fromtimestamp(
                    time.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for time in times
            ]
        )

    def _insolation(
        self, times: np.ndarray, batch_size: int, scale: float
    ) -> torch.Tensor:
        import pandas as pd  # type: ignore[import-untyped]

        target = self._target_datetimes(times)
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

    def _build_condition(self, x: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
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
        scheduler = EDMNoiseScheduler(
            sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=rho
        )

        def predict_x0(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            return self.diffusion_model(
                x.to(self._MODEL_DTYPE),
                t.to(self._MODEL_DTYPE),
                condition=condition,
            ).to(self._SAMPLER_DTYPE)

        steps = scheduler.timesteps(
            self.num_steps, device=latents.device, dtype=self._SAMPLER_DTYPE
        )
        return sample(
            scheduler.get_denoiser(x0_predictor=predict_x0),
            latents.to(self._SAMPLER_DTYPE) * steps[0],
            scheduler,
            self.num_steps,
            solver=self.solver,
            solver_options=_solver_options(
                scheduler,
                self.solver,
                num_steps=self.num_steps,
                options=self.solver_options,
            ),
            time_steps=steps,
        )

    _rng_generator: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the Gaussian initial noise used by the NSRDB sampler.

        Parameters
        ----------
        seed : int
            Seed for the initial-noise stream.
        reset : bool, optional
            Reset an existing stream, by default True.
        """
        if reset or self._rng_generator is None:
            self._rng_generator = torch.Generator().manual_seed(seed)

    def _forward_sample(self, x: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
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
            device=device if self._rng_generator is None else "cpu",
            dtype=self._SAMPLER_DTYPE,
            generator=self._rng_generator,
        ).to(device)
        initial_state = regression / self.sigma_max + noise
        generator = None
        if self._rng_generator is not None and self.solver.startswith(
            "edm_stochastic_"
        ):
            seed = int(
                torch.randint(2**63 - 1, (), generator=self._rng_generator).item()
            )
            generator = torch.Generator(device=device).manual_seed(seed)
        with (
            seeded_rng(generator),
            torch.autocast(device_type=device.type, enabled=self.amp),
        ):
            output = self._edm_sampler(initial_state, condition).to(dtype)
        output = output.reshape(
            batch_size, time_size, len(self.output_variables), height, width
        )
        output = output * (insolation + self._CLEARNESS_INDEX_EPS)
        output = torch.clamp(self._sanitize(output), min=0)
        return torch.where(self.valid_mask, output, torch.nan)

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Generate GHI samples at each input validity time on the model device.

        Parameters
        ----------
        x : xr.DataArray
            GOES observations matching ``input_coords()`` on the model grid.

        Returns
        -------
        xr.DataArray
            Global horizontal irradiance samples matching ``output_coords(x)``,
            with labelled sample and spatial dimensions.
        """
        handshake_nonempty(x)
        output_coords = self.output_coords(x)
        handshake_time(x, dimension=False)
        valid = x.coords["time"]
        if "lead_time" in x.coords:
            handshake_time(x, "lead_time", dimension="lead_time" in x.dims)
            valid = valid + x.coords["lead_time"]
        leading = x.dims[:-3]
        template = xr.DataArray(
            np.empty(tuple(x.sizes[d] for d in leading)), dims=leading
        )
        times = valid.broadcast_like(template).transpose(*leading).values.reshape(-1)
        tensor = x.e2s.to_torch()[0].to(self.latitudes.device)
        tensor = torch.where(self.input_valid_mask, tensor, 0.0)
        # The numerical kernel retains its [batch, time, channel, y, x] layout.
        tensor = tensor.reshape(1, -1, *tensor.shape[-3:])
        samples = [
            self._forward_sample(tensor, times).reshape(
                *[x.sizes[d] for d in leading],
                len(self.output_variables),
                len(self.y),
                len(self.x),
            )
            for _ in range(self.number_of_samples)
        ]
        position = output_coords.dims.index("sample")
        result = from_torch(torch.stack(samples, dim=position), output_coords)
        result.encoding = deepcopy(x.encoding)
        return result
