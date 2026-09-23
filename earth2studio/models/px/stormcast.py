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
from collections.abc import Generator, Iterator
from copy import deepcopy
from itertools import product
from typing import cast

import numpy as np
import torch
import xarray as xr
import zarr

from earth2studio.data import GFS_FX, DataSource, ForecastSource, fetch_data
from earth2studio.grids import ProjectedGrid, resolve_grid
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordinateSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners.legacy import EDMPrecond
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.models.diffusion_unets import StormCastUNet
except ImportError:
    OptionalDependencyFailure("stormcast")
    StormCastUNet = None
    EDMNoiseScheduler = None
    EDMPrecond = None
    OmegaConf = None
    sample = None


# Variables used in StormCastV1 paper
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

INVARIANTS = ["lsm", "orography"]


@check_optional_dependencies()
class StormCast(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """StormCast generative convection-allowing model for regional forecasts consists of
    two core models: a regression and diffusion model. Model time step size is 1 hour,
    taking as input:

    - High-resolution (3km) HRRR state over the central United States (99 vars)
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
    regression_model : torch.nn.Module
        Deterministic model used to make an initial prediction
    diffusion_model : torch.nn.Module
        Generative model correcting the deterministic prediciton
    means : torch.Tensor
        Mean value of each input high-resolution variable
    stds : torch.Tensor
        Standard deviation of each input high-resolution variable
    invariants : torch.Tensor
        Static invariant  quantities
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, defaults to be the StormCastV1 region in central
        United States, by default (273, 785)
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, defaults to be the StormCastV1 region in central
        United States,, by default (579, 1219)
    variables : np.array, optional
        High-resolution variables, by default np.array(VARIABLES)
    conditioning_means : torch.Tensor | None, optional
        Means to normalize conditioning data, by default None
    conditioning_stds : torch.Tensor | None, optional
        Standard deviations to normalize conditioning data, by default None
    conditioning_variables : np.array, optional
        Global variables for conditioning, by default np.array(CONDITIONING_VARIABLES)
    conditioning_data_source : DataSource | ForecastSource | None, optional
        Data Source to use for global conditioning. Required for running in iterator mode, by default None
    sampler_steps : int, optional
        Number of diffusion sampler steps, by default 36
    sampler_args : dict[str, float  |  int], optional
        Arguments to pass to the diffusion sampler, by default None

    Badges
    ------
    region:na class:nowcasting product:wind product:temp product:radar product:atmos year:2024
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        regression_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (273, 785),
        hrrr_lon_lim: tuple[int, int] = (579, 1219),
        variables: np.array = np.array(VARIABLES),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.array = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | ForecastSource | None = None,
        sampler_steps: int = 18,
        sampler_args: dict[str, float | int] | None = None,
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.sampler_steps = sampler_steps
        self.sampler_args: dict[str, float | int] = {
            "sigma_min": 0.002,
            "sigma_max": 800,
            "rho": 7,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1,
        }
        if sampler_args is not None:
            self.sampler_args.update(sampler_args)

        parent = cast(ProjectedGrid, resolve_grid("hrrr"))
        self.grid = ProjectedGrid(
            parent.y[slice(*hrrr_lat_lim)], parent.x[slice(*hrrr_lon_lim)], parent.crs
        )
        self.lat = np.asarray(self.grid.coords()["lat"])
        self.lon = np.asarray(self.grid.coords()["lon"])
        self.hrrr_x, self.hrrr_y = self.grid.x, self.grid.y
        self._conditioning_grid: tuple[np.ndarray, np.ndarray] | None = None
        self._conditioning_interp: LatLonInterpolation | None = None

        self.variables = variables

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source was provided to StormCast, "
                + "set the conditioning_data_source attribute of the model "
                + "before running inference."
            )

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)

        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system"""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input and declare the next hourly forecast without allocation."""
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(1, "h")}
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://nvidia/stormcast-v1-era5-hrrr@6c89a0877a0d6b231033d3b0d8b9828a6f833ed8",
            cache_options={
                "cache_storage": Package.default_cache("stormcast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(),
        sampler_steps: int = 18,
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source to use for global conditioning, by default GFS_FX
        sampler_steps : int, optional
            Number of diffusion sampler steps, by default 18

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            # Likely already registered so skip
            pass

        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        # TODO: remove strict=False once checkpoints/imports updated to new diffusion API
        regression = StormCastUNet.from_checkpoint(
            package.resolve("StormCastUNet.0.0.mdlus"),
            strict=False,
        )
        diffusion = EDMPrecond.from_checkpoint(
            package.resolve("EDMPrecond.0.0.mdlus"),
            strict=False,
        )

        # Load metadata: means, stds, grid
        store = zarr.storage.ZipStore(package.resolve("metadata.zarr.zip"), mode="r")
        metadata = xr.open_zarr(store, zarr_format=2)

        variables = metadata["variable"].values
        conditioning_variables = metadata["conditioning_variable"].values

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
        invariants = metadata["invariants"].sel(invariant=config.data.invariants).values
        invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)

        # EDM sampler arguments
        if config.sampler_args is not None:
            sampler_args = config.sampler_args
        else:
            sampler_args = {}

        return cls(
            regression,
            diffusion,
            means,
            stds,
            invariants,
            variables=variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            sampler_steps=sampler_steps,
            sampler_args=sampler_args,
        )

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:

        # Scale data
        if "conditioning_means" in self._buffers:
            conditioning = conditioning - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            conditioning = conditioning / self.conditioning_stds

        x = (x - self.means) / self.stds

        # Run regression model
        invariant_tensor = self.invariants.repeat(x.shape[0], 1, 1, 1)
        concats = torch.cat((x, conditioning, invariant_tensor), dim=1)

        out = self.regression_model(concats)

        # Concat for diffusion conditioning
        condition = torch.cat((x, out, invariant_tensor), dim=1)
        latents = torch.randn_like(x)
        latents = self.sampler_args["sigma_max"] * latents

        def _conditional_diffusion(
            latent_x: torch.Tensor, t: torch.Tensor
        ) -> torch.Tensor:
            return self.diffusion_model(latent_x, t, condition=condition)

        scheduler = EDMNoiseScheduler(
            sigma_min=self.sampler_args["sigma_min"],
            sigma_max=self.sampler_args["sigma_max"],
            rho=self.sampler_args["rho"],
        )
        denoiser = scheduler.get_denoiser(x0_predictor=_conditional_diffusion)
        edm_out = sample(
            denoiser,
            latents,
            noise_scheduler=scheduler,
            num_steps=self.sampler_steps,
            solver="edm_stochastic_heun",
            solver_options={
                "S_churn": self.sampler_args["S_churn"],
                "S_min": self.sampler_args["S_min"],
                "S_max": self.sampler_args["S_max"],
                "S_noise": self.sampler_args["S_noise"],
            },
        )

        out += edm_out

        out = out * self.stds + self.means

        return out

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : xr.DataArray
            Input field on the declared projected grid.

        Returns
        -------
        xr.DataArray
            Hourly forecast field.

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCast has been called without initializing the model's conditioning_data_source"
            )

        output_coords = self.output_coords(x)
        encoding = deepcopy(x.encoding)
        x, coords = x.e2s.to_torch()
        x = x.to(self.means.device)
        conditioning = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=x.device,
            target_grid=self.grid,
            regridder="linear",
        )
        conditioning = conditioning.transpose(
            "time", "lead_time", "variable", "lat", "lon"
        )
        source = (conditioning.lat.values, conditioning.lon.values)
        if self._conditioning_grid is None or any(
            not np.array_equal(a, b) for a, b in zip(source, self._conditioning_grid)
        ):
            lat, lon = np.meshgrid(*source, indexing="ij")
            self._conditioning_interp = LatLonInterpolation(
                lat, lon, self.lat, self.lon
            )
            self._conditioning_grid = tuple(a.copy() for a in source)
        conditioning, _ = conditioning.e2s.to_torch()
        conditioning = cast(LatLonInterpolation, self._conditioning_interp).to(
            device=x.device, dtype=conditioning.dtype
        )(conditioning)

        # Add a batch dim
        conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)

        x = x.clone()  # prevent editing of argument
        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                for k, _ in enumerate(coords["lead_time"]):
                    x[i, j, k : k + 1] = self._forward(
                        x[i, j, k : k + 1], conditioning[i, j, k : k + 1]
                    )

        out = from_torch(x, output_coords)
        out.attrs = deepcopy(out.attrs)
        out.encoding = encoding
        return out

    def _default_generator(
        self,
        x: xr.DataArray,
    ) -> Generator[xr.DataArray, None, None]:

        handshake_nonempty(x)
        handshake_time(x)
        self.output_coords(x)
        x = x.copy(deep=True)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=True)

        if self.conditioning_data_source is None:
            raise ValueError(
                "A conditioning data source must be available for the iterator to function."
            )

        while True:
            # Front hook
            x = self.front_hook(x.copy(deep=True))
            # Forward
            x = self(x)
            # Rear hook
            x = self.rear_hook(x)
            yield x.copy(deep=True)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : xr.DataArray
            Initial field.

        Yields
        ------
        Iterator[xr.DataArray]
            Initial field followed by hourly forecasts.
        """
        yield from self._default_generator(x)
