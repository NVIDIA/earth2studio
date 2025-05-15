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

import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterator
from itertools import product

import numpy as np
import torch
import xarray as xr
import zarr

try:
    from omegaconf import OmegaConf
    from physicsnemo.models import Module as PhysicsNemoModule
    from physicsnemo.utils.generative import deterministic_sampler
except ImportError:
    PhysicsNemoModule = None
    OmegaConf = None
    deterministic_sampler = None

from earth2studio.data import GFS_FX, HRRR, DataSource, ForecastSource, fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

# Variables used in StormCastV1 paper
VARIABLES = (
    ["u10m", "v10m", "t2m", "mslp"]
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

CONDITIONING_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "mslp", "sp"] + [
    var + str(level)
    for var, level in product(["u", "v", "z", "t", "q"], [1000, 850, 500, 250])
]

INVARIANTS = ["lsm", "orography"]

# Extent of domain in StormCastV1 paper (HRRR Lambert projection indices)
X_START, X_END = 579, 1219
Y_START, Y_END = 273, 785


@check_extra_imports("stormcast", [OmegaConf, PhysicsNemoModule, deterministic_sampler])
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
    sampler_args : dict[str, float  |  int], optional
        Arguments to pass to the diffusion sampler, by default {}
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
        sampler_args: dict[str, float | int] = {},
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.sampler_args = sampler_args

        hrrr_lat, hrrr_lon = HRRR.grid()
        self.lat = hrrr_lat[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.lon = hrrr_lon[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]

        self.hrrr_x = np.arange(hrrr_lon_lim[0], hrrr_lon_lim[1], 1)
        self.hrrr_y = np.arange(hrrr_lat_lim[0], hrrr_lat_lim[1], 1)

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
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

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
        handshake_coords(input_coords, target_input_coords, "hrrr_x")
        handshake_coords(input_coords, target_input_coords, "hrrr_y")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/modulus/stormcast-v1-era5-hrrr@1.0.1",
            cache_options={
                "cache_storage": Package.default_cache("stormcast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_extra_imports(
        "stormcast", [OmegaConf, PhysicsNemoModule, deterministic_sampler]
    )
    def load_model(
        cls,
        package: Package,
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(),
    ) -> DiagnosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source to use for global conditioning, by default GFS_FX

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            # Likely already registered so skip
            pass

        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        regression = PhysicsNemoModule.from_checkpoint(
            package.resolve("StormCastUNet.0.0.mdlus")
        )
        diffusion = PhysicsNemoModule.from_checkpoint(
            package.resolve("EDMPrecond.0.0.mdlus")
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

        # Run diffusion model
        edm_out = deterministic_sampler(
            self.diffusion_model,
            latents=latents,
            img_lr=condition,
            **self.sampler_args,
        )

        out += edm_out

        out = out * self.stds + self.means

        return out

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCast has been called without initializing the model's conditioning_data_source"
            )

        # TODO: Eventually pull out interpolation into model and remove it from fetch
        # data potentially
        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=x.device,
            interp_to=coords | {"_lat": self.lat, "_lon": self.lon},
            interp_method="linear",
        )

        # Add a batch dim
        conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)
        conditioning_coords.update({"batch": np.empty(0)})
        conditioning_coords.move_to_end("batch", last=False)

        # Handshake conditioning coords
        # TODO: ugh the interp... have to deal with this for now, no solution
        # handshake_coords(conditioning_coords, coords, "hrrr_x")
        # handshake_coords(conditioning_coords, coords, "hrrr_y")
        handshake_coords(conditioning_coords, coords, "lead_time")
        handshake_coords(conditioning_coords, coords, "time")

        output_coords = self.output_coords(coords)

        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                for k, _ in enumerate(coords["lead_time"]):
                    x[i, j, k : k + 1] = self._forward(
                        x[i, j, k : k + 1], conditioning[i, j, k : k + 1]
                    )

        return x, output_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()
        self.output_coords(coords)
        yield x, coords

        if self.conditioning_data_source is None:
            raise ValueError(
                "A conditioning data source must be available for the iterator to function."
            )

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            # Forward
            x, coords = self.__call__(x, coords)
            # Rear hook
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
