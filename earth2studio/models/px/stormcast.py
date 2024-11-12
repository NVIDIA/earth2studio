# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from hydra.utils import instantiate
from networks import edm_sampler
from omegaconf import OmegaConf

from earth2studio.data import DataSource, prep_data_array
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = (
    ["u10m", "v10m", "t2m", "mslp"]
    + [
        var + str(level)
        for var, level in product(
            ["u_hl", "v_hl", "t_hl", "q_hl", "z_hl", "p_hl"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30],
        )
        if not ((var == "p_hl") and (level > 20))
    ]
    + [
        "refc",
    ]
)

CONDITIONING_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "mslp", "sp"] + [
    var + str(level)
    for var, level in product(["u", "v", "z", "t", "q"], [1000, 850, 500, 250])
]

INVARIANTS = ["orography", "land_sea_mask"]


class StormCast(torch.nn.Module, AutoModelMixin):
    """


    Note
    ----


    Parameters
    ----------
    residual_model : torch.nn.Module
        Core pytorch model
    regression_model : torch.nn.Module
        Core pytorch model
    in_center : torch.Tensor
        Model input center normalization tensor
    in_scale : torch.Tensor
        Model input scale normalization tensor
    out_center : torch.Tensor
        Model output center normalization tensor
    out_scale : torch.Tensor
        Model output scale normalization tensor
    """

    def __init__(
        self,
        regression_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        lat: np.array,
        lon: np.array,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        variables: np.array = np.array(VARIABLES),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.array = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | None = None,
        sampler_args: dict[str, float | int] = {},
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.lat = lat
        self.lon = lon
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buff("invariants", invariants)

        self.variables = variables

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source is provided, "
                + "conditioning data is expected to be passed."
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
                "variable": np.array(VARIABLES),
                "lat": self.lat,
                "lon": self.lon,
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
                "variable": np.array(VARIABLES),
                "lat": self.lat,
                "lon": self.lon,
            }
        )

        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        return output_coords

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load StormCast model."""
        OmegaConf.register_new_resolver("eval", eval)
        """Load diagnostic from package"""
        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        # For regression, can we remove EasyRegressionV2?
        regression = instantiate(config.regression_model).requires_grad_(False)
        diffusion = instantiate(config.diffusion_model).requires_grad_(False)

        regression.load_state_dict(
            torch.load(package.resolve("regression.pt")), strict=False
        )

        diffusion.load_state_dict(
            torch.load(package.resolve("diffusion.pt")), strict=False
        )

        sampler_args = config.sampler_args

        # Load metadata: means, stds, grid
        metadata = xr.open_zarr("metadata.zarr.zip")

        variables = metadata["variable"].values
        lat = metadata["lat"].values
        lon = metadata["lon"].values
        means = metadata["means"].values
        stds = metadata["stds"].values

        conditioning_variables = metadata["conditioning_variables"].values
        conditioning_means = metadata["conditioning_means"].values
        conditioning_stds = metadata["conditioning_stds"].values

        # Load invariants
        invariants = metadata["invariants"].sel(channel=config.invariants).values
        invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)
        # TODO do we need the below if using RegressionWrapperV2? Not defined
        regression.set_invariant(invariants)

        return cls(
            regression,
            diffusion,
            lat,
            lon,
            means,
            stds,
            invariants,
            variables=variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
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
        invariant_tensor = self.invariant.repeat(x.shape[0], 1, 1, 1)
        concats = torch.cat((x, conditioning, invariant_tensor), dim=1)
        noise = torch.randn([conditioning.shape[0], 1, 1, 1], device=x.device)

        out = self.regression_model(noise, condition=concats)

        # Concat for diffusion conditioning
        condition = torch.cat((x, out, invariant_tensor), dim=1)
        latents = torch.randn_like(x)

        # Run diffusion model
        edm_out = edm_sampler(
            self.diffusion_model,
            latents=latents,
            condition=condition,
            **self.sampler_args,
        )

        out += edm_out

        return out

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
        conditioning_coords: CoordSystem | None = None,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        if conditioning is None:
            if self.conditioning_data_source is None:
                raise ValueError(
                    "If no conditioning data source is provided,"
                    + "then conditioning data must be passed."
                )
            time = coords["time"]
            lead_time = coords["lead_time"]
            ds = self.conditioning_data_source(
                time, lead_time, self.conditioning_variables
            )
            conditioning, conditioning_coords = prep_data_array(ds, device=x.device)
            conditioning = self.interpolate(
                conditioning,
                torch.as_tensor(conditioning_coords["lat"], device=x.device),
                torch.as_tensor(conditioning_coords["lon"], device=x.device),
                torch.as_tensor(self.lat, device=x.device),
                torch.as_tensor(self.lon, device=x.device),
            )
            conditioning_coords["lat"] = self.lat
            conditioning_coords["lon"] = self.lon

            # Need to repeat conditioning to match x
            conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)

        # Handshake conditioning coords
        handshake_coords(conditioning_coords, coords, "lon")
        handshake_coords(conditioning_coords, coords, "lat")
        handshake_coords(conditioning_coords, coords, "lead_time")
        handshake_coords(conditioning_coords, coords, "time")

        output_coords = self.output_coords(coords)

        x, conditioning = (
            x.reshape(-1, x.shape[-3:]),
            conditioning.reshape(-1, x.shape[-3:]),
        )
        for i in range(x.shape[0]):
            x[i : i + 1] = self._forward(x[i : i + 1], conditioning[i])

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
            # Forward is identity operator

            x, coords = self.__call__(x, coords)
            # Rear hook
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    @staticmethod
    def interpolate(
        values: torch.Tensor,
        lat0: torch.Tensor,
        lon0: torch.Tensor,
        lat1: torch.Tensor,
        lon1: torch.Tensor,
    ) -> torch.Tensor:
        """Specialized form of bilinear interpolation intended for optimal use on GPU.

        In particular, the mapped values must be defined on a regular rectangular grid,
        (lat0, lon0). Both lat0 and lon0 are vectors with equal spacing.

        lat1, lon1 are assumed to be 2-dimensional meshgrids with possibly unequal spacing.

        Parameters
        ----------
        values : torch.Tensor [..., W_in, H_in]
            Input values defined over (lat0, lon0) that will be interpolated onto
            (lat1, lon1) grid.
        lat0 : torch.Tensor [W_in, ]
            Vector of input latitude coordinates, assumed to be increasing with
            equal spacing.
        lon0 : torch.Tensor [H_in, ]
            Vector of input longitude coordinates, assumed to be increasing with
            equal spacing.
        lat1 : torch.Tensor [W_out, H_out]
            Tensor of output latitude coordinates
        lon1 : torch.Tensor [W_out, H_out]
            Tensor of output longitude coordinates

        Returns
        -------
        result : torch.Tensor [..., W_out, H_out]
            Tensor of interpolated values onto lat1, lon1 grid.
        """

        # Get input grid shape and flatten
        latshape, lonshape = lat1.shape
        lat1 = lat1.flatten()
        lon1 = lon1.flatten()

        # Get indices of nearest points
        latinds = torch.searchsorted(lat0, lat1) - 1
        loninds = torch.searchsorted(lon0, lon1) - 1

        # Get original grid spacing
        dlat = lat0[1] - lat0[0]
        if dlat < 0:
            # If latitudes are not increasing then need to flip
            lat0 = torch.flip(lat0, (0,))
            values = torch.flip(values, (-2,))

        dlon = lon0[1] - lon0[0]

        # Get unit distances
        normed_lat_distance = (lat1 - lat0[latinds]) / dlat
        normed_lon_distance = (lon1 - lon0[loninds]) / dlon

        # Apply bilinear mapping
        result = (
            values[..., latinds, loninds]
            * (1 - normed_lat_distance)
            * (1 - normed_lon_distance)
        )
        result += (
            values[..., latinds, loninds + 1]
            * (1 - normed_lat_distance)
            * (normed_lon_distance)
        )
        result += (
            values[..., latinds + 1, loninds]
            * (normed_lat_distance)
            * (1 - normed_lon_distance)
        )
        result += (
            values[..., latinds + 1, loninds + 1]
            * (normed_lat_distance)
            * (normed_lon_distance)
        )
        return result.reshape(*values.shape[:-2], latshape, lonshape)

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
