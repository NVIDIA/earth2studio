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

import copy
from collections import OrderedDict
from typing import Callable, Literal

import numpy as np
import torch
import zarr
from modulus.models import Module
from modulus.utils.generative import StackedRandomGenerator, ablation_sampler

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "tcwv",
    "z500",
    "t500",
    "u500",
    "v500",
    "z850",
    "t850",
    "u850",
    "v850",
    "t2m",
    "u10m",
    "v10m",
]

OUT_VARIABLES = ["mrr", "t2m", "u10m", "v10m"]


class CorrDiffTaiwan(torch.nn.Module, AutoModelMixin):
    """

    CorrDiff is a Corrector Diffusion model that learns mappings between
    low- and high-resolution weather data with high fidelity. This particular
    model was trained over a particular region near Taiwan.


    Note
    ----
    This model and checkpoint are from Mardani, Morteza, et al. 2023. For more
    information see the following references:

    - https://arxiv.org/html/2309.15214v
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/corrdiff_inference_package

    Parameters
    ----------
    residual_model : torch.nn.Module
        Core pytorch model
    regression_model : torch.nn.Module
        Core pytorch model
    in_center : torch.Tensor
        Model input center normalization tensor of size [20,1,1]
    in_scale : torch.Tensor
        Model input scale normalization tensor of size [20,1,1]
    out_center : torch.Tensor
        Model output center normalization tensor of size [4,1,1]
    out_scale : torch.Tensor
        Model output scale normalization tensor of size [4,1,1]
    out_lat : torch.Tensor
        Output latitude grid of size [448, 448]
    out_lon : torch.Tensor
        Output longitude grid of size [448, 448]
    number_of_samples : int, optional
        Number of high resolution samples to draw from diffusion model.
        Default is 1
    number_of_steps: int, optional
        Number of langevin diffusion steps during sampling algorithm.
        Default is 8
    solver: Literal['euler', 'heun']
        Discretization of diffusion process. Only 'euler' and 'heun'
        are supported. Default is 'euler'
    """

    def __init__(
        self,
        residual_model: torch.nn.Module,
        regression_model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        out_lat: torch.Tensor,
        out_lon: torch.Tensor,
        number_of_samples: int = 1,
        number_of_steps: int = 8,
        solver: Literal["euler", "heun"] = "euler",
    ):
        super().__init__()
        self.residual_model = residual_model
        self.regression_model = regression_model
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)
        self.register_buffer("out_lat_full", out_lat)
        self.register_buffer("out_lon_full", out_lon)
        self.register_buffer("out_lat", out_lat[1:-1, 1:-1])
        self.register_buffer("out_lon", out_lon[1:-1, 1:-1])

        if not isinstance(number_of_samples, int) and (number_of_samples > 1):
            raise ValueError("`number_of_samples` must be a positive integer.")
        if not isinstance(number_of_steps, int) and (number_of_steps > 1):
            raise ValueError("`number_of_steps` must be a positive integer.")
        if solver not in ["heun", "euler"]:
            raise ValueError(f"{solver} is not supported, must be in ['huen', 'euler']")

        self.number_of_samples = number_of_samples
        self.number_of_steps = number_of_steps
        self.solver = solver

    @property
    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(1),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(19.25, 28, 36, endpoint=True),
                "lon": np.linspace(116, 126, 40, endpoint=False),
            }
        )

    @property
    def output_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(1),
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(OUT_VARIABLES),
                "ilat": np.arange(448),
                "ilon": np.arange(448),
            }
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained corrdiff model package from Nvidia model registry"""
        return Package("ngc://models/nvidia/modulus/corrdiff_inference_package@1")

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""
        cached_path = package.get("corrdiff_inference_package.zip")
        residual = Module.from_checkpoint(
            cached_path + "/corrdiff_inference_package/checkpoints/diffusion.mdlus"
        ).eval()

        regression = Module.from_checkpoint(
            cached_path + "/corrdiff_inference_package/checkpoints/regression.mdlus"
        ).eval()

        # Get dataset for lat/lon grid info and centers/stds
        store = zarr.DirectoryStore(
            cached_path
            + "/corrdiff_inference_package/dataset/2023-01-24-cwb-4years_5times.zarr"
        )
        with zarr.group(store) as root:
            # Get output lat/lon grid
            out_lat = torch.as_tensor(root["XLAT"][:], dtype=torch.float32)
            out_lon = torch.as_tensor(root["XLONG"][:], dtype=torch.float32)

            # get normalization info
            in_inds = [0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19]
            in_center = (
                torch.as_tensor(
                    root["era5_center"][in_inds],
                    dtype=torch.float32,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )

            in_scale = (
                torch.as_tensor(
                    root["era5_scale"][in_inds],
                    dtype=torch.float32,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )

            out_inds = [0, 17, 18, 19]
            out_center = (
                torch.as_tensor(
                    root["cwb_center"][out_inds],
                    dtype=torch.float32,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )

            out_scale = (
                torch.as_tensor(
                    root["cwb_scale"][out_inds],
                    dtype=torch.float32,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )

        return cls(
            residual,
            regression,
            in_center,
            in_scale,
            out_center,
            out_scale,
            out_lat,
            out_lon,
        )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate from input lat/lon (self.lat, self.lon) onto output lat/lon
        (self.lat_grid, self.lon_grid) using bilinear interpolation."""
        return self.interpolate(
            x,
            torch.as_tensor(
                self.input_coords["lat"], device=x.device, dtype=torch.float32
            ),
            torch.as_tensor(
                self.input_coords["lon"], device=x.device, dtype=torch.float32
            ),
            self.out_lat_full,
            self.out_lon_full,
        )[..., 1:-1, 1:-1]

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.solver not in ["euler", "heun"]:
            raise ValueError(
                f"solver must be either 'euler' or 'heun' but got {self.solver}"
            )

        # Interpolate
        x = self._interpolate(x)

        # Add sample dimension
        x = x.unsqueeze(0)
        x = (x - self.in_center) / self.in_scale

        # Create grid channels
        x1 = np.sin(np.linspace(0, 2 * np.pi, 448))
        x2 = np.cos(np.linspace(0, 2 * np.pi, 448))
        y1 = np.sin(np.linspace(0, 2 * np.pi, 448))
        y2 = np.cos(np.linspace(0, 2 * np.pi, 448))
        grid_x1, grid_y1 = np.meshgrid(y1, x1)
        grid_x2, grid_y2 = np.meshgrid(y2, x2)
        grid = torch.as_tensor(
            np.expand_dims(
                np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0), axis=0
            ),
            dtype=torch.float32,
            device=x.device,
        )

        # Concat Grids
        x = torch.cat((x, grid), dim=1)

        # Repeat for sample size
        sample_seeds = torch.arange(self.number_of_samples)
        x = x.repeat(self.number_of_samples, 1, 1, 1)

        # Create latents
        rnd = StackedRandomGenerator(x.device, sample_seeds)
        latents = rnd.randn(
            [
                self.number_of_samples,
                self.regression_model.img_out_channels,
                self.regression_model.img_resolution,
                self.regression_model.img_resolution,
            ],
            device=x.device,
        )

        mean = self.unet_regression(
            self.regression_model,
            torch.zeros_like(latents),
            x,
            num_steps=self.number_of_steps,
        )
        res = ablation_sampler(
            self.residual_model,
            latents,
            x,
            randn_like=rnd.randn_like,
            num_steps=self.number_of_steps,
            solver=self.solver,
        )
        x = mean + res
        x = self.out_scale * x + self.out_center
        return x

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        handshake_dim(coords, "lon", 3)
        handshake_dim(coords, "lat", 2)
        handshake_dim(coords, "variable", 1)
        handshake_coords(coords, self.input_coords, "lon")
        handshake_coords(coords, self.input_coords, "lat")
        handshake_coords(coords, self.input_coords, "variable")

        output_coords = copy.deepcopy(self.output_coords)

        output_coords["batch"] = coords["batch"]

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])

        return out, output_coords

    @staticmethod
    def unet_regression(
        net: torch.nn.Module,
        latents: torch.Tensor,
        img_lr: torch.Tensor,
        class_labels: torch.Tensor = None,
        randn_like: Callable = torch.randn_like,
        num_steps: int = 8,
        sigma_min: float = 0.0,
        sigma_max: float = 0.0,
        rho: int = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform U-Net regression with temporal sampling.

        Parameters
        ----------
        net : torch.nn.Module
            U-Net model for regression.
        latents : torch.Tensor
            Latent representation.
        img_lr : torch.Tensor)
            Low-resolution input image.
        class_labels : torch.Tensor, optional
            Class labels for conditional generation.
        randn_like : function, optional
            Function for generating random noise.
        num_steps : int, optional
            Number of time steps for temporal sampling.
        sigma_min : float, optional
            Minimum noise level.
        sigma_max : float, optional
            Maximum noise level.
        rho : int, optional
            Exponent for noise level interpolation.
        S_churn : float, optional
            Churning parameter.
        S_min : float, optional
            Minimum churning value.
        S_max : float, optional
            Maximum churning value.
        S_noise : float, optional
            Noise level for churning.

        Returns
        -------
        torch.Tensor: Predicted output at the next time step.
        """

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

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
            [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # conditioning
        x_lr = img_lr

        # Main sampling loop.
        x_hat = latents.to(torch.float64) * t_steps[0]
        t_hat = torch.tensor(1.0).to(torch.float64).cuda()

        x_next = net(x_hat, x_lr, t_hat, class_labels).to(torch.float64)

        return x_next

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
