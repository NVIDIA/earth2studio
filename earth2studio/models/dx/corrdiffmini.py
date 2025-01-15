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

from collections import OrderedDict
from functools import partial
import json
from math import ceil
from typing import Literal, Sequence

import numpy as np
import torch
import xarray as xr
from modulus.models import Module
from modulus.utils.generative import deterministic_sampler, stochastic_sampler
from modulus.utils.corrdiff import regression_step, diffusion_step

from earth2studio.utils.interp import latlon_interpolation_regular
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class CorrDiffMini(torch.nn.Module, AutoModelMixin):
    """
    CorrDiff is a Corrector Diffusion model that learns mappings between
    low- and high-resolution weather data with high fidelity. CorrDiff-Mini
    is a small version of the model, with a reference dataset, designed
    for education and use as a baseline for custom CorrDiff versions.


    Note
    ----
    This model and checkpoint are from Mardani, Morteza, et al. 2023. For more
    information see the following references:

    - https://arxiv.org/html/2309.15214v
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/corrdiff_inference_package

    Parameters
    ----------
    input_variables : Sequence[str]
        List of input variable names
    output_variables : Sequence[str]
        List of output variable names
    residual_model : torch.nn.Module
        Residual (i.e. diffusion) pytorch model
    regression_model : torch.nn.Module
        Regression pytorch model
    lat_grid : torch.Tensor
        Latitude grid of the dataset
    lon_grid : torch.Tensor
        Longitude grid of the dataset
    in_center : torch.Tensor
        Model input center normalization tensor of size [len(input_variables),1,1]
    in_scale : torch.Tensor
        Model input scale normalization tensor of size [len(input_variables),1,1]
    invariant_center : torch.Tensor
        Invariant center normalization tensor of size [len(invariants),1,1]
    invariant_scale : torch.Tensor
        Invariant scale normalization tensor of size [len(invariants),1,1]
    out_center : torch.Tensor
        Model output center normalization tensor of size [len(output_variables),1,1]
    out_scale : torch.Tensor
        Model output scale normalization tensor of size [len(output_variables),1,1]
    center_latlon : tuple[float, float]
        The (latitude, longitude) centerpoint of the domain used for inferencing
    invariants : OrderedDict | None
        An OrderedDict mapping invariant names to corresponding data.
        If None (default), no invariants are used.
    number_of_samples : int, optional
        Number of high resolution samples to draw from diffusion model.
        Default is 1
    number_of_steps : int, optional
        Number of diffusion steps during sampling algorithm.
        Default is 8
    solver : Literal['euler', 'heun']
        Discretization of diffusion process. Only 'euler' and 'heun'
        are supported. Default is 'euler'
    sampler_type : Literal["deterministic", "stochastic"]
        The sampler type, either "stochastic" (default) or "deterministic"
    img_shape : tuple[int, int]
        Output image shape, by default (64, 64)
    inference_mode: Literal["regression", "diffusion", "both"]
        Inference mode.
        If "both" (default), both regression and diffusion will be evaluated.
        If "regression", only regression will be performed.
        If "diffusion", only diffusion will be performed.
    seed : int | None
        Random seed used for CorrDiff generation. If None (default), the seed
        is chosen randomly.
    hr_mean_conditioning : bool
        Whether regression model output as used as input to the diffusion model.
        This should match the corresponding setting used in CorrDiff training.
        Default True.
    """

    def __init__(
        self,
        input_variables: Sequence[str],
        output_variables: Sequence[str],        
        residual_model: torch.nn.Module,
        regression_model: torch.nn.Module,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        invariant_center: torch.Tensor,
        invariant_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        center_latlon: tuple[float, float],
        invariants: OrderedDict[str, torch.Tensor] | None = None,
        number_of_samples: int = 1,
        number_of_steps: int = 8,
        solver: Literal["euler", "heun"] = "euler",
        sampler_type: Literal["deterministic", "stochastic"] = "stochastic",
        img_shape: tuple[int, int] = (64, 64),
        inference_mode: Literal["regression", "diffusion", "both"] = "both",
        seed: int | None = None,
        hr_mean_conditioning: bool = True
    ):
        super().__init__()
        self.img_shape = img_shape        
        self.residual_model = residual_model
        self.regression_model = regression_model
        self.invariant_variables = list(invariants.keys())
        self.register_buffer("lat_grid", lat_grid)
        self.register_buffer("lon_grid", lon_grid)
        self.register_buffer("invariants", torch.stack(list(invariants.values()), dim=0))
        num_inputs = len(input_variables) + len(invariants)
        if invariants:
            in_center = torch.concat([in_center, invariant_center], dim=0)
            in_scale = torch.concat([in_scale, invariant_scale], dim=0)
        self.register_buffer("in_center", in_center.view(1, num_inputs, 1, 1))
        self.register_buffer("in_scale", in_scale.view(1, num_inputs, 1, 1))
        num_outputs = len(output_variables)
        self.register_buffer("out_center", out_center.view(1, num_outputs, 1, 1))
        self.register_buffer("out_scale", out_scale.view(1, num_outputs, 1, 1))

        if not isinstance(number_of_samples, int) and (number_of_samples > 1):
            raise ValueError("`number_of_samples` must be a positive integer.")
        if not isinstance(number_of_steps, int) and (number_of_steps > 1):
            raise ValueError("`number_of_steps` must be a positive integer.")
        if solver not in ["heun", "euler"]:
            raise ValueError(f"{solver} is not supported, solver must be in ['heun', 'euler']")
        if inference_mode not in ["regression", "diffusion", "both"]:
            raise ValueError(f"{inference_mode} is not supported, inference_mode must be in ['regression', 'diffusion', 'both']")

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.number_of_samples = number_of_samples
        self.number_of_steps = number_of_steps
        self.solver = solver
        self.inference_mode = inference_mode
        self.seed = seed
        self.hr_mean_conditioning = hr_mean_conditioning
        self.bounds = self._find_bounds(*center_latlon)
        self._set_patch()
        self.sampler = self._setup_sampler(sampler_type)

    def _find_bounds(self, lat, lon):
        nearest = ((lat - self.lat_grid)**2 + (lon - self.lon_grid)**2).argmin()
        (im, jm) = np.unravel_index(nearest, self.lon_grid.shape)
        i0 = max(0, im - self.img_shape[0] // 2)
        j0 = max(0, jm - self.img_shape[1] // 2)
        i0 = min(i0, self.lon_grid.shape[0] - self.img_shape[0])
        j0 = min(j0, self.lon_grid.shape[1] - self.img_shape[1])
        i1 = i0 + self.img_shape[0]
        j1 = j0 + self.img_shape[1]
        return ((i0, i1), (j0, j1))

    def _set_patch(self, bounds=None):
        if bounds is None:
            bounds = self.bounds
        ((i0, i1), (j0, j1)) = bounds
        self.register_buffer("patch_lat", self.lat_grid[i0:i1, j0:j1])
        self.register_buffer("patch_lon", self.lon_grid[i0:i1, j0:j1])
        self.register_buffer("patch_invariants", self.invariants[:, i0:i1, j0:j1])

    def _setup_sampler(self, sampler_type):
        if sampler_type == "deterministic":
            if self.hr_mean_conditioning:
                raise NotImplementedError(
                    "High-res mean conditioning is not yet implemented for the deterministic sampler"
                )
            sampler_fn = partial(
                deterministic_sampler,
                num_steps=self.number_of_steps,
                solver=self.solver,
            )
        elif sampler_type == "stochastic":
            sampler_fn = partial(
                stochastic_sampler,
                img_shape=self.img_shape[1],
                patch_shape=self.img_shape[1],
                num_steps=self.number_of_steps,
                boundary_pix=2,
                overlap_pix=0,
            )
        return sampler_fn

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""

        # Find lat-lon box surrounding patch coordinates
        # This is currently designed for 0.25 deg lat-lon grid inputs
        latlon_res = 0.25
        lat0 = float(self.patch_lat.min())
        lat0 -= lat0 % latlon_res
        lat1 = float(self.patch_lat.max())
        lat1 = lat1 - lat1 % latlon_res + latlon_res
        lon0 = float(self.patch_lon.min())
        lon0 -= lon0 % latlon_res
        lon1 = float(self.patch_lon.max())
        lon1 = lon1 - lon1 % latlon_res + latlon_res
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": np.arange(lat0, lat1+0.01, 0.25),
                "lon": np.arange(lon0, lon1+0.01, 0.25),
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
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(self.output_variables),
                "ilat": np.arange(*self.bounds[0]),
                "ilon": np.arange(*self.bounds[1]),
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
    def load_default_package(cls) -> Package:
        """Default pre-trained corrdiff model package from Nvidia model registry"""
        pass
        # TODO: update this
        return Package(
            "ngc://models/nvidia/modulus/corrdiff_inference_package@1",
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_taiwan"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(cls, package: Package, center_latlon: tuple[float, float]) -> DiagnosticModel:
        """Load diagnostic from package"""
        residual = Module.from_checkpoint(package.resolve("diffusion_mini.mdlus")).eval()
        regression = Module.from_checkpoint(package.resolve("regression_mini.mdlus")).eval()

        with open(package.resolve("metadata.json"), 'r') as f:
            metadata = json.load(f)
        input_variables = metadata["input_variables"]
        output_variables = metadata["output_variables"]

        with xr.open_dataset(package.resolve("grid_data.nc")) as ds:
            lat_grid = torch.Tensor(np.array(ds["latitude"]))
            lon_grid = torch.Tensor(np.array(ds["longitude"]))
            invariants = OrderedDict(
                elev_mean=torch.Tensor(np.array(ds["elev_mean"])),
                lsm_mean=torch.Tensor(np.array(ds["lsm_mean"]))
            )

        with open(package.resolve("stats.json"), 'r') as f:
            stats = json.load(f)
        in_center = torch.Tensor([stats["input"][v]["mean"] for v in input_variables])
        in_scale = torch.Tensor([stats["input"][v]["std"] for v in input_variables])
        invariant_center = torch.Tensor([stats["invariant"][v]["mean"] for v in invariants])
        invariant_scale = torch.Tensor([stats["invariant"][v]["std"] for v in invariants])
        out_center = torch.Tensor([stats["output"][v]["mean"] for v in output_variables])
        out_scale = torch.Tensor([stats["output"][v]["std"] for v in output_variables])

        return cls(
            input_variables=input_variables,
            output_variables=output_variables,
            residual_model=residual,
            regression_model=regression,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            invariants=invariants,
            in_center=in_center,
            in_scale=in_scale,
            invariant_center=invariant_center,            
            invariant_scale=invariant_scale,
            out_center=out_center,
            out_scale=out_scale,
            center_latlon=center_latlon
        )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate from input lat/lon (self.lat, self.lon) onto output lat/lon
        (self.lat_grid, self.lon_grid) using bilinear interpolation."""
        input_coords = self.input_coords()
        return latlon_interpolation_regular(
            x,
            torch.as_tensor(input_coords["lat"], device=x.device, dtype=torch.float32),
            torch.as_tensor(input_coords["lon"], device=x.device, dtype=torch.float32),
            self.patch_lat,
            self.patch_lon,
        )

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.solver not in ["euler", "heun"]:
            raise ValueError(
                f"solver must be either 'euler' or 'heun' but got {self.solver}"
            )

        # Interpolate
        x = self._interpolate(x)

        # Add batch dimension
        (C, H, W) = x.shape
        x = x.view(1, C, H, W)
        
        # Concatenate invariants
        inv_patch = self.patch_invariants.unsqueeze(0)      
        x = torch.concat([x, inv_patch], dim=1)

        # Normalize
        image_lr = (x - self.in_center) / self.in_scale
        image_lr = (
            image_lr.to(torch.float32)
            .to(memory_format=torch.channels_last)
        )

        # Run regression model
        if self.regression_model:
            latents_shape=(1, len(self.output_variables), *self.img_shape)
            x_hat = torch.zeros(latents_shape, dtype=x.dtype, device=x.device)
            t_hat = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            image_reg = self.regression_model(x_hat, image_lr, t_hat)

        # Run ith sample of diffusion
        def generate(i):
            seed = self.seed if self.seed is not None else np.random.randint(2**32)
            if self.residual_model:
                mean_hr = image_reg[:1] if self.hr_mean_conditioning else None
                image_res = diffusion_step(
                    net=self.residual_model,
                    sampler_fn=self.sampler,
                    seed_batch_size=1,
                    img_shape=self.img_shape,
                    img_out_channels=len(self.output_variables),
                    rank_batches=[[seed+i]], #[[self.seed]],
                    img_lr=image_lr,
                    rank=1, # rank=1 only disables progress bar
                    device=x.device,
                    hr_mean=mean_hr,
                )

            if self.inference_mode == "regression":
                image_out = image_reg
            elif self.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res
            return image_out

        image_out = torch.concat([generate(i) for i in range(self.number_of_samples)], dim=0)
        image_out = image_out * self.out_scale + self.out_center

        return image_out


    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )        
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])

        return out, output_coords
