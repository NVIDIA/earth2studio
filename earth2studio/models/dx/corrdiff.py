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
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr
import zarr

try:
    from physicsnemo.models import Module as PhysicsNemoModule
    from physicsnemo.utils.corrdiff import (
        diffusion_step,
        regression_step,
    )
    from physicsnemo.utils.generative import (
        StackedRandomGenerator,
        deterministic_sampler,
        stochastic_sampler,
    )
except ImportError:
    PhysicsNemoModule = None
    StackedRandomGenerator = None
    deterministic_sampler = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
    interp,
)
from earth2studio.utils.type import CoordSystem

# Input variables for the model
INPUT_VARIABLES = ["..."]

# Output variables for the model
OUTPUT_VARIABLES = ["..."]


@check_extra_imports(
    "corrdiff", [PhysicsNemoModule, StackedRandomGenerator, deterministic_sampler]
)
class CorrDiff(torch.nn.Module, AutoModelMixin):
    """CorrDiff is a Corrector Diffusion model that learns mappings between
    low- and high-resolution weather data with high fidelity. This model combines
    regression and diffusion steps to generate high-resolution predictions.

    Note
    ----
    For more information on the model architecture and training, please refer to:

    - https://arxiv.org/html/2309.15214v
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/corrdiff_inference_package

    Parameters
    ----------
    input_variables : Sequence[str]
        List of input variable names
    output_variables : Sequence[str]
        List of output variable names
    residual_model : torch.nn.Module
        Core pytorch model for diffusion step
    regression_model : torch.nn.Module
        Core pytorch model for regression step
    lat_grid : torch.Tensor
        Output latitude grid tensor
    lon_grid : torch.Tensor
        Output longitude grid tensor
    in_center : torch.Tensor
        Model input center normalization tensor
    in_scale : torch.Tensor
        Model input scale normalization tensor
    invariant_center : torch.Tensor
        Invariant features center normalization tensor
    invariant_scale : torch.Tensor
        Invariant features scale normalization tensor
    out_center : torch.Tensor
        Model output center normalization tensor
    out_scale : torch.Tensor
        Model output scale normalization tensor
    invariants : OrderedDict | None, optional
        Dictionary of invariant features, by default None
    number_of_samples : int, optional
        Number of high resolution samples to draw from diffusion model, by default 1
    number_of_steps : int, optional
        Number of langevin diffusion steps during sampling algorithm, by default 18
    solver : Literal["euler", "heun"], optional
        Discretization of diffusion process, by default "euler"
    sampler_type : Literal["deterministic", "stochastic"], optional
        Type of sampler to use, by default "stochastic"
    img_shape : Tuple[int, int], optional
        Shape of the output image, by default (128, 128)
    inference_mode : Literal["regression", "diffusion", "both"], optional
        Which inference mode to use, by default "both"
    hr_mean_conditioning : bool, optional
        Whether to use high-res mean conditioning, by default True
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
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
        invariants: OrderedDict | None = None,
        number_of_samples: int = 1,
        number_of_steps: int = 18,
        solver: Literal["euler", "heun"] = "euler",
        sampler_type: Literal["deterministic", "stochastic"] = "stochastic",
        img_shape: tuple[int, int] = (128, 128),
        inference_mode: Literal["regression", "diffusion", "both"] = "both",
        hr_mean_conditioning: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        # Validate parameters
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("`number_of_samples` must be a positive integer.")
        if not isinstance(number_of_steps, int) or number_of_steps < 1:
            raise ValueError("`number_of_steps` must be a positive integer.")
        if solver not in ["heun", "euler"]:
            raise ValueError(f"{solver} is not supported, must be in ['heun', 'euler']")

        # Store model configuration
        self.img_shape = img_shape
        self.number_of_samples = number_of_samples
        self.number_of_steps = number_of_steps
        self.solver = solver
        self.inference_mode = inference_mode
        self.hr_mean_conditioning = hr_mean_conditioning
        self.seed = seed

        # Store models
        self.residual_model = residual_model
        self.regression_model = regression_model

        # Store variable names
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Register buffers for model parameters
        self._register_buffers(
            lat_grid,
            lon_grid,
            in_center,
            in_scale,
            invariant_center,
            invariant_scale,
            out_center,
            out_scale,
            invariants,
        )

        # Set up sampler
        self.sampler = self._setup_sampler(sampler_type)

    def _register_buffers(
        self,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        invariant_center: torch.Tensor,
        invariant_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        invariants: OrderedDict | None,
    ) -> None:
        """Register model buffers and handle invariants."""
        # Register grid coordinates
        self.register_buffer("lat_grid", lat_grid)
        self.register_buffer("lon_grid", lon_grid)

        # Handle invariants
        if invariants:
            self.invariant_variables = list(invariants.keys())
            self.register_buffer(
                "invariants", torch.stack(list(invariants.values()), dim=0)
            )
            # Combine input normalization with invariants
            in_center = torch.concat([in_center, invariant_center], dim=0)
            in_scale = torch.concat([in_scale, invariant_scale], dim=0)
        else:
            self.invariants = None
            self.invariant_variables = []

        # Register normalization parameters
        num_inputs = len(self.input_variables) + len(self.invariant_variables)
        self.register_buffer("in_center", in_center.view(1, num_inputs, 1, 1))
        self.register_buffer("in_scale", in_scale.view(1, num_inputs, 1, 1))
        self.register_buffer(
            "out_center", out_center.view(1, len(self.output_variables), 1, 1)
        )
        self.register_buffer(
            "out_scale", out_scale.view(1, len(self.output_variables), 1, 1)
        )

    def _setup_sampler(
        self, sampler_type: Literal["deterministic", "stochastic"]
    ) -> Callable:
        """Set up the appropriate sampler based on the type."""
        if sampler_type == "deterministic":
            if self.hr_mean_conditioning:
                raise NotImplementedError(
                    "High-res mean conditioning is not yet implemented for the deterministic sampler"
                )
            return partial(
                deterministic_sampler,
                num_steps=self.number_of_steps,
                solver=self.solver,
            )
        elif sampler_type == "stochastic":
            return partial(
                stochastic_sampler,
                num_steps=self.number_of_steps,
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    def input_coords(self) -> CoordSystem:
        """Get the input coordinate system for the model.

        Returns
        -------
        CoordSystem
            Dictionary containing the input coordinate system
        """
        # Calculate lat-lon box surrounding patch coordinates
        latlon_res = 0.25
        lat_grid_cpu = self.lat_grid.cpu()
        lon_grid_cpu = self.lon_grid.cpu()
        lat0 = np.floor(lat_grid_cpu.min() / latlon_res) * latlon_res
        lat1 = np.ceil(lat_grid_cpu.max() / latlon_res) * latlon_res
        lon0 = np.floor(lon_grid_cpu.min() / latlon_res) * latlon_res
        lon1 = np.ceil(lon_grid_cpu.max() / latlon_res) * latlon_res
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": np.arange(lat0, lat1 + 0.01, 0.25),
                "lon": np.arange(lon0, lon1 + 0.01, 0.25),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Get the output coordinate system for the model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Dictionary containing the output coordinate system
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(self.output_variables),
                "lat": self.lat_grid.cpu().numpy(),
                "lon": self.lon_grid.cpu().numpy(),
            }
        )

        # Validate input coordinates
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained corrdiff model package from Nvidia model registry"""
        raise NotImplementedError

    @classmethod
    @check_extra_imports(
        "corrdiff", [PhysicsNemoModule, StackedRandomGenerator, deterministic_sampler]
    )
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load CorrDiff model from package.

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration

        Returns
        -------
        DiagnosticModel
            Initialized CorrDiff model

        Raises
        ------
        ImportError
            If required dependencies are not installed
        """
        if StackedRandomGenerator is None or deterministic_sampler is None:
            raise ImportError(
                "Additional CorrDiff model dependencies are not installed. See install documentation for details."
            )

        # Load model checkpoints
        residual = PhysicsNemoModule.from_checkpoint(
            package.resolve("diffusion.mdlus")
        ).eval()
        regression = PhysicsNemoModule.from_checkpoint(
            package.resolve("regression.mdlus")
        ).eval()

        # Load normalization statistics
        with open(package.resolve("stats.json")) as f:
            stats = json.load(f)

        # Load input normalization parameters
        in_center = torch.Tensor([stats["input"][v]["mean"] for v in INPUT_VARIABLES])
        in_scale = torch.Tensor([stats["input"][v]["std"] for v in INPUT_VARIABLES])

        # Load output normalization parameters
        out_center = torch.Tensor(
            [stats["output"][v]["mean"] for v in OUTPUT_VARIABLES]
        )
        out_scale = torch.Tensor([stats["output"][v]["std"] for v in OUTPUT_VARIABLES])

        # Load lat/lon grid
        with xr.open_dataset(package.resolve("latlon_grid.nc")) as ds:
            lat_grid = torch.Tensor(np.array(ds["lat"][:]))
            lon_grid = torch.Tensor(np.array(ds["lon"][:]))

        # Load invariants if available
        try:
            with xr.open_dataset(package.resolve("invariants.nc")) as ds:
                invariants = OrderedDict(
                    (var_name, torch.Tensor(np.array(ds[var_name])))
                    for var_name in ds.data_vars
                )

                # Load invariant normalization parameters
                invariant_center = torch.Tensor(
                    [stats["invariants"][v]["mean"] for v in invariants]
                )
                invariant_scale = torch.Tensor(
                    [stats["invariants"][v]["std"] for v in invariants]
                )
        except FileNotFoundError:
            invariants = None
            invariant_center = None
            invariant_scale = None

        return cls(
            input_variables=INPUT_VARIABLES,
            output_variables=OUTPUT_VARIABLES,
            residual_model=residual,
            regression_model=regression,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            in_center=in_center,
            in_scale=in_scale,
            invariant_center=invariant_center,
            invariant_scale=invariant_scale,
            out_center=out_center,
            out_scale=out_scale,
            invariants=invariants,
            img_shape=lat_grid.shape[-2:],
        )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate from input lat/lon onto output lat/lon grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to interpolate

        Returns
        -------
        torch.Tensor
            Interpolated tensor
        """
        input_coords = self.input_coords()
        return interp.latlon_interpolation_regular(
            x,
            torch.as_tensor(input_coords["lat"], device=x.device, dtype=torch.float32),
            torch.as_tensor(input_coords["lon"], device=x.device, dtype=torch.float32),
            self.lat_grid,
            self.lon_grid,
        )

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.solver not in ["euler", "heun"]:
            raise ValueError(
                f"solver must be either 'euler' or 'heun' but got {self.solver}"
            )

        # Interpolate input to output grid
        x = self._interpolate(x)

        # Add batch dimension
        (C, H, W) = x.shape
        x = x.view(1, C, H, W)

        # Concatenate invariants if available
        if self.invariants is not None:
            x = torch.concat([x, self.invariants.unsqueeze(0)], dim=1)

        # Normalize input
        image_lr = (x - self.in_center) / self.in_scale
        image_lr = image_lr.to(torch.float32).to(memory_format=torch.channels_last)

        # Run regression model
        if self.regression_model:
            latents_shape = (1, len(self.output_variables), *self.img_shape)
            image_reg = regression_step(
                net=self.regression_model,
                img_lr=image_lr,
                latents_shape=latents_shape,
            )

        # Generate samples
        def generate(i: int) -> torch.Tensor:
            """Generate a single sample.

            Parameters
            ----------
            i : int
                Sample index

            Returns
            -------
            torch.Tensor
                Generated sample
            """
            seed = self.seed if self.seed is not None else np.random.randint(2**32)

            if self.residual_model and self.inference_mode != "regression":
                mean_hr = image_reg[:1] if self.hr_mean_conditioning else None
                image_res = diffusion_step(
                    net=self.residual_model,
                    sampler_fn=self.sampler,
                    img_shape=self.img_shape,
                    img_out_channels=len(self.output_variables),
                    rank_batches=[[seed + i]],
                    img_lr=image_lr,
                    rank=1,
                    device=x.device,
                    mean_hr=mean_hr,
                )

            if self.inference_mode == "regression":
                return image_reg
            elif self.inference_mode == "diffusion":
                return image_res
            else:
                return image_reg + image_res

        # Generate all samples
        image_out = torch.concat(
            [generate(i) for i in range(self.number_of_samples)], dim=0
        )

        # Denormalize output
        image_out = image_out * self.out_scale + self.out_center

        return image_out

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Execute the model on input data.

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
        """
        output_coords = self.output_coords(coords)

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])

        return out, output_coords


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


@check_extra_imports(
    "corrdiff", [PhysicsNemoModule, StackedRandomGenerator, deterministic_sampler]
)
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

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(19.25, 28, 36, endpoint=True),
                "lon": np.linspace(116, 126, 40, endpoint=False),
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
                "variable": np.array(OUT_VARIABLES),
                "lat": self.out_lat.cpu().numpy(),
                "lon": self.out_lon.cpu().numpy(),
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
        return Package(
            "ngc://models/nvidia/modulus/corrdiff_inference_package@1",
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_taiwan"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports(
        "corrdiff", [PhysicsNemoModule, StackedRandomGenerator, deterministic_sampler]
    )
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic from package"""

        if StackedRandomGenerator is None or deterministic_sampler is None:
            raise ImportError(
                "Additional CorrDiff model dependencies are not installed. See install documentation for details."
            )

        checkpoint_zip = Path(package.resolve("corrdiff_inference_package.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_zip.parent)

        residual = PhysicsNemoModule.from_checkpoint(
            str(
                checkpoint_zip.parent
                / Path("corrdiff_inference_package/checkpoints/diffusion.mdlus")
            )
        ).eval()

        regression = PhysicsNemoModule.from_checkpoint(
            str(
                checkpoint_zip.parent
                / Path("corrdiff_inference_package/checkpoints/regression.mdlus")
            )
        ).eval()

        # Get dataset for lat/lon grid info and centers/stds'
        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            # Fallback to older method if version check fails
            zarr_major_version = 2  # Assume older version if we can't determine

        if zarr_major_version >= 3:
            store = zarr.storage.LocalStore(
                str(
                    checkpoint_zip.parent
                    / Path(
                        "corrdiff_inference_package/dataset/2023-01-24-cwb-4years_5times.zarr"
                    )
                )
            )
        else:
            store = zarr.storage.DirectoryStore(
                str(
                    checkpoint_zip.parent
                    / Path(
                        "corrdiff_inference_package/dataset/2023-01-24-cwb-4years_5times.zarr"
                    )
                )
            )
        root = zarr.group(store)
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
        input_coords = self.input_coords()
        return interp.latlon_interpolation_regular(
            x,
            torch.as_tensor(input_coords["lat"], device=x.device, dtype=torch.float32),
            torch.as_tensor(input_coords["lon"], device=x.device, dtype=torch.float32),
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

        coord = self.output_coords(self.input_coords())
        img_resolution_x = coord["lat"].shape[0]
        img_resolution_y = coord["lon"].shape[1]
        latents = rnd.randn(
            [
                self.number_of_samples,
                self.regression_model.img_out_channels,
                img_resolution_x,
                img_resolution_y,
            ],
            device=x.device,
        )

        mean = self.unet_regression(
            self.regression_model,
            torch.zeros_like(latents),
            x,
            num_steps=self.number_of_steps,
        )
        res = deterministic_sampler(
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
        output_coords = self.output_coords(coords)

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
        t_hat = torch.tensor(1.0).to(torch.float64).to(latents.device)

        x_next = net(x_hat, x_lr, t_hat, class_labels).to(torch.float64)

        return x_next
