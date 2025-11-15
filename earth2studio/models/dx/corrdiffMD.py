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

import zipfile
from collections import OrderedDict
from collections.abc import Callable,Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from functools import partial
import numpy as np
import torch
import zarr
import xarray as xr
from datetime import timezone
from physicsnemo.models import Module

try:
    from physicsnemo.utils.generative import (
        StackedRandomGenerator,
    )
    from physicsnemo.utils.generative import (
        deterministic_sampler as ablation_sampler,
    )
except ImportError:
    StackedRandomGenerator = None
    ablation_sampler = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    interp
)
from earth2studio.utils.interp import latlon_interpolation_regular
from earth2studio.utils.type import CoordSystem
from earth2studio.utils.coords import CoordSystem, map_coords
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem


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
    OptionalDependencyFailure("corrdiff")
    PhysicsNemoModule = None
    StackedRandomGenerator = None
    deterministic_sampler = None

import json
import matplotlib.pyplot as plt
import cftime
import nvtx
import torch
from datetime import datetime
import torch.nn.functional as F
from physicsnemo.utils.zenith_angle import cos_zenith_angle_from_timestamp, cos_zenith_angle
from earth2studio.utils.multidiffusion import MultiDiffusion

@check_optional_dependencies()
class CorrDiffMD(torch.nn.Module, AutoModelMixin):
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
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        invariants: OrderedDict | None = None,
        invariant_center: torch.Tensor | None = None,
        invariant_scale: torch.Tensor | None = None,
        number_of_samples: int = 1,
        number_of_steps: int = 18,
        inference_mode: Literal["regression", "diffusion", "both"] = "both",
        hr_mean_conditioning: bool = True,
        seed: int | None = None,
        stride: int | None = None,
    ):
        super().__init__()

        # Validate parameters
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("`number_of_samples` must be a positive integer.")
        if not isinstance(number_of_steps, int) or number_of_steps < 1:
            raise ValueError("`number_of_steps` must be a positive integer.")
        
        # Store model configuration
        self.number_of_samples = number_of_samples
        self.number_of_steps = number_of_steps
        
        self.inference_mode = inference_mode
        self.hr_mean_conditioning = hr_mean_conditioning
        self.seed = seed
        
        self.img_shape = (lat_output_grid.shape[0], lon_output_grid.shape[0])
        self.img_shape = (320,320)

        self.stride = stride
        # Store models
        self.residual_model = residual_model
        self.regression_model = regression_model

        # Store variable names
        self.input_variables = input_variables
        self.output_variables = output_variables

        

        # Register buffers for model parameters
        self._register_buffers(
            lat_input_grid,
            lon_input_grid,
            lat_output_grid,
            lon_output_grid,
            in_center,
            in_scale,
            invariant_center,
            invariant_scale,
            out_center,
            out_scale,
            invariants,
        )

        lon_input_grid,lat_input_grid = np.meshgrid(self.lon_input_numpy, self.lat_input_numpy)
        lon_output_grid,lat_output_grid = np.meshgrid(self.lon_output_numpy, self.lat_output_numpy)
        self._interpolator = interp.LatLonInterpolation(
                lat_input_grid,
                lon_input_grid,
                lat_output_grid,
                lon_output_grid,
        )

    
    def _register_buffers(
        self,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
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
        
        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)
        input_grid_n_dim = len(lat_input_grid.shape)
        

        self.lat_input_numpy = lat_input_grid.cpu().numpy()
        self.lon_input_numpy = lon_input_grid.cpu().numpy()
        self.lat_output_numpy = lat_output_grid.cpu().numpy()
        self.lon_output_numpy = lon_output_grid.cpu().numpy()

        
        # Handle invariants
        if invariants:
            self.invariant_variables = list(invariants.keys())
            self.register_buffer(
                "invariants", torch.stack(list(invariants.values()), dim=0)
            )
            
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

    def input_coords(self) -> CoordSystem:
        """Get the input coordinate system for the model.

        Returns
        -------
        CoordSystem
            Dictionary containing the input coordinate system
        """

        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": self.lat_input_numpy,
                "lon": self.lon_input_numpy,
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
                "lat": self.lat_output_numpy,
                "lon": self.lon_output_numpy,
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
    @check_optional_dependencies()
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

        # Load normalization statistics
        with open(package.resolve("metadata.json")) as f:
            metadata = json.load(f)
        input_variables = metadata["input_variables"]
        output_variables = metadata["output_variables"]
        
        # Load model parameters (if not provided, use default values)
        number_of_samples = metadata.get("number_of_samples", 1)
        number_of_steps = metadata.get("number_of_steps", 18)
        
        inference_mode = metadata.get("inference_mode", "both")
        hr_mean_conditioning = metadata.get("hr_mean_conditioning", True)
        seed = metadata.get("seed", None)
        stride = metadata.get("stride", 80)

        lat_range = metadata.get("lat_range",None)
        lon_range = metadata.get("lon_range",None)
        
        # Load normalization statistics
        with open(package.resolve("stats.json")) as f:
            stats = json.load(f)

        # Load input normalization parameters
        
        in_center = torch.Tensor([stats["input"][v]["center"] for v in input_variables])
        in_scale = torch.Tensor([stats["input"][v]["scale"] for v in input_variables])
        # Load output normalization parameters
        out_center = torch.Tensor(
            [stats["output"][v]["center"] for v in output_variables]
        )
        out_scale = torch.Tensor([stats["output"][v]["scale"] for v in output_variables])
        # Load lat/lon grid

        if lat_range is not None:
            LR_step = 0.25
            HR_step = 0.05
            print("Generating lat/lon grid")
            print(f"lat range: {lat_range}")
            print(f"lon range: {lon_range}")
            lat_low, lat_high = min(lat_range), max(lat_range)
            lon_low, lon_high = min(lon_range), max(lon_range)

            LR_lon_num = int(round((lon_high - lon_low) / LR_step)) + 1
            LR_lat_num = int(round((lat_high - lat_low) / LR_step)) + 1

            lon_input_grid = torch.linspace(lon_low, lon_high, LR_lon_num)
            lat_input_grid = torch.linspace(lat_low, lat_high, LR_lat_num)

            HR_lon_num = int(round((lon_high - lon_low) / HR_step)) + 1
            HR_lat_num = int(round((lat_high - lat_low) / HR_step)) + 1
            if HR_lon_num<320 or HR_lat_num<320:
                raise ValueError(
                    f"High-resolution grid dimensions are too small. "
                    f"Expected at least 320x320, but got {HR_lat_num}x{HR_lon_num}. "
                    f"Check lat_range and lon_range metadata."
                )
            lon_output_grid = torch.linspace(lon_low, lon_high, HR_lon_num)
            lat_output_grid = torch.linspace(lat_low, lat_high, HR_lat_num)
        else:
            with xr.open_dataset(package.resolve("output_latlon_grid.nc")) as ds:
                lat_output_grid = torch.Tensor(np.array(ds["lat"][:]))
                lon_output_grid = torch.Tensor(np.array(ds["lon"][:]))

            with xr.open_dataset(package.resolve("input_latlon_grid.nc")) as ds:
                lat_input_grid = torch.Tensor(np.array(ds["lat"][:]))
                lon_input_grid = torch.Tensor(np.array(ds["lon"][:]))
        

        # Load invariants if available
        try:
            with xr.open_dataset(package.resolve("invariants.nc")) as ds:
                if lat_range is not None:
                    ds_sliced = ds.sel(latitude=slice(lat_low, lat_high), longitude=slice(lon_low, lon_high))
                else:
                    ds_sliced = ds
                invariants = OrderedDict(
                    (var_name, torch.Tensor(np.array(ds_sliced[var_name])))
                    for var_name in ds_sliced.data_vars
                )
                # Load invariant normalization parameters
                invariant_center = torch.Tensor(
                    [stats["invariants"][v]["center"] for v in invariants]
                )
                invariant_scale = torch.Tensor(
                    [stats["invariants"][v]["scale"] for v in invariants]
                )
        except FileNotFoundError:
            invariants = None
            invariant_center = None
            invariant_scale = None
        
        # Load model checkpoints
        regression = PhysicsNemoModule.from_checkpoint(package.resolve("regression.mdlus")).eval()
        residual = None
        
        if inference_mode != "regression":
            residual = PhysicsNemoModule.from_checkpoint(
                package.resolve("diffusion.mdlus")
            ).eval()
        

        return cls(
            input_variables=input_variables,
            output_variables=output_variables,
            residual_model=residual,
            regression_model=regression,
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            in_center=in_center,
            in_scale=in_scale,
            invariants=invariants,
            invariant_center=invariant_center,
            invariant_scale=invariant_scale,
            out_center=out_center,
            out_scale=out_scale,
            number_of_samples=number_of_samples,
            number_of_steps=number_of_steps,
            inference_mode=inference_mode,
            hr_mean_conditioning=hr_mean_conditioning,
            seed=seed,
            stride=stride
        )

    @staticmethod
    def _infer_input_latlon_grid(
        lat_output_grid: torch.Tensor, lon_output_grid: torch.Tensor, latlon_res: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer the input lat/lon grid from the output lat/lon grid.

        Parameters
        ----------
        lat_output_grid : torch.Tensor
            Output latitude grid
        lon_output_grid : torch.Tensor
            Output longitude grid
        latlon_res : float
            Resolution of the input lat/lon grid

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Input latitude and longitude grids
        """
        lat0 = (torch.floor(lat_output_grid.min() / latlon_res) - 1) * latlon_res
        lon0 = (torch.floor(lon_output_grid.min() / latlon_res) - 1) * latlon_res
        lat1 = (torch.ceil(lat_output_grid.max() / latlon_res) + 1) * latlon_res
        lon1 = (torch.ceil(lon_output_grid.max() / latlon_res) + 1) * latlon_res
        lat_input_grid = torch.arange(lat0, lat1 + latlon_res, latlon_res)
        lon_input_grid = torch.arange(lon0, lon1 + latlon_res, latlon_res)
        return lat_input_grid, lon_input_grid

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
        if self._interpolator is None:

            return interp.latlon_interpolation_regular(
                x,
                self.lat_input_grid,
                self.lon_input_grid,
                self.lat_output_grid,
                self.lon_output_grid,
            )
        return self._interpolator(x)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using model's center and scale parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize

        Returns
        -------
        torch.Tensor
            Normalized input tensor (x - center) / scale
        """
        return (x - self.in_center) / self.in_scale


    def postprocess_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor using model's center and scale parameters.

        Parameters
        ----------
        x : torch.Tensor
            Normalized output tensor to denormalize

        Returns
        -------
        torch.Tensor
            Denormalized output tensor x * scale + center
        """
        return x * self.out_scale + self.out_center

    def get_windows(self, stride=8):
        window_size = 320 
        height, width = len(self.lat_output_numpy),len(self.lon_output_numpy) #801 1101
        
        if window_size > height or window_size > width:
            raise ValueError("window_size cannot be larger than the panorama dimensions")
        h_starts = list(range(0, height - window_size, stride))
        w_starts = list(range(0, width - window_size, stride))
    
        if (height - window_size) not in h_starts:
            h_starts.append(height - window_size)
        
        if (width - window_size) not in w_starts:
            w_starts.append(width - window_size)
        
        windows = []
        for h_s in h_starts:
            for w_s in w_starts:
                h_e = h_s + window_size
                w_e = w_s + window_size
                windows.append((h_s, h_e, w_s, w_e))
            
        return windows
    @torch.inference_mode()
    def _forward(self, x: torch.Tensor, zeith: torch.Tensor) -> torch.Tensor:
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
        
        
        # Interpolate input to output grid
        x = self._interpolate(x.unsqueeze(0)).squeeze(0)

        # Add batch dimension
        (C, H, W) = x.shape
        x = x.view(1, C, H, W)
        
        # Concatenate invariants if available
        if self.invariants is not None:
            x = torch.concat([x, self.invariants.unsqueeze(0)], dim=1)
        x = self.preprocess_input(x)
        
        image_lr = torch.cat([x[:,:-1,:,:],zeith.unsqueeze(0),x[:,-1:,:,:]],dim=1)
        image_lr = image_lr.to(torch.float32).to(memory_format=torch.channels_last)
        
        image_lr_full = image_lr
        # Run regression model
        if self.regression_model:
            latents_shape = (1, len(self.output_variables), *self.img_shape)
            image_reg_full = torch.zeros((1,len(self.output_variables),len(self.lat_output_numpy),len(self.lon_output_numpy))).to(self.device)
            counts = torch.zeros_like(image_reg_full).to(self.device)
            
            self.windows = self.get_windows(stride=self.stride)
            
            for window in self.windows:
                y_start, y_end,x_start, x_end = int(window[0]),int(window[1]),int(window[2]),int(window[3])
                img_lr = image_lr_full[:,:,y_start:y_end,x_start:x_end]
                image_out = regression_step(
                        net=self.regression_model,
                        img_lr=img_lr,
                        latents_shape= latents_shape
                    )
                image_reg_full[:,:,y_start:y_end,x_start:x_end] += image_out
                counts[:,:,y_start:y_end,x_start:x_end] +=1
            image_reg_full = image_reg_full/counts
            
        
        if self.inference_mode == "regression":
            return self.postprocess_output(image_reg_full)

        if self.inference_mode == 'both':
            seed = self.seed if self.seed is not None else np.random.randint(2**32)
            mdiff = MultiDiffusion(image_reg_full.device)
            image_out_full = []
            for i in range(self.number_of_samples):
                #MultiDiffusion:
                rnd = StackedRandomGenerator(image_reg_full.device, [seed+i])
                
                image_res_out_full = mdiff(net=self.residual_model,img_lr=image_lr_full,regression_output=image_reg_full,windows=self.windows,randn_like=rnd.randn_like,num_steps=self.number_of_steps)
            
                image_out_full.append(image_reg_full + image_res_out_full)

        image_out = torch.concat(image_out_full,dim=0)

        # Denormalize output
        image_out = self.postprocess_output(image_out)
        
        return image_out

    #@batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        zeith: torch.Tensor,
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
            out[i] = self._forward(x[i],zeith)

        return out, output_coords

    def to(self, device: torch.device) -> "CorrDiff":
        """Move the model to a device.

        Parameters
        ----------
        device : torch.device
            Device to move the model to
        """
        self = super().to(device)
        if self.residual_model is not None:
            self.residual_model.to(device)
        self.regression_model.to(device)
        return self


class CorrDiffSolarMD(CorrDiffMD):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.last_srx = None
        self.curr_srx = None
        self.sr_keys1 = None
        
    @classmethod
    def load_model(cls, package: Package) -> "CorrDiffSolarMD":
        """Load the CorrDiffSolar model from a package."""

        model = super().load_model(package)

        with open(package.resolve("metadata.json")) as f:
            metadata = json.load(f)
        # The properties of sub-class CorrDiffSolar
        model.sr_keys1 = metadata["sr_keys1"]
        
        return model

    def get_sza_lonlat(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get longitude and latitude arrays for solar zenith angle calculation.

        Parameters
        ----------
        lon : np.ndarray
            Longitude array
        lat : np.ndarray
            Latitude array

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (longitude, latitude) arrays
        """
        grid = np.meshgrid(lon, lat)
        return (grid[0].reshape(-1), grid[1].reshape(-1))

    def compute_sza(self, output_coords: CoordSystem) -> torch.Tensor:
        """Compute solar zenith angle for given coordinates.

        Parameters
        ----------
        output_coords : CoordSystem
            Output coordinate system

        Returns
        -------
        torch.Tensor
            Solar zenith angle tensor
        """
        lon, lat = self.get_sza_lonlat(output_coords["lon"], output_coords["lat"])
        t = output_coords["time"] + output_coords["lead_time"]
        t = datetime.fromtimestamp(
            t.astype("datetime64[s]").astype("int")[0], tz=timezone.utc
        )
        return torch.Tensor(cos_zenith_angle(t, lon, lat))

    def solar_output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        # sadly we do not support any flexible inference
        time_delta = np.array([int(6e11 * (i + 1)) for i in range(6)], dtype='timedelta64[ns]')
        lead_time = np.array([input_coords["lead_time"][0] - delta for delta in time_delta])
        lead_time = np.flip(lead_time, axis=0)
        lead_time = lead_time.astype('timedelta64[m]')
        return OrderedDict(
            {
                "time": input_coords["time"],
                "lead_time": lead_time,
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(self.output_variables),
                "lat": self.lat_output_numpy,
                "lon": self.lon_output_numpy
            }
        )
    
    def get_total_coord(self, time, nsteps) -> CoordSystem:
        # Set up IO backend
        total_coords = OrderedDict({})
        total_coords["time"] = time
        total_coords["lead_time"] = np.asarray(
            [
                int(10 * i) for i in range(6 * nsteps)
            ], dtype='timedelta64[m]').flatten()
        total_coords.move_to_end("lead_time", last=False)
        total_coords.move_to_end("time", last=False)
        total_coords["sample"] = np.arange(self.number_of_samples)
        total_coords['lat'] = self.lat_output_numpy
        total_coords['lon'] = self.lon_output_numpy
        
        return total_coords

    @torch.inference_mode()
    def __call__(
        self,
        pro_out,
        pro_out_coord,
        dia_out = None,
        dia_out_coord = None,
        verbose = True,
        verbose_idx = 0,
    ) -> tuple[torch.Tensor, CoordSystem]:
        
        sr_dict1 = OrderedDict({
            "time": pro_out_coord["time"],
            "lead_time": pro_out_coord["lead_time"],
            "variable": np.array(self.sr_keys1),
            "lat": self.lat_input_numpy,
            "lon": self.lon_input_numpy,
        })


        sr_in_dict = OrderedDict({
                "time": pro_out_coord["time"],
                "lead_time": pro_out_coord["lead_time"],
                "variable": np.array(self.input_variables),
                "lat": self.lat_input_numpy,
                "lon": self.lon_input_numpy,
        })

        curr_srx1, _ = map_coords(pro_out, pro_out_coord, sr_dict1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.last_srx = self.curr_srx
        self.curr_srx = curr_srx1 #atmos_state

        if self.last_srx is None:
            return None, None
        
        srx = torch.stack([self.last_srx, self.curr_srx], dim=3).flatten(start_dim=2, end_dim=3) #cross 

        x = srx.detach().clone()
        x = x[0].to(self.device)

        t = sr_in_dict["time"] + sr_in_dict["lead_time"] - np.array([3600000000000], dtype="timedelta64[ns]")
        # need to shift an hour back
        t = datetime.utcfromtimestamp(t.astype('datetime64[s]').astype('int')[0])
        yy, mm, dd, hh = int(t.year), int(t.month), int(t.day), int(t.hour)
        
        zeith_arr = []
        for miint in range(6):
            ztime = datetime(yy, mm, dd, hh, miint * 10, 0)
            coords_for_sza = {
                "lat": self.lat_output_numpy,
                "lon": self.lon_output_numpy,
                "time": np.array([np.datetime64(ztime)]),
                "lead_time": np.timedelta64(0, 'ns'),
            }
            zeith = self.compute_sza(coords_for_sza).reshape(1,len(self.lat_output_numpy), len(self.lon_output_numpy))
            zeith_arr.append(zeith)

        zeith = torch.cat(zeith_arr, dim=0).to(self.device)
        
        input_coords_for_corrdiff = self.input_coords()

        input_coords_for_corrdiff["batch"] = np.arange(x.shape[0]) #batch size 1

        self.to(self.device)
        out,_ = super().__call__(x,zeith,input_coords_for_corrdiff)
        
        out = torch.clamp(out,min=0)

        return out, self.solar_output_coords(sr_in_dict)
