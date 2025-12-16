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
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr
import zarr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    interp,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
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


@check_optional_dependencies()
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
    lat_input_grid : torch.Tensor
        Input latitude grid of size [in_lat, in_lon]
    lon_input_grid : torch.Tensor
        Input longitude grid of size [in_lat, in_lon]
    lat_output_grid : torch.Tensor
        Output latitude grid of size [out_lat, out_lon]
    lon_output_grid : torch.Tensor
        Output latitude grid of size [out_lat, out_lon]
    in_center : torch.Tensor
        Model input center normalization tensor of size [in_var]
    in_scale : torch.Tensor
        Model input scale normalization tensor of size [in_var]
    out_center : torch.Tensor
        Model output center normalization tensor of size [out_var]
    out_scale : torch.Tensor
        Model output scale normalization tensor of size [out_var]
    invariants : OrderedDict | None, optional
        Dictionary of invariant features, by default None
    invariant_center : torch.Tensor | None, optional
        Model invariant center normalization tensor of size [len(invariants)], by default None
    invariant_scale : torch.Tensor | None, optional
        Model invariant scale normalization tensor of size [len(invariants)], by default None
    number_of_samples : int, optional
        Number of high resolution samples to draw from diffusion model, by default 1
    number_of_steps : int, optional
        Number of langevin diffusion steps during sampling algorithm, by default 18
    solver : Literal["euler", "heun"], optional
        Discretization of diffusion process, by default "euler"
    sampler_type : Literal["deterministic", "stochastic"], optional
        Type of sampler to use, by default "stochastic"
    inference_mode : Literal["regression", "diffusion", "both"], optional
        Which inference mode to use, by default "both"
    hr_mean_conditioning : bool, optional
        Whether to use high-res mean conditioning, by default True
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
    grid_spacing_tolerance : float, optional
        Relative tolerance for checking regular grid spacing. Allows for slight variations
        in grid spacing (e.g., for Gaussian grids). For 1D grids, raises ValueError if
        spacing variations exceed this tolerance; use of a 2D curvilinear grid is suggested then.
        Default is 1e-5 (0.001%).
    grid_bounds_margin : float, optional
        Fraction of input grid range to allow for extrapolation beyond input grid bounds.
        For example, 0.05 allows output grid to extend 5% beyond input grid range.
        Useful for Gaussian grids that don't include poles. For 1D grids, raises ValueError
        if the output grid extends beyond the allowed bounds. Default is 0.0 (no extrapolation).
    sigma_min : float | None, optional
        Minimum noise level for diffusion process. If None, uses sampler-specific defaults
        By default None.
    sigma_max : float | None, optional
        Maximum noise level for diffusion process. If None, uses sampler-specific defaults
        By default None.
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
        solver: Literal["euler", "heun"] = "euler",
        sampler_type: Literal["deterministic", "stochastic"] = "stochastic",
        inference_mode: Literal["regression", "diffusion", "both"] = "both",
        hr_mean_conditioning: bool = True,
        seed: int | None = None,
        grid_spacing_tolerance: float = 1e-5,
        grid_bounds_margin: float = 0.0,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
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
        self.number_of_samples = number_of_samples
        self.number_of_steps = number_of_steps
        self.solver = solver
        self.sampler_type = sampler_type
        self.inference_mode = inference_mode
        self.hr_mean_conditioning = hr_mean_conditioning
        self.seed = seed
        self.img_shape = (lat_output_grid.shape[0], lon_output_grid.shape[0])
        self.invariants_dict = invariants
        self.invariant_center = invariant_center
        self.invariant_scale = invariant_scale
        self.grid_spacing_tolerance = grid_spacing_tolerance
        self.grid_bounds_margin = grid_bounds_margin
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        # Store models
        self.residual_model = residual_model
        self.regression_model = regression_model

        # Store variable names
        # Note: After preprocessing, actual channels = input_variables + invariant_variables
        self.input_variables = list(input_variables)  # Weather variable channels
        self.output_variables = list(output_variables)  # Output channels
        # Note: self.invariant_variables is set in _register_buffers()

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

        # Set up sampler
        self.sampler = self._setup_sampler(sampler_type)

    @classmethod
    def _validate_grid_format(
        cls, lat_grid: torch.Tensor, lon_grid: torch.Tensor, grid_name: str = "grid"
    ) -> None:
        """Validate lat/lon grid format and ordering.

        Validates that:
        - 1D grids (regular/rectilinear) are increasing
        - 2D grids are truly curvilinear (not rectilinear stored as 2D)

        Parameters
        ----------
        lat_grid : torch.Tensor
            Latitude grid to validate (1D or 2D)
        lon_grid : torch.Tensor
            Longitude grid to validate (1D or 2D)
        grid_name : str
            Name of the grid (for error messages), by default "grid"

        Raises
        ------
        ValueError
            If grids are 2D rectilinear (should be 1D) or 1D grids are not increasing
        """
        # Validate grid dimensions
        lat_ndim = len(lat_grid.shape)
        lon_ndim = len(lon_grid.shape)

        # Check for invalid dimensions (>2D or mismatched)
        if lat_ndim > 2 or lon_ndim > 2:
            raise ValueError(
                f"{grid_name.capitalize()} grids must be 1D or 2D. "
                f"Got lat shape {lat_grid.shape} ({lat_ndim}D), "
                f"lon shape {lon_grid.shape} ({lon_ndim}D). "
                "Use 1D for regular grids or 2D for curvilinear grids."
            )

        if lat_ndim != lon_ndim:
            raise ValueError(
                f"{grid_name.capitalize()} grids have mismatched dimensions. "
                f"Got lat shape {lat_grid.shape} ({lat_ndim}D), "
                f"lon shape {lon_grid.shape} ({lon_ndim}D). "
                "Both must be either 1D (regular grid) or 2D (curvilinear grid)."
            )

        # Validate based on grid type
        if lat_ndim == 1:
            # Regular grid - validate that coordinates are increasing
            if lat_grid[0] > lat_grid[-1]:
                raise ValueError(
                    f"{grid_name.capitalize()} latitude must be increasing (South to North). "
                    f"Got {lat_grid[0].item():.2f} to {lat_grid[-1].item():.2f}. "
                    "Please reverse the latitude array in your NetCDF file."
                )
            if lon_grid[0] > lon_grid[-1]:
                raise ValueError(
                    f"{grid_name.capitalize()} longitude must be increasing. "
                    f"Got {lon_grid[0].item():.2f} to {lon_grid[-1].item():.2f}. "
                    "Please reverse the longitude array in your NetCDF file."
                )

        else:  # lat_ndim == 2
            # 2D grids - check if they're actually rectilinear (should be stored as 1D)
            if cls._is_rectilinear(lat_grid, lon_grid):
                raise ValueError(
                    f"{grid_name.capitalize()} grid is rectilinear but stored as 2D. "
                    f"Got lat shape {lat_grid.shape}, lon shape {lon_grid.shape}. "
                    "Rectilinear grids should be stored as 1D arrays (lat[H], lon[W]) "
                    "for efficiency. Only curvilinear grids need 2D storage."
                )
            # 2D curvilinear grids are OK - no ordering requirements

    @staticmethod
    def _is_rectilinear(lat_grid: torch.Tensor, lon_grid: torch.Tensor) -> bool:
        """Check if 2D lat/lon grids represent a rectilinear (regular) grid.

        A rectilinear grid has the property that:
        - lat[i, j] is constant for all j (only varies with i)
        - lon[i, j] is constant for all i (only varies with j)

        Parameters
        ----------
        lat_grid : torch.Tensor
            2D latitude grid [H, W]
        lon_grid : torch.Tensor
            2D longitude grid [H, W]

        Returns
        -------
        bool
            True if the grid is rectilinear, False if curvilinear
        """
        if len(lat_grid.shape) != 2 or len(lon_grid.shape) != 2:
            return False

        # Check if lat is constant along second dimension (columns)
        lat_constant_along_lon = torch.allclose(
            lat_grid[:, 0:1].expand_as(lat_grid),
            lat_grid,
            rtol=1e-5,
            atol=1e-5,
        )

        # Check if lon is constant along first dimension (rows)
        lon_constant_along_lat = torch.allclose(
            lon_grid[0:1, :].expand_as(lon_grid),
            lon_grid,
            rtol=1e-5,
            atol=1e-5,
        )

        return lat_constant_along_lon and lon_constant_along_lat

    def _check_grid_spacing(
        self,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
    ) -> None:
        """Check that regular (1D) grids are regularly spaced.

        For regular (1D) grids, validates that each grid is regularly spaced within itself.
        Lat and lon can have different spacing from each other.

        For curvilinear (2D) grids, this check is skipped.
        """
        # Only check resolution for regular (1D) grids
        # Note: lat and lon spacing do NOT need to be equal to each other,
        # but each must be regularly spaced within itself
        if len(lat_input_grid.shape) == 1:
            # Check latitude is regularly spaced
            lat_diffs = lat_input_grid[1:] - lat_input_grid[:-1]
            if not torch.allclose(
                lat_diffs, lat_diffs[0], rtol=self.grid_spacing_tolerance
            ):
                raise ValueError(
                    f"Input latitude grid must be regularly spaced (within {self.grid_spacing_tolerance*100:.2f}% tolerance), "
                    "but found varying spacing. Consider using a 2D curvilinear grid instead."
                )

            # Check longitude is regularly spaced
            lon_diffs = lon_input_grid[1:] - lon_input_grid[:-1]
            if not torch.allclose(
                lon_diffs, lon_diffs[0], rtol=self.grid_spacing_tolerance
            ):
                raise ValueError(
                    f"Input longitude grid must be regularly spaced (within {self.grid_spacing_tolerance*100:.2f}% tolerance), "
                    "but found varying spacing. Consider using a 2D curvilinear grid instead."
                )

    def _check_grid_bounds(
        self,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
    ) -> None:
        """Check that output grids are within input grid bounds (with optional margin).

        The margin is controlled by grid_bounds_margin (fraction of input range).
        Default is 0.0 (strict bounds), but can be set higher to allow extrapolation.
        """
        lat_input_range = lat_input_grid.max() - lat_input_grid.min()
        lon_input_range = lon_input_grid.max() - lon_input_grid.min()

        lat_margin = self.grid_bounds_margin * lat_input_range
        lon_margin = self.grid_bounds_margin * lon_input_range

        if not torch.all(
            lat_output_grid >= lat_input_grid.min() - lat_margin
        ) or not torch.all(lat_output_grid <= lat_input_grid.max() + lat_margin):
            raise ValueError(
                f"Output latitude grid extends beyond input grid bounds (margin={self.grid_bounds_margin*100:.1f}%). "
                f"Got output range [{lat_output_grid.min():.2f}, {lat_output_grid.max():.2f}], "
                f"input range [{lat_input_grid.min():.2f}, {lat_input_grid.max():.2f}]"
            )
        if not torch.all(
            lon_output_grid >= lon_input_grid.min() - lon_margin
        ) or not torch.all(lon_output_grid <= lon_input_grid.max() + lon_margin):
            raise ValueError(
                f"Output longitude grid extends beyond input grid bounds (margin={self.grid_bounds_margin*100:.1f}%). "
                f"Got output range [{lon_output_grid.min():.2f}, {lon_output_grid.max():.2f}], "
                f"input range [{lon_input_grid.min():.2f}, {lon_input_grid.max():.2f}]"
            )

    def _check_latlon_grid(
        self,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
    ) -> None:
        """Validate lat/lon grid resolution and bounds.

        Calls _check_grid_spacing and _check_grid_bounds, which can be overridden separately.
        """
        self._check_grid_spacing(lat_input_grid, lon_input_grid)
        self._check_grid_bounds(
            lat_input_grid, lon_input_grid, lat_output_grid, lon_output_grid
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
        # Register grid coordinates and validate
        self._check_latlon_grid(
            lat_input_grid, lon_input_grid, lat_output_grid, lon_output_grid
        )

        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)

        # Set up interpolation
        # Determine if input grid is regular (1D) or curvilinear (2D)
        # Note: 2D rectilinear grids are rejected by validation, so 2D = curvilinear
        if len(lat_input_grid.shape) == 1:
            self._interpolator = None  # Use efficient regular grid interpolation
        else:
            # 2D grid = curvilinear (rectilinear 2D grids rejected by validation)
            self._interpolator = interp.LatLonInterpolation(
                lat_input_grid,
                lon_input_grid,
                lat_output_grid,
                lon_output_grid,
            )

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
        sampler_kwargs: dict[str, Any] = {"num_steps": self.number_of_steps}

        # Add sigma parameters if specified (common to both samplers)
        if self.sigma_min is not None:
            sampler_kwargs["sigma_min"] = self.sigma_min
        if self.sigma_max is not None:
            sampler_kwargs["sigma_max"] = self.sigma_max

        if sampler_type == "deterministic":
            if self.hr_mean_conditioning:
                raise NotImplementedError(
                    "High-res mean conditioning is not yet implemented for the deterministic sampler"
                )
            sampler_kwargs["solver"] = self.solver
            return partial(deterministic_sampler, **sampler_kwargs)
        elif sampler_type == "stochastic":
            return partial(stochastic_sampler, **sampler_kwargs)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

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

    @staticmethod
    def _load_json_from_package(package: Package, filename: str) -> dict:
        """Load and parse a JSON file from a package with error handling.

        Parameters
        ----------
        package : Package
            Package containing the JSON file
        filename : str
            Name of the JSON file to load

        Returns
        -------
        dict
            Parsed JSON data

        Raises
        ------
        ValueError
            If the file is empty or contains invalid JSON
        """
        file_path = package.resolve(filename)
        try:
            with open(file_path) as f:
                content = f.read()
                if not content.strip():
                    raise ValueError(f"{filename} is empty at: {file_path}")
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse {filename} at: {file_path}. "
                f"File may be corrupted or contain invalid JSON. Error: {e}"
            )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        device: str | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
    ) -> DiagnosticModel:
        """Load CorrDiff model from package.

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration
        device : str | None, optional
            Device to load model onto (e.g., "cuda:0", "cpu"). By default None.
        sigma_min : float | None, optional
            Minimum noise level for diffusion process. Priority order: (1) this argument,
            (2) metadata.json, (3) sampler internal defaults. By default None.
        sigma_max : float | None, optional
            Maximum noise level for diffusion process. Priority order: (1) this argument,
            (2) metadata.json, (3) sampler internal defaults. By default None.

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

        # Apply inference optimizations (following CorrDiffTaiwan patterns)
        # Disable profiling mode for both models
        residual.profile_mode = False
        regression.profile_mode = False

        # Move to device first (required before channels_last conversion)
        if device is not None:
            residual = residual.to(device)
            regression = regression.to(device)

        # Convert to channels_last memory format for better GPU performance
        residual = residual.to(memory_format=torch.channels_last)
        regression = regression.to(memory_format=torch.channels_last)

        # Configure torch dynamo for potential compilation
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()

        # Load metadata
        metadata = cls._load_json_from_package(package, "metadata.json")
        raw_input_variables = metadata["input_variables"]
        duplicates = [
            var for var, count in Counter(raw_input_variables).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                "metadata['input_variables'] contains duplicate entries: "
                f"{duplicates}. Each variable should appear only once."
            )
        output_variables = metadata["output_variables"]
        invariant_variables = metadata.get("invariant_variables", None)

        # Load model parameters (if not provided, use default values)
        number_of_samples = metadata.get("number_of_samples", 1)
        number_of_steps = metadata.get("number_of_steps", 18)
        solver = metadata.get("solver", "euler")
        sampler_type = metadata.get("sampler_type", "stochastic")
        inference_mode = metadata.get("inference_mode", "both")
        hr_mean_conditioning = metadata.get("hr_mean_conditioning", True)
        seed = metadata.get("seed", None)
        sigma_min_metadata = metadata.get("sigma_min", None)
        sigma_max_metadata = metadata.get("sigma_max", None)
        grid_spacing_tolerance = metadata.get("grid_spacing_tolerance", 1e-5)
        grid_bounds_margin = metadata.get("grid_bounds_margin", 0.0)

        input_variables = list(raw_input_variables)

        # Load normalization statistics
        stats = cls._load_json_from_package(package, "stats.json")

        # Load input normalization parameters
        in_center_values = []
        in_scale_values = []
        for var in input_variables:
            if var not in stats["input"]:
                raise KeyError(
                    f"stats.json is missing normalization statistics for input variable '{var}'."
                )
            in_center_values.append(stats["input"][var]["mean"])
            in_scale_values.append(stats["input"][var]["std"])

        in_center = torch.tensor(in_center_values, device=device)
        in_scale = torch.tensor(in_scale_values, device=device)

        # Load output normalization parameters
        out_center = torch.tensor(
            [stats["output"][v]["mean"] for v in output_variables], device=device
        )
        out_scale = torch.tensor(
            [stats["output"][v]["std"] for v in output_variables], device=device
        )

        # Load output lat/lon grid
        with xr.open_dataset(package.resolve("output_latlon_grid.nc")) as ds:
            lat_output_grid = torch.as_tensor(np.array(ds["lat"][:]), device=device)
            lon_output_grid = torch.as_tensor(np.array(ds["lon"][:]), device=device)

            # Validate output grid format and ordering
            cls._validate_grid_format(
                lat_output_grid, lon_output_grid, grid_name="output"
            )

        # Load input lat/lon grid (or infer from metadata)
        try:
            with xr.open_dataset(package.resolve("input_latlon_grid.nc")) as ds:
                lat_input_grid = torch.as_tensor(np.array(ds["lat"][:]), device=device)
                lon_input_grid = torch.as_tensor(np.array(ds["lon"][:]), device=device)

                # Validate input grid format and ordering
                cls._validate_grid_format(
                    lat_input_grid, lon_input_grid, grid_name="input"
                )
        except FileNotFoundError:
            if "latlon_res" in metadata:
                latlon_res = metadata["latlon_res"]
                lat_input_grid, lon_input_grid = cls._infer_input_latlon_grid(
                    lat_output_grid, lon_output_grid, latlon_res
                )
            else:
                raise FileNotFoundError(
                    "input_latlon_grid.nc not found and latlon_res not in metadata"
                )

        # Load invariants if available
        # Note: Missing file is OK only if metadata doesn't require invariants
        # Wrong variable names or missing required file is an error (configuration problem)
        try:
            invariants_path = package.resolve("invariants.nc")
        except FileNotFoundError:
            # Check if invariants were required by metadata
            if invariant_variables:
                raise FileNotFoundError(
                    f"invariants.nc not found but metadata specifies invariant_variables: {invariant_variables}"
                )
            # No invariants file and none required - model will run without invariants
            invariants = None
            invariant_center = None
            invariant_scale = None
        else:
            # File exists - load and validate
            with xr.open_dataset(invariants_path) as ds:
                # Determine which variables to load and in what order
                if invariant_variables is None:
                    # Load all available variables
                    var_names = list(ds.data_vars)
                else:
                    # Load only specified variables in the specified order
                    var_names = invariant_variables
                    # Validate that all requested variables exist
                    missing_vars = [v for v in var_names if v not in ds.data_vars]
                    if missing_vars:
                        raise ValueError(
                            f"Invariant variables {missing_vars} not found in invariants.nc. "
                            f"Available variables: {list(ds.data_vars)}"
                        )

                invariants = OrderedDict(
                    (var_name, torch.as_tensor(np.array(ds[var_name]), device=device))
                    for var_name in var_names
                )

                # Load invariant normalization parameters
                invariant_center = torch.tensor(
                    [stats["invariants"][v]["mean"] for v in invariants], device=device
                )
                invariant_scale = torch.tensor(
                    [stats["invariants"][v]["std"] for v in invariants], device=device
                )

        # Decide which sigma values to pass into the constructor:
        # 1. Explicit load_model arguments (sigma_min / sigma_max) if provided
        # 2. Otherwise, fall back to metadata values (sigma_min_metadata / sigma_max_metadata)
        # 3. If both are None, __init__ will apply sampler-specific defaults
        effective_sigma_min = sigma_min if sigma_min is not None else sigma_min_metadata
        effective_sigma_max = sigma_max if sigma_max is not None else sigma_max_metadata

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
            solver=solver,
            sampler_type=sampler_type,
            inference_mode=inference_mode,
            hr_mean_conditioning=hr_mean_conditioning,
            seed=seed,
            grid_spacing_tolerance=grid_spacing_tolerance,
            grid_bounds_margin=grid_bounds_margin,
            sigma_min=effective_sigma_min,
            sigma_max=effective_sigma_max,
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
        # Inherit device from output grid
        lat_input_grid = torch.arange(
            lat0, lat1 + latlon_res, latlon_res, device=lat_output_grid.device
        )
        lon_input_grid = torch.arange(
            lon0, lon1 + latlon_res, latlon_res, device=lon_output_grid.device
        )
        return lat_input_grid, lon_input_grid

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate from input lat/lon onto output lat/lon grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to interpolate [C, H_in, W_in]

        Returns
        -------
        torch.Tensor
            Interpolated tensor [C, H_out, W_out]

        Note
        ----
        Override this method in a subclass to implement custom interpolation logic
        or to skip interpolation when data is already on the target grid.
        """
        # Regular grid
        if self._interpolator is None:
            # Input grids are guaranteed to be 1D by validation
            return interp.latlon_interpolation_regular(
                x,
                self.lat_input_grid,
                self.lon_input_grid,
                self.lat_output_grid,
                self.lon_output_grid,
            )

        # Curvilinear grid - use cached interpolator
        return self._interpolator(x)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using model's center and scale parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Normalized input tensor (x - center) / scale
        """
        return (x - self.in_center) / self.in_scale

    def preprocess_input(
        self, x: torch.Tensor, valid_time: datetime | None = None
    ) -> torch.Tensor:
        """Complete input preprocessing pipeline.

        Performs interpolation to output grid, ensures a batch dimension,
        concatenates invariants if available, and normalizes the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[C, H_in, W_in]`` or ``[B, C, H_in, W_in]``.
            If a batch dimension is not present, one will be added.
        valid_time : datetime | None, optional
            Validity time for this sample (when the atmospheric state is valid).
            The base class ignores this parameter. Subclasses can override this
            method to compute time-dependent features such as solar zenith angle
            (SZA). Default is None.

        Returns
        -------
        torch.Tensor
            Preprocessed and normalized input tensor ``[B, C+C_inv, H_out, W_out]``.

        Notes
        -----
        For subclass implementers: ``valid_time`` is provided as an optional hook for
        time-dependent preprocessing (e.g., solar zenith angle). The base
        implementation ignores it; override this method if your model needs it.
        """
        # Accept both [C, H_in, W_in] and [B, C, H_in, W_in]
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if x.ndim != 4:
            raise ValueError(
                f"preprocess_input expected input of shape [C, H, W] or [B, C, H, W], got {tuple(x.shape)}"
            )

        # Interpolate each batch element to output grid; _interpolate expects [C, H, W]
        b, c, h, w = x.shape
        x = torch.stack([self._interpolate(x[i]) for i in range(b)], dim=0)

        # Concatenate invariants if available
        if self.invariants is not None:
            # Invariants are stored as [C_inv, H_out, W_out]; expand to batch and concat
            inv = self.invariants.unsqueeze(0).expand(b, -1, -1, -1)
            x = torch.concat([x, inv], dim=1)

        # Normalize input
        x = self.normalize_input(x)

        return x

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

    def _inference_context(self) -> nullcontext:
        """Return context manager for model inference.

        Base class returns nullcontext (no-op). Subclasses can override
        to add autocast, profiling, or other context-dependent behavior.

        Returns
        -------
        nullcontext
            Context manager to wrap inference operations
        """
        return nullcontext()

    @torch.inference_mode()
    def _forward(
        self, x: torch.Tensor, valid_time: datetime | None = None
    ) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        valid_time : datetime | None, optional
            Validity time of the input sample (when the atmospheric state is valid).
            Used by subclasses for time-dependent preprocessing. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.solver not in ["euler", "heun"]:
            raise ValueError(
                f"solver must be either 'euler' or 'heun' but got {self.solver}"
            )

        # Preprocess input (interpolate, add batch dimension, add invariants, normalize)
        # Base class ignores valid_time; subclasses can override preprocess_input to use it
        image_lr = self.preprocess_input(x, valid_time)
        image_lr = image_lr.to(torch.float32).to(memory_format=torch.channels_last)

        # Run regression model
        if self.regression_model:
            latents_shape = (1, len(self.output_variables), *image_lr.shape[-2:])
            with self._inference_context():
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
                with self._inference_context():
                    image_res = diffusion_step(
                        net=self.residual_model,
                        sampler_fn=self.sampler,
                        img_shape=image_lr.shape[-2:],
                        img_out_channels=len(self.output_variables),
                        rank_batches=[[seed + i]],
                        img_lr=image_lr,
                        rank=1,
                        device=image_lr.device,
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
        image_out = self.postprocess_output(image_out)

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
            Input coordinate system. May optionally contain a ``"time"`` key with an
            array-like of numpy datetime64 values (or a ``list[datetime]``) representing
            the validity time of each sample (i.e., when the atmospheric state is
            valid, not the forecast initialization time). If present, each value is
            passed to ``preprocess_input`` as ``valid_time`` for time-dependent
            preprocessing (e.g., computing solar zenith angle). If absent, ``None`` is
            passed.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system

        Notes
        -----
        Subclass usage: The base class passes ``valid_time`` to ``preprocess_input``
        but does not use it. Subclasses can override ``preprocess_input`` to compute
        time-dependent features like solar zenith angle (SZA). The ``coords["time"]``
        array must have length equal to the batch size (first dimension of x).
        """

        # Pull optional time metadata before any coordinate validation.
        #
        # Design note: CoordSystem was designed for dimensional coords (batch, variable,
        # lat, lon) where each key maps to a tensor axis. "time" here is per-sample
        # metadata (validity timestamp), not a tensor dimension.
        #
        # This method strips "time" before calling:
        # - earth2studio.models.batch.batch_func._compress_batch (enforces len(coords) == x.ndim)
        # - earth2studio.utils.coords.handshake_dim / handshake_coords (assume only dimensional keys)
        #
        # If more models need per-sample metadata, the proper fix is to teach the batching /
        # handshake utilities to ignore or explicitly allow metadata keys (e.g. via a
        # `metadata_keys={"time"}` allowlist on @batch_func), rather than repeating this
        # local workaround in each model.
        time_array = coords.get("time", None)
        coords_no_time = coords
        if time_array is not None:
            coords_no_time = coords.copy()
            del coords_no_time["time"]

        output_coords = self.output_coords(coords_no_time)

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )

        # Extract and validate time information if present in coords
        #
        # Note: we intentionally keep the public coord key as "time" (consistent with
        # earth2studio conventions), but internally treat it as a validity timestamp
        # and pass it to subclasses as `valid_time`.
        valid_time_list: list[datetime | None]
        if time_array is not None:
            # Disallow scalar timestamps: we require one entry per batch element
            if isinstance(time_array, (datetime, np.datetime64)):
                raise TypeError(
                    'coords["time"] must be an array-like of timestamps (one per batch element), '
                    f"but got a scalar {type(time_array)!r}"
                )

            # Validate time array length matches batch size
            if not hasattr(time_array, "__len__"):
                raise TypeError(
                    'coords["time"] must be an array-like of timestamps (supports len()), '
                    f"but got {type(time_array)!r}"
                )
            if len(time_array) != x.shape[0]:
                raise ValueError(
                    f"time array length ({len(time_array)}) must match batch size ({x.shape[0]})"
                )

            # Accept list[datetime] directly (already the desired type for subclasses)
            if isinstance(time_array, (list, tuple)) and all(
                isinstance(t, datetime) for t in time_array
            ):
                valid_time_list = list(time_array)
            else:
                # Normalize to numpy array and require datetime64 dtype
                time_np = np.asarray(time_array)
                if not np.issubdtype(time_np.dtype, np.datetime64):
                    raise TypeError(
                        'coords["time"] must be array-like of numpy datetime64 (e.g., dtype="datetime64[ns]") '
                        f"or a list[datetime], but got {type(time_array)!r}"
                    )
                valid_time_list = timearray_to_datetime(time_np)
        else:
            valid_time_list = [None] * out.shape[0]

        for i in range(out.shape[0]):
            out[i] = self._forward(x[i], valid_time_list[i])

        return out, output_coords

    def to(self, device: torch.device) -> "CorrDiff":
        """Move the model to a device.

        Parameters
        ----------
        device : torch.device
            Device to move the model to
        """
        self = super().to(device)
        self.residual_model.to(device)
        self.regression_model.to(device)
        return self


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


@check_optional_dependencies()
class CorrDiffTaiwan(torch.nn.Module, AutoModelMixin):
    """

    CorrDiff is a Corrector Diffusion model that learns mappings between
    low- and high-resolution weather data with high fidelity. This particular
    model was trained over a particular region near Taiwan.


    Note
    ----
    This model and checkpoint are from Mardani, Morteza, et al. 2023. For more
    information see the following references:

    - https://arxiv.org/abs/2309.15214
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
    seed: int | None, optional
        Random seed for reproducibility. Default is None.
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
        seed: int | None = None,
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
        self.seed = seed
        self.output_variables = OUT_VARIABLES  # Default set of output variables

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
    @check_optional_dependencies()
    def load_model(cls, package: Package, device: str | None = None) -> DiagnosticModel:
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
            ),
            override_args={"use_apex_gn": False},
        )
        residual.use_fp16, residual.profile_mode = True, False
        residual = residual.eval()
        if device is not None:
            residual = residual.to(device)
        residual = residual.to(memory_format=torch.channels_last)

        regression = PhysicsNemoModule.from_checkpoint(
            str(
                checkpoint_zip.parent
                / Path("corrdiff_inference_package/checkpoints/regression.mdlus")
            ),
            override_args={"use_apex_gn": False},
        )
        regression.use_fp16, regression.profile_mode = True, False
        regression = regression.eval()
        if device is not None:
            regression = regression.to(device)
        regression = regression.to(memory_format=torch.channels_last)

        # Compile models
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        # residual = torch.compile(residual)
        # regression = torch.compile(regression)

        store = zarr.storage.LocalStore(
            str(
                checkpoint_zip.parent
                / Path(
                    "corrdiff_inference_package/dataset/2023-01-24-cwb-4years_5times.zarr"
                )
            )
        )

        root = zarr.group(store)
        # Get output lat/lon grid
        out_lat = torch.as_tensor(root["XLAT"][:], dtype=torch.float32, device=device)
        out_lon = torch.as_tensor(root["XLONG"][:], dtype=torch.float32, device=device)

        # get normalization info
        in_inds = [0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19]
        in_center = (
            torch.as_tensor(
                root["era5_center"][in_inds],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

        in_scale = (
            torch.as_tensor(
                root["era5_scale"][in_inds],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

        out_inds = [0, 17, 18, 19]
        out_center = (
            torch.as_tensor(
                root["cwb_center"][out_inds],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )

        out_scale = (
            torch.as_tensor(
                root["cwb_scale"][out_inds],
                dtype=torch.float32,
                device=device,
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
                f"solver must be either 'euler' or 'heun', " f"but got {self.solver}"
            )

        # Interpolate
        x = self._interpolate(x)

        # Add sample dimension
        x = x.unsqueeze(0).to(memory_format=torch.channels_last)
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
        # Concat grid features (only for regression model)
        x_reg = torch.cat((x, grid), dim=1)

        # Create seeds for each sample
        seed = self.seed if self.seed is not None else np.random.randint(2**32)
        if seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)
        else:
            gen = None
        sample_seeds = (
            torch.randint(
                0, 2**32, (self.number_of_samples,), device=x.device, generator=gen
            )
            .cpu()
            .tolist()
        )

        sampler_fn = partial(
            deterministic_sampler,
            num_steps=self.number_of_steps,
            solver=self.solver,
        )

        # Get high-res image shape
        coord = self.output_coords(self.input_coords())
        H_hr = coord["lat"].shape[0]
        W_hr = coord["lon"].shape[1]

        mean_hr = self.unet_regression(
            self.regression_model,
            img_lr=x_reg,
            output_channels=len(OUT_VARIABLES),
            number_of_samples=self.number_of_samples,
        )

        res_hr = diffusion_step(
            net=self.residual_model,
            sampler_fn=sampler_fn,
            img_shape=(H_hr, W_hr),
            img_out_channels=len(self.output_variables),
            rank_batches=[sample_seeds],
            img_lr=x.expand(self.number_of_samples, -1, -1, -1).to(
                memory_format=torch.channels_last
            ),
            rank=1,
            device=x.device,
            mean_hr=mean_hr[0:1],  # Diffusion only takes one mean input
        )

        x_hr = mean_hr + res_hr
        x_hr = self.out_scale * x_hr + self.out_center
        return x_hr

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
        for i in range(x.shape[0]):
            out[i] = self._forward(x[i])

        return out, output_coords

    @staticmethod
    def unet_regression(
        net: torch.nn.Module,
        img_lr: torch.Tensor,
        output_channels: int,
        number_of_samples: int,
    ) -> torch.Tensor:
        """
        Perform U-Net regression.

        Parameters
        ----------
        net : torch.nn.Module
            U-Net model for regression.
        img_lr : torch.Tensor
            Low-resolution input image of shape (1, C_in, H_hr, W_hr).
        output_channels : int
            Number of output channels C_out.
        number_of_samples : int
            Number of samples to generate for the single input batch element.
            Only used to expand the shape of the output tensor.

        Returns
        -------
        torch.Tensor: Predicted output with shape (number_of_samples, C_out,
        H_hr, W_hr).
        """
        mean_hr = regression_step(
            net=net,
            img_lr=img_lr,
            latents_shape=(
                number_of_samples,
                output_channels,
                img_lr.shape[-2],
                img_lr.shape[-1],
            ),
            lead_time_label=None,
        )

        return mean_hr
