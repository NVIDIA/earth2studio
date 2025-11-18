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
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr
import zarr
from torch.nn import functional as F

from earth2studio.data import TimeWindow
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
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("corrdiff")
    PhysicsNemoModule = None
    StackedRandomGenerator = None
    deterministic_sampler = None


# Type alias for datasources that can be wrapped by TimeWindow.
# They are expected to be callable like:
#   datasource(time: datetime, variables: Sequence[str]) -> xarray.DataArray
DataSource = Callable[[datetime, Sequence[str]], xr.DataArray]


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
    time_window : dict | None, optional
        Time window configuration for temporal context. Dictionary containing:
        - 'offsets': List of time offsets (required)
        - 'offsets_units': Units for offsets - "seconds", "hours", or "days" (default: "seconds")
        - 'suffixes': List of suffix strings for variable names (required)
        - 'group_by': Variable ordering - "variable" or "offset" (default: "variable")
        - 'description': Human-readable description (optional)
        Used by create_time_window_wrapper() to automatically configure TimeWindow data wrappers.
        By default None
    grid_spacing_tolerance : float, optional
        Relative tolerance for checking regular grid spacing. Allows for slight variations
        in grid spacing (e.g., for Gaussian grids). Default is 1e-5 (0.001%).
    grid_bounds_margin : float, optional
        Fraction of input grid range to allow for extrapolation beyond input grid bounds.
        For example, 0.05 allows output grid to extend 5% beyond input grid range.
        Useful for Gaussian grids that don't include poles. Default is 0.0 (no extrapolation).
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
        time_window: dict | None = None,
        grid_spacing_tolerance: float = 1e-5,
        grid_bounds_margin: float = 0.0,
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
        self.time_window = time_window
        self.invariants_dict = invariants
        self.invariant_center = invariant_center
        self.invariant_scale = invariant_scale
        self.grid_spacing_tolerance = grid_spacing_tolerance
        self.grid_bounds_margin = grid_bounds_margin

        # Store models
        self.residual_model = residual_model
        self.regression_model = regression_model

        # Store variable names
        # Note: After preprocessing, actual channels = input_variables + invariant_variables
        self.input_variables = list(input_variables)  # Weather variable channels
        self.output_variables = list(output_variables)  # Output channels
        self.invariant_variables = list(invariants.keys()) if invariants else []

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

    def create_time_window_wrapper(self, datasource: DataSource) -> TimeWindow:
        """Create a ``TimeWindow`` wrapper for a compatible datasource.

        The wrapper is configured from this model's ``time_window`` metadata.
        It assumes that the underlying datasource:
        - accepts ``(time: datetime, variables: Sequence[str])`` as arguments
        - returns an ``xarray.DataArray`` with a ``variable`` dimension matching
          the **base** input variables (e.g. ``tas``, ``pr`` without time suffixes).

        In particular, the model's ``input_variables`` may be *suffix-expanded*
        (e.g. ``tas_t-1``, ``tas_t``, ``tas_t+1``), while the datasource only
        needs to provide the corresponding base variables (``tas``); the
        ``TimeWindow`` wrapper is responsible for mapping between the two.

        Parameters
        ----------
        datasource : DataSource
            The underlying datasource to wrap.

        Returns
        -------
        TimeWindow
            Configured ``TimeWindow`` wrapper that fetches data at the required
            time offsets and exposes suffixed variables to the model.

        Raises
        ------
        ValueError
            If the model doesn't have ``time_window`` metadata defined.
        """
        if self.time_window is None:
            raise ValueError(
                "Model does not have time_window metadata defined. "
                "Cannot create TimeWindow wrapper automatically."
            )

        time_window = self._validate_time_window_metadata(self.time_window)
        offsets_config = time_window["offsets"]
        offsets_units = time_window["offsets_units"]
        suffixes = time_window["suffixes"]
        group_by = time_window["group_by"]

        # Convert offsets to timedelta objects
        if offsets_units == "seconds":
            offsets = [timedelta(seconds=offset) for offset in offsets_config]
        elif offsets_units == "hours":
            offsets = [timedelta(hours=offset) for offset in offsets_config]
        elif offsets_units == "days":
            offsets = [timedelta(days=offset) for offset in offsets_config]
        else:
            raise ValueError(
                f"Unknown offsets_units: {offsets_units}. "
                "Must be one of: 'seconds', 'hours', 'days'"
            )

        # Create and return TimeWindow wrapper
        return TimeWindow(
            datasource=datasource,
            offsets=offsets,
            suffixes=suffixes,
            group_by=group_by,
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained corrdiff model package from Nvidia model registry"""
        raise NotImplementedError

    @staticmethod
    def _validate_time_window_metadata(time_window: dict) -> dict:
        """Validate and normalize time window metadata loaded from package."""
        if time_window is None:
            raise ValueError("time_window metadata cannot be None.")
        if not isinstance(time_window, dict):
            raise ValueError(
                f"time_window metadata must be a dictionary, got {type(time_window)}."
            )

        if "offsets" not in time_window:
            raise ValueError("time_window metadata missing required field 'offsets'.")
        offsets = list(time_window["offsets"])
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(
                "time_window 'offsets' must be a non-empty list of numeric values."
            )

        if "suffixes" not in time_window:
            raise ValueError("time_window metadata missing required field 'suffixes'.")
        suffixes = [str(suffix) for suffix in time_window["suffixes"]]
        if not isinstance(suffixes, list) or not suffixes:
            raise ValueError(
                "time_window 'suffixes' must be a non-empty list of strings."
            )
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("time_window 'suffixes' entries must be unique.")
        if len(offsets) != len(suffixes):
            raise ValueError(
                "time_window metadata has mismatched lengths: "
                f"{len(offsets)} offsets but {len(suffixes)} suffixes. "
                "These must be equal length lists."
            )
        if not all(isinstance(suffix, str) and suffix for suffix in suffixes):
            raise ValueError("time_window 'suffixes' must contain non-empty strings.")

        offsets_units = time_window.get("offsets_units", "seconds")
        if offsets_units not in {"seconds", "hours", "days"}:
            raise ValueError(
                "time_window 'offsets_units' must be one of: 'seconds', 'hours', 'days'. "
                f"Got '{offsets_units}'."
            )

        group_by = time_window.get("group_by", "variable")
        if group_by not in {"variable", "offset"}:
            raise ValueError(
                "time_window 'group_by' must be 'variable' or 'offset'. "
                f"Got '{group_by}'."
            )

        validated = {
            "offsets": offsets,
            "suffixes": suffixes,
            "offsets_units": offsets_units,
            "group_by": group_by,
        }
        if "description" in time_window:
            validated["description"] = time_window["description"]
        return validated

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
        grid_spacing_tolerance: float = 1e-5,
        grid_bounds_margin: float = 0.0,
    ) -> DiagnosticModel:
        """Load CorrDiff model from package.

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration
        grid_spacing_tolerance : float, optional
            Relative tolerance for checking regular grid spacing, by default 1e-5
        grid_bounds_margin : float, optional
            Fraction of input grid range to allow for extrapolation, by default 0.0

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
        time_window_metadata = metadata.get("time_window", None)

        base_variables = list(raw_input_variables)
        if time_window_metadata is not None:
            time_window = cls._validate_time_window_metadata(time_window_metadata)
            suffixes = list(time_window["suffixes"])
            suffixes_for_matching = sorted(
                [s for s in suffixes if s], key=len, reverse=True
            )
            input_variables = [
                f"{base}{suffix}" for base in base_variables for suffix in suffixes
            ]
            time_window = dict(time_window)
            time_window["base_variables"] = base_variables
        else:
            time_window = None
            input_variables = base_variables
            suffixes_for_matching = []

        # Load normalization statistics
        stats = cls._load_json_from_package(package, "stats.json")

        # Load input normalization parameters, sharing stats across suffixes
        in_center_values = []
        in_scale_values = []
        for var in input_variables:
            base = var
            if suffixes_for_matching:
                for suffix in suffixes_for_matching:
                    if var.endswith(suffix):
                        base = var[: -len(suffix)]
                        break
            if base not in stats["input"]:
                raise KeyError(
                    f"stats.json is missing normalization statistics for input variable '{base}'."
                )
            in_center_values.append(stats["input"][base]["mean"])
            in_scale_values.append(stats["input"][base]["std"])

        in_center = torch.Tensor(in_center_values)
        in_scale = torch.Tensor(in_scale_values)

        # Load output normalization parameters
        out_center = torch.Tensor(
            [stats["output"][v]["mean"] for v in output_variables]
        )
        out_scale = torch.Tensor([stats["output"][v]["std"] for v in output_variables])

        # Load output lat/lon grid
        with xr.open_dataset(package.resolve("output_latlon_grid.nc")) as ds:
            lat_output_grid = torch.Tensor(np.array(ds["lat"][:]))
            lon_output_grid = torch.Tensor(np.array(ds["lon"][:]))

            # Validate output grid format and ordering
            cls._validate_grid_format(
                lat_output_grid, lon_output_grid, grid_name="output"
            )

        # Load input lat/lon grid (or infer from metadata)
        try:
            with xr.open_dataset(package.resolve("input_latlon_grid.nc")) as ds:
                lat_input_grid = torch.Tensor(np.array(ds["lat"][:]))
                lon_input_grid = torch.Tensor(np.array(ds["lon"][:]))

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
                    (var_name, torch.Tensor(np.array(ds[var_name])))
                    for var_name in var_names
                )

                # Load invariant normalization parameters
                invariant_center = torch.Tensor(
                    [stats["invariants"][v]["mean"] for v in invariants]
                )
                invariant_scale = torch.Tensor(
                    [stats["invariants"][v]["std"] for v in invariants]
                )

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
            time_window=time_window,
            grid_spacing_tolerance=grid_spacing_tolerance,
            grid_bounds_margin=grid_bounds_margin,
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
        self, x: torch.Tensor, time: datetime | None = None
    ) -> torch.Tensor:
        """Complete input preprocessing pipeline.

        Performs interpolation to output grid, adds batch dimension,
        concatenates invariants if available, and normalizes the input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [C, H_in, W_in]
        time : datetime | None, optional
            Time information for time-dependent preprocessing (used by subclasses), by default None

        Returns
        -------
        torch.Tensor
            Preprocessed and normalized input tensor [1, C+C_inv, H_out, W_out]
        """
        # Interpolate input to output grid
        x = self._interpolate(x)

        # Add batch dimension
        x = x.view(1, *x.shape)

        # Concatenate invariants if available
        if self.invariants is not None:
            x = torch.concat([x, self.invariants.unsqueeze(0)], dim=1)

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
    def _forward(self, x: torch.Tensor, time: datetime | None = None) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        time : datetime | None, optional
            Time information for time-dependent preprocessing (used by subclasses), by default None

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
        # Base class ignores time parameter; subclasses can override preprocess_input to use it
        image_lr = self.preprocess_input(x, time)
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

        # Extract time information if present in coords
        time_array = coords.get("time", None)
        if time_array is not None:
            time_list = timearray_to_datetime(time_array)
        else:
            time_list = [None] * out.shape[0]

        for i in range(out.shape[0]):
            out[i] = self._forward(x[i], time_list[i])

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


class CorrDiffCMIP6(CorrDiff):
    """CorrDiff model variant for CMIP6 data.

    This class extends the base CorrDiff model to work with CMIP6 climate model data.
    It provides access to time information in preprocessing for time-dependent operations.

    Key differences from base CorrDiff:
    - Adds time-dependent features (solar zenith angle, hour of day) to inputs
    - Uses relaxed grid validation tolerances suitable for Gaussian grids (1% spacing, 5% bounds)
    - Overrides default interpolation
    - Includes time dimension in coordinate system
    """

    def __init__(
        self,
        *args: Any,
        time_feature_center: torch.Tensor | None = None,
        time_feature_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Extend in_center and in_scale to include time features (sza, hod) at the end
        if time_feature_center is not None and time_feature_scale is not None:
            # Reshape time features to match the 4D format [1, N, 1, 1]
            time_feature_center = time_feature_center.view(1, -1, 1, 1)
            time_feature_scale = time_feature_scale.view(1, -1, 1, 1)

            # Append time features after base variables and invariants
            self.in_center: torch.Tensor = torch.cat(
                [self.in_center, time_feature_center], dim=1
            )
            self.in_scale: torch.Tensor = torch.cat(
                [self.in_scale, time_feature_scale], dim=1
            )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Skip interpolation - CMIP6 data is already on target grid.

        The CMIP6 data is preprocessed to match the model's output grid,
        so no interpolation is needed. This override returns the input unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [C, H, W]

        Returns
        -------
        torch.Tensor
            Input tensor unchanged [C, H, W]
        """
        return x

    def input_coords(self) -> CoordSystem:
        """Get the input coordinate system for the model, including time dimension.

        Returns
        -------
        CoordSystem
            Dictionary containing the input coordinate system with time
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(
                    0
                ),  # Include time so it's preserved through batch compression
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
        # Validate input coordinates (adjusted indices for time dimension)
        target_input_coords = self.input_coords()
        handshake_dim(
            input_coords, "lon", 4
        )  # lon is now at index 4 (batch, time, variable, lat, lon)
        handshake_dim(input_coords, "lat", 3)  # lat is now at index 3
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_dim(input_coords, "variable", 2)  # variable is now at index 2
        handshake_coords(input_coords, target_input_coords, "variable")

        # Build output coords in the correct order: batch, time (if present), sample, variable, lat, lon
        output_coords = OrderedDict()
        output_coords["batch"] = input_coords["batch"]

        output_coords["time"] = input_coords["time"]
        output_coords["sample"] = np.arange(self.number_of_samples)
        output_coords["variable"] = np.array(self.output_variables)
        output_coords["lat"] = self.lat_output_numpy
        output_coords["lon"] = self.lon_output_numpy

        return output_coords

    def _inference_context(self) -> torch.autocast:
        """Return autocast context for float32 precision.

        Returns
        -------
        torch.autocast
            Autocast context manager to ensure float32 computation
        """
        return torch.autocast(device_type="cuda", dtype=torch.float32)

    def preprocess_input(
        self, x: torch.Tensor, time: datetime | None = None
    ) -> torch.Tensor:
        """Complete input preprocessing pipeline with optional time information.

        Performs interpolation to output grid, adds batch dimension,
        concatenates invariants if available, and normalizes the input.

        The time parameter can be used for time-dependent preprocessing operations
        such as seasonal adjustments, time-of-day encoding, etc.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [C, H_in, W_in]
        time : datetime | None, optional
            Time associated with this input for time-dependent preprocessing, by default None

        Returns
        -------
        torch.Tensor
            Preprocessed and normalized input tensor [1, C+C_inv, H_out, W_out]
        """
        channel_names = self.input_variables

        # Convert siconc to sai_cover (sea-air-ice cover) by combining with snow cover
        # Process all time steps (t-1, t, t+1)
        kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0  # 3x3 averaging kernel

        for suffix in ["t-1", "t", "t+1"]:
            siconc_index = self.input_variables.index(f"siconc_{suffix}")
            snc_index = self.input_variables.index(f"snc_{suffix}")

            # Get sea ice and snow cover, replacing NaN with 0
            siconc = torch.nan_to_num(x[:, siconc_index], nan=0.0)
            snc = torch.nan_to_num(x[:, snc_index], nan=0.0)

            # Combine and clip to [0, 100] range
            sai_cover = torch.clip(siconc + snc, 0.0, 100.0)

            # Apply spatial smoothing with 3x3 averaging
            # Pad: circular for longitude (wraps around), replicate for latitude
            sai_cover_pad = F.pad(sai_cover, (1, 1), mode="circular")
            sai_cover_pad = F.pad(sai_cover_pad, (0, 0, 1, 1), mode="replicate")
            sai_cover_smooth = F.conv2d(
                sai_cover_pad.unsqueeze(1), kernel, padding="valid"
            ).squeeze(1)

            # Replace siconc channel with smoothed sai_cover
            x[:, siconc_index] = sai_cover_smooth

        # Interpolate input to output grid
        x = F.interpolate(x, self.img_shape, mode="bilinear")

        # Concatenate invariants if available
        if self.invariants is not None:
            x = torch.concat([x, self.invariants.unsqueeze(0)], dim=1)
            channel_names = channel_names + self.invariant_variables

        # Add batch dimension
        # x = x.view(1, *x.shape)

        # Add time-dependent features AFTER invariants and normalization
        if time is not None:
            # Create meshgrid for lat/lon (cos_zenith_angle expects 2D arrays)
            lon_grid, lat_grid = np.meshgrid(
                self.lon_output_numpy, self.lat_output_numpy
            )

            # Compute cosine of solar zenith angle
            cos_sza = cos_zenith_angle(time, lon_grid, lat_grid).astype(np.float32)
            cos_sza_tensor = (
                torch.from_numpy(cos_sza).unsqueeze(0).unsqueeze(0).to(x.device)
            )
            x = torch.concat([x, cos_sza_tensor], dim=1)
            channel_names = channel_names + ["sza"]

            # Add hour of day as a channel
            hour_tensor = torch.full_like(cos_sza_tensor, float(time.hour))
            x = torch.concat([x, hour_tensor], dim=1)
            channel_names = channel_names + ["hod"]

        # Normalize input (base variables + invariants + time features)
        # The in_center and in_scale have been extended to include time features
        x = self.normalize_input(x)
        x = torch.flip(x, [2])
        x = F.pad(x, (0, 0, 23, 24), mode="reflect")
        x = F.pad(x, (48, 48, 0, 0), mode="circular")

        # Reorder channels: current order is [input_vars, invariants, sza, hod]
        # Required order is: [input_vars, sza, invariants, hod]
        num_input = len(self.input_variables)
        num_inv = len(self.invariant_variables)

        # Build index list for reordering
        indices = (
            list(range(num_input))
            + [num_input + num_inv]  # input variables
            + list(  # sza (currently after invariants)
                range(num_input, num_input + num_inv)
            )
            + [num_input + num_inv + 1]  # invariants  # hod (currently at the end)
        )
        x = x[:, indices]

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
        # crop to original extent
        x = x[:, :, 23:-24, 48:-48]

        # denormalize
        x = x * self.out_scale + self.out_center

        var_limited_0 = [
            "t2m",
            "sp",
            "msl",
            "tcwv",
            "z50",
            "z100",
            "z150",
            "z200",
            "z250",
            "z300",
            "z400",
            "z500",
            "z600",
            "z700",
            "z850",
            "z925",
            "z1000",
            "t50",
            "t100",
            "t150",
            "t200",
            "t250",
            "t300",
            "t400",
            "t500",
            "t600",
            "t700",
            "t850",
            "t925",
            "t1000",
            "q50",
            "q100",
            "q150",
            "q200",
            "q250",
            "q300",
            "q400",
            "q500",
            "q600",
            "q700",
            "q850",
            "q925",
            "q1000",
            "sst",
            "d2m",
        ]
        # identify indices of var_limited_0 in self.output_variables
        indices = [i for i, v in enumerate(self.output_variables) if v in var_limited_0]

        # clip x to enforce lower bound of 0 along the channel dimension
        x[:, indices] = x[:, indices].clamp(min=0)

        # Flip along the latitude dimension to match expected output ordering
        # Model outputs in ascending lat order (South to North), but we need descending (North to South)
        x = torch.flip(x, [2])

        return x

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load CorrDiffCMIP6 model from package with time feature normalization.

        This method extends the base CorrDiff loading to include time feature
        normalization parameters (sza and hod) from stats.json.

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration

        Returns
        -------
        DiagnosticModel
            Initialized CorrDiffCMIP6 model
        """
        # Load stats to get time feature normalization parameters
        stats = CorrDiff._load_json_from_package(package, "stats.json")

        time_feature_center = torch.Tensor(
            [stats["input"]["sza"]["mean"], stats["input"]["hod"]["mean"]]
        )
        time_feature_scale = torch.Tensor(
            [stats["input"]["sza"]["std"], stats["input"]["hod"]["std"]]
        )

        # Load base model with CMIP6-specific grid tolerances
        # Use 1% spacing tolerance for Gaussian grids and 5% bounds margin for pole extrapolation
        model = CorrDiff.load_model.__func__(
            CorrDiff, package, grid_spacing_tolerance=0.01, grid_bounds_margin=0.05
        )

        # Now convert it to CorrDiffCMIP6 by recreating with time feature params
        # Reuse the stored invariants dict from the base model
        # Flatten normalization tensors back to 1D for __init__ (they're 4D buffers in the loaded model)
        num_base_vars = len(model.input_variables)
        in_center_flat = model.in_center.squeeze()[
            :num_base_vars
        ]  # Only base variables, exclude invariants
        in_scale_flat = model.in_scale.squeeze()[:num_base_vars]

        invariant_center_flat = None
        invariant_scale_flat = None
        if model.invariants_dict is not None:
            num_invariants = len(model.invariants_dict)
            invariant_center_flat = model.in_center.squeeze()[
                num_base_vars : num_base_vars + num_invariants
            ]
            invariant_scale_flat = model.in_scale.squeeze()[
                num_base_vars : num_base_vars + num_invariants
            ]

        return cls(
            input_variables=model.input_variables,
            output_variables=model.output_variables,
            residual_model=model.residual_model,
            regression_model=model.regression_model,
            lat_input_grid=model.lat_input_grid,
            lon_input_grid=model.lon_input_grid,
            lat_output_grid=model.lat_output_grid,
            lon_output_grid=model.lon_output_grid,
            in_center=in_center_flat,
            in_scale=in_scale_flat,
            invariants=model.invariants_dict,
            invariant_center=invariant_center_flat,
            invariant_scale=invariant_scale_flat,
            out_center=model.out_center.squeeze(),
            out_scale=model.out_scale.squeeze(),
            number_of_samples=model.number_of_samples,
            number_of_steps=model.number_of_steps,
            solver=model.solver,
            sampler_type=model.sampler_type,
            inference_mode=model.inference_mode,
            hr_mean_conditioning=model.hr_mean_conditioning,
            seed=model.seed,
            time_window=model.time_window,
            time_feature_center=time_feature_center,
            time_feature_scale=time_feature_scale,
            grid_spacing_tolerance=0.01,
            grid_bounds_margin=0.05,
        )


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
