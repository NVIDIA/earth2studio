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
from collections.abc import Generator

import numpy as np
import pandas as pd
import torch
import xarray as xr

try:
    import cupy as cp
except ImportError:
    cp = None

from earth2studio.models.da.base import AssimilationInput
from earth2studio.models.da.utils import (
    validate_input,
    validate_observation_fields,
    validate_observation_type,
)
from earth2studio.utils.type import CoordSystem


class Interp(torch.nn.Module):
    """Interpolation assimilation model that interpolates observations to a lat-lon grid.

    This is a sample implementation of an AssimilationModel that:
    - Accepts DataFrames with observations (time, lat, lon, observation, variable)
    - Interpolates observations to a regular lat-lon grid using specified method
    - Validates input observations against schema constraints
    - Supports multiple interpolation methods: 'nearest', 'linear', 'cubic'

    Parameters
    ----------
    lat : np.ndarray | None, optional
        Latitude coordinates for output grid, by default None (uses default grid)
    lon : np.ndarray | None, optional
        Longitude coordinates for output grid, by default None (uses default grid)
    interpolation_type : str, optional
        Interpolation method to use: 'nearest', 'linear', or 'cubic', by default "linear"

    Raises
    ------
    ValueError
        If interpolation_type is not one of the supported methods
    """

    # Acceptable variables for this model
    VARIABLES = ["t2m", "u10m", "v10m", "sp"]

    def __init__(
        self,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        interpolation_type: str = "linear",
    ) -> None:
        if interpolation_type not in ["nearest", "linear", "cubic"]:
            raise ValueError(
                f"interpolation_type must be one of ['nearest', 'linear', 'cubic'], "
                f"got {interpolation_type}"
            )

        self._lat = (
            lat if lat is not None else np.linspace(25.0, 50.0, 101, dtype=np.float32)
        )
        self._lon = (
            lon if lon is not None else np.linspace(235.0, 295.0, 241, dtype=np.float32)
        )
        self._interpolation_type = interpolation_type
        self.register_buffer("device_buffer", torch.empty(0), persistent=False)

    def input_coords(self) -> tuple[CoordSystem]:
        """Input coordinate system specifying required DataFrame fields.

        Returns
        -------
        tuple[CoordSystem, ...]
            Tuple containing coordinate system dictionary with field names as keys:
            - variable: array of acceptable variable values
            - time, lat, lon, observation: empty arrays (dynamic dimensions)
        """
        return (
            CoordSystem(
                {
                    "time": np.empty(0, dtype="datetime64[ns]"),
                    "lat": np.empty(0, dtype=np.float32),
                    "lon": np.empty(0, dtype=np.float32),
                    "observation": np.empty(0, dtype=np.float32),
                    "variable": np.array(self.VARIABLES, dtype=str),
                }
            ),
        )

    def output_coords(
        self, input_coords: tuple[CoordSystem], x: AssimilationInput
    ) -> CoordSystem:
        """Output coordinate system for assimilated data.

        Parameters
        ----------
        input_coords : tuple[CoordSystem, ...]
            Input coordinate system (CoordSystem for DataFrame input)
        x : AssimilationInput
            Input configuration for the assimilation model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary with time, variable, lat, and lon dimensions

        Raises
        ------
        ValueError
            If input_coords are not valid
        """
        # Extract variables from first input coord system
        if len(input_coords) > 0 and "variable" in input_coords[0]:
            variables = input_coords[0]["variable"]
        else:
            variables = self.VARIABLES

        return CoordSystem(
            {
                "time": x.time,
                "variable": np.array(variables, dtype=str),
                "lat": self._lat,
                "lon": self._lon,
            }
        )

    def __call__(
        self,
        x: AssimilationInput,
    ) -> Generator[
        tuple[pd.DataFrame | xr.DataArray, ...],
        tuple[pd.DataFrame | xr.DataArray, ...],
        None,
    ]:
        """Creates a generator which accepts collection of input observations and
        outputs a collection of assimilated data.

        Parameters
        ----------
        x : AssimilationInput
            Input configuration for the assimilation model

        Yields
        ------
        tuple[*pd.DataFrame | xr.DataArray]
            Generator that yields assimilated data as tuples of pandas DataFrames
            or xarray DataArrays.
        """
        # Validate input configuration
        validate_input(x, required_fields=["time", "lead_time"])
        input_coords = self.input_coords()
        output_coords = self.output_coords(input_coords, x)

        # Initialize generator - first yield is None to start
        observations = yield None  # type: ignore[misc]
        while True:
            # Validate observation types match input_coords
            if not isinstance(observations, tuple):
                observations = (observations,)
            validate_observation_type(observations, input_coords)

            assimilated_data = []

            for obs, coords in zip(observations, input_coords):
                # Validate required fields are present
                validate_observation_fields(obs, required_fields=list(coords.keys()))

                # Interpolate observations to grid
                da = self._interpolate_dataframe(obs, output_coords)
                assimilated_data.append(da)

            # Yield assimilated data and receive next observations
            observations = yield tuple(assimilated_data)

    def _interpolate_dataframe(
        self, df: pd.DataFrame, output_coords: CoordSystem
    ) -> xr.DataArray:
        """Interpolate DataFrame observations to a regular lat-lon grid.

        Parameters
        ----------
        df : pd.DataFrame
            Input observations DataFrame with columns: time, lat, lon, observation, variable
        output_coords : CoordSystem
            Output coordinate system with time, variable, lat, lon dimensions

        Returns
        -------
        xr.DataArray
            Interpolated data array on the output grid
        """
        variables = output_coords["variable"]
        lat_grid = output_coords["lat"]
        lon_grid = output_coords["lon"]
        time_coords = output_coords["time"]

        # Initialize output array: (time, variable, lat, lon)
        n_time = len(time_coords)
        n_var = len(variables)
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)

        # Convert target grid to torch tensors and create meshgrid
        # Get device from model if it has parameters, otherwise use CPU
        device = self.device_buffer.device
        lat_grid_t = torch.tensor(lat_grid, dtype=torch.float32, device=device)
        lon_grid_t = torch.tensor(lon_grid, dtype=torch.float32, device=device)

        # Create meshgrid using torch
        lon_mesh_t, lat_mesh_t = torch.meshgrid(lon_grid_t, lat_grid_t, indexing="ij")
        # Stack and reshape to [n_lat * n_lon, 2] with (lat, lon) pairs
        target_points_t = torch.stack(
            [lat_mesh_t.T.ravel(), lon_mesh_t.T.ravel()], dim=1
        )

        # Initialize output tensor
        interpolated_data = torch.full(
            (n_time, n_var, n_lat, n_lon),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        # Group observations by variable and interpolate each
        for var_idx, variable in enumerate(variables):
            var_df = df[df["variable"] == variable].copy()

            if len(var_df) == 0:
                # No observations for this variable, leave as NaN
                continue

            # Extract observation points and values, convert to torch
            source_lat = torch.tensor(
                var_df["lat"].values, dtype=torch.float32, device=device
            )
            source_lon = torch.tensor(
                var_df["lon"].values, dtype=torch.float32, device=device
            )
            source_points_t = torch.stack([source_lat, source_lon], dim=1)
            values_t = torch.tensor(
                var_df["observation"].values, dtype=torch.float32, device=device
            )

            # Interpolate to grid using PyTorch
            grid_values = self._interpolate_scattered(
                source_points_t, values_t, target_points_t, n_lat, n_lon
            )

            # Assign to all time steps
            for t_idx in range(n_time):
                interpolated_data[t_idx, var_idx, :, :] = grid_values

        # Create DataArray (convert tensor to numpy or cupy based on device)
        if device.type == "cuda" and cp is not None:
            # Convert to cupy array if on CUDA
            data_array = cp.asarray(interpolated_data)
        else:
            # Convert to numpy array if on CPU or cupy not available
            data_array = interpolated_data.cpu().numpy()

        da = xr.DataArray(
            data=data_array,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time_coords,
                "variable": variables,
                "lat": lat_grid,
                "lon": lon_grid,
            },
        )

        return da

    def _interpolate_scattered(
        self,
        source_points: torch.Tensor,
        values: torch.Tensor,
        target_points: torch.Tensor,
        n_lat: int,
        n_lon: int,
    ) -> torch.Tensor:
        """Interpolate scattered points to target grid using PyTorch.

        Parameters
        ----------
        source_points : torch.Tensor
            Source points [N, 2] with (lat, lon) coordinates
        values : torch.Tensor
            Values at source points [N]
        target_points : torch.Tensor
            Target grid points [M, 2] with (lat, lon) coordinates
        n_lat : int
            Number of latitude points in output grid
        n_lon : int
            Number of longitude points in output grid

        Returns
        -------
        torch.Tensor
            Interpolated values [n_lat, n_lon]
        """
        if self._interpolation_type == "nearest":
            # Nearest neighbor interpolation using distance
            # Compute distances: [M, N]
            distances = torch.cdist(target_points, source_points, p=2)
            # Find nearest neighbor indices
            nearest_indices = torch.argmin(distances, dim=1)
            # Get values at nearest neighbors
            interpolated = values[nearest_indices]
        elif self._interpolation_type == "linear":
            # Inverse distance weighting for linear-like interpolation
            # Compute distances: [M, N]
            distances = torch.cdist(target_points, source_points, p=2)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            weights = 1.0 / (distances + epsilon)
            # Normalize weights
            weights = weights / weights.sum(dim=1, keepdim=True)
            # Weighted average
            interpolated = (weights @ values.unsqueeze(1)).squeeze(1)
        elif self._interpolation_type == "cubic":
            # Inverse distance weighting with cubic power for smoother interpolation
            distances = torch.cdist(target_points, source_points, p=2)
            epsilon = 1e-10
            # Use cubic weighting (1/d^3)
            weights = 1.0 / (distances**3 + epsilon)
            weights = weights / weights.sum(dim=1, keepdim=True)
            interpolated = (weights @ values.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(
                f"Unsupported interpolation type: {self._interpolation_type}"
            )

        # Reshape to grid
        return interpolated.reshape(n_lat, n_lon)
