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
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.da.utils import (
    dfseries_to_torch,
    filter_time_range,
    validate_observation_fields,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeTolerance

try:
    import cupy as cp
except ImportError:
    OptionalDependencyFailure("gpu")
    cp = None  # type: ignore[assignment]


@check_optional_dependencies()
class InterpEquirectangular(torch.nn.Module):
    """Interpolation assimilation model that interpolates sparse observations to a
    lat-lon grid.

    This is a sample implementation of an AssimilationModel that:
    - Accepts DataFrames with observations (time, lat, lon, observation, variable)
    - Interpolates observations to a regular lat-lon grid using specified method
    - Validates input observations against schema constraints
    - Supports multiple interpolation methods: 'nearest', 'linear', 'cubic'

    Parameters
    ----------
    lat : np.ndarray | None, optional
        Latitude coordinates for output grid, by default None (uses default grid over
        CONUS)
    lon : np.ndarray | None, optional
        Longitude coordinates for output grid, by default None (uses default
        grid over CONUS)
    interp_method : str, optional
        Interpolation method to use: 'nearest', 'linear', or 'cubic', by default "linear"
    tolerance : TimeTolerance, optional
        Time tolerance for filtering observations. Observations within the tolerance
        window around each requested time will be used for interpolation, by default
        np.timedelta64(10, "m")

    Raises
    ------
    ValueError
        If interp_method is not one of the supported methods
    """

    # Acceptable variables for this model
    VARIABLES = ["t2m", "u10m", "v10m", "sp"]

    def __init__(
        self,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        interp_method: str = "linear",
        tolerance: TimeTolerance = np.timedelta64(10, "m"),
    ) -> None:
        if interp_method not in ["nearest", "linear", "cubic"]:
            raise ValueError(
                f"interp_method must be one of ['nearest', 'linear', 'cubic'], "
                f"got {interp_method}"
            )
        super().__init__()
        self._lat = (
            lat if lat is not None else np.linspace(25.0, 50.0, 101, dtype=np.float32)
        )
        self._lon = (
            lon if lon is not None else np.linspace(235.0, 295.0, 241, dtype=np.float32)
        )
        self.interp_method = interp_method
        self._tolerance = normalize_time_tolerance(tolerance)
        self.register_buffer("device_buffer", torch.empty(0), persistent=False)

    def input_coords(self) -> tuple[FrameSchema]:
        """Input coordinate system specifying required DataFrame fields.

        Returns
        -------
         tuple[FrameSchema]
            Tuple containing coordinate system dictionary with field names as keys:
            - variable: array of acceptable variable values
            - time, lat, lon, observation: empty arrays (dynamic dimensions)
        """
        return (
            FrameSchema(
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
        self,
        input_coords: tuple[CoordSystem],
        request_time: np.ndarray,
        **kwargs: Any,
    ) -> tuple[CoordSystem]:
        """Output coordinate system for assimilated data.

        Parameters
        ----------
        input_coords : tuple[CoordSystem, ...]
            Input coordinate system (CoordSystem for DataFrame input)

        Returns
        -------
        tuple[CoordSystem]
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

        return (
            CoordSystem(
                {
                    "time": request_time,
                    "variable": np.array(variables, dtype=str),
                    "lat": self._lat,
                    "lon": self._lon,
                }
            ),
        )

    def __call__(self, x: pd.DataFrame) -> xr.DataArray:
        """Stateless forward pass"""
        input_coords = self.input_coords()
        (output_coords,) = self.output_coords(input_coords, **x.attrs)
        # Validate observation types match input_coords
        validate_observation_fields(x, required_fields=list(input_coords[0].keys()))
        return self._interpolate_dataframe(x, output_coords)

    def create_generator(self) -> Generator[
        xr.DataArray,
        pd.DataFrame,
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
        xr.DataArray
            Assimilated data output the generator yields

        Receives
        --------
        *DataFrame
            Observations sent via generator.send()
        """
        # Validate input configuration

        # Initialize generator - first yield is None to start
        observations = yield None  # type: ignore[misc]
        try:
            while True:
                da = self.__call__(observations)
                observations = yield da
        except GeneratorExit:
            logger.debug("InterpEquirectangular clean up complete.")

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

        # Process each time step separately
        for t_idx, request_time in enumerate(time_coords):
            # Filter observations within tolerance window for this time step
            time_filtered_df = filter_time_range(
                df, request_time, self._tolerance, time_column="time"
            )

            # Group observations by variable and interpolate each
            for var_idx, variable in enumerate(variables):
                var_df = time_filtered_df[
                    time_filtered_df["variable"] == variable
                ].copy()

                if len(var_df) == 0:
                    # No observations for this variable at this time, leave as NaN
                    continue

                # Extract observation points and values, convert to torch
                source_lat = dfseries_to_torch(
                    var_df["lat"], dtype=torch.float32, device=device
                )
                source_lon = dfseries_to_torch(
                    var_df["lon"], dtype=torch.float32, device=device
                )
                source_points_t = torch.stack([source_lat, source_lon], dim=1)
                values_t = dfseries_to_torch(
                    var_df["observation"], dtype=torch.float32, device=device
                )

                # Interpolate to grid using PyTorch
                grid_values = self._interpolate_scattered(
                    source_points_t, values_t, target_points_t, n_lat, n_lon
                )

                # Assign to this time step
                interpolated_data[t_idx, var_idx, :, :] = grid_values

        # Create DataArray (convert tensor to numpy or cupy based on device)
        if device.type == "cuda" and cp is not None:
            data_array = cp.asarray(interpolated_data)  # Zero-copy
        else:
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

        Accounts for periodic boundary condition in longitude.
        Automatically converts longitude from -180,180 range to 0,360 range if needed.

        Parameters
        ----------
        source_points : torch.Tensor
            Source points [N, 2] with (lat, lon) coordinates. Longitude can be in
            -180,180 or 0,360 range.
        values : torch.Tensor
            Values at source points [N]
        target_points : torch.Tensor
            Target grid points [M, 2] with (lat, lon) coordinates. Longitude can be in
            -180,180 or 0,360 range.
        n_lat : int
            Number of latitude points in output grid
        n_lon : int
            Number of longitude points in output grid

        Returns
        -------
        torch.Tensor
            Interpolated values [n_lat, n_lon]
        """
        target_lat = target_points[:, 0:1]  # [M, 1]
        target_lon = target_points[:, 1:2]  # [M, 1]
        source_lat = source_points[:, 0:1].T  # [1, N]
        source_lon = source_points[:, 1:2].T  # [1, N]

        # Normalize longitude from -180,180 to 0,360 range
        # Convert negative longitudes to positive: -180 -> 180, -90 -> 270, etc.
        target_lon = torch.where(target_lon < 0, target_lon + 360.0, target_lon)
        source_lon = torch.where(source_lon < 0, source_lon + 360.0, source_lon)

        # Latitude distance (not periodic)
        lat_diff = target_lat - source_lat  # [M, N]
        lat_dist_sq = lat_diff**2

        # Longitude distance with periodic boundary (0-360)
        lon_diff = target_lon - source_lon  # [M, N]
        # Compute periodic distance: min(|diff|, 360 - |diff|)
        lon_diff_abs = torch.abs(lon_diff)
        lon_dist = torch.minimum(lon_diff_abs, 360.0 - lon_diff_abs)
        lon_dist_sq = lon_dist**2
        distances = torch.sqrt(lat_dist_sq + lon_dist_sq)

        if self.interp_method == "nearest":
            # Nearest neighbor interpolation using distance
            nearest_indices = torch.argmin(distances, dim=1)
            interpolated = values[nearest_indices]
        elif self.interp_method == "linear":
            # Inverse distance weighting for linear-like interpolation
            epsilon = 1e-10
            weights = 1.0 / (distances + epsilon)
            weights = weights / weights.sum(dim=1, keepdim=True)
            interpolated = (weights @ values.unsqueeze(1)).squeeze(1)
        elif self.interp_method == "cubic":
            # Inverse distance weighting with cubic power for smoother interpolation
            epsilon = 1e-10
            weights = 1.0 / (distances**3 + epsilon)
            weights = weights / weights.sum(dim=1, keepdim=True)
            interpolated = (weights @ values.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unsupported interpolation type: {self.interp_method}")

        # Reshape to grid
        return interpolated.reshape(n_lat, n_lon)
