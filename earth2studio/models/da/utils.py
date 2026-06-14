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
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from functools import reduce
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger

from earth2studio.utils.type import TimeArray

try:
    import cudf
except ImportError:
    cudf = None

try:
    import cupy as cp
    from cupyx.scipy.interpolate import LinearNDInterpolator
except ImportError:
    cp = None
    from scipy.interpolate import LinearNDInterpolator


def validate_observation_fields(
    observation: pd.DataFrame | cudf.DataFrame, required_fields: list[str]
) -> None:
    """Validate that required fields are present as columns in the DataFrame.

    Parameters
    ----------
    observation : pd.DataFrame | cudf.DataFrame
        DataFrame observation to validate
    required_fields : list[str]
        List of required field/column names

    Raises
    ------
    ValueError
        If any required fields are missing from the DataFrame columns
    """
    missing_fields = [
        field for field in required_fields if field not in observation.columns
    ]
    if missing_fields:
        raise ValueError(
            f"DataFrame missing required fields: {missing_fields}. "
            f"Available columns: {list(observation.columns)}"
        )


def filter_time_range(
    df: pd.DataFrame | cudf.DataFrame,
    request_time: np.datetime64 | datetime | str | TimeArray,
    tolerance: tuple[np.timedelta64, np.timedelta64],
    time_column: str = "time",
) -> pd.DataFrame | cudf.DataFrame:
    """Filter DataFrame rows where time column is within the specified tolerance range.

    Filters the DataFrame to include only rows where the time column value is within
    [request_time + lower_bound, request_time + upper_bound]. When *request_time* is a
    :class:`~numpy.ndarray` of datetime64 values, a row is kept if it falls within the
    tolerance window of **any** of the provided times.

    Parameters
    ----------
    df : pd.DataFrame | cudf.DataFrame
        DataFrame to filter. Can be pandas or cudf DataFrame.
    request_time : np.datetime64 | datetime | str | TimeArray
        Reference time(s) for filtering. Observations within the tolerance window
        around this time will be included. If a numpy array of datetime64 is provided,
        the union of all individual tolerance windows is used.
    tolerance : tuple[np.timedelta64, np.timedelta64]
        Tuple of (lower_bound, upper_bound) time deltas defining the tolerance window.
    time_column : str, optional
        Name of the time column in the DataFrame, by default "time"

    Returns
    -------
    pd.DataFrame | cudf.DataFrame
        Filtered DataFrame containing only rows within the time tolerance range.
        Returns the same DataFrame type as the input (pandas or cudf).

    Raises
    ------
    KeyError
        If the time_column is not present in the DataFrame
    """
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure time column is datetime64[ns]
    time_series = df[time_column]
    if time_series.dtype != "datetime64[ns]":
        df = df.copy()
        # Use cudf methods if it's a cudf DataFrame, otherwise use pandas
        if cudf is not None and isinstance(df, cudf.DataFrame):
            df[time_column] = cudf.to_datetime(time_series).astype("datetime64[ns]")
        else:
            df[time_column] = pd.to_datetime(time_series).astype("datetime64[ns]")
    # Ensure request_time is datetime64[ns] array
    if isinstance(request_time, np.ndarray) and request_time.ndim >= 1:
        times_ns = request_time.astype("datetime64[ns]")
    else:
        times_ns = np.array([np.datetime64(request_time, "ns")])

    lower_bound, upper_bound = tolerance
    masks = [
        (df[time_column] >= t + lower_bound) & (df[time_column] <= t + upper_bound)
        for t in times_ns
    ]
    time_mask = reduce(lambda a, b: a | b, masks)
    return df[time_mask]


def dfseries_to_torch(
    series: pd.Series | cudf.Series,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a DataFrame series to a torch tensor with zero-copy for cudf.

    If the series is from a cudf DataFrame, uses dlpack for zero-copy transfer.
    If the series is from pandas but target device is GPU, transfers the data
    and warns that cudf is not being used.

    Parameters
    ----------
    series : pd.Series | cudf.Series
        Series to convert to torch tensor. Can be pandas or cudf Series.
    dtype : torch.dtype, optional
        Desired dtype for the tensor, by default torch.float32
    device : torch.device | str, optional
        Target device for the tensor, by default "cpu"

    Returns
    -------
    torch.Tensor
        Torch tensor with the series data on the specified device

    Raises
    ------
    ImportError
        If cudf is required but not available
    """
    device = torch.device(device)

    # Check if series is from cudf
    if cudf is not None and isinstance(series, cudf.Series):
        # Use dlpack for zero-copy transfer from cudf to torch
        return torch.from_dlpack(series.values).to(dtype=dtype, device=device)

    # Handle pandas Series
    if device.type == "cuda":
        # Warn that cudf is not being used for GPU transfer
        logger.warning(
            "Converting pandas Series to GPU tensor. Consider installing cudf "
            "for zero-copy transfer and better performance."
        )
        return torch.tensor(series.values, dtype=dtype, device=device)

    # CPU case - standard conversion
    return torch.tensor(series.values, dtype=dtype, device=device)


class ObsGridMapping:
    """Maps point observations to and from a 2-D model grid.

    Supports regular (1-D lat/lon axes) and irregular (2-D lat/lon) grids.
    CuPy is used automatically when available and the target device is a GPU.

    Parameters
    ----------
    grid_variables : np.ndarray
        Ordered array of variable names that define the channel dimension.
    grid_lat : np.ndarray or torch.Tensor
        Latitude coordinates. Shape ``(H,)`` for a regular grid or
        ``(H, W)`` for an irregular grid.
    grid_lon : np.ndarray or torch.Tensor
        Longitude coordinates. Shape ``(W,)`` for a regular grid or
        ``(H, W)`` for an irregular grid. Values are expected in [0, 360).
    device : torch.device, str, or None, optional
        Target device. Inferred from ``grid_lat`` when it is a
        ``torch.Tensor``; defaults to CPU otherwise.

    Raises
    ------
    ValueError
        If ``grid_lat`` is neither 1-D nor 2-D.
    """

    def __init__(
        self,
        grid_variables: np.ndarray,
        grid_lat: np.ndarray | torch.Tensor,
        grid_lon: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None
    ):
        if device is None:
            d = getattr(grid_lat, "device", None)
            self.device = torch.device(d) if d is not None else torch.device("cpu")
        else:
            self.device = torch.device(device)
        use_cupy = (cp is not None) and (self.device.type != "cpu")
        device_index = self.device.index if self.device.index is not None else 0
        self.cupy_context = cp.cuda.Device(device_index) if use_cupy else nullcontext()
        self.xp = cp if use_cupy else np

        with self.cupy_context:
            self.grid_lat = self.xp.asarray(grid_lat)
            self.grid_lon = self.xp.asarray(grid_lon)

            self.C = grid_variables.shape[0]
            self.H = grid_lat.shape[0]
            self.W = grid_lon.shape[-1]  # works for either 1D or 2D lat/lon
            self.grid_shape = (self.C, self.H, self.W)

            self.var_to_idx = {str(v): i for i, v in enumerate(grid_variables)}
            self.grid_type: Literal["regular", "irregular"] = "regular"
            if grid_lat.ndim == 2:
                self.grid_type = "irregular"
                (grid_i, grid_j) = self.xp.mgrid[: self.grid_lat.shape[0], : self.grid_lat.shape[1]]
                in_points = self.xp.stack((self.grid_lat.ravel(), self.grid_lon.ravel()), axis=-1)
                self.i_interp = LinearNDInterpolator(in_points, grid_i.ravel())
                self.j_interp = LinearNDInterpolator(in_points, grid_j.ravel())
            elif grid_lat.ndim != 1:
                raise ValueError("grid_lat and grid_lon must be either 1D or 2D.")

    def obs_coords(
        self,
        obs_var: np.ndarray,
        obs_lat: np.ndarray,
        obs_lon: np.ndarray,
        remove_out_of_bounds: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert observation locations to fractional grid indices.

        Parameters
        ----------
        obs_var : np.ndarray
            Variable name for each observation.
        obs_lat : np.ndarray
            Latitude of each observation in degrees.
        obs_lon : np.ndarray
            Longitude of each observation in degrees, in [0, 360).
        remove_out_of_bounds : bool, optional
            When ``True`` (default), observations outside the grid or with an
            unknown variable are removed from the returned arrays.  The full
            boolean mask is always returned as the fourth element.

        Returns
        -------
        obs_c : torch.Tensor
            Channel index for each (retained) observation.
        obs_i : torch.Tensor
            Fractional row index in the grid for each observation.
        obs_j : torch.Tensor
            Fractional column index in the grid for each observation.
        obs_in_bounds : torch.Tensor
            Boolean mask of length ``N`` indicating which of the original
            observations are in-bounds and have a known variable.
        """
        with self.cupy_context:
            obs_lat = self.xp.asarray(obs_lat)
            obs_lon = self.xp.asarray(obs_lon)
            obs_c = self.xp.asarray([self.var_to_idx.get(str(v), -1) for v in obs_var])

            if self.grid_type == "regular":
                obs_i = (obs_lat - self.grid_lat[0]) / (self.grid_lat[1] - self.grid_lat[0])
                obs_j = (obs_lon - self.grid_lon[0]) / (self.grid_lon[1] - self.grid_lon[0])
            else:
                obs_latlon = self.xp.stack((obs_lat.ravel(), obs_lon.ravel()), axis=-1)
                obs_i = self.i_interp(obs_latlon)
                obs_j = self.j_interp(obs_latlon)

            obs_in_bounds = (obs_c != -1) & (obs_i >= 0) & (obs_i <= self.H - 1) & (obs_j >= 0) & (obs_j <= self.W - 1)
            if remove_out_of_bounds:
                obs_c = obs_c[obs_in_bounds]
                obs_i = obs_i[obs_in_bounds]
                obs_j = obs_j[obs_in_bounds]

            return (
                torch.as_tensor(obs_c, device=self.device),
                torch.as_tensor(obs_i, device=self.device),
                torch.as_tensor(obs_j, device=self.device),
                torch.as_tensor(obs_in_bounds, device=self.device)
            )

    def obs_to_grid(
        self,
        obs: pd.DataFrame | None,
        variables: list[str] | None = None,
        request_time: np.datetime64 | None = None,
        time_tolerance: np.timedelta64 | tuple[np.timedelta64, np.timedelta64] | None = None,
        return_empty_grid: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Bin point observations onto the model grid.

        Observations are filtered to ``variables`` and to those within
        ``time_tolerance`` of ``request_time``.  Multiple observations that
        fall in the same grid cell are averaged.

        Parameters
        ----------
        obs : pd.DataFrame or None
            Observation table with columns ``variable``, ``lat``, ``lon``, ``observation``. 
            If ``request_time`` is used, ``obs`` must also have a ``time`` column.
        variables : list[str], optional
            Variable names to retain; others are discarded. ``None`` (default) uses all variables.
        request_time : np.datetime64, optional
            Target valid time for the observations. ``None`` (default) skips time filtering.
        time_tolerance : np.timedelta64, tuple[np.timedelta64, np.timedelta64], optional
            Time window used to filter observations around ``request_time``.
            A single ``np.timedelta64`` creates a symmetric window
            ``[-time_tolerance, +time_tolerance]``.  A 2-tuple
            ``(lower, upper)`` is passed directly to ``filter_time_range``
            as an asymmetric window. ``None`` (default) skips time filtering.
        return_empty_grid : bool, optional
            When ``True``, return zero-filled tensors instead of
            ``(None, None)`` when no observations are gridded.
            Defaults to ``False``.

        Returns
        -------
        y_obs : torch.Tensor or None
            Gridded observations, shape ``(1, C, H, W)``, or ``None`` when no
            observations remain and ``return_empty_grid`` is ``False``.
        mask : torch.Tensor or None
            Observation count per cell (float), same shape as ``y_obs``, or
            ``None`` under the same condition as above.
        """
        def _init_obs_mask():
            y_obs = torch.zeros(1, *self.grid_shape, device=self.device, dtype=torch.float32)
            mask = torch.zeros(1, *self.grid_shape, device=self.device, dtype=torch.float32)
            return (y_obs, mask)

        if obs is None or len(obs) == 0:
            return _init_obs_mask() if return_empty_grid else (None, None)

        if variables is not None:
            obs = obs[obs["variable"].isin(variables)]
        if time_tolerance is not None:
            tolerance_window = (
                time_tolerance
                if isinstance(time_tolerance, tuple)
                else (-time_tolerance, time_tolerance)
            )
            obs = filter_time_range(
                obs, request_time, tolerance_window, time_column="time"
            )

        if len(obs) == 0:
            return _init_obs_mask() if return_empty_grid else (None, None)

        obs_var = obs["variable"].values  # string dtype - must stay on CPU, never convert to CuPy
        with self.cupy_context:
            obs_lat = self.xp.asarray(obs["lat"].values, dtype=np.float32)
            obs_lon = self.xp.asarray(obs["lon"].values, dtype=np.float32) % 360.0  # Normalize lon to 0-360 to match HRRR grid

        (obs_c, obs_i, obs_j, valid) = self.obs_coords(obs_var, obs_lat, obs_lon)

        # Average multiple observations that map to the same grid cell
        if len(obs_c):
            obs_val = torch.as_tensor(obs["observation"].values, device=self.device, dtype=torch.float32)
            obs_i = obs_i.round().to(torch.int64)
            obs_j = obs_j.round().to(torch.int64)
            vals = obs_val[valid]
            (y_obs, mask) = _init_obs_mask()

            # Flatten (c, i, j) into a single linear index for scatter ops
            flat_idx = obs_c * (self.H * self.W) + obs_i * self.W + obs_j
            y_obs = y_obs.flatten()
            mask = mask.flatten()
            y_obs.scatter_add_(0, flat_idx, vals)
            mask.scatter_add_(0, flat_idx, torch.ones_like(vals))

            averaged_obs = (mask >= 2)
            y_obs[averaged_obs] /= mask[averaged_obs]
            y_obs = y_obs.view(1, *self.grid_shape)
            mask = mask.view(1, *self.grid_shape)
            mask = torch.clamp(mask, max=1, out=mask)

            return y_obs, mask
        else:
            return _init_obs_mask() if return_empty_grid else (None, None)

    def grid_to_obs(
        self,
        x: torch.Tensor,
        obs_c: torch.Tensor,
        obs_i: torch.Tensor,
        obs_j: torch.Tensor,
        method: Literal["nearest", "linear"] = "linear"
    ) -> torch.Tensor:
        """Sample grid values at observation locations (differentiable).

        Parameters
        ----------
        x : torch.Tensor
            Grid tensor of shape ``(C, H, W)``.
        obs_c : torch.Tensor
            Channel index for each observation.
        obs_i : torch.Tensor
            Fractional row index for each observation.
        obs_j : torch.Tensor
            Fractional column index for each observation.
        method : {"nearest", "linear"}, optional
            Interpolation method.  ``"linear"`` (default) performs bilinear
            interpolation and supports gradient backpropagation.

        Returns
        -------
        torch.Tensor
            Sampled values, shape ``(N,)``.

        Raises
        ------
        ValueError
            If ``method`` is not ``"nearest"`` or ``"linear"``.
        """
        if x.ndim != 3:
            raise ValueError("x must be of shape (C, H, W).")
        
        obs_c = torch.as_tensor(obs_c)
        obs_i = torch.as_tensor(obs_i)
        obs_j = torch.as_tensor(obs_j)

        if method == "nearest":
            obs_i = obs_i.round().to(dtype=torch.int64)
            obs_j = obs_j.round().to(dtype=torch.int64)
            obs = x[obs_c, obs_i, obs_j]
        elif method == "linear":
            obs_i0 = obs_i.to(dtype=torch.int64)
            obs_j0 = obs_j.to(dtype=torch.int64)
            obs_i0 = obs_i0.clamp(min=0, max=self.H - 2)
            obs_j0 = obs_j0.clamp(min=0, max=self.W - 2)
            obs_i1 = obs_i0 + 1
            obs_j1 = obs_j0 + 1
            dj = obs_j - obs_j0

            obs_00 = x[obs_c, obs_i0, obs_j0]
            obs_01 = x[obs_c, obs_i0, obs_j1]
            obs_10 = x[obs_c, obs_i1, obs_j0]
            obs_11 = x[obs_c, obs_i1, obs_j1]
            obs_0 = obs_00 + dj * (obs_01 - obs_00)
            obs_1 = obs_10 + dj * (obs_11 - obs_10)
            obs = obs_0 + (obs_i - obs_i0) * (obs_1 - obs_0)
        else:
            raise ValueError(f"Unknown interpolation method: {method!r}. Expected 'nearest' or 'linear'.")

        return obs
