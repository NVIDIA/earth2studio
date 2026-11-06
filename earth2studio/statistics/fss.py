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

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from numpy.typing import ArrayLike

from earth2studio.statistics.moments import mean
from earth2studio.statistics.utils import (
    _align_xy,
    _output_signature,
    _spatial_dims_to_end,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class fss:
    """
    Statistic for calculating the fractions skill score (FSS) of one array
    with respect to another over a set of given dimensions.

    If `ensemble_dimension` is provided, computes the probabilistic FSS as defined by
    Necker et al. (2024): https://doi.org/10.1002/qj.4824.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    window_sizes: ArrayLike[int]
        A list of the window sizes (in pixels) applied when calculating FSS.
    thresholds: ArrayLike[float]
        A list of the thresholds applied when calculating FSS.
    spatial_dimensions: Tuple[str, str] | None = None
        Indicates the spatial dimensions. If None, it is assumed that these are the last
        two dimensions.
    ensemble_dimension: str | None = None
        Ensemble dimension for computation of probabilistic FSS. If None (default),
        forecast is interpreted as deterministic.
    batch_update: bool = False
        Whether to apply batch updates to the FSS with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running FSS.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        window_sizes: ArrayLike,
        thresholds: ArrayLike,
        spatial_dimensions: tuple[str, str] | None = None,
        ensemble_dimension: str | None = None,
        batch_update: bool = False,
    ):
        # tracking different means is needed for batch_update to work
        self.fc_mean = {
            ws: mean(reduction_dimensions, batch_update=batch_update)
            for ws in window_sizes
        }
        self.obs_mean = {
            ws: mean(reduction_dimensions, batch_update=batch_update)
            for ws in window_sizes
        }
        self.diff_mean = {
            ws: mean(reduction_dimensions, batch_update=batch_update)
            for ws in window_sizes
        }

        self.window_sizes = np.array(window_sizes).astype(np.float32)
        self.thresholds = np.array(thresholds).astype(np.float32)
        self.spatial_dimensions = spatial_dimensions
        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = reduction_dimensions
        self.windows: dict[int, torch.Tensor] = {}

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["fss"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the statistic output, with trailing `threshold`
            and `window_size` dimensions
        """
        removed_dims = list(self._reduction_dimensions)
        if self.ensemble_dimension is not None:
            removed_dims.append(self.ensemble_dimension)

        return _output_signature(
            input_coords,
            removed_dims,
            add={
                "threshold": self.thresholds.copy(),
                "window_size": self.window_sizes.copy(),
            },
        )

    def _get_window(
        self, window_size: int, dtype: torch.dtype | str, device: torch.device | str
    ) -> torch.Tensor:
        if window_size in self.windows:
            window = self.windows[window_size]
            if window.dtype == dtype and window.device == device:
                return window

        # construct circular window
        window_size_int = int(np.ceil(window_size))
        window_rad = window_size / 2
        wx = torch.arange(window_size_int).to(dtype=dtype, device=device)
        wx = wx - (window_size_int - 1) / 2
        r_sqr = wx[None, None, None, :] ** 2 + wx[None, None, :, None] ** 2
        window = (r_sqr <= window_rad**2).to(dtype=dtype)
        window = window / window.sum()
        window = window.repeat(len(self.thresholds), 1, 1, 1).contiguous()

        self.windows[window_size] = window

        return window

    def _validate_coords(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        for forbidden_dim in ["threshold", "window_size"]:
            if (forbidden_dim in x.dims) or (forbidden_dim in y.dims):
                raise ValueError(
                    f"Dimension '{forbidden_dim}' cannot be present in FSS input dimensions."
                )
        y = _align_xy(x, y, self.ensemble_dimension)

        spatial_dims = (
            [*y.dims][-2:]
            if self.spatial_dimensions is None
            else self.spatial_dimensions
        )
        if any(dim not in self._reduction_dimensions for dim in spatial_dims):
            raise ValueError(
                f"Spatial dimensions {spatial_dims} must be included in reduction_dimensions."
            )
        return y

    def _neighborhood_probability(
        self,
        x: torch.Tensor,
        thresholds: torch.Tensor,
        window_size: int,
        ensemble_dim: int | None = None,
    ) -> torch.Tensor:
        window = self._get_window(window_size, x.dtype, x.device)

        # squeeze batch dimensions
        x_shape = x.shape
        x = x.reshape(int(np.prod(x_shape[:-2])), 1, *x_shape[-2:])

        # mask input points that exceed each threshold
        binary_prob = (x >= thresholds[None, :, None, None]).to(dtype=x.dtype)

        # compute neighborhood probability for each threshold using convolution
        # TODO: for large windows FFT convolution would be better
        neighborhood_prob = F.conv2d(binary_prob, window, groups=len(self.thresholds))

        # move threshold dim to end and restore batch dimensions
        neighborhood_prob = neighborhood_prob.permute(0, 2, 3, 1)
        neighborhood_prob = neighborhood_prob.reshape(
            *x_shape[:-2], *neighborhood_prob.shape[-3:]
        )

        if ensemble_dim is not None:
            # reduce ensemble mean
            neighborhood_prob = neighborhood_prob.mean(dim=ensemble_dim)

        return neighborhood_prob

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array.
            `reduction_dimensions` must be in x, as must `ensemble_dimension` and
            `spatial_dimensions` if provided in constructor.
        y : xr.DataArray
            Input array #2 intended to be used as validation data. Must not contain
            the ensemble dimension.

        Returns
        -------
        xr.DataArray
            Returns fractions skill score array with appropriate reduced coordinates
            and trailing threshold and window_size dimensions.
        """
        y = self._validate_coords(x, y)

        if self.spatial_dimensions is not None:
            x = _spatial_dims_to_end(x, self.spatial_dimensions)
            y = _spatial_dims_to_end(y, self.spatial_dimensions)

        x_t, _ = x.e2s.to_torch()
        y_t, _ = y.e2s.to_torch()

        thresholds = torch.as_tensor(self.thresholds).to(
            dtype=x_t.dtype, device=x_t.device
        )
        out_sig = self.output_coords(x)
        ensemble_dim = (
            None
            if self.ensemble_dimension is None
            else list(x.dims).index(self.ensemble_dimension)
        )
        scores = []

        for window_size in self.window_sizes:
            np_fc = self._neighborhood_probability(
                x_t, thresholds, window_size, ensemble_dim=ensemble_dim
            )
            np_obs = self._neighborhood_probability(y_t, thresholds, window_size)
            margin = _get_margin(window_size)
            template = y.isel(
                {dim: slice(margin[0], y.sizes[dim] - margin[1]) for dim in y.dims[-2:]}
            )
            signature = _output_signature(
                template, add={"threshold": self.thresholds.copy()}
            )
            diff = self.diff_mean[window_size](
                from_torch((np_fc - np_obs) ** 2, signature)
            )
            fc = self.fc_mean[window_size](from_torch(np_fc**2, signature))
            obs = self.obs_mean[window_size](from_torch(np_obs**2, signature))
            diff_t, _ = diff.e2s.to_torch()
            fc_t, _ = fc.e2s.to_torch()
            obs_t, _ = obs.e2s.to_torch()
            scores.append(1 - diff_t / (fc_t + obs_t))

        return from_torch(torch.stack(scores, dim=-1), out_sig)


def _get_margin(window_size: int) -> tuple[int, int]:
    window_size_int = int(np.ceil(window_size))
    if window_size_int % 2:
        margin = (window_size_int // 2, window_size_int // 2)
    else:
        margin = (window_size_int // 2 - 1, window_size_int // 2)
    return margin
