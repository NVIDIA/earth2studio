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

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike

from earth2studio.utils.coords import handshake_dim
from earth2studio.utils.type import CoordSystem

from .moments import mean
from .utils import _spatial_dims_to_end


class fss:
    """
    Statistic for calculating the fractions skill score (FSS) of one tensors
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

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        removed_dims = list(self._reduction_dimensions)
        if self.ensemble_dimension is not None:
            removed_dims.append(self.ensemble_dimension)

        output_coords = input_coords.copy()
        for dimension in removed_dims:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)
        output_coords["threshold"] = self.thresholds.copy()
        output_coords["window_size"] = self.window_sizes.copy()

        return output_coords

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

    def _validate_coords(self, x_coords: CoordSystem, y_coords: CoordSystem) -> None:
        for forbidden_dim in ["threshold", "window_size"]:
            if (forbidden_dim in x_coords) or (forbidden_dim in y_coords):
                raise ValueError(
                    f"Dimension 'f{forbidden_dim}' cannot be present in FSS input coordinates."
                )
        x_coords = x_coords.copy()
        if self.ensemble_dimension is not None:
            if self.ensemble_dimension not in x_coords:
                raise ValueError(
                    f"Ensemble dimension '{self.ensemble_dimension}' set but not present in x_coords."
                )
            if self.ensemble_dimension in y_coords:
                raise ValueError(
                    f"Ensemble dimension '{self.ensemble_dimension}' present in y_coords."
                )
            x_coords.pop(self.ensemble_dimension)
        for (x_dim, x_coord), (y_dim, y_coord) in zip(
            x_coords.items(), y_coords.items()
        ):
            if (x_dim != y_dim) or (x_coord != y_coord).any():
                raise ValueError("Coordinates are incompatible.")

        spatial_dims = (
            list(x_coords)[-2:]
            if self.spatial_dimensions is None
            else self.spatial_dimensions
        )
        if any(dim not in self._reduction_dimensions for dim in spatial_dims):
            raise ValueError(
                f"Spatial dimensions {spatial_dims} must be included in reduction_dimensions."
            )

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
        x = x.view(np.prod(x_shape[:-2]), 1, *x_shape[-2:])

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

    def _neighborhood_probability_coords(
        self, coords: CoordSystem, out_coords: CoordSystem, window_size: int
    ) -> CoordSystem:
        coords = coords.copy()
        spatial_dims = list(coords)[-2:]
        margin = _get_margin(window_size)
        for dim in spatial_dims:
            coords[dim] = coords[dim][margin[0] : -margin[1]].copy()
        coords["threshold"] = out_coords["threshold"].copy()
        return coords

    def __call__(
        self,
        x: torch.Tensor,
        x_coords: CoordSystem,
        y: torch.Tensor,
        y_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, typically the forecast or prediction tensor.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in x_coords, as do `ensemble_dimension` and
            `spatial_dimensions` if provided in constructor.
        y : torch.Tensor
            Input tensor #2 intended to be used as validation data..
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in y_coords, do `spatial_dimensions` if
            provided in constructor.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns root mean squared error tensor with appropriate reduced coordinates.
        """
        self._validate_coords(x_coords, y_coords)

        if self.spatial_dimensions is not None:
            (x, x_coords) = _spatial_dims_to_end(x, x_coords, self.spatial_dimensions)
            (y, y_coords) = _spatial_dims_to_end(y, y_coords, self.spatial_dimensions)

        thresholds = torch.as_tensor(self.thresholds).to(dtype=x.dtype, device=x.device)
        out_coords = self.output_coords(x_coords)
        ensemble_dim = (
            None
            if self.ensemble_dimension is None
            else list(x_coords).index(self.ensemble_dimension)
        )
        fss_shape = [len(c) for c in out_coords.values()]
        fss = torch.empty(fss_shape, dtype=x.dtype, device=x.device)

        for i, window_size in enumerate(self.window_sizes):
            np_fc = self._neighborhood_probability(
                x, thresholds, window_size, ensemble_dim=ensemble_dim
            )
            np_obs = self._neighborhood_probability(y, thresholds, window_size)
            np_coords = self._neighborhood_probability_coords(
                y_coords, out_coords, window_size
            )
            fss[..., i] = 1 - self.diff_mean[window_size](
                (np_fc - np_obs) ** 2, np_coords
            )[0] / (
                self.fc_mean[window_size](np_fc**2, np_coords)[0]
                + self.obs_mean[window_size](np_obs**2, np_coords)[0]
            )

        return (fss, out_coords)


def _get_margin(window_size: int) -> tuple[int, int]:
    window_size_int = int(np.ceil(window_size))
    if window_size_int % 2:
        margin = (window_size_int // 2, window_size_int // 2)
    else:
        margin = (window_size_int // 2 - 1, window_size_int // 2)
    return margin
