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
from numpy.typing import ArrayLike

from earth2studio.utils.coords import handshake_dim
from earth2studio.utils.type import CoordSystem

from .moments import mean


class brier_score:
    """
    Statistic for calculating the Brier score (BS) of one tensors
    with respect to another over a set of given dimensions.

    Follows the definition here:
    https://www.ecmwf.int/sites/default/files/elibrary/2017/17626-ensemble-verification-metrics.pdf

    If `ensemble_dimension` is provided, the forecast probability is inferred
    by averaging over the ensemble members.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    thresholds: ArrayLike[float]
        A list of the thresholds applied when calculating BS.
    ensemble_dimension: str | None = None
        Ensemble dimension for computation of probabilistic BS. If None (default),
        forecast is interpreted as deterministic.
    batch_update: bool = False
        Whether to apply batch updates to the BS with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running BS.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        thresholds: ArrayLike,
        ensemble_dimension: str | None = None,
        batch_update: bool = False,
    ):
        self.mean = mean(reduction_dimensions, batch_update=batch_update)
        self.thresholds = np.array(thresholds).astype(np.float32)
        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = reduction_dimensions

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["brier_score"])

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

        return output_coords

    def _validate_coords(self, x_coords: CoordSystem, y_coords: CoordSystem) -> None:
        if ("threshold" in x_coords) or ("threshold" in y_coords):
            raise ValueError(
                "Dimension 'threshold' cannot be present in brier_score input coordinates."
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
            x_coords.pop("ensemble")
        for (x_dim, x_coord), (y_dim, y_coord) in zip(
            x_coords.items(), y_coords.items()
        ):
            if (x_dim != y_dim) or (x_coord != y_coord).any():
                raise ValueError("Coordinates are incompatible.")

    def _exceedance_probability(
        self, x: torch.Tensor, thresholds: torch.Tensor, ensemble_dim: int | None = None
    ) -> torch.Tensor:
        thresholds = thresholds.view(*((1,) * x.ndim), len(self.thresholds))
        x = x.unsqueeze(-1)
        exc_prob = (x >= thresholds).to(dtype=x.dtype)
        if ensemble_dim is not None:
            exc_prob = exc_prob.mean(dim=ensemble_dim)
        return exc_prob

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

        thresholds = torch.as_tensor(self.thresholds).to(dtype=x.dtype, device=x.device)
        ensemble_dim = (
            None
            if self.ensemble_dimension is None
            else list(x_coords).index(self.ensemble_dimension)
        )
        exc_prob_fc = self._exceedance_probability(
            x, thresholds, ensemble_dim=ensemble_dim
        )
        exc_prob_obs = self._exceedance_probability(y, thresholds)
        exc_prob_coords = y_coords.copy()
        exc_prob_coords["threshold"] = self.thresholds.copy()

        (bs, out_coords) = self.mean((exc_prob_fc - exc_prob_obs) ** 2, exc_prob_coords)

        return (bs, out_coords)
