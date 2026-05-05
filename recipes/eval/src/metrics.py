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

"""Recipe-local metric and statistic classes.

These follow the ``earth2studio.statistics`` Metric / Statistic protocols
and are candidates for upstreaming.

Classes
-------
mse
    Mean squared error (Metric protocol).  Identical to
    ``earth2studio.statistics.rmse`` but returns MSE *without* the
    final square root so that time aggregation can use the correct
    ``RMSE = sqrt(mean_t(MSE_t))`` form.
ensemble_variance
    Spatially-averaged ensemble variance (Statistic protocol).  Computes
    ``var_m(f)`` across the ensemble dimension then averages over
    spatial (reduction) dimensions.  Useful for spread-skill analysis:
    ``Spread = sqrt(mean_t(Var_t))``.
"""

from __future__ import annotations

import torch

from earth2studio.statistics.moments import mean, variance
from earth2studio.utils.coords import CoordSystem, handshake_dim


class mse:
    """Mean squared error of two tensors over a set of given dimensions.

    Identical interface to ``earth2studio.statistics.rmse`` but returns
    MSE (no square root).  This allows correct deferred time aggregation::

        RMSE = sqrt(mean_t(MSE_t))

    Parameters
    ----------
    reduction_dimensions : list[str]
        Dimensions to average the squared error over (e.g. ``["lat", "lon"]``).
    weights : torch.Tensor | None
        Weights for the reduction dimensions.  Must have
        ``ndim == len(reduction_dimensions)``.
    batch_update : bool
        Whether to apply running (batch) updates.
    ensemble_dimension : str | None
        If provided, compute the ensemble mean of *x* before computing
        MSE against *y*.  The ensemble dimension is removed from the
        output coordinates.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor | None = None,
        batch_update: bool = False,
        ensemble_dimension: str | None = None,
    ):
        self.mean = mean(
            reduction_dimensions, weights=weights, batch_update=batch_update
        )
        if ensemble_dimension is not None:
            self.ensemble_dimension: str | None = ensemble_dimension
            self.ensemble_mean = mean([ensemble_dimension])
        else:
            self.ensemble_dimension = None
        self.weights = weights
        self._reduction_dimensions = reduction_dimensions
        self.batch_update = batch_update

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["mse"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Return the coordinate system after reduction.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform.

        Returns
        -------
        CoordSystem
        """
        output_coords = input_coords.copy()
        for dimension in self.reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)
        if self.ensemble_dimension is not None:
            output_coords.pop(self.ensemble_dimension)
        return output_coords

    def __call__(
        self,
        x: torch.Tensor,
        x_coords: CoordSystem,
        y: torch.Tensor,
        y_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Compute MSE between *x* (forecast) and *y* (observation).

        Parameters
        ----------
        x : torch.Tensor
            Forecast tensor.
        x_coords : CoordSystem
            Coordinates describing *x*.
        y : torch.Tensor
            Observation tensor.
        y_coords : CoordSystem
            Coordinates describing *y*.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            ``(mse_values, output_coords)``
        """
        if self.ensemble_dimension is not None:
            x, x_coords = self.ensemble_mean(x, x_coords)
        mse_val, output_coords = self.mean((x - y) ** 2, x_coords)
        return mse_val, output_coords


class ensemble_variance:
    """Spatially-averaged ensemble variance (Statistic protocol).

    Computes the sample variance across the ensemble dimension, then
    averages over spatial (reduction) dimensions.  Follows the
    ``earth2studio.statistics.base.Statistic`` protocol (2-argument
    ``__call__``).

    The ensemble variance uses Bessel's correction (``N-1`` denominator)
    via ``earth2studio.statistics.moments.variance``.

    Useful for WB2-style spread-skill analysis::

        Spread = sqrt(mean_t(Var_t))

    Parameters
    ----------
    ensemble_dimension : str
        Name of the ensemble dimension (e.g. ``"ensemble"``).
    reduction_dimensions : list[str]
        Spatial dimensions to average variance over (e.g.
        ``["lat", "lon"]``).
    weights : torch.Tensor | None
        Weights for the spatial reduction dimensions.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str],
        weights: torch.Tensor | None = None,
    ):
        self.ens_var = variance([ensemble_dimension])
        self.spatial_mean = mean(reduction_dimensions, weights=weights)
        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = [ensemble_dimension] + list(reduction_dimensions)
        self.weights = weights

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["ensemble_variance"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Return the coordinate system after reduction.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform.

        Returns
        -------
        CoordSystem
        """
        output_coords = input_coords.copy()
        for dimension in self._reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)
        return output_coords

    def __call__(
        self,
        x: torch.Tensor,
        x_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Compute spatially-averaged ensemble variance.

        Parameters
        ----------
        x : torch.Tensor
            Forecast tensor with an ensemble dimension.
        x_coords : CoordSystem
            Coordinates describing *x*.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            ``(variance_values, output_coords)``
        """
        var, var_coords = self.ens_var(x, x_coords)
        return self.spatial_mean(var, var_coords)
