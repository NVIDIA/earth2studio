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

Both follow the xarray ``earth2studio.statistics`` Metric and Statistic
protocols (``metric(x, y) -> xr.DataArray``, ``statistic(x) -> xr.DataArray``)
and are candidates for upstreaming.

Classes
-------
mse
    Mean squared error (Metric protocol).  Identical to
    ``earth2studio.statistics.rmse`` but returns MSE *without* the final
    square root, so time aggregation can use the correct
    ``RMSE = sqrt(mean_t(MSE_t))`` form.
ensemble_variance
    Spatially averaged ensemble variance (Statistic protocol).  Computes
    ``var_m(f)`` across the ensemble dimension then averages over the
    spatial (reduction) dimensions.  Useful for spread-skill analysis:
    ``Spread = sqrt(mean_t(Var_t))``.
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr

from earth2studio.statistics.moments import mean, variance
from earth2studio.statistics.rmse import rmse
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class mse(rmse):
    """Mean squared error of two arrays over a set of given dimensions.

    Same interface as ``earth2studio.statistics.rmse`` but returns MSE (no
    square root), which allows the correct deferred time aggregation::

        RMSE = sqrt(mean_t(MSE_t))

    Parameters
    ----------
    reduction_dimensions : list[str]
        Dimensions to average the squared error over (e.g. ``["lat", "lon"]``).
    weights : torch.Tensor | np.ndarray | None
        Weights for the reduction dimensions.  Must have
        ``ndim == len(reduction_dimensions)``.
    batch_update : bool
        Whether to apply running (batch) updates.
    ensemble_dimension : str | None
        If provided, compute the ensemble mean of *x* before computing MSE
        against *y*.  The output has no ensemble dimension.
    """

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["mse"])

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """Compute MSE between *x* (forecast) and *y* (observation).

        Parameters
        ----------
        x : xr.DataArray
            Forecast array; ``reduction_dimensions`` (and the ensemble
            dimension, if any) must be among its dimensions.
        y : xr.DataArray
            Observation array with the dimensions of *x* minus the ensemble
            dimension.

        Returns
        -------
        xr.DataArray
            Mean squared error with the reduced dimensions removed.
        """
        diff = self._difference(x, y)
        diff_t, _ = diff.e2s.to_torch()
        return self.mean(from_torch(diff_t.square(), diff))


class ensemble_variance:
    """Spatially averaged ensemble variance (Statistic protocol).

    Computes the sample variance across the ensemble dimension (Bessel's
    correction, ``N-1`` denominator, via
    ``earth2studio.statistics.moments.variance``), then averages over the
    spatial (reduction) dimensions.

    Useful for WB2-style spread-skill analysis::

        Spread = sqrt(mean_t(Var_t))

    Parameters
    ----------
    ensemble_dimension : str
        Name of the ensemble dimension (e.g. ``"ensemble"``).
    reduction_dimensions : list[str]
        Spatial dimensions to average the variance over (e.g.
        ``["lat", "lon"]``).
    weights : torch.Tensor | np.ndarray | None
        Weights for the spatial reduction dimensions.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str],
        weights: torch.Tensor | np.ndarray | None = None,
    ):
        self.ens_var = variance([ensemble_dimension])
        self.spatial_mean = mean(list(reduction_dimensions), weights=weights)
        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = [ensemble_dimension] + list(reduction_dimensions)
        self.weights = weights

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["ensemble_variance"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Return the coordinate signature after both reductions.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature to transform.

        Returns
        -------
        CoordinateSystem
            Signature with the ensemble and spatial dimensions removed.
        """
        return self.spatial_mean.output_coords(self.ens_var.output_coords(input_coords))

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute the spatially averaged ensemble variance.

        Parameters
        ----------
        x : xr.DataArray
            Forecast array with an ensemble dimension.

        Returns
        -------
        xr.DataArray
            Variance with the ensemble and spatial dimensions removed.
        """
        return self.spatial_mean(self.ens_var(x))
