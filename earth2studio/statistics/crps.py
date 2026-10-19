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
import xarray as xr

from earth2studio.statistics.moments import mean
from earth2studio.statistics.utils import (
    _align_xy,
    _output_signature,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from physicsnemo.metrics.general.crps import kcrps
except ImportError:
    OptionalDependencyFailure("statistics")
    kcrps = None


@check_optional_dependencies()
class crps:
    """
    Compute the Continuous Ranked Probability Score (CRPS).

    Uses this formula
        # int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.


    This statistic reduces over a single dimension, where the presumed ensemble dimension
    does not appear in the truth/observation array.

    Parameters
    ----------
    ensemble_dimension: str
        A name corresponding to a dimension to perform the
        ensemble reduction over. Example: 'ensemble'
    reduction_dimensions: list[str]
        A list of dimensions over which to average the crps over.
        optional, by default none. If none, no additional reduction is done.
    weights: torch.Tensor | np.ndarray, optional
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
        By default None.
    fair: bool, optional
        If true, the CRPS is calculated using the fair CRPS formula.
        By default False.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str] | None = None,
        weights: torch.Tensor | np.ndarray | None = None,
        fair: bool = False,
    ):
        if not isinstance(ensemble_dimension, str):
            raise ValueError(
                "Error! CRPS currently assumes reduction over a single dimension."
            )

        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = reduction_dimensions
        if reduction_dimensions is not None:
            self.mean = mean(reduction_dimensions, weights=weights, batch_update=False)
        self.fair = fair

    def __str__(self) -> str:
        return "_".join(self.reduction_dimensions + ["crps"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return (
            [self.ensemble_dimension]
            if self._reduction_dimensions is None
            else [self.ensemble_dimension] + self._reduction_dimensions
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the statistic output
        """
        return _output_signature(input_coords, self.reduction_dimensions)

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        Parameters
        ----------
        x : xr.DataArray
            Input array of ensemble forecast or prediction data. This is the array
            over which the CRPS/CDF is calculated with respect to.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            Observation or validation array. Must not have the ensemble dimension.

        Returns
        -------
        xr.DataArray
            Returns CRPS array with appropriate reduced coordinates.
        """
        # Do some coordinate checking
        # Assume ensemble_dim is in x but not y
        if self.ensemble_dimension in y.dims:
            raise ValueError(f"{self.ensemble_dimension} should not be in y but is.")
        if x.ndim != y.ndim + 1:
            raise ValueError(
                "x and y must have broadcastable shapes but got"
                + f"{x.shape} and {y.shape}"
            )
        y = _align_xy(x, y, self.ensemble_dimension)

        dim = x.dims.index(self.ensemble_dimension)
        x_t, _ = x.e2s.to_torch()
        y_t, _ = y.e2s.to_torch()
        if self.fair:
            out_t = kcrps(x_t, y_t, dim=dim, biased=False)
        else:
            out_t = _crps_from_empirical_cdf(x_t, y_t, dim=dim)
        out = from_torch(out_t, _output_signature(x, [self.ensemble_dimension]))

        if self._reduction_dimensions is not None:
            out = self.mean(out)
        return out


def _crps_from_empirical_cdf(
    ensemble: torch.Tensor, truth: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """

    Warning
    -------
    This method is being upstreamed to https://github.com/NVIDIA/physicsnemo in the next release.

    Compute the exact CRPS using the CDF method

    Uses this formula
        # int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.

    Parameters
    ----------
    ensemble : torch.Tensor
        array of ensemble members
    truth : torch.Tensor
        array of observations
    dim : int
        Dimension to perform CRPS reduction over.

    Returns
    -------
        array of CRPS scores

    """
    n = ensemble.shape[dim]
    ensemble = torch.sort(ensemble, dim=dim).values
    ans = torch.zeros_like(truth)

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    # val = ensemble[0] - truth
    val = ensemble.select(dim, 0) - truth
    ans += torch.where(val > 0, val, 0.0)

    for i in range(n - 1):
        x0 = ensemble.select(dim, i)
        x1 = ensemble.select(dim, i + 1)

        cdf = (i + 1) / n

        # a. case y < x0
        val = (x1 - x0) * (cdf - 1) ** 2
        mask = truth < x0
        ans += torch.where(mask, val, 0.0)

        # b. case x0 <= y <= x1
        val = (truth - x0) * cdf**2 + (x1 - truth) * (cdf - 1) ** 2
        mask = (truth >= x0) & (truth <= x1)
        ans += torch.where(mask, val, 0.0)

        # c. case x1 < t
        mask = truth > x1
        val = (x1 - x0) * cdf**2
        ans += torch.where(mask, val, 0.0)

    # dx [F(x) - H(x-y)]^2 = dx [1 - 0]^2 = dx
    val = truth - ensemble.select(dim, n - 1)
    ans += torch.where(val > 0, val, 0.0)
    return ans
