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

from earth2studio.statistics.moments import mean, variance
from earth2studio.statistics.utils import (
    _align_xy,
    _output_signature,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class rmse:
    """
    Statistic for calculating the root mean squared error of two arrays
    over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor | np.ndarray | None = None
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    batch_update: bool = False
        Whether to apply batch updates to the rmse with each invocation of __call__.
        This is particularly useful when data is received in a stream of batches. Each
        invocation of __call__ will return the running rmse. In particular, it will apply
        the square root operation after calculating the running mean squared error.
    ensemble_dimension: str | None = None
        Compute ensemble mean over this dimension before computing RMSE. If None (default),
        no reduction is performed.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor | np.ndarray | None = None,
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
        return "_".join(self._reduction_dimensions + ["rmse"])

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
            Coordinate system of the statistic output
        """
        remove = list(self.reduction_dimensions)
        if self.ensemble_dimension is not None:
            remove.append(self.ensemble_dimension)
        return _output_signature(input_coords, remove)

    def _difference(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """Ensemble mean (if requested) and difference between x and y"""
        if self.ensemble_dimension is not None:
            x = self.ensemble_mean(x)
        y = _align_xy(x, y)
        x_t, _ = x.e2s.to_torch()
        y_t, _ = y.e2s.to_torch()
        return from_torch(x_t - y_t, x)

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array, but RMSE is
            symmetric with respect to `x` and `y`.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            Input array #2 intended to be used as validation data, but RMSE is symmetric
            with respect to `x` and `y`.

        Returns
        -------
        xr.DataArray
            Returns root mean squared error array with appropriate reduced coordinates.
        """
        diff = self._difference(x, y)
        diff_t, _ = diff.e2s.to_torch()
        mse = self.mean(from_torch(diff_t.square(), diff))
        mse_t, _ = mse.e2s.to_torch()
        return from_torch(torch.sqrt(mse_t), mse)


class mae(rmse):
    """
    Statistic for calculating the mean absolute error of two arrays
    over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor | np.ndarray | None = None
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    batch_update: bool = False
        Whether to apply batch updates to the rmse with each invocation of __call__.
        This is particularly useful when data is received in a stream of batches. Each
        invocation of __call__ will return the running rmse. In particular, it will apply
        the square root operation after calculating the running mean squared error.
    ensemble_dimension: str | None = None
        Compute ensemble mean over this dimension before computing MAE. If None (default),
        no reduction is performed.
    """

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["mae"])

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample MAE over all seen batches.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array, but MAE is
            symmetric with respect to `x` and `y`.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            Input array #2 intended to be used as validation data, but MAE is symmetric
            with respect to `x` and `y`.

        Returns
        -------
        xr.DataArray
            Returns mean absolute error array with appropriate reduced coordinates.
        """
        diff = self._difference(x, y)
        diff_t, _ = diff.e2s.to_torch()
        return self.mean(from_torch(torch.abs(diff_t), diff))


class spread_skill_ratio:
    """Metric for calculating the spread/skill ratio of an ensemble forecast.

     Specifically, the spread is defined as the standard deviation of the ensemble
     forecast. The skill is defined as the rmse of the ensemble mean prediction. The
     ratio of these two quantities is defined as the spread/skill ratio.

    Parameters
    ----------
    ensemble_dimension : str
        The dimension over which the spread and skill are calculated over.
        This should usually be "ensemble".
    reduction_dimensions : list[str]
        Dimensions to reduce (mean) the spread/skill ratio over. This is commonly done
        over time but can also be the globe or some region.
        Example: ['time', 'lat', 'lon']
    ensemble_weights : torch.Tensor | np.ndarray | None, optional
        A one-dimensional array containing weights to assign to the ensemble_dimension,
        by default None.
    reduction_weights : torch.Tensor | np.ndarray | None, optional
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2. Ignored if None, by default None.
    ensemble_batch_update : bool, optional
        Whether to apply batch updates to the ensemble mean and variance components
        of the spread and skill with each invocation of __call__. This is particularly
        useful when ensemble data is received in a stream of batches. Each invocation of
        __call__ will return the running spread/skill ratio., by default False.
    reduction_batch_update : bool, optional
        Whether to apply batch updates to the reduction rmse and averaging components
        of the spread/skill with each invocation of __call__. This is particularly
        useful when time data is received in a stream of batches., by default False.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str],
        ensemble_weights: torch.Tensor | np.ndarray | None = None,
        reduction_weights: torch.Tensor | np.ndarray | None = None,
        ensemble_batch_update: bool = False,
        reduction_batch_update: bool = False,
    ):
        self.ensemble_dimension = [ensemble_dimension]
        self._reduction_dimensions = reduction_dimensions
        self.ensemble_mean = mean(
            reduction_dimensions=self.ensemble_dimension,
            weights=ensemble_weights,
            batch_update=ensemble_batch_update,
        )
        self.ensemble_var = variance(
            reduction_dimensions=self.ensemble_dimension,
            weights=ensemble_weights,
            batch_update=ensemble_batch_update,
        )
        self.reduced_rmse = rmse(
            reduction_dimensions=reduction_dimensions,
            weights=reduction_weights,
            batch_update=reduction_batch_update,
        )
        self.reduced_mean = mean(
            reduction_dimensions=reduction_dimensions,
            weights=reduction_weights,
            batch_update=reduction_batch_update,
        )

    def __str__(self) -> str:
        return "_".join(
            self.ensemble_dimension + self._reduction_dimensions + ["spread_skill"]
        )

    @property
    def reduction_dimensions(self) -> list[str]:
        return self.ensemble_dimension + self._reduction_dimensions

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

    def _components(
        self, x: xr.DataArray, y: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Squared skill (mse of ensemble mean) and spread (variance) components"""
        em = self.ensemble_mean(x)
        skill = self.reduced_rmse(em, y)
        spread = self.reduced_mean(self.ensemble_var(x))
        return skill, spread

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : xr.DataArray
            The ensemble forecast input array. This is the array over which the
            ensemble mean and spread are calculated with respect to.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            The observation input array.

        Returns
        -------
        xr.DataArray
            Returns spread/skill ratio array with appropriate reduced coordinates.
        """
        skill, spread = self._components(x, y)
        skill_t, _ = skill.e2s.to_torch()
        spread_t, _ = spread.e2s.to_torch()
        return from_torch(torch.sqrt(spread_t) / skill_t, spread)


class skill_spread(spread_skill_ratio):
    """
    Metric for calculating the skill/spread ratio of an ensemble forecast.

    Specifically, the spread is defined as the standard deviation of the ensemble
    forecast. The skill is defined as the rmse of the ensemble mean prediction. The
    ratio of these two quantities is defined as the spread/skill ratio.

    This method, instead of returning the ratio, returns the individual components
    of the ratio, i.e. the spread and the skill.
    """

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the statistic output with a leading `metric`
            dimension
        """
        return _output_signature(
            input_coords,
            self.reduction_dimensions,
            add={"metric": np.array(["mse", "variance"])},
            add_first=True,
        )

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample MSE and variance over all seen batches.

        Parameters
        ----------
        x : xr.DataArray
            The ensemble forecast input array. This is the array over which the
            ensemble mean and spread are calculated with respect to.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            The observation input array.

        Returns
        -------
        xr.DataArray
            Returns an array containing MSE and variance, stacked along a leading
            `metric` dimension, with appropriate reduced coordinates.
        """
        skill, var = self._components(x, y)
        skill_t, _ = skill.e2s.to_torch()
        var_t, _ = var.e2s.to_torch()
        return from_torch(torch.stack((skill_t.square(), var_t)), self.output_coords(x))
