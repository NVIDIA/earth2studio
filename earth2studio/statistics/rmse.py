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

from earth2studio.utils.coords import handshake_dim
from earth2studio.utils.type import CoordSystem

from .moments import mean, variance


class rmse:
    """
    Statistic for calculating the root mean squared error of two tensors
    over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor = None
        A tensor containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    batch_update: bool = False
        Whether to applying batch updates to the rmse with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running rmse. In particular, it will apply
        the square root operation after calculating the running mean squared error.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor = None,
        batch_update: bool = False,
    ):

        self.mean = mean(
            reduction_dimensions, weights=weights, batch_update=batch_update
        )
        self.weights = weights
        self._reduction_dimensions = reduction_dimensions

        self.batch_update = batch_update

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["rmse"])

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
        output_coords = input_coords.copy()
        for dimension in self.reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)

        return output_coords

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

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, typically the forecast or prediction tensor, but RMSE is
            symmetric with respect to `x` and `y`.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
        y : torch.Tensor
            Input tensor #2 intended to be used as validation data, but ACC is symmetric
            with respect to `x` and `y`.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns root mean squared error tensor with appropriate reduced coordinates.
        """
        mse, output_coords = self.mean((x - y) ** 2, x_coords)
        return torch.sqrt(mse), output_coords


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
    ensemble_weights : torch.Tensor | None, optional
        A one-dimensional tensor containing weights to assign to the ensemble_dimension,
        by default None.
    reduction_weights : torch.Tensor, optional
        A tensor containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2. Ignored if None, by default None.
    ensemble_batch_update : bool, optional
        Whether to applying batch updates to the ensemble mean and variance components
        of the spread and skill with each invocation of __call__. This is particularly
        useful when ensemble data is recieved in a stream of batches. Each invocation of
        __call__ will return the running spread/skill ratio., by default False.
    reduction_batch_update : bool, optional
        Whether to applying batch updates to the reduction rmse and averaging components
        of the spread/skill with each invocation of __call__. This is particularly
        useful when time data is recieved in a stream of batches., by default False.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str],
        ensemble_weights: torch.Tensor | None = None,
        reduction_weights: torch.Tensor = None,
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
        output_coords = input_coords.copy()
        for dimension in self.reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)

        return output_coords

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

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : torch.Tensor
            The ensemble forecast input tensor. This is the tensor over which the
            ensemble mean and spread are calculated with respect to.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
        y : torch.Tensor
            The observation input tensor.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns root mean squared error tensor with appropriate reduced coordinates.
        """

        em, output_coords = self.ensemble_mean(x, x_coords)
        skill, output_coords = self.reduced_rmse(em, output_coords, y, y_coords)
        spread, output_coords = self.reduced_mean(*self.ensemble_var(x, x_coords))
        return torch.sqrt(spread) / skill, output_coords


class skill_spread(spread_skill_ratio):
    """
    Metric for calculating the skill/spread ratio of an ensemble forecast.

    Specifically, the spread is defined as the standard deviation of the ensemble
    forecast. The skill is defined as the rmse of the ensemble mean prediction. The
    ratio of these two quantities is defined as the spread/skill ratio.

    This method, instead of returning the ratio, returns the individual components
    of the ratio, i.e. the spread and the skill.
    """

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
        output_coords = input_coords.copy()
        for dimension in self.reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)

        output_coords.update({"metric": np.array(["mse", "variance"])})
        output_coords.move_to_end("metric", last=False)

        return output_coords

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

        If batch_update was passed True upon metric initialization then this method
        returns the running sample MSE and variance over all seen batches.

        Parameters
        ----------
        x : torch.Tensor
            The ensemble forecast input tensor. This is the tensor over which the
            ensemble mean and spread are calculated with respect to.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
        y : torch.Tensor
            The observation input tensor.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns a tensor containing MSE and variance with appropriate reduced coordinates.
        """

        em, output_coords = self.ensemble_mean(x, x_coords)
        skill, output_coords = self.reduced_rmse(em, output_coords, y, y_coords)
        var, output_coords = self.reduced_mean(*self.ensemble_var(x, x_coords))

        mse = torch.square(skill)

        output_coords.update({"metric": np.array(["mse", "variance"])})
        output_coords.move_to_end("metric", last=False)

        return torch.stack((mse, var)), output_coords
