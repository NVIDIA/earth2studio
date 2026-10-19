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

from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.statistics.utils import (
    _broadcast_weights,
    _output_signature,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class mean:
    """
    Statistic for calculating the sample mean over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor | np.ndarray, optional
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
        By default None.
    batch_update: bool, optional
        Whether to applying batch updates to the mean with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running mean.
        By default False.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor | np.ndarray | None = None,
        batch_update: bool = False,
    ):
        if weights is not None:
            if weights.ndim != len(reduction_dimensions):
                raise ValueError(
                    "Error! Weights must be the same dimension as reduction_dimensions"
                )
        self._reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.batch_update = batch_update
        if self.batch_update:
            self.n: Any = 0
            self.sum: Any = None

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["mean"])

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
        return _output_signature(input_coords, self.reduction_dimensions)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """
        Apply the mean operation over the array x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample mean over all seen batches.

        Parameters
        ----------
        x: xr.DataArray
            Input data to compute sample mean.

        Returns
        -------
        xr.DataArray
            Mean with the reduction dimensions removed
        """
        # Empty reduction dimensions represents an identity operation
        if not self._reduction_dimensions:
            return x

        dims = tuple(x.dims.index(d) for d in self._reduction_dimensions)
        signature = self.output_coords(x)
        x_t, _ = x.e2s.to_torch()
        weights = _broadcast_weights(
            self.weights, self._reduction_dimensions, x, x_t.device
        )
        weights_sum = torch.sum(weights)
        total = torch.sum(weights * x_t, dim=dims)

        # If not applying batch updating then return regular mean.
        if not self.batch_update:
            return from_torch(total / weights_sum, signature)

        # If batch updating then calculate updated mean
        if self.n == 0:
            self.sum = total
        else:
            self.sum = self.sum + total
        self.n = self.n + weights_sum
        return from_torch(self.sum / self.n, signature)


class variance:
    """
    Statistic for calculating the sample variance over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor | np.ndarray, optional
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
        By default None.
    batch_update: bool, optional
        Whether to applying batch updates to the variance with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running variance.
        By default False.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor | np.ndarray | None = None,
        batch_update: bool = False,
    ):
        if not reduction_dimensions:
            raise ValueError("variance requires at least one reduction dimension.")
        if weights is not None:
            if weights.ndim != len(reduction_dimensions):
                raise ValueError(
                    "Error! Weights must be the same dimension as reduction_dimensions"
                )
        self._reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.batch_update = batch_update
        if self.batch_update:
            self.n: Any = 0
            self.sum: Any = None
            self.sum2: Any = None

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["variance"])

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

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """
        Apply the sample variance operation over the array x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample variance over all seen batches.

        Parameters
        ----------
        x: xr.DataArray
            Input data to compute sample variance.

        Returns
        -------
        xr.DataArray
            Variance with the reduction dimensions removed
        """
        dims = tuple(x.dims.index(d) for d in self._reduction_dimensions)
        signature = self.output_coords(x)
        x_t, _ = x.e2s.to_torch()
        weights = _broadcast_weights(
            self.weights, self._reduction_dimensions, x, x_t.device
        )
        weights_sum = torch.sum(weights)
        x_t = x_t.to(torch.promote_types(x_t.dtype, weights.dtype))

        # If not applying batch updating then return regular variance.
        if not self.batch_update:
            m = torch.sum(weights * x_t, dim=dims, keepdim=True) / weights_sum
            div = weights_sum - torch.sum(weights**2) / weights_sum
            out = torch.sum(weights * (x_t - m) ** 2, dim=dims) / div
            return from_torch(out, signature)

        # If batch updating then calculate updated variance
        temp_n = weights_sum
        temp_sum = torch.sum(weights * x_t, dim=dims)
        temp_mean = torch.sum(weights * x_t, dim=dims, keepdim=True) / temp_n
        temp_sum2 = torch.sum(weights * (x_t - temp_mean) ** 2, dim=dims)

        # First batch then no correction
        if self.n == 0:
            self.n = temp_n
            self.sum = temp_sum
            self.sum2 = temp_sum2
        # Second-order correction with each batch
        else:
            delta = self.sum * temp_n / self.n - temp_sum
            self.sum = self.sum + temp_sum
            self.sum2 = (
                self.sum2 + temp_sum2 + self.n / temp_n / (self.n + temp_n) * delta**2
            )
            self.n = self.n + temp_n

        return from_torch(self.sum2 / torch.clamp(self.n - 1.0, min=1.0), signature)


class std:
    """
    Statistic for calculating the sample standard deviation over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor | np.ndarray, optional
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
        By default None.
    batch_update: bool, optional
        Whether to applying batch updates to the standard deviation with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running standard deviation.
        By default False.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        weights: torch.Tensor | np.ndarray | None = None,
        batch_update: bool = False,
    ):
        self.var = variance(
            reduction_dimensions, weights=weights, batch_update=batch_update
        )
        self._reduction_dimensions = reduction_dimensions
        self.weights = weights

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["std"])

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
        return _output_signature(input_coords, self.reduction_dimensions)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """
        Apply the sample standard deviation operation over the array x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample standard deviation over all seen batches.

        Parameters
        ----------
        x: xr.DataArray
            Input data to compute sample standard deviation.

        Returns
        -------
        xr.DataArray
            Standard deviation with the reduction dimensions removed
        """
        var = self.var(x)
        var_t, _ = var.e2s.to_torch()
        return from_torch(torch.sqrt(var_t), var)
