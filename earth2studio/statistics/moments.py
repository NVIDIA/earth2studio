# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch

from earth2studio.statistics.utils import _broadcast_weights
from earth2studio.utils.type import CoordSystem


class mean:
    """
    Statistic for calculating the sample mean over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor, optional
        A tensor containing weights to assign to the reduction dimensions.
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
        weights: torch.Tensor = None,
        batch_update: bool = False,
    ):
        if weights is not None:
            if weights.ndim != len(reduction_dimensions):
                raise ValueError(
                    "Error! Weights must be the same dimension as reduction_dimensions"
                )
        self.reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.batch_update = batch_update
        if self.batch_update:
            self.n = 0

    def __str__(self) -> str:
        return "_".join(self.reduction_dimensions + ["mean"])

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Apply the mean operation over the tensor x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample mean over all seen batches.

        Parameters
        ----------
        x: torch.Tensor
            Input data to compute sample mean.
        coords: CoordSystem
            Coordinates referring to the input data, x.
        """
        if not all([rd in coords for rd in self.reduction_dimensions]):
            raise ValueError(
                "Initialized reduction dimensions do not appear in passed coords."
            )

        dims = [list(coords).index(rd) for rd in self.reduction_dimensions]
        output_coords = CoordSystem(
            {key: coords[key] for key in coords if key not in self.reduction_dimensions}
        )
        weights = _broadcast_weights(
            self.weights, self.reduction_dimensions, coords
        ).to(x.device)
        weights_sum = torch.sum(weights)

        # If not applying batch updating then return regular mean.
        if not self.batch_update:
            return torch.sum(weights * x, dim=dims) / weights_sum, output_coords

        # If batch updating then calculate updated mean
        else:
            if self.n == 0:
                self.sum = torch.sum(weights * x, dim=dims)
            else:
                self.sum += torch.sum(weights * x, dim=dims)

            self.n += weights_sum
            return self.sum / self.n, output_coords


class variance:
    """
    Statistic for calculating the sample variance over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor, optional
        A tensor containing weights to assign to the reduction dimensions.
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
        weights: torch.Tensor = None,
        batch_update: bool = False,
    ):
        if weights is not None:
            if weights.ndim != len(reduction_dimensions):
                raise ValueError(
                    "Error! Weights must be the same dimension as reduction_dimensions"
                )
        self.reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.batch_update = batch_update
        if self.batch_update:
            self.n = 0

    def __str__(self) -> str:
        return "_".join(self.reduction_dimensions + ["variance"])

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Apply the sample variance operation over the tensor x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample variance over all seen batches.

        Parameters
        ----------
        x: torch.Tensor
            Input data to compute sample variance.
        coords: CoordSystem
            Coordinates referring to the input data, x.
        """
        if not all([rd in coords for rd in self.reduction_dimensions]):
            raise ValueError(
                "Initialized reduction dimensions do not appear in passed coords."
            )

        dims = [list(coords).index(rd) for rd in self.reduction_dimensions]
        output_coords = CoordSystem(
            {key: coords[key] for key in coords if key not in self.reduction_dimensions}
        )

        weights = _broadcast_weights(
            self.weights, self.reduction_dimensions, coords
        ).to(x.device)
        weights_sum = torch.sum(weights)

        # If not applying batch updating then return regular variance.
        if not self.batch_update:
            m = torch.sum(weights * x, dim=dims, keepdims=True) / weights_sum
            div = weights_sum - torch.sum(weights**2) / weights_sum
            return torch.sum(weights * (x - m) ** 2, dim=dims) / div, output_coords

        # If batch updating then calculate updated mean
        else:
            temp_n = weights_sum
            temp_sum = torch.sum(weights * x, dim=dims)
            temp_sum2 = torch.sum(weights * (x - temp_sum / temp_n) ** 2, dim=dims)

            # First batch then no correction
            if self.n == 0:
                self.n = temp_n
                self.sum = temp_sum
                self.sum2 = temp_sum2
            # Second-order correction with each batch
            else:
                delta = self.sum * temp_n / self.n - temp_sum
                self.sum += temp_sum
                self.sum2 += (
                    temp_sum2 + self.n / temp_n / (self.n + temp_n) * delta**2
                )
                self.n += temp_n

            return (
                self.sum2 / torch.maximum(self.n - 1.0, torch.tensor(1.0)),
                output_coords,
            )


class std:
    """
    Statistic for calculating the sample standard deviation over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: torch.Tensor, optional
        A tensor containing weights to assign to the reduction dimensions.
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
        weights: torch.Tensor = None,
        batch_update: bool = False,
    ):
        self.var = variance(
            reduction_dimensions, weights=weights, batch_update=batch_update
        )
        self.reduction_dimensions = reduction_dimensions
        self.weights = weights

    def __str__(self) -> str:
        return "_".join(self.reduction_dimensions + ["std"])

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Apply the sample standard deviation operation over the tensor x.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample standard deviation over all seen batches.

        Parameters
        ----------
        x: torch.Tensor
            Input data to compute sample standard deviation.
        coords: CoordSystem
            Coordinates referring to the input data, x.
        """
        var, output_coords = self.var(x, coords)
        return torch.sqrt(var), output_coords
