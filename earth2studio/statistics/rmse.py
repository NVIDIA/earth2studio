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

from earth2studio.utils.type import CoordSystem

from .moments import mean


class rmse:
    """
    Statistic for calculating the root mean squared error of two tensors
    over a set of given dimensions.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    weights: Optional[torch.Tensor] = None
        A tensor containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    batch_update: Optional[bool] = False
        Whether to applying batch updates to the rmse with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running rmse. It particular, it will apply
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
        self.reduction_dimensions = reduction_dimensions

        self.batch_update = batch_update

    def __str__(self) -> str:
        return "_".join(self.reduction_dimensions + ["rmse"])

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
            Input tensor #1 intended to apply metric to.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            'reduction_dims' must be in coords.
        y : torch.Tensor
            Input tensor #2 intended to apply statistic to.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            'reduction_dims' must be in coords.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns root mean squared error tensor with appropriate reduced coordinates.
        """
        mse, output_coords = self.mean((x - y) ** 2, x_coords)
        return torch.sqrt(mse), output_coords
