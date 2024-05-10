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

from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class crps:
    """
    Compute the Continuous Ranked Probably Score (CRPS).

    Uses this formula
        # int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.


    This statistic reduces over a single dimension, where the presumed ensemble dimension
    does not appear in the truth/observation tensor.

    Parameters
    ----------
    reduction_dimension: str
        A name corresponding to a dimension to perform the
        statistical reduction over. Example: 'ensemble'
    """

    def __init__(self, reduction_dimension: str):
        if not isinstance(reduction_dimension, str):
            raise ValueError(
                "Error! CRPS currently assumes reduction over a single dimension."
            )

        self.reduction_dimension = reduction_dimension

    def __str__(self) -> str:
        return "crps"

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
        if self.reduction_dimension not in x_coords:
            raise ValueError(
                "Initialized reduction dimension does not appear in passed coords"
            )

        # Do some coordinate checking
        # Assume reduction_dim is in x_coords but not y_coords
        if self.reduction_dimension in y_coords:
            raise ValueError(
                f"{self.reduction_dimension} should not be in y_coords but is."
            )
        if x.ndim != y.ndim + 1:
            raise ValueError(
                "x and y must have broadcastable shapes but got"
                + f"{x.shape} and {y.shape}"
            )
        # Input coordinate checking
        coord_count = 0
        for c in x_coords:
            if c != self.reduction_dimension:
                handshake_dim(y_coords, c, coord_count)
                handshake_coords(y_coords, x_coords, c)
                coord_count += 1

        dim = list(x_coords).index(self.reduction_dimension)

        _crps = _crps_from_empirical_cdf(x, y, dim=dim)
        return _crps, y_coords.copy()


def _crps_from_empirical_cdf(
    ensemble: torch.Tensor, truth: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """

    Warning
    -------
    This method is being upstreamed to https://github.com/NVIDIA/modulus in the next release.

    Compute the exact CRPS using the CDF method

    Uses this formula
        # int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.

    Parameters
    ----------
    ensemble : torch.Tensor
        tensor of ensemble members
    truth : torch.Tensor
        tensor of observations
    dim : int
        Dimension to perform CRPS reduction over.

    Returns
    -------
        tensor of CRPS scores

    """
    n = ensemble.shape[dim]
    device = ensemble.device
    ensemble, _ = torch.sort(ensemble, dim=dim)
    ans = torch.zeros_like(truth)

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    # val = ensemble[0] - truth
    val = (
        torch.index_select(
            ensemble, dim, torch.tensor([0], device=device, dtype=torch.int32)
        ).squeeze(dim)
        - truth
    )
    ans += torch.where(val > 0, val, 0.0)

    for i in range(n - 1):
        x0 = torch.index_select(
            ensemble, dim, torch.tensor([i], device=device, dtype=torch.int32)
        ).squeeze(dim)
        x1 = torch.index_select(
            ensemble, dim, torch.tensor([i + 1], device=device, dtype=torch.int32)
        ).squeeze(dim)

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
    val = truth - torch.index_select(
        ensemble, dim, torch.tensor([n - 1], device=device, dtype=torch.int32)
    ).squeeze(dim)
    ans += torch.where(val > 0, val, 0.0)
    return ans
