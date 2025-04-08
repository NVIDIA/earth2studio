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

import torch

try:
    from physicsnemo.metrics.general.crps import kcrps
except ImportError:
    kcrps = None

from earth2studio.statistics.moments import mean
from earth2studio.utils import check_extra_imports, handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


@check_extra_imports("statistics", [kcrps])
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
    ensemble_dimension: str
        A name corresponding to a dimension to perform the
        ensemble reduction over. Example: 'ensemble'
    reduction_dimensions: list[str]
        A list of dimensions over which to average the crps over.
        optional, by default none. If none, no additional reduction is done.
    weights: torch.Tensor, optional
        A tensor containing weights to assign to the reduction dimensions.
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
        weights: torch.Tensor = None,
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

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of ensemble forecast or prediction data. This is the tensor
            over which the CRPS/CDF is calculated with respect to.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
        y : torch.Tensor
            Observation or validation tensor.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns CRPS tensor with appropriate reduced coordinates.
        """
        if not all([rd in x_coords for rd in self.reduction_dimensions]):
            raise ValueError(
                "Initialized reduction dimension does not appear in passed coords"
            )

        # Do some coordinate checking
        # Assume ensemble_dim is in x_coords but not y_coords
        if self.ensemble_dimension in y_coords:
            raise ValueError(
                f"{self.ensemble_dimension} should not be in y_coords but is."
            )
        if x.ndim != y.ndim + 1:
            raise ValueError(
                "x and y must have broadcastable shapes but got"
                + f"{x.shape} and {y.shape}"
            )
        # Input coordinate checking
        coord_count = 0
        for c in x_coords:
            if c != self.ensemble_dimension:
                handshake_dim(y_coords, c, coord_count)
                handshake_coords(y_coords, x_coords, c)
                coord_count += 1

        dim = list(x_coords).index(self.ensemble_dimension)
        if self.fair:
            out = kcrps(x, y, dim=dim, biased=False)
        else:
            out = _crps_from_empirical_cdf(x, y, dim=dim)
        out_coords = y_coords.copy()

        if self._reduction_dimensions is not None:
            out, out_coords = self.mean(out, out_coords)
        return out, out_coords


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
