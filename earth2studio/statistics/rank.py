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

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo.metrics.general.histogram import _count_bins, linspace
except ImportError:
    OptionalDependencyFailure("statistics")
    _count_bins = None
    linspace = None


@check_optional_dependencies()
class rank_histogram:
    """
    Compute the Rank Histogram for a given set of ensemble forecasts.


    This statistic reduces over a single dimension, where the presumed ensemble dimension
    does not appear in the truth/observation tensor.

    Parameters
    ----------
    ensemble_dimension: str
        A name corresponding to a dimension to perform the ranking over.
        Example: 'ensemble'
    reduction_dimensions: list[str]
        A list of dimensions over which to bin the ranks
    number_of_bins: int | None, optional
        The number of bins to discretize the unit interval over. Best set to
        ensemble_size + 1, which is set automatically when None, by default None
    randomize_ties: bool, optional
        When True randomize the rank in cases where multiple ensemble
        members are exactly equal to the observation. This produces an unbiased
        distribution of ranks in cases where ties occur frequently. If False, the rank
        will be computed as if the observation were larger than the tied ensemble
        members, by default True
    """

    def __init__(
        self,
        ensemble_dimension: str,
        reduction_dimensions: list[str],
        number_of_bins: int | None = None,
        randomize_ties: bool = True,
    ):
        if not isinstance(ensemble_dimension, str):
            raise ValueError(
                "Error! Rank Histogram currently assumes reduction over a single dimension."
            )

        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = reduction_dimensions
        self.number_of_bins = number_of_bins
        self.randomize_ties = randomize_ties

    def __str__(self) -> str:
        return "rank_histogram"

    @property
    def reduction_dimensions(self) -> list[str]:
        return [self.ensemble_dimension] + self._reduction_dimensions

    def _get_number_of_bins(self, input_coords: CoordSystem) -> int:
        if self.number_of_bins is None:
            return len(input_coords[self.ensemble_dimension]) + 1
        else:
            return self.number_of_bins

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
        number_of_bins = self._get_number_of_bins(input_coords)
        for dimension in self.reduction_dimensions:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)

        output_coords = (
            OrderedDict(
                {
                    "histogram_data": np.array(["bin_centers", "bin_counts"]),
                    "bin": np.arange(number_of_bins),
                }
            )
            | output_coords
        )

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
        are broadcastable. While reducing over `reduction_dimensions`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of ensemble data. The rank of observation input tensor `y`
            is determined with respect to the ensemble dimension of `x`.
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
            Returns rank histogram tensor with appropriate reduced coordinates.
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

        number_of_bins = self._get_number_of_bins(x_coords)

        # Get the dimension index of the ensemble dimension
        dim = list(x_coords).index(self.ensemble_dimension)

        # Move the ensemble dimension to the first dimension
        x = torch.movedim(x, dim, 0)

        # Compute the ranks over the ensemble dimension
        _ranks = torch.sum(torch.ge(y, x).to(dtype=torch.int32), dim=0)

        if self.randomize_ties:
            _ranks_gt = torch.sum(torch.gt(y, x).to(dtype=torch.int32), dim=0)
            rank_diff = _ranks - _ranks_gt
            ties = rank_diff > 0
            _ranks[ties] -= (
                torch.rand_like(_ranks[ties], dtype=torch.float32)
                * (rank_diff[ties] + 1)
            ).to(dtype=torch.int32)

        # Reshape ranks using reduction dimensions
        dims = [list(y_coords).index(rd) for rd in self._reduction_dimensions]
        new_dims = list(range(len(dims)))
        _ranks = torch.flatten(
            torch.movedim(_ranks, dims, new_dims),
            start_dim=new_dims[0],
            end_dim=new_dims[-1],
        )

        # Normalize ranks to [0, 1]
        _ranks = _ranks / x.shape[0]

        # Compute histogram
        # note: this version of linspace returns self.number_of_bins + 1 bin edges
        _bin_edges = linspace(
            torch.zeros_like(_ranks[0]), torch.ones_like(_ranks[0]), number_of_bins
        )
        _rank_histogram = _count_bins(_ranks, _bin_edges)

        # Create output tensor and coords
        out = torch.stack((0.5 * (_bin_edges[:-1] + _bin_edges[1:]), _rank_histogram))
        out_coords = y_coords.copy()
        for rd in self._reduction_dimensions:
            out_coords.pop(rd)

        out_coords = (
            OrderedDict(
                {
                    "histogram_data": np.array(["bin_centers", "bin_counts"]),
                    "bin": np.arange(number_of_bins),
                }
            )
            | out_coords
        )
        return out, out_coords
