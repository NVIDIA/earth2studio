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

import torch

from earth2studio.statistics.moments import mean
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class energy_score:
    """
    Compute the Energy Score for multivariate ensemble forecast verification.

    The Energy Score is the multivariate generalization of CRPS. Given an
    ensemble forecast {x_1, ..., x_M} and an observation y, the Energy Score
    is defined as:

        ES = (1/M) * sum_m ||x_m - y|| - 1/(2*M^2) * sum_m sum_m' ||x_m - x_m'||

    where ||.|| denotes the Euclidean norm computed over the multivariate
    dimensions. This is a proper scoring rule that is minimized when the
    forecast distribution matches the true distribution.

    Unlike CRPS which evaluates each variable/grid point independently, the
    Energy Score captures whether the ensemble preserves spatial correlations
    across variables and grid points.

    Parameters
    ----------
    ensemble_dimension: str
        A name corresponding to the dimension to perform the ensemble
        reduction over. Example: 'ensemble'
    multivariate_dimensions: list[str]
        Dimensions over which to compute the Euclidean norm.
        Example: ['variable', 'lat', 'lon'] for full spatial ES, or
        ['variable'] for per-grid-point multivariate ES.
    reduction_dimensions: list[str], optional
        Dimensions over which to average the energy score after computation.
        By default None (no additional reduction).
    weights: torch.Tensor, optional
        Weights for the reduction dimensions. Must have the same number of
        dimensions as passed in reduction_dimensions. By default None.
    fair: bool, optional
        If True, use the fair (unbiased) Energy Score estimator with the
        correction factor M/(M-1). By default False.

    References
    ----------
    Gneiting, T. and Raftery, A. E. (2007), "Strictly Proper Scoring Rules,
    Prediction, and Estimation", Journal of the American Statistical
    Association, 102(477), 359-378.
    """

    def __init__(
        self,
        ensemble_dimension: str,
        multivariate_dimensions: list[str],
        reduction_dimensions: list[str] | None = None,
        weights: torch.Tensor = None,
        fair: bool = False,
    ):
        if not isinstance(ensemble_dimension, str):
            raise ValueError("Error! ensemble_dimension must be a string, not a list.")
        if (
            not isinstance(multivariate_dimensions, list)
            or len(multivariate_dimensions) == 0
        ):
            raise ValueError(
                "Error! multivariate_dimensions must be a non-empty list of strings."
            )

        self.ensemble_dimension = ensemble_dimension
        self.multivariate_dimensions = multivariate_dimensions
        self._reduction_dimensions = reduction_dimensions
        self.fair = fair
        if reduction_dimensions is not None:
            self.mean = mean(reduction_dimensions, weights=weights, batch_update=False)

    def __str__(self) -> str:
        dims = (
            self._reduction_dimensions if self._reduction_dimensions is not None else []
        )
        return "_".join(dims + ["energy_score"])

    @property
    def reduction_dimensions(self) -> list[str]:
        """All dimensions that will be reduced/removed from the output."""
        dims = [self.ensemble_dimension] + self.multivariate_dimensions
        if self._reduction_dimensions is not None:
            dims = dims + self._reduction_dimensions
        return dims

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the computed statistic, corresponding to
        the given input coordinates.

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
        Apply the Energy Score metric to ensemble forecast `x` and observation `y`.

        Parameters
        ----------
        x : torch.Tensor
            Ensemble forecast tensor. Must contain the ensemble dimension.
        x_coords : CoordSystem
            Coordinate system describing the `x` tensor. Must contain
            `ensemble_dimension` and all `multivariate_dimensions`.
        y : torch.Tensor
            Observation tensor. Must not contain the ensemble dimension.
        y_coords : CoordSystem
            Coordinate system describing the `y` tensor. Must contain
            all `multivariate_dimensions`.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Energy Score tensor with appropriate reduced coordinates.
        """
        # Validate reduction dimensions exist in x_coords
        if not all(rd in x_coords for rd in self.reduction_dimensions):
            raise ValueError(
                "Initialized reduction dimension does not appear in passed coords"
            )

        # Ensemble dimension should be in x but not in y
        if self.ensemble_dimension in y_coords:
            raise ValueError(
                f"{self.ensemble_dimension} should not be in y_coords but is."
            )
        if x.ndim != y.ndim + 1:
            raise ValueError(
                "x and y must have broadcastable shapes but got "
                + f"{x.shape} and {y.shape}"
            )

        # Validate multivariate dimensions exist in y_coords
        for mv_dim in self.multivariate_dimensions:
            if mv_dim not in y_coords:
                raise ValueError(
                    f"Multivariate dimension '{mv_dim}' not found in y_coords."
                )

        # Input coordinate checking (skip ensemble dim)
        coord_count = 0
        for c in x_coords:
            if c != self.ensemble_dimension:
                handshake_dim(y_coords, c, coord_count)
                handshake_coords(y_coords, x_coords, c)
                coord_count += 1

        # Compute the Energy Score
        out = _energy_score_compute(
            x,
            y,
            x_coords,
            self.ensemble_dimension,
            self.multivariate_dimensions,
            self.fair,
        )

        # Build output coords: remove ensemble and multivariate dims from y_coords
        out_coords = y_coords.copy()
        for mv_dim in self.multivariate_dimensions:
            out_coords.pop(mv_dim, None)

        # Apply additional reduction if requested
        if self._reduction_dimensions is not None:
            out, out_coords = self.mean(out, out_coords)

        return out, out_coords


def _energy_score_compute(
    ensemble: torch.Tensor,
    truth: torch.Tensor,
    x_coords: CoordSystem,
    ensemble_dimension: str,
    multivariate_dimensions: list[str],
    fair: bool = False,
) -> torch.Tensor:
    """
    Compute the Energy Score.

    ES = (1/M) * sum_m ||x_m - y|| - 1/(2*M^2) * sum_m sum_m' ||x_m - x_m'||

    For the fair (unbiased) estimator:
    ES = (1/M) * sum_m ||x_m - y|| - 1/(2*M*(M-1)) * sum_{m!=m'} ||x_m - x_m'||

    Parameters
    ----------
    ensemble : torch.Tensor
        Ensemble forecast tensor with ensemble dimension.
    truth : torch.Tensor
        Observation tensor without ensemble dimension.
    x_coords : CoordSystem
        Coordinate system for the ensemble tensor.
    ensemble_dimension : str
        Name of the ensemble dimension.
    multivariate_dimensions : list[str]
        Dimensions over which to compute the Euclidean norm.
    fair : bool
        Whether to use the fair (unbiased) estimator.

    Returns
    -------
    torch.Tensor
        Energy Score values.
    """
    coord_keys = list(x_coords.keys())
    ens_dim = coord_keys.index(ensemble_dimension)
    M = ensemble.shape[ens_dim]

    # Get indices for multivariate dims in ensemble tensor
    mv_dims = [coord_keys.index(d) for d in multivariate_dimensions]

    # Term 1: (1/M) * sum_m ||x_m - y||
    # Expand truth to match ensemble shape along ensemble dim
    diff_xy = ensemble - truth.unsqueeze(ens_dim)
    # Compute L2 norm over multivariate dims
    # We need to square, sum over mv_dims, then sqrt
    term1 = _l2_norm_over_dims(diff_xy, mv_dims).mean(dim=ens_dim)

    # Term 2: 1/(2*M^2) * sum_m sum_m' ||x_m - x_m'||
    # or for fair: 1/(2*M*(M-1)) * sum_{m!=m'} ||x_m - x_m'||
    # We use torch.cdist for efficiency

    # We need to reshape ensemble so that ens_dim and mv_dims are isolated
    # Move ens_dim to position 0 and mv_dims to the end, flatten the rest
    # into a batch dimension for cdist

    # Strategy: flatten multivariate dims into a single dim, then use cdist
    # First, move ensemble dim to -2 and multivariate dims to -1 (flattened)

    # Determine permutation: [remaining_dims..., ens_dim, mv_dims_flattened...]
    all_dims = list(range(ensemble.ndim))
    remaining_dims = [d for d in all_dims if d != ens_dim and d not in mv_dims]
    perm = remaining_dims + [ens_dim] + mv_dims
    x_perm = ensemble.permute(*perm)

    # Now shape is (*remaining, M, *mv_sizes)
    remaining_shape = x_perm.shape[: len(remaining_dims)]
    mv_size = 1
    for d in mv_dims:
        mv_size *= ensemble.shape[d]

    # Reshape to (batch, M, D) where D = product of multivariate dim sizes
    batch_size = 1
    for s in remaining_shape:
        batch_size *= s
    x_flat = x_perm.reshape(batch_size, M, mv_size)

    # Use cdist for pairwise distances
    pairwise_dists = torch.cdist(x_flat, x_flat, p=2)  # (batch, M, M)

    if fair:
        # Fair estimator: exclude diagonal (m == m'), divide by M*(M-1)
        if M < 2:
            raise ValueError("Fair Energy Score requires at least 2 ensemble members.")
        # Zero out diagonal then sum, which gives sum_{m!=m'}
        mask = ~torch.eye(M, device=pairwise_dists.device, dtype=torch.bool)
        pairwise_sum = (pairwise_dists * mask.unsqueeze(0)).sum(dim=(-1, -2))
        term2 = pairwise_sum / (2.0 * M * (M - 1))
    else:
        # Standard estimator: sum all pairs including diagonal, divide by 2*M^2
        pairwise_sum = pairwise_dists.sum(dim=(-1, -2))
        term2 = pairwise_sum / (2.0 * M * M)

    # Reshape term2 back to remaining_shape
    term2 = term2.reshape(remaining_shape)

    es = term1 - term2
    return es


def _l2_norm_over_dims(x: torch.Tensor, dims: list[int]) -> torch.Tensor:
    """Compute L2 norm over specified dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dims : list[int]
        Dimensions to compute the norm over (will be reduced).

    Returns
    -------
    torch.Tensor
        Tensor with specified dimensions removed.
    """
    return torch.sqrt((x * x).sum(dim=dims))
