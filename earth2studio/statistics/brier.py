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
from numpy.typing import ArrayLike

from earth2studio.statistics.moments import mean
from earth2studio.statistics.utils import (
    _align_xy,
    _output_signature,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class brier_score:
    """
    Statistic for calculating the Brier score (BS) of one array
    with respect to another over a set of given dimensions.

    Follows the definition here:
    https://www.ecmwf.int/sites/default/files/elibrary/2017/17626-ensemble-verification-metrics.pdf

    If `ensemble_dimension` is provided, the forecast probability is inferred
    by averaging over the ensemble members.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    thresholds: ArrayLike[float]
        A list of the thresholds applied when calculating BS.
    ensemble_dimension: str | None = None
        Ensemble dimension for computation of probabilistic BS. If None (default),
        forecast is interpreted as deterministic.
    batch_update: bool = False
        Whether to apply batch updates to the BS with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running BS.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        thresholds: ArrayLike,
        ensemble_dimension: str | None = None,
        batch_update: bool = False,
    ):
        self.mean = mean(reduction_dimensions, batch_update=batch_update)
        self.thresholds = np.array(thresholds).astype(np.float32)
        self.ensemble_dimension = ensemble_dimension
        self._reduction_dimensions = reduction_dimensions

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["brier_score"])

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
        removed_dims = list(self._reduction_dimensions)
        if self.ensemble_dimension is not None:
            removed_dims.append(self.ensemble_dimension)
        return _output_signature(
            input_coords, removed_dims, add={"threshold": self.thresholds.copy()}
        )

    def _validate_coords(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        if ("threshold" in x.dims) or ("threshold" in y.dims):
            raise ValueError(
                "Dimension 'threshold' cannot be present in brier_score input dimensions."
            )
        return _align_xy(x, y, self.ensemble_dimension)

    def _exceedance_probability(
        self, x: torch.Tensor, ensemble_dim: int | None = None
    ) -> torch.Tensor:
        thresholds = torch.as_tensor(self.thresholds, dtype=x.dtype, device=x.device)
        data = (x[..., None] >= thresholds).to(x.dtype)
        if ensemble_dim is not None:
            data = data.mean(dim=ensemble_dim)
        return data

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array.
            `reduction_dimensions` must be in the dimensions of x, as must
            `ensemble_dimension` if provided in constructor.
        y : xr.DataArray
            Input array #2 intended to be used as validation data. Must not contain
            `ensemble_dimension`. `reduction_dimensions` must be in y.

        Returns
        -------
        xr.DataArray
            Returns Brier score array with appropriate reduced coordinates and
            a trailing threshold dimension.
        """
        y = self._validate_coords(x, y)
        x_t, _ = x.e2s.to_torch()
        y_t, _ = y.e2s.to_torch()
        ensemble_dim = (
            x.dims.index(self.ensemble_dimension)
            if self.ensemble_dimension is not None
            else None
        )
        exc_prob_fc = self._exceedance_probability(x_t, ensemble_dim)
        exc_prob_obs = self._exceedance_probability(y_t)
        signature = _output_signature(y, add={"threshold": self.thresholds.copy()})
        return self.mean(from_torch((exc_prob_fc - exc_prob_obs) ** 2, signature))
