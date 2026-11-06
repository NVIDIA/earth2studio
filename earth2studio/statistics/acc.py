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

from earth2studio.data import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.statistics.utils import (
    _align_xy,
    _broadcast_weights,
    _output_signature,
)
from earth2studio.utils.coords import handshake_dim
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class acc:
    """
    Statistic for calculating the anomaly correlation coefficient of two arrays
    over a set of given dimensions, with respect to some optional climatology.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    climatology: DataSource
        Optional (by default None) climatology to remove from tensors to create
        anomalies before computing the correlation coefficient.
    weights: torch.Tensor | np.ndarray = None
        An array containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        climatology: DataSource | None = None,
        weights: torch.Tensor | np.ndarray | None = None,
    ):

        if weights is not None:
            if weights.ndim != len(reduction_dimensions):
                raise ValueError(
                    "Error! Weights must be the same dimension as reduction_dimensions"
                )
        self._reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.climatology = climatology

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["acc"])

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

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array, but ACC is
            symmetric with respect to `x` and `y`.
            `reduction_dimensions` must be in the dimensions of x.
            "time" and "variable" must be in x.
        y : xr.DataArray
            Input array #2 intended to be used as validation data, but ACC is symmetric
            with respect to `x` and `y`. Must have the same dimensions and coordinates
            as `x`. If "lead_time" is in x, then "lead_time" must also be in y. The
            intention, in this case, is that users will use `fetch_data` to make it
            easier to match validation times.

        Returns
        -------
        xr.DataArray
            Returns anomaly correlation coefficient array with appropriate reduced
            coordinates.

        Note
        ----
        For more information see the following references:

        - https://confluence.ecmwf.int/spaces/FUG/pages/673550781/Section%2B6.2.2%2BAnomaly%2BCorrelation%2BCoefficient
        """

        # Input coordinate checking
        handshake_dim(x, "time")
        handshake_dim(x, "variable")
        dims = tuple(x.dims.index(d) for d in self._reduction_dimensions)
        y = _align_xy(x, y)

        signature = self.output_coords(x)
        times = x.coords["time"].values
        variables = x.coords["variable"].values
        lead_times = x.coords["lead_time"].values if "lead_time" in x.dims else None
        x_t, _ = x.e2s.to_torch()
        y_t, _ = y.e2s.to_torch()
        weights = _broadcast_weights(
            self.weights, self._reduction_dimensions, x, x_t.device
        )
        weights_sum = torch.sum(weights)

        # Get climatology information
        if self.climatology is not None:
            if lead_times is not None:
                climatology = fetch_data(
                    self.climatology,
                    times,
                    variables,
                    lead_time=lead_times,
                    device=x_t.device,
                )
            else:
                climatology = self.climatology(times, variables)
            clim, _ = climatology.transpose(
                *(d for d in x.dims if d in climatology.dims)
            ).e2s.to_torch()
            clim = clim.to(device=x_t.device, dtype=x_t.dtype)
            clim = clim.reshape(
                [climatology.sizes[d] if d in climatology.dims else 1 for d in x.dims]
            )
        else:
            clim = torch.zeros((1,) * x.ndim, dtype=x_t.dtype, device=x_t.device)

        x_hat = x_t - clim
        x_bar = torch.sum(weights * x_hat, dim=dims, keepdim=True) / weights_sum
        x_diff = x_hat - x_bar

        y_hat = y_t - clim
        y_bar = torch.sum(weights * y_hat, dim=dims, keepdim=True) / weights_sum
        y_diff = y_hat - y_bar

        p1 = torch.sum(weights * x_diff * y_diff, dim=dims)
        p2 = torch.sum(weights * x_diff * x_diff, dim=dims)
        p3 = torch.sum(weights * y_diff * y_diff, dim=dims)

        return from_torch(p1 / torch.sqrt(p2 * p3), signature)
