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

from earth2studio.data import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.statistics.utils import _broadcast_weights
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class acc:
    """
    Statistic for calculating the anomaly correlation coefficient of two tensors
    over a set of given dimensions, with respect to some optional climatology.

    Parameters
    ----------
    reduction_dimensions: List[str]
        A list of names corresponding to dimensions to perform the
        statistical reduction over. Example: ['lat', 'lon']
    climatology: DataSource
        Optional (by default None) climatology to remove from tensors to create
        anomalies before computing the correlation coefficient.
    weights: torch.Tensor = None
        A tensor containing weights to assign to the reduction dimensions.
        Note that these weights must have the same number of dimensions
        as passed in reduction_dimensions.
        Example: if reduction_dimensions = ['lat', 'lon'] then
        assert weights.ndim == 2.
    """

    def __init__(
        self,
        reduction_dimensions: list[str],
        climatology: DataSource | None = None,
        weights: torch.Tensor | None = None,
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
        are broadcastable.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, typically the forecast or prediction tensor, but ACC is
            symmetric with respect to `x` and `y`.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
            "time" and "variable" must be in x_coords.
        y : torch.Tensor
            Input tensor #2 intended to be used as validation data, but ACC is symmetric
            with respect to `x` and `y`.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.
            "time" and "variable" must be in y_coords.
            If "lead_time" is in x_coords, then "lead_time" must also be in y_coords. The
            intention, in this case, is that users will use `fetch_data` to make it easier
            to match validation times.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns anomaly correlation coefficient tensor with appropriate reduced coordinates.

        Note
        ----
        Reference: https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2016/ppts_pdfs/ECMWF_ACC_definition.pdf
        """

        # Input coordinate checking
        handshake_dim(x_coords, "time")
        handshake_dim(x_coords, "variable")
        for i, c in enumerate(x_coords):
            handshake_dim(y_coords, c, i)
            handshake_coords(x_coords, y_coords, c)

        dims = [list(x_coords).index(rd) for rd in self._reduction_dimensions]
        output_coords = self.output_coords(x_coords)

        weights = _broadcast_weights(
            self.weights, self._reduction_dimensions, x_coords
        ).to(x.device)
        weights_sum = torch.sum(weights)

        # Get climatology information
        if self.climatology is not None:
            if "lead_time" in output_coords:
                clim, _ = fetch_data(
                    self.climatology,
                    output_coords["time"],
                    output_coords["variable"],
                    lead_time=output_coords["lead_time"],
                    device=x.device,
                )
            else:
                da = self.climatology(output_coords["time"], output_coords["variable"])
                clim = torch.as_tensor(da.values, device=x.device, dtype=x.dtype)
        else:
            clim = torch.zeros_like(x)

        x_hat = x - clim
        x_bar = torch.sum(weights * x_hat, dim=dims, keepdim=True) / weights_sum
        x_diff = x_hat - x_bar

        y_hat = y - clim
        y_bar = torch.sum(weights * y_hat, dim=dims, keepdim=True) / weights_sum
        y_diff = y_hat - y_bar

        p1 = torch.sum(weights * x_diff * y_diff, dim=dims)
        p2 = torch.sum(weights * x_diff * x_diff, dim=dims)
        p3 = torch.sum(weights * y_diff * y_diff, dim=dims)

        return p1 / torch.sqrt(p2 * p3), output_coords
