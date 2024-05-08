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
        self.reduction_dimensions = reduction_dimensions
        self.weights = weights

        self.climatology = climatology

    def __str__(self) -> str:
        return "acc"

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
            Input tensor #1 intended to apply metric to.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            'reduction_dims' must be in coords.
            "time" and "variable" must be in x_coords.
        y : torch.Tensor
            Input tensor #2 intended to be used as validation data.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            'reduction_dims' must be in coords.
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

        dims = [list(x_coords).index(rd) for rd in self.reduction_dimensions]
        output_coords = CoordSystem(
            {
                key: x_coords[key]
                for key in x_coords
                if key not in self.reduction_dimensions
            }
        )

        weights = _broadcast_weights(
            self.weights, self.reduction_dimensions, x_coords
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
        x_bar = torch.sum(weights * x, dim=dims, keepdim=True) / weights_sum
        x_diff = x_hat - x_bar

        y_hat = y - clim
        y_bar = torch.sum(weights * y, dim=dims, keepdim=True) / weights_sum
        y_diff = y_hat - y_bar

        p1 = torch.sum(weights * x_diff * y_diff, dim=dims)
        p2 = torch.sum(weights * x_diff * x_diff, dim=dims)
        p3 = torch.sum(weights * y_diff * y_diff, dim=dims)

        return p1 / torch.sqrt(p2 * p3), output_coords
