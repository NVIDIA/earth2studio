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

from typing import Protocol, runtime_checkable

import torch

from earth2studio.utils.type import CoordSystem


@runtime_checkable
class Statistic(Protocol):
    """Statistic interface."""

    @property
    def reduction_dimensions(self) -> list[str]:
        """Gives the input dimensions of which the statistic performs a reduction
        over. The is used to determine, a priori, the output dimensions of a statistic.
        """
        pass

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
        pass

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply statistic to data `x`, with coordinates `coords` and reduce
        over dimensions `reduction_dimensions`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply statistic to.
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor.
            `reduction_dimensions` must be in coords.
        """
        pass


@runtime_checkable
class Metric(Protocol):
    """Metrics interface."""

    @property
    def reduction_dimensions(self) -> list[str]:
        pass

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
        pass

    def __call__(
        self,
        x: torch.Tensor,
        x_coords: CoordSystem,
        y: torch.Tensor,
        y_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dimensions`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor #1 intended to apply metric to. `x` is typically understood
            to be the forecast or prediction tensor.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in coords.
        y : torch.Tensor
            Input tensor #2 intended to apply statistic to. `y` is typically the observation
            or validation tensor.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in coords.
        """
        pass
