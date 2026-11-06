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

from typing import Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem


# --8<-- [start:statistic-interface]
@runtime_checkable
class Statistic(Protocol):
    """Statistic interface."""

    @property
    def reduction_dimensions(self) -> list[str]:
        """Gives the input dimensions of which the statistic performs a reduction
        over. The is used to determine, a priori, the output dimensions of a statistic.
        """
        pass

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system (xr.DataArray signature) to transform into
            output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the statistic output
        """
        pass

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Apply statistic to data `x` and reduce over dimensions
        `reduction_dimensions`.

        Parameters
        ----------
        x : xr.DataArray
            Input data array intended to apply statistic to. Dimensions and
            coordinates are read from the array itself.
            `reduction_dimensions` must be in the dimensions of x.

        Returns
        -------
        xr.DataArray
            Statistic with the reduced dimensions removed
        """
        pass


# --8<-- [end:statistic-interface]


# --8<-- [start:metric-interface]
@runtime_checkable
class Metric(Protocol):
    """Metrics interface."""

    @property
    def reduction_dimensions(self) -> list[str]:
        """Gives the input dimensions of which the metric performs a reduction
        over. The is used to determine, a priori, the output dimensions of a metric.
        """
        pass

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed metric, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system (xr.DataArray signature) of `x` to transform into
            output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the metric output
        """
        pass

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dimensions`.

        Parameters
        ----------
        x : xr.DataArray
            Input data array #1 intended to apply metric to. `x` is typically understood
            to be the forecast or prediction array.
            `reduction_dimensions` must be in the dimensions of x.
        y : xr.DataArray
            Input data array #2 intended to apply metric to. `y` is typically the
            observation or validation array.

        Returns
        -------
        xr.DataArray
            Metric with the reduced dimensions removed
        """
        pass


# --8<-- [end:metric-interface]
