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

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem


# --8<-- [start:prognostic-model-interface]
@runtime_checkable
class PrognosticModel(Protocol):
    """Prognostic model interface"""

    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Forward pass of the prognostic model, time integrating a single time-step

        Parameters
        ----------
        x : xr.DataArray
            NumPy-backed CPU or CuPy-backed CUDA state with labeled coordinates
            and the metadata required by ``input_coords()``.

        Returns
        -------
        xr.DataArray
            State one time-step into the future, including output coordinates.
        """
        pass

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : xr.DataArray
            Initial state with labeled coordinates and model metadata.

        Yields
        ------
        xr.DataArray
            Initial state followed by successive forecast states.
        """
        pass

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of prognostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature with relative lead times.
        """
        pass

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model give an input coordinate
        system.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input signature or real DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free output signature, retaining concrete leading dimensions.

        Raises
        ------
        ValueError
            If input_coords are not valid
        """
        pass

    def to(self, device: Any) -> PrognosticModel:
        """Moves prognostic model onto inference device, this is typically satisfied via
        `torch.nn.Module`.

        Parameters
        ----------
        device : Any
            Object representing the inference device, typically `torch.device` or str

        Returns
        -------
        PrognosticModel
            Returns instance of prognostic
        """
        pass


# --8<-- [end:prognostic-model-interface]
