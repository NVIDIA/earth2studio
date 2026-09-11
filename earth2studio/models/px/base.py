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

import torch

from earth2studio.utils.coordinate import CoordinateSystem
from earth2studio.utils.type import CoordSystem


# --8<-- [start:prognostic-model-interface]
@runtime_checkable
class PrognosticModel(Protocol):
    """Prognostic model interface"""

    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of the prognostic model, time integrating a single time-step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply diagnostic function on
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate dictionary one time-step into the future
        """
        pass

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, which can be viewed as the initial state of the prognositc
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        pass

    def input_coords(self) -> CoordinateSystem:
        """Return ordered allocation-free input coordinate signatures.

        Returns
        -------
        tuple[xr.DataArray, ...]
            Ordered input signatures
        """
        pass

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Return ordered output signatures for one forecast step.

        Parameters
        ----------
        input_coords : tuple[xr.DataArray, ...]
            Ordered input signatures

        Returns
        -------
        tuple[xr.DataArray, ...]
            Ordered output signatures

        Raises
        ------
        ValueError
            If input_coords are not valid
        """
        pass

    def _input_tensor_coords(self) -> CoordSystem:
        """Return coordinates used by the tensor execution path."""
        pass

    def _output_tensor_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Return tensor output coordinates for one forecast step."""
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
