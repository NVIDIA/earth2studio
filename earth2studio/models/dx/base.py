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
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from earth2studio.utils.type import CoordSystem


@runtime_checkable
class DiagnosticModel(Protocol):
    """Diagnostic model interface"""

    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Execution of the diagnostic model that transforms physical data

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply diagnostic function on
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        pass

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        pass

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the diagnostic model given an input coordinate
        system.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary

        Raises
        ------
        ValueError
            If input_coords are not valid
        """
        pass

    def to(self, device: Any) -> DiagnosticModel:
        """Moves diagnostic model onto inference device, this is typically satisfied via
        `torch.nn.Module`.

        Parameters
        ----------
        device : Any
            Object representing the inference device, typically `torch.device` or str

        Returns
        -------
        DiagnosticModel
            Returns instance of diagnostic
        """
        pass
