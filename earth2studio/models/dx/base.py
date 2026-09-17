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

from typing import Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem


# --8<-- [start:diagnostic-model-interface]
@runtime_checkable
class DiagnosticModel(Protocol):
    """Diagnostic model interface"""

    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Execution of the diagnostic model that transforms physical data

        Parameters
        ----------
        x : xr.DataArray
            NumPy-backed CPU or CuPy-backed CUDA data with labeled coordinates
            and the metadata required by ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Diagnostic output with labeled coordinates and output metadata.
        """
        pass

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        pass

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the diagnostic model given an input coordinate
        system.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input signature or real DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature.

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


# --8<-- [end:diagnostic-model-interface]
