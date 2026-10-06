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
    """Diagnostic model interface

    Inputs and outputs are one DataArray or a tuple of DataArrays ("slots"), shaped
    like ``input_coords()`` and ``output_coords()``, as for prognostic models.
    Diagnostics may also define ``default_sources()``, returning one ``DataSource |
    ForecastSource | None`` per input slot; it is optional here because diagnostics
    share no base class. Drivers read it through
    ``earth2studio.models.px.base.recommended_sources``.

    Until wrappers migrate, ``__call__``, ``input_coords`` and ``output_coords``
    keep single-slot annotations, so current callers type check; they widen to
    tuples with the migration.
    """

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Execution of the diagnostic model that transforms physical data

        Parameters
        ----------
        x : xr.DataArray
            NumPy-backed CPU or CuPy-backed CUDA data matching ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Diagnostic output matching ``output_coords()``.
        """
        pass

    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray input signature, or one per input slot.
        """
        pass

    def output_coords(
        self, input_coords: CoordinateSystem | tuple[CoordinateSystem, ...]
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Output coordinate system of the diagnostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray output signature, or one per output slot.

        Raises
        ------
        ValueError
            If the input coordinates are not valid.
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
