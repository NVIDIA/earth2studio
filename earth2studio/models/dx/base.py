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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource


@runtime_checkable
class DiagnosticModel(Protocol):
    """Diagnostic model interface

    Each input slot is a separate positional DataArray, in the order declared by
    ``input_coords()``. Concrete models must declare a fixed number of named input
    parameters, not variadic inputs. The protocol uses ``*x`` only to represent
    fixed signatures whose arity differs between models. Generic callers use
    slot order rather than parameter names. One output
    is returned directly; multiple outputs are returned as a tuple in the order
    declared by ``output_coords()``. Diagnostics have no forcing slots.
    Diagnostics define ``default_sources()``, returning a source for one input
    slot, a tuple of sources or ``None`` per slot, or ``None`` for no recommendations.
    Drivers read it through
    ``earth2studio.models.utils.recommended_sources``.
    """

    def __call__(self, *x: xr.DataArray) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Execution of the diagnostic model that transforms physical data

        Parameters
        ----------
        *x : xr.DataArray
            xarray DataArrays with array data on the model's device: one positional
            argument per ``input_coords()`` slot, in declared order. Pass multiple slots as
            separate arguments, not as a single tuple argument.

        Returns
        -------
        xr.DataArray | tuple[xr.DataArray, ...]
            Diagnostic output matching ``output_coords()``, or a tuple of outputs
            in declared output-slot order.
        """
        pass

    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray input signature, or a tuple of signatures
            in input-slot order.
        """
        pass

    def output_coords(
        self, *input_coords: CoordinateSystem
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Output coordinate system of the diagnostic model.

        Parameters
        ----------
        *input_coords : CoordinateSystem
            One input signature or DataArray per ``input_coords()`` slot, passed
            as separate positional arguments in declared order. Concrete models
            declare a fixed number of named parameters, matching their input slots.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray output signature, or a tuple of signatures
            in output-slot order.

        Raises
        ------
        ValueError
            If the input coordinates are not valid.
        """
        pass

    def default_sources(
        self,
    ) -> (
        DataSource
        | ForecastSource
        | tuple[DataSource | ForecastSource | None, ...]
        | None
    ):
        """Recommended data sources for each input slot.

        Return one source directly for a single input slot, or a tuple in
        ``input_coords()`` order for multiple slots. ``None`` means no
        recommendations for the model; a ``None`` tuple entry means no
        recommendation for that slot.
        A source whose native grid differs from the slot's is
        returned composed with the recommended regridder; transforms intrinsic
        to the model, whatever the provider, stay inside the wrapper. Models
        never fetch from these themselves.

        Returns
        -------
        DataSource | ForecastSource | tuple[DataSource | ForecastSource | None, ...] | None
            A single source, one source or ``None`` per input slot, or no recommendations.
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
