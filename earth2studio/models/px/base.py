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

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource


def recommended_sources(
    model: Any,
) -> tuple[DataSource | ForecastSource | None, ...]:
    """Recommended source for each input and forcing slot of a model.

    Prognostic models always declare ``default_sources()``; for diagnostics it is
    optional, and models without it recommend nothing.

    Parameters
    ----------
    model : PrognosticModel | DiagnosticModel
        Model whose slots need providers.

    Returns
    -------
    tuple[DataSource | ForecastSource | None, ...]
        One entry per ``input_coords()`` slot, then one per ``forcing_coords()``
        slot.
    """
    declared = getattr(model, "default_sources", None)
    if declared is not None:
        return declared()
    count = 0
    for name in ("input_coords", "forcing_coords"):
        method = getattr(model, name, None)
        signature = method() if method is not None else None
        if signature is not None:
            count += len(signature) if isinstance(signature, tuple) else 1
    return (None,) * count


# --8<-- [start:prognostic-model-interface]
@runtime_checkable
class PrognosticModel(Protocol):
    """Prognostic model interface

    ``initialize`` and ``step`` are the primitives. ``__call__`` and
    ``create_iterator`` derive from them and are supplied by ``PrognosticMixin``,
    so a wrapper implements the transition once and both entry points agree.

    Each argument group is one argument shaped like the coordinate method that
    describes it: a DataArray when that method returns one ``CoordinateSystem``,
    a tuple when it returns a tuple.

    - ``x``: initial fields, described by ``input_coords()``
    - ``forcing``: external fields, described by ``forcing_coords()``: the full
      window at ``initialize``, the newest frames at each ``step``; ``None`` for
      models without forcing
    - ``y``: outputs, described by ``output_coords()``
    - ``state``: everything else a rollout needs, in a model-defined type
    """

    def __call__(
        self,
        x: xr.DataArray | tuple[xr.DataArray, ...],
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Forward pass of the prognostic model, time integrating a single time-step

        Equivalent to ``initialize(x, forcing)`` followed by one ``step`` with the
        same ``forcing``. Applies no hooks.

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            NumPy-backed CPU or CuPy-backed CUDA fields matching ``input_coords()``.
        forcing : xr.DataArray | tuple[xr.DataArray, ...] | None, optional
            Fields matching ``forcing_coords()``. Required when it is not ``None``.

        Returns
        -------
        xr.DataArray | tuple[xr.DataArray, ...]
            Outputs one time-step into the future, matching ``output_coords()``.
        """
        pass

    def create_iterator(
        self,
        x: xr.DataArray | tuple[xr.DataArray, ...],
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        xr.DataArray | tuple[xr.DataArray, ...] | None,
        None,
    ]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        A value sent at a yield is the forcing for the next advance, as ``step``
        takes it. ``next(it)`` is ``send(None)``: at the 0th yield it reuses
        ``forcing``; later it is valid only for models without forcing.

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            Initial fields matching ``input_coords()``.
        forcing : xr.DataArray | tuple[xr.DataArray, ...] | None, optional
            Initial forcing window matching ``forcing_coords()``.

        Yields
        ------
        xr.DataArray | tuple[xr.DataArray, ...]
            Initial condition followed by successive forecasts.
        """
        pass

    def initialize(
        self,
        x: xr.DataArray | tuple[xr.DataArray, ...],
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Start a rollout from initial fields and forcing.

        Splits the input window into the latest fields, returned as ``y``, and a
        model-defined state holding everything else the rollout needs: older
        input and forcing frames, statics, latents, noise states and the RNG
        position derived from the model's seed. The state must be serializable
        (arrays, tensors, scalars, DataArrays, or dataclasses and tuples of them),
        or ``None``.

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            Initial fields matching ``input_coords()``.
        forcing : xr.DataArray | tuple[xr.DataArray, ...] | None, optional
            Initial forcing window matching ``forcing_coords()``. Required when it
            is not ``None``.

        Returns
        -------
        tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]
            ``y``, the initial condition reduced to the final input lead time (the
            iterator's 0th yield), and the state. Later ``y`` match
            ``output_coords()``, so ``step`` accepts both forms.
        """
        pass

    def step(
        self,
        y: xr.DataArray | tuple[xr.DataArray, ...],
        state: Any,
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Advance a rollout by one core computation.

        Modifies neither ``y`` nor ``state``: the same arguments give the same
        result, and earlier outputs stay valid. ``(y, state)`` together are the
        complete continuation of a rollout.

        Parameters
        ----------
        y : xr.DataArray | tuple[xr.DataArray, ...]
            Outputs of ``initialize`` or the previous ``step``.
        state : Any
            State returned alongside ``y``.
        forcing : xr.DataArray | tuple[xr.DataArray, ...] | None, optional
            Newest forcing frames, shaped like ``forcing_coords()``: at least its
            final lead time shifted by each lead time of ``y``. ``step`` selects
            by lead time and keeps older frames in ``state``, so extra frames are
            ignored. Static slots were read at ``initialize`` and may be ``None``.
            Required when ``forcing_coords()`` is not ``None``.

        Returns
        -------
        tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]
            Every output of this core computation, matching ``output_coords()``,
            and the next state. A model computing several lead times per core call
            returns them together; one ``step`` is one iterator yield.

        Raises
        ------
        ValueError
            If required forcing is missing or fails its slot handshakes, or ``y``
            does not continue ``state``.
        """
        pass

    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray input signature with lead times relative to
            initialization, or a tuple of signatures, one per input slot.
        """
        pass

    def forcing_coords(
        self,
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...] | None:
        """Forcing coordinate system of the prognostic model.

        Describes the forcing window the model consumes, with lead times relative
        to initialization, as for ``input_coords()``. ``initialize`` takes the whole
        window; each ``step`` takes only the newest frames, at the window's final
        lead time shifted by the lead time of the ``y`` being advanced. Static
        fields the caller must supply are forcing slots without ``lead_time``.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...] | None
            Allocation-free forcing signature, a tuple of signatures, one per
            forcing slot, or ``None`` for models without forcing.
        """
        pass

    def output_coords(
        self, input_coords: CoordinateSystem | tuple[CoordinateSystem, ...]
    ) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem | tuple[CoordinateSystem, ...]
            Input signature(s) or DataArray(s) to validate and transform.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free output signature(s), retaining concrete leading
            dimensions.

        Raises
        ------
        ValueError
            If the input coordinates are not valid.
        """
        pass

    def default_sources(self) -> tuple[DataSource | ForecastSource | None, ...]:
        """Recommended data sources for each input and forcing slot.

        One entry per ``input_coords()`` slot, then one per ``forcing_coords()``
        slot. ``None`` means no recommendation. A source whose native grid differs
        from the slot's is returned composed with the recommended regridder;
        transforms intrinsic to the model, whatever the provider, stay inside the
        wrapper. Models never fetch from these themselves.

        Returns
        -------
        tuple[DataSource | ForecastSource | None, ...]
            One entry per input slot, then one per forcing slot.
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
