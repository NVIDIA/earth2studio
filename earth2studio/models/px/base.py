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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource


@dataclass(frozen=True)
class ModelState:
    """Per-rollout state of a prognostic model.

    Weights, configuration and the seed set by ``set_rng`` stay on the model;
    everything that advances with one rollout lives here, so a rollout can be
    snapshot, restored or branched without touching the model instance.

    ``fields`` is the public part: the state slots of ``input_coords()`` (the
    rolling input window), in slot order. Iterator front hooks read and replace it.
    Models subclass to carry private per-rollout data (latents, noise states, RNG
    position), which only the model that created the state interprets.

    Subclass fields must be arrays, tensors, scalars or nested ``ModelState``
    objects, so a generic snapshot can serialize any state. Store RNG position as
    a counter or generator state tensor, never a ``torch.Generator`` object.
    ``step`` must not modify the state it receives.

    Example
    -------
    >>> @dataclass(frozen=True)
    ... class AtlasState(ModelState):
    ...     latent: torch.Tensor
    ...     rng_step: int
    """

    fields: xr.DataArray | tuple[xr.DataArray, ...]


@dataclass(frozen=True)
class SourceDefault:
    """Recommended provider for one input slot.

    A recommendation, not a requirement: the slot's signature in ``input_coords()``
    is the requirement. Drivers compose ``source`` with ``regridder`` explicitly;
    callers may keep both, replace only the source (for example with
    ``dataclasses.replace``), or supply their own provider.

    Parameters
    ----------
    source : DataSource | ForecastSource
        Raw data source, not pre-composed with a regridder.
    regridder : Any | None, optional
        Recommended regridder from the source grid onto the slot's grid, for
        example the bilinear interpolation a model was trained on. Typed loosely
        until the ``Regridder`` ABC in ``recipes/eval/src/regrid.py`` is upstreamed.
    """

    source: DataSource | ForecastSource
    regridder: Any | None = None


def recommended_sources(model: Any) -> tuple[SourceDefault | None, ...]:
    """Recommended source for each input slot of a prognostic or diagnostic model.

    Prognostic models always declare ``default_sources()``; for diagnostics it is
    optional, and models without it recommend nothing.

    Parameters
    ----------
    model : PrognosticModel | DiagnosticModel
        Model whose input slots need providers.

    Returns
    -------
    tuple[SourceDefault | None, ...]
        One entry per input slot, aligned with ``input_coords()``.
    """
    declared = getattr(model, "default_sources", None)
    if declared is not None:
        return declared()
    signature = model.input_coords()
    return (None,) * (len(signature) if isinstance(signature, tuple) else 1)


# --8<-- [start:prognostic-model-interface]
@runtime_checkable
class PrognosticModel(Protocol):
    """Prognostic model interface

    ``initialize`` and ``step`` are the primitives. ``__call__`` and
    ``create_iterator`` derive from them and are supplied by ``PrognosticMixin``,
    so a wrapper implements the transition once and both entry points agree.
    """

    def __call__(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Forward pass of the prognostic model, time integrating a single time-step

        Equivalent to ``initialize(x)`` followed by one ``step`` using the step
        input slots already present in ``x``. Applies no hooks.

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            NumPy-backed CPU or CuPy-backed CUDA fields matching ``input_coords()``.

        Returns
        -------
        xr.DataArray | tuple[xr.DataArray, ...]
            Outputs one time-step into the future, matching ``output_coords()``.
        """
        pass

    def create_iterator(self, x: xr.DataArray | tuple[xr.DataArray, ...]) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        tuple[xr.DataArray | None, ...] | None,
        None,
    ]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        A value sent at a yield is the step inputs for the next advance, aligned
        with ``input_coords()``. ``next(it)`` is ``send(None)``: at the 0th yield it
        reuses the step input slots in ``x``; later it is valid only for models
        without step input slots and otherwise raises naming the missing slots.

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            Initial fields matching ``input_coords()``.

        Yields
        ------
        xr.DataArray | tuple[xr.DataArray, ...]
            Initial condition followed by successive forecasts.
        """
        pass

    def initialize(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]:
        """Start a rollout from initial fields.

        Captures the RNG stream for this rollout from the model's seed and derives
        any private state (for example a latent encoding of the input window).

        Parameters
        ----------
        x : xr.DataArray | tuple[xr.DataArray, ...]
            Initial fields matching ``input_coords()``.

        Returns
        -------
        tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]
            Rollout state and the initial condition reduced to the final input
            lead time (the iterator's 0th yield).
        """
        pass

    def step(
        self,
        state: ModelState,
        inputs: tuple[xr.DataArray | None, ...] | None = None,
    ) -> tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]:
        """Advance a rollout by one core computation.

        Rolls the input window internally: the returned state is ready for the
        next ``step`` without caller-side assembly. Pure with respect to
        ``state``: the same state and inputs give the same result.

        Parameters
        ----------
        state : ModelState
            State returned by ``initialize`` or a previous ``step``. Not modified.
        inputs : tuple[xr.DataArray | None, ...] | None, optional
            Step input slots valid at the state's current lead time, aligned with
            ``input_coords()``. Required when the model declares step input slots.

        Returns
        -------
        tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]
            Next state and every output of this core computation, matching
            ``output_coords()``. A model computing several lead times per core call
            returns them together; one ``step`` is one iterator yield.

        Raises
        ------
        ValueError
            If required step inputs are missing or fail their slot handshakes.
        """
        pass

    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free DataArray input signature with relative lead times, or
            a tuple of signatures, one per input slot.
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

    def default_sources(self) -> tuple[SourceDefault | None, ...]:
        """Recommended data sources for each input slot.

        Aligned with ``input_coords()``, including the initial state slots.
        ``None`` means no recommendation. Models never fetch from these
        themselves; drivers fetch next to initial-condition fetching.

        Returns
        -------
        tuple[SourceDefault | None, ...]
            One entry per input slot.
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
