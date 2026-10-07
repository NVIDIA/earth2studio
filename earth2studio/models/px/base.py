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


@runtime_checkable
class PrognosticModel(Protocol):
    """DataArray-based prognostic model interface.

    Implementations must satisfy this protocol's interface and behavior.
    ``PrognosticMixin`` optionally implements ``__call__`` using ``initialize``;
    ``create_iterator`` is currently a stub. Inheriting the mixin is not required.

    Notes
    -----
    The authoritative requirements are in ``dev/spec/MODEL_CONTRACT_SPEC.md``.

    **Arguments and slots (P17)**
        Each declared slot is a separate positional DataArray. Initial arguments
        ``*x`` follow input-slot order, then forcing-slot order, if forcing exists.
        Step arguments ``*y`` follow output-slot order, then time-varying forcing
        slot order; initialization-only static slots are omitted.
        Simple models use ``x`` and ``y``; complex models may give individual
        parameters descriptive names. One output is returned directly; multiple
        outputs are returned as a tuple in declared order.

    **Forecasts and iteration (P7-P9, P20)**
        ``initialize`` computes the first forecast; ``step`` computes the next.
        Each performs one core computation and returns all its lead times together.
        ``create_iterator`` yields complete forecasts, starting with the output of
        ``initialize``. Drivers publish the initial condition separately if needed.
        Without hooks, ``__call__(*x)``, the forecast from ``initialize(*x)``, and
        the first iterator yield agree. Outputs match declared coordinates and
        structural metadata.

    **Hooks (P10)**
        Only iteration applies hooks. The front hook acts on forecast outputs
        before each ``step``, never before ``initialize``; its edits feed back.
        The rear hook changes published outputs only, including the first forecast.

    **Continuation and ownership (P15, P16, P19, P21)**
        The forecast and model-defined state together form a complete continuation.
        State is serializable and holds additional history, latents or RNG state;
        no state base class is required. It avoids duplicating outputs unless their
        published representation is unsuitable for recurrence. Execution borrows
        arrays and coordinates without modifying them. ``step`` also preserves its
        supplied state; replaying the same continuation and external arrays gives
        the same result. Earlier outputs remain stable after later advances.

    **Forcing and sources (P22, P23)**
        Models never fetch data. Callers supply complete forcing windows for
        initialization and new frames for subsequent steps, if forcing exists.
        Missing required forcing raises ``ValueError``. Iterator ``send`` supplies
        forcing for the next step; unforced models support ordinary ``next``.
        Default sources follow input-slot order, then forcing-slot order.

    **Randomness (P11-P14)**
        Models declare ``stochastic``. Stochastic models expose ``set_rng``;
        seeded rollouts are reproducible without perturbing global RNG state.
        Continuation state captures the rollout's RNG state for replay and resume.
    """

    def __call__(self, *x: xr.DataArray) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Advance the prognostic model by one core computation.

        Equivalent to the forecast from ``initialize(*x)``. Applies no hooks.

        Parameters
        ----------
        *x : xr.DataArray
            NumPy-backed CPU or CuPy-backed CUDA arrays: one argument per
            ``input_coords()`` slot, followed by one per ``forcing_coords()`` slot,
            if any exist, in declared order. Forcing uses complete initial windows.
            This concatenates argument sequences, not array contents.

        Returns
        -------
        xr.DataArray | tuple[xr.DataArray, ...]
            Forecast outputs matching ``output_coords()``, including all lead times
            produced by this core computation.
        """
        pass

    def create_iterator(
        self,
        *x: xr.DataArray,
    ) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        tuple[xr.DataArray, ...] | None,
        None,
    ]:
        """Creates an iterator which time-integrates the prognostic model.

        Yields forecasts only, starting with the outputs of ``initialize``; drivers
        publishing the initial condition take it manually from ``x``.
        ``nsteps`` forecasts take ``nsteps`` yields. A value sent at a yield is the
        forcing arrays for the next ``step``, if any exist, as a tuple in declared
        forcing-slot order, omitting initialization-only static slots.
        ``next(it)`` is ``send(None)``, valid when no new forcing is required.

        Parameters
        ----------
        *x : xr.DataArray
            Same arguments as ``__call__`` and ``initialize``: one array per
            ``input_coords()`` slot, followed by the full initialization windows
            for ``forcing_coords()`` slots, if any exist, in declared order.

        Yields
        ------
        xr.DataArray | tuple[xr.DataArray, ...]
            Successive forecasts, each matching ``output_coords()``.
        """
        pass

    def initialize(
        self,
        *x: xr.DataArray,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Start a rollout from initial fields and forcing.

        Computes the first forecast, returned as ``y``, and a model-defined state
        holding everything else the rollout needs: older input and forcing frames,
        statics, latents, noise states and the RNG position, drawn from the
        model's seed. Applies no hooks. The state must be serializable
        (arrays, tensors, scalars, DataArrays, or dataclasses and tuples of them),
        or ``None``.

        Parameters
        ----------
        *x : xr.DataArray
            One argument per ``input_coords()`` slot, followed by one per
            ``forcing_coords()`` slot, if any exist, in declared order. Forcing
            arrays contain the complete initialization windows. This concatenates
            argument sequences, not array contents.

        Returns
        -------
        tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]
            ``y``, the first forecast matching ``output_coords()``, and the state.
        """
        pass

    def step(
        self,
        *y: xr.DataArray,
        state: Any,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Advance a rollout by one core computation.

        Modifies neither ``y`` nor ``state``: the same arguments give the same
        result, and earlier outputs stay valid. ``(y, state)`` together are the
        complete continuation of a rollout.

        Parameters
        ----------
        *y : xr.DataArray
            One argument per ``output_coords()`` slot from ``initialize`` or the
            preceding ``step``, followed by one per ``forcing_coords()`` slot,
            if any exist, in declared order. Unpack multiple outputs into separate
            arguments. Forcing supplies at least its final declared lead time
            shifted by each forecast lead time; extra frames are ignored. Older
            frames and initialization-only statics are retained in ``state``.
            Omit static forcing slots after initialization.
        state : Any
            State returned alongside the forecast, passed as a keyword argument.

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
            initialization, or a tuple of signatures in input-slot order.
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
            Input signatures or DataArrays to validate and transform.

        Returns
        -------
        CoordinateSystem | tuple[CoordinateSystem, ...]
            Allocation-free output signatures, retaining concrete leading dimensions.

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
        """Recommended data sources for each input and forcing slot.

        Return one source directly for a single slot, or a tuple with one source
        per ``input_coords()`` slot, then one per ``forcing_coords()`` slot.
        ``None`` means no recommendations for the model; a ``None`` tuple entry
        means no recommendation for that slot. A source whose native grid differs
        from the slot's is returned composed with the recommended regridder;
        transforms intrinsic to the model, whatever the provider, stay inside the
        wrapper. Models never fetch from these themselves.

        Returns
        -------
        DataSource | ForecastSource | tuple[DataSource | ForecastSource | None, ...] | None
            A single source, sources or ``None`` in input-then-forcing slot order, or no
            recommendations.
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
