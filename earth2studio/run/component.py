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

"""Components: the unit a plan executes.

A component is one model behind a metadata-only :class:`ComponentSpec` and a
steppable :class:`ComponentSession`. The default single-model plan runs exactly
one; a compiled graph runs several. Both read the same declarations, so what a
model requires and publishes is derived once, here, by its adapter.

:class:`PrognosticComponent` adapts any iterator-based prognostic model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import xarray as xr

from earth2studio.models.px.base import ModelState, PrognosticModel, RNGState
from earth2studio.run.schedules import FixedCadence, Schedule
from earth2studio.run.session import CheckpointCapability, OutputPort
from earth2studio.utils.type import CoordinateSystem


@dataclass(frozen=True)
class FieldRequirement:
    """Describe a component input without choosing its provider.

    A single-model plan resolves every requirement to its data source, fetched
    live or read from a cache; a graph may instead resolve one to an edge from
    another component. ``signature`` is an allocation-free DataArray coordinate
    signature; variable labels resolve canonical units through the shared
    lexicon. ``fallback`` names a provider registration, never a value.
    """

    slot: str
    variables: tuple[str, ...]
    signature: CoordinateSystem
    phase: Literal["initialize", "step"]
    schedule: Schedule
    lead_offsets: tuple[np.timedelta64, ...] = ()
    optional: bool = False
    fallback: str | None = None
    freshness: np.timedelta64 | None = None


@dataclass(frozen=True)
class ComponentSpec:
    """Metadata-only description of one scheduled component.

    Constructible without loading weights or fetching data. ``schedule`` is when
    :meth:`ComponentSession.step` runs, which may differ from the valid-time
    pattern an output port publishes.
    """

    name: str
    inputs: tuple[FieldRequirement, ...]
    outputs: tuple[OutputPort, ...]
    schedule: Schedule
    checkpoint_capability: CheckpointCapability
    supports_member_batching: bool = True


@dataclass(frozen=True)
class ComponentSnapshot:
    """Complete state needed to resume one component session."""

    model_state: ModelState
    rng_state: RNGState
    step_index: int
    adapter_state: Mapping[str, xr.DataArray] = field(default_factory=dict)


class ComponentSession(Protocol):
    """Advance one component at its scheduled activation times."""

    @property
    def checkpoint_capability(self) -> CheckpointCapability:
        """Report whether state can be restored."""
        ...

    def step(
        self,
        activation_time: np.datetime64,
        inputs: Mapping[str, xr.DataArray],
    ) -> Mapping[str, xr.DataArray]:
        """Return valid DataArrays keyed by declared output port name."""
        ...

    def snapshot(self) -> ComponentSnapshot:
        """Capture all state needed to continue this session."""
        ...

    def restore(self, snapshot: ComponentSnapshot) -> None:
        """Restore a compatible snapshot before advancing again."""
        ...

    def close(self) -> None:
        """Release live resources; state survives and stepping may resume."""
        ...


class ComponentAdapter(Protocol):
    """Translate a model or callable into a scheduled component session.

    Standard adapters should cover steppable prognostics, iterator-only
    prognostics, single-call models, stateless diagnostics, and plain callables.
    A specialized adapter must still expose real component step boundaries rather
    than run a fused model twice.
    """

    @property
    def spec(self) -> ComponentSpec:
        """Return metadata without opening the component."""
        ...

    def open(
        self,
        initial: Mapping[str, xr.DataArray],
        *,
        rng: RNGState,
    ) -> ComponentSession:
        """Create one session for one work item or member group.

        ``initial`` is keyed by ``initialize``-phase slot. It may be empty when a
        snapshot will be restored before the first step.
        """
        ...


def spec_fingerprint(spec: ComponentSpec) -> str:
    """Return a stable declaration string for folding into plan identity."""
    inputs = (
        f"in:{r.slot}:{r.phase}:{','.join(r.variables)}:{r.schedule.fingerprint()}"
        for r in spec.inputs
    )
    outputs = (f"out:{p.name}:{','.join(p.variables)}" for p in spec.outputs)
    return "|".join(
        [
            spec.name,
            spec.schedule.fingerprint(),
            spec.checkpoint_capability.value,
            *inputs,
            *outputs,
        ]
    )


def _labels(signature: CoordinateSystem, dim: str) -> tuple[str, ...]:
    return tuple(str(value) for value in signature[dim].values)


class PrognosticComponent:
    """Adapt an iterator-based prognostic model into a scheduled component.

    The spec is derived from the model's coordinate contract: one
    ``initial_condition`` requirement at the model's input lead times, one
    ``forecast`` port, and a fixed cadence of the model's step. Activation ``k``
    publishes step ``k``, so the first publishes the initial condition.

    Resume re-seeds the iterator from the last published step. That is exact
    only when the published step is the model's complete recurrent state: a
    single input lead time and identical input and output variables. Other
    models report ``UNSUPPORTED``; they need explicit state
    (:class:`~earth2studio.models.px.base.SteppablePrognosticModel`) to resume.
    RNG state is not captured.

    Parameters
    ----------
    model : PrognosticModel
        Model to adapt.
    name : str, optional
        Component name, by default ``"model"``.
    """

    def __init__(self, model: PrognosticModel, name: str = "model") -> None:
        self.model = model
        input_signature = model.input_coords()
        output_signature = model.output_coords(input_signature)
        step = (
            output_signature["lead_time"].values[-1]
            - input_signature["lead_time"].values[-1]
        )
        self.cadence = FixedCadence(step)

        single_lead = input_signature.sizes["lead_time"] == 1
        same_variables = _labels(input_signature, "variable") == _labels(
            output_signature, "variable"
        )
        capability = (
            CheckpointCapability.SNAPSHOTTABLE
            if single_lead and same_variables
            else CheckpointCapability.UNSUPPORTED
        )
        self.initial_condition = FieldRequirement(
            "initial_condition",
            _labels(input_signature, "variable"),
            input_signature,
            "initialize",
            self.cadence,
            lead_offsets=tuple(input_signature["lead_time"].values),
        )
        self._spec = ComponentSpec(
            name,
            (self.initial_condition,),
            (
                OutputPort(
                    "forecast",
                    _labels(output_signature, "variable"),
                    output_signature,
                    self.cadence,
                ),
            ),
            self.cadence,
            capability,
            supports_member_batching=False,
        )

    @property
    def spec(self) -> ComponentSpec:
        """Return metadata without opening the component."""
        return self._spec

    def open(
        self,
        initial: Mapping[str, xr.DataArray],
        *,
        rng: RNGState = None,
    ) -> PrognosticComponentSession:
        """Create one session seeded from the ``initial_condition`` slot."""
        return PrognosticComponentSession(
            self.model,
            self._spec.checkpoint_capability,
            initial.get("initial_condition"),
        )


class PrognosticComponentSession:
    """Step an iterator-based prognostic model one activation at a time.

    Parameters
    ----------
    model : PrognosticModel
        Model to roll out.
    checkpoint_capability : CheckpointCapability
        Whether re-seeding from the last published step is exact.
    initial : xr.DataArray | None
        Initial condition, or ``None`` if a snapshot will be restored first.
    """

    def __init__(
        self,
        model: PrognosticModel,
        checkpoint_capability: CheckpointCapability,
        initial: xr.DataArray | None,
    ) -> None:
        self.model = model
        self._checkpoint_capability = checkpoint_capability
        self.state = initial
        self.step_index = 0
        self._iterator: Iterator[xr.DataArray] | None = None

    @property
    def checkpoint_capability(self) -> CheckpointCapability:
        """Report whether state can be restored."""
        return self._checkpoint_capability

    def step(
        self,
        activation_time: np.datetime64,
        inputs: Mapping[str, xr.DataArray],
    ) -> Mapping[str, xr.DataArray]:
        """Publish the next step; the first activation publishes the seed."""
        if inputs:
            raise ValueError("Iterator-based prognostics take no step-time inputs")
        if self._iterator is None:
            if self.state is None:
                raise RuntimeError("Session has neither an initial condition nor state")
            self._iterator = self.model.create_iterator(self.state)
            seed = next(self._iterator)  # iterators re-yield their seed first
            if self.step_index == 0:
                return self._advance(seed)
        return self._advance(next(self._iterator))

    def _advance(self, x: xr.DataArray) -> Mapping[str, xr.DataArray]:
        self.state = x
        self.step_index += 1
        return {"forecast": x}

    def snapshot(self) -> ComponentSnapshot:
        """Capture the last published step as the recurrent state."""
        if self.state is None:
            raise RuntimeError("Nothing to snapshot before the first step")
        return ComponentSnapshot({"state": self.state}, None, self.step_index)

    def restore(self, snapshot: ComponentSnapshot) -> None:
        """Resume after ``snapshot.step_index`` published steps."""
        self.close()
        self.state = snapshot.model_state["state"]
        self.step_index = snapshot.step_index

    def close(self) -> None:
        """Close the live iterator; the next step re-seeds it from state."""
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if close is not None:
                close()
            self._iterator = None
