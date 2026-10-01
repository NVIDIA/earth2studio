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

"""Graph wiring: bindings, providers, and the graph that compiles into a plan.

Components themselves -- specs, sessions, adapters, snapshots -- are engine
types in :mod:`earth2studio.run.component`, shared with the single-model plan.
This module adds only what connecting several components needs.

Graph compilation and execution stay in this package. They reuse component
metadata and stepping primitives but do not replace the direct single-model loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
import xarray as xr

from earth2studio.run.component import ComponentAdapter, ComponentSnapshot
from earth2studio.run.session import ExecutionPlan, PortRef, ResolvedRequest

ProviderRegistry: TypeAlias = Mapping[str, object]
"""Named external provider registrations; concrete source types stay separate."""


@dataclass(frozen=True)
class ConnectorSnapshot:
    """Named connector windows and histories needed for continuation."""

    state: Mapping[str, xr.DataArray]


@dataclass(frozen=True)
class GraphSnapshot:
    """Graph-private scientific state carried as a run snapshot payload.

    This is what a compiled graph puts in
    :attr:`~earth2studio.run.session.RunSnapshot.payload`. Supervision persists
    it without inspecting it, which is why component state, connector history,
    and graph event position are declared here rather than in the shared boundary.
    """

    component_states: Mapping[str, ComponentSnapshot]
    connector_states: Mapping[str, ConnectorSnapshot]
    event_cursor: int


class TransformSpec(Protocol):
    """Metadata-only description of a value transform on a binding."""

    def fingerprint(self) -> str:
        """Return stable transform identity for execution-plan hashing."""
        ...


@dataclass(frozen=True)
class Binding:
    """Connect selected fields from one output port to one input slot.

    Scientific timing is declared here and on schedules, then compiled into an
    inspectable event order, rather than encoded by manually ordering actions.
    """

    source: PortRef
    target: PortRef
    fields: tuple[str, ...]
    availability: Literal["current", "previous"]
    transforms: tuple[TransformSpec, ...] = ()


@dataclass(frozen=True)
class ActivationContext:
    """Time context in which a bound provider supplies an input."""

    reference_time: np.datetime64
    activation_time: np.datetime64


class BoundProvider(Protocol):
    """Present a resolved source or component export to a consumer binding.

    Internal and post-resolution. Sources and components keep distinct public
    interfaces and converge only here, so a consumer never branches on which one
    supplied a value. Source-backed providers may prefetch asynchronously behind
    this synchronous boundary; a component-backed provider reads published port
    history.
    """

    def value_at(
        self,
        request: ResolvedRequest,
        context: ActivationContext,
    ) -> xr.DataArray | None:
        """Return a ready value, or ``None`` for a declared optional absence.

        ``None`` never signals an error.
        """
        ...


class ExecutionGraph(Protocol):
    """An unresolved component graph that compiles into an execution plan."""

    @property
    def components(self) -> Mapping[str, ComponentAdapter]:
        """Return the registered component adapters."""
        ...

    @property
    def bindings(self) -> tuple[Binding, ...]:
        """Return unresolved port and field connections."""
        ...

    def compile(self, providers: ProviderRegistry) -> ExecutionPlan:
        """Resolve and validate a reusable plan before model compute or allocation.

        Compilation binds providers and checks the graph once; the resulting plan
        is item-agnostic and opened per work item.
        """
        ...
