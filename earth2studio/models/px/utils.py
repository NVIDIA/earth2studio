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

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import xarray as xr

from earth2studio.models.px.base import ModelState, SourceDefault
from earth2studio.utils.type import CoordinateSystem

Hook = Callable[[xr.DataArray], xr.DataArray]


def _as_tuple(value: Any) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _grid_key(signature: CoordinateSystem) -> Any:
    """Grid identity of a slot: its registered grid ID, else its spatial axes."""
    grid_id = signature.attrs.get("earth2studio_grid_id")
    if grid_id is not None:
        return grid_id
    return tuple(
        (dim, signature.sizes[dim])
        for dim in signature.dims
        if dim not in ("variable", "lead_time") and signature.sizes[dim] > 0
    )


def input_roles(model: Any) -> tuple[str, ...]:
    """Derive the role of each input slot from a model's signatures.

    A variable is state when an output slot on the same grid produces it, a step
    input when it is input-only with ``lead_time``, and static otherwise. Every
    variable in a slot must share one role.

    Parameters
    ----------
    model : PrognosticModel
        Model whose ``input_coords()`` and ``output_coords()`` are inspected.

    Returns
    -------
    tuple[str, ...]
        ``"state"``, ``"step"`` or ``"static"`` for each input slot.

    Raises
    ------
    ValueError
        If an input slot mixes roles.
    """
    inputs = _as_tuple(model.input_coords())
    outputs = _as_tuple(model.output_coords(model.input_coords()))
    produced = {
        (variable, _grid_key(signature))
        for signature in outputs
        for variable in signature["variable"].values
    }
    roles = []
    for index, signature in enumerate(inputs):
        kinds = {
            (
                "state"
                if (variable, _grid_key(signature)) in produced
                else "step" if "lead_time" in signature.dims else "static"
            )
            for variable in signature["variable"].values
        }
        if len(kinds) != 1:
            raise ValueError(f"input slot {index} mixes roles {sorted(kinds)}")
        roles.append(kinds.pop())
    return tuple(roles)


class PrognosticMixin:
    """DataArray iterator hooks around core advances and forecast outputs.

    Hooks take and return one DataArray in the original leading dimensions.
    Hooks belong to the iterator, which is the only path that owns a rollout loop.
    ``__call__`` is the single-step primitive and does not apply them: a caller
    holding a single step can transform the DataArray itself, whereas nothing outside
    ``create_iterator`` can reach the state fed back between steps.

    ``front_hook``/``rear_hook`` are single callable slots, not a registration
    list: a caller with more than one transformation to apply composes them into
    one function and assigns that, so the order they run in is visible at the
    assignment site rather than spread across every place that registered one.
    """

    #: Whether the model draws randomness during a rollout. Stochastic models must
    #: implement ``set_rng(seed, reset=True)``; see ``dev/spec/MODEL_CONTRACT_SPEC.md``.
    stochastic: bool = False

    #: Number of forecast outputs produced by each front-hook/core advance.
    #: Rear hooks run for every output; multi-output cores declare their cadence.
    #: Unused by the derived iterator, where one yield carries a whole step.
    front_hook_interval: int = 1

    @staticmethod
    def _default_hook(x: xr.DataArray) -> xr.DataArray:
        return x

    # Typed as the Hook signature so assigning a plain function — the normal use
    # of this slot — type-checks instead of looking like a method override.
    front_hook: Hook = _default_hook
    rear_hook: Hook = _default_hook

    def clear_hooks(self) -> None:
        """Remove every registered hook, restoring the default pass-through."""
        for name in ("front_hook", "rear_hook"):
            if name in vars(self):
                delattr(self, name)

    # Proposed explicit-state protocol. Wrappers that still define their own
    # ``__call__``/``create_iterator`` override the derived versions below, and
    # inherit ``initialize``/``step`` stubs until they are migrated.

    def initialize(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]:
        """Start a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def step(
        self,
        state: ModelState,
        inputs: tuple[xr.DataArray | None, ...] | None = None,
    ) -> tuple[ModelState, xr.DataArray | tuple[xr.DataArray, ...]]:
        """Advance a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def default_sources(self) -> tuple[SourceDefault | None, ...]:
        """Recommend no source for any input slot."""
        return (None,) * len(_as_tuple(self.input_coords()))  # type: ignore[attr-defined]

    def _step_inputs(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> tuple[xr.DataArray | None, ...] | None:
        """Select the step input slots already present in initial fields."""
        if not isinstance(x, tuple):
            return None
        roles = input_roles(self)
        if "step" not in roles:
            return None
        return tuple(field if role == "step" else None for field, role in zip(x, roles))

    def __call__(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Advance one step from ``x`` without hooks, via ``initialize``/``step``."""
        state, _ = self.initialize(x)
        return self.step(state, self._step_inputs(x))[1]

    # Annotated as an Iterator so unmigrated wrappers returning Iterator still type
    # check as overrides; the generator still accepts step inputs through ``send``.
    def create_iterator(
        self, x: xr.DataArray | tuple[xr.DataArray, ...]
    ) -> Iterator[xr.DataArray | tuple[xr.DataArray, ...]]:
        """Roll out via ``initialize``/``step``, receiving step inputs by ``send``.

        Yields the initial condition first. A value sent at a yield is the step
        inputs for the next advance; sending ``None`` (or calling ``next``) at the
        0th yield reuses the step input slots in ``x``. Each yield is one ``step``:
        a model computing several lead times per core call yields them together.
        """
        # Hooks see whatever payload type the model declares.
        front_hook: Callable[[Any], Any] = self.front_hook
        rear_hook: Callable[[Any], Any] = self.rear_hook

        state, initial = self.initialize(x)
        sent = yield initial
        pending = self._step_inputs(x) if sent is None else sent
        while True:
            state = replace(state, fields=front_hook(state.fields))
            state, out = self.step(state, pending)
            pending = yield rear_hook(out)
