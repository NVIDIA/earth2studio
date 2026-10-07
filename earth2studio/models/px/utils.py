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

from collections.abc import Callable, Generator
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource

Hook = Callable[
    [xr.DataArray | tuple[xr.DataArray, ...]],
    xr.DataArray | tuple[xr.DataArray, ...],
]


def _count(signature: Any) -> int:
    if signature is None:
        return 0
    return len(signature) if isinstance(signature, tuple) else 1


def initial_condition(
    x: xr.DataArray | tuple[xr.DataArray, ...],
) -> xr.DataArray | tuple[xr.DataArray, ...]:
    """Initial condition of a rollout: each input slot at its final lead time.

    ``create_iterator`` yields forecasts only; drivers that publish the starting
    fields take them from the inputs with this helper.

    Parameters
    ----------
    x : xr.DataArray | tuple[xr.DataArray, ...]
        Initial fields matching ``input_coords()``.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, ...]
        ``x`` reduced to its final ``lead_time`` entry, keeping the dimension.
    """
    if isinstance(x, tuple):
        return tuple(slot.isel(lead_time=[-1]) for slot in x)
    return x.isel(lead_time=[-1])


class PrognosticMixin:
    """DataArray iterator hooks around core advances and forecast outputs.

    Hooks take and return the model's output payload (``y``) in the original
    leading dimensions. Hooks belong to the iterator, which is the only path that
    owns a rollout loop. ``__call__`` is the single-step primitive and does not apply
    them: a caller holding a single step can transform the DataArray itself, whereas
    nothing outside ``create_iterator`` can reach the outputs fed back between
    steps.

    ``front_hook``/``rear_hook`` are single callable slots, not a registration
    list: a caller with more than one transformation to apply composes them into
    one function and assigns that, so the order they run in is visible at the
    assignment site rather than spread across every place that registered one.

    Wrappers must declare explicit, fixed signatures for ``__call__``,
    ``initialize``, ``step`` and ``create_iterator`` (contract rule P24).
    The variadic methods below are generic helpers or stubs, not concrete wrapper
    signatures. All four public execution methods are stubs to override.
    Explicit ``__call__`` and ``create_iterator`` methods may delegate to
    ``_default_call`` and ``_default_create_iterator``, respectively. These helpers
    use ``initialize``/``step``; the iterator yields forecasts only and accepts
    time-varying forcing through ``send``. Static forcing is initialization-only.
    Inheriting this mixin is optional; implementations must satisfy the prognostic
    protocol regardless of how their methods are implemented.
    """

    #: Whether the model draws randomness during a rollout. Stochastic models must
    #: implement ``set_rng(seed, reset=True)``; see ``dev/spec/MODEL_CONTRACT_SPEC.md``.
    stochastic: bool = False

    #: Number of forecast outputs produced by each front-hook/core advance.
    #: Rear hooks run for every output; multi-output cores declare their cadence.
    #: Unused by the derived iterator, where one yield carries a whole step.
    front_hook_interval: int = 1

    @staticmethod
    def _default_hook(
        x: xr.DataArray | tuple[xr.DataArray, ...],
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
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

    def initialize(
        self,
        *x: xr.DataArray,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Start a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def step(
        self,
        *y: xr.DataArray,
        state: Any,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Advance a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def forcing_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...] | None:
        """Declare no forcing."""
        return None

    def default_sources(
        self,
    ) -> (
        DataSource
        | ForecastSource
        | tuple[DataSource | ForecastSource | None, ...]
        | None
    ):
        """Recommend no source for any input or forcing slot."""
        return None

    def __call__(self, *x: xr.DataArray) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Predict a forecast; wrappers implement this, optionally via ``_default_call``."""
        raise NotImplementedError(f"{type(self).__name__} must implement __call__")

    def _validate_initial_inputs(self, *x: xr.DataArray) -> None:
        expected = _count(self.input_coords()) + _count(self.forcing_coords())  # type: ignore[attr-defined]
        if len(x) != expected:
            raise ValueError(
                f"{type(self).__name__} requires {expected} input and forcing arrays; "
                f"received {len(x)}"
            )

    def _default_call(
        self, *x: xr.DataArray
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        self._validate_initial_inputs(*x)
        return self.initialize(*x)[0]

    def create_iterator(
        self,
        *x: xr.DataArray,
    ) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        xr.DataArray | tuple[xr.DataArray, ...] | None,
        None,
    ]:
        """Create a forecast iterator; wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_iterator"
        )

    def _default_create_iterator(self, *x: xr.DataArray) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        xr.DataArray | tuple[xr.DataArray, ...] | None,
        None,
    ]:
        self._validate_initial_inputs(*x)
        signatures = self.forcing_coords()
        forcing_slots = (
            ()
            if signatures is None
            else signatures if isinstance(signatures, tuple) else (signatures,)
        )
        expected = sum("lead_time" in slot.dims for slot in forcing_slots)
        y, state = self.initialize(*x)
        while True:
            # Custom hooks may write in place; isolate publication and feedback.
            forcing = yield self.rear_hook(
                y if self.rear_hook is self._default_hook else deepcopy(y)
            )
            forcing = (
                ()
                if forcing is None
                else forcing if isinstance(forcing, tuple) else (forcing,)
            )
            if len(forcing) != expected:
                raise ValueError(
                    f"{type(self).__name__} requires {expected} forcing arrays per step; "
                    f"received {len(forcing)}"
                )
            feedback = self.front_hook(
                y if self.front_hook is self._default_hook else deepcopy(y)
            )
            outputs = feedback if isinstance(feedback, tuple) else (feedback,)
            y, state = self.step(*outputs, *forcing, state=state)
