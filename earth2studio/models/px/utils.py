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
from typing import TYPE_CHECKING, Any, cast

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource
    from earth2studio.models.px.base import PrognosticModel

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
    This mixin supplies no public execution methods; their fixed signatures belong
    to the concrete wrappers. The variadic methods below are private helpers.
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

    def _validate_initial_inputs(self, *x: xr.DataArray) -> None:
        model = cast("PrognosticModel", self)
        expected = _count(model.input_coords()) + _count(self.forcing_coords())
        if len(x) != expected:
            raise ValueError(
                f"{type(self).__name__} requires {expected} input and forcing arrays; "
                f"received {len(x)}"
            )

    def _default_call(
        self, *x: xr.DataArray
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        self._validate_initial_inputs(*x)
        return cast("PrognosticModel", self).initialize(*x)[0]

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
        # Concrete wrappers supply fixed signatures; slot declarations govern
        # the dynamic dispatch through the protocol inside this helper only.
        model = cast("PrognosticModel", self)
        y, state = model.initialize(*x)
        while True:
            y = self.rear_hook(y)
            forcing = yield y
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
            y = self.front_hook(y)
            outputs = y if isinstance(y, tuple) else (y,)
            y, state = model.step(*outputs, *forcing, state=state)
