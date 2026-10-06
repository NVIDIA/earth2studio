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
from typing import TYPE_CHECKING, Any

import xarray as xr

from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.data.base import DataSource, ForecastSource

Hook = Callable[[xr.DataArray], xr.DataArray]


def _count(signature: Any) -> int:
    if signature is None:
        return 0
    return len(signature) if isinstance(signature, tuple) else 1


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

    Wrappers that still define their own ``__call__``/``create_iterator`` override
    the derived versions below, and inherit ``initialize``/``step`` stubs until they
    are migrated.
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

    def initialize(
        self,
        x: xr.DataArray | tuple[xr.DataArray, ...],
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Start a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def step(
        self,
        y: xr.DataArray | tuple[xr.DataArray, ...],
        state: Any,
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> tuple[xr.DataArray | tuple[xr.DataArray, ...], Any]:
        """Advance a rollout; migrated wrappers implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to initialize/step yet"
        )

    def forcing_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...] | None:
        """Declare no forcing."""
        return None

    def default_sources(self) -> tuple[DataSource | ForecastSource | None, ...]:
        """Recommend no source for any input or forcing slot."""
        count = _count(self.input_coords()) + _count(  # type: ignore[attr-defined]
            self.forcing_coords()
        )
        return (None,) * count

    # The derived methods take ``forcing`` through ``*args``/``**kwargs`` only so that
    # unmigrated wrappers defining ``__call__(x)``/``create_iterator(x)`` still type
    # check as overrides. The protocol declares the real signature
    # ``(x, forcing=None)``; restore it here once no wrapper overrides these.

    def __call__(
        self, x: xr.DataArray | tuple[xr.DataArray, ...], *args: Any, **kwargs: Any
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Advance one step from ``x`` without hooks, via ``initialize``/``step``."""
        forcing = args[0] if args else kwargs.get("forcing")
        if forcing is None and self.forcing_coords() is not None:
            raise ValueError(f"{type(self).__name__} requires forcing")
        y, state = self.initialize(x, forcing)
        return self.step(y, state, forcing)[0]

    # Annotated as an Iterator for the same reason; the generator still accepts
    # forcing through ``send``.
    def create_iterator(
        self, x: xr.DataArray | tuple[xr.DataArray, ...], *args: Any, **kwargs: Any
    ) -> Iterator[xr.DataArray | tuple[xr.DataArray, ...]]:
        """Roll out via ``initialize``/``step``, receiving forcing by ``send``.

        Yields the initial condition first. ``forcing`` is the initial window,
        passed to ``initialize``. A value sent at a yield is the forcing for the
        next advance; sending ``None`` (or calling ``next``) at the 0th yield reuses
        ``forcing``, from which ``step`` selects the newest frames. Each yield is
        one ``step``: a model computing several lead times per core call yields them
        together. Sending ``None`` at a later yield is valid only for models without
        forcing.

        The front hook edits ``y`` before it is fed into the next step; the rear
        hook edits only the published output.
        """
        # Hooks see whatever payload type the model declares.
        front_hook: Callable[[Any], Any] = self.front_hook
        rear_hook: Callable[[Any], Any] = self.rear_hook
        forcing = args[0] if args else kwargs.get("forcing")
        forced = self.forcing_coords() is not None
        if forced and forcing is None:
            raise ValueError(f"{type(self).__name__} requires forcing")

        y, state = self.initialize(x, forcing)
        sent = yield y
        if sent is not None:
            forcing = sent
        while True:
            if forced and forcing is None:
                raise ValueError(
                    f"{type(self).__name__} requires forcing: send(...) it at each "
                    "advance after the first"
                )
            y, state = self.step(front_hook(y), state, forcing)
            forcing = yield rear_hook(y)
