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

import warnings
from collections.abc import Callable, Generator, Iterator
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


def initial_condition(
    x: xr.DataArray | tuple[xr.DataArray, ...],
) -> xr.DataArray | tuple[xr.DataArray, ...]:
    """Initial condition of a rollout: each input slot at its final lead time.

    ``rollout_iterator`` yields forecasts only; drivers that publish the starting
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
    nothing outside ``rollout_iterator`` can reach the outputs fed back between
    steps.

    ``front_hook``/``rear_hook`` are single callable slots, not a registration
    list: a caller with more than one transformation to apply composes them into
    one function and assigns that, so the order they run in is visible at the
    assignment site rather than spread across every place that registered one.

    Wrappers that still define their own ``__call__``/``create_iterator`` override
    the derived versions below, and inherit ``initialize``/``step`` stubs until they
    are migrated. ``rollout_iterator`` wraps such a ``create_iterator``, dropping its
    initial-condition yield.
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

    # The derived ``__call__``/``create_iterator`` take ``forcing`` through
    # ``*args``/``**kwargs`` only so that unmigrated wrappers defining
    # ``__call__(x)``/``create_iterator(x)`` still type check as overrides. The
    # protocol declares the real signature ``(x, forcing=None)``; restore it here
    # once no wrapper overrides these.

    def __call__(
        self, x: xr.DataArray | tuple[xr.DataArray, ...], *args: Any, **kwargs: Any
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Advance one step from ``x`` without hooks, via ``initialize``."""
        forcing = args[0] if args else kwargs.get("forcing")
        if forcing is None and self.forcing_coords() is not None:
            raise ValueError(f"{type(self).__name__} requires forcing")
        return self.initialize(x, forcing)[0]

    def rollout_iterator(
        self,
        x: xr.DataArray | tuple[xr.DataArray, ...],
        forcing: xr.DataArray | tuple[xr.DataArray, ...] | None = None,
    ) -> Generator[
        xr.DataArray | tuple[xr.DataArray, ...],
        xr.DataArray | tuple[xr.DataArray, ...] | None,
        None,
    ]:
        """Roll out from input ``x`` and forcing ``forcing`` via ``initialize``/``step``.
        Subseqeuent rollout steps receive forcing at later lead times by ``send``.

        Yields forecasts only, starting with the output of ``initialize``; take the
        initial condition from ``initial_condition(x)``. ``forcing`` is the initial
        window, consumed by ``initialize``. A value sent at a yield is the forcing
        for the next ``step``; sending ``None`` (or calling ``next``) is valid only
        for models without forcing. Each yield is one core computation: a model
        computing several lead times per core call yields them together.

        The front hook edits ``y`` before it is fed into the next step, so it never
        sees the initial condition; edit ``x`` before calling instead. The rear hook
        edits only the published output, including the first forecast.
        """
        if (
            type(self).initialize is PrognosticMixin.initialize
            and type(self).create_iterator is not PrognosticMixin.create_iterator
        ):
            # Unmigrated wrapper: drop the initial condition its own iterator yields.
            legacy = self.create_iterator(x)
            next(legacy)
            yield from legacy
            return

        # Hooks see whatever payload type the model declares.
        front_hook: Callable[[Any], Any] = self.front_hook
        rear_hook: Callable[[Any], Any] = self.rear_hook
        forced = self.forcing_coords() is not None
        if forced and forcing is None:
            raise ValueError(f"{type(self).__name__} requires forcing")

        y, state = self.initialize(x, forcing)
        while True:
            forcing = yield rear_hook(y)
            if forced and forcing is None:
                raise ValueError(
                    f"{type(self).__name__} requires forcing: send(...) it at each "
                    "yield"
                )
            y, state = self.step(front_hook(y), state, forcing)

    # Annotated as an Iterator for the same reason as ``__call__``; the generator
    # still accepts forcing through ``send``.
    def create_iterator(
        self, x: xr.DataArray | tuple[xr.DataArray, ...], *args: Any, **kwargs: Any
    ) -> Iterator[xr.DataArray | tuple[xr.DataArray, ...]]:
        """Deprecated: yield ``initial_condition(x)``, then ``rollout_iterator``.

        Kept so existing loops counting ``nsteps + 1`` yields keep their lead
        times. The front hook no longer runs on the initial condition, and a value
        sent at the 0th yield is ignored: the first forecast uses ``forcing``.
        """
        warnings.warn(
            "create_iterator is deprecated; use rollout_iterator, which yields "
            "forecasts only, and initial_condition(x) for the starting fields",
            DeprecationWarning,
            stacklevel=2,
        )
        forcing = args[0] if args else kwargs.get("forcing")
        yield initial_condition(x)
        yield from self.rollout_iterator(x, forcing)
