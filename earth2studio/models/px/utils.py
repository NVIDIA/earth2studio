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

from collections.abc import Callable

import xarray as xr

Hook = Callable[[xr.DataArray], xr.DataArray]


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
