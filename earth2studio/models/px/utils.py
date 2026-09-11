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

from collections.abc import Callable, Iterator, Sequence

import torch

from earth2studio.utils.coords import CoordSystem

Hook = Callable[[torch.Tensor, CoordSystem], tuple[torch.Tensor, CoordSystem]]


# sphinx - hook chain start
class HookChain:
    """An ordered, composable sequence of prognostic model hooks.

    A chain is itself a hook: calling it applies every member hook in registration
    order, threading the tensor and coordinate system through each one. Assigning a
    bare callable to ``front_hook``/``rear_hook`` remains supported and is promoted
    to a one-element chain the first time a second hook is registered.

    Parameters
    ----------
    hooks : Sequence[Hook], optional
        Initial hooks in application order, by default empty
    """

    def __init__(self, hooks: Sequence[Hook] = ()) -> None:
        self.hooks: list[Hook] = list(hooks)

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply every hook in the chain in registration order.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Coordinate system describing ``x``

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Tensor and coordinate system after the full chain
        """
        for hook in self.hooks:
            x, coords = hook(x, coords)
        return x, coords

    def append(self, hook: Hook) -> None:
        """Add a hook to the end of the chain.

        Parameters
        ----------
        hook : Hook
            Hook to append
        """
        self.hooks.append(hook)

    def remove(self, hook: Hook) -> None:
        """Remove the first occurrence of a hook from the chain.

        Parameters
        ----------
        hook : Hook
            Hook to remove

        Raises
        ------
        ValueError
            If the hook is not registered on this chain
        """
        self.hooks.remove(hook)

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self) -> Iterator[Hook]:
        return iter(self.hooks)


# sphinx - hook chain end


class PrognosticMixin:
    """This utility adds the ability to call hooks into a prognostic iterator.

    Hooks belong to the iterator, which is the only path that owns a rollout loop.
    ``__call__`` is the single-step primitive and does not apply them: a caller
    holding a single step can transform the tensor itself, whereas nothing outside
    ``create_iterator`` can reach the state fed back between steps.
    """

    #: Whether the model draws randomness during a rollout. Stochastic models must
    #: implement ``set_rng(seed, reset=True)``; see ``dev/spec/MODEL_CONTRACT_SPEC.md``.
    stochastic: bool = False

    def _default_hook(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        return x, coords

    front_hook = _default_hook
    rear_hook = _default_hook

    def _hook_chain(self, name: str) -> HookChain:
        """Return the named hook slot as a chain, promoting it if needed.

        The default hook becomes an empty chain; a previously assigned bare callable
        becomes the chain's first element, so direct assignment and registration can
        be mixed without losing a hook.

        Parameters
        ----------
        name : str
            Hook slot name, either ``"front_hook"`` or ``"rear_hook"``

        Returns
        -------
        HookChain
            The chain installed on the slot
        """
        current = getattr(self, name)
        if isinstance(current, HookChain):
            return current
        is_default = getattr(current, "__func__", None) is PrognosticMixin._default_hook
        chain = HookChain(() if is_default else (current,))
        setattr(self, name, chain)
        return chain

    def add_front_hook(self, hook: Hook) -> Hook:
        """Register a hook applied before each model step.

        Parameters
        ----------
        hook : Hook
            Hook to register

        Returns
        -------
        Hook
            The registered hook, so this may be used as a decorator
        """
        self._hook_chain("front_hook").append(hook)
        return hook

    def add_rear_hook(self, hook: Hook) -> Hook:
        """Register a hook applied after each model step.

        Parameters
        ----------
        hook : Hook
            Hook to register

        Returns
        -------
        Hook
            The registered hook, so this may be used as a decorator
        """
        self._hook_chain("rear_hook").append(hook)
        return hook

    def clear_hooks(self) -> None:
        """Remove every registered hook, restoring the default pass-through."""
        for name in ("front_hook", "rear_hook"):
            if name in vars(self):
                delattr(self, name)
