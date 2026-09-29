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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import torch


class RNGMixin:
    """Isolated RNG control for stochastic backends using global random draws."""

    stochastic = True
    _rng_generator: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Set the model's random stream without changing global RNG state.

        Parameters
        ----------
        seed : int
            Seed for reproducible sampling.
        reset : bool, optional
            Replace an existing stream. If False, initialize only when unseeded,
            by default True.
        """
        if reset or self._rng_generator is None:
            self._rng_generator = torch.Generator().manual_seed(seed)

    @contextmanager
    def _rng_context(self) -> Iterator[None]:
        """Isolate one numerical call; never hold this context across a yield."""
        if self._rng_generator is None:
            yield
            return
        seed = int(torch.randint(2**32, (), generator=self._rng_generator))
        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_initialized()
            else []
        )
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            np.random.seed(seed)
            try:
                yield
            finally:
                np.random.set_state(numpy_state)


F = TypeVar("F", bound=Callable[..., Any])


def seeded(function: F) -> F:
    """Run a numerical model method in its isolated, advancing random stream."""

    @wraps(function)
    def wrapped(self: RNGMixin, *args: Any, **kwargs: Any) -> Any:
        with self._rng_context():
            return function(self, *args, **kwargs)

    return cast(F, wrapped)
