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

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def fork_rng(
    states: dict[str, torch.Tensor] | None,
    seed: int | None,
    device: torch.device,
) -> Iterator[None]:
    """Resume model-owned Torch RNG state for a numerical sampling call.

    Parameters
    ----------
    states : dict[str, torch.Tensor] | None
        Model-owned states keyed by ``cpu`` and ``cuda:N``. Updated in place
        with the advanced states, including when sampling raises an exception.
    seed : int | None
        Initial seed for a previously unused device. None leaves global RNG
        behavior unchanged. Seeded calls require an initialized CPU state.
    device : torch.device
        Sampling device. CPU state is preserved alongside CUDA state.

    Notes
    -----
    Restores the caller's global state on exit. Do not span iterator yields or
    use concurrently with other code accessing the same global generators.
    """
    if seed is None:
        yield
        return
    if states is None:
        raise ValueError("Seeded sampling requires initialized RNG states")
    devices = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(states["cpu"])
        for index in devices:
            key = f"cuda:{index}"
            if key not in states:
                states[key] = torch.Generator(device=key).manual_seed(seed).get_state()
            torch.cuda.set_rng_state(states[key], index)
        try:
            yield
        finally:
            states["cpu"] = torch.get_rng_state()
            for index in devices:
                states[f"cuda:{index}"] = torch.cuda.get_rng_state(index)
