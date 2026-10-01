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

from contextlib import nullcontext

import pytest
import torch

from earth2studio.models.utils import fork_rng


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    return torch.device(request.param)


def test_fork_rng_persistent_state(device):
    states = {"cpu": torch.Generator().manual_seed(1).get_state()}
    reference = torch.Generator(device=device).manual_seed(1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for fail in (False, True, False):
        expectation = (
            pytest.raises(RuntimeError, match="sampling failed")
            if fail
            else nullcontext()
        )
        with expectation:
            with fork_rng(states, 1, device):
                actual = torch.randn(8, device=device)
                if fail:
                    raise RuntimeError("sampling failed")
        assert torch.equal(actual, torch.randn(8, device=device, generator=reference))
        assert torch.equal(cpu_state, torch.get_rng_state())
        if cuda_state is not None:
            assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))


def test_fork_rng_unseeded_uses_global_rng(device):
    reference = torch.Generator(device=device)
    state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else torch.get_rng_state()
    )
    reference.set_state(state)
    with fork_rng(None, None, device):
        first = torch.randn(8, device=device)
    second = torch.randn(8, device=device)
    assert torch.equal(first, torch.randn(8, device=device, generator=reference))
    assert torch.equal(second, torch.randn(8, device=device, generator=reference))
