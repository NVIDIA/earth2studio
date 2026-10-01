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

import numpy as np
import pytest
import torch

from earth2studio.models.px import DiagnosticWrapper, StormCast
from earth2studio.models.px.stormscope import StormScopeBase
from earth2studio.models.utils import fork_rng


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    return torch.device(request.param)


def make_model(model_type):
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    return model


@pytest.fixture
def stormscope():
    return make_model(StormScopeBase)


@pytest.fixture
def stormcast():
    model = make_model(StormCast)
    model._sample = lambda x, conditioning: torch.randn_like(x)
    return model


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


def test_rng_stream_reset_and_isolation(device, stormscope):
    model = stormscope
    template = torch.empty(8, device=device)
    model.set_rng(42, reset=False)
    first = model._randn_like(template)
    model.set_rng(999, reset=False)
    second = model._randn_like(template)
    assert not torch.equal(first, second)
    model.set_rng(42)
    cpu_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert torch.equal(first, model._randn_like(template))
    assert torch.equal(second, model._randn_like(template))
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    for before, after in zip(
        cuda_states, torch.cuda.get_rng_state_all() if cuda_states else []
    ):
        assert torch.equal(before, after)
    model.set_rng(43)
    assert not torch.equal(first, model._randn_like(template))


def test_unseeded_model_uses_global_rng(stormscope):
    state = torch.get_rng_state()
    stormscope._randn_like(torch.empty(8))
    assert not torch.equal(state, torch.get_rng_state())


def test_wrapper_dispatches_rng():
    class Component(torch.nn.Module):
        stochastic = True
        _rng_generator = None

        def set_rng(self, seed, reset=True):
            if reset or self._rng_generator is None:
                self._rng_generator = torch.Generator().manual_seed(seed)

    components = [Component(), Component()]
    wrapper = DiagnosticWrapper(*components)
    assert wrapper.stochastic
    wrapper.set_rng(12, reset=False)
    states = [model._rng_generator.get_state() for model in components]
    wrapper.set_rng(99, reset=False)
    for model, state in zip(components, states):
        assert torch.equal(model._rng_generator.get_state(), state)


def test_stormcast_sampler_stream(device, stormcast):
    model = stormcast
    x = torch.empty(8, device=device)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if x.is_cuda else None
    model.set_rng(42, reset=False)
    first = model._forward(x, x)
    reference = torch.Generator(device=device).manual_seed(42)
    assert torch.equal(first, torch.randn(x.shape, device=device, generator=reference))
    other = make_model(StormCast)
    other._sample = model._sample
    other.set_rng(43)
    other_first = other._forward(x, x)
    model.set_rng(99, reset=False)
    second = model._forward(x, x)
    assert torch.equal(second, torch.randn(x.shape, device=device, generator=reference))
    assert not torch.equal(second, other_first)
    assert not torch.equal(first, second)
    model.set_rng(42)
    assert torch.equal(first, model._forward(x, x))
    assert torch.equal(second, model._forward(x, x))
    assert torch.equal(cpu_state, torch.get_rng_state())
    if x.is_cuda:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))


def test_stormcast_sampler_exception_isolation(device, stormcast):
    x = torch.empty(8, device=device)
    stormcast.set_rng(42)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if x.is_cuda else None

    def fail(x, conditioning):
        torch.randn_like(x)
        raise RuntimeError("sampler failed")

    stormcast._sample = fail
    with pytest.raises(RuntimeError, match="sampler failed"):
        stormcast._forward(x, x)
    assert torch.equal(cpu_state, torch.get_rng_state())
    if x.is_cuda:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
