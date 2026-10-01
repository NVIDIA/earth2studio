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

import numpy as np
import pytest
import torch

from earth2studio.models.px.stormscope import StormScopeBase


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fork_rng_persistent_state(device):
    from earth2studio.models.rng import fork_rng

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    device = torch.device(device)
    states = {"cpu": torch.Generator().manual_seed(1).get_state()}
    reference = torch.Generator(device=device).manual_seed(1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for fail in (False, True, False):
        try:
            with fork_rng(states, 1, device):
                actual = torch.randn(8, device=device)
                if fail:
                    raise RuntimeError("sampling failed")
        except RuntimeError:
            assert fail
        assert torch.equal(actual, torch.randn(8, device=device, generator=reference))
        assert torch.equal(cpu_state, torch.get_rng_state())
        if cuda_state is not None:
            assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
    with fork_rng(None, None, device):
        torch.rand(8)
    assert not torch.equal(cpu_state, torch.get_rng_state())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA missing"
            ),
        ),
    ],
)
def test_rng_stream_reset_and_isolation(device):
    model = StormScopeBase.__new__(StormScopeBase)
    torch.nn.Module.__init__(model)
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


def test_unseeded_model_uses_global_rng():
    model = StormScopeBase.__new__(StormScopeBase)
    torch.nn.Module.__init__(model)
    state = torch.get_rng_state()
    model._randn_like(torch.empty(8))
    assert not torch.equal(state, torch.get_rng_state())


def test_wrapper_dispatches_rng():
    from earth2studio.models.px import DiagnosticWrapper

    class Component(torch.nn.Module):
        stochastic = True
        _rng_generator = None

        def set_rng(self, seed, reset=True):
            if reset or self._rng_generator is None:
                self._rng_generator = torch.Generator().manual_seed(seed)

    wrapper = DiagnosticWrapper(Component(), Component())
    assert wrapper.stochastic
    wrapper.set_rng(12, reset=False)
    states = [
        m._rng_generator.get_state() for m in [wrapper.px_model, *wrapper.dx_model]
    ]
    wrapper.set_rng(99, reset=False)
    for model, state in zip([wrapper.px_model, *wrapper.dx_model], states):
        assert torch.equal(model._rng_generator.get_state(), state)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_sampler_state_and_exception_isolation(device):
    from earth2studio.models.px import StormCast

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    model = StormCast.__new__(StormCast)
    torch.nn.Module.__init__(model)
    x = torch.empty(8, device=device)
    model._sample = lambda x, conditioning: torch.randn_like(x)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if x.is_cuda else None
    model.set_rng(42, reset=False)
    first = model._forward(x, x)
    reference = torch.Generator(device=device).manual_seed(42)
    assert torch.equal(first, torch.randn(x.shape, device=device, generator=reference))
    other = StormCast.__new__(StormCast)
    torch.nn.Module.__init__(other)
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

    def fail(x, conditioning):
        torch.randn_like(x)
        raise RuntimeError("sampler failed")

    model._sample = fail
    with pytest.raises(RuntimeError, match="sampler failed"):
        model._forward(x, x)
    assert torch.equal(cpu_state, torch.get_rng_state())
    if x.is_cuda:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))
