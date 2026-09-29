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

from earth2studio.models.rng import RNGMixin, seeded


class SamplingModel(RNGMixin):
    @seeded
    def sample(self, device: str = "cpu") -> torch.Tensor:
        return torch.randn(8, device=device) + np.random.random()


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
    model = SamplingModel()
    model.set_rng(42, reset=False)
    first = model.sample(device)
    model.set_rng(999, reset=False)
    second = model.sample(device)
    assert not torch.equal(first, second)
    model.set_rng(42)
    cpu_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert torch.equal(first, model.sample(device))
    assert torch.equal(second, model.sample(device))
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    for before, after in zip(
        cuda_states, torch.cuda.get_rng_state_all() if cuda_states else []
    ):
        assert torch.equal(before, after)
    model.set_rng(43)
    assert not torch.equal(first, model.sample(device))


def test_unseeded_model_uses_global_rng():
    model = SamplingModel()
    state = torch.get_rng_state()
    model.sample()
    assert not torch.equal(state, torch.get_rng_state())


def test_wrapper_dispatches_rng():
    from earth2studio.models.px import DiagnosticWrapper

    class Component(torch.nn.Module, RNGMixin):
        pass

    wrapper = DiagnosticWrapper(Component(), Component())
    assert wrapper.stochastic
    wrapper.set_rng(12, reset=False)
    states = [
        m._rng_generator.get_state() for m in [wrapper.px_model, *wrapper.dx_model]
    ]
    wrapper.set_rng(99, reset=False)
    for model, state in zip([wrapper.px_model, *wrapper.dx_model], states):
        assert torch.equal(model._rng_generator.get_state(), state)
