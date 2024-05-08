# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from collections import OrderedDict

import pytest
import torch

from earth2studio.perturbation import BredVector, Brown, Gaussian


@pytest.fixture
def model():
    class FooModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.Tensor([0.1]))
            self.index = 0

        def forward(self, x, coords):
            self.index += 1
            return self.scale * x, coords

    return FooModel()


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(2, 16, 16, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
        [
            torch.randn(2, 8, 32, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude,steps,ensemble",
    [[1.0, 5, False], [1.0, 3, True]],
)
@pytest.mark.parametrize(
    "seeding_perturbation_method",
    [Brown(), Gaussian()],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_bred_vec(
    model, x, coords, amplitude, steps, ensemble, seeding_perturbation_method, device
):

    model = model.to(device)
    model.index = 0
    x = x.to(device)

    prtb = BredVector(model, amplitude, steps, ensemble, seeding_perturbation_method)
    xout, coords = prtb(x, coords)
    dx = xout - x

    # Don't have a good statistical test for this at the moment
    assert dx.shape == x.shape
    assert dx.device == x.device
    assert model.index == steps + 1
