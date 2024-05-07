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

import numpy as np
import pytest
import torch

from earth2studio.perturbation import Zero


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(4, 16, 16),
        torch.randn(2, 4, 16, 16),
        torch.randn(1, 2, 4, 16, 16),
    ],
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
def test_zero(x, device):

    x = x.to(device)
    coords = OrderedDict([(f"{i}", np.arange(x.shape[i])) for i in range(x.ndim)])

    prtb = Zero()
    xout, coords = prtb(x, coords)
    dx = xout - x

    assert dx.shape == x.shape
    assert torch.sum(dx) == 0
    assert dx.device == x.device
