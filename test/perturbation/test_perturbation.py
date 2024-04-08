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

from earth2studio.perturbation import Brown, Gaussian
from earth2studio.perturbation.utils import Perturbation


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(2, 4, 16, 16),
            OrderedDict(
                [
                    ("a", []),
                    ("variable", ["a", "b", "c", "d"]),
                    ("lat", []),
                    ("lon", []),
                ]
            ),
        ]
    ],
)
@pytest.mark.parametrize("method", [Gaussian(), Brown()])
@pytest.mark.parametrize(
    "variables,center,scale",
    [
        [None, None, None],
        [["a", "b"], None, None],
        [["b", "d"], np.random.randn(2), None],
        [["c", "a"], np.random.randn(2), np.random.randn(2)],
        [None, np.random.randn(4), np.random.randn(4)],
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
def test_perturbation(x, coords, method, variables, center, scale, device):
    x = x.to(device)
    x0 = torch.clone(x)  # Pertubation is inplace

    prtb = Perturbation(method, variables, center, scale)
    y, ycoord = prtb(x, coords)

    assert y.shape == x0.shape
    assert y.device == x0.device

    cdim = list(coords.keys()).index("variable")
    variables = coords["variable"] if variables is None else variables
    for i, variable in enumerate(coords["variable"]):
        if variable in variables:
            assert not torch.allclose(
                torch.select(x0, dim=cdim, index=i), torch.select(y, dim=cdim, index=i)
            )
        else:
            assert torch.allclose(
                torch.select(x0, dim=cdim, index=i), torch.select(y, dim=cdim, index=i)
            )


@pytest.mark.parametrize(
    "center,scale",
    [
        [np.random.randn(2), np.random.randn(3)],
        [np.random.randn(2), np.random.randn(2, 2)],
        [np.random.randn(4, 2), np.random.randn(4)],
    ],
)
def test_perturbation_failure_norm(center, scale):
    with pytest.raises(ValueError):
        Perturbation(Gaussian(), center=center, scale=scale)


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict(
            [
                ("a", []),
                ("lat", []),
                ("variable", []),
                ("lon", []),
            ]
        ),
        OrderedDict(
            [
                ("variable", []),
                ("a", []),
                ("lat", []),
                ("lon", []),
            ]
        ),
    ],
)
def test_perturbation_failure_coord(coords):
    with pytest.raises(ValueError):
        prtb = Perturbation(Gaussian())
        prtb(torch.zeros(1), coords)
