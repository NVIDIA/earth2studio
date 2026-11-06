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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.perturbation import Brown, Perturbation
from earth2studio.utils.cupy import from_torch


def _da(x, dims):
    """Wrap tensor as DataArray with integer coordinates"""
    return from_torch(
        x, OrderedDict((d, np.arange(n)) for d, n in zip(dims, x.shape)), attrs={"a": 1}
    )


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(2, 16, 16, 16),
            ("a", "variable", "lat", "lon"),
        ],
        [
            torch.randn(2, 32, 16),
            ("variable", "lat", "lon"),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude,reddening",
    [[1.0, 2], [0.05, 3], ["tensor", 2]],
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
def test_brown(x, coords, amplitude, reddening, device):

    x = x.to(device)
    xda = _da(x, coords)

    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )
    prtb = Brown(amplitude, reddening)
    assert isinstance(prtb, Perturbation)
    out = prtb(xda)
    xout = out.e2s.to_torch()[0]
    dx = xout - x

    # Don't have a good statistical test for this at the moment
    assert out.dims == xda.dims
    assert out.attrs == xda.attrs
    assert dx.shape == x.shape
    assert torch.allclose(
        torch.mean(dx), torch.Tensor([0]).to(device), rtol=1e-2, atol=1e-1
    )
    assert dx.device == x.device
    assert out.e2s.is_cupy == xda.e2s.is_cupy


@pytest.mark.parametrize(
    "x, coords, error",
    [
        [
            torch.randn(2, 8, 8, 8),
            ("a", "b", "not_lat", "lon"),
            KeyError,
        ],
        [
            torch.randn(2, 8, 8, 8),
            ("a", "lat", "b", "lon"),
            ValueError,
        ],
    ],
)
def test_brown_failure(x, coords, error):
    with pytest.raises(error):
        prtb = Brown()
        prtb(_da(x, coords))
