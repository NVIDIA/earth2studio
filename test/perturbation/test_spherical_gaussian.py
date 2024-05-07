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

from earth2studio.perturbation import SphericalGaussian


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(1, 2, 16, 32),
            OrderedDict([("a", []), ("b", []), ("lat", []), ("lon", [])]),
        ],
        [torch.randn(2, 17, 32), OrderedDict([("a", []), ("lat", []), ("lon", [])])],
    ],
)
@pytest.mark.parametrize(
    "amplitude,alpha,tau,sigma",
    [[1.0, 2.0, 3.0, None], [0.05, 1.0, 10.0, 2.0]],
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
def test_spherical_gaussian(x, coords, amplitude, alpha, tau, sigma, device):

    x = x.to(device)

    prtb = SphericalGaussian(amplitude, alpha, tau, sigma)
    xout, coords = prtb(x, coords)
    dx = xout - x

    assert dx.shape == x.shape
    assert torch.allclose(
        torch.mean(dx), torch.Tensor([0]).to(device), rtol=1e-2, atol=1e-1
    )
    assert dx.device == x.device


@pytest.mark.parametrize(
    "x, coords, error",
    [
        [torch.randn(2, 4), OrderedDict([("not_lat", []), ("lon", [])]), KeyError],
        [torch.randn(2, 4), OrderedDict([("lat", []), ("not_lon", [])]), KeyError],
        [torch.randn(4, 2), OrderedDict([("lon", []), ("lat", [])]), ValueError],
        [torch.randn(4, 4), OrderedDict([("lat", []), ("lon", [])]), ValueError],
    ],
)
def test_spherical_gaussian_failure(x, coords, error):
    with pytest.raises(error):
        prtb = SphericalGaussian()
        prtb(x, coords)
