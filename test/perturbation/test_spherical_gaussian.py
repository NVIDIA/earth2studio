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

from earth2studio.perturbation import CorrelatedSphericalGaussian, SphericalGaussian
from earth2studio.utils.cupy import from_torch


def _da(x, dims):
    return from_torch(
        x, OrderedDict((d, np.arange(n)) for d, n in zip(dims, x.shape)), attrs={"a": 1}
    )


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(1, 2, 16, 32),
            ("a", "variable", "lat", "lon"),
        ],
        [
            torch.randn(2, 17, 32),
            ("variable", "lat", "lon"),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude,alpha,tau,sigma",
    [[1.0, 2.0, 3.0, None], [0.05, 1.0, 10.0, 2.0], ["tensor", 1.0, 10.0, 2.0]],
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
    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )
    xda = _da(x, coords)
    prtb = SphericalGaussian(amplitude, alpha, tau, sigma)
    xout = prtb(xda)
    dx = xout.e2s.as_numpy().values - xda.e2s.as_numpy().values

    assert xout.dims == xda.dims
    assert xout.shape == xda.shape
    assert xout.attrs == xda.attrs
    assert np.isclose(dx.mean(), 0.0, rtol=1e-2, atol=1e-1)
    assert xout.e2s.is_cupy == (device != "cpu")


@pytest.mark.parametrize(
    "x, coords, error",
    [
        [torch.randn(2, 4), ("not_lat", "lon"), KeyError],
        [torch.randn(2, 4), ("lat", "not_lon"), KeyError],
        [torch.randn(4, 2), ("lon", "lat"), ValueError],
        [torch.randn(4, 4), ("lat", "lon"), ValueError],
    ],
)
def test_spherical_gaussian_failure(x, coords, error):
    with pytest.raises(error):
        prtb = SphericalGaussian()
        prtb(_da(x, coords))


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(1, 2, 16, 32),
            ("a", "variable", "lat", "lon"),
        ],
        [
            torch.randn(1, 16, 32),
            ("variable", "lat", "lon"),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude,sigma,length_scale,time_scale",
    [[1.0, 2.0, 3.0, 20.0], [0.05, 1.0, 10.0, 48.0], ["tensor", 1.0, 10.0, 28.0]],
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
def test_correlated_spherical_gaussian(
    x, coords, amplitude, sigma, length_scale, time_scale, device
):

    x = x.to(device)
    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )
    xda = _da(x, coords)
    prtb = CorrelatedSphericalGaussian(amplitude, sigma, length_scale, time_scale)
    xout = prtb(xda)

    assert xout.dims == xda.dims
    assert xout.shape == xda.shape
    assert xout.attrs == xda.attrs
    assert xout.e2s.is_cupy == (device != "cpu")


@pytest.mark.parametrize(
    "x, coords, error",
    [
        [torch.randn(2, 4), ("not_lat", "lon"), KeyError],
        [torch.randn(2, 4), ("lat", "not_lon"), KeyError],
        [torch.randn(4, 2), ("lon", "lat"), ValueError],
        [torch.randn(4, 4), ("lat", "lon"), ValueError],
    ],
)
def test_correlated_spherical_gaussian_failure(x, coords, error):
    with pytest.raises(error):
        prtb = CorrelatedSphericalGaussian(48.0)
        prtb(_da(x, coords))
