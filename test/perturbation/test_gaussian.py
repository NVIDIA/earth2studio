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

import pytest
import torch

from earth2studio.perturbation import CorrelatedSphericalGaussian, Gaussian


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(4, 16, 16, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
        [
            torch.randn(4, 32, 16),
            OrderedDict([("variable", []), ("lat", []), ("lon", [])]),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude",
    [1.0, 0.05, "tensor"],
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
def test_gaussian(x, coords, amplitude, device):

    x = x.to(device)

    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )
    prtb = Gaussian(amplitude)
    xout, coords = prtb(x, coords)
    dx = xout - x

    assert dx.shape == x.shape
    assert torch.allclose(
        torch.mean(dx), torch.Tensor([0]).to(device), rtol=1e-2, atol=1e-1
    )

    if not isinstance(amplitude, torch.Tensor):
        assert torch.allclose(
            torch.std(dx), torch.Tensor([amplitude]).to(device), rtol=1e-2, atol=1e-1
        )
    assert dx.device == x.device


def test_correlated_spherical_gaussian_no_amplitude():
    """Test that CorrelatedSphericalGaussian raises error without amplitude"""
    with pytest.raises(ValueError):
        CorrelatedSphericalGaussian(noise_amplitude=None)


def test_correlated_spherical_gaussian_wrong_ratio():
    """Test that incorrect lat/lon ratio raises error"""
    x = torch.randn(4, 16, 15, 16)  # Wrong ratio
    coords = OrderedDict([("variable", []), ("time", []), ("lat", []), ("lon", [])])

    prtb = CorrelatedSphericalGaussian(noise_amplitude=0.05)
    with pytest.raises(ValueError):
        prtb(x, coords)


@pytest.mark.parametrize("latents", [[2], [2, 4], [2, 4, 2]])
@pytest.mark.parametrize("nlat", [32, 33])  # Test both even and odd latitudes
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
def test_correlated_spherical_gaussian_odd_even_lats(latents: list[int], nlat, device):
    """Test CorrelatedSphericalGaussian handles both odd and even latitude counts"""
    x = torch.randn(*latents, nlat, 2 * 32).to(device)  # Keep lon count even
    coords = OrderedDict([("batch", []), ("variable", []), ("lat", []), ("lon", [])])

    prtb = CorrelatedSphericalGaussian(
        noise_amplitude=0.05, sigma=1.0, length_scale=5.0e5, time_scale=48.0
    )

    output, _ = prtb(x, coords)
    assert output.shape == x.shape
    assert output.device == x.device
    # Need a good statistical test for this
