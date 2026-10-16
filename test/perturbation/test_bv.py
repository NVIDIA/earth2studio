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
import xarray as xr

from earth2studio.perturbation import (
    BredVector,
    Brown,
    Gaussian,
)
from earth2studio.utils.cupy import from_torch


def _da(x, dims):
    return from_torch(
        x, OrderedDict((d, np.arange(n)) for d, n in zip(dims, x.shape)), attrs={"a": 1}
    )


@pytest.fixture
def model():
    class FooModel:
        def __init__(self):
            self.scale = 0.1
            self.index = 0

        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            self.index += 1
            return x * self.scale

    return FooModel()


@pytest.mark.parametrize(
    "x, coords",
    [
        [torch.randn(2, 16, 16, 16), ("a", "variable", "lat", "lon")],
        [torch.randn(2, 8, 32, 16), ("a", "variable", "lat", "lon")],
    ],
)
@pytest.mark.parametrize(
    "amplitude,steps,ensemble",
    [[1.0, 5, False], [1.0, 3, True], ["tensor", 2, True]],
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

    model.index = 0
    x = x.to(device)
    xda = _da(x, coords)

    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )

    prtb = BredVector(model, amplitude, steps, ensemble, seeding_perturbation_method)
    xout = prtb(xda)

    # Don't have a good statistical test for this at the moment
    assert xout.dims == xda.dims
    assert xout.shape == xda.shape
    assert xout.attrs == xda.attrs
    assert xout.e2s.is_cupy == (device != "cpu")
    assert not np.allclose(xout.e2s.as_numpy().values, xda.e2s.as_numpy().values)
    assert model.index == steps + 1
