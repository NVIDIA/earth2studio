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

"""Native diagnostic cases to adapt in test/models/dx/test_<model>.py.

Use the existing fixture's mock weights; retain numerical, coordinate, invalid
input and conformance assertions. Preserve marked package tests for explicit
real-weight validation. Configure domains and variables before signature lookup.
For generative models, retain sample-count and seed reproducibility assertions in
the existing cases, and make the mock exercise the actual sampler RNG path.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.utils.coords import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


def make_input(model, device: str = "cpu") -> xr.DataArray:
    """Build a concrete field retaining configured geometry and metadata."""
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    return from_torch(torch.randn(signature.shape, device=device), signature)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_model_call(model, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model.to(device)
    x = make_input(model, device)
    before = x.copy(deep=True)
    result = model(x)
    handshake_dataarray(result, model.output_coords(x))
    xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())
    # Preserve the model-specific expected numerical output assertion here.


def test_model_exceptions(model):
    x = make_input(model)
    with pytest.raises(ValueError):
        model(x.transpose(*reversed(x.dims)))


def test_model_conformance(model):
    skipped = check_diagnostic_contract(model)
    assert skipped == (
        []
        if getattr(model, "stochastic", False)
        else ["D10: model does not declare itself stochastic"]
    )


def test_model_deterministic_seed(model):
    x = make_input(model)
    model.set_rng(42)
    first = model(x).e2s.as_numpy()
    model.set_rng(42)
    xr.testing.assert_identical(first, model(x).e2s.as_numpy())
    model.set_rng(43)
    assert not np.array_equal(first.values, model(x).e2s.as_numpy().values)
