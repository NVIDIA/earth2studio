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

"""Native prognostic test patterns to adapt inside test/models/px/test_<model>.py.

Supply the existing model fixture with mock weights. Retain call, iterator,
exceptions, conformance and package cases; avoid parallel migration suites or
redundant matrices. Package tests stay marked package and use real cached weights
only when explicitly enabled. Never replace core numerical assertions with shape
checks alone. Stochastic mocks must exercise the wrapper's real seeding mechanism.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.utils.coords import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


def make_input(model, device: str = "cpu") -> xr.DataArray:
    """Concretize the configured signature without materializing its placeholder."""
    signature = model.input_coords()
    replacements = {"batch": [0]}
    if "time" in signature.attrs.get("earth2studio_dynamic_dims", ()):
        replacements["time"] = np.array(["2024-01-01"], dtype="datetime64[ns]")
    signature = coord_array_like(signature, replacements)
    return from_torch(torch.randn(signature.shape, device=device), signature)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_model_call(model, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model.to(device)
    x = make_input(model, device)
    before = x.copy(deep=True)
    output = model(x)
    handshake_dataarray(output, model.output_coords(x))
    xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())
    # Add model-specific normalization, ordering and numerical assertions here.


def test_model_iter(model):
    x = make_input(model)
    iterator = model.create_iterator(x)
    first = next(iterator)
    saved = first.copy(deep=True)
    xr.testing.assert_identical(first, x.isel(lead_time=slice(-1, None)))
    prediction = next(iterator)
    handshake_dataarray(prediction, model.output_coords(x))
    next(iterator)
    xr.testing.assert_identical(first, saved)
    iterator.close()


def test_model_exceptions(model):
    x = make_input(model)
    with pytest.raises(ValueError):
        model(x.transpose(*reversed(x.dims)))


def test_model_conformance(model):
    # The returned list contains structurally unevaluated rules. Pin any expected
    # entries specifically; actual violations raise and must not be suppressed.
    skipped = check_prognostic_contract(model)
    assert skipped == (
        [] if model.stochastic else ["P14: model does not declare itself stochastic"]
    )
