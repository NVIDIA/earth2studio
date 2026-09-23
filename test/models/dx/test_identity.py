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

from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import Identity
from earth2studio.utils.cupy import from_torch


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict({"a": np.random.rand(8), "b": np.random.rand(8)}),
        OrderedDict(
            {"a": np.random.rand(16), "b": np.random.rand(16), "c": np.random.rand(16)}
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_diagnostic_identity(coords, device):
    # Initialize random tensor for data
    dims = [value.shape[0] for value in coords.values()]
    data = torch.rand(*dims).to(device)

    df = Identity().to(device)

    field = from_torch(data, coords, name="weather", attrs={"experiment": "identity"})
    field = field.assign_coords(height=("a", np.arange(field.sizes["a"])))
    field.encoding["source"] = "identity-fixture"
    output = df(field)

    xr.testing.assert_identical(output, field)
    assert output.encoding == field.encoding
    assert output.e2s.is_cupy == field.e2s.is_cupy
    signature = df.output_coords(field)
    assert signature.data.nbytes == 0
    assert signature.dims == field.dims
    assert df.input_coords().data.nbytes == 0


def test_identity_conformance():
    model = Identity()
    assert check_diagnostic_contract(model) == [
        "D4: model declares fewer than three dimensions",
        "D10: model does not declare itself stochastic",
    ]
