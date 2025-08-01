# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the CMIP6 lexicon."""

import pytest
import torch

from earth2studio.lexicon import CMIP6Lexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["u10m", "v200"],
        ["tcwv", "z500", "q850"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cmip6_lexicon(variable, device):
    """Basic shape / device checks and simple key assertions."""
    inp = torch.randn(len(variable), 8).to(device)

    for v in variable:
        (var_id, level), modifier = CMIP6Lexicon[v]
        out = modifier(inp)

        # shape / device preserved
        assert out.shape == inp.shape
        assert out.device == inp.device

        if level != -1:
            # the variable name should start with the corresponding parameter code
            assert v[0] in ["u", "v", "t", "z", "r", "q", "w"]
            # pressure level matches encoded hPa
            assert isinstance(level, int)
            assert f"{level}" in v
