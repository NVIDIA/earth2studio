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

import pytest
import torch

from earth2studio.lexicon import WB2Lexicon


@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v200"], ["msl", "t150", "q700"]]
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_deterministic(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = WB2Lexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device


@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v200"], ["msl", "t150", "q700"]]
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_failure(variable, device):
    with pytest.raises(KeyError):
        label, modifier = WB2Lexicon["t3m"]
