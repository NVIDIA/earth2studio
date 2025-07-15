# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.lexicon import GOESLexicon


@pytest.mark.parametrize(
    "variable",
    [["vis047"], ["vis064", "nir086"], ["ir1035", "ir1120", "ir1230"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_goes_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = GOESLexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = GOESLexicon[v]
