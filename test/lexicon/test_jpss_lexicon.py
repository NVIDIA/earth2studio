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

import pytest
import torch

from earth2studio.lexicon import JPSSLexicon


@pytest.mark.parametrize(
    "variable",
    [["viirs01i"], ["viirs02m", "viirs03m"], ["lst", "aod", "cmask"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_jpss_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            product_type, folder, dataset, modifier = JPSSLexicon[v]
            output = modifier(input)
            assert isinstance(product_type, str)
            assert isinstance(folder, str)
            assert isinstance(dataset, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                product_type, folder, dataset, modifier = JPSSLexicon[v]
