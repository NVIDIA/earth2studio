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

from earth2studio.lexicon import MeteosatFCILexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["fci04vis"],  # single VIS band
        ["fci63wv", "fci87ir"],  # WV + IR pair
        ["fci06vis", "fci22nir", "fci105ir"],  # HRFI-capable channels
        ["fci04vis", "fci16nir", "fci38ir", "fci133ir"],  # mixed
        ["foo"],  # unknown variable → KeyError
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_meteosat_fci_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MeteosatFCILexicon[v]  # type: ignore[misc]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                MeteosatFCILexicon[v]  # type: ignore[misc]
