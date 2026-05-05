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

import numpy as np
import pytest

from earth2studio.lexicon import GHCNLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m_max"],
        ["t2m_min", "tp"],
        [
            "t2m_max",
            "t2m_min",
            "t2m",
            "d2m",
            "r2m",
            "tp",
            "sf",
            "sd",
            "sde",
            "ws10m",
            "fg10m",
            "tcc",
        ],
    ],
)
def test_ghcn_lexicon(variable):
    input = np.random.randn(len(variable), 8).astype(np.float32)
    for v in variable:
        label, modifier = GHCNLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
