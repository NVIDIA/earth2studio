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

from earth2studio.lexicon import DynamicalLexicon
from earth2studio.lexicon.base import E2STUDIO_VOCAB


@pytest.mark.parametrize(
    ("variable", "label"),
    [
        ("t2m", "temperature_2m"),
        ("tpf", "precipitation_surface"),
        ("ptype", "categorical_precipitation_type_surface"),
        ("z500", "geopotential_height_500hpa"),
    ],
)
def test_dynamical_lexicon(variable, label):
    input = np.array([1.0], dtype=np.float32)

    output_label, modifier = DynamicalLexicon[variable]
    output = modifier(input)

    assert output_label == label
    assert output.shape == input.shape


def test_dynamical_ptype_is_standard_vocab():
    assert "ptype" in E2STUDIO_VOCAB
