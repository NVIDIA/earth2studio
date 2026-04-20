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

from earth2studio.lexicon import NClimGridLexicon


@pytest.mark.parametrize(
    "variable",
    [["t2m_max"], ["t2m_min", "tp"], ["t2m_max", "t2m_min", "tp", "spi"]],
)
def test_nclimgrid_lexicon(variable):
    input = np.random.randn(len(variable), 8).astype(np.float32)
    for v in variable:
        label, modifier = NClimGridLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape


def test_nclimgrid_lexicon_modifiers():
    # Temperature: Celsius -> Kelvin
    _, mod = NClimGridLexicon["t2m_max"]
    np.testing.assert_allclose(mod(np.array([0.0, 25.0])), [273.15, 298.15])

    _, mod = NClimGridLexicon["t2m_min"]
    np.testing.assert_allclose(mod(np.array([-10.0])), [263.15])

    # Precipitation: mm -> m
    _, mod = NClimGridLexicon["tp"]
    np.testing.assert_allclose(mod(np.array([0.0, 100.0])), [0.0, 0.1])

    # SPI: identity (dimensionless)
    _, mod = NClimGridLexicon["spi"]
    np.testing.assert_allclose(mod(np.array([1.2, -0.5])), [1.2, -0.5])


def test_nclimgrid_lexicon_invalid_key():
    with pytest.raises(KeyError):
        NClimGridLexicon["invalid_key"]
