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

from earth2studio.lexicon import AIFSLexicon, IFSLexicon
from earth2studio.lexicon.ecmwf import (
    AIFS_ACCUMULATION_HOURS,
    IFS_ACCUMULATION_HOURS,
)


@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v200"], ["msl", "z500", "q700"], ["sf06", "ro06"]]
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ifs_lexicon(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = IFSLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device


@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v200"], ["hcc", "z500", "q700"], ["sf06", "ro06"]]
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifs_lexicon(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = AIFSLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device


def test_ecmwf_accumulation_aliases():
    expected = {
        "cp06": 6,
        "ro06": 6,
        "sf06": 6,
        "ssrd06": 6,
        "strd06": 6,
        "tp06": 6,
    }

    assert IFS_ACCUMULATION_HOURS == expected
    assert AIFS_ACCUMULATION_HOURS == expected
    assert IFSLexicon.VOCAB["ro06"] == "ro::sfc::"
    assert IFSLexicon.VOCAB["sf06"] == "sf::sfc::"
    assert AIFSLexicon.VOCAB["ro06"] == "rowe::sfc::"
    assert AIFSLexicon.VOCAB["sf06"] == "sf::sfc::"
