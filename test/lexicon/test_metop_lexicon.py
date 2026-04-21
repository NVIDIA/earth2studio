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

from earth2studio.lexicon import (
    MetOpAMSUALexicon,
    MetOpAVHRRLexicon,
    MetOpIASILexicon,
    MetOpMHSLexicon,
    MetOpMTGLexicon,
)


@pytest.mark.parametrize(
    "variable",
    [["amsua"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_metop_amsua_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MetOpAMSUALexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = MetOpAMSUALexicon[v]


@pytest.mark.parametrize(
    "variable",
    [["avhrr"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_metop_avhrr_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MetOpAVHRRLexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = MetOpAVHRRLexicon[v]


@pytest.mark.parametrize(
    "variable",
    [["iasi"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_metop_iasi_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MetOpIASILexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = MetOpIASILexicon[v]


@pytest.mark.parametrize(
    "variable",
    [["mhs"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_metop_mhs_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MetOpMHSLexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = MetOpMHSLexicon[v]


@pytest.mark.parametrize(
    "variable",
    [["fci01"], ["fci07", "fci09"], ["fci01", "fci06", "fci12"], ["foo"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_metop_mtg_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MetOpMTGLexicon[v]
            output = modifier(input)
            assert isinstance(label, str)
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                label, modifier = MetOpMTGLexicon[v]
