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

from datetime import datetime, timedelta

import pytest
import torch

from earth2studio.lexicon import HRRRFXLexicon, HRRRLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["u10m", "v200"],
        ["u80m", "z500", "q700"],
        ["u1hl", "v4hl", "t20hl", "p30hl"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_deterministic(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = HRRRLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device
        if label.split("::")[0] == "HGT":
            torch.allclose(output, 9.81 * input)


def test_hrrr_index_regex():
    time = datetime(2024, 1, 1)
    lead_time = timedelta(hours=1)

    # Test surface variable
    regex = HRRRLexicon.index_regex("t2m", time, lead_time)
    assert regex == "TMP:2 m above ground"

    # Test pressure level variable
    regex = HRRRLexicon.index_regex("u500", time, lead_time)
    assert regex == "UGRD:500 mb"

    # Test hybrid level variable
    regex = HRRRLexicon.index_regex("t20hl", time, lead_time)
    assert regex == "TMP:20 hybrid level"


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["u10m", "v500"],
        ["u80m", "tp", "tcwv"],
        ["u1hl", "v4hl", "t20hl", "p30hl"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_hrrrfx_run_deterministic(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = HRRRFXLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device
        if label.split("::")[3] == "HGT" and v.startswith("z"):
            torch.allclose(output, 9.81 * input)


def test_hrrrfx_index_regex():
    time = datetime(2024, 1, 1)
    # Test total precipitation with different lead times
    lead_time = timedelta(hours=1)
    regex = HRRRFXLexicon.index_regex("tp", time, lead_time)
    assert regex == "APCP:surface:0-1 hour acc"

    lead_time = timedelta(hours=3)
    regex = HRRRFXLexicon.index_regex("tp", time, lead_time)
    assert regex == "APCP:surface:2-3 hour acc"

    # Test regular variable
    regex = HRRRFXLexicon.index_regex("t2m", time, lead_time)
    assert regex == "TMP:2 m above ground"
