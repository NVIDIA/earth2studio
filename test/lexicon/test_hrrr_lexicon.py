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
def test_hrrr_lexicon(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = HRRRLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device
        if label.split("::")[1] == "HGT" and v.startswith("z"):
            torch.allclose(output, 9.81 * input)


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
        if label.split("::")[1] == "HGT" and v.startswith("z"):
            torch.allclose(output, 9.81 * input)


@pytest.mark.parametrize("lexicon", [HRRRLexicon, HRRRFXLexicon])
def test_hrrr_lexicon_snow_water_equivalent(lexicon):
    # WEASD is stored in kg m-2, sd is snow depth in m of water equivalent
    label, modifier = lexicon["sd"]
    assert label.split("::")[1] == "WEASD"
    assert modifier(np.array([25.0])) == pytest.approx(np.array([0.025]))
