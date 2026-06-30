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

from earth2studio.lexicon import CosmoLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["u10m", "v10m", "sp"],
        ["tcc", "d2m", "tot_precip", "h_pbl", "tke_l40", "u3d_l45"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cosmo_lexicon_get_item(variable, device):
    """Earth2Studio names resolve to COSMO name(s) + a shape/device-preserving
    unit modifier."""
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        cosmo_name, modifier = CosmoLexicon[v]
        assert v in CosmoLexicon  # __contains__ over the vocab
        output = modifier(input)
        assert input.shape == output.shape
        assert input.device == output.device


def test_cosmo_lexicon_to_e2studio():
    """COSMO output names map back to Earth2Studio names + a unit scale, accepting
    either resolution's spelling; unknown names fall back to lowercase."""
    # both resolution spellings of the same field -> one Earth2Studio name
    assert CosmoLexicon.to_e2studio("U_10M") == ("u10m", 1.0)  # REA6 spelling
    assert CosmoLexicon.to_e2studio("10U") == ("u10m", 1.0)  # REA2 spelling
    assert CosmoLexicon.to_e2studio("2MT") == ("t2m", 1.0)
    # cloud cover carries a % -> 0-1 fraction rescale
    assert CosmoLexicon.to_e2studio("CLCT") == ("tcc", 0.01)
    # COSMO-specific fields keep a descriptive lowercase name, scale 1.0
    assert CosmoLexicon.to_e2studio("H_PBL") == ("h_pbl", 1.0)
    assert CosmoLexicon.to_e2studio("U3D_L45") == ("u3d_l45", 1.0)
    # unknown name -> lowercased fallback (no raise)
    assert CosmoLexicon.to_e2studio("FUTURE_VAR") == ("future_var", 1.0)


def test_cosmo_lexicon_tcc_modifier_converts_percent_to_fraction():
    """The tcc modifier converts COSMO percent to the Earth2Studio 0-1 fraction."""
    _, modifier = CosmoLexicon["tcc"]
    out = modifier(torch.tensor([0.0, 50.0, 100.0]))
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))
