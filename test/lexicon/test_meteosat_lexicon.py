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

from earth2studio.lexicon import MeteosatFCILexicon, MeteosatLILexicon


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


@pytest.mark.parametrize(
    "variable",
    [
        ["lightning_flash_radiance"],  # LFL measurement
        ["lightning_flash_count", "lightning_flash_duration"],  # synthetic + native
        ["lightning_group_radiance", "lightning_group_count"],  # LGR
        ["lightning_event_radiance", "lightning_event_count"],  # LEF
        ["lightning_flash_footprint_pixels"],  # native pixel count
        ["foo"],  # unknown variable -> KeyError
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_meteosat_li_lexicon(variable, device):
    input = torch.randn(len(variable), 100, 100).to(device)
    for v in variable:
        if v != "foo":
            label, modifier = MeteosatLILexicon[v]  # type: ignore[misc]
            output = modifier(input)
            assert isinstance(label, str)
            # Keys are "{product}::{field}" and name a real LI L2 collection
            product, field = label.split("::")
            assert product in ("LFL", "LGR", "LEF")
            assert field
            assert input.shape == output.shape
            assert input.device == output.device
        else:
            with pytest.raises(KeyError):
                MeteosatLILexicon[v]  # type: ignore[misc]


def test_meteosat_li_lexicon_shares_glm_naming():
    """LI and GLM agree on the unified lightning variable naming scheme."""
    from earth2studio.lexicon import GOESGLMLexicon

    for name in MeteosatLILexicon.VOCAB:
        assert name.startswith("lightning_")
    # Every tier of the detection hierarchy is expressible on both
    # instruments under one id
    for level in ("event", "group", "flash"):
        assert f"lightning_{level}_count" in MeteosatLILexicon.VOCAB
        assert f"lightning_{level}_count" in GOESGLMLexicon.VOCAB
