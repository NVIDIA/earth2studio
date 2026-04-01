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

from earth2studio.lexicon import CAMSLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["aod550"],
        ["duaod550", "tcno2"],
        ["bcaod550", "ssaod550", "suaod550"],
        ["tcco", "tco3", "tcso2"],
        ["gtco3", "omaod550"],
    ],
)
def test_cams_lexicon(variable):
    data = np.random.randn(len(variable), 8)
    for v in variable:
        label, modifier = CAMSLexicon[v]
        output = modifier(data)
        assert isinstance(label, str)
        assert data.shape == output.shape


def test_cams_lexicon_invalid():
    with pytest.raises(KeyError):
        CAMSLexicon["nonexistent_variable"]


def test_cams_lexicon_vocab_format():
    for key, value in CAMSLexicon.VOCAB.items():
        parts = value.split("::")
        assert len(parts) == 4, (
            f"VOCAB entry '{key}' must have format 'dataset::api_var::nc_key::level'"
        )
        assert parts[0] == "cams-global-atmospheric-composition-forecasts", (
            f"Expected global dataset in VOCAB entry '{key}', got '{parts[0]}'"
        )


def test_cams_lexicon_all_global_vars():
    expected = [
        "aod550",
        "duaod550",
        "omaod550",
        "bcaod550",
        "ssaod550",
        "suaod550",
        "tcco",
        "tcno2",
        "tco3",
        "tcso2",
        "gtco3",
    ]
    for var in expected:
        assert var in CAMSLexicon.VOCAB, f"Missing global variable: {var}"
    assert len(CAMSLexicon.VOCAB) == len(expected)
