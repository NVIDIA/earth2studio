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
        ["dust"],
        ["so2sfc", "pm2p5"],
        ["no2sfc", "o3sfc", "cosfc"],
        ["aod550"],
        ["duaod550", "tcno2"],
        ["dust_500m", "pm2p5_1000m"],
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
            f"VOCAB entry '{key}' must have format "
            "'dataset::api_var::nc_key::level'"
        )
        dataset = parts[0]
        assert dataset in (
            "cams-europe-air-quality-forecasts",
            "cams-global-atmospheric-composition-forecasts",
        ), f"Unknown dataset in VOCAB entry '{key}': {dataset}"


def test_cams_lexicon_all_levels_covered():
    levels = [50, 100, 250, 500, 750, 1000, 2000, 3000, 5000]
    pollutants = ["dust", "pm2p5", "pm10", "so2", "no2", "o3", "co", "nh3", "no"]
    for p in pollutants:
        for lev in levels:
            key = f"{p}_{lev}m"
            assert key in CAMSLexicon.VOCAB, f"Missing level entry: {key}"
