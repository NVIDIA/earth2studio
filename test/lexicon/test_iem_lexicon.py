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

import pandas as pd
import pytest

from earth2studio.lexicon import IEM_ASOSLexicon

_PARSED_OBSERVATION = pd.DataFrame(
    {
        "tmpf": [83.0],
        "dwpf": [65.0],
        "relh": [54.71],
        "drct": [110.0],
        "sknt": [8.0],
        "p01i": [0.0],
        "mslp": [1013.1],
        "gust": [12.0],
        "skyc1": ["FEW"],
        "skyc2": ["SCT"],
        "skyc3": [None],
        "skyc4": [None],
    }
)


@pytest.mark.parametrize(
    "variable, expected",
    [
        ("t2m", 301.483333),
        ("d2m", 291.483333),
        ("r2m", 54.71),
        ("ws10m", 4.115552),
        ("u10m", -3.867354),
        ("v10m", 1.407602),
        ("fg10m", 6.173328),
        ("tp01", 0.0),
        ("msl", 101310.0),
        ("tcc", 0.5),
    ],
)
def test_iem_asos_lexicon(variable, expected):
    _, modifier = IEM_ASOSLexicon[variable]
    assert modifier(_PARSED_OBSERVATION).iat[0] == pytest.approx(expected, rel=1.0e-5)
