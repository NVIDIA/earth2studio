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

from earth2studio.lexicon import NNJAObsConvLexicon


def test_nnja_obs_conv_lexicon_parse():
    """Every NNJAObsConvLexicon entry resolves to a non-empty key plus a callable modifier."""
    for var in NNJAObsConvLexicon.VOCAB:
        key, modifier = NNJAObsConvLexicon[var]
        assert isinstance(key, str)
        assert key
        assert callable(modifier)


def test_nnja_obs_conv_lexicon_modifiers():
    """The conv lexicon modifiers convert raw PrepBUFR units to Earth2Studio standards."""
    # t: degrees C -> Kelvin
    _, mod = NNJAObsConvLexicon["t"]
    df = mod(pd.DataFrame({"observation": [0.0]}))
    assert df["observation"].iloc[0] == pytest.approx(273.15)

    # q: mg/kg -> kg/kg
    _, mod = NNJAObsConvLexicon["q"]
    df = mod(pd.DataFrame({"observation": [1e6]}))
    assert df["observation"].iloc[0] == pytest.approx(1.0)

    # pres: hPa -> Pa
    _, mod = NNJAObsConvLexicon["pres"]
    df = mod(pd.DataFrame({"observation": [1.0]}))
    assert df["observation"].iloc[0] == pytest.approx(100.0)

    # u, v: identity (already SI)
    for var in ("u", "v"):
        _, mod = NNJAObsConvLexicon[var]
        df = mod(pd.DataFrame({"observation": [3.14]}))
        assert df["observation"].iloc[0] == pytest.approx(3.14)


def test_nnja_obs_conv_lexicon_routes():
    """Conv lexicon entries are route-prefixed with 'prepbufr::'."""
    for var, vocab in NNJAObsConvLexicon.VOCAB.items():
        route, _, rest = vocab.partition("::")
        assert route == "prepbufr", f"{var}: unexpected route '{route}'"
        assert rest, f"{var}: empty payload after route prefix"
