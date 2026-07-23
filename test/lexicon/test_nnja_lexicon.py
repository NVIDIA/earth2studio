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

from earth2studio.lexicon import NNJAObsConvLexicon, NNJAObsSatLexicon


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

    # u, v, gps: identity (already SI)
    for var in ("u", "v", "gps"):
        _, mod = NNJAObsConvLexicon[var]
        df = mod(pd.DataFrame({"observation": [3.14]}))
        assert df["observation"].iloc[0] == pytest.approx(3.14)


def test_nnja_obs_conv_lexicon_routes():
    """Conv lexicon entries are route-prefixed with known NNJA routes."""
    for var, vocab in NNJAObsConvLexicon.VOCAB.items():
        route, _, rest = vocab.partition("::")
        assert route in {"prepbufr", "gpsro"}, f"{var}: unexpected route '{route}'"
        assert rest, f"{var}: empty payload after route prefix"

    assert NNJAObsConvLexicon.VOCAB["gps"] == "gpsro::15037"
    assert "gps_t" not in NNJAObsConvLexicon.VOCAB
    assert "gps_q" not in NNJAObsConvLexicon.VOCAB


def test_nnja_obs_sat_lexicon_routes_and_quantity_identity():
    assert NNJAObsSatLexicon.VOCAB == {
        "atms": "atms::TMBR",
        "atms_antenna_temperature": "atms::TMANT",
        "mhs": "mhs::TMBR",
        "amsua": "amsua::TMBR",
        "amsub": "amsub::TMBR",
    }

    frame = pd.DataFrame({"observation": [201.25]})
    for variable in NNJAObsSatLexicon.VOCAB:
        source_key, modifier = NNJAObsSatLexicon[variable]
        assert source_key == NNJAObsSatLexicon.VOCAB[variable]
        assert modifier(frame) is frame

    with pytest.raises(KeyError):
        NNJAObsSatLexicon["crisfsr"]
