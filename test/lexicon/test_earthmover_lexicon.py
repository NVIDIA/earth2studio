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

from earth2studio.lexicon import EarthMoverIFSLexicon, IFSLexicon
from earth2studio.lexicon.earthmover import make_modifier, normalize_units


def test_earthmover_ifs_lexicon_matches_ifs_vocab():
    assert EarthMoverIFSLexicon.VOCAB == IFSLexicon.VOCAB


def test_earthmover_ifs_lexicon_specs():
    assert EarthMoverIFSLexicon.spec("t850").param_id == 130
    assert EarthMoverIFSLexicon.spec("t850").level == 850
    assert EarthMoverIFSLexicon.spec("z500").short_name == "gh"
    assert EarthMoverIFSLexicon.spec("u10").level_type == "isobaric"
    assert EarthMoverIFSLexicon.spec("u10m").level_type == "surface"
    assert (
        EarthMoverIFSLexicon.spec("u10").param_id
        != EarthMoverIFSLexicon.spec("u10m").param_id
    )


def test_earthmover_ifs_lexicon_keys():
    for variable in EarthMoverIFSLexicon.VOCAB:
        source_key, modifier = EarthMoverIFSLexicon[variable]
        assert source_key == EarthMoverIFSLexicon.VOCAB[variable]
        assert callable(modifier)


def test_earthmover_ifs_lexicon_invalid():
    with pytest.raises(KeyError):
        EarthMoverIFSLexicon.spec("not_a_variable")


def test_earthmover_unit_normalization():
    assert normalize_units("m s**-1") == normalize_units("m s-1")
    assert normalize_units("degree_Celsius") == "degc"
    assert normalize_units("(0 - 1)") == "1"
    assert normalize_units("percent") == "percent"


@pytest.mark.parametrize(
    "variable,src_units,raw,expected",
    [
        ("t2m", "degree_Celsius", np.array([0.0]), np.array([273.15])),
        ("z500", "m", np.array([1.0]), np.array([9.80665])),
        ("z500", "m2 s-2", np.array([5.0]), np.array([5.0])),
        ("tcc", "percent", np.array([50.0]), np.array([0.5])),
    ],
)
def test_earthmover_unit_conversions(variable, src_units, raw, expected):
    spec = EarthMoverIFSLexicon.spec(variable)
    np.testing.assert_allclose(make_modifier(spec, src_units)(raw), expected)
