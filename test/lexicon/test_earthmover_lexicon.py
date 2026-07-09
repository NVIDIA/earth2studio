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

from earth2studio.lexicon import EarthMoverERA5Lexicon, EarthMoverIFSLexicon
from earth2studio.lexicon.base import E2STUDIO_VOCAB
from earth2studio.lexicon.earthmover import make_modifier, normalize_units

IFS_FORECAST_VOCAB = {
    "u100m": "100u::sfc::",
    "v100m": "100v::sfc::",
    "u10m": "10u::sfc::",
    "v10m": "10v::sfc::",
    "d2m": "2d::sfc::",
    "t2m": "2t::sfc::",
    "cp": "cp::sfc::",
    "fdir": "fdir::sfc::",
    "hcc": "hcc::sfc::",
    "lcc": "lcc::sfc::",
    "mcc": "mcc::sfc::",
    "msl": "msl::sfc::",
    "sd": "sd::sfc::",
    "ssrd": "ssrd::sfc::",
    "tp": "tp::sfc::",
}
ERA5_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ERA5_SINGLE_VARIABLES = {
    "blh",
    "cape",
    "cp",
    "d2m",
    "fdir",
    "fg10m",
    "fsr",
    "hcc",
    "ie",
    "lcc",
    "lsp",
    "mcc",
    "msl",
    "sd",
    "sf",
    "skt",
    "slhf",
    "sp",
    "ssr",
    "ssrd",
    "sst",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "swvl1",
    "t2m",
    "tcc",
    "tcw",
    "tcwv",
    "tisr",
    "tp",
    "tsr",
    "u10m",
    "u100m",
    "v10m",
    "v100m",
    "zust",
}


def test_earthmover_ifs_lexicon_matches_marketplace_variables():
    assert EarthMoverIFSLexicon.VOCAB == IFS_FORECAST_VOCAB


def test_earthmover_ifs_lexicon_specs():
    assert EarthMoverIFSLexicon.spec("t2m").param_id == 167
    assert EarthMoverIFSLexicon.spec("t2m").short_name == "2t"
    assert EarthMoverIFSLexicon.spec("fdir").short_name == "fdir"
    assert EarthMoverIFSLexicon.spec("u10m").level_type == "surface"
    assert all(
        EarthMoverIFSLexicon.spec(variable).level_type == "surface"
        for variable in EarthMoverIFSLexicon.VOCAB
    )
    assert (
        EarthMoverIFSLexicon.spec("u100m").param_id
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
    with pytest.raises(KeyError):
        EarthMoverIFSLexicon["z500"]


def test_earthmover_era5_lexicon_matches_marketplace_variables():
    assert set(EarthMoverERA5Lexicon.VOCAB).issuperset(ERA5_SINGLE_VARIABLES)
    for name in ("pv", "q", "r", "t", "u", "v", "w", "z"):
        for level in ERA5_LEVELS:
            variable = f"{name}{level}"
            assert EarthMoverERA5Lexicon.VOCAB[variable] == f"{name}::pl::{level}"


def test_earthmover_era5_lexicon_specs():
    assert EarthMoverERA5Lexicon.spec("t2m").short_name == "t2m"
    assert EarthMoverERA5Lexicon.spec("t2m").level_type == "surface"
    assert EarthMoverERA5Lexicon.spec("z500").short_name == "z"
    assert EarthMoverERA5Lexicon.spec("z500").level_type == "isobaric"
    assert EarthMoverERA5Lexicon.spec("z500").level == 500
    assert EarthMoverERA5Lexicon.spec("pv500").short_name == "pv"


def test_earthmover_era5_lexicon_keys():
    for variable in EarthMoverERA5Lexicon.VOCAB:
        source_key, modifier = EarthMoverERA5Lexicon[variable]
        assert source_key == EarthMoverERA5Lexicon.VOCAB[variable]
        assert callable(modifier)


def test_earthmover_era5_lexicon_base_vocab_coverage():
    assert set(EarthMoverERA5Lexicon.VOCAB) <= set(E2STUDIO_VOCAB)


def test_earthmover_unit_normalization():
    assert normalize_units("m s**-1") == normalize_units("m s-1")
    assert normalize_units("degree_Celsius") == "degc"
    assert normalize_units("(0 - 1)") == "1"
    assert normalize_units("percent") == "percent"


@pytest.mark.parametrize(
    "variable,src_units,raw,expected",
    [
        ("t2m", "degree_Celsius", np.array([0.0]), np.array([273.15])),
        ("cp", "kg m-2", np.array([1000.0]), np.array([1.0])),
        ("hcc", "percent", np.array([50.0]), np.array([0.5])),
        ("fdir", "J m**-2", np.array([5.0]), np.array([5.0])),
    ],
)
def test_earthmover_unit_conversions(variable, src_units, raw, expected):
    spec = EarthMoverIFSLexicon.spec(variable)
    np.testing.assert_allclose(make_modifier(spec, src_units)(raw), expected)
