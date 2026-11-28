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

from earth2studio.lexicon import (
    EarthmoverERA5Lexicon,
    EarthmoverIFSInitialConditionLexicon,
    EarthmoverIFSLexicon,
)
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
IFS_ANALYSIS_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
IFS_ANALYSIS_SURFACE_VOCAB = {
    "u100m": "100u::sfc::",
    "v100m": "100v::sfc::",
    "u10m": "10u::sfc::",
    "v10m": "10v::sfc::",
    "d2m": "2d::sfc::",
    "t2m": "2t::sfc::",
    "hcc": "hcc::sfc::",
    "lcc": "lcc::sfc::",
    "mcc": "mcc::sfc::",
    "msl": "msl::sfc::",
    "skt": "skt::sfc::",
    "sp": "sp::sfc::",
    "sst": "sst::sfc::",
    "stl1": "stl1::sfc::",
    "stl2": "stl2::sfc::",
    "swvl1": "swvl1::sfc::",
    "swvl2": "swvl2::sfc::",
    "tcc": "tcc::sfc::",
    "tcw": "tcw::sfc::",
    "tcwv": "tcwv::sfc::",
}
IFS_ANALYSIS_PRESSURE_VOCAB = {
    f"{name}{level}": f"{name}::pl::{level}"
    for name in ("q", "t", "u", "v", "w", "z")
    for level in IFS_ANALYSIS_LEVELS
}
IFS_INITIAL_CONDITION_VOCAB = {
    **IFS_ANALYSIS_SURFACE_VOCAB,
    **IFS_ANALYSIS_PRESSURE_VOCAB,
}

ERA5_LEVELS = IFS_ANALYSIS_LEVELS
ERA5_SURFACE_VOCAB = {
    "blh": "blh::sfc::",
    "cape": "cape::sfc::",
    "cp": "cp::sfc::",
    "d2m": "d2m::sfc::",
    "fdir": "fdir::sfc::",
    "fg10m": "fg10::sfc::",
    "fsr": "fsr::sfc::",
    "hcc": "hcc::sfc::",
    "ie": "ie::sfc::",
    "lcc": "lcc::sfc::",
    "lsp": "lsp::sfc::",
    "mcc": "mcc::sfc::",
    "msl": "msl::sfc::",
    "sd": "sd::sfc::",
    "sf": "sf::sfc::",
    "skt": "skt::sfc::",
    "slhf": "slhf::sfc::",
    "sp": "sp::sfc::",
    "ssr": "ssr::sfc::",
    "ssrd": "ssrd::sfc::",
    "sst": "sst::sfc::",
    "stl1": "stl1::sfc::",
    "stl2": "stl2::sfc::",
    "stl3": "stl3::sfc::",
    "stl4": "stl4::sfc::",
    "swvl1": "swvl1::sfc::",
    "swvl2": "swvl2::sfc::",
    "t2m": "t2m::sfc::",
    "tcc": "tcc::sfc::",
    "tcw": "tcw::sfc::",
    "tcwv": "tcwv::sfc::",
    "tisr": "tisr::sfc::",
    "tp": "tp::sfc::",
    "tsr": "tsr::sfc::",
    "u10m": "u10::sfc::",
    "u100m": "u100::sfc::",
    "v10m": "v10::sfc::",
    "v100m": "v100::sfc::",
    "zust": "zust::sfc::",
}
ERA5_PRESSURE_VOCAB = {
    f"{name}{level}": f"{name}::pl::{level}"
    for name in ("pv", "q", "r", "t", "u", "v", "w", "z")
    for level in ERA5_LEVELS
}
ERA5_VOCAB = {
    **ERA5_SURFACE_VOCAB,
    **ERA5_PRESSURE_VOCAB,
}


def test_earthmover_era5_lexicon_matches_marketplace_variables():
    assert EarthmoverERA5Lexicon.VOCAB == ERA5_VOCAB
    assert EarthmoverERA5Lexicon.PRESSURE_LEVELS == ERA5_LEVELS


def test_earthmover_ifs_initial_condition_lexicon_matches_marketplace_variables():
    assert EarthmoverIFSInitialConditionLexicon.VOCAB == IFS_INITIAL_CONDITION_VOCAB


def test_earthmover_ifs_forecast_lexicon_matches_marketplace_variables():
    assert EarthmoverIFSLexicon.VOCAB == IFS_FORECAST_VOCAB


def test_earthmover_era5_lexicon_specs():
    assert EarthmoverERA5Lexicon.spec("t2m").short_name == "t2m"
    assert EarthmoverERA5Lexicon.spec("t2m").param_id == 167
    assert EarthmoverERA5Lexicon.spec("t2m").canonical_units == "K"
    assert "2t" in EarthmoverERA5Lexicon.spec("t2m").aliases
    assert EarthmoverERA5Lexicon.spec("msl").param_id == 151
    assert EarthmoverERA5Lexicon.spec("fg10m").short_name == "fg10"
    assert EarthmoverERA5Lexicon.spec("fg10m").param_id == 49
    assert EarthmoverERA5Lexicon.spec("u10m").level_type == "surface"
    assert EarthmoverERA5Lexicon.spec("u10m").param_id == 165
    assert EarthmoverERA5Lexicon.spec("sf").canonical_units == "kg m-2"
    assert EarthmoverERA5Lexicon.spec("q500").level_type == "isobaric"
    assert EarthmoverERA5Lexicon.spec("q500").level == 500
    assert EarthmoverERA5Lexicon.spec("pv500").short_name == "pv"
    assert EarthmoverERA5Lexicon.spec("pv500").param_id == 60
    assert EarthmoverERA5Lexicon.spec("z50").short_name == "z"


def test_earthmover_ifs_initial_condition_lexicon_specs():
    assert EarthmoverIFSInitialConditionLexicon.spec("t2m").param_id == 167
    assert EarthmoverIFSInitialConditionLexicon.spec("t2m").short_name == "2t"
    assert EarthmoverIFSInitialConditionLexicon.spec("sst").param_id == 34
    assert EarthmoverIFSInitialConditionLexicon.spec("stl1").short_name == "stl1"
    assert EarthmoverIFSInitialConditionLexicon.spec("u10m").level_type == "surface"
    assert EarthmoverIFSInitialConditionLexicon.spec("q500").level_type == "isobaric"
    assert EarthmoverIFSInitialConditionLexicon.spec("q500").level == 500
    assert EarthmoverIFSInitialConditionLexicon.spec("z50").short_name == "z"
    assert (
        EarthmoverIFSInitialConditionLexicon.spec("u100m").param_id
        != EarthmoverIFSInitialConditionLexicon.spec("u10m").param_id
    )


def test_earthmover_ifs_forecast_lexicon_specs():
    assert EarthmoverIFSLexicon.spec("t2m").param_id == 167
    assert EarthmoverIFSLexicon.spec("t2m").short_name == "2t"
    assert EarthmoverIFSLexicon.spec("fdir").short_name == "fdir"
    assert EarthmoverIFSLexicon.spec("u10m").level_type == "surface"
    assert (
        EarthmoverIFSLexicon.spec("u100m").param_id
        != EarthmoverIFSLexicon.spec("u10m").param_id
    )


def test_earthmover_lexicon_keys():
    for lexicon in (
        EarthmoverERA5Lexicon,
        EarthmoverIFSInitialConditionLexicon,
        EarthmoverIFSLexicon,
    ):
        for variable in lexicon.VOCAB:
            source_key, modifier = lexicon[variable]
            assert source_key == lexicon.VOCAB[variable]
            assert callable(modifier)


def test_earthmover_lexicon_invalid():
    with pytest.raises(KeyError):
        EarthmoverERA5Lexicon.spec("not_a_variable")
    with pytest.raises(KeyError):
        EarthmoverIFSLexicon["r500"]
    with pytest.raises(KeyError):
        EarthmoverIFSLexicon.spec("q500")
    with pytest.raises(KeyError):
        EarthmoverIFSInitialConditionLexicon.spec("fdir")


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
    spec = EarthmoverIFSLexicon.spec(variable)
    np.testing.assert_allclose(make_modifier(spec, src_units)(raw), expected)


@pytest.mark.parametrize(
    "variable,src_units,raw,expected",
    [
        ("t2m", "degree_Celsius", np.array([0.0]), np.array([273.15])),
        ("sf", "m of water equivalent", np.array([0.002]), np.array([2.0])),
    ],
)
def test_earthmover_era5_unit_conversions(variable, src_units, raw, expected):
    spec = EarthmoverERA5Lexicon.spec(variable)
    np.testing.assert_allclose(make_modifier(spec, src_units)(raw), expected)
