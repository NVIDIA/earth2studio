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
    EarthMoverIFSInitialConditionLexicon,
    EarthMoverIFSLexicon,
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


def test_earthmover_ifs_initial_condition_lexicon_matches_marketplace_variables():
    assert EarthMoverIFSInitialConditionLexicon.VOCAB == IFS_INITIAL_CONDITION_VOCAB


def test_earthmover_ifs_forecast_lexicon_matches_marketplace_variables():
    assert EarthMoverIFSLexicon.VOCAB == IFS_FORECAST_VOCAB


def test_earthmover_ifs_initial_condition_lexicon_specs():
    assert EarthMoverIFSInitialConditionLexicon.spec("t2m").param_id == 167
    assert EarthMoverIFSInitialConditionLexicon.spec("t2m").short_name == "2t"
    assert EarthMoverIFSInitialConditionLexicon.spec("sst").param_id == 34
    assert EarthMoverIFSInitialConditionLexicon.spec("stl1").short_name == "stl1"
    assert EarthMoverIFSInitialConditionLexicon.spec("u10m").level_type == "surface"
    assert EarthMoverIFSInitialConditionLexicon.spec("q500").level_type == "isobaric"
    assert EarthMoverIFSInitialConditionLexicon.spec("q500").level == 500
    assert EarthMoverIFSInitialConditionLexicon.spec("z50").short_name == "z"
    assert (
        EarthMoverIFSInitialConditionLexicon.spec("u100m").param_id
        != EarthMoverIFSInitialConditionLexicon.spec("u10m").param_id
    )


def test_earthmover_ifs_forecast_lexicon_specs():
    assert EarthMoverIFSLexicon.spec("t2m").param_id == 167
    assert EarthMoverIFSLexicon.spec("t2m").short_name == "2t"
    assert EarthMoverIFSLexicon.spec("fdir").short_name == "fdir"
    assert EarthMoverIFSLexicon.spec("u10m").level_type == "surface"
    assert (
        EarthMoverIFSLexicon.spec("u100m").param_id
        != EarthMoverIFSLexicon.spec("u10m").param_id
    )


def test_earthmover_ifs_lexicon_keys():
    for variable in EarthMoverIFSLexicon.VOCAB:
        source_key, modifier = EarthMoverIFSLexicon[variable]
        assert source_key == EarthMoverIFSLexicon.VOCAB[variable]
        assert callable(modifier)
    for variable in EarthMoverIFSInitialConditionLexicon.VOCAB:
        source_key, modifier = EarthMoverIFSInitialConditionLexicon[variable]
        assert source_key == EarthMoverIFSInitialConditionLexicon.VOCAB[variable]
        assert callable(modifier)


def test_earthmover_ifs_lexicon_invalid():
    with pytest.raises(KeyError):
        EarthMoverIFSLexicon.spec("not_a_variable")
    with pytest.raises(KeyError):
        EarthMoverIFSLexicon["r500"]
    with pytest.raises(KeyError):
        EarthMoverIFSLexicon.spec("q500")
    with pytest.raises(KeyError):
        EarthMoverIFSInitialConditionLexicon.spec("fdir")


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
