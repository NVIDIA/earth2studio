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

from earth2studio.lexicon import CAMSGlobalLexicon


@pytest.mark.parametrize(
    "variable",
    [
        # Surface variables
        ["u10m", "v10m", "t2m"],
        ["d2m", "sp", "msl"],
        ["tcwv", "tp", "tcc"],
        # Aerosol optical depth
        ["aod550"],
        ["duaod550", "tcno2"],
        ["bcaod550", "ssaod550", "suaod550"],
        # Trace gases
        ["tcco", "tco3", "tcso2"],
        ["omaod550"],
        # Pressure level variables
        ["u500", "v500", "t500"],
        ["z850", "q700"],
    ],
)
def test_cams_lexicon(variable):
    data = np.random.randn(len(variable), 8)
    for v in variable:
        label, modifier = CAMSGlobalLexicon[v]
        output = modifier(data)
        assert isinstance(label, str)
        assert data.shape == output.shape


def test_cams_lexicon_invalid():
    with pytest.raises(KeyError):
        CAMSGlobalLexicon["nonexistent_variable"]


def test_cams_lexicon_vocab_format():
    for key, value in CAMSGlobalLexicon.VOCAB.items():
        parts = value.split("::")
        assert (
            len(parts) == 4
        ), f"VOCAB entry '{key}' must have format 'dataset::api_var::nc_key::level'"
        assert (
            parts[0] == "cams-global-atmospheric-composition-forecasts"
        ), f"Expected global dataset in VOCAB entry '{key}', got '{parts[0]}'"


def test_cams_lexicon_surface_vars():
    """Test that all expected surface variables are present."""
    expected_surface = [
        "u10m",
        "v10m",
        "t2m",
        "d2m",
        "sp",
        "msl",
        "tcwv",
        "tp",
        "skt",
        "z",
        "lsm",
        "tcc",
    ]
    for var in expected_surface:
        assert var in CAMSGlobalLexicon.VOCAB, f"Missing surface variable: {var}"


def test_cams_lexicon_aod_vars():
    """Test that all aerosol optical depth variables are present."""
    expected_aod = [
        "aod550",
        "duaod550",
        "omaod550",
        "bcaod550",
        "ssaod550",
        "suaod550",
    ]
    for var in expected_aod:
        assert var in CAMSGlobalLexicon.VOCAB, f"Missing AOD variable: {var}"


def test_cams_lexicon_trace_gas_vars():
    """Test that all trace gas variables are present."""
    expected_gases = ["tcco", "tcno2", "tco3", "tcso2"]
    for var in expected_gases:
        assert var in CAMSGlobalLexicon.VOCAB, f"Missing trace gas variable: {var}"


def test_cams_lexicon_pressure_level_vars():
    """Test that pressure level variables are present for common levels."""
    pressure_levels = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    variables = ["u", "v", "t", "z", "q"]

    for var in variables:
        for level in pressure_levels:
            key = f"{var}{level}"
            assert (
                key in CAMSGlobalLexicon.VOCAB
            ), f"Missing pressure level variable: {key}"
