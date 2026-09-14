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

from earth2studio.lexicon import CAMulatorLexicon
from earth2studio.lexicon.camulator import (
    ACCUMULATED_FLUX_VARIABLES,
    CAMULATOR_LEVELS,
    CAMULATOR_STEP_SECONDS,
)
from earth2studio.models.px.camulator import (
    FORCING_VARIABLES,
    OUTPUT_VARIABLES,
    PROGNOSTIC_VARIABLES,
)


@pytest.mark.parametrize(
    "variable",
    [
        # 3D prognostic, model levels
        ["u0k", "v15k", "t31k", "qtot7k"],
        # 2D prognostic
        ["sp", "t2m"],
        # Diagnostics
        ["tp06", "skt", "hcc", "lcc", "mcc", "iews", "inss", "ws10m", "e06"],
        ["msdwswrf", "msdwlwrf", "msuwswrf", "msuwlwrf", "msshf", "mslhf"],
        ["mtuwswrf", "mtuwlwrf"],
        # Forcing
        ["mtdwswrf", "sst", "sic", "global_mean_co2"],
    ],
)
def test_camulator_lexicon(variable):
    data = np.random.randn(len(variable), 8)
    for v in variable:
        label, modifier = CAMulatorLexicon[v]
        output = modifier(data)
        assert isinstance(label, str)
        assert data.shape == output.shape


def test_camulator_lexicon_invalid():
    with pytest.raises(KeyError):
        CAMulatorLexicon["nonexistent_variable"]
    with pytest.raises(KeyError):
        CAMulatorLexicon["u32k"]  # only 32 levels, 0..31
    with pytest.raises(KeyError):
        CAMulatorLexicon.get_e2s_from_cesm("NOTAVAR")


def test_camulator_lexicon_vocab_format():
    assert len(CAMulatorLexicon.VOCAB) == 4 * CAMULATOR_LEVELS + 2 + 17 + 4
    for key, value in CAMulatorLexicon.VOCAB.items():
        parts = value.split("::")
        assert len(parts) in (1, 2), f"VOCAB entry '{key}' must be 'NAME' or 'NAME::k'"
        if len(parts) == 2:
            assert key.endswith("k")
            assert 0 <= int(parts[1]) < CAMULATOR_LEVELS
    # CESM names are unique, so the reverse mapping is a bijection
    assert len(set(CAMulatorLexicon.VOCAB.values())) == len(CAMulatorLexicon.VOCAB)
    for e2s, cesm in CAMulatorLexicon.VOCAB.items():
        assert CAMulatorLexicon.get_e2s_from_cesm(cesm) == e2s


def test_camulator_lexicon_model_variables():
    # Every variable the prognostic model exposes must resolve in the lexicon
    for v in list(OUTPUT_VARIABLES) + list(FORCING_VARIABLES):
        assert v in CAMulatorLexicon
    assert len(OUTPUT_VARIABLES) == 147
    assert len(PROGNOSTIC_VARIABLES) == 130
    for var, cesm in [("u", "U"), ("v", "V"), ("t", "T"), ("qtot", "Qtot")]:
        for k in range(CAMULATOR_LEVELS):
            assert CAMulatorLexicon.VOCAB[f"{var}{k}k"] == f"{cesm}::{k}"


def test_camulator_lexicon_modifiers():
    data = np.array([1.0, 21600.0], dtype=np.float32)

    # CO2: mol mol-1 -> ppm
    label, modifier = CAMulatorLexicon["global_mean_co2"]
    assert label == "co2vmr_3d"
    np.testing.assert_allclose(modifier(np.float32(3.7e-4)), 370.0, rtol=1e-6)

    # 6 h accumulated fluxes (J m-2) -> mean rate (W m-2)
    for v in ACCUMULATED_FLUX_VARIABLES:
        _, modifier = CAMulatorLexicon[v]
        np.testing.assert_allclose(modifier(data), data / CAMULATOR_STEP_SECONDS)
    assert CAMULATOR_STEP_SECONDS == 6 * 3600

    # Everything else is an identity
    for v in ["sp", "t2m", "tp06", "e06", "mtuwswrf", "mtuwlwrf", "sst", "sic"]:
        _, modifier = CAMulatorLexicon[v]
        np.testing.assert_array_equal(modifier(data), data)
