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

from earth2studio.lexicon import GHCNDailyLexicon, GHCNHourlyLexicon


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m_max"],
        ["t2m_min", "tp"],
        [
            "t2m_max",
            "t2m_min",
            "t2m",
            "d2m",
            "r2m",
            "tp",
            "sf",
            "sd",
            "sde",
            "ws10m",
            "fg10m",
            "tcc",
        ],
    ],
)
def test_ghcn_lexicon(variable):
    input = np.random.randn(len(variable), 8).astype(np.float32)
    for v in variable:
        label, modifier = GHCNDailyLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["d2m", "tp"],
        ["t2m", "d2m", "ws10m", "fg10m", "tp", "u10m", "v10m", "tcc"],
    ],
)
def test_ghcn_hourly_lexicon(variable):
    input = np.random.randn(len(variable), 8).astype(np.float32)
    for v in variable:
        col, modifier = GHCNHourlyLexicon[v]
        output = modifier(input)
        assert col is None or isinstance(col, str)
        assert input.shape == output.shape


class TestGHCNDailyLexicon:
    @pytest.mark.parametrize(
        "var, element",
        [
            ("t2m_max", "TMAX"),
            ("t2m_min", "TMIN"),
            ("t2m", "TAVG"),
            ("d2m", "ADPT"),
            ("r2m", "RHAV"),
            ("tp", "PRCP"),
            ("sf", "WESF"),
            ("sd", "SNWD"),
            ("sde", "SNOW"),
            ("ws10m", "AWND"),
            ("fg10m", "WSF2"),
            ("tcc", "ACMH"),
        ],
    )
    def test_element_map(self, var, element):
        assert GHCNDailyLexicon.VOCAB[var] == element

    def test_lexicon_keys(self):
        for var in GHCNDailyLexicon.VOCAB:
            element, mod = GHCNDailyLexicon[var]
            assert element == GHCNDailyLexicon.VOCAB[var]
            assert callable(mod)

    @pytest.mark.parametrize(
        "var, raw, expected",
        [
            ("t2m_max", np.array([200.0]), np.array([293.15])),  # 20.0 C -> K
            ("t2m_min", np.array([-100.0]), np.array([263.15])),  # -10.0 C -> K
            ("t2m", np.array([200.0]), np.array([293.15])),  # 20.0 C -> K
            ("d2m", np.array([150.0]), np.array([288.15])),  # 15.0 C -> K
            ("r2m", np.array([75.0]), np.array([75.0])),  # percent (no conv)
            ("tp", np.array([100.0]), np.array([0.01])),  # 10.0 mm -> 0.01 m
            ("sf", np.array([100.0]), np.array([0.01])),  # 10.0 mm -> 0.01 m
            ("sd", np.array([500.0]), np.array([0.5])),  # 500 mm -> 0.5 m
            ("sde", np.array([250.0]), np.array([0.25])),  # 250 mm -> 0.25 m
            ("ws10m", np.array([50.0]), np.array([5.0])),  # 50 tenths -> 5 m/s
            ("fg10m", np.array([120.0]), np.array([12.0])),  # 120 tenths -> 12 m/s
            ("tcc", np.array([80.0]), np.array([0.8])),  # 80% -> 0.8 fraction
        ],
    )
    def test_unit_conversions(self, var, raw, expected):
        _, mod = GHCNDailyLexicon[var]
        np.testing.assert_allclose(mod(raw), expected, atol=1e-6)

    def test_invalid_variable(self):
        with pytest.raises(KeyError):
            GHCNDailyLexicon["nonexistent"]
